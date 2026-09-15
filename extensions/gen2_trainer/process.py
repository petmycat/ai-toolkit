"""Native process bridge and extension-owned training lifecycle."""
from __future__ import annotations

import copy
import gc
import json
import random
import time
import traceback
import uuid
from pathlib import Path

import torch

from .config import resolve_process_config, SPEC_SHA256
from .data import assert_writable_path, preflight_datasets, make_native_loader, NativeDataStream, prepare_batch
from .recording import Recorder, write_json, json_safe


def specification_path(config):
    candidate = Path(config["gen2"]["spec_path"])
    if not candidate.is_absolute():
        # The example is portable across clones and independent of cwd.
        extension = Path(__file__).resolve().parent
        candidate = extension / candidate if (extension / candidate).is_file() else Path.cwd() / candidate
    return candidate.resolve()


def native_configuration(config):
    """Native constructors remain authoritative for the inherited native schemas."""
    from toolkit.config_modules import (ModelConfig, NetworkConfig, TrainConfig, SaveConfig, SampleConfig,
        DatasetConfig, LoggingConfig, preprocess_dataset_raw_config, validate_configs)
    native = {"train": TrainConfig(**copy.deepcopy(config["train"])),
              "model": ModelConfig(**copy.deepcopy(config["model"])),
              "network": NetworkConfig(**copy.deepcopy(config["network"])),
              "save": SaveConfig(**copy.deepcopy(config["save"])),
              "sample": SampleConfig(**copy.deepcopy(config["sample"])),
              "logging": LoggingConfig(**copy.deepcopy(config.get("logging", {})))}
    native["datasets"] = [DatasetConfig(**raw) for raw in preprocess_dataset_raw_config(copy.deepcopy(config["datasets"]))]
    validate_configs(native["train"], native["model"], native["save"], native["datasets"])
    return native


def resume_contract(config):
    """Runtime output locations/resume pointer are not the training mechanism."""
    result = copy.deepcopy(config)
    result.pop("_gen2_resolved", None)
    result.pop("name", None)
    result.pop("training_folder", None)
    result["gen2"]["checkpoint"]["resume_from"] = None
    result["gen2"]["spec_path"] = "immutable_sha256:" + SPEC_SHA256
    return result


class Gen2Runner:
    """One process, one GPU, no inherited optimizer/accumulation/skip loop."""
    def __init__(self, config, name, *, requested_config=None):
        self.config = resolve_process_config(config)
        self.name = name
        self.requested_config = copy.deepcopy(requested_config or config)
        self.save_root = assert_writable_path(Path(self.config["training_folder"]) / name)
        self.root = self.save_root / "gen2"
        self.recorder = None
        self.engine = self.backend = self.loader = self.stream = self.evaluation = None
        self.checkpoints = None
        self.run_id = None
        self.saved = set()
        self.last_checkpoint = None
        self.initial_frozen_hashes = None
        self.logger = None
        self.logger_started = False
        self.frozen_verified_at = None

    def _load(self):
        import numpy as np
        from toolkit.accelerator import get_accelerator
        from toolkit.logging_aitk import create_logger
        from extensions_built_in.diffusion_models.ideogram4.ideogram4 import Ideogram4Model
        from .backend_ideogram4 import Ideogram4Backend
        from .checkpointing import CheckpointManager, load_manifest
        from .diagnostics import capture_rng_state, restore_rng_state
        from .engine import Gen2Engine
        from .evaluation import Evaluation
        from .provenance import environment_manifest, frozen_state_hash

        config = self.config
        resume = config["gen2"]["checkpoint"]["resume_from"]
        previous = load_manifest(resume) if resume else None
        self.run_id = previous["metadata"]["run_id"] if previous else str(uuid.uuid4())
        existing = self.root / "run_manifest.json"
        if existing.exists() and not resume:
            raise ValueError(f"Run already exists: {self.root}. Select complete checkpoint resume_from or a new config.name")
        self.root.mkdir(parents=True, exist_ok=True)
        self.recorder = Recorder(self.root, self.run_id, config=config["gen2"]["recording"])
        self.recorder.event("initializing", requested_resume=resume)
        self.checkpoints = CheckpointManager(self.root / "checkpoints", specification_path(config),
            config["gen2"]["expected_spec_sha256"] or SPEC_SHA256, config["save"]["max_step_saves_to_keep"])
        if self.checkpoints.spec_sha256 != SPEC_SHA256:
            raise ValueError("This implementation requires the approved immutable v1 reference bytes")
        seed = config["gen2"]["execution"]["training_seed"]
        random.seed(seed)
        np.random.seed(seed % 2 ** 32)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(config["gen2"]["execution"]["deterministic_algorithms"])
        accelerator = get_accelerator()
        if accelerator.num_processes != 1 or not str(config["device"]).startswith("cuda"):
            raise ValueError("Gen2 production runs require exactly one CUDA process/GPU")
        if not torch.cuda.is_available():
            raise RuntimeError("Gen2 real-backend training requires CUDA; run this configuration on the VM")
        if config["train"]["dtype"] in ("fp16", "float16") and accelerator.scaler is None:
            raise ValueError("fp16 requires native accelerator scaling: launch with ACCELERATE_MIXED_PRECISION=fp16")
        native = native_configuration(config)
        self.logging_config = native["logging"]
        self.logger = create_logger(native["logging"], json_safe(config, redact=True), str(self.save_root))
        manifest = preflight_datasets(config)
        self.recorder.event("dataset_preflight_passed", images=len(manifest))
        model = Ideogram4Model(config["device"], native["model"], dtype=config["train"]["dtype"])
        model.load_model()
        model.noise_scheduler.set_train_timesteps(config["train"]["num_train_timesteps"],
            device=model.device_torch, timestep_type="linear")
        self.recorder.event("hashing_original_weights", method="full_native_serialized_state_streaming")
        self.initial_frozen_hashes = {name: frozen_state_hash(module) for name, module in
            (("diffusion", model.transformer), ("text_encoder", model.text_encoder), ("vae", model.vae))}
        self.initial_frozen_hashes["unconditional_lora"] = frozen_state_hash(model.unconditional_lora) if getattr(model, "unconditional_lora", None) is not None else None
        self.backend = Ideogram4Backend.from_native(model, config["gen2"], config["network"],
            config["trigger_word"], gradient_checkpointing=config["train"]["gradient_checkpointing"])
        self.engine = Gen2Engine(config, self.backend, accelerator=accelerator, recorder=self.recorder)
        self.loader = make_native_loader(config, model, manifest)
        self.stream = NativeDataStream(self.loader, seed)
        self.manifest_by_path = {row["path"]: row for row in manifest}
        self.evaluation = Evaluation(self.backend, config, self.recorder, self.root)
        self.evaluation.model_identities = self.initial_frozen_hashes
        self.engine.external_diagnostic_bytes = self.evaluation.retained_bytes
        self.evaluation.external_diagnostic_bytes = self.engine.retained_diagnostic_bytes
        self.metadata = {"run_id": self.run_id, "name": self.name, "resolved_config": config,
            "requested_config": self.requested_config, "training_contract": resume_contract(config),
            "model_identities": self.initial_frozen_hashes, "model_source": config["model"],
            "module_mapping": [{k: v for k, v in row.items() if k not in ("device",)} for row in self.backend.module_manifest()],
            "environment": environment_manifest(), "spec_sha256": self.checkpoints.spec_sha256,
            "tokenizer_identity": {"name_or_path": model.tokenizer.name_or_path,
                "vocabulary_sha256": __import__("hashlib").sha256(json.dumps(model.tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
                "chat_template": model.tokenizer.chat_template,
                "encoder_commit": getattr(model.text_encoder.config, "_commit_hash", None)},
            "dataset_identity": [{"sample_id": row["sample_id"], "content_hash": row["content_hash"],
                "canonical_caption_hash": row["canonical_caption_hash"]} for row in manifest]}
        if previous:
            expected = {key: self.metadata[key] for key in ("training_contract", "model_identities", "module_mapping", "tokenizer_identity", "dataset_identity")}
            loaded = self.checkpoints.load(resume, expected_metadata=expected, components=self.backend.components())
            state = loaded["engine_state"]
            self.engine.load_state_dict(state)
            self.stream.load_state_dict(state["data_stream"])
            self.evaluation.load_state_dict(state["evaluation"])
            self.recorder.resume(loaded["recorder_state"])
            restore_rng_state(loaded["rng_state"])
            self.last_checkpoint = Path(resume)
            self.saved.add(self.engine.logical_update)
        else:
            self.evaluation.initialize(self.loader, manifest)
            for row in manifest:
                self.recorder.record("dataset_manifest", row)
        write_json(self.root / "run_manifest.json", {**self.metadata, "optimizer_factories": self.engine.factory_manifest,
            "family_horizons": self.engine.schedule.horizons,
            "loader_replay": "exact reference: num_workers=0" if self.loader.num_workers == 0 else "worker prefetch cannot be exactly restored",
            "dataset_initialization_seed_rule": "training_seed + preprocessed native dataset index; isolated from training RNG",
            "precision": {"trainable_masters": "float32", "compute": str(model.torch_dtype)},
            "timing_policy": "perf_counter without per-region CUDA synchronization; kernel timing is approximate",
            "neutral_preservation_scope": "re-noised target dataset latents with paired content captions"})
        write_json(self.root / "module_manifest.json", self.backend.module_manifest())
        config["_gen2_resolved"]["native_schema_validation"] = "passed native configuration constructors and validate_configs"
        config["_gen2_resolved"]["factory_manifest"] = self.engine.factory_manifest
        self._write_config("config.requested.yaml", self.requested_config)
        self._write_config("config.resolved.yaml", config)
        self.recorder.event("initialized", logical_update=self.engine.logical_update,
            family_horizons=self.engine.schedule.horizons, factories=self.engine.factory_manifest)
        self.logger.start()
        self.logger_started = True

    def _write_config(self, filename, value):
        import yaml
        (self.root / filename).write_text(yaml.safe_dump(json_safe(value, redact=True), sort_keys=False, allow_unicode=True), encoding="utf-8")

    def _set_context(self):
        update = self.engine.logical_update
        schedule = self.engine.schedule
        self.recorder.set_context(logical_update=update, update_attempt_id=self.engine.update_attempt,
            stage=schedule.stage_at(update), update_kind=schedule.kind_at(update) if update < schedule.total else "complete",
            checkpoint_hash=None)

    def save(self, reasons, protected=False):
        from .diagnostics import capture_rng_state
        if self.engine.logical_update in self.saved:
            return self.last_checkpoint
        checkpoint_started = time.perf_counter()
        self.recorder.event("checkpoint_begin", reasons=reasons)
        self.recorder.flush()
        state = self.engine.state_dict()
        state["data_stream"] = self.stream.state_dict()
        state["evaluation"] = self.evaluation.state_dict()
        dtype = {"float32": torch.float32, "fp32": torch.float32, "float16": torch.float16,
                 "fp16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}[self.config["save"]["dtype"]]
        self.last_checkpoint = self.checkpoints.save(self.engine.logical_update, self.backend.components(),
            state, capture_rng_state(), self.metadata, recorder_state=self.recorder.state_dict(),
            protected=protected, reasons=reasons, export_dtype=dtype)
        self.saved.add(self.engine.logical_update)
        self.recorder.event("checkpoint_complete", checkpoint=str(self.last_checkpoint), reasons=reasons,
            io_seconds=time.perf_counter()-checkpoint_started)
        return self.last_checkpoint

    def events(self, initial=False, force_save=False):
        from .evaluation import bundle_identity
        diagnostic_started = time.perf_counter()
        update = self.engine.logical_update
        schedule, dg, evaluation = self.engine.schedule, self.config["gen2"]["diagnostics"], self.config["gen2"]["evaluation"]
        boundary = update in schedule.boundaries
        final = update == schedule.total
        self._set_context()
        reasons = (["initial"] if initial else []) + (["stage_boundary"] if boundary else []) + (["final"] if final else [])
        numeric_due = boundary or any(update % interval == 0 for interval in
            (dg["probes"]["every"], dg["gradient_probe_every"], dg["spectra_every"], dg["gate_log_every"]))
        save_due = boundary or force_save or update % self.config["save"]["save_every"] == 0
        milestone = boundary or update % evaluation["milestone_every"] == 0
        preview = update >= self.config["sample"]["sample_start_step"] and update % self.config["sample"]["sample_every"] == 0
        validation = self.config["train"].get("validation_config")
        validation_due = validation and (boundary or update % validation.get("validate_every_n_steps", 10) == 0)
        package_hash = hashes = None
        if numeric_due or save_due or validation_due or ((milestone or preview) and not self.config["train"]["disable_sampling"]):
            package_hash, hashes = bundle_identity(self.backend)
            self.recorder.set_context(checkpoint_hash=package_hash)
            self.recorder.event("component_state_hashes", component_hashes=hashes, package_state_hash=package_hash,
                checkpoint_identity_kind="full_component_state_sha256")
        if boundary:
            self.recorder.event("stage_boundary", next_stage=schedule.stage_at(update), family_steps=self.engine.family_steps)
        if boundary or update % dg["probes"]["every"] == 0:
            self.evaluation.numerical_probes(update, reasons or ["probe_interval"])
        if boundary or update % dg["gradient_probe_every"] == 0:
            self.evaluation.gradient_probes(update)
        spectra = boundary or update % dg["spectra_every"] == 0
        calibration_save = update > self.config["gen2"]["phases"]["warmup_updates"] + self.config["gen2"]["phases"]["refinement_updates"] and update % self.config["save"]["save_every"] == 0
        if spectra or calibration_save or update % dg["gate_log_every"] == 0:
            self.evaluation.representations(spectra=spectra)
        if validation_due:
            self.evaluation.validate(update)
        if not self.config["train"]["disable_sampling"] and (milestone or preview) and not (initial and self.config["train"]["skip_first_sample"]):
            # Union identical sampling requests across interval/boundary causes.
            modes = list(dict.fromkeys((evaluation["preview_modes"] if preview else []) + (evaluation["milestone_modes"] if milestone else [])))
            seeds = list(dict.fromkeys([self.config["sample"]["seed"]] + (evaluation["additional_seeds"] if milestone else [])))
            self.evaluation.sample(update, modes, seeds, reasons + (["milestone"] if milestone else []) + (["preview"] if preview else []), package_hash, hashes)
        if save_due:
            self.save(reasons or ["save_interval"], protected=boundary)
        if boundary:
            self.recorder.flush()
        self.recorder.event("scheduled_diagnostics_complete", seconds=time.perf_counter()-diagnostic_started,
            timing_includes_sampling_and_checkpoint_io=True, approximate_cuda_timing=True)

    def _verify_frozen(self):
        from .provenance import frozen_state_hash
        if self.frozen_verified_at == self.engine.logical_update:
            return
        self.backend.assert_frozen()
        excluded = [p for parameters in self.backend.parameter_families().values() for p in parameters]
        model = self.backend.model
        hashes = {name: frozen_state_hash(module, excluded) for name, module in
            (("diffusion", model.transformer), ("text_encoder", model.text_encoder), ("vae", model.vae))}
        uncond = getattr(model, "unconditional_lora", None)
        hashes["unconditional_lora"] = frozen_state_hash(uncond) if uncond is not None else None
        if hashes != self.initial_frozen_hashes:
            raise RuntimeError("Full original model state changed during Gen2 training")
        self.frozen_verified_at = self.engine.logical_update
        self.recorder.event("frozen_weights_verified", hashes=hashes, method="full_native_serialized_state_streaming")

    def run(self, *, stop_after=None):
        """stop_after is an acceptance-harness boundary, never a training option."""
        from .diagnostics import summarize_run
        try:
            self._load()
            if self.engine.logical_update == 0:
                self.events(initial=True)
            while self.engine.logical_update < self.engine.schedule.total:
                if stop_after is not None and self.engine.logical_update >= stop_after:
                    self._verify_frozen()
                    self.save(["acceptance_harness_stop"], protected=True)
                    self.recorder.event("acceptance_harness_stopped", logical_update=self.engine.logical_update)
                    break
                self._set_context()
                before_data = time.perf_counter()
                window = [prepare_batch(self.stream.next(), self.backend.model, self.config, self.manifest_by_path)
                          for _ in range(self.config["train"]["gradient_accumulation_steps"])]
                data_seconds = time.perf_counter() - before_data
                due = (self.engine.logical_update + 1) % self.config["gen2"]["diagnostics"]["activation_every"] == 0
                with self.evaluation.activation_context(due):
                    result = self.engine.step(window)
                del window
                self._set_context()
                self.recorder.event("data_preparation", seconds=data_seconds, approximate_cuda_timing=True)
                self.recorder.event("recorder_status", **self.recorder.status())
                print(f"Gen2 {self.engine.logical_update}/{self.engine.schedule.total} {result['update_kind']}: {result.get('losses', {})}", flush=True)
                if self.engine.logical_update % self.logging_config.log_every == 0:
                    prefix = f"gen2/{result['stage']}/{result['update_kind']}"
                    self.logger.log({f"{prefix}/{key}": value for key, value in result["losses"].items() if value is not None})
                    self.logger.commit(step=self.engine.logical_update)
                if self.engine.logical_update == self.engine.schedule.total:
                    self._verify_frozen()
                self.events()
            self._verify_frozen()
            complete = self.engine.logical_update == self.engine.schedule.total
            self.recorder.event("run_complete" if complete else "run_paused_for_acceptance",
                logical_update=self.engine.logical_update, checkpoint=str(self.last_checkpoint))
            self.recorder.flush()
            summarize_run(self.root)
            return self.last_checkpoint
        except BaseException as error:
            if self.recorder is not None:
                try:
                    self.recorder.event("run_aborted", error_type=type(error).__name__, error=str(error),
                        traceback=traceback.format_exc(), partial_commit=bool(getattr(self.engine, "partial_commit", False)),
                        last_complete_checkpoint=str(self.last_checkpoint) if self.last_checkpoint else None)
                    self.recorder.flush()
                except Exception:
                    # Disk/budget failures still produce a visible stderr failure.
                    traceback.print_exc()
            raise
        finally:
            try:
                if self.logger_started:
                    self.logger.finish()
                    self.logger_started = False
            finally:
                if self.recorder is not None:
                    self.recorder.close()

    def release(self):
        self.engine = self.backend = self.loader = self.stream = self.evaluation = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Gen2TrainProcess:
    """Compose native BaseTrainProcess without inheriting its SD stepping loop."""
    def __init__(self, process_id, job, config):
        resolved = resolve_process_config(config)
        name = config.get("name", job.name)
        assert_writable_path(Path(resolved["training_folder"]) / name)
        log_dir = config.get("log_dir", getattr(job, "log_dir", None))
        if log_dir:
            assert_writable_path(log_dir)
        if (Path(resolved["training_folder"]) / name / "gen2" / "run_manifest.json").exists() and not resolved["gen2"]["checkpoint"]["resume_from"]:
            raise ValueError("Run already exists; choose a new name or a complete checkpoint resume_from")
        from jobs.process.BaseTrainProcess import BaseTrainProcess
        self.native = BaseTrainProcess(process_id, job, copy.deepcopy(resolved))
        self.runner = Gen2Runner(resolved, name, requested_config=config)

    def __getattr__(self, name):
        return getattr(self.native, name)

    def run(self):
        self.native.run()
        try:
            return self.runner.run()
        finally:
            self.runner.release()

    def on_error(self, error):
        # Runner owns error recording and never advances a skipped update.
        pass
