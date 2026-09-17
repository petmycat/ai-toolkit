"""V2 lifecycle: one token optimizer, native data, independent image/save clocks."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import random
import time
import traceback
import uuid
from pathlib import Path

import torch

from ..data import assert_writable_path, preflight_datasets, make_native_loader, NativeDataStream
from ..data import prepare_batch as prepare_native_batch
from ..diagnostics import capture_rng_state, restore_rng_state, isolated_rng
from ..recording import Recorder, write_json, json_safe, iter_records
from .config import resolve_process_config, SPEC_SHA256


def specification_path(config):
    path = Path(config["gen2"]["spec_path"])
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if hashlib.sha256(path.read_bytes()).hexdigest() != SPEC_SHA256:
        raise ValueError("V2 specification bytes do not match the approved immutable reference")
    return path.resolve()


def native_configuration(config):
    from toolkit.config_modules import (ModelConfig, TrainConfig, SaveConfig, SampleConfig,
        DatasetConfig, LoggingConfig, preprocess_dataset_raw_config, validate_configs)
    native = {name: cls(**copy.deepcopy(config.get(name, {}))) for name, cls in (
        ("model", ModelConfig), ("train", TrainConfig), ("save", SaveConfig),
        ("sample", SampleConfig), ("logging", LoggingConfig))}
    native["datasets"] = [DatasetConfig(**raw) for raw in
        preprocess_dataset_raw_config(copy.deepcopy(config["datasets"]))]
    validate_configs(native["train"], native["model"], native["save"], native["datasets"])
    return native


def training_contract(config):
    contract = copy.deepcopy(config)
    for key in ("_gen2_resolved", "name", "training_folder", "log_dir"):
        contract.pop(key, None)
    contract["gen2"]["checkpoint"]["resume_from"] = None
    contract["gen2"]["spec_path"] = "sha256:" + SPEC_SHA256
    return contract


def prepare_batch(dto, model, config, manifest_by_path):
    # Reuse native latent/noise/augmentation preparation, but v2 must retain all
    # markers. V1's canonical q deliberately removes them and is not our input.
    batch = prepare_native_batch(dto, model, config, manifest_by_path)
    batch["qs"] = [row["original_caption"] for row in batch["metadata"]]
    for row in batch["metadata"]:
        row["q"] = row["original_caption"]
    return batch


def build_token_report(config, manifest, tokenizer):
    from .text import NativePromptCompiler
    gen2 = config["gen2"]
    compiler = NativePromptCompiler(tokenizer, config["trigger_word"],
        gen2["conditioning"]["num_tokens"], config["model"]["model_kwargs"]["max_text_length"],
        overflow_policy=gen2["conditioning"]["overflow_policy"])
    entries, failures = [], []
    for row in manifest:
        identity = {"path": row["path"], "caption_provenance": row["caption_provenance"],
                    "sample_id": row["sample_id"], "kind": "training"}
        try:
            compiled = compiler.compile(row["original_caption"], require_trigger=True)
            entries.append({**identity, **compiled.metadata})
        except (ValueError, RuntimeError) as error:
            failures.append({**identity, "error": str(error)})
    for index, prompt in enumerate(config["sample"]["prompts"]):
        identity = {"kind": "sampling", "prompt_index": index}
        try:
            comparisons = compiler.comparison(prompt, named_phrase=gen2["evaluation"]["named_phrase"])
            entries.extend({**identity, "mode": mode, **item.metadata} for mode, item in comparisons.items())
        except (ValueError, RuntimeError) as error:
            failures.append({**identity, "error": str(error)})
    return {"schema_version": "2.0.0", "passed": not failures, "failures": failures,
            "entries": entries, "captions_checked": len(manifest) + len(config["sample"]["prompts"]),
            "training_truncation_count": sum(row.get("truncated", False) for row in entries if row["kind"] == "training"),
            "maximum_resulting_length": max((row["resulting_length"] for row in entries), default=0),
            "total_token_limit": config["model"]["model_kwargs"]["max_text_length"],
            "overflow_policy": gen2["conditioning"]["overflow_policy"]}


def memory_stats(device):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        return {"device": str(device), "cuda_available": False}
    return {"device": str(device), "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device)}


def summarize_run(root):
    root = Path(root)
    rows = list(iter_records(root, "updates"))
    events = list(iter_records(root, "events"))
    # A resumed run can replay updates after its last durable checkpoint. Keep
    # the most recent attempt per committed index; never count replay twice.
    by_update = {row["logical_update"]: row for row in rows}
    ordered = [by_update[index] for index in sorted(by_update)]
    summary = {"schema_version": "2.0.0", "committed_updates_recorded": len(ordered),
        "latest_update": ordered[-1] if ordered else None,
        "run_complete": any(row.get("event") == "run_complete" for row in events),
        "frozen_weights_verified": any(row.get("event") == "frozen_weights_verified" for row in events),
        "visual_acceptance": "pending_user_review",
        "visual_acceptance_rule": "At least named-phrase style fidelity with content preserved; loss alone cannot pass."}
    write_json(root / "summary.json", summary)
    return summary


class V2Runner:
    def __init__(self, config, name, *, requested_config=None):
        self.config = resolve_process_config(config)
        self.requested_config = copy.deepcopy(requested_config or config)
        self.name = name
        self.save_root = assert_writable_path(Path(self.config["training_folder"]) / name)
        self.root = self.save_root / "gen2_v2"
        self.backend = self.engine = self.loader = self.stream = self.evaluation = None
        self.recorder = self.logger = None
        self.logger_started = False
        self.last_checkpoint = None
        self.initial_frozen_hashes = None

    def _load(self):
        import numpy as np
        from toolkit.accelerator import get_accelerator
        from toolkit.logging_aitk import create_logger
        from extensions_built_in.diffusion_models.ideogram4.ideogram4 import Ideogram4Model
        from ..text_preflight import load_tokenizer
        from ..original_unconditional import load_original_unconditional
        from ..package import tokenizer_identity
        from ..provenance import environment_manifest, frozen_model_hashes
        from .backend import V2Backend
        from .checkpoint import V2CheckpointManager, load_manifest
        from .engine import V2Engine
        from .evaluation import V2Evaluation

        config = self.config
        spec = specification_path(config)
        resume = config["gen2"]["checkpoint"]["resume_from"]
        previous = load_manifest(resume, expected_sha=SPEC_SHA256) if resume else None
        existing_run = (self.root / "run_manifest.json").exists()
        if existing_run and not resume:
            raise ValueError("V2 run already exists; choose a new name or resume_latest checkpoint")
        self.run_id = previous["metadata"]["run_id"] if previous else str(uuid.uuid4())
        self.root.mkdir(parents=True, exist_ok=True)
        self.recorder = Recorder(self.root, self.run_id, config=config["gen2"]["recording"])
        self.recorder.event("initializing", trainer_version="2.0.0", requested_resume=resume)
        self.checkpoints = V2CheckpointManager(self.root / "checkpoints", spec, SPEC_SHA256)
        seed = config["gen2"]["execution"]["training_seed"]
        random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
        torch.use_deterministic_algorithms(config["gen2"]["execution"]["deterministic_algorithms"])
        accelerator = get_accelerator()
        if accelerator.num_processes != 1 or not str(config["device"]).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("V2 real-backend training requires one CUDA GPU; run the smoke on the VM")
        native = native_configuration(config)
        self.logging_config = native["logging"]
        self.logger = create_logger(native["logging"], json_safe(config, redact=True), str(self.save_root))
        manifest = preflight_datasets(config)
        # V2 identity includes complete source captions, including every marker.
        for row in manifest:
            row["q"] = row["original_caption"]
            row["canonical_caption_hash"] = row["caption_hash"]
        tokenizer = load_tokenizer(config)
        report = build_token_report(config, manifest, tokenizer)
        report_path = write_json(self.root / "caption_token_report.json", report)
        if not report["passed"]:
            raise ValueError(f"V2 caption preflight rejected {len(report['failures'])} input(s); "
                             f"report: {report_path}; failures: {report['failures']}")
        del tokenizer
        self.recorder.event("caption_preflight_passed", report=str(report_path), images=len(manifest))
        print(f"[Gen2 v2] caption preflight passed: {len(manifest)} training captions; "
              f"{report['training_truncation_count']} truncated; longest sequence "
              f"{report['maximum_resulting_length']} tokens. Report: {report_path}", flush=True)
        model = Ideogram4Model(config["device"], native["model"], dtype=config["train"]["dtype"])
        model.load_model()
        load_original_unconditional(model, config["gen2"])
        model.noise_scheduler.set_train_timesteps(config["train"]["num_train_timesteps"],
            device=model.device_torch, timestep_type="linear")
        self.recorder.event("hashing_original_weights")
        self.initial_frozen_hashes = frozen_model_hashes(model)
        self.backend = V2Backend(model, config)
        self.engine = V2Engine(config, self.backend, accelerator=accelerator)
        self.loader = make_native_loader(config, model, manifest)
        self.stream = NativeDataStream(self.loader, seed)
        self.manifest_by_path = {row["path"]: row for row in manifest}
        self.evaluation = V2Evaluation(self.backend, config, self.recorder, self.root)
        self.metadata = {"run_id": self.run_id, "name": self.name, "resolved_config": config,
            "requested_config": self.requested_config, "training_contract": training_contract(config),
            "model_identities": self.initial_frozen_hashes, "tokenizer_identity": tokenizer_identity(model),
            "dataset_identity": [{key: row[key] for key in ("sample_id", "content_hash", "caption_hash")} for row in manifest],
            "spec_sha256": SPEC_SHA256, "environment": environment_manifest(),
            "optimizer_factory": self.engine.factory_manifest,
            "initialization": self.backend.tokens.provenance(),
            "precision": {"trainable_masters": "float32", "compute": str(model.torch_dtype)}}
        if previous:
            expected = {key: self.metadata[key] for key in
                        ("training_contract", "model_identities", "tokenizer_identity", "dataset_identity")}
            runtime = self.checkpoints.load(resume, self.backend, self.engine, expected_metadata=expected)
            self.stream.load_state_dict(runtime["data"])
            self.evaluation.load_state_dict(runtime["evaluation"])
            if "recorder" in runtime and existing_run:
                self.recorder.resume(runtime["recorder"])
            elif "recorder" in runtime:
                self.recorder.event("resumed_into_new_output_directory", parent_checkpoint=str(resume),
                                    parent_recorder=runtime["recorder"])
            restore_rng_state(runtime["rng"])
            self.last_checkpoint = Path(resume)
        else:
            for row in manifest:
                self.recorder.record("dataset_manifest", row)
        write_json(self.root / "run_manifest.json", self.metadata)
        import yaml
        for filename, value in (("config.requested.yaml", self.requested_config), ("config.resolved.yaml", config)):
            (self.root / filename).write_text(yaml.safe_dump(json_safe(value, redact=True), sort_keys=False,
                                                          allow_unicode=True), encoding="utf-8")
        self.recorder.event("initialized", logical_update=self.engine.logical_update,
            effective_batch_size=config["train"]["batch_size"] * config["train"]["gradient_accumulation_steps"],
            optimizer=self.engine.factory_manifest, memory=memory_stats(config["device"]))
        if config["gen2"]["diagnostics"]["mechanical_probes"]:
            with isolated_rng(seed):
                longest = max((row for row in report["entries"] if row["kind"] == "training"),
                              key=lambda row: row["resulting_length"])
                captions = [self.manifest_by_path[longest["path"]]["original_caption"]]
                self.recorder.event("native_features_verified", records=self.backend.verify_native(captions))
                probe = self.backend.probe_conditioning(captions[0])
                self.recorder.event("conditioning_gradient_probe", result=probe)
                if not all(probe[key] for key in ("finite_features", "gradient_finite", "gradient_nonzero")):
                    raise RuntimeError("V2 ordinary-caption gradient path failed its mechanical probe")
            if self.engine.logical_update == 0:
                from .mechanical import run_memory_probes
                run_memory_probes(self.backend, self.loader, config, self.manifest_by_path, self.recorder)
        self.logger.start()
        self.logger_started = True

    def _context(self):
        self.recorder.set_context(logical_update=self.engine.logical_update,
            update_attempt_id=getattr(self.engine, "update_attempt", self.engine.logical_update),
            stage="standalone_activator", update_kind="embedding")

    def save(self, *, final=False):
        self.recorder.flush()
        runtime = {"rng": capture_rng_state(), "data": self.stream.state_dict(),
                   "evaluation": self.evaluation.state_dict(), "recorder": self.recorder.state_dict()}
        self.last_checkpoint = self.checkpoints.save(self.backend, self.engine, self.metadata, runtime, final=final)
        self.recorder.event("checkpoint_complete", path=str(self.last_checkpoint), final=final)
        print(f"[Gen2 v2] saved {'final export + rolling resume' if final else 'rolling resume'}: {self.last_checkpoint}", flush=True)
        return self.last_checkpoint

    def events(self, *, initial=False):
        self._context()
        update = self.engine.logical_update
        train, sample = self.config["train"], self.config["sample"]
        sampling_due = (initial and not train["skip_first_sample"]) or (
            not initial and update > 0 and update >= sample.get("sample_start_step", 0)
            and update % sample["sample_every"] == 0)
        if sampling_due and not train["disable_sampling"]:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.config["device"])
            self.evaluation.sample(update)
            self.recorder.event("sampling_memory", **memory_stats(self.config["device"]))
        if not initial:
            final = update == train["steps"]
            if final or update % self.config["save"]["save_every"] == 0:
                self.save(final=final)

    def _verify_frozen(self):
        from ..provenance import frozen_model_hashes
        self.backend.assert_frozen()
        hashes = frozen_model_hashes(self.backend.model)
        if hashes != self.initial_frozen_hashes:
            raise RuntimeError("V2 frozen original model state changed")
        self.recorder.event("frozen_weights_verified", hashes=hashes)

    def run(self, *, stop_after=None):
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            self._load()
            if self.engine.logical_update == 0:
                self.events(initial=True)
            elif self.engine.logical_update == self.config["train"]["steps"]:
                # The rolling state is published before the final inference
                # package. Recover an interrupted export without retraining.
                self._context()
                self._verify_frozen()
                self.save(final=True)
            while self.engine.logical_update < self.config["train"]["steps"]:
                if stop_after is not None and self.engine.logical_update >= stop_after:
                    self._verify_frozen()
                    self.save()
                    self.recorder.event("acceptance_harness_stopped")
                    break
                self._context()
                started = time.perf_counter()
                window = [prepare_batch(self.stream.next(), self.backend.model, self.config, self.manifest_by_path)
                          for _ in range(self.config["train"]["gradient_accumulation_steps"])]
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(self.config["device"])
                result = self.engine.step(window)
                del window
                self._context()
                result = {**result, "trainer_version": "2.0.0", "seconds": time.perf_counter()-started,
                          "memory": memory_stats(self.config["device"])}
                self.recorder.record("updates", result)
                print(f"[Gen2 v2] activator update {self.engine.logical_update}/{self.config['train']['steps']} "
                      f"| loss {result.get('loss', result.get('losses'))} | {result['seconds']:.2f}s "
                      f"| peak allocated {result['memory'].get('peak_allocated_bytes', 0)/2**30:.2f} GiB", flush=True)
                if self.engine.logical_update % self.logging_config.log_every == 0:
                    numeric = {f"gen2_v2/{key}": value for key, value in result.items()
                               if isinstance(value, (int, float)) and not isinstance(value, bool)}
                    self.logger.log(numeric); self.logger.commit(step=self.engine.logical_update)
                if self.engine.logical_update == self.config["train"]["steps"]:
                    self._verify_frozen()
                self.events()
            self.recorder.event("run_complete" if self.engine.logical_update == self.config["train"]["steps"]
                                else "run_paused_for_acceptance", visual_acceptance="pending_user_review")
            self.recorder.flush()
            summarize_run(self.root)
            return self.last_checkpoint
        except BaseException as error:
            if self.recorder is not None:
                try:
                    self.recorder.event("run_aborted", error_type=type(error).__name__, error=str(error),
                        traceback=traceback.format_exc(), last_complete_checkpoint=str(self.last_checkpoint),
                        partial_commit=bool(getattr(self.engine, "partial_commit", False)))
                    self.recorder.flush()
                except Exception:
                    traceback.print_exc()
            raise
        finally:
            try:
                if self.logger_started:
                    self.logger.finish()
                if self.recorder is not None:
                    self.recorder.close()
            finally:
                torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn)

    def release(self):
        self.engine = self.backend = self.loader = self.stream = self.evaluation = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class V2TrainProcess:
    def __init__(self, process_id, job, config):
        resolved = resolve_process_config(config)
        name = config.get("name", job.name)
        root = assert_writable_path(Path(resolved["training_folder"]) / name)
        log_dir = config.get("log_dir", getattr(job, "log_dir", None))
        if log_dir:
            assert_writable_path(log_dir)
        if (root / "gen2_v2" / "run_manifest.json").exists() and not resolved["gen2"]["checkpoint"]["resume_from"]:
            raise ValueError("V2 run exists; select a new name or resume_latest checkpoint")
        from jobs.process.BaseTrainProcess import BaseTrainProcess
        self.native = BaseTrainProcess(process_id, job, copy.deepcopy(resolved))
        self.runner = V2Runner(resolved, name, requested_config=config)

    def __getattr__(self, name):
        return getattr(self.native, name)

    def run(self):
        self.native.run()
        try:
            return self.runner.run()
        finally:
            self.runner.release()

    def on_error(self, error):
        pass
