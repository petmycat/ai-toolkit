from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping

from jobs.process import BaseExtensionProcess

from extensions_built_in.ideogram4_v3_residual_ablation.helpers import (
    parse_activator_a2_contract,
    require_file_hash,
    sha256_file,
)
from toolkit.residual_gating import (
    ResidualGateRouter,
    ResidualGateRuntime,
    active_registry_fingerprint,
    bind_active_registry,
    build_module_registry,
    filter_active_registry,
    residual_gate_runtime_context,
    serialize_active_registry,
)
from extensions_built_in.ideogram4_v3_residual_ablation.process import Ideogram4V3ResidualAblationProcess
from .config import load_config
from .helpers import (
    DeterministicStratifiedTimestepSampler,
    append_jsonl,
    artifact_ref,
    atomic_write_json,
    detect_active_groups,
    gate_regularization,
    load_json,
    load_jsonl,
    rewrite_jsonl,
    router_config_payload,
    select_best_validation,
    temporal_smoothness,
    validate_registry_contract,
    validation_grid,
)


class Ideogram4V3RouterPhaseCProcess(Ideogram4V3ResidualAblationProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        BaseExtensionProcess.__init__(self, process_id, job, config)
        self.phase_c = load_config(self.get_conf("phase_c_router", required=True))
        self.device = self.get_conf("device", getattr(job, "device", "cuda"))
        self.output_root = Path(self.phase_c.output_root).resolve()
        self.ablation = {
            "a2_contract": self.phase_c.a2_contract,
            "dataset_root": self.phase_c.dataset_root,
            "split_manifest": self.phase_c.split_manifest,
        }
        self.model_definition = self.get_conf("model")
        self.network_definition = self.get_conf("network")
        self.run_label = str(self.get_conf("run_id", self.name))
        self.sd = None
        self.network = None
        self.text_activator = None
        self._run_started_at = None
        self._calibrated_embed_cache = {}
        self._stock_embed_cache = {}
        self._latent_cache = {}

    def _phase_c_progress(self, message: str) -> None:
        from toolkit.print import print_acc

        elapsed = 0.0 if self._run_started_at is None else time.monotonic() - self._run_started_at
        print_acc(f"[Phase C +{self._format_duration(elapsed)}] {message}")

    def run(self):
        BaseExtensionProcess.run(self)
        import torch
        from tqdm.auto import tqdm

        self._run_started_at = time.monotonic()
        self._phase_c_progress(f"starting run {self.run_label!r}; output={self.output_root}")
        self._phase_c_progress("resolving A2, residual-ablation, registry, and split contracts")
        self._resolve_inputs()
        self._resolve_phase_c_inputs()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._phase_c_progress("loading Ideogram4 model")
        self.sd = self._load_model()
        self._phase_c_progress("loading frozen V3 LoRA")
        self.network = self._load_lora(torch)
        self._phase_c_progress("installing frozen A2 activator")
        self.text_activator = self._build_text_activator(torch)
        self._load_and_install_activator()
        self._freeze()
        self._enable_checkpointing()
        self.network.is_active = True
        self._phase_c_progress("model, V3 LoRA, activator, and checkpointing ready")

        self._phase_c_progress("building and validating active residual registry")
        actual_registry = build_module_registry(self.network.get_all_modules())
        validate_registry_contract(actual_registry, self.recorded_registry)
        active_groups = detect_active_groups(actual_registry)
        active_registry = filter_active_registry(actual_registry, norm_threshold=0.0)
        if tuple(sorted({str(row["group_id"]) for row in active_registry})) != active_groups:
            raise RuntimeError("active registry filtering disagrees with Phase C active-group detection")
        bind_active_registry(self.network.get_all_modules(), active_registry)
        active_payload = serialize_active_registry(active_registry)
        registry_fingerprint = active_registry_fingerprint(active_registry)
        group_count = int(active_payload["group_count"])

        self._phase_c_progress(
            f"active registry ready: {group_count} groups across {len(active_registry)} LoRA modules"
        )
        self._seed_everything(torch)
        router = self._build_canonical_router(group_count).to(self.sd.device_torch, dtype=torch.float32)
        optimizer = self._build_optimizer(router)
        self._phase_c_progress(
            f"router and optimizer ready: optimizer={self.phase_c.optimizer}, steps={self.phase_c.steps}"
        )
        paths = self._artifact_paths()
        source_manifest = self._source_manifest(active_payload, registry_fingerprint)
        router_config = router_config_payload(self.phase_c, active_payload)
        run_fingerprint = self._run_fingerprint(source_manifest, router_config)
        start_step = (
            self._resume(router, optimizer, registry_fingerprint, run_fingerprint, torch)
            if self.phase_c.resume
            else 0
        )
        if start_step > self.phase_c.steps:
            raise RuntimeError(
                f"Phase C checkpoint step {start_step} exceeds configured steps {self.phase_c.steps}"
            )
        self._assert_router_only_trainable(router)
        self._sanitize_resume_artifacts(paths, start_step)
        if start_step:
            self._phase_c_progress(f"resumed compatible checkpoint at step {start_step}/{self.phase_c.steps}")
        else:
            self._phase_c_progress("starting from fresh zero-initialized router")

        train_items, validation_items = self._load_items()
        self._validation_item_count = len(validation_items)
        self._phase_c_progress(
            f"precomputing {len(train_items)} train and {len(validation_items)} validation items"
        )
        self._precompute_items(train_items + validation_items, torch)
        self._phase_c_progress("running real-model residual-gate preflight invariants")
        self._assert_runtime_invariants(router, train_items[0], registry_fingerprint, start_step, torch)
        self._phase_c_progress("preflight invariants passed; entering router training")
        source_manifest["run_fingerprint"] = run_fingerprint
        atomic_write_json(paths["source_manifest"], source_manifest)
        router_config["run_fingerprint"] = run_fingerprint
        atomic_write_json(paths["router_config"], router_config)

        sampler = DeterministicStratifiedTimestepSampler(
            self.phase_c.seed, self.phase_c.timestep_bins
        )
        progress = tqdm(
            range(start_step + 1, self.phase_c.steps + 1),
            total=self.phase_c.steps,
            initial=start_step,
            desc="Phase C router training",
            unit="step",
            dynamic_ncols=True,
            leave=True,
        )
        for step in progress:
            timestep, bin_index = sampler.sample(step)
            item = train_items[(step - 1) % len(train_items)]
            prepared = self._prepare_training_case(item, timestep, self.phase_c.seed + step, torch)
            normalized = torch.tensor([timestep / 1000.0], device=self.sd.device_torch, dtype=torch.float32)
            q_values = router(normalized)
            gates = 1.0 + q_values
            runtime = ResidualGateRuntime(gates, registry_fingerprint=registry_fingerprint)
            optimizer.zero_grad(set_to_none=True)
            with residual_gate_runtime_context(self.network, runtime):
                prediction = self._predict(prepared, prepared["prompt_template"], "full")
            reconstruction = torch.mean((prediction.float() - prepared["target"].float()).square())
            gate_penalty = gate_regularization(q_values)
            smoothness = reconstruction.new_zeros(())
            if self.phase_c.temporal_smoothness.enabled:
                smoothness = temporal_smoothness(
                    router, normalized, self.phase_c.temporal_smoothness.delta
                )
            loss = (
                reconstruction
                + self.phase_c.lambda_gate * gate_penalty
                + self.phase_c.temporal_smoothness.weight * smoothness
            )
            self._assert_finite_training_values(
                step,
                loss=loss,
                reconstruction=reconstruction,
                gate_penalty=gate_penalty,
                temporal_smoothness=smoothness,
                q_values=q_values,
                gates=gates,
            )
            loss.backward()
            self._assert_router_only_gradients(router, torch)
            optimizer.step()
            self._assert_finite_router(router, step, torch)
            append_jsonl(paths["training_metrics"], {
                "schema": "ai-toolkit.ideogram4-v3-phase-c-training-metric",
                "schema_version": 1,
                "step": step,
                "split": "train",
                "item_id": item["dataset_relative_item_id"],
                "timestep": timestep,
                "timestep_bin": bin_index,
                "noise_seed": self.phase_c.seed + step,
                "loss": float(loss.detach().item()),
                "reconstruction_loss": float(reconstruction.detach().item()),
                "gate_penalty": float(gate_penalty.detach().item()),
                "temporal_smoothness": float(smoothness.detach().item()),
                "mean_q": float(q_values.detach().mean().item()),
                "mean_abs_q": float(q_values.detach().abs().mean().item()),
                "max_abs_q": float(q_values.detach().abs().max().item()),
                "std_q": float(q_values.detach().float().std(unbiased=False).item()),
                "saturation_fraction": float((q_values.detach().abs() >= self.phase_c.q_max * 0.99).float().mean().item()),
                "gate_min": float(gates.detach().min().item()),
                "gate_max": float(gates.detach().max().item()),
                "gate_mean": float(gates.detach().mean().item()),
            })
            progress.set_postfix(
                loss=f"{float(loss.detach().item()):.4g}",
                recon=f"{float(reconstruction.detach().item()):.4g}",
                abs_q=f"{float(q_values.detach().abs().mean().item()):.3g}",
                timestep=timestep,
                refresh=False,
            )
            validation_due = step % self.phase_c.validation.every == 0 or step == self.phase_c.steps
            if validation_due:
                progress.write(f"[Phase C] validating checkpoint candidate at step {step}")
                self._validate(router, validation_items, step, registry_fingerprint, paths, torch)
                progress.write(f"[Phase C] validation completed at step {step}")
            if validation_due or step % self.phase_c.checkpoint_every == 0:
                progress.write(f"[Phase C] saving checkpoint at step {step}")
                self._save_checkpoint(
                    router,
                    optimizer,
                    step,
                    registry_fingerprint,
                    run_fingerprint,
                    router_config,
                    torch,
                )
                progress.write(f"[Phase C] checkpoint saved at step {step}")
        progress.close()

        self._phase_c_progress("training loop finished; selecting best checkpoint and finalizing artifacts")
        self._finalize(router, registry_fingerprint, router_config, paths, torch)
        self._phase_c_progress(
            f"COMPLETED {self.phase_c.steps}/{self.phase_c.steps} steps; "
            f"best/final artifacts and completion manifest written to {self.output_root}"
        )

    def _resolve_phase_c_inputs(self) -> None:
        self.residual_manifest_path = require_file_hash(
            self.phase_c.residual_manifest, sha256_file(self.phase_c.residual_manifest), "residual ablation manifest"
        )
        residual_manifest = load_json(self.residual_manifest_path)
        if residual_manifest.get("schema") != "ai-toolkit.ideogram4-v3-residual-ablation":
            raise RuntimeError("invalid residual ablation manifest schema")
        if int(residual_manifest.get("schema_version", 0)) < 5:
            raise RuntimeError("Phase C requires residual ablation manifest schema version 5 or newer")
        if residual_manifest.get("status") != "completed" or not residual_manifest.get("diagnostics_only"):
            raise RuntimeError("Phase C requires a completed diagnostics-only residual ablation")
        residual_inputs = residual_manifest.get("inputs", {})
        for key in ("a2_contract", "a2_snapshot", "v3_weights", "best_embedding", "best_te_adapter", "best_manifest"):
            current_ref = self.input_refs.get(key, {})
            residual_ref = residual_inputs.get(key, {})
            if current_ref.get("sha256") != residual_ref.get("sha256"):
                raise RuntimeError(f"residual ablation input disagrees with current Phase C source: {key}")
        self.registry_path = Path(self.phase_c.registry)
        if not self.registry_path.is_file():
            raise FileNotFoundError(self.registry_path)
        recorded_ref = residual_manifest.get("artifacts", {}).get("registry", {})
        manifest_registry_path = (self.residual_manifest_path.parent / str(recorded_ref.get("path", ""))).resolve()
        if manifest_registry_path != self.registry_path.resolve():
            raise RuntimeError("configured registry path does not match residual ablation manifest artifact")
        if recorded_ref.get("sha256") != sha256_file(self.registry_path):
            raise RuntimeError("configured registry does not match residual ablation manifest")
        self.recorded_registry = load_json(self.registry_path)
        self.split_manifest_path = Path(self.phase_c.split_manifest)
        self.dataset_root = Path(self.phase_c.dataset_root)
        if not self.dataset_root.is_dir():
            raise NotADirectoryError(self.dataset_root)

    def _build_canonical_router(self, group_count: int):
        values = {
            "group_count": group_count,
            "timestep_embed_dim": self.phase_c.timestep_embed_dim,
            "hidden_dim": self.phase_c.hidden_dim,
            "router_rank": self.phase_c.router_rank,
            "q_max": self.phase_c.q_max,
        }
        signature = inspect.signature(ResidualGateRouter)
        supported = {key: value for key, value in values.items() if key in signature.parameters}
        return ResidualGateRouter(**supported)

    def _build_optimizer(self, router):
        from toolkit.optimizer import get_optimizer

        optimizer_params = dict(self.phase_c.optimizer_params)
        optimizer_params["weight_decay"] = self.phase_c.weight_decay
        try:
            optimizer = get_optimizer(
                router.parameters(),
                self.phase_c.optimizer,
                learning_rate=self.phase_c.learning_rate,
                optimizer_params=optimizer_params,
            )
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                f"Phase C optimizer {self.phase_c.optimizer!r} requires an unavailable dependency: {error}"
            ) from error
        if optimizer is None or not hasattr(optimizer, "step") or not hasattr(optimizer, "state_dict"):
            raise RuntimeError(f"Phase C optimizer factory returned an invalid optimizer for {self.phase_c.optimizer!r}")
        return optimizer

    def _seed_everything(self, torch) -> None:
        random.seed(self.phase_c.seed)
        torch.manual_seed(self.phase_c.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.phase_c.seed)

    def _run_fingerprint(self, source_manifest, router_config) -> str:
        payload = {
            "run_label": self.run_label,
            "source": source_manifest,
            "router_config": router_config,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _sanitize_resume_artifacts(self, paths, start_step: int) -> None:
        for key in ("training_metrics", "validation_metrics", "gate_profiles"):
            records = load_jsonl(paths[key])
            kept = [record for record in records if int(record.get("step", -1)) <= int(start_step)]
            if len(kept) != len(records) or (start_step == 0 and records):
                rewrite_jsonl(paths[key], kept)
        for key in ("handoff_manifest", "completion_manifest"):
            try:
                paths[key].unlink()
            except FileNotFoundError:
                pass
        for directory_name in (self.phase_c.artifacts.best_dir, self.phase_c.artifacts.final_dir):
            directory = self.output_root / directory_name
            if directory.is_dir():
                shutil.rmtree(directory)

    def _assert_runtime_invariants(self, router, item, registry_fingerprint, start_step, torch) -> None:
        normalized = torch.tensor([0.5], device=self.sd.device_torch, dtype=torch.float32)
        with torch.no_grad():
            q_values = router(normalized)
        if q_values.shape != (1, router.group_count):
            raise RuntimeError(f"Phase C router produced invalid shape {tuple(q_values.shape)}")
        if not bool(torch.isfinite(q_values).all()) or bool((q_values.abs() > self.phase_c.q_max + 1.0e-6).any()):
            raise RuntimeError("Phase C router output is non-finite or outside q_max")
        if start_step == 0 and not torch.equal(q_values, torch.zeros_like(q_values)):
            raise RuntimeError("fresh Phase C router must initialize to exact q=0 identity")

        prepared = self._prepare_training_case(item, 500, self.phase_c.seed, torch)
        identity_runtime = ResidualGateRuntime(torch.ones_like(q_values), registry_fingerprint=registry_fingerprint)
        zero_runtime = ResidualGateRuntime(torch.zeros_like(q_values), registry_fingerprint=registry_fingerprint)
        with torch.no_grad():
            with residual_gate_runtime_context(self.network, identity_runtime):
                identity_prediction = self._predict(prepared, prepared["prompt_template"], "full")
            with residual_gate_runtime_context(self.network, zero_runtime):
                zero_prediction = self._predict(prepared, prepared["prompt_template"], "full")
            was_active = self.network.is_active
            try:
                self.network.is_active = False
                disabled_prediction = self._predict(prepared, prepared["prompt_template"], "full")
            finally:
                self.network.is_active = was_active
        if not torch.allclose(zero_prediction.float(), disabled_prediction.float(), rtol=1.0e-4, atol=1.0e-5):
            difference = float((zero_prediction.float() - disabled_prediction.float()).abs().max().item())
            raise RuntimeError(f"zero residual gates do not match disabled V3; max_abs_difference={difference}")
        if start_step == 0:
            with torch.no_grad():
                ungated_prediction = self._predict(prepared, prepared["prompt_template"], "full")
            if not torch.allclose(identity_prediction.float(), ungated_prediction.float(), rtol=1.0e-4, atol=1.0e-5):
                difference = float((identity_prediction.float() - ungated_prediction.float()).abs().max().item())
                raise RuntimeError(f"identity residual gates do not match ordinary V3; max_abs_difference={difference}")
        if not bool(torch.isfinite(identity_prediction).all()):
            raise RuntimeError("Phase C preflight prediction is non-finite")

    def _assert_finite_training_values(self, step: int, **values) -> None:
        for name, value in values.items():
            if not bool(value.detach().isfinite().all()):
                raise RuntimeError(f"Phase C step {step} produced non-finite {name}")

    def _assert_finite_router(self, router, step: int, torch) -> None:
        invalid = [name for name, parameter in router.named_parameters() if not bool(torch.isfinite(parameter).all())]
        if invalid:
            raise RuntimeError(f"Phase C step {step} produced non-finite router parameters: {invalid}")

    def _assert_router_only_trainable(self, router) -> None:
        offenders = []
        for label, module in (
            ("model", self.sd.model),
            ("text_encoder", self.sd.text_encoder),
            ("vae", self.sd.vae),
            ("v3", self.network),
            ("activator", self.text_activator),
        ):
            offenders.extend(f"{label}.{name}" for name, parameter in module.named_parameters() if parameter.requires_grad)
        if offenders:
            raise RuntimeError(f"Phase C found non-router trainable parameters: {offenders[:10]}")
        if not any(parameter.requires_grad for parameter in router.parameters()):
            raise RuntimeError("canonical router has no trainable parameters")

    def _assert_router_only_gradients(self, router, torch) -> None:
        router_gradients = [parameter.grad for parameter in router.parameters() if parameter.requires_grad]
        if not router_gradients or not any(
            gradient is not None and bool(torch.isfinite(gradient).all()) and float(gradient.float().norm().item()) > 0.0
            for gradient in router_gradients
        ):
            raise RuntimeError("Phase C router did not receive a finite non-zero gradient")
        offenders = []
        for label, module in (
            ("model", self.sd.model),
            ("text_encoder", self.sd.text_encoder),
            ("vae", self.sd.vae),
            ("v3", self.network),
            ("activator", self.text_activator),
        ):
            offenders.extend(f"{label}.{name}" for name, parameter in module.named_parameters() if parameter.grad is not None)
        if offenders:
            raise RuntimeError(f"Phase C frozen parameters received gradients: {offenders[:10]}")

    def _load_items(self):
        from extensions_built_in.ideogram4_tap_ablation.helpers import select_probe_items
        from toolkit.trigger_data_split import load_data_split_manifest

        split = load_data_split_manifest(self.split_manifest_path, expected_seed=self.phase_c.seed)
        train_limit = min(
            len(split.train_item_ids),
            self.phase_c.train_item_limit or len(split.train_item_ids),
        )
        validation_limit = min(
            len(split.heldout_item_ids),
            self.phase_c.validation_item_limit or len(split.heldout_item_ids),
        )
        items = select_probe_items(
            self.dataset_root,
            split.as_dict(),
            train_limit=train_limit,
            expected_heldout=len(split.heldout_item_ids),
            placeholder=self.placeholder,
        )
        if validation_limit < len(split.heldout_item_ids):
            heldout_ids = set(sorted(split.heldout_item_ids)[:validation_limit])
            items = [
                item
                for item in items
                if item["split"] == "train" or item["dataset_relative_item_id"] in heldout_ids
            ]
        train = [item for item in items if item["split"] == "train"]
        validation = [{**item, "split": "validation"} for item in items if item["split"] == "heldout"]
        if not train or not validation:
            raise RuntimeError("Phase C requires non-empty train and validation splits")
        return train, validation

    def _precompute_items(self, items, torch) -> None:
        from tqdm.auto import tqdm

        for item in tqdm(items, desc="Phase C precompute", unit="item", dynamic_ncols=True, leave=True):
            item_id = item["dataset_relative_item_id"]
            case = SimpleNamespace(item_id=item_id, timestep=0, noise_seed=self.phase_c.seed)
            prepared = self._prepare_case(case, item, torch)
            with torch.no_grad():
                self._calibrated_embed_cache[item_id] = self._calibrated_prompt_embeds(prepared["prompt_template"]).to("cpu")
            self._latent_cache[item_id] = self._latent_cache[item_id].to("cpu")
        from toolkit.basic import flush

        self.sd.text_encoder.to("cpu")
        self.text_activator.to("cpu")
        self.sd.vae.to("cpu")
        flush()

    def _prepare_training_case(self, item, timestep, noise_seed, torch):
        case = SimpleNamespace(
            item_id=item["dataset_relative_item_id"],
            timestep=int(timestep),
            noise_seed=int(noise_seed),
        )
        prepared = self._prepare_case(case, item, torch)
        prepared["calibrated_embeds"] = self._calibrated_embed_cache[case.item_id]
        return prepared

    def _validate(self, router, items, step, registry_fingerprint, paths, torch) -> None:
        router.eval()
        profiles = {}
        with torch.no_grad():
            for spec in validation_grid(
                self.phase_c.validation.canonical_timesteps,
                self.phase_c.validation.dense_timesteps,
                self.phase_c.validation.seeds,
            ):
                normalized = torch.tensor([spec["timestep"] / 1000.0], device=self.sd.device_torch)
                q_values = router(normalized)
                gates = 1.0 + q_values
                profiles[(spec["grid"], spec["timestep"])] = gates.detach().float().cpu()[0].tolist()
                if spec["grid"] != "canonical":
                    continue
                runtime = ResidualGateRuntime(gates, registry_fingerprint=registry_fingerprint)
                for item in items:
                    prepared = self._prepare_training_case(item, spec["timestep"], spec["seed"], torch)
                    with residual_gate_runtime_context(self.network, runtime):
                        prediction = self._predict(prepared, prepared["prompt_template"], "full")
                    identity_runtime = ResidualGateRuntime(torch.ones_like(gates), registry_fingerprint=registry_fingerprint)
                    with residual_gate_runtime_context(self.network, identity_runtime):
                        baseline_prediction = self._predict(prepared, prepared["prompt_template"], "full")
                    loss = torch.mean((prediction.float() - prepared["target"].float()).square())
                    baseline_loss = torch.mean((baseline_prediction.float() - prepared["target"].float()).square())
                    normalized_gain = 1.0 - loss / baseline_loss.clamp_min(1.0e-12)
                    append_jsonl(paths["validation_metrics"], {
                        "schema": "ai-toolkit.ideogram4-v3-phase-c-validation-metric",
                        "schema_version": 1,
                        "step": step,
                        "split": "validation",
                        "grid": spec["grid"],
                        "item_id": item["dataset_relative_item_id"],
                        "timestep": spec["timestep"],
                        "noise_seed": spec["seed"],
                        "loss": float(loss.item()),
                        "normal_v3_loss": float(baseline_loss.item()),
                        "normalized_improvement_vs_v3": float(normalized_gain.item()),
                        "positive_improvement": bool(loss.item() < baseline_loss.item()),
                    })
            for (grid, timestep), gates in sorted(profiles.items()):
                append_jsonl(paths["gate_profiles"], {
                    "schema": "ai-toolkit.ideogram4-v3-phase-c-gate-profile",
                    "schema_version": 1,
                    "step": step,
                    "split": "validation",
                    "grid": grid,
                    "timestep": timestep,
                    "style_strength": 1.0,
                    "q": [float(value - 1.0) for value in gates],
                    "gates": gates,
                })
        router.train()

    def _artifact_paths(self) -> Dict[str, Path]:
        artifacts = self.phase_c.artifacts
        return {
            "training_metrics": self.output_root / artifacts.training_metrics,
            "validation_metrics": self.output_root / artifacts.validation_metrics,
            "gate_profiles": self.output_root / artifacts.gate_profiles,
            "source_manifest": self.output_root / artifacts.source_manifest,
            "handoff_manifest": self.output_root / artifacts.handoff_manifest,
            "completion_manifest": self.output_root / artifacts.completion_manifest,
            "router_config": self.output_root / artifacts.router_config_filename,
        }

    def _safe_checkpoint_dir(self, step: int) -> Path:
        root = (self.output_root / self.phase_c.artifacts.checkpoints_dir).resolve()
        path = (root / f"step_{int(step):06d}").resolve()
        if root not in path.parents or path.parent != root:
            raise RuntimeError("unsafe Phase C checkpoint path")
        return path

    def _save_checkpoint(self, router, optimizer, step, fingerprint, run_fingerprint, router_config, torch) -> None:
        from safetensors.torch import save_file

        destination = self._safe_checkpoint_dir(step)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir = destination.parent / f".{destination.name}.tmp"
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        temporary_dir.mkdir()
        try:
            router_path = temporary_dir / self.phase_c.artifacts.router_filename
            save_file({key: value.detach().cpu() for key, value in router.state_dict().items()}, str(router_path))
            torch.save({
                "schema": "ai-toolkit.ideogram4-v3-phase-c-checkpoint",
                "schema_version": 1,
                "step": int(step),
                "registry_fingerprint": fingerprint,
                "run_fingerprint": run_fingerprint,
                "optimizer": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "python_rng_state": random.getstate(),
            }, temporary_dir / "training_state.pt")
            atomic_write_json(temporary_dir / self.phase_c.artifacts.router_config_filename, router_config)
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(temporary_dir, destination)
        except BaseException:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
            raise

    def _resume(self, router, optimizer, fingerprint, run_fingerprint, torch) -> int:
        root = self.output_root / self.phase_c.artifacts.checkpoints_dir
        if not root.is_dir():
            return 0
        candidates = []
        for path in root.glob("step_*"):
            if not path.is_dir():
                continue
            try:
                step = int(path.name.removeprefix("step_"))
            except ValueError:
                continue
            required = (
                path / "training_state.pt",
                path / self.phase_c.artifacts.router_filename,
                path / self.phase_c.artifacts.router_config_filename,
            )
            if all(candidate.is_file() for candidate in required):
                candidates.append((step, path))
        if not candidates:
            return 0
        checkpoint_root = root.resolve()
        _, latest_path = max(candidates, key=lambda item: item[0])
        latest = latest_path.resolve()
        if checkpoint_root not in latest.parents or latest.parent != checkpoint_root:
            raise RuntimeError("unsafe Phase C resume checkpoint path")
        state = torch.load(latest / "training_state.pt", map_location="cpu", weights_only=False)
        if state.get("registry_fingerprint") != fingerprint:
            raise RuntimeError("checkpoint active registry fingerprint mismatch")
        if state.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError("checkpoint Phase C source/config fingerprint mismatch")
        if int(state.get("step", -1)) < 0:
            raise RuntimeError("checkpoint Phase C step is invalid")
        from safetensors.torch import load_file

        router.load_state_dict(load_file(str(latest / self.phase_c.artifacts.router_filename), device="cpu"), strict=True)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng_state"])
        random.setstate(state["python_rng_state"])
        return int(state["step"])

    def _source_manifest(self, active_registry, fingerprint):
        return {
            "schema": "ai-toolkit.ideogram4-v3-phase-c-source",
            "schema_version": 1,
            "status": "resolved",
            "inputs": {
                **self.input_refs,
                "residual_manifest": {"path": str(self.residual_manifest_path), "sha256": sha256_file(self.residual_manifest_path)},
                "registry": {"path": str(self.registry_path), "sha256": sha256_file(self.registry_path)},
                "split_manifest": {"path": str(self.split_manifest_path), "sha256": sha256_file(self.split_manifest_path)},
            },
            "active_registry_fingerprint": fingerprint,
            "active_registry": active_registry,
            "frozen": ["Ideogram4", "V3", "A2 best-heldout activator"],
            "trainable": ["ResidualGateRouter"],
        }

    def _read_validation_records(self, path: Path):
        return load_jsonl(path)

    def _validate_completion_records(self, validation_records, validation_item_count: int) -> None:
        expected_per_step = (
            validation_item_count
            * len(self.phase_c.validation.seeds)
            * len(self.phase_c.validation.canonical_timesteps)
        )
        validation_steps = {
            step
            for step in range(1, self.phase_c.steps + 1)
            if step % self.phase_c.validation.every == 0 or step == self.phase_c.steps
        }
        keys = []
        counts = {}
        for record in validation_records:
            if record.get("split") != "validation" or record.get("grid") != "canonical":
                continue
            step = int(record["step"])
            key = (step, str(record["item_id"]), int(record["timestep"]), int(record["noise_seed"]))
            keys.append(key)
            counts[step] = counts.get(step, 0) + 1
        if len(keys) != len(set(keys)):
            raise RuntimeError("Phase C canonical validation records contain duplicate probe keys")
        if set(counts) != validation_steps:
            raise RuntimeError(
                f"Phase C canonical validation steps are incomplete: expected={sorted(validation_steps)}, actual={sorted(counts)}"
            )
        invalid = {step: count for step, count in counts.items() if count != expected_per_step}
        if invalid:
            raise RuntimeError(
                f"Phase C canonical validation probe counts are incomplete; expected_per_step={expected_per_step}, actual={invalid}"
            )

    def _copy_router_artifacts(self, source: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for filename in (self.phase_c.artifacts.router_filename, self.phase_c.artifacts.router_config_filename):
            temporary = destination / (filename + ".tmp")
            shutil.copyfile(source / filename, temporary)
            os.replace(temporary, destination / filename)

    def _finalize(self, router, fingerprint, router_config, paths, torch) -> None:
        final_checkpoint = self._safe_checkpoint_dir(self.phase_c.steps)
        final_dir = self.output_root / self.phase_c.artifacts.final_dir
        self._copy_router_artifacts(final_checkpoint, final_dir)
        validation_records = self._read_validation_records(paths["validation_metrics"])
        self._validate_completion_records(validation_records, self._validation_item_count)
        best_step, best_loss = select_best_validation(validation_records)
        best_checkpoint = self._safe_checkpoint_dir(best_step)
        best_dir = self.output_root / self.phase_c.artifacts.best_dir
        self._copy_router_artifacts(best_checkpoint, best_dir)
        handoff = {
            "schema": "ai-toolkit.ideogram4-v3-phase-c-handoff",
            "schema_version": 1,
            "status": "completed",
            "active_registry_fingerprint": fingerprint,
            "best": {
                "step": best_step,
                "canonical_validation_loss": best_loss,
                "router": artifact_ref(best_dir / self.phase_c.artifacts.router_filename, self.output_root),
                "config": artifact_ref(best_dir / self.phase_c.artifacts.router_config_filename, self.output_root),
            },
            "final": {
                "step": self.phase_c.steps,
                "router": artifact_ref(final_dir / self.phase_c.artifacts.router_filename, self.output_root),
                "config": artifact_ref(final_dir / self.phase_c.artifacts.router_config_filename, self.output_root),
            },
        }
        atomic_write_json(paths["handoff_manifest"], handoff)
        atomic_write_json(paths["completion_manifest"], {
            "schema": "ai-toolkit.ideogram4-v3-phase-c-completion",
            "schema_version": 1,
            "status": "completed",
            "steps": self.phase_c.steps,
            "optimizer": self.phase_c.optimizer,
            "optimizer_params": dict(self.phase_c.optimizer_params),
            "fresh_optimizer": True,
            "router_only": True,
            "validation_term": "validation",
            "artifacts": {
                "source": artifact_ref(paths["source_manifest"], self.output_root),
                "handoff": artifact_ref(paths["handoff_manifest"], self.output_root),
                "training_metrics": artifact_ref(paths["training_metrics"], self.output_root),
                "validation_metrics": artifact_ref(paths["validation_metrics"], self.output_root),
                "gate_profiles": artifact_ref(paths["gate_profiles"], self.output_root),
            },
        })
