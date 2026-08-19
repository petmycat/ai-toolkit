from __future__ import annotations

import hashlib
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
    aggregate_activator_occurrences,
    active_registry_fingerprint,
    bind_active_registry,
    build_module_registry,
    filter_active_registry,
    effective_gates,
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
        legacy_config = self.get_conf("phase_c_router", None)
        if legacy_config is not None:
            raise ValueError("legacy phase_c_router configuration is not supported; use phase_c_v2")
        self.phase_c = load_config(self.get_conf("phase_c_v2", required=True))
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
        self._projected_activator_cache = {}
        self._projected_activator_fingerprints = {}
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
        self._active_registry_payload = active_payload
        registry_fingerprint = active_registry_fingerprint(active_registry)
        group_count = int(active_payload["group_count"])

        self._phase_c_progress(
            f"active registry ready: {group_count} groups across {len(active_registry)} LoRA modules"
        )
        self._seed_everything(torch)
        conditioning_dim = int(self.sd.model.config.emb_dim)
        router = self._build_canonical_router(group_count, conditioning_dim).to(
            self.sd.device_torch, dtype=torch.float32
        )
        optimizer = self._build_optimizer(router)
        self._phase_c_progress(
            f"router and optimizer ready: optimizer={self.phase_c.optimizer}, steps={self.phase_c.steps}"
        )
        paths = self._artifact_paths()
        train_items, validation_items = self._load_items()
        self._validation_item_count = len(validation_items)
        self._validation_item_ids = tuple(item["dataset_relative_item_id"] for item in validation_items)
        source_manifest = self._source_manifest(active_payload, registry_fingerprint)
        router_config = router_config_payload(self.phase_c, active_payload, conditioning_dim)
        router_config["data_selection"] = {
            "train_item_ids": [item["dataset_relative_item_id"] for item in train_items],
            "validation_item_ids": list(self._validation_item_ids),
        }
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
        self._assert_output_root_schema(paths, start_step, run_fingerprint)
        self._sanitize_resume_artifacts(paths, start_step)
        if start_step:
            self._validate_resume_prefix(paths, start_step)
            self._phase_c_progress(f"resumed compatible checkpoint at step {start_step}/{self.phase_c.steps}")
        else:
            self._phase_c_progress("starting from fresh zero-initialized router")

        self._phase_c_progress(
            f"precomputing {len(train_items)} train and {len(validation_items)} validation items"
        )
        self._precompute_items(train_items + validation_items, torch)
        self._phase_c_progress("running real-model residual-gate preflight invariants")
        self._assert_runtime_invariants(router, train_items, registry_fingerprint, start_step, torch)
        self._phase_c_progress("preflight invariants passed; entering router training")
        source_manifest["run_fingerprint"] = run_fingerprint
        atomic_write_json(paths["source_manifest"], source_manifest)
        atomic_write_json(paths["group_registry"], active_payload)
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
            selected_items = self._select_same_timestep_items(train_items, step)
            normalized = torch.tensor([timestep / 1000.0], device=self.sd.device_torch, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            reconstructions = []
            contextual_penalties = []
            q_rows = []
            universal_rows = []
            contextual_rows = []
            activator_inputs = []
            item_ids = []
            for content_index, item in enumerate(selected_items):
                noise_seed = self.phase_c.seed + step
                prepared = self._prepare_training_case(item, timestep, noise_seed, torch)
                projected = self._projected_activator_cache[item["dataset_relative_item_id"]].to(
                    self.sd.device_torch, dtype=torch.float32
                )
                activator_code = router.encode_activator(projected)
                universal_q, contextual_q, q_values = router.components(normalized, activator_code)
                gates = 1.0 + q_values
                runtime = ResidualGateRuntime(gates, registry_fingerprint=registry_fingerprint)
                with residual_gate_runtime_context(self.network, runtime):
                    prediction = self._predict(prepared, prepared["prompt_template"], "full")
                reconstruction = torch.mean((prediction.float() - prepared["target"].float()).square())
                contextual_penalty = contextual_q.float().square().mean()
                micro_loss = (
                    reconstruction
                    + self.phase_c.lambda_contextual * contextual_penalty
                ) / float(len(selected_items))
                self._assert_finite_training_values(
                    step,
                    reconstruction=reconstruction,
                    contextual_penalty=contextual_penalty,
                    q_values=q_values,
                    gates=gates,
                )
                micro_loss.backward()
                reconstructions.append(reconstruction.detach())
                contextual_penalties.append(contextual_penalty.detach())
                q_rows.append(q_values.detach())
                universal_rows.append(universal_q.detach())
                contextual_rows.append(contextual_q.detach())
                activator_inputs.append(projected.detach())
                item_ids.append(item["dataset_relative_item_id"])
            batch_codes = router.encode_activator(torch.cat(activator_inputs, dim=0))
            _, contextual_batch, _ = router.components(normalized, batch_codes)
            contextual_mean_penalty = contextual_batch.mean(dim=0).square().mean()
            contextual_mean_regularization = (
                self.phase_c.lambda_contextual
                * self.phase_c.contextual_mean_multiplier
                * contextual_mean_penalty
            )
            self._assert_finite_training_values(
                step,
                contextual_mean_penalty=contextual_mean_penalty,
                contextual_mean_regularization=contextual_mean_regularization,
            )
            contextual_mean_regularization.backward()
            universal_penalty = router.universal_penalty()
            smoothness = router.universal_temporal_smoothness() if self.phase_c.temporal_smoothness.enabled else universal_penalty.new_zeros(())
            regularization = (
                self.phase_c.lambda_universal * universal_penalty
                + self.phase_c.temporal_smoothness.weight * smoothness
            )
            self._assert_finite_training_values(
                step,
                universal_penalty=universal_penalty,
                temporal_universal_smoothness=smoothness,
                regularization=regularization,
            )
            regularization.backward()
            self._assert_router_only_gradients(router, torch)
            optimizer.step()
            self._assert_finite_optimizer(optimizer, step, torch)
            self._assert_finite_router(router, step, torch)
            reconstruction = torch.stack(reconstructions).mean()
            contextual_penalty = torch.stack(contextual_penalties).mean()
            q_values = torch.cat(q_rows, dim=0)
            universal_q = torch.cat(universal_rows, dim=0)
            contextual_q = torch.cat(contextual_rows, dim=0)
            loss = (
                reconstruction
                + self.phase_c.lambda_contextual * contextual_penalty
                + contextual_mean_regularization.detach()
                + regularization.detach()
            )
            append_jsonl(paths["training_metrics"], {
                "schema": "ai-toolkit.ideogram4-v3-phase-c-v2-training-metric",
                "schema_version": 2,
                "step": step,
                "split": "train",
                "item_ids": item_ids,
                "same_timestep_content_batch": len(item_ids),
                "distinct_item_count": len(set(item_ids)),
                "timestep": timestep,
                "timestep_bin": bin_index,
                "loss": float(loss.item()),
                "reconstruction_loss": float(reconstruction.item()),
                "universal_penalty": float(universal_penalty.detach().item()),
                "contextual_penalty": float(contextual_penalty.item()),
                "contextual_batch_mean_penalty": float(contextual_mean_penalty.detach().item()),
                "temporal_universal_smoothness": float(smoothness.detach().item()),
                "mean_abs_q_universal": float(universal_q.abs().mean().item()),
                "mean_abs_q_contextual": float(contextual_q.abs().mean().item()),
                "mean_abs_q_total": float(q_values.abs().mean().item()),
                "max_abs_q": float(q_values.abs().max().item()),
                "saturation_fraction": float((q_values.abs() >= self.phase_c.q_max * 0.99).float().mean().item()),
            })
            progress.set_postfix(
                loss=f"{float(loss.item()):.4g}",
                recon=f"{float(reconstruction.item()):.4g}",
                abs_q=f"{float(q_values.abs().mean().item()):.3g}",
                contents=len(item_ids),
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

    def _build_canonical_router(self, group_count: int, conditioning_dim: int):
        return ResidualGateRouter(
            group_count=group_count,
            conditioning_dim=conditioning_dim,
            activator_token_count=self.phase_c.activator_token_count,
            activator_token_dim=self.phase_c.activator_token_dim,
            temporal_anchor_count=self.phase_c.temporal_anchor_count,
            contextual_rank=self.phase_c.contextual_rank,
            q_max=self.phase_c.q_max,
        )

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

    def _assert_output_root_schema(self, paths, start_step: int, run_fingerprint: str) -> None:
        expected = {
            "source_manifest": "ai-toolkit.ideogram4-v3-phase-c-v2-source",
            "router_config": "ai-toolkit.ideogram4-v3-phase-c-v2-router-config",
        }
        for key, schema in expected.items():
            path = paths[key]
            if not path.is_file():
                continue
            payload = load_json(path)
            if payload.get("schema") != schema or int(payload.get("schema_version", 0)) != 2:
                raise RuntimeError(
                    f"output root contains legacy or incompatible Phase C artifact: {path.name}; "
                    "use a fresh Phase C V2 output_root"
                )
            if key == "router_config" and int(payload.get("contract_revision", 0)) != 4:
                raise RuntimeError("output root predates the audited Phase C V2 contract revision; use a fresh output_root")
            recorded_fingerprint = payload.get("run_fingerprint")
            if start_step > 0 and not recorded_fingerprint:
                raise RuntimeError(f"resumed Phase C V2 artifact lacks run_fingerprint: {path.name}")
            if start_step > 0 and recorded_fingerprint != run_fingerprint:
                raise RuntimeError(f"resumed Phase C V2 artifact run_fingerprint mismatch: {path.name}")
        if start_step == 0:
            for key in ("training_metrics", "validation_metrics", "temporal_profiles", "contextual_profiles"):
                records = load_jsonl(paths[key])
                if records and any("phase-c-v2" not in str(record.get("schema", "")) for record in records):
                    raise RuntimeError(
                        f"output root contains legacy Phase C records in {paths[key].name}; "
                        "use a fresh Phase C V2 output_root"
                    )

    def _sanitize_resume_artifacts(self, paths, start_step: int) -> None:
        checkpoint_root = self.output_root / self.phase_c.artifacts.checkpoints_dir
        if start_step == 0 and checkpoint_root.is_dir():
            shutil.rmtree(checkpoint_root)
        for key in ("training_metrics", "validation_metrics", "temporal_profiles", "contextual_profiles"):
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

    def _validate_resume_prefix(self, paths, start_step: int) -> None:
        training_records = load_jsonl(paths["training_metrics"])
        steps = [int(record.get("step", -1)) for record in training_records]
        if steps != list(range(1, start_step + 1)):
            raise RuntimeError("Phase C resume training-metric prefix is incomplete or unordered")
        completed_validation_steps = {
            step for step in range(1, start_step + 1)
            if step % self.phase_c.validation.every == 0 or step == self.phase_c.steps
        }
        validation_records = load_jsonl(paths["validation_metrics"])
        actual_validation_steps = {int(record.get("step", -1)) for record in validation_records}
        if actual_validation_steps != completed_validation_steps:
            raise RuntimeError("Phase C resume validation prefix is incomplete")
        temporal_records = load_jsonl(paths["temporal_profiles"])
        if {int(record.get("step", -1)) for record in temporal_records} != completed_validation_steps:
            raise RuntimeError("Phase C resume temporal-profile prefix is incomplete")

    def _assert_runtime_invariants(self, router, items, registry_fingerprint, start_step, torch) -> None:
        item = items[0]
        normalized = torch.tensor([0.5], device=self.sd.device_torch, dtype=torch.float32)
        projected = self._projected_activator_cache[item["dataset_relative_item_id"]].to(
            self.sd.device_torch, dtype=torch.float32
        )
        with torch.no_grad():
            activator_code = router.encode_activator(projected)
            repeated_code = router.encode_activator(projected.clone())
            q_values = router(normalized, activator_code)
        if not torch.allclose(activator_code, repeated_code, rtol=0.0, atol=1.0e-7):
            raise RuntimeError("Phase C V2 activator encoding is not deterministic")
        fingerprints = [self._projected_activator_fingerprints[item["dataset_relative_item_id"]] for item in items]
        if len(fingerprints) != len(set(fingerprints)):
            raise RuntimeError("different captions produced bit-identical projected activator states")
        if len(items) > 1:
            other_projected = self._projected_activator_cache[items[1]["dataset_relative_item_id"]].to(
                self.sd.device_torch, dtype=torch.float32
            )
            with torch.no_grad():
                other_code = router.encode_activator(other_projected)
            if torch.equal(activator_code, other_code):
                raise RuntimeError("different captions produced bit-identical activator bottleneck codes")
        midpoint_gates = effective_gates(q_values, 0.5)
        if not torch.equal(midpoint_gates, torch.ones_like(midpoint_gates)):
            raise RuntimeError("Phase C V2 style_strength midpoint does not produce exact V3 gates")
        if q_values.shape != (1, router.group_count):
            raise RuntimeError(f"Phase C V2 router produced invalid shape {tuple(q_values.shape)}")
        if not bool(torch.isfinite(q_values).all()) or bool((q_values.abs() > self.phase_c.q_max + 1.0e-6).any()):
            raise RuntimeError("Phase C V2 router output is non-finite or outside q_max")
        if start_step == 0 and not torch.equal(q_values, torch.zeros_like(q_values)):
            raise RuntimeError("fresh Phase C V2 router must initialize to exact q=0 identity")

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

    def _assert_finite_optimizer(self, optimizer, step: int, torch) -> None:
        invalid = []
        for parameter_index, state in enumerate(optimizer.state.values()):
            for name, value in state.items():
                if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                    invalid.append(f"state[{parameter_index}].{name}")
        if invalid:
            raise RuntimeError(f"Phase C step {step} produced non-finite optimizer state: {invalid[:10]}")

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
        if len(train) < self.phase_c.same_timestep_content_batch:
            raise RuntimeError("Phase C training split is smaller than same_timestep_content_batch")
        if self.phase_c.validation.evaluate_context_shuffle and len(validation) < 2:
            raise RuntimeError("Phase C context-shuffled validation requires at least two heldout items")
        return train, validation

    def _precompute_items(self, items, torch) -> None:
        from tqdm.auto import tqdm

        for item in tqdm(items, desc="Phase C precompute", unit="item", dynamic_ncols=True, leave=True):
            item_id = item["dataset_relative_item_id"]
            case = SimpleNamespace(item_id=item_id, timestep=0, noise_seed=self.phase_c.seed)
            prepared = self._prepare_case(case, item, torch)
            with torch.no_grad():
                embeds = self._calibrated_prompt_embeds(prepared["prompt_template"])
                self._calibrated_embed_cache[item_id] = embeds.to("cpu")
                projected = self._extract_projected_activator_states(embeds, torch).to("cpu")
                self._projected_activator_cache[item_id] = projected
                self._projected_activator_fingerprints[item_id] = hashlib.sha256(
                    projected.contiguous().numpy().tobytes()
                ).hexdigest()
            self._latent_cache[item_id] = self._latent_cache[item_id].to("cpu")
        from toolkit.basic import flush

        self.sd.text_encoder.to("cpu")
        self.text_activator.to("cpu")
        self.sd.vae.to("cpu")
        flush()

    def _extract_projected_activator_states(self, embeds, torch):
        features = embeds.text_embeds[0]
        mask = embeds.trigger_masks[0].to(dtype=torch.bool)
        if mask.shape[0] != features.shape[0]:
            raise RuntimeError("private activator mask length does not match cached Qwen conditioning")
        positions = torch.nonzero(mask, as_tuple=False).reshape(-1)
        token_count = self.phase_c.activator_token_count
        occurrence_count = self.phase_c.activator_occurrence_count
        expected_positions = token_count * occurrence_count
        if positions.numel() != expected_positions:
            raise RuntimeError(
                f"Phase C V2 requires {occurrence_count} activator occurrences with "
                f"{token_count} virtual-token positions each; expected {expected_positions}, "
                f"found {positions.numel()}"
            )
        transformer = self.sd.model
        selected = features[positions].to(self.sd.device_torch, dtype=self.sd.torch_dtype)
        projected = transformer.llm_cond_proj(transformer.llm_cond_norm(selected)).float()
        states = aggregate_activator_occurrences(
            projected,
            occurrence_count=occurrence_count,
            token_count=token_count,
            mode=self.phase_c.activator_occurrence_mode,
        )
        if not bool(torch.isfinite(states).all()):
            raise RuntimeError("projected private activator states are non-finite")
        return states

    def _select_same_timestep_items(self, train_items, step):
        count = self.phase_c.same_timestep_content_batch
        if len(train_items) < count:
            raise RuntimeError("training split cannot provide distinct same-timestep contents")
        batch_index = int(step) - 1
        batches_per_cycle = (len(train_items) + count - 1) // count
        cycle = batch_index // batches_per_cycle
        offset = (batch_index % batches_per_cycle) * count
        permutation = list(range(len(train_items)))
        random.Random((self.phase_c.seed << 32) ^ cycle ^ 0xD1B54A32D192ED03).shuffle(permutation)
        indices = [permutation[(offset + index) % len(permutation)] for index in range(count)]
        if len(set(indices)) != count:
            raise RuntimeError("same-timestep round-robin selected duplicate dataset items")
        return [train_items[index] for index in indices]

    def _prepare_training_case(self, item, timestep, noise_seed, torch):
        case = SimpleNamespace(
            item_id=item["dataset_relative_item_id"],
            timestep=int(timestep),
            noise_seed=int(noise_seed),
        )
        prepared = self._prepare_case(case, item, torch)
        prepared["calibrated_embeds"] = self._calibrated_embed_cache[case.item_id]
        return prepared

    def _validation_q(self, router, item_id, timestep, mode, shuffled_item_id=None):
        import torch

        normalized = torch.tensor([timestep / 1000.0], device=self.sd.device_torch, dtype=torch.float32)
        projected_id = shuffled_item_id if mode == "context_shuffled" else item_id
        projected = self._projected_activator_cache[projected_id].to(self.sd.device_torch, dtype=torch.float32)
        code = router.encode_activator(projected)
        universal_raw, contextual_raw, total = router.components(normalized, code)
        universal_applied = router.bound(universal_raw)
        if mode == "normal_v3":
            return torch.zeros_like(total), universal_raw, contextual_raw
        if mode == "universal_only":
            return universal_applied, universal_raw, torch.zeros_like(contextual_raw)
        return total, universal_raw, contextual_raw

    def _family_statistics(self, vector, torch):
        groups = self._active_registry_payload["groups"]
        statistics = {}
        families = sorted({str(group["group_id"]).rsplit(":", 1)[-1] for group in groups})
        for family in families:
            indices = [
                int(group["group_index"])
                for group in groups
                if str(group["group_id"]).endswith(f":{family}")
            ]
            if not indices:
                continue
            values = vector[..., indices].float().reshape(-1)
            statistics[family] = {
                "mean": float(values.mean().item()),
                "mean_abs": float(values.abs().mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "max_abs": float(values.abs().max().item()),
            }
        return statistics

    @staticmethod
    def _svd_summary(matrix, torch):
        values = torch.linalg.svdvals(matrix.float())
        energy = values.square()
        total = energy.sum()
        total_value = float(total.item())
        if total_value == 0.0:
            return {
                "status": "zero_field",
                "numerical_rank": 0,
                "top_singular_energy_fraction": 0.0,
                "effective_rank": 0.0,
                "total_energy": float(total.item()),
            }
        probability = energy / total
        entropy = -(probability * probability.clamp_min(1.0e-20).log()).sum()
        return {
            "status": "valid",
            "numerical_rank": int(torch.linalg.matrix_rank(matrix.float()).item()),
            "top_singular_energy_fraction": float(probability[0].item()),
            "effective_rank": float(torch.exp(entropy).item()),
            "total_energy": float(total.item()),
        }

    @staticmethod
    def _safe_cosine(left, right, torch):
        left_norm = float(left.float().norm().item())
        right_norm = float(right.float().norm().item())
        if min(left_norm, right_norm) <= 1.0e-12:
            return None
        return float(torch.nn.functional.cosine_similarity(left.float(), right.float(), dim=0).item())

    def _validate(self, router, items, step, registry_fingerprint, paths, torch) -> None:
        router.eval()
        item_ids = [item["dataset_relative_item_id"] for item in items]
        modes = ["normal_v3", "universal_only", "full_router"]
        if self.phase_c.validation.evaluate_context_shuffle and len(items) > 1:
            modes.append("context_shuffled")
        with torch.no_grad():
            for spec in validation_grid(
                self.phase_c.validation.canonical_timesteps,
                self.phase_c.validation.dense_timesteps,
                self.phase_c.validation.seeds,
            ):
                for item_index, item in enumerate(items):
                    item_id = item_ids[item_index]
                    shuffled_id = item_ids[(item_index + 1) % len(item_ids)]
                    full_q, universal_raw, contextual_raw = self._validation_q(
                        router, item_id, spec["timestep"], "full_router"
                    )
                    universal_applied = router.bound(universal_raw)
                    contextual_applied = full_q - universal_applied
                    if spec["grid"] == "dense" or spec["seed"] == self.phase_c.validation.seeds[0]:
                        append_jsonl(paths["contextual_profiles"], {
                            "schema": "ai-toolkit.ideogram4-v3-phase-c-v2-contextual-profile",
                            "schema_version": 2,
                            "step": step,
                            "grid": spec["grid"],
                            "item_id": item_id,
                            "timestep": spec["timestep"],
                            "raw_universal_norm": float(universal_raw.norm().item()),
                            "raw_contextual_norm": float(contextual_raw.norm().item()),
                            "applied_universal_norm": float(universal_applied.norm().item()),
                            "applied_contextual_increment_norm": float(contextual_applied.norm().item()),
                            "applied_total_norm": float(full_q.norm().item()),
                            "family_statistics": {
                                "applied_universal": self._family_statistics(universal_applied, torch),
                                "applied_contextual_increment": self._family_statistics(contextual_applied, torch),
                                "applied_total": self._family_statistics(full_q, torch),
                            },
                            "q_total": full_q.cpu()[0].tolist(),
                        })
                    if spec["grid"] != "canonical":
                        continue
                    prepared = self._prepare_training_case(item, spec["timestep"], spec["seed"], torch)
                    losses = {}
                    for mode in modes:
                        q_values, _, _ = self._validation_q(
                            router, item_id, spec["timestep"], mode, shuffled_item_id=shuffled_id
                        )
                        runtime = ResidualGateRuntime(1.0 + q_values, registry_fingerprint=registry_fingerprint)
                        with residual_gate_runtime_context(self.network, runtime):
                            prediction = self._predict(prepared, prepared["prompt_template"], "full")
                        losses[mode] = torch.mean((prediction.float() - prepared["target"].float()).square())
                    baseline = losses["normal_v3"].clamp_min(1.0e-12)
                    full_loss = losses["full_router"]
                    shuffled_loss = losses.get("context_shuffled")
                    append_jsonl(paths["validation_metrics"], {
                        "schema": "ai-toolkit.ideogram4-v3-phase-c-v2-validation-metric",
                        "schema_version": 2,
                        "step": step,
                        "split": "validation",
                        "grid": spec["grid"],
                        "item_id": item_id,
                        "timestep": spec["timestep"],
                        "noise_seed": spec["seed"],
                        "normal_v3_loss": float(losses["normal_v3"].item()),
                        "universal_only_loss": float(losses["universal_only"].item()),
                        "full_router_loss": float(full_loss.item()),
                        "context_shuffled_loss": float(shuffled_loss.item()) if shuffled_loss is not None else None,
                        "loss": float(full_loss.item()),
                        "normalized_improvement_vs_v3": float((1.0 - full_loss / baseline).item()),
                        "contextual_incremental_gain": float((losses["universal_only"] - full_loss).item()),
                        "contextual_specificity_gain": float((shuffled_loss - full_loss).item()) if shuffled_loss is not None else None,
                    })

            dense = torch.tensor(self.phase_c.validation.dense_timesteps, device=self.sd.device_torch, dtype=torch.float32) / 1000.0
            universal_matrix = torch.stack([router.universal(value.reshape(1))[0] for value in dense])
            temporal_record = {
                "schema": "ai-toolkit.ideogram4-v3-phase-c-v2-temporal-profile",
                "schema_version": 2,
                "step": step,
                "universal_anchor_raw": self._svd_summary(router.universal_anchors, torch),
                "universal_applied_dense": self._svd_summary(universal_matrix, torch),
                "prompts": [],
                "cross_content": {},
            }
            fields_by_item = {}
            for item_id in item_ids:
                projected = self._projected_activator_cache[item_id].to(self.sd.device_torch, dtype=torch.float32)
                code = router.encode_activator(projected)
                field = torch.cat([router(value.reshape(1), code) for value in dense], dim=0)
                centered_field = field - field.mean(dim=0, keepdim=True)
                fields_by_item[item_id] = field
                canonical = [router(torch.tensor([value / 1000.0], device=self.sd.device_torch), code)[0] for value in self.phase_c.validation.canonical_timesteps]
                cosines = {}
                for left in range(len(canonical)):
                    for right in range(left + 1, len(canonical)):
                        key = f"{self.phase_c.validation.canonical_timesteps[left]}_{self.phase_c.validation.canonical_timesteps[right]}"
                        cosines[key] = self._safe_cosine(canonical[left], canonical[right], torch)
                temporal_record["prompts"].append({
                    "item_id": item_id,
                    "uncentered": self._svd_summary(field, torch),
                    "centered": self._svd_summary(centered_field, torch),
                    "canonical_cosines": cosines,
                })
            if len(item_ids) > 1:
                for timestep in self.phase_c.validation.canonical_timesteps:
                    normalized_timestep = torch.tensor(
                        [timestep / 1000.0], device=self.sd.device_torch, dtype=torch.float32
                    )
                    canonical_fields = {}
                    for item_id in item_ids:
                        projected = self._projected_activator_cache[item_id].to(
                            self.sd.device_torch, dtype=torch.float32
                        )
                        code = router.encode_activator(projected)
                        canonical_fields[item_id] = router(normalized_timestep, code)[0]
                    pairwise = []
                    for left in range(len(item_ids)):
                        for right in range(left + 1, len(item_ids)):
                            cosine = self._safe_cosine(
                                canonical_fields[item_ids[left]],
                                canonical_fields[item_ids[right]],
                                torch,
                            )
                            if cosine is not None:
                                pairwise.append(cosine)
                    temporal_record["cross_content"][str(timestep)] = {
                        "valid_pair_count": len(pairwise),
                        "degenerate_pair_count": len(item_ids) * (len(item_ids) - 1) // 2 - len(pairwise),
                        "mean_cosine": float(sum(pairwise) / len(pairwise)) if pairwise else None,
                        "std_cosine": float(torch.tensor(pairwise).std(unbiased=False).item()) if pairwise else None,
                    }
            append_jsonl(paths["temporal_profiles"], temporal_record)
        router.train()

    def _artifact_paths(self) -> Dict[str, Path]:
        artifacts = self.phase_c.artifacts
        return {
            "training_metrics": self.output_root / artifacts.training_metrics,
            "validation_metrics": self.output_root / artifacts.validation_metrics,
            "temporal_profiles": self.output_root / artifacts.temporal_profiles,
            "contextual_profiles": self.output_root / artifacts.contextual_profiles,
            "group_registry": self.output_root / artifacts.group_registry,
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
            router_config = dict(router_config)
            router_config["run_fingerprint"] = run_fingerprint
            router_path = temporary_dir / self.phase_c.artifacts.router_filename
            save_file({key: value.detach().cpu() for key, value in router.state_dict().items()}, str(router_path))
            torch.save({
                "schema": "ai-toolkit.ideogram4-v3-phase-c-v2-checkpoint",
                "schema_version": 2,
                "step": int(step),
                "registry_fingerprint": fingerprint,
                "run_fingerprint": run_fingerprint,
                "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
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
        partial = []
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
            present = [candidate.is_file() for candidate in required]
            if all(present):
                candidates.append((step, path))
            elif any(present):
                partial.append((step, path, [candidate.name for candidate, exists in zip(required, present) if not exists]))
        if partial:
            highest_partial = max(partial, key=lambda item: item[0])
            highest_complete = max((step for step, _ in candidates), default=-1)
            if highest_partial[0] >= highest_complete:
                raise RuntimeError(
                    f"highest Phase C checkpoint is partial at step {highest_partial[0]}; "
                    f"missing={highest_partial[2]}"
                )
        if not candidates:
            return 0
        checkpoint_root = root.resolve()
        directory_step, latest_path = max(candidates, key=lambda item: item[0])
        latest = latest_path.resolve()
        if checkpoint_root not in latest.parents or latest.parent != checkpoint_root:
            raise RuntimeError("unsafe Phase C resume checkpoint path")
        state = torch.load(latest / "training_state.pt", map_location="cpu", weights_only=False)
        if state.get("schema") != "ai-toolkit.ideogram4-v3-phase-c-v2-checkpoint" or int(state.get("schema_version", 0)) != 2:
            raise RuntimeError("legacy Phase C V1 checkpoint cannot be resumed by Phase C V2")
        if int(state.get("step", -1)) != directory_step:
            raise RuntimeError("checkpoint state step does not match its directory name")
        checkpoint_config = load_json(latest / self.phase_c.artifacts.router_config_filename)
        if checkpoint_config.get("schema") != "ai-toolkit.ideogram4-v3-phase-c-v2-router-config" or int(checkpoint_config.get("schema_version", 0)) != 2:
            raise RuntimeError("checkpoint router_config is not a Phase C V2 artifact")
        if int(checkpoint_config.get("contract_revision", 0)) != 4:
            raise RuntimeError("checkpoint predates the audited Phase C V2 contract revision; start from a fresh output_root")
        if checkpoint_config.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError("checkpoint router_config fingerprint mismatch")
        if state.get("registry_fingerprint") != fingerprint:
            raise RuntimeError("checkpoint active registry fingerprint mismatch")
        if state.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError("checkpoint Phase C source/config fingerprint mismatch")
        from safetensors.torch import load_file

        if "optimizer" not in state or "torch_rng_state" not in state or "python_rng_state" not in state:
            raise RuntimeError("Phase C V2 checkpoint is missing optimizer or RNG state")
        router.load_state_dict(load_file(str(latest / self.phase_c.artifacts.router_filename), device="cpu"), strict=True)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        random.setstate(state["python_rng_state"])
        return int(state["step"])

    def _source_manifest(self, active_registry, fingerprint):
        return {
            "schema": "ai-toolkit.ideogram4-v3-phase-c-v2-source",
            "schema_version": 2,
            "contract_revision": 4,
            "status": "resolved",
            "inputs": {
                **self.input_refs,
                "residual_manifest": {"path": str(self.residual_manifest_path), "sha256": sha256_file(self.residual_manifest_path)},
                "registry": {"path": str(self.registry_path), "sha256": sha256_file(self.registry_path)},
                "split_manifest": {"path": str(self.split_manifest_path), "sha256": sha256_file(self.split_manifest_path)},
            },
            "active_registry_fingerprint": fingerprint,
            "active_registry": active_registry,
            "frozen": ["Ideogram4", "Qwen", "trigger embeddings", "A1/A2 TE adapter", "V3 diffusion LoRA"],
            "trainable": ["Phase C V2 activator projector and local temporal router"],
            "trigger_token_count_per_occurrence": self.phase_c.activator_token_count,
            "activator_occurrence_count": self.phase_c.activator_occurrence_count,
            "activator_occurrence_mode": self.phase_c.activator_occurrence_mode,
            "activator_mask_schema": "a1-a2-trigger-mask-v1",
        }

    def _read_validation_records(self, path: Path):
        return load_jsonl(path)

    def _validate_training_records(self, training_records) -> None:
        steps = [int(record.get("step", -1)) for record in training_records]
        expected = list(range(1, self.phase_c.steps + 1))
        if steps != expected:
            raise RuntimeError("Phase C training metrics are not a complete ordered 1..steps sequence")
        if any(record.get("schema") != "ai-toolkit.ideogram4-v3-phase-c-v2-training-metric" for record in training_records):
            raise RuntimeError("Phase C training metrics contain an incompatible schema")

    def _validate_profile_records(self, temporal_records, contextual_records) -> None:
        validation_steps = {
            step for step in range(1, self.phase_c.steps + 1)
            if step % self.phase_c.validation.every == 0 or step == self.phase_c.steps
        }
        temporal_steps = [int(record.get("step", -1)) for record in temporal_records]
        if len(temporal_steps) != len(set(temporal_steps)) or set(temporal_steps) != validation_steps:
            raise RuntimeError("Phase C temporal profiles are incomplete or duplicated")
        expected = {
            (step, item_id, grid, int(timestep))
            for step in validation_steps
            for item_id in self._validation_item_ids
            for grid, timesteps in (
                ("canonical", self.phase_c.validation.canonical_timesteps),
                ("dense", self.phase_c.validation.dense_timesteps),
            )
            for timestep in timesteps
        }
        actual = {
            (int(record.get("step", -1)), str(record.get("item_id")), str(record.get("grid")), int(record.get("timestep", -1)))
            for record in contextual_records
        }
        if len(actual) != len(contextual_records) or actual != expected:
            raise RuntimeError("Phase C contextual profiles are incomplete or duplicated")

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
        expected_keys = {
            (step, str(item_id), int(timestep), int(seed))
            for step in validation_steps
            for item_id in self._validation_item_ids
            for timestep in self.phase_c.validation.canonical_timesteps
            for seed in self.phase_c.validation.seeds
        }
        actual_keys = set(keys)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)[:5]
            extra = sorted(actual_keys - expected_keys)[:5]
            raise RuntimeError(
                f"Phase C canonical validation probe identities are incomplete; missing={missing}, extra={extra}"
            )
        invalid = {step: count for step, count in counts.items() if count != expected_per_step}
        if invalid:
            raise RuntimeError(
                f"Phase C canonical validation probe counts are incomplete; expected_per_step={expected_per_step}, actual={invalid}"
            )

    def _copy_router_artifacts(self, source: Path, destination: Path) -> None:
        temporary_dir = destination.parent / f".{destination.name}.tmp"
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        temporary_dir.mkdir(parents=True)
        try:
            for filename in (self.phase_c.artifacts.router_filename, self.phase_c.artifacts.router_config_filename):
                shutil.copyfile(source / filename, temporary_dir / filename)
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(temporary_dir, destination)
        except BaseException:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
            raise

    def _finalize(self, router, fingerprint, router_config, paths, torch) -> None:
        final_checkpoint = self._safe_checkpoint_dir(self.phase_c.steps)
        final_dir = self.output_root / self.phase_c.artifacts.final_dir
        self._copy_router_artifacts(final_checkpoint, final_dir)
        training_records = load_jsonl(paths["training_metrics"])
        validation_records = self._read_validation_records(paths["validation_metrics"])
        temporal_records = load_jsonl(paths["temporal_profiles"])
        contextual_records = load_jsonl(paths["contextual_profiles"])
        self._validate_training_records(training_records)
        self._validate_completion_records(validation_records, self._validation_item_count)
        self._validate_profile_records(temporal_records, contextual_records)
        best_step, best_loss = select_best_validation(validation_records)
        best_checkpoint = self._safe_checkpoint_dir(best_step)
        best_dir = self.output_root / self.phase_c.artifacts.best_dir
        self._copy_router_artifacts(best_checkpoint, best_dir)
        handoff = {
            "schema": "ai-toolkit.ideogram4-v3-phase-c-v2-handoff",
            "schema_version": 2,
            "status": "completed",
            "active_registry_fingerprint": fingerprint,
            "run_fingerprint": router_config.get("run_fingerprint"),
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
            "schema": "ai-toolkit.ideogram4-v3-phase-c-v2-completion",
            "schema_version": 2,
            "status": "completed",
            "steps": self.phase_c.steps,
            "optimizer": self.phase_c.optimizer,
            "optimizer_params": dict(self.phase_c.optimizer_params),
            "fresh_optimizer": True,
            "router_only": True,
            "run_fingerprint": router_config.get("run_fingerprint"),
            "active_registry_fingerprint": fingerprint,
            "validation_term": "full_router canonical validation loss",
            "artifacts": {
                "source": artifact_ref(paths["source_manifest"], self.output_root),
                "handoff": artifact_ref(paths["handoff_manifest"], self.output_root),
                "training_metrics": artifact_ref(paths["training_metrics"], self.output_root),
                "validation_metrics": artifact_ref(paths["validation_metrics"], self.output_root),
                "temporal_profiles": artifact_ref(paths["temporal_profiles"], self.output_root),
                "contextual_profiles": artifact_ref(paths["contextual_profiles"], self.output_root),
                "group_registry": artifact_ref(paths["group_registry"], self.output_root),
            },
        })
