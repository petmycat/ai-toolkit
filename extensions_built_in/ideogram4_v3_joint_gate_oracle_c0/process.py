from __future__ import annotations

import os
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from jobs.process import BaseExtensionProcess
from extensions_built_in.ideogram4_v3_residual_ablation.process import Ideogram4V3ResidualAblationProcess

from .config import load_config
from .helpers import (
    FAMILY_ORDER,
    atomic_write_json,
    atomic_write_jsonl,
    build_prior_vectors,
    cosine,
    load_json,
    load_jsonl,
    metric_payload,
    q_statistics,
    select_balanced_batch,
    safe_prompt_slug,
    sha256_file,
    sign_agreement,
    summarize_validation,
    validate_visual_prompt_placeholders,
)


class Ideogram4V3JointGateOracleC0Process(Ideogram4V3ResidualAblationProcess):
    """Independent, discardable C0 oracle reusing the audited A2/V3 loaders."""

    def __init__(self, process_id: int, job, config: OrderedDict):
        BaseExtensionProcess.__init__(self, process_id, job, config)
        self.c0 = load_config(dict(self.raw_process_config.get("c0_joint_gate_oracle", {})))
        self.device = self.get_conf("device", getattr(job, "device", "cuda"))
        self.output_root = Path(self.c0.output_root)
        self.run_label = str(self.get_conf("run_id", self.name))
        self.ablation = {"a2_contract": self.c0.a2_contract, "helpers": ["unused-c0-loader-placeholder"]}
        self.model_definition = None
        self.network_definition = None
        self.sd = None
        self.network = None
        self.text_activator = None
        self._calibrated_embed_cache = {}
        self._stock_embed_cache = {}
        self._latent_cache = {}
        self._run_started_at = time.monotonic()

    def _progress(self, message: str) -> None:
        from toolkit.print import print_acc

        elapsed = max(0, int(time.monotonic() - self._run_started_at))
        print_acc(
            f"[C0 joint gate oracle +{elapsed // 3600}h "
            f"{(elapsed % 3600) // 60:02d}m {elapsed % 60:02d}s] {message}"
        )

    def _enable_checkpointing(self):
        enable = getattr(self.sd.model, "enable_gradient_checkpointing", None)
        if not callable(enable):
            raise RuntimeError("C0 gate gradients require transformer checkpointing support")
        enable(every_n_blocks=self.c0.gradient_checkpointing_every_n_blocks)
        if not getattr(self.sd.model, "gradient_checkpointing", False):
            raise RuntimeError("C0 gate gradients require transformer checkpointing")

    def _resolve_c0_inputs(self):
        self._resolve_inputs()
        residual_manifest = load_json(self.c0.residual_manifest)
        if residual_manifest.get("schema") != "ai-toolkit.ideogram4-v3-residual-ablation":
            raise RuntimeError("invalid C0 residual-ablation manifest schema")
        if int(residual_manifest.get("schema_version", 0)) < 5 or residual_manifest.get("status") != "completed":
            raise RuntimeError("C0 requires a completed schema-v5 residual ablation")
        for key in ("a2_contract", "a2_snapshot", "v3_weights", "best_embedding", "best_te_adapter", "best_manifest"):
            if residual_manifest.get("inputs", {}).get(key, {}).get("sha256") != self.input_refs.get(key, {}).get("sha256"):
                raise RuntimeError(f"C0 residual-ablation source disagrees with A2 source: {key}")
        registry_path = Path(self.c0.registry)
        records_path = Path(self.c0.residual_records)
        for path in (registry_path, records_path, Path(self.c0.split_manifest), Path(self.c0.dataset_root)):
            if not path.exists():
                raise FileNotFoundError(path)
        recorded_ref = residual_manifest.get("artifacts", {}).get("registry", {})
        manifest_registry_path = (Path(self.c0.residual_manifest).parent / str(recorded_ref.get("path", ""))).resolve()
        if manifest_registry_path != registry_path.resolve() or recorded_ref.get("sha256") != sha256_file(registry_path):
            raise RuntimeError("configured C0 registry does not match residual-ablation manifest")
        manifest_records_path = (Path(self.c0.residual_manifest).parent / str(residual_manifest.get("artifacts", {}).get("records", {}).get("path", ""))).resolve()
        recorded_records_ref = residual_manifest.get("artifacts", {}).get("records", {})
        if manifest_records_path != records_path.resolve() or recorded_records_ref.get("sha256") != sha256_file(records_path):
            raise RuntimeError("configured C0 residual records do not match residual-ablation manifest")
        records = load_jsonl(records_path)
        run_id = str(residual_manifest.get("run_id", ""))
        if not run_id or any(str(row.get("run_id", "")) != run_id for row in records):
            raise RuntimeError("C0 residual records do not all belong to the manifest run_id")
        record_timesteps = {int(row.get("timestep", -1)) for row in records}
        if not set(self.c0.timesteps).issubset(record_timesteps):
            raise RuntimeError(f"C0 residual records lack required timesteps: {sorted(set(self.c0.timesteps) - record_timesteps)}")
        return load_json(registry_path), records, residual_manifest

    def _preflight_visual_prompts(self):
        visual = self.c0.novel_visuals
        if not visual.enabled or not visual.prompts:
            return []
        expected_occurrences = None
        if visual.include_phase_c_v2:
            handoff_path = Path(visual.phase_c_v2_handoff)
            handoff = load_json(handoff_path)
            if handoff.get("schema") != "ai-toolkit.ideogram4-v3-phase-c-v2-handoff" or int(handoff.get("schema_version", 0)) != 2 or handoff.get("status") != "completed":
                raise RuntimeError("invalid or incomplete Phase C V2 handoff for C0 visual prompt preflight")
            refs = handoff.get(visual.phase_c_v2_checkpoint, {})
            config_ref = refs.get("config", {}) if isinstance(refs, Mapping) else {}
            config_path = (handoff_path.parent / str(config_ref.get("path", ""))).resolve()
            if not config_path.is_file() or sha256_file(config_path) != config_ref.get("sha256"):
                raise RuntimeError("Phase C V2 visual router config file/hash mismatch during prompt preflight")
            router_config = load_json(config_path)
            if router_config.get("schema") != "ai-toolkit.ideogram4-v3-phase-c-v2-router-config" or int(router_config.get("schema_version", 0)) != 2:
                raise RuntimeError("invalid Phase C V2 router config during visual prompt preflight")
            expected_occurrences = int(router_config.get("activator_occurrence_count", 0))
        return validate_visual_prompt_placeholders(
            visual.prompts, self.placeholder, expected_occurrences=expected_occurrences
        )

    def _build_active_registry(self, recorded_registry):
        from toolkit.residual_gating import (
            active_registry_fingerprint,
            bind_active_registry,
            build_module_registry,
            filter_active_registry,
            serialize_active_registry,
        )
        from extensions_built_in.ideogram4_v3_router_phase_c.helpers import validate_registry_contract

        actual = build_module_registry(self.network.get_all_modules())
        validate_registry_contract(actual, recorded_registry)
        active = filter_active_registry(actual, norm_threshold=0.0)
        bind_active_registry(self.network.get_all_modules(), active)
        payload = serialize_active_registry(active)
        if int(payload["group_count"]) != 102:
            raise RuntimeError(f"C0 requires exactly 102 active groups, found {payload['group_count']}")
        rows = []
        for group in payload["groups"]:
            group_id = str(group["group_id"])
            block_text, block_index, family = group_id.split(":")
            if block_text != "block" or family not in FAMILY_ORDER:
                raise RuntimeError(f"C0 found unexpected active group: {group_id}")
            rows.append({
                "group_id": group_id,
                "index": int(group["group_index"]),
                "block_index": int(block_index),
                "family": family,
                "modules": list(group["module_names"]),
            })
        rows.sort(key=lambda row: row["index"])
        if [row["index"] for row in rows] != list(range(102)):
            raise RuntimeError("active registry group_index must be contiguous in [0,102)")
        expected = {(block, family) for block in range(34) for family in FAMILY_ORDER}
        if {(row["block_index"], row["family"]) for row in rows} != expected:
            raise RuntimeError("active registry is not the exact 34 x 3 C0 partition")
        return rows, payload, active_registry_fingerprint(active)

    def _load_items(self):
        from extensions_built_in.ideogram4_tap_ablation.helpers import select_probe_items
        from toolkit.trigger_data_split import load_data_split_manifest

        split = load_data_split_manifest(self.c0.split_manifest, expected_seed=self.c0.seed)
        train_limit = min(len(split.train_item_ids), self.c0.train_item_limit or len(split.train_item_ids))
        validation_limit = min(len(split.heldout_item_ids), self.c0.validation_item_limit or len(split.heldout_item_ids))
        expected_train = set(sorted(split.train_item_ids)[:train_limit])
        items = select_probe_items(
            self.c0.dataset_root, split.as_dict(), train_limit=train_limit,
            expected_heldout=len(split.heldout_item_ids), placeholder=self.placeholder,
        )
        selected_heldout = set(sorted(split.heldout_item_ids)[:validation_limit])
        train = [item for item in items if item["split"] == "train" and item["dataset_relative_item_id"] in expected_train]
        heldout = [item for item in items if item["split"] == "heldout" and item["dataset_relative_item_id"] in selected_heldout]
        if {item["dataset_relative_item_id"] for item in train} != expected_train:
            raise RuntimeError("C0 train item selection does not exactly match the split-manifest contract")
        if {item["dataset_relative_item_id"] for item in heldout} != selected_heldout:
            raise RuntimeError("C0 heldout item selection does not exactly match the split-manifest contract")
        if len(train) < self.c0.same_timestep_content_batch or not heldout:
            raise RuntimeError("C0 requires enough distinct train contents and a non-empty heldout split")
        return train, heldout

    def _precompute_items(self, items, torch):
        from tqdm.auto import tqdm
        from toolkit.basic import flush

        for item in tqdm(items, desc="C0 precompute", unit="item", dynamic_ncols=True, leave=True):
            item_id = item["dataset_relative_item_id"]
            case = SimpleNamespace(item_id=item_id, timestep=0, noise_seed=self.c0.seed)
            prepared = self._prepare_case(case, item, torch)
            with torch.no_grad():
                self._calibrated_embed_cache[item_id] = self._calibrated_prompt_embeds(prepared["prompt_template"]).to("cpu")
            self._latent_cache[item_id] = self._latent_cache[item_id].to("cpu")
        self.sd.text_encoder.to("cpu")
        self.text_activator.to("cpu")
        self.sd.vae.to("cpu")
        flush()

    def _prepare_item_case(self, item, timestep, noise_seed, torch):
        case = SimpleNamespace(item_id=item["dataset_relative_item_id"], timestep=int(timestep), noise_seed=int(noise_seed))
        prepared = self._prepare_case(case, item, torch)
        prepared["calibrated_embeds"] = self._calibrated_embed_cache[case.item_id]
        return prepared

    @staticmethod
    def _select_train_items(items, step, count, seed):
        return select_balanced_batch(items, step, count, seed)

    def _prediction_loss(self, prepared, q, registry_fingerprint, torch):
        from toolkit.residual_gating import ResidualGateRuntime, residual_gate_runtime_context

        if q.ndim != 1 or q.shape[0] != 102:
            raise ValueError("C0 q must have shape [102]")
        if not bool(torch.isfinite(q).all()):
            raise RuntimeError("C0 q contains non-finite values")
        if bool((q.abs() > self.c0.q_bound + 1.0e-6).any()):
            raise RuntimeError("C0 q exceeds the configured bounded intervention domain")
        gates = 1.0 + q.unsqueeze(0)
        runtime = ResidualGateRuntime(gates, registry_fingerprint=registry_fingerprint)
        with residual_gate_runtime_context(self.network, runtime):
            prediction = self._predict(prepared, prepared["prompt_template"], "full")
        loss = torch.mean((prediction.float() - prepared["target"].float()).square())
        return loss

    def _mean_loss(self, items, timestep, seeds, q, registry_fingerprint, torch):
        values = []
        with torch.no_grad():
            for seed in seeds:
                for item in items:
                    prepared = self._prepare_item_case(item, timestep, seed, torch)
                    values.append(self._prediction_loss(prepared, q, registry_fingerprint, torch))
        if not values:
            raise RuntimeError("C0 mean loss requires at least one item/seed case")
        return float(torch.stack(values).mean().item())

    def _gradient_at_zero(self, train_items, timestep, registry_fingerprint, torch):
        gradients = []
        cases = []
        for seed in self.c0.optimizer.noise_seeds:
            parameter = torch.zeros(102, device=self.sd.device_torch, dtype=torch.float32, requires_grad=True)
            for item in train_items:
                prepared = self._prepare_item_case(item, timestep, seed, torch)
                q_micro = self.c0.q_bound * torch.tanh(parameter)
                loss = self._prediction_loss(prepared, q_micro, registry_fingerprint, torch)
                (loss / len(train_items)).backward()
                cases.append({"item_id": item["dataset_relative_item_id"], "noise_seed": seed})
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"C0 q={timestep} zero-point gradient is missing or non-finite")
            gradients.append(parameter.grad.detach().clone())
        gradient = torch.stack(gradients).mean(dim=0)
        if float(gradient.norm().item()) == 0.0:
            raise RuntimeError(f"C0 q={timestep} mean zero-point gradient is exactly zero")
        return gradient, cases

    def _validation_records(self, q, fd_q, gradient_at_zero, train_items, heldout_items, timestep, registry_fingerprint, torch):
        candidates = {"v3": torch.zeros_like(q), "fd_composed": fd_q, "oracle": q}
        for scale in self.c0.validation.global_scales:
            candidates[f"global_{scale:g}"] = torch.full_like(q, scale - 1.0)
        for magnitude in self.c0.validation.gradient_sign_scales:
            candidates[f"gradient_sign_{magnitude:g}"] = -float(magnitude) * torch.sign(gradient_at_zero)
        rows = []
        for split_name, items in (("train", train_items), ("heldout", heldout_items)):
            for seed in self.c0.validation.noise_seeds:
                for item in items:
                    prepared = self._prepare_item_case(item, timestep, seed, torch)
                    losses = {}
                    with torch.no_grad():
                        for name, candidate in candidates.items():
                            losses[name] = float(
                                self._prediction_loss(prepared, candidate, registry_fingerprint, torch).item()
                            )
                    v3_loss = losses["v3"]
                    for name, loss in losses.items():
                        rows.append({
                            "schema": "ai-toolkit.ideogram4-v3-c0-validation", "schema_version": 1,
                            "timestep": timestep, "split": split_name,
                            "item_id": item["dataset_relative_item_id"], "noise_seed": seed,
                            "candidate": name, **metric_payload(v3_loss, loss),
                        })
        return rows

    def _diagnostics(self, q, fd_q, prior_gradient, heldout_items, timestep, registry_fingerprint, torch):
        diagnostics = {"topk": [], "magnitude_path": [], "family_combinations": []}
        ranking = torch.argsort(q.abs(), descending=True)
        for count in self.c0.validation.topk:
            candidate = torch.zeros_like(q)
            candidate[ranking[:count]] = q[ranking[:count]]
            diagnostics["topk"].append({
                "active_groups": count,
                "mean_heldout_loss": self._mean_loss(
                    heldout_items, timestep, self.c0.validation.noise_seeds, candidate, registry_fingerprint, torch
                ),
            })
        for alpha in self.c0.validation.magnitude_alphas:
            diagnostics["magnitude_path"].append({
                "alpha": alpha,
                "mean_heldout_loss": self._mean_loss(
                    heldout_items, timestep, self.c0.validation.noise_seeds, q * alpha, registry_fingerprint, torch
                ),
            })
        masks = {}
        for family in FAMILY_ORDER:
            mask = torch.tensor([row["family"] == family for row in self._group_rows], device=q.device)
            masks[family] = mask
        combinations = [
            ("adaln",), ("attention",), ("mlp",),
            ("mlp", "attention"), ("mlp", "adaln"), ("attention", "adaln"),
        ]
        for combination in combinations:
            mask = torch.zeros_like(q, dtype=torch.bool)
            for family in combination:
                mask |= masks[family]
            candidate = torch.where(mask, q, torch.zeros_like(q))
            diagnostics["family_combinations"].append({
                "families": list(combination),
                "mean_heldout_loss": self._mean_loss(
                    heldout_items, timestep, self.c0.validation.noise_seeds, candidate, registry_fingerprint, torch
                ),
            })
        diagnostics["prior_comparison"] = {
            "oracle_vs_prior_gradient_cosine": cosine(q, -prior_gradient),
            "oracle_vs_fd_cosine": cosine(q, fd_q),
            "fd_sign_agreement": sign_agreement(q, fd_q),
        }
        return diagnostics

    def _assert_invariants(self, train_items, registry_fingerprint, torch):
        item = train_items[0]
        prepared = self._prepare_item_case(item, 100, self.c0.optimizer.noise_seeds[0], torch)
        zero = torch.zeros(102, device=self.sd.device_torch, dtype=torch.float32)
        with torch.no_grad():
            raw = self._predict(prepared, prepared["prompt_template"], "full")
            zero_loss = self._prediction_loss(prepared, zero, registry_fingerprint, torch)
            raw_loss = torch.mean((raw.float() - prepared["target"].float()).square())
        if not torch.equal(raw_loss, zero_loss):
            raise RuntimeError("C0 q=0 does not exactly reproduce ordinary V3")
        parameter = torch.zeros(102, device=self.sd.device_torch, dtype=torch.float32, requires_grad=True)
        q = self.c0.q_bound * torch.tanh(parameter)
        loss = self._prediction_loss(prepared, q, registry_fingerprint, torch)
        loss.backward()
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) or float(parameter.grad.norm().item()) == 0.0:
            raise RuntimeError("C0 oracle scalar vector did not receive a finite non-zero gradient")
        offenders = []
        for label, module in (("model", self.sd.model), ("text_encoder", self.sd.text_encoder), ("vae", self.sd.vae), ("v3", self.network), ("activator", self.text_activator)):
            offenders.extend(f"{label}.{name}" for name, value in module.named_parameters() if value.grad is not None)
        if offenders:
            raise RuntimeError(f"C0 frozen parameters received gradients: {offenders[:10]}")
        for index, expected_scale in ((0, 0.75), (1, 1.25)):
            probe = torch.zeros_like(zero)
            probe[index] = expected_scale - 1.0
            if float((1.0 + probe[index]).item()) != expected_scale:
                raise RuntimeError("C0 gate parameterization failed exact FD endpoint identity")
        self.sd.model.zero_grad(set_to_none=True)
        self.network.zero_grad(set_to_none=True)

    def run(self):
        BaseExtensionProcess.run(self)
        import torch
        from tqdm.auto import tqdm

        self._run_started_at = time.monotonic()
        self._visual_summary = {"status": "not_started", "image_count": 0, "prompt_count": 0}
        self._progress("resolving V9 A2, V3, residual-ablation, registry, and split contracts")
        recorded_registry, fd_records, residual_manifest = self._resolve_c0_inputs()
        visual_prompt_audit = self._preflight_visual_prompts()
        if visual_prompt_audit:
            self._progress(f"validated {len(visual_prompt_audit)} novel visual prompt placeholder contracts")
        if self.output_root.exists() and any(self.output_root.iterdir()):
            raise RuntimeError(
                f"C0 output_root is not empty and resume is disabled: {self.output_root}; "
                "use a fresh output_root or explicitly remove the prior run"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._progress("loading frozen Ideogram4, V3 LoRA, and calibrated A2 activator")
        self.sd = self._load_model()
        self.network = self._load_lora(torch)
        self.text_activator = self._build_text_activator(torch)
        self._load_and_install_activator()
        self._freeze()
        self._enable_checkpointing()
        self.network.is_active = True
        self._group_rows, active_registry, registry_fingerprint = self._build_active_registry(recorded_registry)
        self._registry_fingerprint = registry_fingerprint
        train_items, heldout_items = self._load_items()
        self._precompute_items(train_items + heldout_items, torch)
        self._progress("running exact endpoint, freeze, registry, and gradient preflight invariants")
        self._assert_invariants(train_items, registry_fingerprint, torch)
        all_validation = []
        oracle_vectors = {}
        prior_vectors = {}
        for timestep in self.c0.timesteps:
            prior = build_prior_vectors(fd_records, self._group_rows, timestep)
            prior_vectors[timestep] = prior
            fd_q = torch.tensor(prior["fd_q"], device=self.sd.device_torch, dtype=torch.float32)
            prior_gradient = torch.tensor(prior["mean_gradient"], device=self.sd.device_torch, dtype=torch.float32)
            parameter = torch.zeros(102, device=self.sd.device_torch, dtype=torch.float32, requires_grad=True)
            optimizer = torch.optim.AdamW([parameter], lr=self.c0.optimizer.learning_rate, weight_decay=0.0)
            metrics = []
            checkpoint_dir = self.output_root / "canonical" / f"t{timestep}"
            self._save_q(checkpoint_dir / "q_step000000.json", timestep, 0, torch.zeros_like(parameter), registry_fingerprint)
            progress = tqdm(range(1, self.c0.optimizer.steps + 1), desc=f"C0 q_{timestep}", unit="step", dynamic_ncols=True, leave=True)
            for step in progress:
                selected = self._select_train_items(train_items, step, self.c0.same_timestep_content_batch, self.c0.seed + timestep)
                noise_seed = self.c0.optimizer.noise_seeds[(step - 1) % len(self.c0.optimizer.noise_seeds)]
                optimizer.zero_grad(set_to_none=True)
                q_before_update = (self.c0.q_bound * torch.tanh(parameter)).detach()
                micro_losses = []
                v3_losses = []
                for item in selected:
                    prepared = self._prepare_item_case(item, timestep, noise_seed, torch)
                    q_micro = self.c0.q_bound * torch.tanh(parameter)
                    loss = self._prediction_loss(prepared, q_micro, registry_fingerprint, torch)
                    (loss / len(selected)).backward()
                    micro_losses.append(loss.detach())
                    with torch.no_grad():
                        v3_losses.append(self._prediction_loss(prepared, torch.zeros_like(q_micro), registry_fingerprint, torch))
                grad_norm = float(parameter.grad.detach().norm().item())
                if not bool(torch.isfinite(parameter.grad).all()) or grad_norm == 0.0:
                    raise RuntimeError(f"C0 q_{timestep} step {step} has invalid oracle gradient")
                optimizer.step()
                if not bool(torch.isfinite(parameter).all()):
                    raise RuntimeError(f"C0 q_{timestep} step {step} produced non-finite parameters")
                q_logged = self.c0.q_bound * torch.tanh(parameter.detach())
                oracle_loss = float(torch.stack(micro_losses).mean().item())
                v3_loss = float(torch.stack(v3_losses).mean().item())
                metrics.append({
                    "schema": "ai-toolkit.ideogram4-v3-c0-training", "schema_version": 1,
                    "timestep": timestep, "optimizer_step": step,
                    "item_ids": [item["dataset_relative_item_id"] for item in selected],
                    "noise_seeds": [noise_seed] * len(selected),
                    "train_loss_v3": v3_loss, "train_loss_oracle": oracle_loss,
                    "train_absolute_gain": v3_loss - oracle_loss,
                    "train_normalized_gain": (v3_loss - oracle_loss) / (v3_loss + 1.0e-12),
                    "gate_grad_norm": grad_norm,
                    "q_before_update": q_statistics(q_before_update, self._group_rows),
                    "q_after_update": q_statistics(q_logged, self._group_rows),
                })
                progress.set_postfix(loss=f"{oracle_loss:.5g}", gain=f"{v3_loss - oracle_loss:.3g}", abs_q=f"{q_logged.abs().mean().item():.3g}", refresh=False)
                if step in self.c0.optimizer.snapshot_steps:
                    self._save_q(checkpoint_dir / f"q_step{step:06d}.json", timestep, step, q_logged, registry_fingerprint)
            progress.close()
            q_final = self.c0.q_bound * torch.tanh(parameter.detach())
            oracle_vectors[timestep] = q_final
            atomic_write_jsonl(checkpoint_dir / "training_metrics.jsonl", metrics)
            self._save_q(checkpoint_dir / "q_final.json", timestep, self.c0.optimizer.steps, q_final, registry_fingerprint)
            current_gradient, gradient_cases = self._gradient_at_zero(
                train_items, timestep, registry_fingerprint, torch
            )
            validation = self._validation_records(q_final, fd_q, current_gradient, train_items, heldout_items, timestep, registry_fingerprint, torch)
            all_validation.extend(validation)
            diagnostics = self._diagnostics(q_final, fd_q, prior_gradient, heldout_items, timestep, registry_fingerprint, torch)
            diagnostics["current_c0_gradient_sign_baseline"] = {
                "cases": gradient_cases,
                "aggregation": "mean gradient across every selected train item, then mean across optimizer noise seeds",
                "gradient": [float(value) for value in current_gradient.cpu().tolist()],
            }
            diagnostics["q_statistics"] = q_statistics(q_final, self._group_rows)
            diagnostics["prior_fd"] = prior
            atomic_write_json(checkpoint_dir / "diagnostics.json", diagnostics)
        self._finalize_artifacts(recorded_registry, active_registry, registry_fingerprint, residual_manifest, all_validation, oracle_vectors, prior_vectors)
        self._generate_novel_visuals(oracle_vectors, registry_fingerprint, torch)
        atomic_write_json(self.output_root / self.c0.artifacts.completion_manifest, {
            "schema": "ai-toolkit.ideogram4-v3-c0-completion", "schema_version": 1,
            "status": "completed", "run_id": self.run_label, "timesteps": list(self.c0.timesteps),
            "optimizer_steps_per_timestep": self.c0.optimizer.steps,
            "registry_fingerprint": registry_fingerprint,
            "visual_status": self._visual_summary["status"],
            "visual_prompt_count": self._visual_summary["prompt_count"],
            "visual_image_count": self._visual_summary["image_count"],
        })
        self._progress(
            f"C0 completed with numerical diagnostics; visuals={self._visual_summary['status']} "
            f"images={self._visual_summary['image_count']}"
        )

    def _save_q(self, path, timestep, step, q, registry_fingerprint):
        atomic_write_json(path, {
            "schema": "ai-toolkit.ideogram4-v3-c0-oracle-gates", "schema_version": 1,
            "timestep": timestep, "optimizer_step": step, "q_bound": self.c0.q_bound,
            "registry_fingerprint": registry_fingerprint,
            "q": [float(value) for value in q.detach().cpu().tolist()],
            "group_values": {row["group_id"]: float(q[int(row["index"])].item()) for row in self._group_rows},
        })

    def _finalize_artifacts(self, recorded_registry, active_registry, registry_fingerprint, residual_manifest, validation, oracle_vectors, prior_vectors):
        from dataclasses import asdict

        artifacts = self.c0.artifacts
        atomic_write_json(self.output_root / artifacts.config, {
            "schema": "ai-toolkit.ideogram4-v3-c0-config", "schema_version": 1,
            "config": asdict(self.c0),
        })
        atomic_write_json(self.output_root / artifacts.source_manifest, {
            "schema": "ai-toolkit.ideogram4-v3-c0-source", "schema_version": 1,
            "inputs": {
                "a2_contract": {"path": self.c0.a2_contract, "sha256": sha256_file(self.c0.a2_contract)},
                "residual_manifest": {"path": self.c0.residual_manifest, "sha256": sha256_file(self.c0.residual_manifest)},
                "registry": {"path": self.c0.registry, "sha256": sha256_file(self.c0.registry)},
                "residual_records": {"path": self.c0.residual_records, "sha256": sha256_file(self.c0.residual_records)},
                "split_manifest": {"path": self.c0.split_manifest, "sha256": sha256_file(self.c0.split_manifest)},
            },
            "residual_run_id": residual_manifest.get("run_id"),
            "registry_fingerprint": registry_fingerprint,
        })
        atomic_write_json(self.output_root / artifacts.group_registry, active_registry)
        atomic_write_jsonl(self.output_root / "validation_metrics.jsonl", validation)
        atomic_write_json(self.output_root / artifacts.canonical_summary, summarize_validation(validation))
        atomic_write_json(self.output_root / artifacts.gate_comparison, {
            str(timestep): {
                "q": [float(value) for value in q.cpu().tolist()],
                "statistics": q_statistics(q, self._group_rows),
                "prior_fd": prior_vectors[timestep],
            } for timestep, q in oracle_vectors.items()
        })
        atomic_write_json(self.output_root / artifacts.family_summary, {
            family: {str(timestep): q_statistics(q, self._group_rows)["families"][family] for timestep, q in oracle_vectors.items()}
            for family in FAMILY_ORDER
        })

    def _generate_novel_visuals(self, oracle_vectors, registry_fingerprint, torch):
        from extensions_built_in.diffusion_models.ideogram4.src.pipeline import (
            get_ideogram4_sigmas, pad_text_features, predict_velocity,
        )
        from toolkit.residual_gating import ResidualGateRuntime, residual_gate_runtime_context

        visual = self.c0.novel_visuals
        manifest = {
            "schema": "ai-toolkit.ideogram4-v3-c0-visuals", "schema_version": 1,
            "prompt_count": len(visual.prompts), "seeds": list(visual.seeds), "images": [],
            "prompt_placeholder_audit": validate_visual_prompt_placeholders(visual.prompts, self.placeholder) if visual.prompts else [],
            "c0_trajectory": "piecewise linear interpolation of q_100/q_500/q_900 with endpoint clamping",
        }
        if not visual.enabled or not visual.prompts:
            manifest["status"] = "skipped_empty_prompt_list" if visual.enabled else "disabled"
            self._visual_summary = {"status": manifest["status"], "image_count": 0, "prompt_count": len(visual.prompts)}
            atomic_write_json(self.output_root / self.c0.artifacts.visual_manifest, manifest)
            return
        self.sd.text_encoder.to(self.sd.device_torch)
        self.text_activator.to(self.sd.device_torch)
        self.sd.vae.to(self.sd.device_torch)
        conditions = ["v3", "global_best", "c0_three_anchor_interpolation"]
        if visual.include_phase_c_v2 and visual.phase_c_v2_handoff:
            conditions.append("phase_c_v2")
        elif visual.include_phase_c_v2:
            manifest["phase_c_v2_status"] = "skipped_no_handoff_configured"
        best_global = self._best_global_by_timestep()
        phase_c = self._load_phase_c_v2(torch) if "phase_c_v2" in conditions else None
        for prompt_index, prompt in enumerate(visual.prompts):
            embeds = self._calibrated_prompt_embeds(prompt).to(self.sd.device_torch, dtype=self.sd.torch_dtype)
            slug = safe_prompt_slug(prompt_index, prompt)
            for seed in visual.seeds:
                for condition in conditions:
                    with torch.inference_mode():
                        image = self._sample_visual(
                            embeds, seed, condition, oracle_vectors, best_global, phase_c,
                            registry_fingerprint, torch, get_ideogram4_sigmas, pad_text_features,
                            predict_velocity, ResidualGateRuntime, residual_gate_runtime_context,
                        )
                    destination = self.output_root / "visuals" / slug / f"seed_{seed}" / f"{condition}.png"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    image.save(destination)
                    manifest["images"].append({
                        "prompt_index": prompt_index, "prompt": prompt, "seed": seed,
                        "condition": condition, "path": os.path.relpath(destination, self.output_root).replace(os.sep, "/"),
                        "sha256": sha256_file(destination),
                    })
        expected_images = len(visual.prompts) * len(visual.seeds) * len(conditions)
        if len(manifest["images"]) != expected_images:
            raise RuntimeError(f"C0 visual completion count mismatch: {len(manifest['images'])} != {expected_images}")
        manifest["status"] = "completed"
        self._visual_summary = {"status": "completed", "image_count": expected_images, "prompt_count": len(visual.prompts)}
        atomic_write_json(self.output_root / self.c0.artifacts.visual_manifest, manifest)

    def _best_global_by_timestep(self):
        records = load_jsonl(self.output_root / "validation_metrics.jsonl")
        candidates = {}
        for record in records:
            if record["split"] == "train" and str(record["candidate"]).startswith("global_"):
                identity = (int(record["timestep"]), str(record["candidate"]))
                candidates.setdefault(identity, []).append(float(record["loss"]))
        output = {}
        for timestep in self.c0.timesteps:
            scoped = {name: values for (value_timestep, name), values in candidates.items() if value_timestep == timestep}
            if not scoped:
                raise RuntimeError(f"C0 validation lacks train global-scalar candidates for timestep {timestep}")
            name = min(scoped, key=lambda key: sum(scoped[key]) / len(scoped[key]))
            output[timestep] = float(name.split("_", 1)[1]) - 1.0
        return output

    def _sample_visual(self, embeds, seed, condition, oracle_vectors, best_global, phase_c, registry_fingerprint, torch, get_sigmas, pad_text_features, predict_velocity, Runtime, runtime_context):
        from diffusers.utils.torch_utils import randn_tensor

        model = self.sd
        device, dtype = model.device_torch, model.torch_dtype
        transformer = model.transformer
        patch = model.patch_size
        visual = self.c0.novel_visuals
        sigmas = get_sigmas(
            visual.steps, visual.width, visual.height,
            mu=float(model.model_config.model_kwargs.get("ideogram_schedule_mu", 0.0)),
            std=float(model.model_config.model_kwargs.get("ideogram_schedule_std", 1.75)), device=device,
        )
        gh = visual.height // (model.vae_scale_factor * patch)
        gw = visual.width // (model.vae_scale_factor * patch)
        generator = torch.Generator(device=device).manual_seed(int(seed))
        latents = randn_tensor((1, transformer.config.in_channels, gh, gw), generator=generator, device=device, dtype=torch.float32) * sigmas[0]
        cond_feats, cond_mask = pad_text_features(embeds.text_embeds, device, dtype)
        uncond_feats = torch.zeros(1, 0, cond_feats.shape[-1], device=device, dtype=dtype)
        uncond_mask = torch.zeros(1, 0, dtype=torch.long, device=device)
        for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
            canonical_timestep = float(sigma.item()) * 1000.0
            q = self._visual_q(condition, canonical_timestep, oracle_vectors, best_global, phase_c, embeds, torch)
            conditional_runtime = Runtime(1.0 + q.unsqueeze(0), registry_fingerprint=registry_fingerprint)
            unconditional_runtime = Runtime(torch.ones_like(q).unsqueeze(0), registry_fingerprint=registry_fingerprint)
            t01 = sigma.expand(1)
            with runtime_context(self.network, conditional_runtime):
                v_cond = predict_velocity(transformer, latents.to(dtype), t01, cond_feats, cond_mask)
            with runtime_context(self.network, unconditional_runtime):
                v_uncond = predict_velocity(transformer, latents.to(dtype), t01, uncond_feats, uncond_mask)
            velocity = v_uncond + visual.guidance_scale * (v_cond - v_uncond)
            latents = latents + velocity.float() * (sigma_next - sigma)
        images = model.decode_latents(latents, device=device, dtype=dtype).float().clamp(-1.0, 1.0)
        images = ((images + 1.0) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        from PIL import Image
        return Image.fromarray(images[0])

    def _visual_q(self, condition, timestep, oracle_vectors, best_global, phase_c, embeds, torch):
        if condition == "v3":
            return torch.zeros(102, device=self.sd.device_torch, dtype=torch.float32)
        if condition == "c0_three_anchor_interpolation":
            anchors = sorted(oracle_vectors)
            if timestep <= anchors[0]:
                return oracle_vectors[anchors[0]]
            if timestep >= anchors[-1]:
                return oracle_vectors[anchors[-1]]
            left = max(value for value in anchors if value <= timestep)
            right = min(value for value in anchors if value >= timestep)
            if left == right:
                return oracle_vectors[left]
            alpha = (timestep - left) / (right - left)
            return oracle_vectors[left] * (1.0 - alpha) + oracle_vectors[right] * alpha
        if condition == "global_best":
            anchors = sorted(best_global)
            if timestep <= anchors[0]:
                value = best_global[anchors[0]]
            elif timestep >= anchors[-1]:
                value = best_global[anchors[-1]]
            else:
                left = max(value for value in anchors if value <= timestep)
                right = min(value for value in anchors if value >= timestep)
                alpha = (timestep - left) / (right - left)
                value = best_global[left] * (1.0 - alpha) + best_global[right] * alpha
            return torch.full((102,), value, device=self.sd.device_torch, dtype=torch.float32)
        if condition == "phase_c_v2":
            router, router_config = phase_c
            projected = self._project_visual_activator(embeds, router_config, torch)
            code = router.encode_activator(projected)
            normalized = torch.tensor([timestep / 1000.0], device=self.sd.device_torch, dtype=torch.float32)
            return router(normalized, code).reshape(-1)
        raise ValueError(f"unknown C0 visual condition: {condition}")

    def _load_phase_c_v2(self, torch):
        from safetensors.torch import load_file
        from toolkit.residual_gating import ResidualGateRouter

        handoff_path = Path(self.c0.novel_visuals.phase_c_v2_handoff)
        handoff = load_json(handoff_path)
        if handoff.get("schema") != "ai-toolkit.ideogram4-v3-phase-c-v2-handoff" or int(handoff.get("schema_version", 0)) != 2 or handoff.get("status") != "completed":
            raise RuntimeError("invalid or incomplete Phase C V2 handoff for C0 visuals")
        if handoff.get("active_registry_fingerprint") != self._registry_fingerprint:
            raise RuntimeError("Phase C V2 visual router registry disagrees with C0 active registry")
        key = self.c0.novel_visuals.phase_c_v2_checkpoint
        refs = handoff.get(key, {})
        if not isinstance(refs, Mapping):
            raise RuntimeError(f"Phase C V2 handoff lacks {key} checkpoint")
        router_ref = refs.get("router", {})
        config_ref = refs.get("config", {})
        router_path = (handoff_path.parent / str(router_ref.get("path", ""))).resolve()
        config_path = (handoff_path.parent / str(config_ref.get("path", ""))).resolve()
        if not router_path.is_file() or sha256_file(router_path) != router_ref.get("sha256"):
            raise RuntimeError("Phase C V2 visual router file/hash mismatch")
        if not config_path.is_file() or sha256_file(config_path) != config_ref.get("sha256"):
            raise RuntimeError("Phase C V2 visual router config file/hash mismatch")
        config = load_json(config_path)
        if config.get("schema") != "ai-toolkit.ideogram4-v3-phase-c-v2-router-config" or int(config.get("schema_version", 0)) != 2:
            raise RuntimeError("invalid Phase C V2 router config for C0 visuals")
        active = config.get("active_registry", {})
        if active.get("group_count") != 102:
            raise RuntimeError("Phase C V2 visual router does not contain 102 active groups")
        architecture = config
        router = ResidualGateRouter(
            group_count=102, conditioning_dim=int(architecture["conditioning_dim"]),
            activator_token_count=int(architecture["activator_token_count"]),
            activator_token_dim=int(architecture["activator_token_dim"]),
            temporal_anchor_count=int(architecture["temporal_anchor_count"]),
            contextual_rank=int(architecture["contextual_rank"]), q_max=float(architecture["q_max"]),
        ).to(self.sd.device_torch, dtype=torch.float32)
        router.load_state_dict(load_file(str(router_path), device=str(self.sd.device_torch)), strict=True)
        router.eval()
        return router, config

    def _project_visual_activator(self, embeds, router_config, torch):
        from toolkit.residual_gating import aggregate_activator_occurrences

        architecture = router_config
        features = embeds.text_embeds[0]
        mask = embeds.trigger_masks[0].to(dtype=torch.bool)
        positions = torch.nonzero(mask, as_tuple=False).reshape(-1)
        token_count = int(architecture["activator_token_count"])
        occurrence_count = int(architecture.get("activator_occurrence_count", 3))
        if positions.numel() != token_count * occurrence_count:
            raise RuntimeError("novel prompt activator occurrence count disagrees with Phase C V2 handoff")
        transformer = self.sd.model
        selected = features[positions].to(self.sd.device_torch, dtype=self.sd.torch_dtype)
        projected = transformer.llm_cond_proj(transformer.llm_cond_norm(selected)).float()
        return aggregate_activator_occurrences(projected, occurrence_count=occurrence_count, token_count=token_count, mode=str(architecture.get("activator_occurrence_mode", "additive")))
