from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence

from jobs.process import BaseExtensionProcess

from .helpers import (
    ResidualAblationRuntime,
    aggregate_records,
    annotate_viability_metrics,
    atomic_write_json,
    atomic_write_jsonl,
    build_helper_response_bases,
    build_module_registry,
    build_report,
    finite_difference_metrics,
    gate_gradients,
    load_json,
    load_jsonl,
    parse_activator_a2_contract,
    registry_maps,
    require_file_hash,
    residual_runtime_context,
    resume_key,
    sha256_file,
    validate_complete_partition,
)


class Ideogram4V3ResidualAblationProcess(BaseExtensionProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.device = self.get_conf("device", getattr(job, "device", "cuda"))
        self.output_root = Path(self.get_conf("output_root", required=True))
        self.ablation = self.get_conf("ablation", required=True)
        self.model_definition = self.get_conf("model")
        self.network_definition = self.get_conf("network")
        self.run_label = str(self.get_conf("run_id", self.name))
        forbidden = sorted(key for key in ("optimizer", "scheduler", "train", "steps") if key in self.raw_process_config)
        if forbidden:
            raise ValueError(f"diagnostics-only residual ablation rejects training fields: {forbidden}")
        self.sd = None
        self.network = None
        self.text_activator = None
        self._run_started_at = None
        self._calibrated_embed_cache = {}
        self._stock_embed_cache = {}
        self._latent_cache = {}

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {seconds:02d}s"
        if minutes:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"

    def _progress(self, message: str) -> None:
        from toolkit.print import print_acc

        elapsed = 0.0 if self._run_started_at is None else time.monotonic() - self._run_started_at
        print_acc(f"[V3 residual ablation +{self._format_duration(elapsed)}] {message}")

    def _log_cuda_memory(self, label: str, torch) -> None:
        if not torch.cuda.is_available():
            return
        gib = 1024 ** 3
        allocated = torch.cuda.memory_allocated() / gib
        reserved = torch.cuda.memory_reserved() / gib
        peak = torch.cuda.max_memory_allocated() / gib
        free, total = torch.cuda.mem_get_info()
        self._progress(
            f"CUDA memory {label}: allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB, "
            f"peak={peak:.2f} GiB, free={free / gib:.2f}/{total / gib:.2f} GiB"
        )

    def run(self):
        super().run()
        import torch

        self._run_started_at = time.monotonic()
        self._progress("resolving A2 contract and artifact hashes")
        self._resolve_inputs()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._progress("loading Ideogram4 model")
        self.sd = self._load_model()
        self._progress("loading frozen V3 LoRA")
        self.network = self._load_lora(torch)
        self._progress("installing calibrated A2 activator")
        self.text_activator = self._build_text_activator(torch)
        self._load_and_install_activator()
        self._freeze()
        self._enable_checkpointing()
        self._progress("model, V3 LoRA, and activator ready")
        self._log_cuda_memory("after model setup", torch)

        registry = build_module_registry(self.network.get_all_modules())
        registry_summary = validate_complete_partition(registry)
        module_to_group, group_to_modules = registry_maps(registry)
        registry_path = self.output_root / self.ablation.get("registry_filename", "lora_module_registry.json")
        atomic_write_json(registry_path, {"summary": registry_summary, "modules": registry, "groups": group_to_modules})

        probes, probe_manifest = self._build_probes()
        probe_path = self.output_root / self.ablation.get("probe_manifest_filename", "residual_ablation_probe_manifest.json")
        atomic_write_json(probe_path, probe_manifest)
        self._precompute_probe_inputs(probes, torch)
        self._progress(
            f"prepared {len(probes)} probe cases across {len(group_to_modules)} groups; "
            f"optimized workload per probe: {len(self.helper_phrases) + 2} references, "
            f"1 no-grad center capture, checkpointed gate-gradient batches of "
            f"{int(self.ablation.get('gate_gradient_batch_size', 1))}, at most "
            f"{2 * int(self.ablation.get('finite_difference_top_k_groups', 12))} core side-scale passes "
            f"plus family coverage"
        )

        from toolkit.trigger_binding_artifacts import fingerprint

        run_id = fingerprint({
            "label": self.run_label,
            "a2_contract_sha256": sha256_file(self.a2_contract_path),
            "snapshot_sha256": sha256_file(self.snapshot_path),
            "v3_weights_sha256": sha256_file(self.lora_path),
            "embedding_sha256": sha256_file(self.embedding_path),
            "te_adapter_sha256": sha256_file(self.te_adapter_path),
            "probe_manifest_hash": probe_manifest["probe_manifest_hash"],
            "registry": registry,
            "helpers": self.helper_phrases,
            "candidate_scales": self._candidate_scales(),
            "finite_difference_top_k_groups": int(self.ablation.get("finite_difference_top_k_groups", 12)),
            "gate_gradient_batch_size": int(self.ablation.get("gate_gradient_batch_size", 1)),
            "schema_version": 5,
        })
        records_path = self.output_root / self.ablation.get("records_filename", "residual_ablation_records.jsonl")
        existing = load_jsonl(records_path) if self.ablation.get("resume", True) else []
        unique = {}
        for record in existing:
            key = resume_key(record)
            if key in unique:
                raise RuntimeError(f"duplicate resume record key: {key}")
            unique[key] = record
        records = [record for record in unique.values() if record.get("run_id") == run_id]
        completed = {resume_key(record) for record in records}

        runtime = ResidualAblationRuntime(
            module_to_group,
            capture=True,
            capture_vectors=True,
            sketch_size=int(self.ablation.get("residual_sketch_size", 256)),
        )
        self.network.is_active = True
        completed_probe_count = sum(
            all((run_id, case.probe_case_id, group) in completed for group in group_to_modules)
            for case, _ in probes
        )
        probe_durations = []
        for case_index, (case, item) in enumerate(probes, 1):
            pending = [group for group in group_to_modules if (run_id, case.probe_case_id, group) not in completed]
            if not pending:
                self._progress(f"probe {case_index}/{len(probes)} already complete; resuming past it")
                continue
            probe_started = time.monotonic()
            self._progress(
                f"probe {case_index}/{len(probes)} start: split={case.split}, timestep={case.timestep}, "
                f"pending_groups={len(pending)}"
            )
            prepared = self._prepare_case(case, item, torch)
            embed_key = str(case.item_id)
            if embed_key not in self._calibrated_embed_cache:
                raise RuntimeError(f"missing precomputed calibrated embedding for {embed_key}")
            self._progress(f"probe {case_index}/{len(probes)} reused calibrated text embedding")
            prepared["calibrated_embeds"] = self._calibrated_embed_cache[embed_key]
            helper_basis, helper_norms, helper_spectra, reference_summary = self._capture_reference_responses(
                prepared, runtime, torch, case_index=case_index, case_count=len(probes)
            )
            runtime.set_projection_basis(helper_basis)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._log_cuda_memory(f"before probe {case_index} center capture", torch)
            case_records = self._evaluate_case(
                run_id, case, prepared, pending, registry, runtime,
                helper_norms, helper_spectra, reference_summary, torch,
                case_index=case_index, case_count=len(probes),
            )
            records.extend(case_records)
            records = annotate_viability_metrics(records)
            atomic_write_jsonl(records_path, sorted(records, key=resume_key))
            completed.update(resume_key(record) for record in case_records)
            completed_probe_count += 1
            duration = time.monotonic() - probe_started
            probe_durations.append(duration)
            remaining = len(probes) - completed_probe_count
            eta = (sum(probe_durations) / len(probe_durations)) * remaining
            self._progress(
                f"probe {case_index}/{len(probes)} complete in {self._format_duration(duration)}; "
                f"records={len(records)}/{len(probes) * len(group_to_modules)}, ETA={self._format_duration(eta)}"
            )

        expected = len(probes) * len(group_to_modules)
        if len(records) != expected:
            raise RuntimeError(f"residual ablation completion contract failed: {len(records)} != {expected}")
        records = annotate_viability_metrics(records)
        atomic_write_jsonl(records_path, sorted(records, key=resume_key))
        aggregates_path = self.output_root / self.ablation.get("aggregates_filename", "residual_ablation_aggregates.jsonl")
        report_path = self.output_root / self.ablation.get("report_filename", "residual_ablation_report.md")
        manifest_path = self.output_root / self.ablation.get("manifest_filename", "residual_ablation_manifest.json")
        atomic_write_jsonl(aggregates_path, aggregate_records(records))
        report_path.write_text(build_report(records, registry_summary), encoding="utf-8", newline="\n")
        artifacts = {
            "registry": self._artifact_ref(registry_path),
            "probes": self._artifact_ref(probe_path),
            "records": self._artifact_ref(records_path),
            "aggregates": self._artifact_ref(aggregates_path),
            "report": self._artifact_ref(report_path),
        }
        atomic_write_json(manifest_path, {
            "schema": "ai-toolkit.ideogram4-v3-residual-ablation",
            "schema_version": 5,
            "status": "completed",
            "run_id": run_id,
            "diagnostics_only": True,
            "phase_c_started": False,
            "calibrated_activator_installed": True,
            "helper_basis_source": "actual_v3_lora_residual_responses",
            "helper_spectrum": {
                "matrix": "per_probe_per_group_helper_residual_sketch_columns",
                "top_energy_fraction": "sigma_1^2 / sum_i sigma_i^2",
                "effective_rank": "exp(-sum_i p_i log p_i)",
                "stored_in": "records.helper_response_spectrum",
            },
            "candidate_scales": self._candidate_scales(),
            "evaluation_strategy": {
                "shared_center_capture_pass_per_probe": True,
                "gate_gradients": "checkpointed_batched_backward_without_residual_capture",
                "gate_gradient_batch_size": int(self.ablation.get("gate_gradient_batch_size", 1)),
                "finite_difference_selection": "top_absolute_gradient_plus_family_coverage",
                "finite_difference_top_k_groups": int(self.ablation.get("finite_difference_top_k_groups", 12)),
                "side_scale_passes_per_selected_group": 2,
                "calibrated_text_embeddings_cached_per_item": True,
                "stock_text_embeddings_cached_per_prompt": True,
                "latents_cached_per_item": True,
                "conditioning_and_vae_offloaded_after_precompute": True,
                "residual_statistics": "chunked_without_full_tensor_boolean_copy",
            },
            "probe_case_count": len(probes),
            "group_count": len(group_to_modules),
            "module_count": len(registry),
            "expected_record_count": expected,
            "actual_record_count": len(records),
            "inputs": self.input_refs,
            "artifacts": artifacts,
        })

    def _resolve_inputs(self):
        self.a2_contract_path = Path(self.ablation["a2_contract"])
        if not self.a2_contract_path.is_file():
            raise FileNotFoundError(self.a2_contract_path)
        contract = load_json(self.a2_contract_path)
        parsed = parse_activator_a2_contract(contract)
        self.snapshot_path = require_file_hash(parsed["snapshot"], parsed["snapshot_sha256"], "A2 config snapshot")
        self.lora_path = require_file_hash(parsed["v3_weights"], parsed["v3_weights_sha256"], "V3 source weights")
        if parsed.get("best_manifest_sha256"):
            best_manifest_path = require_file_hash(
                parsed["best_manifest"], parsed["best_manifest_sha256"], "A2 best-heldout manifest"
            )
        else:
            best_manifest_path = Path(parsed["best_manifest"])
            if not best_manifest_path.is_file():
                raise FileNotFoundError(best_manifest_path)
        best_manifest = load_json(best_manifest_path)
        if best_manifest.get("schema") != "ai-toolkit.ideogram4-v3-activator-best-heldout" or int(best_manifest.get("schema_version", 0)) != 1:
            raise RuntimeError("invalid A2 best-heldout manifest")
        best_artifacts = best_manifest.get("artifacts", {})
        embedding_ref = best_artifacts.get("embedding", {})
        te_ref = best_artifacts.get("te_adapter", {})
        if Path(str(parsed["best_embedding"])).resolve() != Path(str(embedding_ref.get("path", ""))).resolve():
            raise RuntimeError("A2 contract best_embedding disagrees with best-heldout manifest")
        if Path(str(parsed["best_te_adapter"])).resolve() != Path(str(te_ref.get("path", ""))).resolve():
            raise RuntimeError("A2 contract best_te_adapter disagrees with best-heldout manifest")
        if parsed.get("best_embedding_sha256") and parsed["best_embedding_sha256"] != embedding_ref.get("sha256"):
            raise RuntimeError("A2 contract best_embedding hash disagrees with best-heldout manifest")
        if parsed.get("best_te_adapter_sha256") and parsed["best_te_adapter_sha256"] != te_ref.get("sha256"):
            raise RuntimeError("A2 contract best_te_adapter hash disagrees with best-heldout manifest")
        self.embedding_path = require_file_hash(parsed["best_embedding"], embedding_ref.get("sha256"), "A2 best embedding")
        self.te_adapter_path = require_file_hash(parsed["best_te_adapter"], te_ref.get("sha256"), "A2 best TE adapter")
        source_records = contract.get("sources", {})
        canonical_sources = json.dumps(source_records, sort_keys=True).encode("utf-8")
        if hashlib.sha256(canonical_sources).hexdigest() != contract.get("source_fingerprint"):
            raise RuntimeError("A2 source fingerprint mismatch")
        self.input_refs = {
            "a2_contract": {"path": str(self.a2_contract_path), "sha256": sha256_file(self.a2_contract_path)},
            "a2_snapshot": {"path": str(self.snapshot_path), "sha256": sha256_file(self.snapshot_path)},
            "v3_weights": {"path": str(self.lora_path), "sha256": sha256_file(self.lora_path)},
            "best_embedding": {"path": str(self.embedding_path), "sha256": sha256_file(self.embedding_path)},
            "best_te_adapter": {"path": str(self.te_adapter_path), "sha256": sha256_file(self.te_adapter_path)},
            "best_manifest": {"path": str(best_manifest_path), "sha256": sha256_file(best_manifest_path)},
        }
        self._load_snapshot_config()

    def _load_snapshot_config(self):
        import yaml

        with self.snapshot_path.open("r", encoding="utf-8") as handle:
            snapshot = yaml.safe_load(handle)
        processes = snapshot.get("config", {}).get("process", ()) if isinstance(snapshot, Mapping) else ()
        source = next((value for value in processes if isinstance(value, Mapping)), None)
        if source is None:
            raise RuntimeError("A2 stage snapshot lacks process configuration")
        phase = source.get("three_phase_trigger_training", {})
        if phase.get("objective_mode") != "ideogram4_v3_activator":
            raise RuntimeError("A2 snapshot objective is not ideogram4_v3_activator")
        if phase.get("runtime", {}).get("active_phase") != "a2":
            raise RuntimeError("A2 snapshot runtime phase is not a2")
        runtime_objective = phase.get("phase_runtime", {}).get("objective", {})
        if not runtime_objective.get("v3_active_frozen") or not runtime_objective.get("te_only"):
            raise RuntimeError("A2 snapshot does not record frozen active V3 with TE-only calibration")
        snapshot_model = dict(source.get("model") or {})
        snapshot_network = dict(source.get("network") or {})
        if not snapshot_model or snapshot_model.get("arch") != "ideogram4":
            raise RuntimeError("A2 snapshot lacks Ideogram4 model configuration")
        pretrained = snapshot_network.pop("pretrained_lora_path", None)
        if pretrained and Path(str(pretrained)).resolve() != self.lora_path.resolve():
            raise RuntimeError("A2 snapshot V3 weight source disagrees with contract")
        self.model_definition = snapshot_model
        self.network_definition = snapshot_network
        self.trigger_config = dict(phase.get("trigger") or {})
        self.activator_config = dict(phase.get("text_activator") or {})
        helper_schedule = runtime_objective.get("helper_schedule", {})
        configured_helpers = self.ablation.get("helpers")
        self.helper_phrases = [str(value) for value in (configured_helpers or helper_schedule.get("helpers") or ())]
        if not self.helper_phrases:
            raise RuntimeError("A2 snapshot/config does not provide helper phrases")
        self.data_split_config = dict(phase.get("data_split") or {})
        self.validation_config = dict(phase.get("validation") or {})
        self.placeholder = str(self.trigger_config.get("placeholder", "[trigger]"))
        self.literal = str(self.trigger_config.get("literal", ""))
        if not self.literal:
            raise RuntimeError("A2 snapshot lacks trigger literal")
        self.neutral_phrase = str(self.ablation.get("neutral_phrase", ""))
        self.far_phrase = str(self.ablation.get("far_phrase", "landscape photograph"))

    def _load_model(self):
        from toolkit.config_modules import ModelConfig
        from toolkit.util.get_model import get_model_class

        model_config = ModelConfig(**self.model_definition)
        model_class = get_model_class(model_config)
        sd = model_class(device=self.device, model_config=model_config, dtype=model_config.dtype, noise_scheduler=model_class.get_train_scheduler())
        sd.load_model()
        for module in (sd.model, sd.text_encoder, sd.vae):
            module.eval()
            module.requires_grad_(False)
        return sd

    def _load_lora(self, torch):
        from safetensors.torch import load_file
        from toolkit.config_modules import NetworkConfig
        from toolkit.lora_special import LoRASpecialNetwork

        weights = load_file(str(self.lora_path), device="cpu")
        rank = next((int(value.shape[0]) for key, value in weights.items() if key.endswith("lora_A.weight") or key.endswith("lora_down.weight")), None)
        if rank is None:
            raise RuntimeError("unable to infer LoRA rank from V3 weights")
        network_values = {"type": "lora", "linear": rank, "linear_alpha": rank, "transformer_only": True} | dict(self.network_definition or {})
        network_values.pop("pretrained_lora_path", None)
        config = NetworkConfig(**network_values)
        network = LoRASpecialNetwork(
            text_encoder=None, unet=self.sd.model, lora_dim=config.linear, multiplier=1.0,
            alpha=config.linear_alpha, train_unet=True, train_text_encoder=False,
            network_config=config, network_type=config.type, transformer_only=config.transformer_only,
            is_transformer=True, target_lin_modules=self.sd.target_lora_modules, base_model=self.sd,
            **config.network_kwargs,
        )
        network.apply_to(None, self.sd.model, apply_text_encoder=False, apply_unet=True)
        network.force_to(self.sd.device_torch, dtype=torch.float32)
        network._update_torch_multiplier()
        network.load_weights(str(self.lora_path))
        network.eval()
        return network

    def _build_text_activator(self, torch):
        from toolkit.models.ideogram4_trigger_activator import TextActivator
        from toolkit.train_tools import get_torch_dtype

        config = self.activator_config
        te = config.get("te_adapter", {})
        if config.get("architecture_mode") != "module_lora" or te.get("mode") != "module_lora":
            raise RuntimeError("A2 activator must use module_lora topology")
        language_model = self.sd.text_encoder.language_model
        hidden_size = int(language_model.config.hidden_size)
        self.sd.tokenizer.add_tokens([self.literal], special_tokens=True)
        atomic_ids = self.sd.tokenizer(self.literal, add_special_tokens=False)["input_ids"]
        if len(atomic_ids) != 1:
            raise RuntimeError(f"trigger literal is not atomic after registration: {atomic_ids}")
        embedding = config.get("embedding", {})
        activator = TextActivator(
            embedding_dim=hidden_size,
            hidden_size=hidden_size,
            embedding_tokens=int(embedding.get("tokens", 4)),
            initializer=torch.zeros(1, hidden_size, device=self.sd.device_torch),
            te_adapter_mode="module_lora",
            te_rank=int(te.get("rank", 4)),
            te_alpha=float(te.get("alpha", te.get("rank", 4))),
            te_dropout=float(te.get("dropout", 0.0)),
            te_target_modules=te.get("target_modules", te.get("child_modules", ("down_proj",))),
            te_parent_modules=te.get("parent_modules", ()),
            te_layers=te.get("layers", "all"),
            create_tap_adapters=False,
            gamma_trainable=False,
        ).to(self.sd.device_torch, dtype=get_torch_dtype(embedding.get("dtype", "bf16")))
        activator.atomic_token_id = int(atomic_ids[0])
        activator.lookup_token_id = int(atomic_ids[0])
        activator.install_module_lora(language_model)
        return activator

    def _load_and_install_activator(self):
        from toolkit.trigger_binding_artifacts import load_artifact

        embedding, _ = load_artifact(self.embedding_path, expected_type="embedding", expected_file_sha256=sha256_file(self.embedding_path))
        te_adapter, _ = load_artifact(self.te_adapter_path, expected_type="te_adapter", expected_file_sha256=sha256_file(self.te_adapter_path))
        self.text_activator.embedding.load_state_dict(embedding, strict=True)
        self.text_activator.te_adapter_artifact_module().load_state_dict(te_adapter, strict=True)
        installer = getattr(self.sd, "install_text_activator", None) or getattr(self.sd, "set_text_activator", None)
        if not callable(installer):
            raise RuntimeError("Ideogram4 model does not expose text activator installation")
        # A1/A2 deliberately have no tap adapters. semantic_only is the complete
        # calibrated activator path for this topology: virtual embedding + TE LoRA.
        installer(self.text_activator, runtime_mode="semantic_only")

    def _freeze(self):
        for module in (self.sd.model, self.sd.text_encoder, self.sd.vae, self.network, self.text_activator):
            module.requires_grad_(False)
            module.eval()
        trainable = [name for name, parameter in self.text_activator.named_parameters() if parameter.requires_grad]
        trainable += [name for name, parameter in self.network.named_parameters() if parameter.requires_grad]
        if trainable:
            raise RuntimeError(f"diagnostics found trainable parameters: {trainable[:10]}")

    def _enable_checkpointing(self):
        enable = getattr(self.sd.model, "enable_gradient_checkpointing", None)
        if callable(enable):
            enable()
        if not getattr(self.sd.model, "gradient_checkpointing", False):
            raise RuntimeError("residual ablation gate gradients require transformer checkpointing")

    def _candidate_scales(self):
        values = tuple(float(value) for value in self.ablation.get("candidate_scales", (0.75, 1.0, 1.25)))
        if values != (0.75, 1.0, 1.25):
            raise ValueError("candidate_scales must be exactly [0.75, 1.0, 1.25]")
        return values

    def _build_probes(self):
        from extensions_built_in.ideogram4_tap_ablation.helpers import select_probe_items
        from toolkit.semantic_scaffold_calibration import build_fixed_probe_cases
        from toolkit.trigger_binding_artifacts import fingerprint
        from toolkit.trigger_data_split import load_data_split_manifest

        split_manifest = self.ablation.get("split_manifest") or self.data_split_config.get("manifest_path") or self.validation_config.get("data_split_manifest")
        if not split_manifest:
            raise RuntimeError("A2 snapshot/config lacks fixed data split manifest")
        seed = int(self.ablation.get("seed", self.data_split_config.get("seed", self.validation_config.get("seed", 42))))
        split = load_data_split_manifest(split_manifest, expected_seed=seed)
        items = select_probe_items(
            self.ablation["dataset_root"], split.as_dict(), train_limit=int(self.ablation.get("train_probe_count", 8)),
            expected_heldout=int(self.ablation.get("expected_heldout_count", 4)), placeholder=self.placeholder,
        )
        cases = build_fixed_probe_cases(
            items,
            {"train_item_ids": [item["dataset_relative_item_id"] for item in items if item["split"] == "train"], "heldout_item_ids": [item["dataset_relative_item_id"] for item in items if item["split"] == "heldout"]},
            noise_seeds=[seed], fixed_timesteps=[int(value) for value in self.ablation.get("timesteps", self.validation_config.get("fixed_timesteps", ()))],
            probe_scope="split", target_mode="flow",
        )
        item_by_id = {item["dataset_relative_item_id"]: item for item in items}
        probes = [(case, item_by_id[case.item_id]) for case in cases]
        payload = {"schema": "ai-toolkit.ideogram4-v3-residual-ablation-probes", "schema_version": 2, "items": items, "cases": [case.as_dict() for case in cases], "helpers": self.helper_phrases, "neutral_phrase": self.neutral_phrase, "far_phrase": self.far_phrase}
        payload["probe_manifest_hash"] = fingerprint(payload)
        return probes, payload

    def _precompute_probe_inputs(self, probes, torch):
        from toolkit.basic import flush

        unique_items = {}
        for case, item in probes:
            unique_items.setdefault(str(case.item_id), (case, item))
        self._progress(f"precomputing latents and text embeddings for {len(unique_items)} unique contents")
        for item_index, (item_id, (case, item)) in enumerate(unique_items.items(), 1):
            prepared = self._prepare_case(case, item, torch)
            template = prepared["prompt_template"]
            with torch.no_grad():
                self._calibrated_embed_cache[item_id] = self._calibrated_prompt_embeds(template).to("cpu")
                phrases = list(self.helper_phrases) + [self.neutral_phrase, self.far_phrase]
                for phrase in phrases:
                    prompt = template.replace(self.placeholder, phrase)
                    if prompt not in self._stock_embed_cache:
                        self._stock_embed_cache[prompt] = self.sd.get_prompt_embeds(
                            [prompt], runtime_mode="stock_literal"
                        ).to("cpu")
            self._latent_cache[item_id] = self._latent_cache[item_id].to("cpu")
            self._progress(f"precomputed content {item_index}/{len(unique_items)}")

        self.sd.text_encoder.to("cpu")
        self.text_activator.to("cpu")
        self.sd.vae.to("cpu")
        flush()
        self._progress("offloaded Qwen text encoder, activator, and VAE after precompute")
        self._log_cuda_memory("after conditioning/VAE offload", torch)

    def _prepare_case(self, case, item, torch):
        from PIL import Image
        from torchvision import transforms

        item_id = str(case.item_id)
        latent = self._latent_cache.get(item_id)
        if latent is None:
            with Image.open(item["image_path"]) as image:
                image = image.convert("RGB")
                width, height = max(16, image.width // 16 * 16), max(16, image.height // 16 * 16)
                tensor = transforms.ToTensor()(image.resize((width, height), Image.Resampling.BICUBIC)) * 2.0 - 1.0
            latent = self.sd.encode_images([tensor], device=self.sd.device_torch, dtype=self.sd.torch_dtype)
            self._latent_cache[item_id] = latent
        else:
            latent = latent.to(device=self.sd.device_torch, dtype=self.sd.torch_dtype)
        generator = torch.Generator(device=latent.device).manual_seed(int(case.noise_seed))
        noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)
        schedule = max(0.0, min(1.0, float(case.timestep) / 1000.0))
        return {"latent": latent, "noise": noise, "noisy_latents": latent * (1.0 - schedule) + noise * schedule, "timesteps": torch.tensor([float(case.timestep)], device=latent.device, dtype=latent.dtype), "target": noise - latent, "prompt_template": item["caption"]}

    def _calibrated_prompt_embeds(self, prompt_template):
        from extensions_built_in.diffusion_models.ideogram4.src.pipeline import get_qwen3_vl_features
        from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
        from toolkit.trigger_binding import bind_trigger_batch

        batch = bind_trigger_batch(
            self.sd.tokenizer, [prompt_template], self.literal, placeholder=self.placeholder,
            max_length=getattr(self.sd, "max_text_length", None), require_placeholder=True,
            mask_all_occurrences=True, require_atomic=True, expected_token_id=self.text_activator.atomic_token_id,
            virtual_tokens=self.text_activator.virtual_tokens,
        ).to(self.sd.text_encoder.device)
        features = get_qwen3_vl_features(
            self.sd.text_encoder, batch.input_ids, batch.attention_mask,
            (batch.attention_mask.cumsum(dim=-1) - 1).clamp(min=0).long(),
            trigger_mask=batch.trigger_mask, token_indices=batch.token_indices,
            text_activator=self.text_activator, runtime_mode="semantic_only", runtime_metadata=batch.runtime_metadata(),
        )
        valid = int(batch.attention_mask[0].sum().item())
        return AdvancedPromptEmbeds(
            text_embeds=[features[0, :valid].to(self.sd.torch_dtype)],
            trigger_masks=[batch.trigger_mask[0, :valid]],
        )

    def _predict(self, prepared, prompt, runtime_mode):
        if runtime_mode == "full":
            embeds = prepared.get("calibrated_embeds")
            if embeds is None:
                embeds = self._calibrated_prompt_embeds(prompt)
        else:
            embeds = self._stock_embed_cache.get(prompt)
            if embeds is None:
                embeds = self.sd.get_prompt_embeds([prompt], runtime_mode="stock_literal")
                self._stock_embed_cache[prompt] = embeds
        batch = SimpleNamespace(latents=prepared["latent"], file_items=[], audio_pred_slot=None)
        return self.sd.predict_noise(
            latents=prepared["noisy_latents"], timestep=prepared["timesteps"],
            conditional_embeddings=embeds.to(self.sd.device_torch, dtype=prepared["noisy_latents"].dtype),
            unconditional_embeddings=None, guidance_scale=1.0, guidance_embedding_scale=1.0,
            bypass_guidance_embedding=False, batch=batch,
        )

    def _capture_reference_responses(self, prepared, runtime, torch, *, case_index, case_count):
        captured = {}
        summaries = {}
        reference_count = len(self.helper_phrases) + 2
        reference_index = 0
        with residual_runtime_context(self.network, runtime):
            runtime.set_gates({})
            for helper in self.helper_phrases:
                reference_index += 1
                reference_started = time.monotonic()
                runtime.reset()
                prompt = prepared["prompt_template"].replace(self.placeholder, helper)
                with torch.no_grad():
                    prediction = self._predict(prepared, prompt, "stock_literal")
                self._progress(
                    f"probe {case_index}/{case_count} reference {reference_index}/{reference_count} "
                    f"helper={helper!r} finished in {self._format_duration(time.monotonic() - reference_started)}"
                )
                captured[helper] = runtime.captured_group_vectors()
                summaries[f"helper:{helper}"] = {
                    "loss": float(torch.mean((prediction.float() - prepared["target"].float()).square()).item()),
                    "groups": runtime.summary()["groups"],
                }
            for name, phrase in (("neutral", self.neutral_phrase), ("far", self.far_phrase)):
                reference_index += 1
                reference_started = time.monotonic()
                runtime.reset()
                prompt = prepared["prompt_template"].replace(self.placeholder, phrase)
                with torch.no_grad():
                    prediction = self._predict(prepared, prompt, "stock_literal")
                self._progress(
                    f"probe {case_index}/{case_count} reference {reference_index}/{reference_count} "
                    f"{name} finished in {self._format_duration(time.monotonic() - reference_started)}"
                )
                summaries[name] = {
                    "phrase": phrase,
                    "loss": float(torch.mean((prediction.float() - prepared["target"].float()).square()).item()),
                    "groups": runtime.summary()["groups"],
                }
        bases, mean_norms, spectra = build_helper_response_bases(captured)
        return bases, mean_norms, spectra, summaries

    def _evaluate_case(
        self, run_id, case, prepared, pending, registry, runtime,
        helper_norms, helper_spectra, references, torch,
        *, case_index, case_count,
    ):
        by_group = {row["group_id"]: row for row in registry}
        scales = self._candidate_scales()
        side_scales = [scale for scale in scales if scale != 1.0]
        self._progress(f"probe {case_index}/{case_count} center capture pass for all {len(pending)} groups")
        center_started = time.monotonic()
        runtime.capture = True
        runtime.capture_vectors = True
        with residual_runtime_context(self.network, runtime):
            runtime.reset()
            runtime.set_gates({})
            with torch.no_grad():
                prediction = self._predict(prepared, prepared["prompt_template"], "full")
            center_loss = float(torch.mean((prediction.float() - prepared["target"].float()).square()).item())
            center_summary = runtime.summary()
            center_vectors = runtime.captured_group_vectors()
        self._progress(
            f"probe {case_index}/{case_count} center capture finished in "
            f"{self._format_duration(time.monotonic() - center_started)}"
        )
        self._log_cuda_memory(f"after probe {case_index} center capture", torch)

        gradient_batch_size = int(self.ablation.get("gate_gradient_batch_size", 1))
        if gradient_batch_size <= 0:
            raise ValueError("gate_gradient_batch_size must be positive")
        gradients = {}
        runtime.capture = False
        runtime.capture_vectors = False
        gradient_batches = [pending[index:index + gradient_batch_size] for index in range(0, len(pending), gradient_batch_size)]
        for batch_index, batch_groups in enumerate(gradient_batches, 1):
            batch_started = time.monotonic()
            gates = {
                group: torch.tensor(1.0, device=prepared["target"].device, dtype=torch.float32, requires_grad=True)
                for group in batch_groups
            }
            with residual_runtime_context(self.network, runtime):
                runtime.reset()
                runtime.set_gates(gates)
                prediction = self._predict(prepared, prepared["prompt_template"], "full")
                loss = torch.mean((prediction.float() - prepared["target"].float()).square())
                gradients.update(gate_gradients(loss, gates))
            del prediction, loss, gates
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            remaining = len(gradient_batches) - batch_index
            elapsed = time.monotonic() - batch_started
            self._progress(
                f"probe {case_index}/{case_count} gate-gradient batch {batch_index}/{len(gradient_batches)} "
                f"groups={batch_groups} finished in {self._format_duration(elapsed)}; "
                f"gradient ETA={self._format_duration(elapsed * remaining)}"
            )
            self._log_cuda_memory(f"after gradient batch {batch_index}", torch)
        runtime.capture = True
        runtime.capture_vectors = True

        finite_difference_top_k = int(self.ablation.get("finite_difference_top_k_groups", 12))
        if finite_difference_top_k <= 0:
            raise ValueError("finite_difference_top_k_groups must be positive")
        ranked = sorted(
            pending,
            key=lambda group: abs(gradients[group]) if gradients.get(group) is not None else -1.0,
            reverse=True,
        )
        selected = set(ranked[:finite_difference_top_k])
        for kind in ("attention", "mlp", "adaln", "other"):
            candidates = [group for group in ranked if by_group[group]["kind"] == kind]
            if candidates:
                selected.add(candidates[0])
        self._progress(
            f"probe {case_index}/{case_count} selected {len(selected)}/{len(pending)} groups for finite differences; "
            "all groups retain exact joint dL/dg"
        )

        output = []
        group_durations = []
        selected_completed = 0
        for group_index, group in enumerate(pending, 1):
            group_started = time.monotonic()
            losses = {1.0: center_loss}
            if group in selected:
                with residual_runtime_context(self.network, runtime):
                    for scale in side_scales:
                        runtime.reset()
                        runtime.set_gates({group: scale})
                        with torch.no_grad():
                            prediction = self._predict(prepared, prepared["prompt_template"], "full")
                        losses[scale] = float(torch.mean((prediction.float() - prepared["target"].float()).square()).item())
                metrics = finite_difference_metrics(losses)
                metrics["fd_result"] = "prefer_" + str(metrics["best_scale"])
                selected_completed += 1
            else:
                metrics = {
                    "left_slope": None,
                    "right_slope": None,
                    "central_secant": None,
                    "curvature": None,
                    "best_scale": None,
                    "best_loss": center_loss,
                    "center_loss": center_loss,
                    "fd_result": "not_selected_joint_gradient_only",
                }
            metrics["dL_dg"] = gradients[group]
            group_stats = center_summary.get("groups", {}).get(group, {})
            residual_rms = group_stats.get("rms")
            helper_norm = float(helper_norms.get(group, 0.0))
            activator_vector = center_vectors.get(group)
            activator_norm = float(activator_vector.norm().item()) if activator_vector is not None else 0.0
            helper_spectrum = helper_spectra.get(group, {})
            metrics.update({
                "residual_rms": residual_rms,
                "residual_max_abs": group_stats.get("max_abs"),
                "residual_nonfinite_count": group_stats.get("nonfinite_count"),
                "residual_call_count": group_stats.get("call_count"),
                "normalized_rms": residual_rms / (helper_norm / (runtime.sketch_size ** 0.5) + 1.0e-12) if residual_rms is not None else None,
                "projection_p_j": center_summary.get("helper_subspace_projection", {}).get(group, {}).get("energy_fraction"),
                "magnitude_ratio_m_j": activator_norm / (helper_norm + 1.0e-12),
                "helper_mean_norm": helper_norm,
                "helper_top_energy_fraction": helper_spectrum.get("top_energy_fraction"),
                "helper_effective_rank": helper_spectrum.get("effective_rank"),
                "helper_numerical_rank": helper_spectrum.get("numerical_rank"),
                "helper_count": helper_spectrum.get("helper_count"),
                "finite_difference_selected": group in selected,
            })
            row = by_group[group]
            output.append({
                "run_id": run_id, "probe_case_id": case.probe_case_id, "split": case.split,
                "item_id": case.item_id, "timestep": case.timestep, "noise_seed": case.noise_seed,
                "group_id": group, "block_index": row["block_index"], "family": row["kind"], "kind": row["kind"],
                "reference_type": "calibrated_activator", "candidate_losses": {str(scale): value for scale, value in losses.items()},
                "reference_summary": references,
                "helper_response_spectrum": helper_spectrum,
                "metrics": metrics,
            })
            duration = time.monotonic() - group_started
            if group in selected:
                group_durations.append(duration)
                remaining_selected = len(selected) - selected_completed
                eta = (sum(group_durations) / len(group_durations)) * remaining_selected
                self._progress(
                    f"probe {case_index}/{case_count} finite difference {selected_completed}/{len(selected)} "
                    f"{group} complete in {self._format_duration(duration)}; FD ETA={self._format_duration(eta)}"
                )
        return output

    def _artifact_ref(self, path: Path) -> Dict[str, Any]:
        return {"path": os.path.relpath(path, self.output_root).replace(os.sep, "/"), "sha256": sha256_file(path)}
