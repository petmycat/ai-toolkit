from __future__ import annotations

import importlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence

from jobs.process import BaseExtensionProcess

from .helpers import (
    aggregate_records,
    assert_frozen,
    assert_tap_state_equal,
    atomic_write_json,
    atomic_write_jsonl,
    completed_case_keys,
    compute_layer_metrics,
    condition_specs,
    load_jsonl,
    resume_key,
    select_probe_items,
    sha256_file,
    tap_mask_context,
    tap_state,
)


class Ideogram4TapCausalAblationProcess(BaseExtensionProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.device = self.get_conf("device", getattr(job, "device", "cuda"))
        self.model_definition = self.get_conf("model")
        self.trigger_config = self.get_conf("trigger")
        self.activator_config = self.get_conf("text_activator")
        self.ablation_config = self.get_conf("ablation", required=True)
        self.output_root = Path(self.get_conf("output_root", required=True))
        self.run_label = str(self.get_conf("run_id", self.name))
        self.run_id = self.run_label
        forbidden = sorted(
            key for key in ("optimizer", "scheduler", "train", "steps", "datasets", "network")
            if key in self.raw_process_config
        )
        if forbidden:
            raise ValueError(f"inference-only tap ablation rejects training fields: {forbidden}")
        self.sd = None
        self.text_activator = None

    def run(self):
        super().run()
        torch = importlib.import_module("torch")
        with torch.inference_mode():
            self._run_inference(torch)

    def _run_inference(self, torch):
        self._load_snapshot_architecture()
        self.output_root.mkdir(parents=True, exist_ok=True)
        records_path = self.output_root / self.ablation_config.get("records_filename", "tap_ablation_records.jsonl")
        records = load_jsonl(records_path) if self.ablation_config.get("resume", True) else []
        unique_records = {}
        for record in records:
            key = resume_key(record)
            if key in unique_records:
                raise RuntimeError(f"duplicate resume record key: {key}")
            unique_records[key] = record
        records = list(unique_records.values())

        items, cases, probe_manifest = self._build_probes()
        from toolkit.trigger_binding_artifacts import fingerprint

        self.run_id = fingerprint({
            "label": self.run_label,
            "probe_manifest_hash": probe_manifest["probe_manifest_hash"],
            "checkpoint_steps": [int(step) for step in self.ablation_config["checkpoint_steps"]],
            "checkpoint_paths": self.ablation_config.get("checkpoint_paths", {}),
            "tap_layers": list(self.activator_config["tap_adapters"]["tap_layers"]),
            "evaluator_schema_version": 1,
        })
        records = [record for record in records if record.get("run_id") == self.run_id]
        probe_path = self.output_root / self.ablation_config.get("probe_manifest_filename", "tap_ablation_probe_manifest.json")
        atomic_write_json(probe_path, probe_manifest)
        item_by_id = {item["dataset_relative_item_id"]: item for item in items}

        self.sd = self._load_model()
        self.text_activator = self._build_text_activator(torch)
        self._install_activator()
        self._freeze_all_parameters()
        assert_frozen(self.sd.model)
        assert_frozen(self.sd.text_encoder)
        assert_frozen(self.text_activator)

        tap_layers = tuple(int(layer) for layer in self.activator_config["tap_adapters"]["tap_layers"])
        conditions = condition_specs(tap_layers)
        checkpoint_steps = [int(step) for step in self.ablation_config["checkpoint_steps"]]
        completed = completed_case_keys(records, tap_layers)
        checkpoint_refs = []
        for checkpoint_step in checkpoint_steps:
            checkpoint_refs.append(self._load_checkpoint(checkpoint_step))
            first_pending = True
            for case in cases:
                case_key = (self.run_id, checkpoint_step, case.probe_case_id)
                if case_key in completed:
                    continue
                prepared = self._prepare_case(case, item_by_id[case.item_id], torch)
                predictions = self._predict_conditions(prepared, conditions, torch)
                if first_pending:
                    self._sanity_check(prepared, predictions, torch)
                    first_pending = False
                case_records = []
                for layer in tap_layers:
                    metrics = compute_layer_metrics(
                        predictions["none"],
                        predictions["all"],
                        predictions[f"solo:{layer}"],
                        predictions[f"leave_one_out:{layer}"],
                        prepared["target"],
                    )
                    case_records.append({
                        "run_id": self.run_id,
                        "checkpoint_step": checkpoint_step,
                        "probe_case_id": case.probe_case_id,
                        "split": case.split,
                        "item_id": case.item_id,
                        "timestep": case.timestep,
                        "noise_seed": case.noise_seed,
                        "tap_layer": layer,
                        "metrics": metrics,
                    })
                records.extend(case_records)
                atomic_write_jsonl(records_path, sorted(records, key=resume_key))
                completed.add(case_key)

        aggregates = aggregate_records(records)
        aggregates_path = self.output_root / self.ablation_config.get("aggregates_filename", "tap_ablation_aggregates.jsonl")
        summary_path = self.output_root / self.ablation_config.get("summary_filename", "tap_ablation_summary.json")
        manifest_path = self.output_root / self.ablation_config.get("final_manifest_filename", "tap_ablation_manifest.json")
        atomic_write_jsonl(aggregates_path, aggregates)
        summary = self._summary(records, cases, checkpoint_steps, tap_layers)
        atomic_write_json(summary_path, summary)
        artifacts = {
            "probe_manifest": self._artifact_ref(probe_path),
            "records": self._artifact_ref(records_path),
            "aggregates": self._artifact_ref(aggregates_path),
            "summary": self._artifact_ref(summary_path),
        }
        if len(records) != len(checkpoint_steps) * len(cases) * len(tap_layers):
            raise RuntimeError("tap ablation completion contract failed before final manifest")
        final_manifest = {
            "schema": "ai-toolkit.ideogram4-tap-causal-ablation",
            "schema_version": 1,
            "status": "completed",
            "definitions": {
                "gain_baseline": "tap_none",
                "tap_gain_full_vs_none": "(loss_none-loss_full)/(abs(loss_none)+epsilon)",
                "solo_gain_vs_none": "(loss_none-loss_solo)/(abs(loss_none)+epsilon)",
                "marginal_gain_in_full": "(loss_leave_one_out-loss_full)/(abs(loss_none)+epsilon)",
                "interaction_gain": "marginal_gain_in_full-solo_gain_vs_none",
            },
            "run_id": self.run_id,
            "run_label": self.run_label,
            "checkpoint_steps": checkpoint_steps,
            "tap_layers": list(tap_layers),
            "probe_case_count": len(cases),
            "condition_count": len(conditions),
            "expected_prediction_count": len(checkpoint_steps) * len(cases) * len(conditions),
            "expected_record_count": len(checkpoint_steps) * len(cases) * len(tap_layers),
            "actual_record_count": len(records),
            "actual_prediction_count": len(checkpoint_steps) * len(cases) * len(conditions),
            "checkpoint_artifacts": checkpoint_refs,
            "artifacts": artifacts,
        }
        atomic_write_json(manifest_path, final_manifest)

    def _load_snapshot_architecture(self):
        snapshot_path = self.ablation_config.get("snapshot_yaml")
        if snapshot_path:
            import yaml

            with open(snapshot_path, "r", encoding="utf-8") as handle:
                snapshot = yaml.safe_load(handle)
            candidates = []
            if isinstance(snapshot, Mapping):
                candidates.append(snapshot)
                config_root = snapshot.get("config")
                if isinstance(config_root, Mapping):
                    processes = config_root.get("process", ())
                    if isinstance(processes, Sequence) and not isinstance(processes, (str, bytes)):
                        candidates.extend(value for value in processes if isinstance(value, Mapping))
            source = next(
                (
                    candidate
                    for candidate in candidates
                    if "model" in candidate and isinstance(candidate.get("three_phase_trigger_training"), Mapping)
                ),
                None,
            )
            if source is None:
                raise RuntimeError(f"snapshot YAML lacks model/three_phase_trigger_training: {snapshot_path}")
            phase = source["three_phase_trigger_training"]
            snapshot_model = dict(source["model"])
            snapshot_train = source.get("train")
            if "dtype" not in snapshot_model and isinstance(snapshot_train, Mapping):
                snapshot_model["dtype"] = snapshot_train.get("dtype", "bf16")
            snapshot_trigger = phase.get("trigger")
            snapshot_activator = phase.get("text_activator")
            if not isinstance(snapshot_trigger, Mapping) or not isinstance(snapshot_activator, Mapping):
                raise RuntimeError(f"snapshot YAML lacks trigger/text_activator architecture: {snapshot_path}")
            for name, explicit, loaded in (
                ("model", self.model_definition, snapshot_model),
                ("trigger", self.trigger_config, snapshot_trigger),
                ("text_activator", self.activator_config, snapshot_activator),
            ):
                if explicit is not None:
                    disagreements = {
                        key: (value, loaded.get(key))
                        for key, value in explicit.items()
                        if key not in loaded or loaded.get(key) != value
                    }
                    if disagreements:
                        raise RuntimeError(
                            f"explicit {name} configuration disagrees with snapshot YAML: {disagreements}"
                        )
            self.model_definition = dict(snapshot_model)
            self.trigger_config = dict(snapshot_trigger)
            self.activator_config = dict(snapshot_activator)
        if not all(isinstance(value, Mapping) for value in (self.model_definition, self.trigger_config, self.activator_config)):
            raise RuntimeError("configure snapshot_yaml or explicit model, trigger and text_activator mappings")

    def _load_model(self):
        from toolkit.config_modules import ModelConfig
        from toolkit.util.get_model import get_model_class

        model_config = ModelConfig(**self.model_definition)
        model_class = get_model_class(model_config)
        if getattr(model_class, "arch", None) != "ideogram4":
            raise RuntimeError(f"tap ablation requires Ideogram4 model class, got {model_class}")
        scheduler_factory = getattr(model_class, "get_train_scheduler", None)
        scheduler = scheduler_factory() if callable(scheduler_factory) else None
        sd = model_class(
            device=self.device,
            model_config=model_config,
            dtype=model_config.dtype,
            noise_scheduler=scheduler,
        )
        sd.load_model()
        sd.model.eval()
        sd.text_encoder.eval()
        sd.vae.eval()
        return sd

    def _build_text_activator(self, torch):
        from toolkit.models.ideogram4_trigger_activator import TextActivator
        from toolkit.train_tools import get_torch_dtype

        config = self.activator_config
        if config.get("architecture_mode") != "module_lora" or config["te_adapter"].get("mode") != "module_lora":
            raise RuntimeError("snapshot text activator must use module_lora architecture")
        language_model = self.sd.text_encoder.language_model
        hidden_size = int(language_model.config.hidden_size)
        literal = str(self.trigger_config["literal"])
        self.sd.tokenizer.add_tokens([literal], special_tokens=True)
        atomic_ids = self.sd.tokenizer(literal, add_special_tokens=False)["input_ids"]
        if len(atomic_ids) != 1:
            raise RuntimeError(f"trigger literal is not atomic after registration: {atomic_ids}")
        init_ids = self.sd.tokenizer(config["embedding"]["init_words"], add_special_tokens=False)["input_ids"]
        embedding_table = language_model.embed_tokens
        safe_ids = [token_id for token_id in init_ids if token_id < embedding_table.num_embeddings]
        if not safe_ids:
            raise RuntimeError("text activator initializer has no valid token IDs")
        initializer_ids = torch.tensor(safe_ids, device=embedding_table.weight.device, dtype=torch.long)
        initializer = embedding_table(initializer_ids).float().mean(dim=0, keepdim=True)
        te, taps = config["te_adapter"], config["tap_adapters"]
        amplification = config["amplification"]
        activator = TextActivator(
            embedding_dim=hidden_size,
            hidden_size=hidden_size,
            embedding_tokens=int(config["embedding"]["tokens"]),
            initializer=initializer,
            te_adapter_mode="module_lora",
            te_rank=int(te["rank"]),
            te_alpha=float(te["alpha"]),
            te_dropout=float(te["dropout"]),
            te_target_modules=te.get("target_modules", te.get("child_modules", ())),
            te_parent_modules=te.get("parent_modules", ()),
            te_layers=te.get("layers", "all"),
            tap_layers=taps["tap_layers"],
            tap_rank=int(taps["rank"]),
            tap_alpha=float(taps["alpha"]),
            tap_dropout=float(taps["dropout"]),
            tap_learnable_scale=bool(taps.get("learnable_scale", False)),
            tap_scale_init=float(taps.get("scale_init", 1.0)),
            per_tap=taps.get("per_tap", {}),
            gamma_init=float(amplification["initial"]),
            gamma_min=float(amplification["minimum"]),
            gamma_max=float(amplification["maximum"]),
            gamma_trainable=False,
        ).to(self.sd.device_torch, dtype=get_torch_dtype(config["embedding"]["dtype"]))
        activator.atomic_token_id = int(atomic_ids[0])
        activator.lookup_token_id = int(safe_ids[0])
        activator.install_module_lora(language_model)
        return activator

    def _install_activator(self):
        installer = getattr(self.sd, "install_text_activator", None) or getattr(self.sd, "set_text_activator", None)
        if not callable(installer):
            raise RuntimeError("Ideogram4 model does not expose text activator installation")
        installer(self.text_activator, runtime_mode="full")

    def _freeze_all_parameters(self):
        for module in (self.sd.model, self.sd.text_encoder, self.sd.vae, self.text_activator):
            module.requires_grad_(False)
            module.eval()

    def _checkpoint_dir(self, step: int) -> Path:
        overrides = self.ablation_config.get("checkpoint_paths", {})
        configured = overrides.get(step, overrides.get(str(step))) if isinstance(overrides, Mapping) else None
        if configured:
            path = Path(str(configured))
            return path if path.is_absolute() else Path(self.ablation_config["snapshot_root"]) / path
        pattern = str(self.ablation_config.get("checkpoint_dir_pattern", "phase_a1/checkpoints/{step}"))
        return Path(self.ablation_config["snapshot_root"]) / pattern.format(step=step)

    def _load_checkpoint(self, step: int) -> Dict[str, Any]:
        from toolkit.trigger_binding_artifacts import load_artifact, load_checkpoint_manifest

        checkpoint_dir = self._checkpoint_dir(step)
        manifest_filename = self.ablation_config.get("manifest_filename")
        allow_legacy = bool(self.ablation_config.get("allow_legacy_raw_safetensors", False))
        artifact_filenames = self.ablation_config.get("artifact_filenames", {})
        artifact_keys = self.ablation_config.get("artifact_manifest_keys", {
            "embedding": "embedding",
            "te_adapter": "te_adapter",
            "tap_adapter": "tap_adapter",
        })
        component_modules = {
            "embedding": self.text_activator.embedding,
            "te_adapter": self.text_activator.te_adapter_artifact_module(),
            "tap_adapter": self.text_activator.tap_adapters,
        }
        loaded_refs = {}
        if manifest_filename:
            manifest_path = checkpoint_dir / str(manifest_filename)
            manifest = load_checkpoint_manifest(manifest_path, verify_artifacts=True)
            if int(manifest["step"]) != step:
                raise RuntimeError(f"checkpoint manifest step mismatch at {manifest_path}")
            for artifact_type, module in component_modules.items():
                key = artifact_keys[artifact_type]
                reference = manifest["artifacts"].get(key)
                if reference is None:
                    raise RuntimeError(f"checkpoint manifest lacks required artifact {key!r}")
                path = Path(reference["path"])
                if not path.is_absolute():
                    path = manifest_path.parent / path
                state, artifact_manifest = load_artifact(
                    path,
                    expected_type=artifact_type,
                    expected_phase_fingerprint=manifest["phase_fingerprint"],
                    expected_source_fingerprint=manifest["source_fingerprint"],
                    expected_config_fingerprint=manifest["config_fingerprint"],
                    expected_file_sha256=reference["sha256"],
                )
                module.load_state_dict(state, strict=True)
                loaded_refs[artifact_type] = {
                    "path": str(path),
                    "sha256": reference["sha256"],
                    "artifact_manifest_sha256": reference["manifest_sha256"],
                    "artifact_schema": artifact_manifest["artifact_schema"],
                }
            return {"step": step, "manifest": str(manifest_path), "artifacts": loaded_refs}
        if not artifact_filenames:
            raise RuntimeError("fail-closed: configure manifest_filename or explicit artifact_filenames")
        for artifact_type, module in component_modules.items():
            filename = artifact_filenames.get(artifact_type)
            if not filename:
                raise RuntimeError(f"fail-closed: missing explicit filename for {artifact_type}")
            path = checkpoint_dir / filename
            try:
                state, artifact_manifest = load_artifact(path, expected_type=artifact_type)
            except Exception:
                if not allow_legacy:
                    raise
                from safetensors.torch import load_file
                state, artifact_manifest = load_file(str(path)), None
            module.load_state_dict(state, strict=True)
            loaded_refs[artifact_type] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "artifact_schema": None if artifact_manifest is None else artifact_manifest["artifact_schema"],
            }
        return {"step": step, "manifest": None, "artifacts": loaded_refs}

    def _build_probes(self):
        from toolkit.semantic_scaffold_calibration import build_fixed_probe_cases
        from toolkit.trigger_binding_artifacts import fingerprint
        from toolkit.trigger_data_split import load_data_split_manifest

        dataset_root = self.ablation_config["dataset_root"]
        manifest = load_data_split_manifest(
            self.ablation_config["split_manifest"],
            expected_seed=int(self.ablation_config.get("seed", 42)),
        )
        items = select_probe_items(
            dataset_root,
            manifest.as_dict(),
            train_limit=int(self.ablation_config.get("train_probe_count", 8)),
            expected_heldout=int(self.ablation_config.get("expected_heldout_count", 4)),
            placeholder=str(self.trigger_config.get("placeholder", "[trigger]")),
        )
        cases = build_fixed_probe_cases(
            items,
            {
                "train_item_ids": [item["dataset_relative_item_id"] for item in items if item["split"] == "train"],
                "heldout_item_ids": [item["dataset_relative_item_id"] for item in items if item["split"] == "heldout"],
            },
            noise_seeds=[int(self.ablation_config.get("seed", 42))],
            fixed_timesteps=[int(value) for value in self.ablation_config["timesteps"]],
            probe_scope="split",
            target_mode="flow",
        )
        payload = {
            "schema": "ai-toolkit.ideogram4-tap-causal-ablation-probes",
            "schema_version": 1,
            "split_manifest": self.ablation_config["split_manifest"],
            "dataset_root": dataset_root,
            "items": items,
            "cases": [case.as_dict() for case in cases],
        }
        payload["probe_manifest_hash"] = fingerprint(payload)
        expected_cases = (int(self.ablation_config.get("train_probe_count", 8)) + int(self.ablation_config.get("expected_heldout_count", 4))) * len(self.ablation_config["timesteps"])
        if len(cases) != expected_cases:
            raise RuntimeError(f"probe case count mismatch: {len(cases)} != {expected_cases}")
        return items, cases, payload

    def _prepare_case(self, case, item: Mapping[str, Any], torch):
        from PIL import Image
        from torchvision import transforms
        from toolkit.train_tools import get_torch_dtype

        with Image.open(item["image_path"]) as image:
            image = image.convert("RGB")
            width, height = max(16, image.width // 16 * 16), max(16, image.height // 16 * 16)
            tensor = transforms.ToTensor()(image.resize((width, height), Image.Resampling.BICUBIC)) * 2.0 - 1.0
        dtype = get_torch_dtype(self.model_definition.get("dtype", "bf16"))
        latent = self.sd.encode_images([tensor], device=self.sd.device_torch, dtype=dtype)
        generator = torch.Generator(device=latent.device).manual_seed(int(case.noise_seed))
        noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)
        schedule = max(0.0, min(1.0, float(case.timestep) / 1000.0))
        noisy = latent * (1.0 - schedule) + noise * schedule
        return {
            "latent": latent,
            "noise": noise,
            "noisy_latents": noisy,
            "timesteps": torch.tensor([float(case.timestep)], device=latent.device, dtype=latent.dtype),
            "target": noise - latent,
            "prompt": item["caption"],
        }

    def _encode_bound_prompt(self, prompt: str, runtime_mode: str):
        from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
        from toolkit.trigger_binding import bind_trigger_batch

        placeholder = str(self.trigger_config.get("placeholder", "[trigger]"))
        if placeholder not in prompt:
            raise RuntimeError(f"bound prompt lacks required placeholder {placeholder!r}")
        batch = bind_trigger_batch(
            self.sd.tokenizer,
            [prompt],
            str(self.trigger_config["literal"]),
            placeholder=placeholder,
            max_length=getattr(self.sd, "max_text_length", None),
            require_placeholder=True,
            mask_all_occurrences=bool(self.trigger_config.get("mask_all_occurrences", True)),
            require_atomic=True,
            expected_token_id=self.text_activator.atomic_token_id,
            virtual_tokens=self.text_activator.virtual_tokens,
        ).to(self.sd.text_encoder.device)
        pipeline = importlib.import_module("extensions_built_in.diffusion_models.ideogram4.src.pipeline")
        features = pipeline.get_qwen3_vl_features(
            self.sd.text_encoder,
            batch.input_ids,
            batch.attention_mask,
            (batch.attention_mask.cumsum(dim=-1) - 1).clamp(min=0).long(),
            trigger_mask=batch.trigger_mask,
            token_indices=getattr(batch, "token_indices", None),
            text_activator=self.text_activator,
            runtime_mode=runtime_mode,
            runtime_metadata=batch.runtime_metadata() if hasattr(batch, "runtime_metadata") else None,
        )
        embeds = AdvancedPromptEmbeds(
            text_embeds=[features[0].to(self.sd.torch_dtype)],
            trigger_masks=[batch.trigger_mask[0, :int(batch.attention_mask[0].sum().item())]],
        )
        embeds.trigger_runtime_metadata = batch.runtime_metadata() if hasattr(batch, "runtime_metadata") else None
        return embeds

    def _predict(self, prepared: Mapping[str, Any], runtime_mode: str):
        embeds = self._encode_bound_prompt(prepared["prompt"], runtime_mode)
        batch = SimpleNamespace(latents=prepared["latent"], file_items=[], audio_pred_slot=None)
        return self.sd.predict_noise(
            latents=prepared["noisy_latents"],
            timestep=prepared["timesteps"],
            conditional_embeddings=embeds.to(self.sd.device_torch, dtype=prepared["noisy_latents"].dtype),
            unconditional_embeddings=None,
            guidance_scale=1.0,
            guidance_embedding_scale=1.0,
            bypass_guidance_embedding=False,
            batch=batch,
        ).detach().clone()

    def _predict_conditions(self, prepared: Mapping[str, Any], conditions: Sequence[Mapping[str, Any]], torch):
        predictions = {}
        base_state = tap_state(self.text_activator)
        for condition in conditions:
            with tap_mask_context(self.text_activator, condition["active_layers"]):
                predictions[condition["name"]] = self._predict(prepared, "full")
            assert_tap_state_equal(tap_state(self.text_activator), base_state)
        return predictions

    def _sanity_check(self, prepared: Mapping[str, Any], predictions: Mapping[str, Any], torch):
        with tap_mask_context(self.text_activator, ()): 
            semantic_only = self._predict(prepared, "semantic_only")
        if not torch.allclose(predictions["none"], semantic_only, rtol=1.0e-5, atol=1.0e-6):
            raise RuntimeError("sanity failed: none-mask full does not match semantic_only")
        all_layers = self.activator_config["tap_adapters"]["tap_layers"]
        with tap_mask_context(self.text_activator, all_layers):
            repeated_all = self._predict(prepared, "full")
        if not torch.equal(predictions["all"], repeated_all):
            raise RuntimeError("sanity failed: repeated all-tap prediction is not deterministic")
        assert_frozen(self.sd.model)
        assert_frozen(self.sd.text_encoder)
        assert_frozen(self.text_activator)

    def _summary(self, records, cases, checkpoint_steps, tap_layers):
        expected_records = len(cases) * len(checkpoint_steps) * len(tap_layers)
        if len(records) != expected_records:
            raise RuntimeError(f"record count mismatch: {len(records)} != {expected_records}")
        splits = {}
        for split in ("train", "heldout"):
            split_records = [record for record in records if record["split"] == split]
            splits[split] = {"record_count": len(split_records), "case_count": len({record["probe_case_id"] for record in split_records})}
        return {
            "run_id": self.run_id,
            "checkpoint_count": len(checkpoint_steps),
            "probe_case_count": len(cases),
            "tap_layer_count": len(tap_layers),
            "record_count": len(records),
            "expected_prediction_count": len(cases) * len(checkpoint_steps) * 28,
            "expected_record_count": expected_records,
            "splits": splits,
        }

    def _artifact_ref(self, path: Path) -> Dict[str, Any]:
        return {"path": os.path.relpath(path, self.output_root).replace(os.sep, "/"), "sha256": sha256_file(path)}
