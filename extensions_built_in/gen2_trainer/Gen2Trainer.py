from __future__ import annotations

import copy
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import ConcatDataset

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.progress_bar import ToolkitProgressBar

from .activator import (
    PlaceholderContract,
    QwenModuleLoRABank,
    SoftTokenBank,
    adapter_enabled_context,
    encode_qwen_inputs_embeds,
    install_qwen_module_lora,
    normalize_inline_helpers,
    replace_token_spans_with_soft_tokens,
    resample_embedding_sequence,
)
from .artifacts import load_tensor_artifact, save_tensor_artifact
from .checkpoint import load_phase_checkpoint, save_phase_checkpoint
from .config import Gen2RuntimeConfig, validate_gen2_config
from .controller import AdaptiveJointController, isolated_gradient_cosine
from .probes import (
    deterministic_probe_noise,
    deterministic_probe_noise_seed,
    deterministic_probe_timestep,
    load_fixed_probes,
    probe_fingerprint,
    save_fixed_probes,
    select_probe_pairs,
    validate_regions,
)
from .split import _item_pair_hash, ensure_split, make_dataset_view, make_train_dataset_view
from toolkit.data_loader import DataLoader, dto_collation, get_dataloader_datasets
from toolkit.data_loader import get_dataloader_from_datasets
from toolkit.optimizer import get_optimizer
from toolkit.scheduler import get_lr_scheduler
from .registry import AdapterRuntimeContext, enable_adapter_training, install_ideogram_adapters
from .sampling import build_validation_matrix, make_flowmatch_noisy_latents, sample_stratified_timesteps
from .temporal_rank_field import time_smooth_regularizer, time_mean_diagnostic
from .phase_a_math import (
    dataset_mse,
    detached_ema_prototype,
    helper_subset_direction_loss,
    prototype_consistency_loss,
    spatial_response,
    timestep_region_masks,
)


class Gen2Trainer(BaseSDTrainProcess):
    """Two-phase Ideogram 4 style trainer with Gen2-native artifacts."""

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs: Any) -> None:
        self.gen2_config: Gen2RuntimeConfig = validate_gen2_config(config)
        super().__init__(process_id, job, config, **kwargs)
        self.placeholder_contract = PlaceholderContract(self.gen2_config.placeholder)
        self.gen2_root = Path(self.save_root) / "gen2"
        self.phase_a_root = self.gen2_root / "phase_a"
        self.phase_b_root = self.gen2_root / "phase_b"
        self.soft_tokens: SoftTokenBank | None = None
        self.frozen_e0: torch.Tensor | None = None
        self.qwen_module_lora: QwenModuleLoRABank | None = None
        self._phase_a_prototype: dict[str, torch.Tensor] | None = None
        self._gen2_split = None
        self._last_phase_a_diagnostics: dict[str, float] = {}
        self._phase_a_history: list[dict[str, Any]] = []
        self._phase_b_history: list[dict[str, Any]] = []
        self._phase_a_probes: list[dict[str, Any]] = []
        self._phase_a_heldout_probes: list[dict[str, Any]] = []
        self._phase_b_probes: list[dict[str, Any]] = []
        self._gen2_fixed_probe_metadata: dict[str, Any] = {}
        self.official_unconditional = None
        self.helpers = normalize_inline_helpers(self.gen2_config.helpers)
        self._adapter_bank = None
        self._timestep_bin_orders: dict[str, torch.Tensor] = {}
        self._timestep_bin_cursors: dict[str, int] = {"a": 0, "b": 0}
        self._validation_prompts: tuple[str, ...] = self.gen2_config.validation_prompts
        self._designated_helper_id = str((self.config.get("sample") or {}).get("designated_helper_id", ""))

    def _assert_qwen_frozen(self) -> None:
        allowed = set()
        if self.qwen_module_lora is not None:
            allowed.update(id(parameter) for parameter in self.qwen_module_lora.parameters())
        trainable = [name for name, parameter in self.sd.text_encoder.named_parameters() if parameter.requires_grad and id(parameter) not in allowed]
        if trainable:
            raise RuntimeError(f"Gen2 requires frozen Qwen parameters, but these are trainable: {trainable[:8]}")
    def _rebuild_dataloader_for_phase(self, phase: str) -> None:
        settings = self.gen2_config.phase_a if phase == "a" else self.gen2_config.phase_b
        batch_size = int(settings.get("batch_size", self.train_config.batch_size))
        if batch_size < 1:
            raise ValueError(f"phase_{phase}.batch_size must be positive")
        if self._gen2_split is not None and hasattr(self, "_gen2_train_datasets"):
            self.data_loader = self._make_gen2_loader(self._gen2_train_datasets, batch_size)
        elif self.data_loader is not None:
            self.data_loader = get_dataloader_from_datasets(get_dataloader_datasets(self.data_loader), batch_size, self.sd)
        if self.datasets_reg is not None:
            raise ValueError("Gen2 forbids datasets_reg and regularization loaders")
        self.train_config.batch_size = batch_size
        if self.accelerator.is_main_process:
            print(f"Gen2 Phase {phase.upper()} dataloader batch_size={batch_size}")

    def _phase_steps(self, phase: str) -> int:
        settings = self.gen2_config.phase_a if phase == "a" else self.gen2_config.phase_b
        if not settings.get("enabled", True):
            return 0
        return int(settings.get("steps", 0))

    def _write_manifest(self, status: str, completed_phase: str | None = None, error: str | None = None) -> None:
        self.gen2_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "trainer": "gen2_trainer",
            "status": status,
            "completed_phase": completed_phase,
            "mode": self.gen2_config.mode,
            "placeholder": self.gen2_config.placeholder,
            "official_unconditional_required": True,
            "training_scope": "conditional_only",
            "artifacts": {
                "activator": "phase_a/activator.safetensors",
                "style_adapter": "phase_b/adapter.safetensors",
            },
        }
        if error is not None:
            manifest["error"] = error
        (self.gen2_root / "gen2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def hook_after_model_load(self):
        if self.sd.arch != "ideogram4":
            raise ValueError("Gen2 only supports the Ideogram 4 model")
        self.sd.model.requires_grad_(False)
        self.sd.text_encoder.requires_grad_(False)
        self.sd.vae.requires_grad_(False)
        qwen_config = self.gen2_config.qwen.get("per_layer_adapter") or {}
        self.qwen_module_lora = install_qwen_module_lora(
            self.sd.text_encoder,
            rank=int(qwen_config.get("rank", 4)),
            alpha=float(qwen_config.get("alpha", 4.0)),
            enabled=bool(qwen_config.get("enabled", True)),
            layers=qwen_config.get("layers", "all"),
        )
        if self.qwen_module_lora is not None:
            self.qwen_module_lora.to(self.device_torch)
        self._assert_qwen_frozen()
        network = self.config.get("network") or {}
        temporal = network.get("temporal") or {}
        self._adapter_bank = install_ideogram_adapters(
            self.sd.model,
            rank=int(network.get("rank", 32)),
            alpha=float(network.get("alpha", network.get("rank", 32))),
            knots=int(temporal.get("knots", 8)),
            delta_max=float(temporal.get("delta_max", 1.0)),
        )
        self.sd.transformer = self.sd.model
        self.soft_tokens.to(self.device_torch)
        self.modules_being_trained = [self.soft_tokens]
        if self.qwen_module_lora is not None:
            self.modules_being_trained.append(self.qwen_module_lora)

    def hook_before_train_loop(self):
        self._preflight_loaded_datasets()
        # The immutable split must exist before any phase loader is rebuilt.
        self._prepare_gen2_split()
        if self.accelerator.is_main_process:
            self.logger.start()
        self.prepare_accelerator()

    def prepare_accelerator(self):
        self.accelerator.even_batches = False
        self.sd.vae = self.accelerator.prepare(self.sd.vae)
        self.sd.unet = self.accelerator.prepare(self.sd.unet)
        modules = [self.soft_tokens]
        if self.qwen_module_lora is not None:
            modules.append(self.qwen_module_lora)
        prepared = self.accelerator.prepare(*modules, self.optimizer)
        index = 0
        self.soft_tokens = prepared[index]
        index += 1
        if self.qwen_module_lora is not None:
            self.qwen_module_lora = prepared[index]
            index += 1
        self.optimizer = prepared[index]
        if self.lr_scheduler is not None:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
        self.modules_being_trained = [self.soft_tokens]
        if self.qwen_module_lora is not None:
            self.modules_being_trained.append(self.qwen_module_lora)
        self._assert_qwen_frozen()

    def hook_add_extra_train_params(self, params):
        params.clear()
        phase_a = self.gen2_config.phase_a
        self.train_config.optimizer = phase_a.get("optimizer", self.train_config.optimizer)
        self.train_config.optimizer_params = dict(phase_a.get("optimizer_params", self.train_config.optimizer_params))
        self.train_config.lr = float(phase_a.get("lr", self.train_config.lr))
        self.train_config.lr_scheduler = phase_a.get("lr_scheduler", self.train_config.lr_scheduler)
        self.train_config.lr_scheduler_params = dict(phase_a.get("lr_scheduler_params", self.train_config.lr_scheduler_params))
        params.append({
            "params": [self.soft_tokens.A],
            "lr": self.train_config.lr,
        })
        if self.qwen_module_lora is not None:
            qwen_config = self.gen2_config.qwen.get("per_layer_adapter") or {}
            params.append({
                "params": list(self.qwen_module_lora.parameters()),
                "lr": float(qwen_config["lr"]),
                "weight_decay": float(qwen_config.get("weight_decay", 0.0)),
            })
        return params

    def before_dataset_load(self):
        return None

    def _preflight_loaded_datasets(self) -> None:
        missing = []
        datasets = [] if self.data_loader is None else [self.data_loader.dataset]
        for dataset in datasets:
            for item in getattr(dataset, "file_list", []):
                try:
                    item.load_caption(getattr(dataset, "caption_dict", None))
                    self.preflight_caption(item.raw_caption, item.path)
                except Exception as error:
                    missing.append(f"{getattr(item, 'path', '<unknown>')}: {error}")
        for index, prompt in enumerate(self.gen2_config.validation_prompts):
            try:
                self.preflight_caption(prompt, f"validation_prompts[{index}]")
            except Exception as error:
                missing.append(str(error))
        if missing:
            raise ValueError("Gen2 placeholder preflight failed:\n" + "\n".join(missing))

    def _next_batch(self, iterator, dataloader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(dataloader)
            return next(iterator), iterator

    def _prepare_real_batch(self, batch, phase: str):
        noisy_latents, noise, _, _, imgs = self.process_general_training_batch(batch)
        raw_prompts = [item.raw_caption for item in batch.file_items]
        timesteps = self._sample_gen2_timesteps(noisy_latents.shape[0], phase, noisy_latents.device)
        noisy_latents = self._make_flowmatch_noisy_latents(batch.latents, noise, timesteps)
        return noisy_latents, noise, timesteps, raw_prompts, imgs

    def _sample_gen2_timesteps(self, batch_size: int, phase: str, device: torch.device) -> torch.Tensor:
        settings = self.gen2_config.phase_a if phase == "a" else self.gen2_config.phase_b
        bins = int(settings.get("timestep_bins", 10))
        if settings.get("timestep_sampling", "stratified_uniform") != "stratified_uniform":
            raise ValueError(f"unsupported {phase} timestep sampling")
        phase_key = phase.lower()
        timesteps, order, cursor = sample_stratified_timesteps(
            batch_size,
            bins,
            self._timestep_bin_orders.get(phase_key),
            self._timestep_bin_cursors.get(phase_key, 0),
            device,
        )
        self._timestep_bin_orders[phase_key] = order
        self._timestep_bin_cursors[phase_key] = cursor
        return timesteps

    def _make_flowmatch_noisy_latents(self, clean: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return make_flowmatch_noisy_latents(clean, noise, timestep)

    def _encode_soft_prompt(self, raw_caption: str, diagnostics: dict[str, Any] | None = None, return_pooled: bool = False, token_bank: SoftTokenBank | None = None, adapter_enabled: bool = True):
        from extensions_built_in.diffusion_models.ideogram4.src.pipeline import QWEN3_VL_ACTIVATION_LAYERS
        from toolkit.ideogram_caption import digest_caption_string

        self.preflight_caption(raw_caption)
        normalized = digest_caption_string(raw_caption)
        messages = [{"role": "user", "content": [{"type": "text", "text": normalized}]}]
        text = self.sd.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        encoded = self.sd.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.sd.max_text_length,
        )
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            raise RuntimeError("Qwen tokenizer must support offset_mapping for Gen2 soft tokens")
        spans = []
        cursor = 0
        while True:
            start = text.find("[trigger]", cursor)
            if start < 0:
                break
            end = start + len("[trigger]")
            token_indices = [index for index, (left, right) in enumerate(offsets) if right > start and left < end]
            if not token_indices:
                raise RuntimeError("failed to map [trigger] to Qwen token span")
            spans.append((token_indices[0], token_indices[-1] + 1))
            cursor = end
        token_ids = torch.tensor(encoded["input_ids"], device=self.sd.text_encoder.device, dtype=torch.long)
        ordinary = self.sd.text_encoder.language_model.embed_tokens(token_ids)
        active_bank = token_bank or self.soft_tokens
        if active_bank is None:
            raise RuntimeError("Gen2 soft-token bank is not initialized")
        expanded, expanded_spans = replace_token_spans_with_soft_tokens(ordinary, spans, active_bank)
        activator_mask = torch.zeros(expanded.shape[0], device=expanded.device, dtype=torch.long)
        for start, end in expanded_spans:
            activator_mask[start:end] = 1
        if diagnostics is not None:
            ordinary_norm = ordinary.float().norm(dim=-1)
            soft_norm = active_bank.A.float().norm(dim=-1)
            diagnostics.update(
                {
                    "raw_token_length": int(ordinary.shape[0]),
                    "expanded_token_length": int(expanded.shape[0]),
                    "placeholder_spans": [list(span) for span in spans],
                    "expanded_spans": [list(span) for span in expanded_spans],
                    "attention_mask_sum": int(expanded.shape[0]),
                    "position_id_min": 0,
                    "position_id_max": int(expanded.shape[0] - 1),
                    "ordinary_embedding_norm_mean": float(ordinary_norm.mean().item()),
                    "ordinary_embedding_norm_max": float(ordinary_norm.max().item()),
                    "soft_token_norm_mean": float(soft_norm.mean().item()),
                    "soft_token_norm_max": float(soft_norm.max().item()),
                }
            )
        expanded = expanded.unsqueeze(0)
        attention_mask = torch.ones(expanded.shape[:2], device=expanded.device, dtype=torch.long)
        pos_2d = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        with adapter_enabled_context(adapter_enabled):
            encoded = encode_qwen_inputs_embeds(
                self.sd.text_encoder,
                expanded,
                attention_mask,
                pos_2d,
                QWEN3_VL_ACTIVATION_LAYERS,
                activator_mask=activator_mask.unsqueeze(0),
                return_details=diagnostics is not None or return_pooled,
                gradient_checkpointing=bool(self.gen2_config.phase_a.get("gradient_checkpointing", self.config.get("train", {}).get("gradient_checkpointing", False))),
            )
        if diagnostics is not None or return_pooled:
            features, pooled, ordinary_pooled = encoded
            if diagnostics is not None:
                diagnostics["activator_pooled_norm"] = float(pooled.float().norm().item())
                diagnostics["activator_mask_count"] = int(activator_mask.sum().item())
            if return_pooled:
                return features[0].to(self.sd.torch_dtype), pooled[0].to(self.sd.torch_dtype), ordinary_pooled[0].to(self.sd.torch_dtype)
            return features[0].to(self.sd.torch_dtype)
        return encoded[0].to(self.sd.torch_dtype)

    def _encode_frozen_e0_prompt(self, raw_caption: str, diagnostics: dict[str, Any] | None = None):
        """Encode E0 with identical expansion geometry and all adapters disabled."""
        if self.frozen_e0 is None:
            raise RuntimeError("immutable E0 is not initialized")
        bank = SoftTokenBank(self.soft_tokens.tokens, self.soft_tokens.dimension, self.frozen_e0.detach())
        with torch.no_grad():
            return self._encode_soft_prompt(
                raw_caption,
                diagnostics=diagnostics,
                token_bank=bank,
                adapter_enabled=False,
            )

    def _qwen_adapter_tensors(self) -> dict[str, torch.Tensor]:
        if self.qwen_module_lora is None:
            return {}
        return {f"qwen_adapter.{key}": value.detach().cpu() for key, value in self.qwen_module_lora.state_dict().items()}

    def _qwen_adapter_metadata(self) -> dict[str, Any]:
        config = self.gen2_config.qwen.get("per_layer_adapter") or {}
        return {
            "qwen_per_layer_adapter": self.qwen_module_lora is not None,
            "qwen_per_layer_adapter_layers": list(self.qwen_module_lora.layer_indices) if self.qwen_module_lora is not None else [],
            "qwen_per_layer_adapter_rank": int(config.get("rank", 4)),
            "qwen_per_layer_adapter_alpha": float(config.get("alpha", 4.0)),
        }

    def _load_qwen_adapter_state(self, tensors: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
        expected = self.qwen_module_lora is not None
        if bool(metadata.get("qwen_per_layer_adapter", False)) != expected:
            raise ValueError("artifact/checkpoint Qwen per-layer adapter setting does not match current configuration")
        state = {key.removeprefix("qwen_adapter."): value for key, value in tensors.items() if key.startswith("qwen_adapter.")}
        if expected:
            if not state:
                raise ValueError("artifact/checkpoint is missing Qwen per-layer adapter state")
            expected_layers = list(self.qwen_module_lora.layer_indices)
            if list(metadata.get("qwen_per_layer_adapter_layers", [])) != expected_layers:
                raise ValueError("artifact/checkpoint Qwen adapter layer metadata does not match")
            config = self.gen2_config.qwen.get("per_layer_adapter") or {}
            if int(metadata.get("qwen_per_layer_adapter_rank", -1)) != int(config.get("rank", 4)):
                raise ValueError("artifact/checkpoint Qwen adapter rank metadata does not match")
            if float(metadata.get("qwen_per_layer_adapter_alpha", -1.0)) != float(config.get("alpha", 4.0)):
                raise ValueError("artifact/checkpoint Qwen adapter alpha metadata does not match")
            self.qwen_module_lora.load_state_dict(state, strict=True)
        elif state:
            raise ValueError("artifact/checkpoint contains Qwen adapter state but adapter is disabled")

    def _phase_checkpoint(self, phase: str, step: int, tensors: dict[str, torch.Tensor], metadata: dict[str, Any], optimizer, scheduler=None) -> None:
        root = self.phase_a_root if phase == "a" else self.phase_b_root
        tensors = dict(tensors)
        tensors.update(self._qwen_adapter_tensors())
        metadata = dict(metadata)
        metadata.update(self._qwen_adapter_metadata())
        path = root / f"step_{step:06d}"
        phase_key = phase.lower()
        order = self._timestep_bin_orders.get(phase_key)
        metadata = dict(metadata)
        metadata["timestep_sampler"] = {
            "order": order.detach().cpu().tolist() if order is not None else None,
            "cursor": int(self._timestep_bin_cursors.get(phase_key, 0)),
        }
        save_phase_checkpoint(path, tensors, metadata, optimizer=optimizer, scheduler=scheduler)
        phase_save = ((self.config.get("save") or {}).get(f"phase_{phase}") or {})
        keep = int(phase_save.get("max_step_saves_to_keep", 0))
        if keep > 0:
            checkpoints = sorted(
                (item for item in root.iterdir() if item.is_dir() and item.name.startswith("step_")),
                key=lambda item: int(item.name.removeprefix("step_")),
            )
            for old_checkpoint in checkpoints[:-keep]:
                shutil.rmtree(old_checkpoint)

    def _prompt_prediction(self, prompts, noisy_latents, timesteps, target=None, adapter_context=None):
        from extensions_built_in.gen2_trainer.registry import clear_adapter_context

        clear_adapter_context(self.sd.transformer)
        embeds = self.sd.get_prompt_embeds(prompts)
        prediction = self.sd.get_noise_prediction(noisy_latents, timesteps, embeds, adapter_context=adapter_context)
        clear_adapter_context(self.sd.transformer)
        if target is None:
            return prediction
        return torch.nn.functional.mse_loss(prediction.float(), target.float(), reduction="none").flatten(1).mean(1)

    @staticmethod
    def _advanced_prompt_embeds(features):
        from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
        return AdvancedPromptEmbeds(text_embeds=features)

    def _literal_prompt_embeds(self, prompt: str):
        if self.frozen_e0 is None:
            raise RuntimeError("literal E0 is unavailable")
        return self._advanced_prompt_embeds([self._encode_frozen_e0_prompt(prompt).detach()])

    def _frozen_e0_prediction(self, prompts, noisy_latents, timesteps, target=None):
        """Predict with the same expanded geometry using immutable E0 and no Qwen adapter."""
        from extensions_built_in.gen2_trainer.registry import clear_adapter_context
        clear_adapter_context(self.sd.transformer)
        features = [self._encode_frozen_e0_prompt(prompt) for prompt in prompts]
        embeds = self._advanced_prompt_embeds(features)
        prediction = self.sd.get_noise_prediction(noisy_latents, timesteps, embeds)
        clear_adapter_context(self.sd.transformer)
        if target is None:
            return prediction
        return torch.nn.functional.mse_loss(prediction.float(), target.float(), reduction="none").flatten(1).mean(1)

    def _activator_prediction(self, prompts, noisy_latents, timesteps, target=None):
        from extensions_built_in.gen2_trainer.registry import clear_adapter_context

        clear_adapter_context(self.sd.transformer)
        features = [self._encode_soft_prompt(prompt) for prompt in prompts]
        embeds = self._advanced_prompt_embeds(features)
        prediction = self.sd.get_noise_prediction(noisy_latents, timesteps, embeds)
        clear_adapter_context(self.sd.transformer)
        if target is None:
            return prediction
        return torch.nn.functional.mse_loss(prediction.float(), target.float(), reduction="none").flatten(1).mean(1)

    def _collect_phase_a_probes(self, count: int, iterator):
        probes = []
        while len(probes) < count:
            batch, iterator = self._next_batch(iterator, self.data_loader)
            noisy, noise, timesteps, prompts, _ = self._prepare_real_batch(batch, "a")
            target = self.sd.get_loss_target(noise=noise, batch=batch)
            for index, prompt in enumerate(prompts):
                probes.append((noisy[index:index + 1].detach(), timesteps[index:index + 1].detach(), target[index:index + 1].detach(), prompt))
                if len(probes) >= count:
                    break
            if hasattr(batch, "cleanup"):
                batch.cleanup()
        return probes, iterator


    def _prepare_validation_prompts(self) -> None:
        if self._gen2_split is None or not hasattr(self, "_gen2_dataset"):
            raise RuntimeError("Gen2 validation prompts require an active dataset split")
        heldout_ids = set(self._gen2_split.heldout_pair_sha256)
        heldout_prompts: list[str] = []
        for item in getattr(self._gen2_dataset, "file_list", []):
            if _item_pair_hash(item, self._gen2_dataset) in heldout_ids:
                prompt = str(item.raw_caption)
                if prompt in heldout_prompts:
                    raise ValueError("heldout validation prompts contain duplicates")
                heldout_prompts.append(prompt)
        merged = heldout_prompts + list(self.gen2_config.validation_prompts)
        if len(set(merged)) != len(merged):
            raise ValueError("heldout and YAML validation prompts must not overlap")
        self._validation_prompts = tuple(merged)

    def _prepare_gen2_split(self) -> None:
        if self._gen2_split is not None and hasattr(self, "_gen2_source_datasets") and hasattr(self, "_gen2_train_datasets"):
            split_settings = self.gen2_config.dataset_split
            heldout_count = int(split_settings.get("heldout_count", 1))
            seed = int(split_settings.get("seed", 0))
            split_path = Path(split_settings.get("artifact_path", self.gen2_root / "dataset_split.json"))
            if not split_path.is_absolute():
                split_path = self.gen2_root.parent / split_path
            canonical = self._gen2_source_datasets[0]
            validated = ensure_split(split_path, canonical, heldout_count, seed)
            if validated.dataset_fingerprint != self._gen2_split.dataset_fingerprint:
                raise ValueError("Gen2 in-memory split no longer matches the immutable split artifact")
            return
        if self.datasets_reg is not None:
            raise ValueError("Gen2 forbids datasets_reg")
        if self.data_loader is None:
            raise ValueError("Gen2 requires the real dataloader before split preflight")
        datasets = get_dataloader_datasets(self.data_loader)
        if not datasets:
            raise ValueError("Gen2 requires a non-empty actual dataset")
        configured_datasets = self.config.get("datasets") or []
        if len(configured_datasets) != 1:
            raise ValueError("Gen2 frozen plan requires exactly one configured dataset")
        configured_dataset = (self.config.get("datasets") or [])[0]
        if len(datasets) > 1:
            roots = {str(getattr(source, "dataset_path", None) or getattr(getattr(source, "dataset_config", None), "folder_path", None)) for source in datasets}
            if len(roots) != 1:
                raise ValueError("Gen2 requires resolution-split datasets to share one source folder")
            datasets = sorted(datasets, key=lambda source: int(getattr(getattr(source, "dataset_config", None), "resolution", 0)))
            canonical = datasets[0]
            canonical_pairs = {_item_pair_hash(item, canonical) for item in canonical.file_list}
            if not canonical_pairs:
                raise ValueError("Gen2 canonical dataset shard is empty")
            for source in datasets[1:]:
                shard_pairs = {_item_pair_hash(item, source) for item in source.file_list}
                if shard_pairs != canonical_pairs:
                    raise ValueError("Gen2 resolution shards do not contain identical image-caption pairs")
            dataset = canonical
            if self.accelerator.is_main_process:
                resolutions = [int(getattr(source.dataset_config, "resolution", 0)) for source in datasets]
                print(f"Gen2 using canonical split across resolution shards: {resolutions}")
        else:
            dataset = datasets[0]
        self._gen2_dataset = dataset
        self._gen2_source_dataset = dataset
        self._gen2_source_datasets = datasets
        self._gen2_configured_dataset = configured_dataset
        split_settings = self.gen2_config.dataset_split
        heldout_count = int(split_settings.get("heldout_count", 1))
        seed = int(split_settings.get("seed", 0))
        split_path = Path(split_settings.get("artifact_path", self.gen2_root / "dataset_split.json"))
        if not split_path.is_absolute():
            split_path = self.gen2_root.parent / split_path
        self._gen2_split = ensure_split(split_path, dataset, heldout_count, seed)
        # Zero-heldout is an explicit full-train mode; it is not an error.
        self._gen2_train_datasets = [make_train_dataset_view(source, self._gen2_split) for source in self._gen2_source_datasets]
        self._gen2_train_dataset = self._gen2_train_datasets[0]
        self._prepare_validation_prompts()
        self.data_loader = self._make_gen2_loader(self._gen2_train_datasets, int(self.gen2_config.phase_a.get("batch_size", self.train_config.batch_size)))

    def _make_gen2_loader(self, dataset, batch_size: int):
        """Rebuild a loader from one or more train-only dataset views."""
        datasets = list(dataset) if isinstance(dataset, (list, tuple)) else [dataset]
        if not datasets or any(not getattr(item, "file_list", None) for item in datasets):
            raise ValueError("Gen2 train-only loader requires non-empty dataset views")
        bucketed = [bool(getattr(item.dataset_config, "buckets", False)) for item in datasets]
        if any(bucketed) and not all(bucketed):
            raise ValueError("Gen2 resolution shards must agree on bucket mode")
        if all(bucketed):
            for item in datasets:
                item.batch_size = batch_size
                item.buckets = {}
                item.batch_indices = []
                item.epoch_num = 0
                item.setup_buckets(quiet=True)
            combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
            return DataLoader(combined, batch_size=None, shuffle=True, drop_last=False, collate_fn=dto_collation, num_workers=0)
        combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        return DataLoader(combined, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=dto_collation, num_workers=0)

    def _load_gen2_fixed_probes(self) -> None:
        settings = self.gen2_config.phase_a.get("probes") or {}
        if self._gen2_split is None:
            raise RuntimeError("Gen2 dataset split must be prepared before fixed probes")
        # Probe artifact locations are run-scoped and shared by both phases.
        json_path = self.gen2_root / "fixed_probes.json"
        tensor_path = self.gen2_root / "fixed_probes.safetensors"
        configured_json = settings.get("json_path") or settings.get("artifact_path") or settings.get("path")
        configured_tensor = settings.get("safetensors_path") or settings.get("tensor_artifact_path")
        if configured_json:
            json_path = Path(configured_json)
            if not json_path.is_absolute():
                json_path = self.gen2_root.parent / json_path
        if configured_tensor:
            tensor_path = Path(configured_tensor)
            if not tensor_path.is_absolute():
                tensor_path = self.gen2_root.parent / tensor_path
        timestep_count = int(settings.get("timestep_count", 1))
        expected_pair_count = len(self._gen2_split.heldout_records)
        regions = validate_regions({name: {**metadata, "timestep_count": timestep_count, "pair_count": expected_pair_count} for name, metadata in (settings.get("regions") or {}).items()})
        if not json_path.exists() or not tensor_path.exists():
            self._create_gen2_fixed_probes(json_path, tensor_path, regions)
        probes, metadata = load_fixed_probes(
            json_path,
            tensor_path,
            expected_split_fingerprint=self._gen2_split.dataset_fingerprint,
            expected_regions=regions,
        )
        self._phase_a_probes = [item for item in probes if item.get("split_label") == "train"]
        self._phase_a_heldout_probes = [item for item in probes if item.get("split_label") == "heldout"]
        if self._gen2_split.heldout_count > 0 and not self._phase_a_heldout_probes:
            raise ValueError("fixed probe artifact is missing required heldout probes")
        self._phase_b_probes = list(self._phase_a_heldout_probes)
        self._gen2_fixed_probe_metadata = metadata

    def _create_gen2_fixed_probes(self, json_path: Path, tensor_path: Path, regions: dict[str, dict[str, Any]]) -> None:
        """Create exact fixed probes once; never replace an existing artifact."""
        if json_path.exists() or tensor_path.exists():
            raise FileExistsError("Gen2 fixed probe artifact is incomplete; refusing silent regeneration")
        split = self._gen2_split
        train_records = list(split.train_records)
        heldout_records = list(split.heldout_records)
        pair_count = len(heldout_records)
        if pair_count > len(train_records):
            raise ValueError("Gen2 fixed probes require at least heldout_count train pairs")
        seed = int((self.gen2_config.phase_a.get("probes") or {})["probe_seed"])
        selected_ids = set(select_probe_pairs([record.pair_sha256 for record in train_records], pair_count, seed, "train"))
        selected_train = [record for record in train_records if record.pair_sha256 in selected_ids]
        # Build a separate heldout view without ever attaching it to self.data_loader.
        full_dataset = self._gen2_source_dataset
        heldout_dataset = make_dataset_view(full_dataset, split.heldout_pair_sha256, require_nonempty=False)
        probe_loader = self._make_gen2_loader(self._gen2_train_dataset, 1)
        heldout_loader = self._make_gen2_loader(heldout_dataset, 1) if heldout_dataset.file_list else None
        train_iter, heldout_iter = iter(probe_loader), iter(heldout_loader) if heldout_loader is not None else None
        probes: list[dict[str, Any]] = []
        for split_label, records, iterator, dataset in (("train", selected_train, train_iter, self._gen2_train_dataset), ("heldout", heldout_records, heldout_iter, heldout_dataset)):
            if not records:
                continue
            record_ids = {record.pair_sha256 for record in records}
            pending = set(record_ids)
            while pending:
                if iterator is None:
                    raise RuntimeError("fixed probe dataloader is unavailable")
                batch, iterator = self._next_batch(iterator, probe_loader if split_label == "train" else heldout_loader)
                noisy, noise, timesteps, prompts, _ = self._prepare_real_batch(batch, "a")
                target = self.sd.get_loss_target(noise=noise, batch=batch).detach()
                for index, item in enumerate(batch.file_items):
                    pair = _item_pair_hash(item, dataset)
                    if pair not in pending:
                        continue
                    prompt = prompts[index]
                    for region, metadata in regions.items():
                        start, end, count = int(metadata.get("min", metadata.get("start"))), int(metadata.get("max", metadata.get("end"))), int(metadata["timestep_count"])
                        for ordinal in range(count):
                            timestep_value = deterministic_probe_timestep(seed, pair, region, ordinal, start, end, split_label, count)
                            timestep_tensor = torch.tensor([float(timestep_value)], device=noisy.device)
                            clean_one = batch.latents[index:index + 1]
                            noise_one, noise_seed = deterministic_probe_noise(clean_one, seed, pair, region, ordinal, split_label)
                            expected_noise_seed = deterministic_probe_noise_seed(seed, pair, region, ordinal, split_label)
                            if noise_seed != expected_noise_seed:
                                raise RuntimeError("fixed probe derived noise seed mismatch")
                            noisy_one = self._make_flowmatch_noisy_latents(clean_one, noise_one, timestep_tensor)
                            target_one = (noise_one - clean_one).detach()
                            probes.append({"region": region, "split_label": split_label, "pair_fingerprint": pair, "ordinal": ordinal, "fingerprint": probe_fingerprint(seed, pair, region, ordinal, split_label), "prompt": prompt, "noise_seed": noise_seed, "noisy_latent": noisy_one.detach().cpu().float(), "target": target_one.cpu().float(), "timestep": timestep_tensor.detach().cpu()})
                    pending.pop(pair)
                if hasattr(batch, "cleanup"):
                    batch.cleanup()
        for region, metadata in regions.items():
            metadata["pair_count"] = pair_count
        save_fixed_probes(json_path, tensor_path, probes, split_fingerprint=split.dataset_fingerprint, seed=seed, regions=regions)
        # Force a post-write reload/hash validation before training can proceed.
        load_fixed_probes(json_path, tensor_path, expected_split_fingerprint=split.dataset_fingerprint, expected_regions=regions)

    def _probe_loss_and_direction(self, probes, parameters):
        if not probes:
            return None, None, {"valid": False, "reason": "no_fixed_probes"}
        dataset_losses, helper_losses = [], []
        responses_by_region: dict[str, list[torch.Tensor]] = {}
        for probe in probes:
            noisy = probe["noisy_latent"].to(self.device_torch)
            target = probe["target"].to(self.device_torch)
            timestep = probe["timestep"].to(self.device_torch)
            prompt = probe["prompt"]
            activator = self._activator_prediction([prompt], noisy, timestep)
            dataset_losses.append(dataset_mse(activator, target))
            with torch.no_grad():
                baseline = self._frozen_e0_prediction([prompt], noisy, timestep)
                helper_stack = torch.stack([
                    self._prompt_prediction(
                        [self.placeholder_contract.replace(prompt, helper["replacement"])], noisy, timestep
                    ) for helper in self.helpers
                ])
            helper_loss, cosines, valid = helper_subset_direction_loss(activator, helper_stack, baseline)
            if bool(valid.any()):
                helper_losses.append(helper_loss)
            responses_by_region.setdefault(str(probe["region"]), []).append(spatial_response((activator - baseline).detach())[0])
        if not dataset_losses:
            return None, None, {"valid": False, "reason": "empty_fixed_probes"}
        dataset_loss = torch.stack(dataset_losses).mean()
        helper_loss = torch.stack(helper_losses).mean() if helper_losses else dataset_loss.detach() * 0.0
        cosine = isolated_gradient_cosine(dataset_loss, helper_loss, parameters, retain_graph=False)
        prototype = {region: torch.stack(values).mean(0) for region, values in responses_by_region.items()}
        return dataset_loss.detach(), prototype, {
            "valid": cosine is not None,
            "count": len(dataset_losses),
            "helper_valid_count": len(helper_losses),
            "helper_loss": float(helper_loss.detach().item()),
            "global_gradient_cosine": float(cosine.item()) if cosine is not None else None,
            "prototype_regions": sorted(responses_by_region),
        }

    def _run_frozen_phase_a(self) -> None:
        steps = self._phase_steps("a")
        if steps <= 0:
            return
        self._prepare_gen2_split()
        self._rebuild_dataloader_for_phase("a")
        self._load_gen2_fixed_probes()
        self._initialize_soft_tokens_from_literal_trigger()
        settings = self.gen2_config.phase_a
        helpers_per_step = int(settings.get("helpers_per_step", self.gen2_config.helpers_per_step))
        if not 1 <= helpers_per_step <= 5 or helpers_per_step > len(self.helpers):
            raise ValueError("phase_a.helpers_per_step must be between 1 and the number of helpers")
        accumulation = int((settings.get("curriculum") or {}).get("accumulation_steps", settings.get("gradient_accumulation_steps", 1)))
        if accumulation < 1:
            raise ValueError("phase_a accumulation steps must be positive")
        iterator = iter(self.data_loader)
        parameters = [self.soft_tokens.A]
        if self.qwen_module_lora is not None:
            parameters.extend(self.qwen_module_lora.parameters())
        self.phase_a_root.mkdir(parents=True, exist_ok=True)
        prototype_settings = settings.get("prototype") or self.gen2_config.prototype
        prototype_decay = float(prototype_settings.get("ema_decay", 0.95))
        regions = (settings.get("probes") or {}).get("regions") or {}
        controller_cfg = dict(self.gen2_config.controller)
        controller = AdaptiveJointController(controller_cfg)
        helper_seed = int(settings.get("helper_sampling_seed", 0))
        import random
        helper_rng = random.Random(helper_seed)
        with ToolkitProgressBar(total=steps, desc="Phase A frozen objective") as progress:
            for step in range(steps):
                optimizer = self.optimizer
                optimizer.zero_grad(set_to_none=True)
                dataset_sum = torch.zeros((), device=self.device_torch)
                dataset_count = 0
                direction_sum = torch.zeros_like(dataset_sum)
                direction_count = 0
                prototype_sum = torch.zeros_like(dataset_sum)
                prototype_count = 0
                pending_prototype_samples: dict[str, list[torch.Tensor]] = {}
                direction_vectors = []
                selected = helper_rng.sample(self.helpers, min(helpers_per_step, len(self.helpers)))
                for _micro in range(accumulation):
                    batch, iterator = self._next_batch(iterator, self.data_loader)
                    noisy, noise, timesteps, prompts, _ = self._prepare_real_batch(batch, "a")
                    target = self.sd.get_loss_target(noise=noise, batch=batch).detach()
                    activator = self._activator_prediction(prompts, noisy, timesteps)
                    dataset_values = (activator.float() - target.float()).square().flatten(1).mean(1)
                    dataset_sum = dataset_sum + dataset_values.sum()
                    dataset_count += int(dataset_values.shape[0])
                    with torch.no_grad():
                        baseline = self._frozen_e0_prediction(prompts, noisy, timesteps)
                    region_masks = timestep_region_masks(timesteps, regions)
                    response = spatial_response(activator - baseline.detach())
                    for region, mask in region_masks.items():
                        if mask.any():
                            pending_prototype_samples.setdefault(region, []).append(response[mask].detach())
                    if self._phase_a_prototype:
                        for region, mask in region_masks.items():
                            if region in self._phase_a_prototype and mask.any():
                                region_losses, region_valid = prototype_consistency_loss(
                                    response[mask],
                                    self._phase_a_prototype[region].to(response.device),
                                    floor=float(prototype_settings.get("response_floor", 1e-8)),
                                )
                                if region_valid.any():
                                    prototype_sum = prototype_sum + region_losses[region_valid].sum()
                                    prototype_count += int(region_valid.sum().item())
                    if controller.fraction > 0.0:
                        helper_predictions = []
                        with torch.no_grad():
                            for helper in selected:
                                helper_prompts = [self.placeholder_contract.replace(prompt, helper["replacement"]) for prompt in prompts]
                                helper_predictions.append(self._prompt_prediction(helper_prompts, noisy, timesteps))
                        helper_stack = torch.stack(helper_predictions)
                        direction_loss, cosines, valid = helper_subset_direction_loss(activator, helper_stack, baseline)
                        valid_sample_count = int(valid.sum().item())
                        if valid_sample_count > 0:
                            direction_sum = direction_sum + direction_loss * valid_sample_count
                            direction_count += valid_sample_count
                        direction_vectors.append(cosines.detach()[valid.unsqueeze(0).expand_as(cosines)])
                    if hasattr(batch, "cleanup"):
                        batch.cleanup()
                if dataset_count == 0:
                    raise RuntimeError("Phase A update contains no valid dataset samples")
                dataset_total = dataset_sum / dataset_count
                direction_total = direction_sum / direction_count if direction_count > 0 else dataset_total.detach() * 0.0
                prototype_total = prototype_sum / prototype_count if prototype_count > 0 else dataset_total.detach() * 0.0
                dataset_weight = float(controller_cfg.get("total_semantic_budget", 1.0)) * (1.0 - controller.fraction)
                helper_weight = float(controller_cfg.get("total_semantic_budget", 1.0)) * controller.fraction
                prototype_weight = float(prototype_settings.get("weight", 0.1))
                total_loss = dataset_weight * dataset_total + helper_weight * direction_total + prototype_weight * prototype_total
                gradient_cosine = isolated_gradient_cosine(dataset_total, direction_total, parameters) if direction_count > 0 else None
                self.accelerator.backward(total_loss)
                gradient_norm = self.accelerator.clip_grad_norm_(parameters, self.train_config.max_grad_norm)
                optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                for region, samples in pending_prototype_samples.items():
                    values = torch.cat(samples, dim=0)
                    self._phase_a_prototype = detached_ema_prototype(self._phase_a_prototype, {region: values}, prototype_decay)
                controller_diag = controller.diagnostics()
                evaluation_every = int(controller_cfg.get("evaluation_every", 5))
                if (step + 1) % evaluation_every == 0:
                    train_probe_loss, train_probe_proto, train_probe_diag = self._probe_loss_and_direction(self._phase_a_probes, parameters)
                    heldout_probe_loss, heldout_probe_proto, heldout_probe_diag = self._probe_loss_and_direction(self._phase_a_heldout_probes, parameters)
                    heldout_enabled = self._gen2_split is not None and self._gen2_split.heldout_count > 0
                    train_cosine = train_probe_diag.get("global_gradient_cosine")
                    heldout_cosine = heldout_probe_diag.get("global_gradient_cosine")
                    if heldout_enabled:
                        probe_cosine = min(train_cosine, heldout_cosine) if train_cosine is not None and heldout_cosine is not None else None
                        probe_loss = heldout_probe_loss
                        probe_proto = heldout_probe_proto
                        probe_reason = "conservative_min_train_heldout" if probe_cosine is not None else "train_or_heldout_signal_unavailable"
                    else:
                        probe_cosine = train_cosine
                        probe_loss = train_probe_loss
                        probe_proto = train_probe_proto
                        probe_reason = "explicit_zero_heldout_train_only"
                    train_loss_value = float(train_probe_loss.item()) if train_probe_loss is not None else None
                    heldout_loss_value = float(heldout_probe_loss.item()) if heldout_probe_loss is not None else None
                    probe_diag = {"valid": probe_cosine is not None, "aggregation": probe_reason, "controller_mode": "observational_gradient_cosine_hysteresis", "experimental_multisignal_gates_applied": False, "global_gradient_cosine": probe_cosine, "train": train_probe_diag, "heldout": heldout_probe_diag, "train_loss": train_loss_value, "heldout_loss": heldout_loss_value, "heldout_minus_train_loss": (heldout_loss_value - train_loss_value) if train_loss_value is not None and heldout_loss_value is not None else None, "prototype_signal_available": probe_proto is not None and bool(probe_proto)}
                    controller_diag = controller.update(probe_cosine, probe_proto, gradient_diagnostics={"norm": float(gradient_norm.item()), "probe": probe_diag})
                else:
                    probe_loss, probe_diag = None, {"valid": False, "evaluated": False}
                current_step = step + 1
                self._last_phase_a_diagnostics = {"dataset_loss": float(dataset_total.item()), "helper_direction_loss": float(direction_total.item()), "prototype_loss": float(prototype_total.item()), "probe_loss": float(probe_loss.item()) if probe_loss is not None else 0.0, "helper_direction_cosine": float(torch.cat(direction_vectors).mean().item()) if direction_vectors else 0.0, "dataset_helper_gradient_cosine": float(gradient_cosine.item()) if gradient_cosine is not None else None, "helpers_per_step": helpers_per_step, "microbatch_accumulation": accumulation, "controller": controller_diag, "gradient_norm": float(gradient_norm.item())}
                self._phase_a_history.append({"step": current_step, **self._last_phase_a_diagnostics})
                if self.accelerator.is_main_process:
                    (self.phase_a_root / "diagnostics.json").write_text(json.dumps({"history": self._phase_a_history}, ensure_ascii=False, indent=2), encoding="utf-8")
                    self.logger.log({f"gen2/phase_a/{key}": value for key, value in self._last_phase_a_diagnostics.items() if isinstance(value, (int, float))})
                    self.logger.commit()
                progress.update(1)
                save_every = int(((self.config.get("save") or {}).get("phase_a") or {}).get("save_every", 0))
                if save_every and current_step % save_every == 0:
                    self._phase_checkpoint(
                        "a", current_step,
                        {"E_A": self.soft_tokens.A.detach().cpu(), "E0": self.frozen_e0.detach().cpu()} if self.frozen_e0 is not None else {"E_A": self.soft_tokens.A.detach().cpu()},
                        {"phase": "a", "step": current_step, "prototype": {key: value.detach().cpu().tolist() for key, value in (self._phase_a_prototype or {}).items()}, "history_length": len(self._phase_a_history)},
                        self.optimizer,
                        scheduler=self.lr_scheduler,
                    )
        if self.frozen_e0 is None:
            raise RuntimeError("frozen E0 was not initialized")
        activator_tensors = {"E_A": self.soft_tokens.A.detach().cpu(), "E0": self.frozen_e0.detach().cpu(), **self._qwen_adapter_tensors()}
        activator_metadata = {"artifact": "gen2_activator", "schema_version": 5, "initializer": "literal_trigger_resampled", "literal": ((self.config.get("activator") or {}).get("initialization") or {}).get("literal"), "placeholder": self.placeholder_contract.placeholder, "tokens": self.soft_tokens.tokens, "dimension": self.soft_tokens.dimension, "training_objective": "dataset_helper_direction_timestep_region_ema_prototype", "dataset_split_fingerprint": self._gen2_split.dataset_fingerprint if self._gen2_split else None, "split_ref": self.gen2_config.dataset_split.get("artifact_path"), "probe_ref": (self.gen2_config.phase_a.get("probes") or {}).get("json_path") or (self.gen2_config.phase_a.get("probes") or {}).get("artifact_path"), "prototype": {key: value.detach().cpu().tolist() for key, value in (self._phase_a_prototype or {}).items()}, "controller": controller.state_dict(), "diagnostics_ref": "diagnostics.json", **self._qwen_adapter_metadata()}
        save_tensor_artifact(self.phase_a_root / "activator.safetensors", activator_tensors, activator_metadata)

    def _run_phase_a(self) -> None:
        return self._run_frozen_phase_a()


    def _install_phase_b_optimizer(self) -> None:
        if self._adapter_bank is None:
            raise RuntimeError("Phase B adapter bank has not been installed")
        trainable = enable_adapter_training(self._adapter_bank)
        self.train_config.steps = self._phase_steps("b")
        settings = self.gen2_config.phase_b
        network_params = []
        temporal_params = []
        for module in self._adapter_bank.modules():
            if module is self._adapter_bank:
                continue
            if module.__class__.__name__ == "TemporalField":
                temporal_params.extend(module.parameters())
        temporal_ids = {id(parameter) for parameter in temporal_params}
        network_params = [parameter for parameter in self._adapter_bank.parameters() if id(parameter) not in temporal_ids]
        self.params = [
            {"params": network_params, "lr": float(settings.get("network_lr", 1e-4)), "weight_decay": float(settings.get("weight_decay", 1e-4))},
            {"params": temporal_params, "lr": float(settings.get("temporal_lr", 5e-4)), "weight_decay": float(settings.get("temporal_weight_decay", 0.0))},
        ]
        optimizer_type = settings.get("optimizer", self.train_config.optimizer)
        optimizer_params = dict(settings.get("optimizer_params", self.train_config.optimizer_params))
        self.optimizer = get_optimizer(
            self.params,
            optimizer_type=optimizer_type,
            learning_rate=float(settings.get("network_lr", 1e-4)),
            optimizer_params=optimizer_params,
        )
        self.optimizer = self.accelerator.prepare(self.optimizer)
        scheduler_name = settings.get("lr_scheduler", self.train_config.lr_scheduler)
        scheduler_params = dict(settings.get("lr_scheduler_params", self.train_config.lr_scheduler_params))
        scheduler_params.setdefault("total_iters", self.train_config.steps)
        self.lr_scheduler = get_lr_scheduler(scheduler_name, self.optimizer, **scheduler_params)
        self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)

    def _load_activator_artifact(self) -> None:
        artifact_path = (self.config.get("activator") or {}).get("artifact_path")
        if self.gen2_config.mode not in {"phase_b_from_activator", "resume_phase_b"}:
            return
        if not artifact_path:
            raise ValueError(f"{self.gen2_config.mode} requires activator.artifact_path")
        if Path(artifact_path).suffix != ".safetensors":
            raise ValueError("Gen2 activator artifacts must use safetensors")
        tensors, metadata = load_tensor_artifact(artifact_path)
        artifact = metadata.get("artifact")
        if artifact != "gen2_activator":
            raise ValueError(f"unsupported activator artifact type: {artifact!r}")
        if int(metadata.get("schema_version", -1)) != 5:
            raise ValueError("activator artifact schema_version must be 5 and include immutable E0")
        expected_literal = ((self.config.get("activator") or {}).get("initialization") or {}).get("literal")
        if metadata.get("literal") != expected_literal:
            raise ValueError("activator artifact literal bootstrap does not match the current configuration")
        if metadata.get("initializer") != "literal_trigger_resampled":
            raise ValueError("activator artifact was not initialized from the literal trigger")
        if metadata.get("placeholder") != self.placeholder_contract.placeholder:
            raise ValueError("activator artifact placeholder does not match the current contract")
        activator = tensors.get("E_A")
        e0 = tensors.get("E0")
        if activator is None or e0 is None:
            raise ValueError("activator artifact must contain tensors E_A and immutable E0")
        if tuple(activator.shape) != tuple(self.soft_tokens.A.shape) or tuple(e0.shape) != tuple(self.soft_tokens.A.shape):
            raise ValueError("activator artifact E_A/E0 shapes do not match configured soft tokens")
        if int(metadata.get("tokens", -1)) != self.soft_tokens.tokens or int(metadata.get("dimension", -1)) != self.soft_tokens.dimension:
            raise ValueError("activator artifact dimensions do not match the configured soft token bank")
        self.soft_tokens.A.data.copy_(activator.to(self.soft_tokens.A))
        self.frozen_e0 = e0.to(device=self.soft_tokens.A.device, dtype=self.soft_tokens.A.dtype).detach().clone()
        self._load_qwen_adapter_state(tensors, metadata)

    def _restore_phase_b_checkpoint(self, optimizer) -> int:
        if self.gen2_config.mode != "resume_phase_b":
            return 0
        candidates = [item for item in self.phase_b_root.iterdir() if item.is_dir() and item.name.startswith("step_")]
        if not candidates:
            raise FileNotFoundError("resume_phase_b requires a phase_b/step_<step> checkpoint")
        checkpoint = max(candidates, key=lambda item: int(item.name.removeprefix("step_")))
        tensors, metadata = load_phase_checkpoint(checkpoint, optimizer=optimizer, scheduler=self.lr_scheduler)
        adapter_tensors = {key: value for key, value in tensors.items() if not key.startswith("qwen_adapter.")}
        self._adapter_bank.load_state_dict(adapter_tensors, strict=True)
        self._load_qwen_adapter_state(tensors, metadata)
        sampler_state = metadata.get("timestep_sampler") or {}
        order = sampler_state.get("order")
        if order is not None:
            self._timestep_bin_orders["b"] = torch.tensor(order, device=self.device_torch, dtype=torch.long)
            self._timestep_bin_cursors["b"] = int(sampler_state.get("cursor", 0))
        expected_phase = "b"
        if metadata.get("phase") != expected_phase:
            raise ValueError(f"resume checkpoint phase must be {expected_phase}")
        return int(metadata.get("step", 0))

    def _run_phase_b(self) -> None:
        steps = self._phase_steps("b")
        if steps <= 0:
            return
        if self.data_loader is None or self._adapter_bank is None:
            raise RuntimeError("Phase B requires a loaded Ideogram model and real dataloader")
        self._rebuild_dataloader_for_phase("b")
        self._adapter_bank.train()
        settings = self.gen2_config.phase_b
        if self.optimizer is None:
            raise RuntimeError("Phase B optimizer was not installed")
        optimizer = self.optimizer
        trainable = [parameter for parameter in self._adapter_bank.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("Phase B adapter bank has no trainable parameters at loop start")
        self.modules_being_trained = [self._adapter_bank]
        self.phase_b_root.mkdir(parents=True, exist_ok=True)
        start_step = self._restore_phase_b_checkpoint(optimizer)
        iterator = iter(self.data_loader)
        if not self._phase_b_probes:
            if self._gen2_split is None or self._gen2_split.heldout_records:
                raise RuntimeError("Phase B requires strictly loaded heldout fixed probes")
            # Explicit heldout_count=0 mode uses train fixed probes for diagnostics only.
            self._phase_b_probes = list(self._phase_a_probes)
        with ToolkitProgressBar(total=steps, desc="Phase B2 TRF-LoRA") as progress:
            for step in range(start_step, steps):
                self._assert_qwen_frozen()
                batch, iterator = self._next_batch(iterator, self.data_loader)
                noisy_latents, noise, timesteps, prompts, _ = self._prepare_real_batch(batch, "b")
                embeds = type(self.sd.get_prompt_embeds(prompts))(
                    text_embeds=[self._encode_soft_prompt(prompt).detach() for prompt in prompts]
                )
                target = self.sd.get_loss_target(noise=noise, batch=batch)
                optimizer.zero_grad(set_to_none=True)
                batch_size = noisy_latents.shape[0]
                style_gate = torch.ones(batch_size, device=self.device_torch, dtype=torch.float32)
                training_context = AdapterRuntimeContext(timesteps.float() / 1000.0, style_gate, 1.0)
                prediction = self.sd.get_noise_prediction(
                    noisy_latents,
                    timesteps,
                    embeds,
                    adapter_context=training_context,
                )
                saved_training_context = getattr(self.sd.transformer, "_gen2_adapter_context", None)
                if saved_training_context is None:
                    raise RuntimeError("Phase B training forward did not retain adapter context for checkpoint recomputation")
                if not prediction.requires_grad:
                    raise RuntimeError("Phase B prediction has no grad_fn; adapter parameters are not connected")
                fm = torch.nn.functional.mse_loss(prediction.float(), target.float())
                with torch.no_grad():
                    probe_on_losses = []
                    probe_off_losses = []
                    probe_delta_norms = []
                    from extensions_built_in.gen2_trainer.registry import clear_adapter_context
                    for probe in self._phase_b_probes:
                        if isinstance(probe, dict):
                            probe_noisy = probe["noisy_latent"].to(self.device_torch)
                            probe_timesteps = probe["timestep"].to(self.device_torch)
                            probe_target = probe["target"].to(self.device_torch)
                            probe_prompt = probe["prompt"]
                        else:
                            probe_noisy, probe_timesteps, probe_target, probe_prompt = probe
                            probe_noisy, probe_timesteps, probe_target = (item.to(self.device_torch) for item in (probe_noisy, probe_timesteps, probe_target))
                        probe_embeds = type(self.sd.get_prompt_embeds([probe_prompt]))(text_embeds=[self._encode_soft_prompt(probe_prompt).detach()])
                        probe_on = self.sd.get_noise_prediction(probe_noisy, probe_timesteps, probe_embeds, adapter_context=AdapterRuntimeContext(probe_timesteps.float() / 1000.0, torch.ones(1, device=self.device_torch), 1.0))
                        clear_adapter_context(self.sd.transformer)
                        probe_off = self.sd.get_noise_prediction(probe_noisy, probe_timesteps, probe_embeds, adapter_context=AdapterRuntimeContext(probe_timesteps.float() / 1000.0, torch.zeros(1, device=self.device_torch), 1.0))
                        clear_adapter_context(self.sd.transformer)
                        probe_on_losses.append(torch.nn.functional.mse_loss(probe_on.float(), probe_target.float()))
                        probe_off_losses.append(torch.nn.functional.mse_loss(probe_off.float(), probe_target.float()))
                        probe_delta_norms.append((probe_on - probe_off).float().norm())
                    probe_on_loss = torch.stack(probe_on_losses).mean()
                    probe_off_loss = torch.stack(probe_off_losses).mean()
                    probe_on_loss_std = torch.stack(probe_on_losses).std(unbiased=False)
                    probe_off_loss_std = torch.stack(probe_off_losses).std(unbiased=False)
                    probe_delta_norm = torch.stack(probe_delta_norms).mean()
                self.sd.transformer._gen2_adapter_context = saved_training_context
                smooth = time_smooth_regularizer(self._adapter_bank.temporal_fields())
                loss = fm + float(settings.get("temporal_smooth_weight", 1e-4)) * smooth
                self.accelerator.backward(loss)
                if not any(parameter.grad is not None for parameter in trainable):
                    raise RuntimeError("Phase B backward produced no adapter gradients")
                gradient_norm = float(self.accelerator.clip_grad_norm_(trainable, self.train_config.max_grad_norm).item())
                optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                current_step = step + 1
                adapter_norm = torch.stack([parameter.detach().float().norm() for parameter in trainable]).mean()
                temporal_values = time_mean_diagnostic(self._adapter_bank.temporal_fields()).detach()
                with torch.no_grad():
                    from extensions_built_in.gen2_trainer.registry import clear_adapter_context
                    clear_adapter_context(self.sd.transformer)
                    prediction_off = self.sd.get_noise_prediction(
                        noisy_latents,
                        timesteps,
                        embeds,
                        adapter_context=AdapterRuntimeContext(timesteps.float() / 1000.0, torch.zeros(batch_size, device=self.device_torch), 1.0),
                    )
                    clear_adapter_context(self.sd.transformer)
                    current_prediction = self.sd.get_noise_prediction(
                        noisy_latents,
                        timesteps,
                        embeds,
                        adapter_context=AdapterRuntimeContext(timesteps.float() / 1000.0, torch.ones(batch_size, device=self.device_torch), 1.0),
                    )
                    clear_adapter_context(self.sd.transformer)
                    residual_scale = (current_prediction - prediction_off).float().norm() / current_prediction.float().norm().clamp_min(1e-8)
                phase_b_diagnostics = {
                    "fm_loss": float(fm.detach().item()),
                    "temporal_smooth_loss": float(smooth.detach().item()),
                    "adapter_parameter_norm": float(adapter_norm.item()),
                    "temporal_field_abs_mean": float(temporal_values.abs().mean().item()),
                    "temporal_field_abs_max": float(temporal_values.abs().max().item()),
                    "gradient_norm": gradient_norm,
                    "probe_on_loss": float(probe_on_loss.item()),
                    "probe_on_loss_std": float(probe_on_loss_std.item()),
                    "probe_off_loss": float(probe_off_loss.item()),
                    "probe_off_loss_std": float(probe_off_loss_std.item()),
                    "probe_gain": float((probe_off_loss - probe_on_loss).item()),
                    "probe_adapter_delta_norm": float(probe_delta_norm.item()),
                    "training_residual_relative_norm": float(residual_scale.item()),
                }
                self._phase_b_history.append({"step": current_step, **phase_b_diagnostics})
                if self.accelerator.is_main_process:
                    (self.phase_b_root / "diagnostics.json").write_text(json.dumps({"history": self._phase_b_history}, ensure_ascii=False, indent=2), encoding="utf-8")
                save_every = int(((self.config.get("save") or {}).get("phase_b") or {}).get("save_every", 0))
                if save_every and current_step % save_every == 0:
                    self._phase_checkpoint(
                        "b",
                        current_step,
                        {key: value.detach().cpu() for key, value in self._adapter_bank.state_dict().items()},
                        {"phase": "b", "step": current_step, "temporal_mean": temporal_values.cpu().tolist(), "diagnostics": phase_b_diagnostics, "history_length": len(self._phase_b_history)},
                        optimizer,
                        scheduler=self.lr_scheduler,
                    )
                if self.accelerator.is_main_process:
                    self.logger.log({f"gen2/phase_b/{key}": value for key, value in phase_b_diagnostics.items()})
                    self.logger.commit()
                progress.set_postfix(step=current_step, total=steps, fm=f"{fm.item():.4g}", time_smooth=f"{smooth.item():.4g}", adapter_norm=f"{adapter_norm.item():.3g}")
                progress.update(1)
                if hasattr(batch, "cleanup"):
                    batch.cleanup()
        from extensions_built_in.gen2_trainer.registry import clear_adapter_context
        clear_adapter_context(self.sd.transformer)
        adapter_path = self.phase_b_root / "adapter.pt"
        final_step = steps
        self._phase_checkpoint(
            "b",
            final_step,
            {key: value.detach().cpu() for key, value in self._adapter_bank.state_dict().items()},
            {"phase": "b", "step": final_step, "temporal_mean": time_mean_diagnostic(self._adapter_bank.temporal_fields()).detach().cpu().tolist()},
            optimizer,
            scheduler=self.lr_scheduler,
        )
        network = self.config.get("network") or {}
        temporal = network.get("temporal") or {}
        (self.phase_b_root / "diagnostics.json").write_text(json.dumps({"history": self._phase_b_history}, ensure_ascii=False, indent=2), encoding="utf-8")
        save_tensor_artifact(
            self.phase_b_root / "adapter.safetensors",
            dict(self._adapter_bank.state_dict()),
            {
                "artifact": "gen2_temporal_rank_field_lora",
                "rank": int(network.get("rank", 32)),
                "alpha": float(network.get("alpha", network.get("rank", 32))),
                "temporal_knots": int(temporal.get("knots", 8)),
                "temporal_delta_max": float(temporal.get("delta_max", 1.0)),
                "registry": self._adapter_bank.registry,
                "training_scope": "conditional_only",
                "image_token_only": True,
            },
        )
        torch.save({"state_dict": self._adapter_bank.state_dict(), "registry": self._adapter_bank.registry}, adapter_path)
        if self.official_unconditional is not None:
            unconditional_bank = getattr(self.official_unconditional.transformer, "_gen2_adapter_bank", None)
            if unconditional_bank is None:
                raise RuntimeError("official unconditional transformer is missing Gen2 adapter bank")
            unconditional_bank.load_state_dict(self._adapter_bank.state_dict(), strict=True)

    def _run_official_cfg_validation(self) -> None:
        sample = self.config.get("sample") or {}
        if bool(sample.get("disable_sampling", False)):
            raise RuntimeError("Gen2 smoke/validation requires sampling; disable_sampling=true is not allowed")
        cases = build_validation_matrix(
            len(self._validation_prompts),
            self.gen2_config.seeds,
            sample.get("unconditional_scale_sweep") or [],
            include_sweep=True,
            designated_helper_index=next(index for index, helper in enumerate(self.helpers) if helper["id"] == self._designated_helper_id),
        )
        if not cases:
            raise RuntimeError("Gen2 validation matrix is empty")
        self.validation_cases = cases
        output_root = self.gen2_root / "samples" / f"step_{self._phase_steps('b'):06d}"
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(f"Gen2 validation output already exists; refusing overwrite: {output_root}")
        output_root.mkdir(parents=True, exist_ok=False)
        diagnostics_path = output_root / "conditioning_diagnostics.json"
        diagnostics_records: list[dict[str, Any]] = []
        progress = ToolkitProgressBar(total=len(cases), desc=f"Gen2 official CFG validation ({len(cases)} cases)")
        try:
            for case_index, case in enumerate(cases, start=1):
                if case.eta_u > 0 and self.official_unconditional is None:
                    raise RuntimeError("positive eta_u requires official unconditional transformer")
                raw_prompt = self._validation_prompts[case.prompt_index]
                case_diagnostics: dict[str, Any] = {
                    "case_index": case_index,
                    "prompt_index": case.prompt_index,
                    "seed": case.seed,
                    "conditioning_mode": case.conditioning_mode,
                    "style_gate": case.style_gate,
                    "eta_c": case.eta_c,
                    "eta_u": case.eta_u,
                    "adapter_conditional_enabled": case.style_gate != 0.0 and case.eta_c != 0.0,
                    "adapter_unconditional_enabled": case.style_gate != 0.0 and case.eta_u != 0.0,
                }
                if case.conditioning_mode == "helper":
                    materialized = self.placeholder_contract.replace(raw_prompt, self.helpers[case.helper_index]["replacement"])
                    embeds = self.sd.get_prompt_embeds([materialized])
                    case_diagnostics["materialized_helper_id"] = self.helpers[case.helper_index]["id"]
                    case_diagnostics["text_length"] = int(embeds.text_embeds[0].shape[0])
                    case_diagnostics["text_feature_norm_mean"] = float(embeds.text_embeds[0].float().norm(dim=-1).mean().item())
                    case_diagnostics["text_feature_norm_max"] = float(embeds.text_embeds[0].float().norm(dim=-1).max().item())
                elif case.conditioning_mode == "activator":
                    embeds = type(self.sd.get_prompt_embeds([raw_prompt]))(
                        text_embeds=[self._encode_soft_prompt(raw_prompt, case_diagnostics).detach()]
                    )
                elif case.conditioning_mode == "literal_e0":
                    embeds = self._literal_prompt_embeds(raw_prompt)
                    case_diagnostics["literal_e0"] = True
                    case_diagnostics["text_length"] = int(embeds.text_embeds[0].shape[0])
                    case_diagnostics["text_feature_norm_mean"] = float(embeds.text_embeds[0].float().norm(dim=-1).mean().item())
                    case_diagnostics["text_feature_norm_max"] = float(embeds.text_embeds[0].float().norm(dim=-1).max().item())
                else:
                    raise RuntimeError(f"unknown validation conditioning mode: {case.conditioning_mode}")
                generator = torch.Generator(device=self.device_torch).manual_seed(case.seed)
                denoise_diagnostics: list[dict[str, Any]] = []
                denoise_progress = ToolkitProgressBar(
                    total=int(sample.get("steps", 4)),
                    desc=f"CFG case {case_index}/{len(cases)}",
                    leave=False,
                )
                try:
                    images = self.sd.pipeline(
                        conditional_embeds=embeds,
                        unconditional_embeds=None,
                        height=int(sample.get("height", 512)),
                        width=int(sample.get("width", 512)),
                        num_inference_steps=int(sample.get("steps", 4)),
                        guidance_scale=float(sample.get("guidance_scale", 7.0)),
                        generator=generator,
                        require_official_unconditional=True,
                        official_unconditional_transformer=self.official_unconditional,
                        style_adapter_conditional_scale=case.eta_c,
                        style_adapter_unconditional_scale=case.eta_u,
                        style_gate=torch.tensor([case.style_gate], device=self.device_torch),
                        step_callback=lambda _step, _total: denoise_progress.update(1),
                        diagnostics_callback=denoise_diagnostics.append,
                    )
                finally:
                    denoise_progress.close()
                case_root = output_root / f"prompt_{case.prompt_index:02d}" / f"seed_{case.seed}"
                case_root.mkdir(parents=True, exist_ok=True)
                name = f"{case.conditioning_mode}_b_{'on' if case.style_gate and case.eta_c else 'off'}_eta_u_{case.eta_u:g}"
                image_path = case_root / f"{name}.png"
                if image_path.exists():
                    raise FileExistsError(f"Gen2 validation refuses to overwrite {image_path}")
                images[0].save(image_path)
                case_diagnostics["denoise"] = denoise_diagnostics
                diagnostics_records.append(case_diagnostics)
                diagnostics_path.write_text(json.dumps(diagnostics_records, ensure_ascii=False, indent=2), encoding="utf-8")
                progress.set_postfix(case=case_index, mode=case.conditioning_mode, eta_u=f"{case.eta_u:g}")
                progress.update(1)
                self.official_unconditional.transformer.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            progress.close()
        if self.accelerator.is_main_process:
            self.logger.log({"gen2/official_cfg_validation_cases": len(cases)})
            self.logger.commit()
            print(f"Gen2 official CFG validation completed: {len(cases)} cases")

    def run(self) -> None:
        mode = self.gen2_config.mode
        self._write_manifest("running")
        try:
            self._run_impl(mode)
        except KeyboardInterrupt:
            self._write_manifest("interrupted", error="KeyboardInterrupt")
            raise
        except Exception as error:
            self._write_manifest("failed", error=f"{type(error).__name__}: {error}")
            raise

    def _run_impl(self, mode: str) -> None:
        self._initialize_soft_tokens()
        self.train_config = copy.copy(self.train_config)
        self.train_config.steps = self._phase_steps("a")
        self.train_config.train_unet = False
        self.train_config.train_text_encoder = False
        self.train_config.train_refiner = False
        self.network_config = None
        self.is_fine_tuning = True
        self._prepare_only = True
        super().run()
        self._prepare_only = False
        if mode in {"auto", "phase_a_only"}:
            self._run_phase_a()
            self._write_manifest("phase_a_training_completed", completed_phase="phase_a")
        else:
            self._prepare_gen2_split()
            self._load_gen2_fixed_probes()
            self._load_activator_artifact()
        if mode == "phase_a_only":
            print("Gen2 training completed: Phase A only")
            self.accelerator.end_training()
            return
        official = self.load_official_unconditional()
        official_transformer = official.transformer
        network = self.config.get("network") or {}
        temporal = network.get("temporal") or {}
        unconditional_bank = install_ideogram_adapters(
            official_transformer,
            rank=int(network.get("rank", 32)),
            alpha=float(network.get("alpha", network.get("rank", 32))),
            knots=int(temporal.get("knots", 8)),
            delta_max=float(temporal.get("delta_max", 1.0)),
        )
        unconditional_bank.requires_grad_(False)
        if len(unconditional_bank.registry) != len(self._adapter_bank.registry) or unconditional_bank.registry != self._adapter_bank.registry:
            raise RuntimeError("official unconditional adapter registry is not exactly compatible")
        self.sd.pipeline.official_unconditional_transformer = official_transformer
        self.sd.pipeline.require_official_unconditional = True
        style_adapter = (self.config.get("sample") or {}).get("style_adapter") or {}
        self.sd.pipeline.style_adapter_conditional_scale = float(style_adapter.get("conditional_scale", 1.0))
        self.sd.pipeline.style_adapter_unconditional_scale = float(style_adapter.get("unconditional_scale", 0.0))
        if mode in {"auto", "phase_b_from_activator", "resume_phase_b"}:
            self._install_phase_b_optimizer()
            self._run_phase_b()
            self._write_manifest("phase_b_training_completed", completed_phase="phase_b")
            self._run_official_cfg_validation()
            self._write_manifest("completed", completed_phase="phase_b")
        print("Gen2 training completed: Phase A + Phase B + official CFG validation")
        self.accelerator.end_training()

    def _initialize_soft_tokens(self) -> None:
        model = self.config.get("model") or {}
        dimension = int(model.get("qwen_hidden_size", 4096))
        tokens = int((self.config.get("activator") or {}).get("tokens", 24))
        self.soft_tokens = SoftTokenBank(tokens, dimension)

    def _initialize_soft_tokens_from_literal_trigger(self) -> None:
        language_model = self.sd.text_encoder.language_model
        embedding_weight = language_model.embed_tokens.weight
        if self.soft_tokens.dimension != embedding_weight.shape[-1]:
            raise ValueError(
                "activator dimension must match the loaded Qwen embedding dimension: "
                f"configured={self.soft_tokens.dimension}, actual={embedding_weight.shape[-1]}"
            )
        literal = ((self.config.get("activator") or {}).get("initialization") or {}).get("literal", "")
        if not isinstance(literal, str) or not literal.strip():
            raise ValueError("activator.initialization.literal must be a non-empty string")
        encoded = self.sd.tokenizer(
            literal,
            add_special_tokens=False,
            return_tensors="pt",
        )
        token_ids = encoded.get("input_ids")
        if token_ids is None or token_ids.numel() == 0:
            raise ValueError("literal activator placeholder produced no tokenizer tokens")
        token_ids = token_ids.to(device=embedding_weight.device, dtype=torch.long)
        with torch.no_grad():
            literal_embeddings = language_model.embed_tokens(token_ids)[0].float()
            initialization = resample_embedding_sequence(literal_embeddings, self.soft_tokens.tokens)
        initialized = initialization.to(device=self.soft_tokens.A.device, dtype=self.soft_tokens.A.dtype)
        self.soft_tokens.A.data.copy_(initialized)
        self.frozen_e0 = initialized.detach().clone()
        if not torch.equal(self.soft_tokens.A.detach(), self.frozen_e0):
            raise RuntimeError("Gen2 initialization invariant failed: E_A and frozen E0 differ")
        if self.frozen_e0.requires_grad:
            raise RuntimeError("Gen2 initialization invariant failed: frozen E0 requires gradients")
    def preflight_caption(self, raw_caption: str, source: str = "caption") -> int:
        _, occurrences = self.placeholder_contract.parse(raw_caption)
        if not occurrences:
            raise ValueError(f"{source} is missing the literal [trigger] placeholder")
        return len(occurrences)

    def load_official_unconditional(self):
        from .unconditional import OfficialIdeogramUnconditionalLoader
        from toolkit.train_tools import get_torch_dtype

        model = self.config.get("model") or {}
        loader = OfficialIdeogramUnconditionalLoader(
            model["name_or_path"],
            dtype=get_torch_dtype(model.get("dtype", "bf16")),
            device=self.device_torch,
            quantize=bool(model.get("quantize", True)),
            qtype=model.get("qtype", "qfloat8"),
        )
        self.official_unconditional = loader.load()
        from extensions_built_in.gen2_trainer.unconditional import assert_official_backend

        assert_official_backend(self.official_unconditional)
        return self.official_unconditional
