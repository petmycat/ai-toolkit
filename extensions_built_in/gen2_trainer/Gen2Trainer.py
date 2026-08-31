from __future__ import annotations

import copy
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.progress_bar import ToolkitProgressBar

from .activator import (
    PlaceholderContract,
    SoftTokenBank,
    TriggerLocalTEAdapter,
    encode_qwen_inputs_embeds,
    normalize_inline_helpers,
    replace_token_spans_with_soft_tokens,
    resample_embedding_sequence,
)
from .artifacts import load_tensor_artifact, save_tensor_artifact
from .checkpoint import load_phase_checkpoint, save_phase_checkpoint
from .config import Gen2RuntimeConfig, validate_gen2_config
from toolkit.optimizer import get_optimizer
from toolkit.scheduler import get_lr_scheduler
from .registry import AdapterRuntimeContext, enable_adapter_training, install_ideogram_adapters
from .sampling import build_validation_matrix, make_flowmatch_noisy_latents, sample_stratified_timesteps
from .temporal_rank_field import time_smooth_regularizer, time_mean_diagnostic
from .phase_a_math import calibrate_helper_losses, disturbance_projection, robust_competence, semantic_direction_loss, smooth_helper_weight


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
        self._soft_tokens_initial: torch.Tensor | None = None
        self.trigger_local_adapter: TriggerLocalTEAdapter | None = None
        self._helper_calibration: dict[str, Any] | None = None
        self._helper_latched = False
        self._helper_release_streak = 0
        self._heldout_competence: float | None = None
        self._last_phase_a_diagnostics: dict[str, float] = {}
        self._phase_a_history: list[dict[str, Any]] = []
        self._phase_b_history: list[dict[str, Any]] = []
        self._phase_a_probes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]] = []
        self._phase_a_heldout_probes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]] = []
        self._phase_b_probes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]] = []
        self.adapter = torch.nn.ModuleDict()
        self.official_unconditional = None
        self.helpers = normalize_inline_helpers(self.gen2_config.helpers)
        self._prepared = False
        self._adapter_bank = None
        self._timestep_bin_orders: dict[str, torch.Tensor] = {}
        self._timestep_bin_cursors: dict[str, int] = {"a": 0, "b": 0}

    def _assert_qwen_frozen(self) -> None:
        trainable = [name for name, parameter in self.sd.text_encoder.named_parameters() if parameter.requires_grad]
        if trainable:
            raise RuntimeError(f"Gen2 requires frozen Qwen parameters, but these are trainable: {trainable[:8]}")

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
        self._assert_qwen_frozen()
        self.sd.vae.requires_grad_(False)
        activator_config = self.config.get("activator") or {}
        local_config = activator_config.get("trigger_local_adapter") or {}
        self.trigger_local_adapter = TriggerLocalTEAdapter(
            self.soft_tokens.dimension,
            rank=int(local_config.get("rank", 4)),
            alpha=float(local_config.get("alpha", 4.0)),
        ) if bool(local_config.get("enabled", False)) else None
        if self.trigger_local_adapter is not None:
            self.trigger_local_adapter.to(self.device_torch)
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
        if self.trigger_local_adapter is not None:
            self.modules_being_trained.append(self.trigger_local_adapter)

    def hook_before_train_loop(self):
        self._preflight_loaded_datasets()
        if self.accelerator.is_main_process:
            self.logger.start()
        self.prepare_accelerator()

    def prepare_accelerator(self):
        self.accelerator.even_batches = False
        self.sd.vae = self.accelerator.prepare(self.sd.vae)
        self.sd.unet = self.accelerator.prepare(self.sd.unet)
        modules = [self.soft_tokens]
        if self.trigger_local_adapter is not None:
            modules.append(self.trigger_local_adapter)
        prepared = self.accelerator.prepare(*modules, self.optimizer)
        if self.trigger_local_adapter is not None:
            self.soft_tokens, self.trigger_local_adapter, self.optimizer = prepared
        else:
            self.soft_tokens, self.optimizer = prepared
        self._prepared = True
        if self.lr_scheduler is not None:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
        self.modules_being_trained = [self.soft_tokens]
        if self.trigger_local_adapter is not None:
            self.modules_being_trained.append(self.trigger_local_adapter)

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
        if self.trigger_local_adapter is not None:
            local_config = ((self.config.get("activator") or {}).get("trigger_local_adapter") or {})
            params.append({
                "params": list(self.trigger_local_adapter.parameters()),
                "lr": float(local_config.get("lr", self.train_config.lr * 0.25)),
                "weight_decay": float(local_config.get("weight_decay", 0.0)),
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

    def _encode_soft_prompt(self, raw_caption: str, diagnostics: dict[str, Any] | None = None, return_pooled: bool = False):
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
        expanded, expanded_spans = replace_token_spans_with_soft_tokens(ordinary, spans, self.soft_tokens)
        activator_mask = torch.zeros(expanded.shape[0], device=expanded.device, dtype=torch.long)
        for start, end in expanded_spans:
            activator_mask[start:end] = 1
        if diagnostics is not None:
            ordinary_norm = ordinary.float().norm(dim=-1)
            soft_norm = self.soft_tokens.A.float().norm(dim=-1)
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
        encoded = encode_qwen_inputs_embeds(
            self.sd.text_encoder,
            expanded,
            attention_mask,
            pos_2d,
            QWEN3_VL_ACTIVATION_LAYERS,
            activator_mask=activator_mask.unsqueeze(0),
            trigger_local_adapter=self.trigger_local_adapter,
            return_details=diagnostics is not None or return_pooled,
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

    def _phase_checkpoint(self, phase: str, step: int, tensors: dict[str, torch.Tensor], metadata: dict[str, Any], optimizer, scheduler=None) -> None:
        root = self.phase_a_root if phase == "a" else self.phase_b_root
        path = root / f"step_{step:06d}"
        phase_key = phase.lower()
        order = self._timestep_bin_orders.get(phase_key)
        metadata = dict(metadata)
        metadata["timestep_sampler"] = {
            "order": order.detach().cpu().tolist() if order is not None else None,
            "cursor": int(self._timestep_bin_cursors.get(phase_key, 0)),
        }
        save_phase_checkpoint(path, tensors, metadata, optimizer=optimizer, scheduler=scheduler)
        keep = int((self.config.get("save") or {}).get("max_step_saves_to_keep", 0))
        if keep > 0:
            checkpoints = sorted(
                (item for item in root.iterdir() if item.is_dir() and item.name.startswith("step_")),
                key=lambda item: int(item.name.removeprefix("step_")),
            )
            for old_checkpoint in checkpoints[:-keep]:
                shutil.rmtree(old_checkpoint)

    def _native_ordinary_pooled(self, prompts, replacements):
        from toolkit.ideogram_caption import digest_caption_string

        pooled = []
        for prompt, replacement in zip(prompts, replacements):
            materialized = self.placeholder_contract.replace(prompt, replacement)
            normalized = digest_caption_string(materialized)
            messages = [{"role": "user", "content": [{"type": "text", "text": normalized}]}]
            text = self.sd.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            encoded = self.sd.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=self.sd.max_text_length)
            offsets = encoded.get("offset_mapping")
            features = self.sd.get_prompt_embeds([materialized]).text_embeds[0].float()
            if offsets is None:
                pooled.append(features.mean(dim=0))
                continue
            replacement_indices = set()
            cursor = 0
            while True:
                start = text.find(replacement, cursor)
                if start < 0:
                    break
                end = start + len(replacement)
                replacement_indices.update(index for index, (left, right) in enumerate(offsets) if right > start and left < end)
                cursor = end
            keep = [index for index in range(features.shape[0]) if index not in replacement_indices]
            pooled.append(features[keep].mean(dim=0) if keep else features.mean(dim=0))
        return torch.stack(pooled)

    def _native_span_pooled(self, prompts, replacements):
        from toolkit.ideogram_caption import digest_caption_string

        pooled = []
        for prompt, replacement in zip(prompts, replacements):
            materialized = self.placeholder_contract.replace(prompt, replacement)
            normalized = digest_caption_string(materialized)
            messages = [{"role": "user", "content": [{"type": "text", "text": normalized}]}]
            text = self.sd.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            encoded = self.sd.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=self.sd.max_text_length)
            offsets = encoded.get("offset_mapping")
            features = self.sd.get_prompt_embeds([materialized]).text_embeds[0].float()
            if offsets is None:
                pooled.append(features.mean(dim=0))
                continue
            spans = []
            cursor = 0
            while True:
                start = text.find(replacement, cursor)
                if start < 0:
                    break
                end = start + len(replacement)
                indices = [index for index, (left, right) in enumerate(offsets) if right > start and left < end]
                if indices:
                    spans.append((indices[0], indices[-1] + 1))
                cursor = end
            if not spans:
                pooled.append(features.mean(dim=0))
                continue
            pieces = [features[start:min(end, features.shape[0])] for start, end in spans if start < features.shape[0]]
            pooled.append(torch.stack([piece.mean(dim=0) for piece in pieces]).mean(dim=0))
        return torch.stack(pooled)

    def _prompt_prediction(self, prompts, noisy_latents, timesteps, target=None, adapter_context=None):
        from extensions_built_in.gen2_trainer.registry import clear_adapter_context

        clear_adapter_context(self.sd.transformer)
        embeds = self.sd.get_prompt_embeds(prompts)
        prediction = self.sd.get_noise_prediction(noisy_latents, timesteps, embeds, adapter_context=adapter_context)
        clear_adapter_context(self.sd.transformer)
        if target is None:
            return prediction
        return torch.nn.functional.mse_loss(prediction.float(), target.float(), reduction="none").flatten(1).mean(1)

    def _activator_prediction(self, prompts, noisy_latents, timesteps, target=None):
        from extensions_built_in.gen2_trainer.registry import clear_adapter_context

        clear_adapter_context(self.sd.transformer)
        base_embeds = self.sd.get_prompt_embeds(prompts)
        features = [self._encode_soft_prompt(prompt) for prompt in prompts]
        embeds = type(base_embeds)(text_embeds=features)
        prediction = self.sd.get_noise_prediction(noisy_latents, timesteps, embeds)
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

    def _literal_prompt(self, prompt: str) -> str:
        literal = ((self.config.get("activator") or {}).get("initialization") or {}).get("literal", "")
        return self.placeholder_contract.replace(prompt, literal)

    def _run_helper_calibration(self, probes):
        calibration = self.gen2_config.phase_a.get("calibration") or {}
        literal_losses = []
        helper_losses = [[] for _ in self.helpers]
        evaluations_per_probe = 1 + len(self.helpers)
        total_evaluations = len(probes) * evaluations_per_probe
        with ToolkitProgressBar(total=total_evaluations, desc="Phase A helper calibration") as progress:
            with torch.no_grad():
                for probe_index, (noisy, timesteps, target, prompt) in enumerate(probes, start=1):
                    literal_prompt = self._literal_prompt(prompt)
                    literal = self._prompt_prediction([literal_prompt], noisy, timesteps, target).detach()[0]
                    literal_losses.append(literal)
                    progress.update(1)
                    progress.set_postfix(probe=f"{probe_index}/{len(probes)}", evaluation="literal")
                    for helper_index, helper in enumerate(self.helpers, start=1):
                        helper_prompt = self.placeholder_contract.replace(prompt, helper["replacement"])
                        helper_losses[helper_index - 1].append(self._prompt_prediction([helper_prompt], noisy, timesteps, target).detach()[0])
                        progress.update(1)
                        progress.set_postfix(probe=f"{probe_index}/{len(probes)}", evaluation=f"helper {helper_index}/{len(self.helpers)}")
        result = calibrate_helper_losses(
            torch.stack(literal_losses),
            torch.stack([torch.stack(losses) for losses in helper_losses]),
            float(calibration.get("temperature", 0.1)),
            float(calibration.get("min_gain", 0.0)),
            float(calibration.get("min_positive_fraction", 0.6)),
        )
        self._helper_calibration = {
            "weights": result.weights.detach(),
            "reliable_mask": result.reliable_mask.detach(),
            "median_gains": result.median_gains.detach(),
            "literal_losses": torch.stack(literal_losses),
            "helper_losses": torch.stack([torch.stack(losses) for losses in helper_losses]),
        }
        return result

    def _competence_probe(self, probes):
        reference = self._helper_calibration
        if reference is None or not probes:
            return 0.0
        literal_losses = []
        helper_losses = [[] for _ in self.helpers]
        activator_losses = []
        for noisy, timesteps, target, prompt in probes:
            with torch.no_grad():
                literal_losses.append(self._prompt_prediction([self._literal_prompt(prompt)], noisy, timesteps, target)[0])
                activator_losses.append(self._activator_prediction([prompt], noisy, timesteps, target)[0])
                for helper_index, helper in enumerate(self.helpers):
                    helper_prompt = self.placeholder_contract.replace(prompt, helper["replacement"])
                    helper_losses[helper_index].append(self._prompt_prediction([helper_prompt], noisy, timesteps, target)[0])
        literal = torch.stack(literal_losses)
        helpers = torch.stack([torch.stack(losses) for losses in helper_losses])
        weighted_helper = (helpers * reference["weights"].to(helpers.device).unsqueeze(1)).sum(dim=0)
        competence, _ = robust_competence(literal, torch.stack(activator_losses), weighted_helper)
        return float(competence.item())

    def _run_phase_a(self) -> None:
        steps = self._phase_steps("a")
        if steps <= 0:
            return
        if self.data_loader is None:
            raise RuntimeError("Phase A requires the real dataset dataloader")
        self._initialize_soft_tokens_from_literal_trigger()
        settings = self.gen2_config.phase_a
        curriculum = settings.get("curriculum") or {}
        optimizer = self.optimizer
        if optimizer is None:
            raise RuntimeError("Phase A optimizer was not prepared")
        iterator = iter(self.data_loader)
        calibration_settings = settings.get("calibration") or {}
        calibration_seed = int(calibration_settings.get("seed", 1234))
        torch.random.manual_seed(calibration_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(calibration_seed)
        probe_count = int(calibration_settings.get("probe_count", 8))
        heldout_count = int((settings.get("calibration") or {}).get("heldout_count", 4))
        probes, iterator = self._collect_phase_a_probes(probe_count + heldout_count, iterator)
        self._phase_a_probes = probes[:probe_count]
        self._phase_a_heldout_probes = probes[probe_count:]
        calibration_result = self._run_helper_calibration(self._phase_a_probes)
        if self._phase_a_heldout_probes:
            self._heldout_competence = self._competence_probe(self._phase_a_heldout_probes)
        physical_batch = int(settings.get("batch_size", 1))
        microbatch_size = int(curriculum.get("microbatch_size", physical_batch))
        effective_batch = int(curriculum.get("effective_batch_size", physical_batch))
        if effective_batch % microbatch_size != 0:
            raise ValueError("Phase A effective batch must be divisible by the microbatch size")
        accumulation = max(1, effective_batch // microbatch_size)
        if physical_batch != microbatch_size:
            raise ValueError("Phase A physical batch and microbatch must match the loaded dataloader batch contract")
        self.phase_a_root.mkdir(parents=True, exist_ok=True)
        with ToolkitProgressBar(total=steps, desc="Phase A curriculum") as progress:
            for step in range(steps):
                self._assert_qwen_frozen()
                optimizer.zero_grad(set_to_none=True)
                dataset_loss = torch.zeros((), device=self.device_torch)
                semantic_loss = torch.zeros_like(dataset_loss)
                disturbance_loss = torch.zeros_like(dataset_loss)
                preserve_loss = torch.zeros_like(dataset_loss)
                trust_region_loss = torch.zeros_like(dataset_loss)
                cross_content_deltas = []
                disturbance_alpha = []
                for _ in range(accumulation):
                    batch, iterator = self._next_batch(iterator, self.data_loader)
                    noisy, noise, timesteps, prompts, _ = self._prepare_real_batch(batch, "a")
                    target = self.sd.get_loss_target(noise=noise, batch=batch)
                    activator_prediction = self._activator_prediction(prompts, noisy, timesteps)
                    dataset_loss = dataset_loss + torch.nn.functional.mse_loss(activator_prediction.float(), target.float()) / accumulation
                    with torch.no_grad():
                        literal_prompts = [self._literal_prompt(prompt) for prompt in prompts]
                        literal_prediction = self._prompt_prediction(literal_prompts, noisy, timesteps)
                        helper_predictions = []
                        helper_features = []
                        for helper in self.helpers:
                            helper_prompts = [self.placeholder_contract.replace(prompt, helper["replacement"]) for prompt in prompts]
                            helper_predictions.append(self._prompt_prediction(helper_prompts, noisy, timesteps))
                            helper_features.append(self._native_span_pooled(prompts, [helper["replacement"] for _ in prompts]))
                        helper_prediction = torch.stack(helper_predictions)
                        weights = self._helper_calibration["weights"].to(helper_prediction.device, helper_prediction.dtype)
                        weighted_helper_prediction = (helper_prediction * weights.view(-1, 1, 1, 1, 1)).sum(dim=0)
                        literal_features = self._native_span_pooled(prompts, [((self.config.get("activator") or {}).get("initialization") or {}).get("literal", "") for _ in prompts])
                        weighted_helper_features = (torch.stack(helper_features) * weights.view(-1, 1, 1)).sum(dim=0)
                    activator_encoded = [self._encode_soft_prompt(prompt, return_pooled=True) for prompt in prompts]
                    activator_features = torch.stack([item[1].float() for item in activator_encoded])
                    activator_ordinary_features = torch.stack([item[2].float() for item in activator_encoded])
                    semantic_loss = semantic_loss + semantic_direction_loss(activator_features - literal_features, weighted_helper_features - literal_features) / accumulation
                    alpha, beta, valid = disturbance_projection(activator_prediction - literal_prediction, weighted_helper_prediction - literal_prediction)
                    disturbance_loss = disturbance_loss + (beta[valid].mean() if torch.any(valid) else activator_prediction.sum() * 0.0) / accumulation
                    delta = activator_features - literal_features.detach()
                    cross_content_deltas.append(delta)
                    literal_ordinary_features = self._native_ordinary_pooled(prompts, [((self.config.get("activator") or {}).get("initialization") or {}).get("literal", "") for _ in prompts])
                    preserve_loss = preserve_loss + (activator_ordinary_features - literal_ordinary_features.detach()).square().mean() / accumulation
                    trust_region_loss = trust_region_loss + (self.soft_tokens.A.float() - self._soft_tokens_initial.float()).square().mean() / accumulation
                    if torch.any(valid):
                        disturbance_alpha.append(alpha[valid].mean().detach())
                    if hasattr(batch, "cleanup"):
                        batch.cleanup()
                evaluate_competence = (step + 1) % int(curriculum.get("competence_eval_every", 10)) == 0
                competence = self._competence_probe(self._phase_a_probes) if evaluate_competence else 0.0
                heldout_competence = self._competence_probe(self._phase_a_heldout_probes) if evaluate_competence and self._phase_a_heldout_probes else 0.0
                if competence >= float(curriculum.get("release_threshold", 0.75)):
                    self._helper_release_streak += 1
                elif competence != 0.0:
                    self._helper_release_streak = 0
                if self._helper_release_streak >= int(curriculum.get("release_consecutive", 3)):
                    self._helper_latched = True
                helper_weight = 0.0 if self._helper_latched else smooth_helper_weight(competence, float(curriculum.get("semantic_weight", 0.25)), float(curriculum.get("decay_threshold", 0.25)), float(curriculum.get("release_threshold", 0.75)))
                if len(cross_content_deltas) > 1:
                    stacked_deltas = torch.cat(cross_content_deltas, dim=0)
                    mean_delta = stacked_deltas.mean(dim=0, keepdim=True)
                    cross_content_loss = (1.0 - torch.nn.functional.cosine_similarity(stacked_deltas, mean_delta, dim=-1)).mean()
                else:
                    cross_content_loss = dataset_loss * 0.0
                self._last_phase_a_diagnostics = {
                    "dataset_loss": float(dataset_loss.detach().item()),
                    "semantic_loss": float(semantic_loss.detach().item()),
                    "disturbance_loss": float(disturbance_loss.detach().item()),
                    "content_preserve_loss": float(preserve_loss.detach().item()),
                    "cross_content_loss": float(cross_content_loss.detach().item()),
                    "trust_region_loss": float(trust_region_loss.detach().item()),
                    "disturbance_alpha": float(torch.stack(disturbance_alpha).mean().item()) if disturbance_alpha else 0.0,
                    "competence": float(competence),
                    "helper_weight": float(helper_weight),
                    "helper_latched": float(self._helper_latched),
                    "helper_release_streak": float(self._helper_release_streak),
                }
                self._last_phase_a_diagnostics["heldout_competence"] = float(heldout_competence)
                self._phase_a_history.append({"step": step + 1, **self._last_phase_a_diagnostics})
                calibration_snapshot = None
                if self._helper_calibration is not None:
                    calibration_snapshot = {key: value.detach().cpu().tolist() if torch.is_tensor(value) else value for key, value in self._helper_calibration.items()}
                if self.accelerator.is_main_process:
                    (self.phase_a_root / "diagnostics.json").write_text(json.dumps({"calibration": calibration_snapshot, "heldout_competence": self._heldout_competence, "history": self._phase_a_history}, ensure_ascii=False, indent=2), encoding="utf-8")
                total_loss = (
                    float(curriculum.get("dataset_weight", 1.0)) * dataset_loss
                    + helper_weight * semantic_loss
                    + float(curriculum.get("disturbance_weight", 0.01)) * disturbance_loss
                    + float(curriculum.get("content_preserve_weight", 0.05)) * preserve_loss
                    + float(curriculum.get("cross_content_weight", 0.01)) * cross_content_loss
                    + float(curriculum.get("trust_region_weight", 0.001)) * trust_region_loss
                )
                self.accelerator.backward(total_loss)
                trainable = [self.soft_tokens.A]
                if self.trigger_local_adapter is not None:
                    trainable.extend(self.trigger_local_adapter.parameters())
                torch.nn.utils.clip_grad_norm_(trainable, self.train_config.max_grad_norm)
                optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                current_step = step + 1
                if self.accelerator.is_main_process:
                    self.logger.log({f"gen2/phase_a/{key}": value for key, value in self._last_phase_a_diagnostics.items()})
                    self.logger.commit()
                save_every = int(self.config.get("save", {}).get("save_every", 0))
                if save_every and current_step % save_every == 0:
                    tensors = {"A": self.soft_tokens.A.detach().cpu()}
                    if self.trigger_local_adapter is not None:
                        tensors.update({f"te_adapter.{key}": value.detach().cpu() for key, value in self.trigger_local_adapter.state_dict().items()})
                    self._phase_checkpoint("a", current_step, tensors, {"phase": "a", "step": current_step, "helper_latched": self._helper_latched, "competence": competence, "heldout_competence": heldout_competence, "diagnostics": self._last_phase_a_diagnostics, "history_length": len(self._phase_a_history)}, optimizer)
                progress.set_postfix(step=current_step, total=steps, dataset=f"{dataset_loss.item():.4g}", semantic=f"{semantic_loss.item():.4g}", disturbance=f"{disturbance_loss.item():.4g}", competence=f"{competence:.3f}", latch=int(self._helper_latched))
                progress.update(1)
        tail_steps = int(curriculum.get("independent_tail_steps", 0)) if self._helper_latched else 0
        with ToolkitProgressBar(total=tail_steps, desc="Phase A dataset-only tail") as tail_progress:
            for _ in range(tail_steps):
                optimizer.zero_grad(set_to_none=True)
                tail_loss = torch.zeros((), device=self.device_torch)
                last_batch = None
                for _ in range(accumulation):
                    batch, iterator = self._next_batch(iterator, self.data_loader)
                    last_batch = batch
                    noisy, noise, timesteps, prompts, _ = self._prepare_real_batch(batch, "a")
                    target = self.sd.get_loss_target(noise=noise, batch=batch)
                    tail_loss = tail_loss + self._activator_prediction(prompts, noisy, timesteps, target).mean() / accumulation
                    if hasattr(batch, "cleanup"):
                        batch.cleanup()
                self.accelerator.backward(tail_loss)
                trainable = [self.soft_tokens.A]
                if self.trigger_local_adapter is not None:
                    trainable.extend(self.trigger_local_adapter.parameters())
                torch.nn.utils.clip_grad_norm_(trainable, self.train_config.max_grad_norm)
                optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                tail_progress.set_postfix(loss=f"{tail_loss.item():.4g}")
                tail_progress.update(1)
        tensors = {"A": self.soft_tokens.A}
        metadata = {"artifact": "gen2_activator", "schema_version": 3, "initializer": "literal_trigger_resampled", "literal": ((self.config.get("activator") or {}).get("initialization") or {}).get("literal"), "placeholder": self.placeholder_contract.placeholder, "tokens": self.soft_tokens.tokens, "dimension": self.soft_tokens.dimension, "trigger_local_adapter": self.trigger_local_adapter is not None, "trigger_local_adapter_rank": self.trigger_local_adapter.rank if self.trigger_local_adapter is not None else 0, "trigger_local_adapter_alpha": self.trigger_local_adapter.alpha if self.trigger_local_adapter is not None else 0.0, "helper_latched": self._helper_latched, "helper_release_streak": self._helper_release_streak, "heldout_competence": self._heldout_competence, "phase_a_diagnostics": self._last_phase_a_diagnostics}
        if self._helper_calibration is not None:
            metadata["helper_calibration"] = {key: value.detach().cpu().tolist() if torch.is_tensor(value) else value for key, value in self._helper_calibration.items()}
        if self.trigger_local_adapter is not None:
            tensors.update({f"te_adapter.{key}": value for key, value in self.trigger_local_adapter.state_dict().items()})
        (self.phase_a_root / "diagnostics.json").write_text(json.dumps({"calibration": metadata.get("helper_calibration"), "heldout_competence": self._heldout_competence, "history": self._phase_a_history}, ensure_ascii=False, indent=2), encoding="utf-8")
        save_tensor_artifact(self.phase_a_root / "activator.safetensors", tensors, metadata)

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
        if int(metadata.get("schema_version", -1)) != 3:
            raise ValueError("activator artifact schema_version must be 3")
        expected_literal = ((self.config.get("activator") or {}).get("initialization") or {}).get("literal")
        if metadata.get("literal") != expected_literal:
            raise ValueError("activator artifact literal bootstrap does not match the current configuration")
        if metadata.get("initializer") != "literal_trigger_resampled":
            raise ValueError("activator artifact was not initialized from the literal trigger")
        if metadata.get("placeholder") != self.placeholder_contract.placeholder:
            raise ValueError("activator artifact placeholder does not match the current contract")
        activator = tensors.get("A")
        if activator is None:
            raise ValueError("activator artifact must contain tensor A")
        if tuple(activator.shape) != tuple(self.soft_tokens.A.shape):
            raise ValueError("activator artifact shape does not match configured soft tokens")
        if int(metadata.get("tokens", -1)) != self.soft_tokens.tokens or int(metadata.get("dimension", -1)) != self.soft_tokens.dimension:
            raise ValueError("activator artifact dimensions do not match the configured soft token bank")
        self.soft_tokens.A.data.copy_(activator.to(self.soft_tokens.A))
        artifact_has_adapter = bool(metadata.get("trigger_local_adapter", False))
        config_has_adapter = self.trigger_local_adapter is not None
        if artifact_has_adapter != config_has_adapter:
            raise ValueError("activator artifact trigger-local adapter setting does not match the current configuration")
        if config_has_adapter:
            adapter_state = {key.removeprefix("te_adapter."): value for key, value in tensors.items() if key.startswith("te_adapter.")}
            if not adapter_state:
                raise ValueError("activator artifact is missing trigger-local adapter state")
            self.trigger_local_adapter.load_state_dict(adapter_state, strict=True)

    def _restore_phase_b_checkpoint(self, optimizer) -> int:
        if self.gen2_config.mode != "resume_phase_b":
            return 0
        candidates = [item for item in self.phase_b_root.iterdir() if item.is_dir() and item.name.startswith("step_")]
        if not candidates:
            raise FileNotFoundError("resume_phase_b requires a phase_b/step_<step> checkpoint")
        checkpoint = max(candidates, key=lambda item: int(item.name.removeprefix("step_")))
        tensors, metadata = load_phase_checkpoint(checkpoint, optimizer=optimizer, scheduler=self.lr_scheduler)
        self._adapter_bank.load_state_dict(tensors, strict=True)
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
            self._phase_b_probes = self._phase_a_heldout_probes or self._phase_a_probes
            if not self._phase_b_probes:
                self._phase_b_probes, iterator = self._collect_phase_a_probes(4, iterator)
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
                prediction = self.sd.get_noise_prediction(
                    noisy_latents,
                    timesteps,
                    embeds,
                    adapter_context=AdapterRuntimeContext(timesteps.float() / 1000.0, style_gate, 1.0),
                )
                if not prediction.requires_grad:
                    raise RuntimeError("Phase B prediction has no grad_fn; adapter parameters are not connected")
                fm = torch.nn.functional.mse_loss(prediction.float(), target.float())
                with torch.no_grad():
                    probe_on_losses = []
                    probe_off_losses = []
                    probe_delta_norms = []
                    from extensions_built_in.gen2_trainer.registry import clear_adapter_context
                    for probe_noisy, probe_timesteps, probe_target, probe_prompt in self._phase_b_probes:
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
                save_every = int(self.config.get("save", {}).get("save_every", 0))
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
            len(self.gen2_config.validation_prompts),
            self.gen2_config.seeds,
            sample.get("unconditional_scale_sweep") or [0.0],
            include_sweep=True,
        )
        if not cases:
            raise RuntimeError("Gen2 validation matrix is empty")
        self.validation_cases = cases
        output_root = self.gen2_root / "samples" / f"step_{self._phase_steps('b'):06d}"
        output_root.mkdir(parents=True, exist_ok=True)
        diagnostics_path = output_root / "conditioning_diagnostics.json"
        diagnostics_records: list[dict[str, Any]] = []
        progress = ToolkitProgressBar(total=len(cases), desc=f"Gen2 official CFG validation ({len(cases)} cases)")
        try:
            for case_index, case in enumerate(cases, start=1):
                if case.eta_u > 0 and self.official_unconditional is None:
                    raise RuntimeError("positive eta_u requires official unconditional transformer")
                raw_prompt = self.gen2_config.validation_prompts[case.prompt_index]
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
                if case.conditioning_mode == "native_helper":
                    materialized = self.placeholder_contract.replace(raw_prompt, self.helpers[case.helper_index]["replacement"])
                    embeds = self.sd.get_prompt_embeds([materialized])
                    case_diagnostics["materialized_helper_id"] = self.helpers[case.helper_index]["id"]
                    case_diagnostics["text_length"] = int(embeds.text_embeds[0].shape[0])
                    case_diagnostics["text_feature_norm_mean"] = float(embeds.text_embeds[0].float().norm(dim=-1).mean().item())
                    case_diagnostics["text_feature_norm_max"] = float(embeds.text_embeds[0].float().norm(dim=-1).max().item())
                elif case.conditioning_mode == "soft_tokens":
                    embeds = type(self.sd.get_prompt_embeds([raw_prompt]))(
                        text_embeds=[self._encode_soft_prompt(raw_prompt, case_diagnostics).detach()]
                    )
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
                name = f"{case.conditioning_mode}_adapter_{'on' if case.style_gate and case.eta_c else 'off'}_eta_u_{case.eta_u:g}"
                images[0].save(case_root / f"{name}.png")
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
        tokens = int((self.config.get("activator") or {}).get("tokens", 8))
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
        self.soft_tokens.A.data.copy_(initialization.to(self.soft_tokens.A.device, self.soft_tokens.A.dtype))
        self._soft_tokens_initial = self.soft_tokens.A.detach().clone()
        if self.trigger_local_adapter is not None:
            nn.init.zeros_(self.trigger_local_adapter.up.weight)
            self.trigger_local_adapter.zero_reference = tuple(parameter.detach().clone() for parameter in self.trigger_local_adapter.parameters())

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
