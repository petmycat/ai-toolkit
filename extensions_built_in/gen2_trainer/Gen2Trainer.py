from __future__ import annotations

import copy
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.progress_bar import ToolkitProgressBar

from .activator import (
    PlaceholderContract,
    SoftTokenBank,
    encode_qwen_inputs_embeds,
    initialize_from_helper_embeddings,
    normalize_inline_helpers,
    replace_token_spans_with_soft_tokens,
)
from .checkpoint import load_phase_checkpoint, save_phase_checkpoint
from .config import Gen2RuntimeConfig, validate_gen2_config
from toolkit.optimizer import get_optimizer
from toolkit.scheduler import get_lr_scheduler
from .registry import AdapterRuntimeContext, enable_adapter_training, install_ideogram_adapters
from .sampling import build_validation_matrix, make_flowmatch_noisy_latents, sample_stratified_timesteps
from .temporal_rank_field import time_smooth_regularizer, time_mean_diagnostic


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
        self.adapter = torch.nn.ModuleDict()
        self.official_unconditional = None
        self.helpers = normalize_inline_helpers(self.gen2_config.helpers)
        self._prepared = False
        self._adapter_bank = None
        self._timestep_bin_orders: dict[str, torch.Tensor] = {}
        self._timestep_bin_cursors: dict[str, int] = {"a": 0, "b": 0}

    def _phase_steps(self, phase: str) -> int:
        settings = self.gen2_config.phase_a if phase == "a" else self.gen2_config.phase_b
        if not settings.get("enabled", True):
            return 0
        return int(settings.get("steps", 0))

    def _write_manifest(self, completed_phase: str) -> None:
        self.gen2_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "trainer": "gen2_trainer",
            "completed_phase": completed_phase,
            "mode": self.gen2_config.mode,
            "placeholder": self.gen2_config.placeholder,
            "official_unconditional_required": True,
            "training_scope": "conditional_only",
        }
        (self.gen2_root / "gen2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def hook_after_model_load(self):
        if self.sd.arch != "ideogram4":
            raise ValueError("Gen2 only supports the Ideogram 4 model")
        self.sd.model.requires_grad_(False)
        self.sd.text_encoder.requires_grad_(False)
        self.sd.vae.requires_grad_(False)
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

    def hook_before_train_loop(self):
        self._preflight_loaded_datasets()
        if self.accelerator.is_main_process:
            self.logger.start()
        self.prepare_accelerator()

    def prepare_accelerator(self):
        self.accelerator.even_batches = False
        self.sd.vae = self.accelerator.prepare(self.sd.vae)
        self.sd.unet = self.accelerator.prepare(self.sd.unet)
        self.soft_tokens, self.optimizer = self.accelerator.prepare(self.soft_tokens, self.optimizer)
        self._prepared = True
        if self.lr_scheduler is not None:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
        self.modules_being_trained = [self.soft_tokens]

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
        return params

    def hook_train_loop(self, batches):
        batch_list = batches if isinstance(batches, list) else [batches]
        self.optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=self.device_torch)
        for batch in batch_list:
            noisy_latents, noise, timesteps, prompts, _ = self._prepare_real_batch(batch, "a")
            activator_features = [self._encode_soft_prompt(prompt) for prompt in prompts]
            conditioned = type(self.sd.get_prompt_embeds(prompts))(text_embeds=activator_features)
            prediction = self.sd.get_noise_prediction(noisy_latents, timesteps, conditioned)
            target = self.sd.get_loss_target(noise=noise, batch=batch)
            loss = torch.nn.functional.mse_loss(prediction.float(), target.float()) / len(batch_list)
            self.accelerator.backward(loss)
            total = total + loss.detach()
        if not self.is_grad_accumulation_step:
            self.accelerator.clip_grad_norm_([self.soft_tokens.A], self.train_config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
        return OrderedDict(loss=total.item())

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

    def _encode_soft_prompt(self, raw_caption: str):
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
        expanded, _ = replace_token_spans_with_soft_tokens(ordinary, spans, self.soft_tokens)
        expanded = expanded.unsqueeze(0)
        attention_mask = torch.ones(expanded.shape[:2], device=expanded.device, dtype=torch.long)
        pos_2d = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        return encode_qwen_inputs_embeds(
            self.sd.text_encoder,
            expanded,
            attention_mask,
            pos_2d,
            QWEN3_VL_ACTIVATION_LAYERS,
        )[0].to(self.sd.torch_dtype)

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

    def _run_phase_a(self) -> None:
        steps = self._phase_steps("a")
        if steps <= 0:
            return
        if self.data_loader is None:
            raise RuntimeError("Phase A requires the real dataset dataloader")
        self._initialize_soft_tokens_from_helpers()
        settings = self.gen2_config.phase_a
        optimizer = self.optimizer
        if optimizer is None:
            raise RuntimeError("Phase A optimizer was not prepared")
        iterator = iter(self.data_loader)
        self.phase_a_root.mkdir(parents=True, exist_ok=True)
        with ToolkitProgressBar(total=steps, desc="Phase A-lite") as progress:
            for step in range(steps):
                batch, iterator = self._next_batch(iterator, self.data_loader)
                noisy_latents, noise, timesteps, prompts, _ = self._prepare_real_batch(batch, "a")
                base_embeds = self.sd.get_prompt_embeds(prompts)
                target = self.sd.get_loss_target(noise=noise, batch=batch)
                optimizer.zero_grad(set_to_none=True)
                activator_features = [self._encode_soft_prompt(prompt) for prompt in prompts]
                conditioned = type(base_embeds)(text_embeds=activator_features)
                prediction = self.sd.get_noise_prediction(noisy_latents, timesteps, conditioned)
                loss_target = torch.nn.functional.mse_loss(prediction.float(), target.float())
                loss_margin = torch.zeros_like(loss_target)
                phase_a_settings = self.gen2_config.phase_a
                margin_settings = phase_a_settings.get("helper_margin") or {}
                if margin_settings.get("enabled", False) and (step + 1) % int(margin_settings.get("every", 1)) == 0:
                    helper_prompt = [self.placeholder_contract.replace(prompt, self.helpers[0]["replacement"]) for prompt in prompts]
                    with torch.no_grad():
                        helper_embeds = self.sd.get_prompt_embeds(helper_prompt)
                        helper_prediction = self.sd.get_noise_prediction(noisy_latents, timesteps, helper_embeds)
                        helper_loss = torch.nn.functional.mse_loss(helper_prediction.float(), target.float())
                    margin = float(margin_settings.get("margin", 0.0))
                    if margin_settings.get("mode", "absolute") == "relative":
                        loss_margin = torch.relu((loss_target - helper_loss) / helper_loss.clamp_min(1e-8) + margin)
                    else:
                        loss_margin = torch.relu(loss_target - helper_loss + margin)
                anchor = torch.zeros_like(loss_target)
                anchor_settings = phase_a_settings.get("init_anchor") or {}
                if self._soft_tokens_initial is not None and anchor_settings:
                    fraction = min(1.0, (step + 1) / max(steps, 1))
                    decay_fraction = float(anchor_settings.get("decay_fraction", 0.5))
                    progress_fraction = min(1.0, fraction / max(decay_fraction, 1e-8))
                    weight_start = float(anchor_settings.get("weight_start", 0.0))
                    weight_end = float(anchor_settings.get("weight_end", 0.0))
                    anchor_weight = weight_start + (weight_end - weight_start) * progress_fraction
                    anchor = (self.soft_tokens.A.float() - self._soft_tokens_initial.float()).square().mean() * anchor_weight
                total_loss = loss_target + float(margin_settings.get("weight", 1.0)) * loss_margin + anchor
                self.accelerator.backward(total_loss)
                torch.nn.utils.clip_grad_norm_([self.soft_tokens.A], self.train_config.max_grad_norm)
                optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                current_step = step + 1
                save_every = int(self.config.get("save", {}).get("save_every", 0))
                if save_every and current_step % save_every == 0:
                    self._phase_checkpoint(
                        "a",
                        current_step,
                        {"A": self.soft_tokens.A.detach().cpu()},
                        {"phase": "a", "step": current_step, "helpers": self.helpers},
                        optimizer,
                    )
                progress.set_postfix(step=current_step, total=steps, target=f"{loss_target.item():.4g}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                progress.update(1)
                if hasattr(batch, "cleanup"):
                    batch.cleanup()
        torch.save({"A": self.soft_tokens.A.detach().cpu(), "helpers": self.helpers}, self.phase_a_root / "activator.pt")

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
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        if "A" not in payload:
            raise ValueError("activator artifact must contain A")
        if tuple(payload["A"].shape) != tuple(self.soft_tokens.A.shape):
            raise ValueError("activator artifact shape does not match configured soft tokens")
        self.soft_tokens.A.data.copy_(payload["A"].to(self.soft_tokens.A))

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
        with ToolkitProgressBar(total=steps, desc="Phase B2 TRF-LoRA") as progress:
            for step in range(start_step, steps):
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
                    adapter_context=AdapterRuntimeContext(timesteps.float(), style_gate, 1.0),
                )
                if not prediction.requires_grad:
                    raise RuntimeError("Phase B prediction has no grad_fn; adapter parameters are not connected")
                fm = torch.nn.functional.mse_loss(prediction.float(), target.float())
                smooth = time_smooth_regularizer(self._adapter_bank.temporal_fields())
                loss = fm + float(settings.get("temporal_smooth_weight", 1e-4)) * smooth
                self.accelerator.backward(loss)
                if not any(parameter.grad is not None for parameter in trainable):
                    raise RuntimeError("Phase B backward produced no adapter gradients")
                self.accelerator.clip_grad_norm_(trainable, self.train_config.max_grad_norm)
                optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                current_step = step + 1
                save_every = int(self.config.get("save", {}).get("save_every", 0))
                if save_every and current_step % save_every == 0:
                    self._phase_checkpoint(
                        "b",
                        current_step,
                        {key: value.detach().cpu() for key, value in self._adapter_bank.state_dict().items()},
                        {"phase": "b", "step": current_step, "temporal_mean": time_mean_diagnostic(self._adapter_bank.temporal_fields()).detach().cpu().tolist()},
                        optimizer,
                        scheduler=self.lr_scheduler,
                    )
                progress.set_postfix(step=current_step, total=steps, fm=f"{fm.item():.4g}", time_smooth=f"{smooth.item():.4g}")
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
        for case in cases:
            if case.eta_u > 0 and self.official_unconditional is None:
                raise RuntimeError("positive eta_u requires official unconditional transformer")
            raw_prompt = self.gen2_config.validation_prompts[case.prompt_index]
            if case.helper_off:
                materialized = self.placeholder_contract.replace(raw_prompt, self.helpers[0]["replacement"])
                embeds = self.sd.get_prompt_embeds([materialized])
            else:
                base = self.sd.get_prompt_embeds([raw_prompt])
                embeds = type(base)(text_embeds=[self._encode_soft_prompt(raw_prompt).detach()])
            generator = torch.Generator(device=self.device_torch).manual_seed(case.seed)
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
            )
            case_root = output_root / f"prompt_{case.prompt_index:02d}" / f"seed_{case.seed}" 
            case_root.mkdir(parents=True, exist_ok=True)
            name = "helper_off" if case.helper_off else f"activator_on_eta_u_{case.eta_u:g}"
            images[0].save(case_root / f"{name}.png")
            self.official_unconditional.transformer.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if self.accelerator.is_main_process:
            self.logger.log({"gen2/official_cfg_validation_cases": len(cases)})
            self.logger.commit()

    def run(self) -> None:
        mode = self.gen2_config.mode
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
            self._write_manifest("phase_a")
        else:
            self._load_activator_artifact()
        if mode == "phase_a_only":
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
            self._write_manifest("phase_b")
            self._run_official_cfg_validation()
        self.accelerator.end_training()

    def _initialize_soft_tokens(self) -> None:
        model = self.config.get("model") or {}
        dimension = int(model.get("qwen_hidden_size", 4096))
        tokens = int((self.config.get("activator") or {}).get("tokens", 8))
        self.soft_tokens = SoftTokenBank(tokens, dimension)

    def _initialize_soft_tokens_from_helpers(self) -> None:
        language_model = self.sd.text_encoder.language_model
        embeddings = []
        with torch.no_grad():
            for helper in self.helpers:
                ids = self.sd.tokenizer(
                    helper["replacement"],
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"].to(language_model.embed_tokens.weight.device)
                embeddings.append(language_model.embed_tokens(ids)[0].float())
        initialization = initialize_from_helper_embeddings(embeddings, self.soft_tokens.tokens)
        self.soft_tokens.A.data.copy_(initialization.to(self.soft_tokens.A.device, self.soft_tokens.A.dtype))
        self._soft_tokens_initial = self.soft_tokens.A.detach().clone()

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
