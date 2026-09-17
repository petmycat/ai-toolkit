"""Frozen native Ideogram backend for the standalone v2 token activator."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch.utils.checkpoint import checkpoint

from .conditioning import TokenBank, replace_input_embeddings
from .text import NativePromptCompiler, CompiledPrompt

EXPECTED_TAPS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)


@dataclass
class Conditioning:
    features: list[torch.Tensor]
    metadata: list[dict]

    @property
    def text_embeds(self):
        return self.features


def qwen_features(text_encoder, inputs_embeds, attention_mask, positions, *,
                  gradient_checkpointing=True, taps=EXPECTED_TAPS, causal_mask_factory=None):
    """Differentiable sibling of native extraction, retaining ALL text features."""
    if causal_mask_factory is None:
        from transformers.masking_utils import create_causal_mask
        causal_mask_factory = create_causal_mask
    language = text_encoder.language_model
    position_ids = positions[None, ...].expand(4, positions.shape[0], -1)
    text_positions, mrope_positions = position_ids[0], position_ids[1:]
    mask = causal_mask_factory(config=language.config, inputs_embeds=inputs_embeds,
        attention_mask=attention_mask, past_key_values=None, position_ids=text_positions)
    rotary = language.rotary_emb(inputs_embeds, mrope_positions)
    hidden, captured = inputs_embeds, {}
    for index, layer in enumerate(language.layers):
        def forward(value, layer=layer):
            output = layer(value, attention_mask=mask, position_ids=text_positions,
                           past_key_values=None, position_embeddings=rotary)
            if not torch.is_tensor(output):
                raise TypeError("Native Qwen decoder-output contract changed")
            return output
        hidden = checkpoint(forward, hidden, use_reentrant=False) if gradient_checkpointing and torch.is_grad_enabled() else forward(hidden)
        if index in taps:
            captured[index] = hidden
    if set(taps)-captured.keys():
        raise ValueError("Loaded native Qwen decoder does not contain every required activation tap")
    # Same native c*K+j ordering: (K,B,L,d) -> (B,L,d,K).
    packed = torch.stack([captured[index] for index in taps], 0).permute(1, 2, 3, 0)
    packed = packed.reshape(*inputs_embeds.shape[:2], -1)
    return packed*attention_mask.to(packed.dtype).unsqueeze(-1)


class V2Backend:
    def __init__(self, model, config):
        from extensions_built_in.diffusion_models.ideogram4.src.pipeline import (
            get_qwen3_vl_features, pad_text_features, predict_velocity)
        from extensions_built_in.diffusion_models.ideogram4.src.transformer import QWEN3_VL_ACTIVATION_LAYERS
        if getattr(model, "arch", None) != "ideogram4":
            raise ValueError("V2 currently requires the native Ideogram4Model")
        if tuple(QWEN3_VL_ACTIVATION_LAYERS) != EXPECTED_TAPS:
            raise ValueError("Native Qwen feature taps changed; review backend compatibility")
        if getattr(model, "network", None) is not None:
            raise ValueError("V2 standalone activator cannot load a personalization diffusion network")
        if getattr(model, "unconditional_lora", None) is not None:
            raise ValueError("V2 requires the original unconditional transformer, not a correction adapter")
        self.unconditional = getattr(model, "unconditional_transformer", None)
        if self.unconditional is None or self.unconditional is model.transformer:
            raise ValueError("V2 requires a separately loaded original unconditional transformer")
        self.model, self.config = model, config
        self.trigger_word = config["trigger_word"]
        self.native_features, self.pad_features, self.velocity = get_qwen3_vl_features, pad_text_features, predict_velocity
        self.encoder_checkpointing = config["gen2"]["execution"]["encoder_gradient_checkpointing"]
        for component in (model.transformer, model.text_encoder, model.vae, self.unconditional):
            component.eval().requires_grad_(False)
        self.unconditional.disable_gradient_checkpointing()
        self.unconditional.set_attention_backend(config["gen2"]["execution"]["dit_attention_backend"])
        # Native low_vram loading can leave complete components on CPU. Move
        # them before ANY graph exists, using the same native placement seam.
        self._ensure_device(model.text_encoder)
        self._ensure_device(model.transformer)
        if config["train"]["gradient_checkpointing"]:
            model.transformer.enable_gradient_checkpointing()
        else:
            model.transformer.disable_gradient_checkpointing()
        model.transformer.set_attention_backend(config["gen2"]["execution"]["dit_attention_backend"])
        options = config["gen2"]["conditioning"]
        self.compiler = NativePromptCompiler(model.tokenizer, config["trigger_word"], options["num_tokens"],
            config["model"]["model_kwargs"]["max_text_length"], options["overflow_policy"])
        self.tokens = TokenBank(model.text_encoder.language_model.embed_tokens, model.tokenizer,
            options["num_tokens"], options["initializer_seed"], options["initialization_sample_size"])
        self.tokens.to(model.device_torch, dtype=torch.float32)
        components = [model.transformer, model.text_encoder, model.vae, self.unconditional]
        self._frozen = tuple(parameter for component in components for parameter in component.parameters())

    def _ensure_device(self, component):
        device = getattr(component, "device", None)
        if device is None:
            device = next(component.parameters()).device
        if torch.device(device).type == "cpu" and torch.device(self.model.device_torch).type != "cpu":
            component.to(self.model.device_torch)

    def frozen_parameters(self):
        return iter(self._frozen)

    def assert_frozen(self):
        if any(parameter.requires_grad for parameter in self._frozen):
            raise RuntimeError("An original v2 model/vocabulary/unconditional parameter became trainable")
        if self.tokens.E.dtype != torch.float32 or not bool(torch.isfinite(self.tokens.E).all()):
            raise FloatingPointError("V2 token masters must remain finite FP32 values")
        if getattr(self.model, "unconditional_lora", None) is not None:
            raise RuntimeError("An unsupported correction adapter appeared in the v2 model")

    def encode(self, prompts, mode="learned", gradients=False, compiled=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        if gradients and not torch.is_grad_enabled():
            raise RuntimeError("V2 activator encoding entered an outer no_grad context")
        if gradients and mode != "learned":
            raise ValueError("Only learned activator conditioning is a training gradient path")
        if compiled is None:
            phrase = self.config["gen2"].get("evaluation", {}).get("named_phrase")
            compiled = [self.compiler.compile(prompt, mode=mode, named_phrase=phrase) for prompt in prompts]
        elif isinstance(compiled, CompiledPrompt):
            compiled = [compiled]
        if len(compiled) != len(prompts):
            raise ValueError("Compiled prompt batch size differs from input prompt count")
        self._ensure_device(self.model.text_encoder)
        embedding = self.model.text_encoder.language_model.embed_tokens
        features, metadata = [], []
        with torch.set_grad_enabled(gradients):
            for prompt, item in zip(prompts, compiled):
                if item.mode != mode or item.metadata["original_caption"] != prompt:
                    raise ValueError("Compiled conditioning differs from the requested caption/mode")
                ids = torch.tensor([item.ids], device=embedding.weight.device, dtype=torch.long)
                mask = torch.ones_like(ids)
                positions = mask.cumsum(-1)-1
                if item.soft_positions:
                    inputs = replace_input_embeddings(embedding, ids, item.soft_positions,
                                                      item.soft_bank_indices, self.tokens(mode))
                    value = qwen_features(self.model.text_encoder, inputs, mask, positions,
                                          gradient_checkpointing=self.encoder_checkpointing)
                else:
                    # Marker-absent and plain benchmark routes are EXACT native
                    # extraction, using the unchanged original vocabulary.
                    value = self.native_features(self.model.text_encoder, ids, mask, positions)
                features.append(value[0].to(self.model.torch_dtype))
                metadata.append(dict(item.metadata))
        return Conditioning(features, metadata)

    def predict(self, latents, tau, conditioning):
        self._ensure_device(self.model.transformer)
        features, mask = self.pad_features(conditioning.features, self.model.device_torch, self.model.torch_dtype)
        return self.velocity(self.model.transformer, latents.to(self.model.device_torch, self.model.torch_dtype),
                             tau, features, mask)

    @torch.no_grad()
    def predict_unconditional(self, latents, tau):
        self._ensure_device(self.unconditional)
        dimension = self.tokens.E.shape[1]*len(EXPECTED_TAPS)
        features = torch.empty(latents.shape[0], 0, dimension, device=self.model.device_torch, dtype=self.model.torch_dtype)
        mask = torch.empty(latents.shape[0], 0, device=self.model.device_torch, dtype=torch.long)
        return self.velocity(self.unconditional, latents.to(self.model.device_torch, self.model.torch_dtype),
                             tau, features, mask)

    @torch.no_grad()
    def verify_native(self, prompts, atol=.001, rtol=.01):
        records = []
        embedding = self.model.text_encoder.language_model.embed_tokens
        self._ensure_device(self.model.text_encoder)
        for index, prompt in enumerate(prompts):
            item = self.compiler.compile(prompt, mode="base")
            ids = torch.tensor([item.ids], device=embedding.weight.device, dtype=torch.long)
            mask = torch.ones_like(ids); positions = mask.cumsum(-1)-1
            native = self.native_features(self.model.text_encoder, ids, mask, positions)
            repeat = self.native_features(self.model.text_encoder, ids, mask, positions)
            sibling = qwen_features(self.model.text_encoder, embedding(ids), mask, positions, gradient_checkpointing=False)
            for comparison, value in (("native_repeat_baseline", repeat), ("differentiable_native_parity", sibling)):
                difference = (native.float()-value.float()).abs()
                passed = torch.allclose(native, value, atol=atol, rtol=rtol)
                records.append({"example_index": index, "comparison": comparison,
                    "max_abs": float(difference.max()), "difference_rms": float(difference.square().mean().sqrt()),
                    "passed": passed, "atol": atol, "rtol": rtol, "conditioning": item.metadata})
        if any(not record["passed"] for record in records):
            error = RuntimeError("V2 frozen native feature-path comparison failed")
            error.records = records
            raise error
        return records

    def probe_conditioning(self, caption):
        """Isolated ordinary-token-only gradient probe; never modifies .grad."""
        item = self.compiler.compile(caption, require_trigger=True)
        ordinary = [index for index in item.metadata["ordinary_caption_positions"] if index > min(item.soft_positions)]
        if not ordinary:
            raise ValueError("Conditioning gradient probe needs ordinary caption tokens after an activator")
        previous = self.tokens.E.requires_grad
        self.tokens.E.requires_grad_(True)
        try:
            with torch.enable_grad():
                encoded = self.encode([caption], gradients=True, compiled=[item])
                features = encoded.features[0][ordinary].float()
                weights = torch.linspace(-1., 1., features.shape[-1], device=features.device)
                scalar = (features*weights).mean()
                gradient, = torch.autograd.grad(scalar, (self.tokens.E,))
            return {"probe": "ordinary_caption_features_only", "conditioning": item.metadata,
                    "ordinary_positions": ordinary, "excluded_soft_positions": item.soft_positions,
                    "finite_features": bool(torch.isfinite(features).all()),
                    "gradient_finite": bool(torch.isfinite(gradient).all()),
                    "gradient_norm": float(gradient.float().norm()), "gradient_nonzero": bool((gradient != 0).any())}
        finally:
            self.tokens.E.requires_grad_(previous)
