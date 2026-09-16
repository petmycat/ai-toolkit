"""Thin native Ideogram bridge, with differentiable suffix extraction.

Heavy toolkit imports are deliberately lazy: inspecting config and testing the
new math must not load the multi-billion-parameter production backend.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
import weakref

import torch
from torch.utils.checkpoint import checkpoint

from .conditioning import (Conditioning, LearnedTokenBank,
                           encoder_projection_scope, make_masked_lora_class, native_lora_residual,
                           dequantize_projection_input)
from .gates import CubicTimeGates
from .text_preflight import tokenize_caption, require_token_budget

EXPECTED_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
DIFFUSION_PROJECTIONS = ("attention.qkv", "attention.o", "feed_forward.w1",
                         "feed_forward.w2", "feed_forward.w3")
RECOMPUTING = ContextVar("gen2_recomputing", default=False)


class _ReusableContext:
    """Create a fresh scope for each replay of a retained checkpoint graph.

    Non-reentrant checkpointing retains the context returned by context_fn.
    Chunked autograd.grad can enter that same context more than once, and CUDA
    autograd workers do not inherit the calling thread's ContextVar bindings.
    Keep entry stacks local to the execution context, too.
    """
    def __init__(self, factory):
        self.factory = factory
        self._stack = ContextVar(f"gen2_replay_scopes_{id(self)}", default=())

    def __enter__(self):
        scope = self.factory()
        value = scope.__enter__()
        entry = [scope]
        entry.append(self._stack.set((*self._stack.get(), entry)))
        return value

    def __exit__(self, *error):
        scope, token = self._stack.get()[-1]
        # Reset rather than storing an empty stack: autograd worker contexts
        # must not retain a new ContextVar key for every completed graph.
        self._stack.reset(token)
        return scope.__exit__(*error)


def recomputation_context():
    @contextmanager
    def scope():
        token = RECOMPUTING.set(True)
        try:
            yield
        finally:
            RECOMPUTING.reset(token)
    return _ReusableContext(scope)


def pack_activation_taps(selected: list[torch.Tensor], attention_mask: torch.Tensor):
    """Exact native feature coordinate: channel * number_of_taps + tap."""
    stacked = torch.stack(selected, 0).permute(1, 2, 3, 0)
    packed = stacked.reshape(*selected[0].shape[:2], -1)
    return packed*attention_mask.to(packed.dtype).unsqueeze(-1)


def differentiable_qwen_features(text_encoder, inputs_embeds, attention_mask, pos_2d,
                                 suffix_mask, *, adapter_enabled=True,
                                 gradient_checkpointing=False, diagnostic=None,
                                 activation_layers=EXPECTED_ACTIVATION_LAYERS,
                                 causal_mask_factory=None):
    """Native decoder loop, returning R_T explicitly through every checkpoint.

    This sibling intentionally does not call the native @no_grad helper or the
    encoder top-level forward. Native RoPE and causal-mask builders are reused.
    """
    if causal_mask_factory is None:
        from transformers.masking_utils import create_causal_mask
        causal_mask_factory = create_causal_mask
    language_model = text_encoder.language_model
    positions = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
    text_positions, mrope_positions = positions[0], positions[1:]
    causal_mask = causal_mask_factory(
        config=language_model.config, inputs_embeds=inputs_embeds,
        attention_mask=attention_mask, past_key_values=None, position_ids=text_positions)
    position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_positions)
    hidden, captured, regularizers, numerators, denominators = inputs_embeds, {}, [], [], []
    for layer_id, decoder in enumerate(language_model.layers):
        # Defaults bind the current layer and immutable mask to recomputation.
        def layer_forward(h, decoder=decoder, layer_id=layer_id):
            callback = None if RECOMPUTING.get() else diagnostic
            with encoder_projection_scope(suffix_mask, adapter_enabled, callback) as scope:
                result = decoder(h, attention_mask=causal_mask, position_ids=text_positions,
                                 past_key_values=None, position_embeddings=position_embeddings)
                if not torch.is_tensor(result):
                    raise TypeError("Native Qwen decoder output contract changed; expected Tensor")
                if adapter_enabled and bool(suffix_mask.any()):
                    if scope.result is None:
                        raise RuntimeError(f"Qwen layer {layer_id} did not execute its masked down_proj")
                    rt, numerator, denominator = scope.result
                else:
                    rt = result.new_zeros(result.shape[0], dtype=torch.float32)
                    numerator = denominator = result.new_empty(0, dtype=torch.float32)
                # Only reduced suffix samples retained, never full layer graphs.
                return result, rt, numerator.detach(), denominator.detach()
        if gradient_checkpointing and torch.is_grad_enabled():
            hidden, rt, numerator, denominator = checkpoint(
                layer_forward, hidden, use_reentrant=False,
                context_fn=lambda: (nullcontext(), recomputation_context()))
        else:
            hidden, rt, numerator, denominator = layer_forward(hidden)
        regularizers.append(rt)
        numerators.append(numerator)
        denominators.append(denominator)
        if layer_id in activation_layers:
            captured[layer_id] = hidden
    missing = set(activation_layers)-set(captured)
    if missing:
        raise ValueError(f"Missing Qwen decoder-output activation taps: {sorted(missing)}")
    features = pack_activation_taps([captured[i] for i in activation_layers], attention_mask)
    return features, torch.stack(regularizers).mean(0), torch.cat(numerators), torch.cat(denominators)


@dataclass(frozen=True)
class DiffusionBranch:
    tau: torch.Tensor
    gate_values: torch.Tensor
    enabled: bool
    strength: float
    unconditional: bool
    name: str
    observed: set = field(default_factory=set, compare=False)
    started_components: set = field(default_factory=set, compare=False)


def make_gated_lora_class(native_lora_class, branch_variable, diagnostic_variable):
    class GatedDiffusionLoRA(native_lora_class):
        def forward(self, x, *args, **kwargs):
            state = branch_variable.get()
            if state is None:
                raise RuntimeError("Gen2 diffusion prediction requires an explicit branch context")
            base = self.org_forward(x, *args, **kwargs)
            return _gated_projection(self, x, base, state, diagnostic_variable)
    return GatedDiffusionLoRA


def _gated_projection(adapter, x, base, state, diagnostic_variable):
    """One residual equation for both backbones, using the live native factors."""
    if not state.enabled or not adapter.network_ref().is_active or state.strength == 0:
        return base
    residual = native_lora_residual(adapter, x, base.dtype)
    gate = state.gate_values[:, adapter.gen2_block_id].to(residual.device)
    if gate.shape[0] != residual.shape[0]:
        raise RuntimeError("Gate batch and native projection batch disagree")
    applied = residual*gate.reshape(gate.shape[0], *([1]*(residual.ndim-1)))*state.strength
    diagnostic = diagnostic_variable.get()
    if diagnostic is not None and not RECOMPUTING.get() and id(adapter) not in state.observed:
        state.observed.add(id(adapter))
        callback, regions = diagnostic
        callback(adapter, dequantize_projection_input(x.detach()), base.detach(), residual.detach(), applied.detach(), regions)
    return base+applied.to(base.dtype)


def bind_unconditional_lora(transformer, diffusion_network, branch_variable, diagnostic_variable):
    """Bind every original-unconditional projection to the existing LoRA factors.

    Only forwards are wrapped. No parameters/modules are registered on the
    frozen model and no factor copies, merges or synchronization steps exist.
    Validate the entire mapping before replacing any forward.
    """
    expected = {f"layers.{block}.{projection}": block
                for block in range(len(transformer.layers)) for projection in DIFFUSION_PROJECTIONS}
    adapters = {}
    for adapter in diffusion_network.unet_loras:
        path = adapter.gen2_original_path
        if path in adapters or path not in expected or adapter.gen2_block_id != expected[path]:
            raise ValueError(f"Invalid or duplicate original unconditional LoRA mapping: {path}")
        adapters[path] = adapter
    if set(adapters) != set(expected):
        raise ValueError(f"Missing original unconditional LoRA mappings: {sorted(set(expected)-set(adapters))}")
    bindings, seen = [], set()
    for path, block in expected.items():
        adapter = adapters[path]
        try:
            target = transformer.get_submodule(path)
        except AttributeError as error:
            raise ValueError(f"Missing original unconditional projection: {path}") from error
        source = adapter.orig_module_ref()
        if source is None or target is source or id(target) in seen:
            raise ValueError(f"Aliased original unconditional projection: {path}")
        if hasattr(target, "_gen2_shared_lora"):
            raise ValueError(f"Original unconditional projection already bound: {path}")
        shape = (getattr(target, "out_features", None), getattr(target, "in_features", None))
        if shape != (source.out_features, source.in_features):
            raise ValueError(f"Original unconditional projection shape mismatch at {path}: {shape}")
        seen.add(id(target))
        bindings.append((path, block, target, adapter))
    records = []
    for path, block, target, adapter in bindings:
        original_forward, adapter_ref = target.forward, weakref.ref(adapter)

        def forward(x, *args, original_forward=original_forward, adapter_ref=adapter_ref, **kwargs):
            state = branch_variable.get()
            if state is None or not state.unconditional:
                raise RuntimeError("Original unconditional projection requires an unconditional branch context")
            if torch.is_grad_enabled():
                raise RuntimeError("Original unconditional model is inference-only; use torch.no_grad")
            source_adapter = adapter_ref()
            if source_adapter is None:
                raise RuntimeError("Original unconditional projection lost its shared diffusion LoRA")
            base = original_forward(x, *args, **kwargs)
            return _gated_projection(source_adapter, x, base, state, diagnostic_variable)

        target.forward = forward
        target._gen2_shared_lora = adapter_ref
        records.append({"family": "diffusion", "branch": "unconditional",
            "original_path": "unconditional_transformer."+path,
            "block_id": block, "shape": [target.out_features, target.in_features],
            "rank": adapter.lora_dim, "alpha": float(adapter.alpha),
            "native_key": adapter.lora_name, "shared_with_native_key": adapter.lora_name,
            "parameter_count": 0, "dtype": str(adapter.lora_down.weight.dtype),
            "device": str(adapter.lora_down.weight.device), "frozen_base": True})
    return records


def _resolved_targets(root, paths, role, rank, alpha):
    targets = {}
    for block_id, path in paths:
        try:
            module = root.get_submodule(path)
        except AttributeError as exc:
            raise ValueError(f"Missing required Gen2 {role} target {path}") from exc
        if id(module) in targets or not hasattr(module, "in_features") or not hasattr(module, "out_features"):
            raise ValueError(f"Duplicate or non-linear target {path}")
        if rank > min(module.in_features, module.out_features):
            raise ValueError(f"Rank {rank} exceeds target dimensions at {path}")
        targets[id(module)] = {"original_path": path, "block_id": block_id, "family": role,
            "shape": [module.out_features, module.in_features], "rank": rank, "alpha": alpha}
    return targets


class Ideogram4Backend:
    @classmethod
    def load(cls, model_config, gen2_config, network_config, trigger_word,
             device="cuda:0", dtype="bf16", gradient_checkpointing=True):
        from extensions_built_in.diffusion_models.ideogram4.ideogram4 import Ideogram4Model
        model = Ideogram4Model(device=device, model_config=model_config, dtype=dtype)
        model.load_model()
        from .original_unconditional import load_original_unconditional
        load_original_unconditional(model, gen2_config)
        return cls.from_native(model, gen2_config, network_config, trigger_word, gradient_checkpointing)

    @classmethod
    def from_native(cls, model, gen2_config, network_config, trigger_word, gradient_checkpointing=True):
        return cls(model, gen2_config, network_config, trigger_word, gradient_checkpointing)

    def __init__(self, model, gen2_config, network_config, trigger_word, gradient_checkpointing=True):
        from toolkit.config_modules import NetworkConfig
        from toolkit.lora_special import LoRAModule, LoRASpecialNetwork
        from extensions_built_in.diffusion_models.ideogram4.src.pipeline import (
            pad_text_features, predict_velocity, get_qwen3_vl_features)
        from extensions_built_in.diffusion_models.ideogram4.src.transformer import QWEN3_VL_ACTIVATION_LAYERS
        if getattr(model, "arch", None) != "ideogram4":
            raise ValueError("Gen2 v1 supports the native Ideogram4Model only")
        if tuple(QWEN3_VL_ACTIVATION_LAYERS) != EXPECTED_ACTIVATION_LAYERS:
            raise ValueError("Native Qwen feature taps changed; backend compatibility review required")
        self.model, self.config, self.trigger_word = model, gen2_config, trigger_word
        self.pad_text_features, self.predict_velocity = pad_text_features, predict_velocity
        self.native_features = get_qwen3_vl_features
        self.gradient_checkpointing = gradient_checkpointing
        self.encoder_checkpointing = gen2_config["execution"]["encoder_gradient_checkpointing"]
        original_unconditional = getattr(model, "unconditional_transformer", None)
        requested_unconditional = gen2_config["inference"].get("unconditional_model_path")
        if bool(requested_unconditional) != (original_unconditional is not None):
            raise ValueError("Original unconditional model configuration and loaded component disagree")
        if original_unconditional is model.transformer:
            raise ValueError("Original unconditional model must be a separate transformer instance")
        self._branch = ContextVar(f"gen2_branch_{id(self)}", default=None)
        self._diagnostic = ContextVar(f"gen2_diagnostic_{id(self)}", default=None)
        # Native block checkpointing keeps its forward/arguments. This callback
        # carries Gen2's exact forward context into autograd worker replays.
        model.transformer._gradient_checkpointing_func = self._checkpoint
        for component in (model.transformer, model.text_encoder, model.vae):
            component.eval().requires_grad_(False)
        if original_unconditional is not None:
            original_unconditional.eval().requires_grad_(False)
            original_unconditional.disable_gradient_checkpointing()
            original_unconditional.set_attention_backend(gen2_config["execution"]["dit_attention_backend"])
            for key in ("in_channels", "emb_dim", "num_heads", "llm_features_dim"):
                if getattr(original_unconditional.config, key) != getattr(model.transformer.config, key):
                    raise ValueError(f"Original unconditional architecture differs at {key}")
        uncond = getattr(model, "unconditional_lora", None)
        if original_unconditional is not None and uncond is not None:
            raise ValueError("Original unconditional model cannot be combined with an unconditional correction LoRA")
        if uncond is not None:
            uncond.eval().requires_grad_(False)
            uncond.is_active = False
        if gradient_checkpointing:
            model.transformer.enable_gradient_checkpointing()
        else:
            model.transformer.disable_gradient_checkpointing()
        model.transformer.set_attention_backend(gen2_config["execution"]["dit_attention_backend"])
        c = gen2_config["conditioning"]
        ids = model.tokenizer(c["initializer_text"], add_special_tokens=False)["input_ids"]
        embedding = model.text_encoder.language_model.embed_tokens
        with torch.no_grad():
            rows = embedding(torch.tensor(ids, dtype=torch.long, device=embedding.weight.device))
        self.tokens = LearnedTokenBank(rows, ids, c["num_tokens"], c["initializer_seed"], c["initializer_jitter"])
        self.tokens.to(model.device_torch, dtype=torch.float32)
        self.gates = CubicTimeGates(len(model.transformer.layers), gen2_config["gates"]["amplitude"],
                                  gen2_config["gates"]["regularization_grid_points"]).to(model.device_torch)
        rank = network_config.get("linear", 32) if isinstance(network_config, dict) else network_config.linear
        alpha = network_config.get("linear_alpha", rank) if isinstance(network_config, dict) else network_config.linear_alpha
        diffusion_paths = [(b, f"layers.{b}.{p}") for b in range(len(model.transformer.layers)) for p in DIFFUSION_PROJECTIONS]
        self._diffusion_targets = _resolved_targets(model.transformer, diffusion_paths, "diffusion", rank, alpha)
        language_model = model.text_encoder.language_model
        text_paths = [(i, f"layers.{i}.mlp.down_proj") for i in range(len(language_model.layers))]
        self._text_targets = _resolved_targets(language_model, text_paths, "text_adapter", c["adapter_rank"], c["adapter_alpha"])
        def create_native(root, targets, rank, alpha, module_class, base_model=None):
            native_config = NetworkConfig(type="lora", linear=rank, linear_alpha=alpha, transformer_only=False)
            # Native PEFT construction otherwise silently forces alpha=rank.
            # Its per-module dimensions/alpha seam preserves the requested scale.
            native_keys = ["transformer$$"+entry["original_path"].replace(".", "$$") for entry in targets.values()]
            net = LoRASpecialNetwork(text_encoder=None, unet=root, lora_dim=rank, alpha=alpha,
                multiplier=1., train_unet=True, train_text_encoder=False, network_type="lora",
                network_config=native_config, module_class=module_class, use_bias=False,
                dropout=None, rank_dropout=None, module_dropout=None, transformer_only=False,
                target_lin_modules=[root.__class__.__name__],
                only_if_contains=[entry["original_path"] for entry in targets.values()],
                modules_dim={key: rank for key in native_keys},
                modules_alpha={key: alpha for key in native_keys},
                is_transformer=True, base_model=base_model)
            seen = set()
            for adapter in net.unet_loras:
                identity = id(adapter.orig_module_ref())
                if identity not in targets or identity in seen:
                    raise ValueError("Native LoRA target discovery produced an extra or duplicate projection")
                seen.add(identity)
                adapter.gen2_block_id = targets[identity]["block_id"]
                adapter.gen2_original_path = targets[identity]["original_path"]
                if adapter.lora_dim != rank or float(adapter.alpha) != float(alpha):
                    raise ValueError("Native LoRA did not preserve the configured rank and alpha")
                adapter.can_merge_in = False
            if seen != set(targets):
                raise ValueError("Native LoRA target discovery omitted required projections")
            net.apply_to(None, root, apply_text_encoder=False, apply_unet=True)
            net.force_to(model.device_torch, dtype=torch.float32)
            net.eval()
            net.is_active = True
            return net
        self.diffusion_network = create_native(model.transformer, self._diffusion_targets, rank, alpha,
            make_gated_lora_class(LoRAModule, self._branch, self._diagnostic), model)
        self._unconditional_mapping = (bind_unconditional_lora(original_unconditional,
            self.diffusion_network, self._branch, self._diagnostic) if original_unconditional is not None else [])
        if self._unconditional_mapping:
            print(f"[Gen2] Original unconditional transformer: {len(self._unconditional_mapping)} projections "
                  "share the live diffusion LoRA parameters and time gates", flush=True)
        self.text_network = create_native(language_model, self._text_targets, c["adapter_rank"], c["adapter_alpha"],
                                         make_masked_lora_class(LoRAModule))
        # Native adapters live outside the frozen modules' parameter registration.
        frozen = [model.transformer, model.text_encoder, model.vae]
        frozen.extend(component for component in (original_unconditional, uncond) if component is not None)
        self._frozen_parameters = tuple(p for m in frozen for p in m.parameters())

    def parameter_families(self):
        return {"diffusion": list(self.diffusion_network.parameters()), "embedding": list(self.tokens.parameters()),
                "text_adapter": list(self.text_network.parameters()), "gates": list(self.gates.parameters())}

    def components(self):
        return {"diffusion": self.diffusion_network, "embedding": self.tokens,
                "text_adapter": self.text_network, "gates": self.gates}

    def module_manifest(self):
        records = []
        for network, targets in ((self.diffusion_network, self._diffusion_targets), (self.text_network, self._text_targets)):
            for adapter in network.unet_loras:
                entry = dict(targets[id(adapter.orig_module_ref())])
                entry.update(native_key=adapter.lora_name, parameter_count=sum(p.numel() for p in adapter.parameters()),
                             dtype=str(adapter.lora_down.weight.dtype), device=str(adapter.lora_down.weight.device),
                             frozen_base=True)
                if entry["family"] == "text_adapter":
                    entry["original_path"] = "language_model."+entry["original_path"]
                records.append(entry)
        return records + getattr(self, "_unconditional_mapping", [])

    def assert_frozen(self):
        if any(p.requires_grad for p in self._frozen_parameters):
            raise RuntimeError("Original model parameters became trainable")
        if any(p.dtype != torch.float32 for ps in self.parameter_families().values() for p in ps):
            raise RuntimeError("A Gen2 trainable master is no longer float32")
        # Called again after native optimizer steps and before logical commit.
        # Reject a finite but zero/floor-sized U immediately, before a package
        # can be published with an undefined normalized-token representation.
        with torch.no_grad():
            self.tokens()

    def frozen_parameters(self):
        return iter(self._frozen_parameters)

    @property
    def current_branch(self):
        return self._branch.get()

    def _checkpoint(self, function, *args, **kwargs):
        """Bind each native block replay to the state from its own forward.

        Holding branch() around backward is insufficient on CUDA workers:
        ContextVars belong to a Python execution context, not the autograd graph.
        Capture the original gate tensor without detaching it so G gradients
        still reach the gate coefficients. Restore all ambient state on exit.
        """
        state = self._branch.get()
        if state is None:
            raise RuntimeError("Gen2 checkpoint requires an explicit forward branch context")
        if kwargs.get("use_reentrant", False) or "context_fn" in kwargs:
            raise ValueError("Gen2 requires non-reentrant checkpointing with its bound replay context")
        kwargs["use_reentrant"] = False
        network = self.diffusion_network
        uncond = getattr(self.model, "unconditional_lora", None)
        active = network.is_active
        unconditional_active = uncond.is_active if uncond is not None else None

        @contextmanager
        def replay():
            branch_token = self._branch.set(state)
            diagnostic_token = self._diagnostic.set(None)
            recomputing_token = RECOMPUTING.set(True)
            previous_active = network.is_active
            previous_unconditional = uncond.is_active if uncond is not None else None
            network.is_active = active
            if uncond is not None:
                uncond.is_active = unconditional_active
            try:
                yield
            finally:
                network.is_active = previous_active
                if uncond is not None:
                    uncond.is_active = previous_unconditional
                RECOMPUTING.reset(recomputing_token)
                self._diagnostic.reset(diagnostic_token)
                self._branch.reset(branch_token)

        return checkpoint(function, *args, **kwargs,
                          context_fn=lambda: (nullcontext(), _ReusableContext(replay)))

    def _ensure_native_device(self, component, name):
        """Mirror native Ideogram's low_vram CPU-to-compute-device guard.

        This runs before a component's initial forward, never during decoder or
        DiT checkpoint recomputation. Native .to and its offload hooks retain
        ownership of placement; Gen2 provides no offload implementation.
        """
        device = getattr(component, "device", None)
        if device is None:
            device = next(component.parameters()).device
        if torch.device(device).type == "cpu" and torch.device(self.model.device_torch).type != "cpu":
            branch = self._branch.get()
            if branch is not None and name in branch.started_components:
                raise RuntimeError(f"Cannot move {name} while its branch graph may still recompute")
            component.to(self.model.device_torch)

    @torch.no_grad()
    def verify_prefix(self, qs, atol=.001, rtol=.01):
        """Measure native numerical baseline, sibling parity and every styled tap."""
        native = self.encode(qs, styled=False)
        repeat = self.encode(qs, styled=False)
        styled = self.encode(qs, styled=True)
        rows = []
        language_model = self.model.text_encoder.language_model
        device = language_model.embed_tokens.weight.device
        for i, item in enumerate(native.metadata):
            ids = torch.tensor([item["original_ids"]], device=device, dtype=torch.long)
            mask = torch.ones_like(ids)
            positions = mask.cumsum(-1)-1
            inputs = language_model.embed_tokens(ids)
            sibling, _, _, _ = differentiable_qwen_features(self.model.text_encoder, inputs, mask,
                positions, torch.zeros_like(mask, dtype=torch.bool), adapter_enabled=False)
            comparisons = {"native_repeat": repeat.features[i], "neutral_sibling": sibling[0],
                           "styled_prefix": styled.features[i][:item["original_length"]]}
            for label, features in comparisons.items():
                for tap_index, layer in enumerate(EXPECTED_ACTIVATION_LAYERS):
                    expected = native.features[i][..., tap_index::len(EXPECTED_ACTIVATION_LAYERS)].float()
                    observed = features[..., tap_index::len(EXPECTED_ACTIVATION_LAYERS)].float()
                    difference = (expected-observed).abs()
                    passed = torch.allclose(expected, observed, atol=atol, rtol=rtol)
                    row = {"example_index": i, "comparison": label, "tap": layer,
                           "max_abs": difference.max().item(), "difference_rms": difference.square().mean().sqrt().item(),
                           "reference_rms": expected.square().mean().sqrt().item(), "atol": atol, "rtol": rtol,
                           "passed": passed}
                    rows.append(row)
        failures = [row for row in rows if not row["passed"]]
        if failures:
            error = RuntimeError(f"Gen2 native prefix/packing acceptance failed: {failures}")
            error.records = rows
            raise error
        return rows

    def encode(self, qs, styled=True, gradients=False, token_mode="learned", adapter_enabled=True):
        if gradients and not torch.is_grad_enabled():
            raise RuntimeError("A conditioning encode entered an outer no_grad context")
        self._ensure_native_device(self.model.text_encoder, "text_encoder")
        features, metadata, rt, numerators, denominators = [], [], [], [], []
        language_model = self.model.text_encoder.language_model
        device = language_model.embed_tokens.weight.device
        with torch.set_grad_enabled(gradients):
            for example_index, caption in enumerate(qs):
                item = tokenize_caption(caption, self.trigger_word, self.model.tokenizer)
                serialized, ids = item["serialized_text"], item["original_ids"]
                m = self.tokens.U.shape[0] if styled else 0
                # Reserve M for both paired routes, so C0 never accepts a sample C+ cannot represent.
                reserve = self.tokens.U.shape[0]
                require_token_budget(item, reserve, self.model.max_text_length)
                token_ids = torch.tensor([ids], device=device, dtype=torch.long)
                original = language_model.embed_tokens(token_ids)
                inputs = original
                if styled:
                    suffix = self.tokens(token_mode).to(device=device, dtype=original.dtype).unsqueeze(0)
                    inputs = torch.cat((original, suffix), dim=1)
                mask = torch.ones(inputs.shape[:2], device=device, dtype=torch.long)
                positions = (mask.cumsum(-1)-1).clamp(min=0).long()
                suffix_mask = positions >= len(ids)
                if not styled:
                    packed = self.native_features(self.model.text_encoder, token_ids, mask, positions)
                    one_rt = packed.new_zeros(1, dtype=torch.float32)
                    numerator = denominator = packed.new_empty(0, dtype=torch.float32)
                else:
                    callback = None
                    current = self._diagnostic.get()
                    if current is not None:
                        def callback(adapter, *values, example_index=example_index, observer=current[0]):
                            old_index = getattr(adapter, "gen2_example_index", None)
                            adapter.gen2_example_index = example_index
                            try:
                                observer(adapter, *values)
                            finally:
                                adapter.gen2_example_index = old_index
                    packed, one_rt, numerator, denominator = differentiable_qwen_features(
                        self.model.text_encoder, inputs, mask, positions, suffix_mask,
                        adapter_enabled=adapter_enabled, gradient_checkpointing=self.encoder_checkpointing,
                        diagnostic=callback)
                features.append(packed[0].to(self.model.torch_dtype))
                rt.append(one_rt[0]); numerators.append(numerator); denominators.append(denominator)
                item.update(serialized_text=serialized, original_ids=ids, original_length=len(ids),
                            total_length=len(ids)+m, suffix_positions=list(range(len(ids), len(ids)+m)),
                            positions=list(range(len(ids)+m)), suffix_mask=suffix_mask[0].tolist(),
                            overflow=False, token_mode=token_mode if styled else "absent",
                            adapter_enabled=bool(styled and adapter_enabled))
                metadata.append(item)
        if gradients and self._branch.get() is not None:
            self._branch.get().started_components.add("text_encoder")
        return Conditioning(features, metadata, torch.stack(rt), torch.cat(numerators), torch.cat(denominators))

    @contextmanager
    def branch(self, tau, lora_enabled=True, gate_mode="one", strength=1., unconditional=False,
               name="student", allow_unconditional_lora=False):
        if tau.ndim != 1 or not bool(torch.isfinite(tau).all()) or bool(((tau < 0)|(tau > 1)).any()):
            raise ValueError("Branch tau must be a finite batch vector in [0,1]")
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("Branch LoRA strength must be finite and nonnegative")
        original_unconditional = getattr(self.model, "unconditional_transformer", None)
        if unconditional and original_unconditional is not None and torch.is_grad_enabled():
            raise ValueError("Original unconditional model is inference-only; use torch.no_grad")
        if unconditional and lora_enabled:
            if not allow_unconditional_lora:
                raise ValueError("CFG unconditional personalization requires explicit inference opt-in")
            if torch.is_grad_enabled():
                raise ValueError("CFG unconditional personalization is an inference-only diagnostic; use torch.no_grad")
        # Resolve all fallible inputs before touching either network's live flag.
        state = DiffusionBranch(tau, self.gates.values(tau, gate_mode), bool(lora_enabled),
                                float(strength), bool(unconditional), name)
        uncond = getattr(self.model, "unconditional_lora", None)
        if original_unconditional is not None and uncond is not None:
            raise ValueError("Original unconditional model cannot be combined with an unconditional correction LoRA")
        old_uncond = uncond.is_active if uncond is not None else None
        old_active = self.diffusion_network.is_active
        self.diffusion_network.is_active = bool(lora_enabled)
        if uncond is not None:
            uncond.is_active = bool(unconditional)
        token = self._branch.set(state)
        try:
            yield state
        finally:
            self._branch.reset(token)
            self.diffusion_network.is_active = old_active
            if uncond is not None:
                uncond.is_active = old_uncond

    @contextmanager
    def diagnostics(self, callback):
        # None is an explicit nested suppression scope for isolated probes.
        token = self._diagnostic.set(None if callback is None else (callback, {}))
        try:
            yield
        finally:
            self._diagnostic.reset(token)

    def predict(self, latents, tau, conditioning):
        state = self._branch.get()
        if state is None:
            raise RuntimeError("Predict must run inside backend.branch through its backward")
        if not torch.equal(state.tau.to(tau.device), tau):
            raise ValueError("Prediction tau differs from its bound gate/branch context")
        transformer = self.model.transformer
        component_name = "transformer"
        if state.unconditional and getattr(self.model, "unconditional_transformer", None) is not None:
            if torch.is_grad_enabled():
                raise RuntimeError("Original unconditional prediction is inference-only; use torch.no_grad")
            if any(feature.shape[0] != 0 for feature in conditioning.features):
                raise ValueError("Original unconditional prediction requires image-only conditioning with zero text tokens")
            transformer = self.model.unconditional_transformer
            component_name = "unconditional_transformer"
        self._ensure_native_device(transformer, component_name)
        features, mask = self.pad_text_features(conditioning.features, self.model.device_torch, self.model.torch_dtype)
        current = self._diagnostic.get()
        token = None
        if current is not None:
            b, length = mask.shape
            image_count = latents.shape[-1]*latents.shape[-2]
            original = torch.zeros(b, length+image_count, device=mask.device, dtype=torch.bool)
            suffix = torch.zeros_like(original); image = torch.zeros_like(original); padding = torch.zeros_like(original)
            for i, item in enumerate(conditioning.metadata):
                original[i, :item["original_length"]] = True
                suffix[i, item["original_length"]:item["total_length"]] = True
            image[:, length:] = True
            padding[:, :length] = ~mask.bool()
            token = self._diagnostic.set((current[0], {"original": original, "suffix": suffix,
                                                     "image": image, "padding": padding}))
        try:
            # Native helper owns both reversed model time and velocity negation.
            result = self.predict_velocity(transformer,
                latents.to(self.model.device_torch, self.model.torch_dtype), tau, features, mask)
            if torch.is_grad_enabled() and result.requires_grad:
                state.started_components.add("transformer")
            return result
        finally:
            if token is not None:
                self._diagnostic.reset(token)

    def empty_conditioning(self, batch_size, feature_dim):
        return Conditioning([torch.empty(0, feature_dim, device=self.model.device_torch,
                              dtype=self.model.torch_dtype) for _ in range(batch_size)],
                            [{"original_length": 0, "total_length": 0} for _ in range(batch_size)],
                            torch.zeros(batch_size, device=self.model.device_torch))
