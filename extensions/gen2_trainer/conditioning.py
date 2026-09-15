"""Reserved trigger compilation, normalized soft tokens and masked native LoRA."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .objectives import text_adapter_ratio

EPSILON_EMBEDDING = 1e-8


def dequantize_projection_input(x):
    # Ordinary Tensor.dequantize introduces an unsupported autograd operation.
    # Native ToolkitModuleMixin dequantizes Quanto activations only.
    quantized = bool(getattr(x, "is_quantized", False)) or type(x).__module__.startswith(("optimum.quanto", "torchao"))
    return x.dequantize() if quantized else x


def native_lora_residual(adapter, x, compute_dtype):
    """Keep native low-rank execution, with one explicit compute-dtype boundary.

    This scope also applies during inference, where no trainer autocast context
    exists. Parameter masters stay fp32; casts remain differentiable.
    """
    plain = dequantize_projection_input(x)
    mixed = compute_dtype in (torch.float16, torch.bfloat16)
    with torch.autocast(device_type=plain.device.type, dtype=compute_dtype if mixed else torch.bfloat16, enabled=mixed):
        return adapter._call_forward(plain.to(adapter.lora_down.weight.dtype))


def compile_trigger(caption: str, trigger_word: str) -> dict:
    if not isinstance(trigger_word, str) or not trigger_word:
        raise ValueError("process.trigger_word must be a nonempty reserved literal")
    expanded = caption.replace("[trigger]", trigger_word)
    return {"original_caption": caption, "marker_expanded_caption": expanded,
            "q": expanded.replace(trigger_word, "").strip(),
            "trigger_present": trigger_word in expanded}


def normalized_vectors(raw: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    """Float64 remains available only for independent analytic reference tests."""
    work = raw if raw.dtype == torch.float64 else raw.float()
    norms = work.norm(dim=-1, keepdim=True)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= EPSILON_EMBEDDING).any()):
        raise FloatingPointError("Learned token raw norm reached epsilon floor or is nonfinite")
    return radii.to(work.dtype).unsqueeze(-1)*work/norms.clamp_min(EPSILON_EMBEDDING)


class LearnedTokenBank(nn.Module):
    def __init__(self, vocabulary_rows: torch.Tensor, initializer_ids: list[int],
                 num_tokens: int = 4, seed: int = 271828, jitter: float = .01):
        super().__init__()
        if not initializer_ids or vocabulary_rows.ndim != 2 or len(initializer_ids) != len(vocabulary_rows):
            raise ValueError("Initializer needs nonempty token IDs and their vocabulary rows")
        if not 1 <= num_tokens <= 32 or not 0 <= jitter <= 1:
            raise ValueError("num_tokens must be in [1,32], jitter in [0,1]")
        rows = vocabulary_rows.detach().to("cpu", torch.float32)
        # Conventional median averages the two middle norms for an even count.
        a, radius = rows.mean(0), rows.norm(dim=-1).quantile(.5)
        if not bool(torch.isfinite(rows).all()) or radius <= 0 or a.norm() <= 0:
            raise ValueError("Initializer vocabulary rows have invalid norms/mean")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        xi = torch.randn(num_tokens, rows.shape[1], generator=generator)
        xi = xi/xi.norm(dim=-1, keepdim=True)
        raw = a/a.norm()+jitter*xi
        raw = radius*raw/raw.norm(dim=-1, keepdim=True)
        self.U = nn.Parameter(raw)
        self.register_buffer("r", radius.expand(num_tokens).clone())
        self.register_buffer("e_init", raw.clone())
        self.register_buffer("initializer_ids", torch.tensor(initializer_ids, dtype=torch.long))
        self.register_buffer("initializer_seed", torch.tensor(seed, dtype=torch.long))
        self.register_buffer("initializer_jitter", torch.tensor(jitter, dtype=torch.float32))

    def forward(self, mode: str = "learned") -> torch.Tensor:
        if mode == "init":
            return self.e_init
        if mode != "learned":
            raise ValueError(f"Unknown token view: {mode}")
        return normalized_vectors(self.U, self.r)


@dataclass
class ProjectionScope:
    suffix_mask: torch.Tensor
    enabled: bool
    # Local to one layer invocation, returned explicitly by its checkpoint wrapper.
    result: tuple | None = None
    diagnostic: Callable | None = None


ENCODER_SCOPE: ContextVar[ProjectionScope | None] = ContextVar("gen2_encoder_scope", default=None)


@contextmanager
def encoder_projection_scope(suffix_mask, enabled=True, diagnostic=None):
    scope = ProjectionScope(suffix_mask, enabled, diagnostic=diagnostic)
    token = ENCODER_SCOPE.set(scope)
    try:
        yield scope
    finally:
        ENCODER_SCOPE.reset(token)


def make_masked_lora_class(native_lora_class):
    """Use native factors, initialization, runtime alpha/rank, activation and state."""
    class MaskedEncoderLoRA(native_lora_class):
        def forward(self, x, *args, **kwargs):
            scope = ENCODER_SCOPE.get()
            base = self.org_forward(x, *args, **kwargs)
            # A native C0 pass never has an encoder adaptation scope.
            if scope is None or not scope.enabled or not self.network_ref().is_active:
                return base
            if scope.result is not None:
                raise RuntimeError("An encoder projection ran twice in one decoder wrapper")
            residual = native_lora_residual(self, x, base.dtype)
            applied = residual*scope.suffix_mask.to(residual.dtype).unsqueeze(-1)
            scope.result = text_adapter_ratio(base, residual, scope.suffix_mask)
            if scope.diagnostic is not None:
                scope.diagnostic(self, dequantize_projection_input(x.detach()), base.detach(), residual.detach(), applied.detach(),
                                 {"suffix": scope.suffix_mask, "original": ~scope.suffix_mask})
            return base + applied.to(base.dtype)
    return MaskedEncoderLoRA


@dataclass
class Conditioning:
    features: list[torch.Tensor]
    metadata: list[dict]
    rt_per_example: torch.Tensor
    rt_numerators: torch.Tensor | None = None
    rt_denominators: torch.Tensor | None = None

    @property
    def text_embeds(self):
        return self.features

    @property
    def rt(self):
        return self.rt_per_example.mean()
