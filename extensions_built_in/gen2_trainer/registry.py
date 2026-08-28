from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .temporal_rank_field import TemporalRankFieldLoRA


@dataclass
class AdapterRuntimeContext:
    timestep: torch.Tensor
    style_gate: torch.Tensor
    branch_scale: torch.Tensor | float
    image_token_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class TargetSpec:
    name: str
    module: nn.Module
    row_slice: tuple[int, int] | None


class FusedQKVTemporalAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int, alpha: float, knots: int, delta_max: float) -> None:
        super().__init__()
        self.q = TemporalRankFieldLoRA(hidden_size, hidden_size, rank, alpha, knots, delta_max, (0, hidden_size))
        self.k = TemporalRankFieldLoRA(hidden_size, hidden_size, rank, alpha, knots, delta_max, (hidden_size, 2 * hidden_size))
        self.v = TemporalRankFieldLoRA(hidden_size, hidden_size, rank, alpha, knots, delta_max, (2 * hidden_size, 3 * hidden_size))

    def residual(self, x: torch.Tensor, context: AdapterRuntimeContext) -> torch.Tensor:
        parts = [
            item(x, context.timestep, context.image_token_mask, context.style_gate, context.branch_scale)
            for item in (self.q, self.k, self.v)
        ]
        return torch.cat(parts, dim=-1)


class Gen2LinearWrapper(nn.Module):
    def __init__(self, base: nn.Module, adapter: nn.Module, context_owner: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter
        object.__setattr__(self, "_context_owner", context_owner)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        context = getattr(self._context_owner, "_gen2_adapter_context", None)
        if context is None or torch.count_nonzero(context.style_gate).item() == 0:
            return output
        if isinstance(self.adapter, FusedQKVTemporalAdapter):
            residual = self.adapter.residual(x, context)
        else:
            residual = self.adapter(
                x,
                context.timestep,
                context.image_token_mask,
                context.style_gate,
                context.branch_scale,
            )
        return output + residual.to(output.dtype)


class Gen2AdapterBank(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapters = nn.ModuleDict()
        self.registry: list[dict] = []

    def add(self, key: str, adapter: nn.Module, target: TargetSpec) -> None:
        safe_key = key.replace(".", "__")
        self.adapters[safe_key] = adapter
        self.registry.append({
            "name": target.name,
            "row_slice": target.row_slice,
            "in_features": target.module.in_features,
            "out_features": target.module.out_features,
        })

    def temporal_fields(self):
        for module in self.adapters.values():
            if isinstance(module, FusedQKVTemporalAdapter):
                yield module.q.temporal
                yield module.k.temporal
                yield module.v.temporal
            else:
                yield module.temporal


def install_ideogram_adapters(transformer: nn.Module, rank: int, alpha: float, knots: int, delta_max: float) -> Gen2AdapterBank:
    if getattr(transformer, "_gen2_adapter_bank", None) is not None:
        return transformer._gen2_adapter_bank
    bank = Gen2AdapterBank()
    hidden_size = transformer.config.emb_dim
    for block_index, block in enumerate(transformer.layers):
        prefix = f"layers.{block_index}"
        qkv_target = TargetSpec(f"{prefix}.attention.qkv", block.attention.qkv, (0, hidden_size * 3))
        qkv_adapter = FusedQKVTemporalAdapter(hidden_size, rank, alpha, knots, delta_max)
        bank.add(qkv_target.name, qkv_adapter, qkv_target)
        block.attention.qkv = Gen2LinearWrapper(block.attention.qkv, qkv_adapter, transformer)
        targets = [
            ("attention.o", block.attention, "o"),
            ("feed_forward.w1", block.feed_forward, "w1"),
            ("feed_forward.w2", block.feed_forward, "w2"),
            ("feed_forward.w3", block.feed_forward, "w3"),
        ]
        for suffix, owner, attribute in targets:
            base = getattr(owner, attribute)
            target = TargetSpec(f"{prefix}.{suffix}", base, None)
            adapter = TemporalRankFieldLoRA(base.in_features, base.out_features, rank, alpha, knots, delta_max)
            bank.add(target.name, adapter, target)
            setattr(owner, attribute, Gen2LinearWrapper(base, adapter, transformer))
    transformer.add_module("_gen2_adapter_bank", bank)
    transformer._gen2_adapter_context = None
    return bank


def set_adapter_context(transformer: nn.Module, timestep: torch.Tensor, indicator: torch.Tensor, style_gate: torch.Tensor, branch_scale: torch.Tensor | float) -> None:
    if timestep.ndim != 1 or timestep.shape[0] != indicator.shape[0]:
        raise ValueError("adapter timestep must have shape (batch,)")
    if style_gate.ndim != 1 or style_gate.shape[0] != indicator.shape[0]:
        raise ValueError("adapter style_gate must have shape (batch,)")
    image_mask = (indicator == 2).to(dtype=torch.float32).unsqueeze(-1)
    transformer._gen2_adapter_context = AdapterRuntimeContext(
        timestep=timestep,
        style_gate=style_gate,
        branch_scale=branch_scale,
        image_token_mask=image_mask,
    )


def clear_adapter_context(transformer: nn.Module) -> None:
    transformer._gen2_adapter_context = None


def iter_ideogram_targets(transformer: nn.Module) -> list[TargetSpec]:
    targets: list[TargetSpec] = []
    hidden_size = transformer.config.emb_dim
    for block_index, block in enumerate(transformer.layers):
        prefix = f"layers.{block_index}"
        qkv = block.attention.qkv.base if isinstance(block.attention.qkv, Gen2LinearWrapper) else block.attention.qkv
        targets.extend([
            TargetSpec(f"{prefix}.attention.qkv.q", qkv, (0, hidden_size)),
            TargetSpec(f"{prefix}.attention.qkv.k", qkv, (hidden_size, 2 * hidden_size)),
            TargetSpec(f"{prefix}.attention.qkv.v", qkv, (2 * hidden_size, 3 * hidden_size)),
        ])
        for suffix, module in (
            ("attention.o", block.attention.o),
            ("feed_forward.w1", block.feed_forward.w1),
            ("feed_forward.w2", block.feed_forward.w2),
            ("feed_forward.w3", block.feed_forward.w3),
        ):
            base = module.base if isinstance(module, Gen2LinearWrapper) else module
            targets.append(TargetSpec(f"{prefix}.{suffix}", base, None))
    return targets


def validate_registry_compatibility(left: list[TargetSpec], right: list[TargetSpec]) -> None:
    left_signature = [(item.name, item.row_slice, item.module.in_features, item.module.out_features) for item in left]
    right_signature = [(item.name, item.row_slice, item.module.in_features, item.module.out_features) for item in right]
    if left_signature != right_signature:
        raise ValueError("official unconditional adapter registry is not exactly compatible")
