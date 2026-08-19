from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn


GROUP_KINDS = ("attention", "mlp", "adaln", "other")
_BLOCK_RE = re.compile(r"(?:^|[.$_])layers(?:[.$_])(\d+)(?:[.$_]|$)")
_CURRENT_RESIDUAL_GATES: contextvars.ContextVar[Optional[torch.Tensor]] = contextvars.ContextVar(
    "ai_toolkit_residual_gates", default=None
)


def module_display_name(module: Any) -> str:
    name = str(getattr(module, "lora_name", ""))
    return name.replace("$$", ".").replace("diffusion_model.", "transformer.")


def classify_lora_module(name: str) -> Tuple[Optional[int], str]:
    normalized = name.replace("$$", ".").lower()
    match = _BLOCK_RE.search(normalized)
    block_index = int(match.group(1)) if match else None
    if ".attention." in normalized or ".attn." in normalized:
        kind = "attention"
    elif ".feed_forward." in normalized or ".mlp." in normalized or any(
        token in normalized for token in (".w1", ".w2", ".w3")
    ):
        kind = "mlp"
    elif "adaln" in normalized or "ada_ln" in normalized:
        kind = "adaln"
    else:
        kind = "other"
    return block_index, kind


def _fp32_finite_norm(tensor: Any) -> Tuple[Optional[float], bool]:
    if tensor is None:
        return None, False
    value = tensor.detach().to(device="cpu", dtype=torch.float32)
    finite = bool(torch.isfinite(value).all())
    norm = float(torch.linalg.vector_norm(value).item()) if finite else None
    return norm, finite


def build_module_registry(modules: Sequence[Any]) -> List[Dict[str, Any]]:
    registry: List[Dict[str, Any]] = []
    seen = set()
    for index, module in enumerate(modules):
        name = module_display_name(module)
        if not name or name in seen:
            raise ValueError(f"LoRA registry requires unique non-empty names: {name!r}")
        seen.add(name)
        block_index, kind = classify_lora_module(name)
        group_id = f"block:{block_index}:{kind}" if block_index is not None else f"global:{kind}"
        down = getattr(getattr(module, "lora_down", None), "weight", None)
        up = getattr(getattr(module, "lora_up", None), "weight", None)
        down_norm, down_finite = _fp32_finite_norm(down)
        up_norm, up_finite = _fp32_finite_norm(up)
        registry.append({
            "module_index": index,
            "module_name": name,
            "block_index": block_index,
            "kind": kind,
            "group_id": group_id,
            "rank": int(getattr(module, "lora_dim", down.shape[0] if down is not None else 0)),
            "down_shape": list(down.shape) if down is not None else None,
            "up_shape": list(up.shape) if up is not None else None,
            "down_fp32_norm": down_norm,
            "up_fp32_norm": up_norm,
            "down_finite": down_finite,
            "up_finite": up_finite,
        })
    if not registry:
        raise ValueError("LoRA registry is empty")
    return registry


def validate_complete_partition(registry: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    names = [str(row["module_name"]) for row in registry]
    if len(names) != len(set(names)):
        raise ValueError("LoRA registry contains duplicate module names")
    invalid = [row for row in registry if row.get("kind") not in GROUP_KINDS or not row.get("group_id")]
    if invalid:
        raise ValueError("LoRA registry partition is incomplete")
    counts = {kind: sum(row["kind"] == kind for row in registry) for kind in GROUP_KINDS}
    groups = sorted({str(row["group_id"]) for row in registry})
    return {"module_count": len(registry), "group_count": len(groups), "kind_counts": counts, "groups": groups}


def registry_maps(registry: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    module_to_group = {str(row["module_name"]): str(row["group_id"]) for row in registry}
    group_to_modules: Dict[str, List[str]] = defaultdict(list)
    for module_name, group_id in module_to_group.items():
        group_to_modules[group_id].append(module_name)
    return module_to_group, {key: sorted(value) for key, value in group_to_modules.items()}


def filter_active_registry(
    registry: Sequence[Mapping[str, Any]],
    norm_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    threshold = float(norm_threshold)
    if threshold < 0.0:
        raise ValueError("norm_threshold must be non-negative")
    active_groups = set()
    for row in registry:
        down_norm = row.get("down_fp32_norm")
        up_norm = row.get("up_fp32_norm")
        module_active = (
            bool(row.get("down_finite"))
            and bool(row.get("up_finite"))
            and down_norm is not None
            and up_norm is not None
            and float(down_norm) > threshold
            and float(up_norm) > threshold
        )
        if module_active:
            active_groups.add(str(row["group_id"]))
    if not active_groups:
        raise ValueError("active residual registry is empty after FP32 finite norm filtering")
    ordered_groups = sorted(active_groups)
    group_indices = {group: index for index, group in enumerate(ordered_groups)}
    active = []
    for source_row in registry:
        group_id = str(source_row["group_id"])
        if group_id not in active_groups:
            continue
        row = dict(source_row)
        row["module_active"] = (
            bool(row.get("down_finite"))
            and bool(row.get("up_finite"))
            and row.get("down_fp32_norm") is not None
            and row.get("up_fp32_norm") is not None
            and float(row["down_fp32_norm"]) > threshold
            and float(row["up_fp32_norm"]) > threshold
        )
        row["group_index"] = group_indices[group_id]
        active.append(row)
    return active


def serialize_active_registry(registry: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [dict(row) for row in registry]
    rows.sort(key=lambda row: (int(row["group_index"]), int(row["module_index"]), str(row["module_name"])))
    groups = []
    for group_id in sorted({str(row["group_id"]) for row in rows}, key=lambda group: next(
        int(row["group_index"]) for row in rows if str(row["group_id"]) == group
    )):
        group_rows = [row for row in rows if str(row["group_id"]) == group_id]
        groups.append({
            "group_id": group_id,
            "group_index": int(group_rows[0]["group_index"]),
            "module_names": [str(row["module_name"]) for row in group_rows],
        })
    return {
        "schema": "ai-toolkit.residual-gate-registry",
        "schema_version": 1,
        "group_count": len(groups),
        "module_count": len(rows),
        "groups": groups,
        "modules": rows,
    }


def active_registry_fingerprint(registry: Sequence[Mapping[str, Any]]) -> str:
    payload = serialize_active_registry(registry)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bind_active_registry(modules: Sequence[Any], registry: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    module_by_name = {module_display_name(module): module for module in modules}
    for module in modules:
        module._residual_gate_group_index = -1
    bound: Dict[str, int] = {}
    for row in registry:
        name = str(row["module_name"])
        module = module_by_name.get(name)
        if module is None:
            raise KeyError(f"active registry module is not installed: {name}")
        group_index = int(row["group_index"])
        module._residual_gate_group_index = group_index
        bound[name] = group_index
    return bound


def normalize_canonical_timestep(timestep: torch.Tensor) -> torch.Tensor:
    if not isinstance(timestep, torch.Tensor):
        raise TypeError("canonical timestep must be a torch.Tensor")
    if timestep.ndim == 1:
        value = timestep.unsqueeze(-1)
    elif timestep.ndim == 2 and timestep.shape[1] == 1:
        value = timestep
    elif timestep.ndim == 2:
        first = timestep[:, :1]
        if not torch.allclose(timestep.float(), first.expand_as(timestep).float(), rtol=0.0, atol=1.0e-6):
            raise ValueError("timestep-only routing requires one canonical timestep per batch row")
        value = first
    else:
        raise ValueError("canonical timestep must have shape [B], [B,1], or row-constant [B,L]")
    value = value.to(torch.float32)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("canonical timestep must be finite")
    if bool(((value < 0.0) | (value > 1.0)).any()):
        raise ValueError("canonical timestep must already be normalized to [0,1]")
    return value


def effective_gates(q: torch.Tensor, style_strength: Any) -> torch.Tensor:
    if q.ndim != 2:
        raise ValueError("q must have shape [B,G]")
    strength = torch.as_tensor(style_strength, device=q.device, dtype=q.dtype)
    if strength.ndim == 0:
        strength = strength.reshape(1, 1)
    elif strength.ndim == 1:
        strength = strength.unsqueeze(-1)
    elif strength.ndim != 2 or strength.shape[1] != 1:
        raise ValueError("style_strength must be scalar, [B], or [B,1]")
    if not bool(torch.isfinite(strength).all()) or bool(((strength < 0.0) | (strength > 1.0)).any()):
        raise ValueError("style_strength must be finite and in [0,1]")
    if strength.shape[0] not in (1, q.shape[0]):
        raise ValueError("style_strength batch size must be one or match q")
    lower = (2.0 * strength).expand_as(q)
    upper = 1.0 + (2.0 * strength - 1.0) * q
    return torch.where(strength <= 0.5, lower, upper)


def timestep_fourier_features(timestep: torch.Tensor, feature_dim: int = 32) -> torch.Tensor:
    if feature_dim <= 0 or feature_dim % 2:
        raise ValueError("timestep Fourier feature_dim must be a positive even integer")
    tau = normalize_canonical_timestep(timestep)
    frequencies = torch.arange(1, feature_dim // 2 + 1, device=tau.device, dtype=tau.dtype)
    phases = torch.pi * tau * frequencies.unsqueeze(0)
    return torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1)


class ResidualGateRouter(nn.Module):
    def __init__(
        self,
        group_count: int,
        timestep_embed_dim: int = 32,
        hidden_dim: int = 64,
        router_rank: int = 16,
        q_max: float = 0.5,
    ) -> None:
        super().__init__()
        if group_count <= 0:
            raise ValueError("group_count must be positive")
        if timestep_embed_dim <= 0 or timestep_embed_dim % 2:
            raise ValueError("timestep_embed_dim must be a positive even integer")
        if hidden_dim <= 0 or router_rank <= 0:
            raise ValueError("router hidden_dim and router_rank must be positive")
        if not 0.0 < q_max <= 0.5:
            raise ValueError("q_max must be in (0,0.5]")
        self.group_count = int(group_count)
        self.timestep_embed_dim = int(timestep_embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.router_rank = int(router_rank)
        self.q_max = float(q_max)
        self.net = nn.Sequential(
            nn.Linear(self.timestep_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.router_rank),
            nn.SiLU(),
            nn.Linear(self.router_rank, self.group_count),
        )
        final_projection = self.net[-1]
        nn.init.zeros_(final_projection.weight)
        nn.init.zeros_(final_projection.bias)

    def forward(self, canonical_tau: torch.Tensor) -> torch.Tensor:
        raw = self.net(timestep_fourier_features(canonical_tau, self.timestep_embed_dim))
        return self.q_max * torch.tanh(raw)


@dataclass(frozen=True)
class ResidualGateRuntime:
    gates: torch.Tensor
    registry_fingerprint: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.gates, torch.Tensor) or self.gates.ndim != 2:
            raise ValueError("residual gates must be a [B,G] tensor")
        if not bool(torch.isfinite(self.gates).all()):
            raise ValueError("residual gates must be finite")

    def apply(self, module: Any, residual: torch.Tensor) -> torch.Tensor:
        group_index = getattr(module, "_residual_gate_group_index", None)
        if group_index is None:
            raise RuntimeError(f"LoRA module has no pre-bound residual group index: {module_display_name(module)}")
        return apply_group_gate(residual, self.gates, int(group_index))


def apply_group_gate(residual: torch.Tensor, gates: torch.Tensor, group_index: int) -> torch.Tensor:
    if gates.ndim != 2:
        raise ValueError("residual gates must have shape [B,G]")
    if group_index < 0 or group_index >= gates.shape[1]:
        raise IndexError(f"residual group index {group_index} is outside [0,{gates.shape[1]})")
    gate = gates[:, group_index]
    if residual.shape[0] != gate.shape[0]:
        if residual.shape[0] % gate.shape[0] != 0:
            raise ValueError("residual batch size must match or be an integer multiple of gate batch size")
        gate = gate.repeat(residual.shape[0] // gate.shape[0])
    while gate.ndim < residual.ndim:
        gate = gate.unsqueeze(-1)
    return residual * gate.to(device=residual.device, dtype=residual.dtype)


def current_residual_gates() -> Optional[torch.Tensor]:
    return _CURRENT_RESIDUAL_GATES.get()


@contextlib.contextmanager
def residual_gate_tensor_context(gates: Optional[torch.Tensor]) -> Iterator[Optional[torch.Tensor]]:
    if gates is not None and gates.ndim != 2:
        raise ValueError("residual gates must have shape [B,G]")
    token = _CURRENT_RESIDUAL_GATES.set(gates)
    try:
        yield gates
    finally:
        _CURRENT_RESIDUAL_GATES.reset(token)


@contextlib.contextmanager
def residual_gate_runtime_context(network: Any, runtime: ResidualGateRuntime) -> Iterator[ResidualGateRuntime]:
    previous = getattr(network, "_residual_gate_runtime", None)
    network._residual_gate_runtime = runtime
    try:
        with residual_gate_tensor_context(runtime.gates):
            yield runtime
    finally:
        network._residual_gate_runtime = previous
