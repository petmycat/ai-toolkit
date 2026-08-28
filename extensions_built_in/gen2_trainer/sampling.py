from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from .unconditional import official_asymmetric_cfg


@dataclass(frozen=True)
class ValidationCase:
    prompt_index: int
    seed: int
    style_gate: float
    helper_off: bool
    eta_c: float
    eta_u: float


def build_validation_matrix(prompt_count: int, seeds: Iterable[int], eta_u_values: Iterable[float], include_sweep: bool = True) -> list[ValidationCase]:
    cases = []
    for prompt_index in range(prompt_count):
        for seed in seeds:
            cases.append(ValidationCase(prompt_index, int(seed), 1.0, False, 1.0, 0.0))
            cases.append(ValidationCase(prompt_index, int(seed), 0.0, True, 0.0, 0.0))
            if include_sweep:
                for eta_u in eta_u_values:
                    cases.append(ValidationCase(prompt_index, int(seed), 1.0, False, 1.0, float(eta_u)))
    return cases


def assert_gate_noop(base: torch.Tensor, gated: torch.Tensor, atol: float = 0.0, rtol: float = 0.0) -> None:
    if not torch.allclose(base, gated, atol=atol, rtol=rtol):
        raise AssertionError("style_gate=0 must preserve the base output")


def cfg_velocity_with_residuals(v_c: torch.Tensor, v_u: torch.Tensor, guidance_scale: float, delta_c: torch.Tensor, delta_u: torch.Tensor, eta_c: float, eta_u: float) -> torch.Tensor:
    return official_asymmetric_cfg(v_c, v_u, guidance_scale, eta_c, eta_u, delta_c, delta_u)


def sample_stratified_timesteps(
    batch_size: int,
    bins: int,
    order: torch.Tensor | None,
    cursor: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if bins < 1:
        raise ValueError("timestep_bins must be positive")
    if order is None or order.numel() != bins:
        order = torch.randperm(bins, device=device)
    else:
        order = order.to(device=device, dtype=torch.long)
    positions = torch.arange(cursor, cursor + batch_size, device=device)
    indices = order[positions.remainder(bins)]
    next_cursor = int((cursor + batch_size) % bins)
    offsets = torch.rand(batch_size, device=device)
    timesteps = ((indices.float() + offsets) / bins).clamp(0.0, 1.0) * 1000.0
    return timesteps, order, next_cursor


def make_flowmatch_noisy_latents(clean: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    t = (timestep.to(device=clean.device, dtype=torch.float32) / 1000.0).to(clean.dtype)
    t = t.view(-1, 1, 1, 1)
    if not torch.all((t >= 0.0) & (t <= 1.0)):
        raise ValueError("flow-matching timesteps must be on the 0..1000 model scale")
    return (1.0 - t) * clean + t * noise
