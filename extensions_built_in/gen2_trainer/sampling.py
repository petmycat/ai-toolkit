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
    conditioning_mode: str
    style_gate: float
    eta_c: float
    eta_u: float
    helper_index: int = 0

    @property
    def helper_off(self) -> bool:
        return self.conditioning_mode == "helper" and self.style_gate == 0.0


def validation_condition_names() -> tuple[str, ...]:
    return (
        "helper_b_off", "helper_b_on", "activator_b_off",
        "activator_b_on_eta_u_0", "activator_b_on_eta_u_sweep_1",
        "activator_b_on_eta_u_sweep_2", "literal_e0_b_off", "literal_e0_b_on",
    )


def build_validation_matrix(
    prompt_count: int,
    seeds: Iterable[int],
    eta_u_values: Iterable[float],
    include_sweep: bool = True,
    designated_helper_index: int = 0,
) -> list[ValidationCase]:
    """Build exactly eight frozen conditions per prompt/seed."""
    sweep = [float(value) for value in eta_u_values if float(value) != 0.0]
    if include_sweep and len(sweep) != 2:
        raise ValueError("validation requires exactly two non-zero eta_u sweep values")
    if not include_sweep:
        sweep = []
    if not isinstance(designated_helper_index, int) or designated_helper_index < 0:
        raise ValueError("designated helper index must be a non-negative integer")
    cases: list[ValidationCase] = []
    for prompt_index in range(prompt_count):
        for seed in seeds:
            seed = int(seed)
            base_cases = [
                ValidationCase(prompt_index, seed, "helper", 0.0, 0.0, 0.0, designated_helper_index),
                ValidationCase(prompt_index, seed, "helper", 1.0, 1.0, 0.0, designated_helper_index),
                ValidationCase(prompt_index, seed, "activator", 0.0, 0.0, 0.0),
                ValidationCase(prompt_index, seed, "activator", 1.0, 1.0, 0.0),
            ]
            if include_sweep:
                base_cases.extend([
                    ValidationCase(prompt_index, seed, "activator", 1.0, 1.0, sweep[0]),
                    ValidationCase(prompt_index, seed, "activator", 1.0, 1.0, sweep[1]),
                ])
            base_cases.extend([
                ValidationCase(prompt_index, seed, "literal_e0", 0.0, 0.0, 0.0),
                ValidationCase(prompt_index, seed, "literal_e0", 1.0, 1.0, 0.0),
            ])
            cases.extend(base_cases)
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
