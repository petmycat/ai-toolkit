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
