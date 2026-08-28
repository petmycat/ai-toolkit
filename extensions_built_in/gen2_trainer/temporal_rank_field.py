from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class AdapterTarget:
    name: str
    kind: str
    row_slice: tuple[int, int] | None
    input_features: int
    output_features: int


class TemporalField(nn.Module):
    def __init__(self, knots: int = 8, rank: int = 32, delta_max: float = 1.0) -> None:
        super().__init__()
        if knots < 2 or rank < 1 or delta_max < 0:
            raise ValueError("invalid temporal field dimensions")
        self.knots = knots
        self.rank = rank
        self.delta_max = float(delta_max)
        self.values = nn.Parameter(torch.zeros(knots, rank))
        self.register_buffer("knot_positions", torch.linspace(0.0, 1.0, knots), persistent=True)

    def centered_values(self) -> torch.Tensor:
        values = self.values.float()
        step = 1.0 / (self.knots - 1)
        weights = torch.ones(self.knots, device=values.device, dtype=values.dtype)
        weights[[0, -1]] = 0.5
        trapezoid_mean = step * (values * weights[:, None]).sum(dim=0)
        return values - trapezoid_mean[None, :]

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim != 1:
            raise ValueError("TemporalField expects timestep with shape (batch,)")
        t = timestep.float().clamp(0.0, 1.0)
        values = self.centered_values()
        scaled = t * (self.knots - 1)
        left = scaled.floor().long().clamp(max=self.knots - 2)
        right = left + 1
        frac = (scaled - left).unsqueeze(-1)
        interpolated = values[left] * (1.0 - frac) + values[right] * frac
        return torch.exp(self.delta_max * torch.tanh(interpolated))


class TemporalRankFieldLoRA(nn.Module):
    def __init__(self, input_features: int, output_features: int, rank: int = 32, alpha: float | None = None, knots: int = 8, delta_max: float = 1.0, row_slice: tuple[int, int] | None = None) -> None:
        super().__init__()
        if row_slice is not None and row_slice[1] - row_slice[0] != output_features:
            raise ValueError("row slice length must equal output_features")
        self.input_features = input_features
        self.output_features = output_features
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        self.row_slice = row_slice
        self.down = nn.Parameter(torch.empty(rank, input_features))
        self.up = nn.Parameter(torch.zeros(output_features, rank))
        self.temporal = TemporalField(knots, rank, delta_max)
        nn.init.kaiming_uniform_(self.down, a=5 ** 0.5)

    def residual(self, x: torch.Tensor, timestep: torch.Tensor, image_token_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, sequence, features)")
        if image_token_mask is None:
            image_token_mask = torch.ones(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)
        scale = self.temporal(timestep).to(dtype=x.dtype).unsqueeze(1)
        raw = torch.matmul(torch.matmul(x, self.down.t()) * scale, self.up.t())
        return raw * image_token_mask.to(dtype=raw.dtype) * (self.alpha / self.rank)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, image_token_mask: torch.Tensor | None = None, style_gate: torch.Tensor | None = None, branch_scale: torch.Tensor | float = 1.0) -> torch.Tensor:
        if style_gate is None:
            style_gate = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        gate = style_gate.reshape(-1, 1, 1).to(dtype=x.dtype)
        scale = torch.as_tensor(branch_scale, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
        return self.residual(x, timestep, image_token_mask) * gate * scale


def time_smooth_regularizer(fields: Iterable[TemporalField]) -> torch.Tensor:
    values = [field.centered_values() for field in fields]
    if not values:
        raise ValueError("at least one temporal field is required")
    losses = []
    for value in values:
        losses.append((value[2:] - 2 * value[1:-1] + value[:-2]).square().sum())
    return torch.stack(losses).sum()


def time_mean_diagnostic(fields: Iterable[TemporalField]) -> torch.Tensor:
    means = [field.values.float().mean(dim=0) for field in fields]
    if not means:
        raise ValueError("at least one temporal field is required")
    return torch.stack(means)
