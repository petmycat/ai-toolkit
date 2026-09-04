from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass
class ControllerState:
    fraction: float = 1.0
    released: bool = False
    unavailable: bool = False
    unavailable_reason: str | None = None
    ema_cosine: float | None = None
    ema_prototype: list[float] | dict[str, list[float]] | None = None
    good_streak: int = 0
    bad_streak: int = 0
    hold_steps: int = 0
    valid_evals: int = 0


def _flatten_gradient(gradients: tuple[torch.Tensor | None, ...] | list[torch.Tensor | None]) -> torch.Tensor | None:
    flat = [gradient.detach().float().reshape(-1) for gradient in gradients if gradient is not None]
    if not flat:
        return None
    vector = torch.cat(flat)
    return vector if torch.isfinite(vector).all() else None


def isolated_global_gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...],
    *,
    retain_graph: bool = False,
) -> tuple[list[torch.Tensor | None], dict[str, Any]]:
    if not loss.requires_grad:
        return [None for _ in parameters], {"valid": False, "reason": "loss_detached"}
    gradients = torch.autograd.grad(loss, tuple(parameters), retain_graph=retain_graph, allow_unused=True)
    vector = _flatten_gradient(gradients)
    if vector is None:
        return list(gradients), {"valid": False, "reason": "no_connected_parameters"}
    return list(gradients), {"valid": True, "reason": None, "norm": float(vector.norm().item()), "parameter_count": sum(g is not None for g in gradients)}


def gradient_cosine(
    gradients_left: list[torch.Tensor | None] | tuple[torch.Tensor | None, ...],
    gradients_right: list[torch.Tensor | None] | tuple[torch.Tensor | None, ...],
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor | None:
    if len(gradients_left) != len(gradients_right):
        raise ValueError("gradient collections must have equal length")
    left_parts, right_parts = [], []
    for left, right in zip(gradients_left, gradients_right):
        if left is None and right is None:
            continue
        reference = left if left is not None else right
        if left is not None and right is not None and left.shape != right.shape:
            raise ValueError("paired gradients must have equal shapes")
        left_parts.append(torch.zeros_like(reference, dtype=torch.float32).reshape(-1) if left is None else left.float().reshape(-1))
        right_parts.append(torch.zeros_like(reference, dtype=torch.float32).reshape(-1) if right is None else right.float().reshape(-1))
    if not left_parts:
        return None
    left_flat, right_flat = torch.cat(left_parts), torch.cat(right_parts)
    if not torch.isfinite(left_flat).all() or not torch.isfinite(right_flat).all():
        return None
    if left_flat.norm() <= epsilon or right_flat.norm() <= epsilon:
        return None
    return F.cosine_similarity(left_flat[None], right_flat[None], dim=1, eps=epsilon)[0]


def isolated_gradient_cosine(
    loss_left: torch.Tensor,
    loss_right: torch.Tensor,
    parameters: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...],
    *,
    epsilon: float = 1e-8,
    retain_graph: bool = True,
) -> torch.Tensor | None:
    if not loss_left.requires_grad or not loss_right.requires_grad:
        return None
    grads_left = torch.autograd.grad(loss_left, tuple(parameters), retain_graph=True, allow_unused=True)
    grads_right = torch.autograd.grad(loss_right, tuple(parameters), retain_graph=retain_graph, allow_unused=True)
    return gradient_cosine(grads_left, grads_right, epsilon=epsilon)


class AdaptiveJointController:
    """Hysteretic helper-fraction controller driven by valid post-step agreement probes."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.min_fraction = float(config.get("min_fraction", 0.0))
        self.max_fraction = float(config.get("max_fraction", 1.0))
        self.step_up = float(config.get("release_step", config.get("step_up", 0.1)))
        self.step_down = float(config.get("recovery_step", config.get("step_down", 0.05)))
        self.release_cosine = float(config.get("recovery_threshold", config.get("release_cosine", 0.0)))
        self.recovery_cosine = float(config.get("conflict_threshold", config.get("recovery_cosine", -0.1)))
        self.ema_decay = float(config.get("ema_decay", 0.8))
        self.patience = int(config.get("release_patience", config.get("patience", 2)))
        self.recovery_patience = int(config.get("recovery_patience", 2))
        self.min_hold_steps = int(config.get("min_hold_steps", 1))
        if not 0 <= self.min_fraction <= self.max_fraction <= 1 or self.step_up < 0 or self.step_down < 0 or not 0 <= self.ema_decay < 1 or self.patience < 1 or self.recovery_patience < 1 or self.min_hold_steps < 0:
            raise ValueError("invalid controller configuration")
        initial_fraction = float(config.get("initial_fraction", self.max_fraction))
        if not self.min_fraction <= initial_fraction <= self.max_fraction:
            raise ValueError("initial_fraction must be within controller fraction bounds")
        self.state = ControllerState(fraction=initial_fraction, hold_steps=0)

    @property
    def fraction(self) -> float:
        return self.state.fraction

    def mark_unavailable(self, reason: str) -> None:
        self.state.unavailable, self.state.unavailable_reason = True, str(reason)
        self.state.good_streak = self.state.bad_streak = 0

    def update(self, cosine: torch.Tensor | float | None, prototype: torch.Tensor | Mapping[str, torch.Tensor] | None = None, *, gradient_diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
        if cosine is None:
            self.mark_unavailable("no_probe_diagnostics")
            result = self.diagnostics()
            if gradient_diagnostics is not None:
                result["gradients"] = gradient_diagnostics
            return result
        value = torch.as_tensor(cosine, dtype=torch.float32)
        if value.numel() == 0 or not torch.isfinite(value).all():
            self.mark_unavailable("non_finite_probe_diagnostics")
            result = self.diagnostics()
            if gradient_diagnostics is not None:
                result["gradients"] = gradient_diagnostics
            return result
        current = float(value.mean())
        self.state.unavailable = False
        self.state.unavailable_reason = None
        self.state.valid_evals += 1
        self.state.ema_cosine = current if self.state.ema_cosine is None else self.ema_decay * self.state.ema_cosine + (1 - self.ema_decay) * current
        if prototype is not None:
            if isinstance(prototype, Mapping):
                self.state.ema_prototype = {str(k): v.detach().float().cpu().tolist() for k, v in prototype.items()}
            else:
                self.state.ema_prototype = prototype.detach().float().cpu().tolist()
        if self.state.valid_evals < 2:
            return self.diagnostics()
        self.state.hold_steps += 1
        if self.state.hold_steps < self.min_hold_steps:
            return self.diagnostics()
        if self.state.released:
            self.state.bad_streak = self.state.bad_streak + 1 if current >= self.release_cosine else 0
            self.state.good_streak = 0
            if self.state.bad_streak >= self.recovery_patience:
                self.state.released = False
                self.state.fraction = min(self.max_fraction, self.state.fraction + self.step_down)
                self.state.hold_steps = self.state.bad_streak = 0
        elif self.state.ema_cosine <= self.recovery_cosine:
            self.state.good_streak += 1
            self.state.bad_streak = 0
            if self.state.good_streak >= self.patience:
                self.state.released = True
                self.state.fraction = max(self.min_fraction, self.state.fraction - self.step_up)
                self.state.hold_steps = 0
        else:
            self.state.good_streak = 0
        result = self.diagnostics()
        if gradient_diagnostics is not None:
            result["gradients"] = gradient_diagnostics
        return result

    def diagnostics(self) -> dict[str, Any]:
        return {"fraction": self.state.fraction, **asdict(self.state)}

    def state_dict(self) -> dict[str, Any]:
        return asdict(self.state)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.state = ControllerState(**{key: state[key] for key in asdict(self.state) if key in state})
        self.state.fraction = min(self.max_fraction, max(self.min_fraction, self.state.fraction))


__all__ = ["ControllerState", "AdaptiveJointController", "gradient_cosine", "isolated_global_gradients", "isolated_gradient_cosine"]
