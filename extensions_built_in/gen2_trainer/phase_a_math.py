from __future__ import annotations

import torch
import torch.nn.functional as F


def positive_cone_projection(
    helper_effects: torch.Tensor,
    activator_effect: torch.Tensor,
    iterations: int = 24,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project each activator effect onto the non-negative helper-effect cone."""
    if helper_effects.ndim < 3 or activator_effect.ndim < 2:
        raise ValueError("helper effects must have shape (helpers, batch, ...) and activator effect (batch, ...)")
    if helper_effects.shape[1] != activator_effect.shape[0]:
        raise ValueError("helper and activator batch dimensions must match")
    if iterations < 1:
        raise ValueError("cone projection iterations must be positive")
    helpers = helper_effects.detach().float().permute(1, 0, *range(2, helper_effects.ndim)).flatten(2)
    activator = activator_effect.float().flatten(1)
    gram = torch.bmm(helpers, helpers.transpose(1, 2))
    rhs = torch.bmm(helpers, activator.unsqueeze(2)).squeeze(2)
    coefficients = torch.linalg.lstsq(gram, rhs.unsqueeze(2)).solution.squeeze(2).clamp_min(0.0)
    lipschitz = gram.diagonal(dim1=1, dim2=2).sum(dim=1).clamp_min(epsilon)
    for _ in range(iterations):
        gradient = torch.bmm(gram, coefficients.unsqueeze(2)).squeeze(2) - rhs
        coefficients = (coefficients - gradient / lipschitz.unsqueeze(1)).clamp_min(0.0)
    projection_flat = torch.bmm(helpers.transpose(1, 2), coefficients.unsqueeze(2)).squeeze(2)
    return projection_flat.view_as(activator_effect), coefficients


def positive_cone_geometry(
    helper_effects: torch.Tensor,
    activator_effect: torch.Tensor,
    iterations: int = 24,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    projection, coefficients = positive_cone_projection(helper_effects, activator_effect, iterations, epsilon)
    activator = activator_effect.float().flatten(1)
    projected = projection.float().flatten(1)
    residual = activator - projected
    activator_norm = activator.norm(dim=1)
    projection_norm = projected.norm(dim=1)
    alignment = (activator * projected).sum(dim=1) / (activator_norm * projection_norm).clamp_min(epsilon)
    orthogonal_fraction = residual.norm(dim=1) / activator_norm.clamp_min(epsilon)
    valid = (activator_norm > epsilon) & (projection_norm > epsilon) & (coefficients.sum(dim=1) > epsilon)
    return {
        "projection": projection,
        "coefficients": coefficients,
        "alignment": torch.where(valid, alignment.clamp(-1.0, 1.0), torch.zeros_like(alignment)),
        "orthogonal_fraction": torch.where(activator_norm > epsilon, orthogonal_fraction, torch.zeros_like(orthogonal_fraction)),
        "activator_norm": activator_norm,
        "projection_norm": projection_norm,
        "valid": valid,
    }


def normalized_teacher_distillation_loss(
    activator_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
    effect_scale: torch.Tensor | float = 1.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if activator_prediction.shape != teacher_prediction.shape:
        raise ValueError("activator and teacher predictions must have equal shapes")
    scale = torch.as_tensor(effect_scale, device=activator_prediction.device, dtype=torch.float32).clamp_min(epsilon)
    error = (activator_prediction.float() - teacher_prediction.detach().float()).flatten(1).square().mean(dim=1)
    return (error / scale.square()).mean()


def effect_geometry_gate(
    train_geometry: dict[str, float],
    heldout_geometry: dict[str, float],
    min_alignment: float = 0.5,
    max_orthogonal_fraction: float = 0.5,
    min_magnitude_ratio: float = 0.1,
    max_magnitude_ratio: float = 2.0,
) -> tuple[bool, dict[str, bool]]:
    if not 0.0 <= min_alignment <= 1.0:
        raise ValueError("minimum alignment must be in 0..1")
    if not 0.0 <= max_orthogonal_fraction <= 1.0:
        raise ValueError("maximum orthogonal fraction must be in 0..1")
    if min_magnitude_ratio <= 0.0 or max_magnitude_ratio < min_magnitude_ratio:
        raise ValueError("magnitude release range is invalid")
    checks = {
        "train_alignment": train_geometry.get("alignment", 0.0) >= min_alignment,
        "heldout_alignment": heldout_geometry.get("alignment", 0.0) >= min_alignment,
        "train_orthogonal": train_geometry.get("orthogonal_fraction", 1.0) <= max_orthogonal_fraction,
        "heldout_orthogonal": heldout_geometry.get("orthogonal_fraction", 1.0) <= max_orthogonal_fraction,
        "train_magnitude": min_magnitude_ratio <= train_geometry.get("magnitude_ratio", 0.0) <= max_magnitude_ratio,
        "heldout_magnitude": min_magnitude_ratio <= heldout_geometry.get("magnitude_ratio", 0.0) <= max_magnitude_ratio,
        "train_valid": train_geometry.get("valid_fraction", 0.0) > 0.0,
        "heldout_valid": heldout_geometry.get("valid_fraction", 0.0) > 0.0,
    }
    return all(checks.values()), checks
