from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HelperEffectGeometry:
    projection: torch.Tensor
    orthogonal: torch.Tensor
    activator_norm: torch.Tensor
    projection_norm: torch.Tensor
    helper_norm_median: torch.Tensor
    alignment: torch.Tensor
    orthogonal_fraction: torch.Tensor
    magnitude_ratio: torch.Tensor
    valid: torch.Tensor
    rank: torch.Tensor


def helper_effect_geometry(
    helper_effects: torch.Tensor,
    activator_effect: torch.Tensor,
    rank: int = 0,
    energy_threshold: float = 0.99,
    epsilon: float = 1e-8,
) -> HelperEffectGeometry:
    if helper_effects.ndim < 3 or activator_effect.ndim < 2:
        raise ValueError("helper effects must have shape (helpers, batch, ...) and activator effect (batch, ...)")
    if helper_effects.shape[1] != activator_effect.shape[0]:
        raise ValueError("helper and activator batch dimensions must match")
    if rank < 0:
        raise ValueError("helper effect rank cannot be negative")
    if not 0.0 < energy_threshold <= 1.0:
        raise ValueError("helper effect energy threshold must be in (0, 1]")
    helpers = helper_effects.detach().float().permute(1, 0, *range(2, helper_effects.ndim)).flatten(2)
    activator = activator_effect.float().flatten(1)
    batch_size, helper_count, feature_count = helpers.shape
    helper_norms = helpers.norm(dim=2)
    gram = torch.bmm(helpers, helpers.transpose(1, 2))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0.0)
    descending = torch.argsort(eigenvalues, dim=1, descending=True)
    eigenvalues = torch.gather(eigenvalues, 1, descending)
    eigenvectors = torch.gather(eigenvectors, 2, descending.unsqueeze(1).expand(-1, helper_count, -1))
    cumulative = eigenvalues.cumsum(dim=1)
    total_energy = cumulative[:, -1:].clamp_min(epsilon)
    needed = (cumulative / total_energy >= energy_threshold).to(torch.long).argmax(dim=1) + 1
    positive_rank = (eigenvalues > epsilon).sum(dim=1)
    max_rank = helper_count if rank == 0 else min(rank, helper_count)
    selected_rank = needed.clamp_max(max_rank).minimum(positive_rank)
    components = torch.bmm(helpers.transpose(1, 2), eigenvectors)
    components = components / eigenvalues.clamp_min(epsilon).sqrt().unsqueeze(1)
    component_mask = torch.arange(helper_count, device=helpers.device).view(1, -1) < selected_rank.view(-1, 1)
    basis = components * component_mask.unsqueeze(1).to(components.dtype)
    coefficients = torch.bmm(basis.transpose(1, 2), activator.unsqueeze(2)).squeeze(2)
    projection = torch.bmm(basis, coefficients.unsqueeze(2)).squeeze(2).view_as(activator_effect)
    orthogonal = activator_effect.float() - projection.view_as(activator_effect)
    activator_norm = activator.norm(dim=1)
    projection_flat = projection
    projection_norm = projection_flat.norm(dim=1)
    helper_norm_median = helper_norms.median(dim=1).values
    alignment = (activator * projection_flat).sum(dim=1) / (activator_norm * projection_norm).clamp_min(epsilon)
    orthogonal_fraction = orthogonal.flatten(1).norm(dim=1) / activator_norm.clamp_min(epsilon)
    magnitude_ratio = projection_norm / helper_norm_median.clamp_min(epsilon)
    valid = (selected_rank > 0) & (helper_norm_median > epsilon)
    alignment = torch.where(valid, alignment.clamp(-1.0, 1.0), torch.zeros_like(alignment))
    orthogonal_fraction = torch.where(activator_norm > epsilon, orthogonal_fraction, torch.zeros_like(orthogonal_fraction))
    magnitude_ratio = torch.where(valid, magnitude_ratio, torch.zeros_like(magnitude_ratio))
    return HelperEffectGeometry(
        projection=projection.view_as(activator_effect),
        orthogonal=orthogonal,
        activator_norm=activator_norm,
        projection_norm=projection_norm,
        helper_norm_median=helper_norm_median,
        alignment=alignment,
        orthogonal_fraction=orthogonal_fraction,
        magnitude_ratio=magnitude_ratio,
        valid=valid,
        rank=selected_rank,
    )


def effect_magnitude_loss(
    projection_norm: torch.Tensor,
    helper_norm: torch.Tensor,
    target_ratio: float = 0.5,
    min_ratio: float = 0.1,
    max_ratio: float = 2.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if projection_norm.shape != helper_norm.shape:
        raise ValueError("projection and helper norms must have equal shapes")
    if not 0.0 < min_ratio <= target_ratio <= max_ratio:
        raise ValueError("magnitude ratios must satisfy 0 < min_ratio <= target_ratio <= max_ratio")
    ratio = projection_norm / helper_norm.clamp_min(epsilon)
    target = torch.as_tensor(target_ratio, device=ratio.device, dtype=ratio.dtype)
    log_ratio = torch.log(ratio.clamp_min(epsilon))
    target_error = (log_ratio - torch.log(target)).square()
    lower_error = F.softplus(min_ratio - ratio).square()
    upper_error = F.softplus(ratio - max_ratio).square()
    return (target_error + lower_error + upper_error).mean()


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


def geometric_alignment_loss(activator_effect: torch.Tensor, projection: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    if activator_effect.shape != projection.shape:
        raise ValueError("activator effect and projection must have equal shapes")
    activator = activator_effect.float().flatten(1)
    projected = projection.float().flatten(1)
    valid = (activator.norm(dim=1) > epsilon) & (projected.norm(dim=1) > epsilon)
    if not torch.any(valid):
        return activator_effect.sum() * 0.0
    return (1.0 - F.cosine_similarity(activator[valid], projected[valid], dim=-1)).mean()


def geometric_orthogonal_loss(orthogonal: torch.Tensor, activator_norm: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    if orthogonal.ndim < 2 or activator_norm.ndim != 1 or orthogonal.shape[0] != activator_norm.shape[0]:
        raise ValueError("orthogonal effect and activator norm shapes are incompatible")
    fraction = orthogonal.float().flatten(1).norm(dim=1) / activator_norm.float().clamp_min(epsilon)
    valid = activator_norm.float() > epsilon
    if not torch.any(valid):
        return orthogonal.sum() * 0.0
    return fraction[valid].mean()

