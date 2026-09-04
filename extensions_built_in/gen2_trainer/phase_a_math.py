from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


def _check_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have equal shapes")
    if prediction.ndim < 2:
        raise ValueError("tensors must have a batch dimension")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("inputs must be finite")


def effective_sample_weighted_reduce(
    values: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    dim: int | tuple[int, ...] | None = None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Reduce sample losses using normalized effective (non-zero) sample weights."""
    values = values.float()
    if not torch.isfinite(values).all():
        raise ValueError("values must be finite")
    if weights is None:
        return values.mean() if dim is None else values.mean(dim=dim)
    weights = torch.as_tensor(weights, device=values.device, dtype=values.dtype)
    if weights.ndim != 1 or values.shape[0] != weights.numel():
        raise ValueError("weights must be a vector matching the batch dimension")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and non-negative")
    shape = (weights.numel(),) + (1,) * (values.ndim - 1)
    weighted = values * weights.reshape(shape)
    denominator = weights.sum().clamp_min(epsilon)
    if dim is None:
        return weighted.sum() / denominator
    return weighted.sum(dim=dim) / denominator


def dataset_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared dataset loss, optionally weighted per effective sample."""
    _check_pair(prediction, target)
    if sample_weights is not None and weights is not None:
        raise ValueError("provide only one sample weight argument")
    per_sample = (prediction.float() - target.float()).square().flatten(1).mean(1)
    return effective_sample_weighted_reduce(per_sample, sample_weights if sample_weights is not None else weights)


def helper_direction_cosine(
    activator_prediction: torch.Tensor,
    helper_prediction: torch.Tensor,
    baseline_prediction: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample helper/activator direction cosine and its validity mask."""
    if activator_prediction.shape != helper_prediction.shape or activator_prediction.shape != baseline_prediction.shape:
        raise ValueError("activator, helper, and baseline predictions must have equal shapes")
    if activator_prediction.ndim < 2:
        raise ValueError("predictions must have a batch dimension")
    activator = (activator_prediction - baseline_prediction).float().flatten(1)
    helper = (helper_prediction - baseline_prediction).detach().float().flatten(1)
    finite = torch.isfinite(activator).all(dim=1) & torch.isfinite(helper).all(dim=1)
    safe_activator = torch.where(torch.isfinite(activator), activator, torch.zeros_like(activator))
    safe_helper = torch.where(torch.isfinite(helper), helper, torch.zeros_like(helper))
    activator_norm, helper_norm = safe_activator.norm(dim=1), safe_helper.norm(dim=1)
    valid = finite & (activator_norm > epsilon) & (helper_norm > epsilon)
    cosine = F.cosine_similarity(safe_activator, safe_helper, dim=1, eps=epsilon)
    return torch.where(valid, cosine.clamp(-1, 1), torch.zeros_like(cosine)), valid


def helper_subset_direction_loss(
    activator_prediction: torch.Tensor,
    helper_predictions: torch.Tensor,
    baseline_prediction: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce helper directions per helper, then per sample, without helper averaging first."""
    if helper_predictions.ndim != activator_prediction.ndim + 1:
        raise ValueError("helper predictions must be [helpers, batch, ...]")
    losses, cosines, validities = [], [], []
    for helper_prediction in helper_predictions:
        cosine, valid = helper_direction_cosine(activator_prediction, helper_prediction, baseline_prediction, epsilon=epsilon)
        losses.append(1.0 - cosine)
        cosines.append(cosine)
        validities.append(valid)
    loss_matrix = torch.stack(losses)
    cosine_matrix = torch.stack(cosines)
    valid_matrix = torch.stack(validities)
    weights = None if sample_weights is None else torch.as_tensor(sample_weights, device=loss_matrix.device, dtype=loss_matrix.dtype)
    sample_losses = []
    for index in range(loss_matrix.shape[1]):
        valid = valid_matrix[:, index]
        if valid.any():
            sample_losses.append(loss_matrix[:, index][valid].mean())
        else:
            sample_losses.append(loss_matrix[:, index].sum() * 0.0)
    sample_losses_tensor = torch.stack(sample_losses)
    sample_valid = valid_matrix.any(dim=0)
    effective_weights = sample_valid.float() if weights is None else weights * sample_valid.float()
    if sample_valid.any():
        reduced = effective_sample_weighted_reduce(sample_losses_tensor, effective_weights)
    else:
        reduced = sample_losses_tensor.sum() * 0.0
    return reduced, cosine_matrix, sample_valid


def helper_direction_loss(
    activator_prediction: torch.Tensor,
    helper_prediction: torch.Tensor,
    baseline_prediction: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine, valid = helper_direction_cosine(activator_prediction, helper_prediction, baseline_prediction, epsilon=epsilon)
    losses = 1.0 - cosine
    if valid.any():
        weights = None if sample_weights is None else torch.as_tensor(sample_weights, device=losses.device, dtype=losses.dtype) * valid.float()
        loss = effective_sample_weighted_reduce(losses, weights)
    else:
        loss = losses.sum() * 0.0
    return loss, cosine


def response_field_signature(effect: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Create a resolution-independent, finite two-statistic response signature."""
    if effect.ndim < 2:
        raise ValueError("response field must have a batch dimension")
    values = effect.float().flatten(1)
    if not torch.isfinite(values).all():
        raise ValueError("response field must be finite")
    mean = values.mean(1)
    rms = values.square().mean(1).sqrt()
    return torch.stack((mean, rms), dim=1)


def timestep_region_masks(
    timesteps: torch.Tensor,
    regions: Mapping[str, Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    """Return mutually exclusive masks for contiguous user-defined timestep regions."""
    values = timesteps.detach().float().reshape(-1)
    result: dict[str, torch.Tensor] = {}
    for name, metadata in regions.items():
        start = int(metadata.get("min", metadata.get("start", -1)))
        end = int(metadata.get("max", metadata.get("end", -1)))
        upper = values <= end if end == 1000 else values < end
        result[str(name)] = (values >= start) & upper
    if result and not torch.stack(tuple(result.values())).sum(0).eq(1).all():
        raise ValueError("every timestep must belong to exactly one prototype region")
    return result


def spatial_response(effect: torch.Tensor) -> torch.Tensor:
    """Reduce the known [B,C,H,W] prediction layout over spatial dimensions only."""
    if effect.ndim != 4:
        raise ValueError("prototype response requires prediction layout [B,C,H,W]")
    values = effect.float().mean(dim=(-2, -1))
    if not torch.isfinite(values).all():
        raise ValueError("prototype response must be finite")
    return values


def prototype_consistency_loss(
    response: torch.Tensor,
    prototype: torch.Tensor,
    *,
    floor: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample cosine dissimilarity to a detached region prototype."""
    if response.ndim != 2 or prototype.ndim != 1 or response.shape[1] != prototype.numel():
        raise ValueError("prototype response must be [N,C] and prototype must be [C]")
    response = response.float()
    prototype = prototype.detach().float()
    response_finite = torch.isfinite(response).all(dim=1)
    prototype_finite = torch.isfinite(prototype).all()
    safe_response = torch.where(torch.isfinite(response), response, torch.zeros_like(response))
    safe_prototype = torch.where(torch.isfinite(prototype), prototype, torch.zeros_like(prototype))
    if not prototype_finite:
        return torch.zeros(response.shape[0], device=response.device, dtype=response.dtype), torch.zeros(response.shape[0], device=response.device, dtype=torch.bool)
    valid = response_finite & (safe_response.norm(dim=1) > floor) & (safe_prototype.norm() > floor)
    cosine = F.cosine_similarity(safe_response, safe_prototype.unsqueeze(0), dim=1, eps=floor)
    losses = 1.0 - cosine
    return torch.where(valid, losses, torch.zeros_like(losses)), valid


def detached_ema_prototype(
    previous: torch.Tensor | Mapping[str, torch.Tensor] | None,
    samples: torch.Tensor | Mapping[str, torch.Tensor],
    decay: float,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Update detached response prototypes; mappings provide independent timestep regions."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    if isinstance(samples, Mapping):
        old = previous if isinstance(previous, Mapping) else {}
        result = {str(region): value.detach().float().clone() for region, value in old.items()}
        for region, value in samples.items():
            key = str(region)
            result[key] = detached_ema_prototype(old.get(key), value, decay)
        return result
    prototype = samples.detach().float().mean(0)
    if previous is None:
        return prototype.detach()
    if isinstance(previous, Mapping):
        raise ValueError("previous EMA prototype type does not match samples")
    if previous.shape != prototype.shape:
        raise ValueError("EMA prototype shape must remain stable")
    return (decay * previous.detach().float() + (1.0 - decay) * prototype).detach()


def response_prototype_ema(
    previous: torch.Tensor | Mapping[str, torch.Tensor] | None,
    responses: torch.Tensor | Mapping[str, torch.Tensor],
    decay: float = 0.9,
) -> torch.Tensor | dict[str, torch.Tensor]:
    return detached_ema_prototype(previous, responses, decay)


__all__ = [
    "dataset_mse", "effective_sample_weighted_reduce", "helper_direction_cosine",
    "helper_direction_loss", "helper_subset_direction_loss", "prototype_consistency_loss", "response_field_signature", "spatial_response",
    "timestep_region_masks", "detached_ema_prototype", "response_prototype_ema",
]
