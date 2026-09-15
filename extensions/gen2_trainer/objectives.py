"""Explicit fp32 reductions. These functions contain no scheduler/sign conversion."""
from __future__ import annotations

import torch


def per_example_mse(prediction: torch.Tensor, target: torch.Tensor,
                    valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("Prediction and target need the same batched latent shape")
    squared = (prediction.float()-target.float()).square()
    if valid_mask is None:
        return squared.flatten(1).mean(1)
    mask = torch.broadcast_to(valid_mask.to(squared.device, torch.bool), squared.shape)
    counts = mask.flatten(1).sum(1)
    if bool((counts == 0).any()):
        raise ValueError("Each example must contain at least one valid latent element")
    return (squared*mask).flatten(1).sum(1)/counts


def styled_loss(prediction, target, valid_mask=None):
    return per_example_mse(prediction, target, valid_mask).mean()


def neutral_loss(prediction, teacher, valid_mask=None):
    return per_example_mse(prediction, teacher.detach(), valid_mask).mean()


def text_adapter_ratio(base: torch.Tensor, residual: torch.Tensor,
                       suffix_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-example ratio and suffix numerator/denominator samples from one layer."""
    if base.shape != residual.shape or base.shape[:2] != suffix_mask.shape:
        raise ValueError("Encoder residual/base/mask shapes disagree")
    numerator = residual.float().square().mean(-1)
    denominator = base.float().square().mean(-1).detach() + 1e-6
    count = suffix_mask.sum(-1)
    if bool((count == 0).any()):
        raise ValueError("R_T requires suffix positions in every example")
    ratio = ((numerator/denominator)*suffix_mask).sum(-1)/count
    return ratio, numerator.masked_select(suffix_mask), denominator.masked_select(suffix_mask)
