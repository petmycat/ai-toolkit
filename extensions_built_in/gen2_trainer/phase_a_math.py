from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HelperCalibrationResult:
    gains: torch.Tensor
    median_gains: torch.Tensor
    positive_fractions: torch.Tensor
    reliable_mask: torch.Tensor
    weights: torch.Tensor


def calibrate_helper_losses(
    literal_losses: torch.Tensor,
    helper_losses: torch.Tensor,
    temperature: float,
    min_gain: float,
    min_positive_fraction: float,
) -> HelperCalibrationResult:
    if literal_losses.ndim != 1 or helper_losses.ndim != 2:
        raise ValueError("calibration losses must have shapes (probes,) and (helpers, probes)")
    if helper_losses.shape[1] != literal_losses.shape[0]:
        raise ValueError("helper and literal probe counts must match")
    if temperature <= 0:
        raise ValueError("helper calibration temperature must be positive")
    gains_by_probe = literal_losses.unsqueeze(0) - helper_losses
    gains = gains_by_probe.mean(dim=1)
    median_gains = gains_by_probe.median(dim=1).values
    positive_fractions = (gains_by_probe > 0).float().mean(dim=1)
    reliable_mask = (median_gains > min_gain) & (positive_fractions >= min_positive_fraction)
    if not torch.any(reliable_mask):
        raise RuntimeError("no helper reliably outperforms the literal bootstrap on calibration probes")
    logits = torch.full_like(gains, float("-inf"))
    logits[reliable_mask] = F.softplus(gains[reliable_mask] / temperature).log()
    weights = torch.softmax(logits, dim=0)
    weights = torch.where(reliable_mask, weights, torch.zeros_like(weights))
    return HelperCalibrationResult(gains, median_gains, positive_fractions, reliable_mask, weights)


def semantic_direction_loss(activator_delta: torch.Tensor, helper_delta: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    if activator_delta.shape != helper_delta.shape:
        raise ValueError("semantic deltas must have equal shape")
    valid = (activator_delta.norm(dim=-1) > epsilon) & (helper_delta.norm(dim=-1) > epsilon)
    if not torch.any(valid):
        return activator_delta.sum() * 0.0
    return (1.0 - F.cosine_similarity(activator_delta[valid].float(), helper_delta[valid].float(), dim=-1)).mean()


def disturbance_projection(
    activator_effect: torch.Tensor,
    helper_effect: torch.Tensor,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if activator_effect.shape != helper_effect.shape or activator_effect.ndim < 2:
        raise ValueError("prediction effects must have equal batched shapes")
    activator = activator_effect.float().flatten(1)
    helper = helper_effect.float().flatten(1)
    helper_power = helper.square().sum(dim=1)
    valid = helper_power > epsilon
    alpha = torch.zeros_like(helper_power)
    alpha[valid] = (activator[valid] * helper[valid]).sum(dim=1) / helper_power[valid]
    perpendicular = activator - alpha.unsqueeze(1) * helper
    activator_norm = activator.norm(dim=1)
    beta = perpendicular.norm(dim=1) / activator_norm.clamp_min(epsilon)
    beta = torch.where(activator_norm > epsilon, beta, torch.zeros_like(beta))
    return alpha, beta, valid


def robust_competence(
    literal_losses: torch.Tensor,
    activator_losses: torch.Tensor,
    helper_losses: torch.Tensor,
    minimum_advantage: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if literal_losses.shape != activator_losses.shape or literal_losses.shape != helper_losses.shape:
        raise ValueError("competence losses must have equal shapes")
    advantage = literal_losses - helper_losses
    valid = advantage > minimum_advantage
    if not torch.any(valid):
        raise RuntimeError("competence cannot be evaluated without helper-positive probes")
    ratios = (literal_losses[valid] - activator_losses[valid]) / advantage[valid]
    return ratios.median(), valid


def smooth_helper_weight(competence: float, initial_weight: float, decay_threshold: float, release_threshold: float) -> float:
    if release_threshold <= decay_threshold:
        raise ValueError("release threshold must exceed decay threshold")
    if competence < decay_threshold:
        return float(initial_weight)
    if competence >= release_threshold:
        return 0.0
    x = (release_threshold - competence) / (release_threshold - decay_threshold)
    smooth = x * x * (3.0 - 2.0 * x)
    return float(initial_weight * smooth)
