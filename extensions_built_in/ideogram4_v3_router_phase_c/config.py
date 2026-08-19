from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import PurePath
from typing import Any, Mapping, Optional, Tuple


@dataclass(frozen=True)
class TemporalSmoothnessConfig:
    enabled: bool = True
    universal_only: bool = True
    weight: float = 1.0e-4


@dataclass(frozen=True)
class StyleStrengthConfig:
    min: float = 0.0
    max: float = 1.0
    default: float = 0.5
    step: float = 0.01


@dataclass(frozen=True)
class ValidationConfig:
    every: int = 25
    seeds: Tuple[int, ...] = (42, 314159, 271828)
    canonical_timesteps: Tuple[int, ...] = (100, 500, 900)
    dense_timesteps: Tuple[int, ...] = tuple(range(0, 1001, 50))
    evaluate_universal_only: bool = True
    evaluate_full_router: bool = True
    evaluate_context_shuffle: bool = True
    log_temporal_svd: bool = True
    log_cross_timestep_cosine: bool = True
    log_cross_content_cosine: bool = True


@dataclass(frozen=True)
class ArtifactConfig:
    training_metrics: str = "training_metrics.jsonl"
    validation_metrics: str = "validation_metrics.jsonl"
    temporal_profiles: str = "temporal_profiles.jsonl"
    contextual_profiles: str = "contextual_profiles.jsonl"
    group_registry: str = "group_registry.json"
    checkpoints_dir: str = "checkpoints"
    best_dir: str = "best"
    final_dir: str = "final"
    router_filename: str = "router.safetensors"
    router_config_filename: str = "router_config.json"
    source_manifest: str = "source_manifest.json"
    handoff_manifest: str = "handoff_manifest.json"
    completion_manifest: str = "completion_manifest.json"


@dataclass(frozen=True)
class PhaseCRouterConfig:
    a2_contract: str
    residual_manifest: str
    registry: str
    dataset_root: str
    split_manifest: str
    output_root: str
    conditioning_source: str = "projected_private_activator_states"
    activator_token_count: int = 4
    activator_token_dim: int = 4
    temporal_anchor_count: int = 16
    temporal_interpolation: str = "linear"
    universal_branch: bool = True
    contextual_branch: bool = True
    contextual_rank: int = 4
    q_max: float = 0.5
    style_strength: StyleStrengthConfig = field(default_factory=StyleStrengthConfig)
    same_timestep_content_batch: int = 4
    distinct_items_per_update: bool = True
    timestep_sampling: str = "stratified_uniform"
    timestep_bins: int = 16
    optimizer: str = "adamw8bit"
    optimizer_params: Mapping[str, Any] = field(default_factory=dict)
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-4
    lambda_universal: float = 1.0e-3
    lambda_contextual: float = 2.0e-3
    contextual_mean_multiplier: float = 4.0
    temporal_smoothness: TemporalSmoothnessConfig = field(default_factory=TemporalSmoothnessConfig)
    steps: int = 500
    seed: int = 42
    checkpoint_every: int = 25
    resume: bool = True
    train_item_limit: Optional[int] = None
    validation_item_limit: Optional[int] = None
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)


def _tuple_of_ints(value: Any, label: str) -> Tuple[int, ...]:
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a sequence of integers") from error


def _reject_unknown_keys(raw: Mapping[str, Any], allowed, label: str) -> None:
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {label} fields: {unknown}")


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a YAML boolean, not {type(value).__name__}")
    return value


def load_config(raw: Mapping[str, Any]) -> PhaseCRouterConfig:
    _reject_unknown_keys(raw, {item.name for item in fields(PhaseCRouterConfig)}, "phase_c_v2")
    smooth_raw = dict(raw.get("temporal_smoothness", {}))
    strength_raw = dict(raw.get("style_strength", {}))
    validation_raw = dict(raw.get("validation", {}))
    artifacts_raw = dict(raw.get("artifacts", {}))
    _reject_unknown_keys(smooth_raw, {item.name for item in fields(TemporalSmoothnessConfig)}, "temporal_smoothness")
    _reject_unknown_keys(strength_raw, {item.name for item in fields(StyleStrengthConfig)}, "style_strength")
    _reject_unknown_keys(validation_raw, {item.name for item in fields(ValidationConfig)}, "validation")
    _reject_unknown_keys(artifacts_raw, {item.name for item in fields(ArtifactConfig)}, "artifacts")
    required = {name: str(raw.get(name, "")) for name in (
        "a2_contract", "residual_manifest", "registry", "dataset_root", "split_manifest", "output_root"
    )}
    config = PhaseCRouterConfig(
        **required,
        conditioning_source=str(raw.get("conditioning_source", "projected_private_activator_states")),
        activator_token_count=int(raw.get("activator_token_count", 4)),
        activator_token_dim=int(raw.get("activator_token_dim", 4)),
        temporal_anchor_count=int(raw.get("temporal_anchor_count", 16)),
        temporal_interpolation=str(raw.get("temporal_interpolation", "linear")),
        universal_branch=_strict_bool(raw.get("universal_branch", True), "universal_branch"),
        contextual_branch=_strict_bool(raw.get("contextual_branch", True), "contextual_branch"),
        contextual_rank=int(raw.get("contextual_rank", 4)),
        q_max=float(raw.get("q_max", 0.5)),
        style_strength=StyleStrengthConfig(**strength_raw),
        same_timestep_content_batch=int(raw.get("same_timestep_content_batch", 4)),
        distinct_items_per_update=_strict_bool(raw.get("distinct_items_per_update", True), "distinct_items_per_update"),
        timestep_sampling=str(raw.get("timestep_sampling", "stratified_uniform")),
        timestep_bins=int(raw.get("timestep_bins", 16)),
        optimizer=str(raw.get("optimizer", "adamw8bit")).strip().lower(),
        optimizer_params=dict(raw.get("optimizer_params", {})),
        learning_rate=float(raw.get("learning_rate", 5.0e-4)),
        weight_decay=float(raw.get("weight_decay", 1.0e-4)),
        lambda_universal=float(raw.get("lambda_universal", 1.0e-3)),
        lambda_contextual=float(raw.get("lambda_contextual", 2.0e-3)),
        contextual_mean_multiplier=float(raw.get("contextual_mean_multiplier", 4.0)),
        temporal_smoothness=TemporalSmoothnessConfig(
            enabled=_strict_bool(smooth_raw.get("enabled", True), "temporal_smoothness.enabled"),
            universal_only=_strict_bool(smooth_raw.get("universal_only", True), "temporal_smoothness.universal_only"),
            weight=float(smooth_raw.get("weight", 1.0e-4)),
        ),
        steps=int(raw.get("steps", 500)),
        seed=int(raw.get("seed", 42)),
        checkpoint_every=int(raw.get("checkpoint_every", 25)),
        resume=_strict_bool(raw.get("resume", True), "resume"),
        train_item_limit=int(raw["train_item_limit"]) if raw.get("train_item_limit") is not None else None,
        validation_item_limit=int(raw["validation_item_limit"]) if raw.get("validation_item_limit") is not None else None,
        validation=ValidationConfig(
            every=int(validation_raw.get("every", 25)),
            seeds=_tuple_of_ints(validation_raw.get("seeds", (42, 314159, 271828)), "validation.seeds"),
            canonical_timesteps=_tuple_of_ints(validation_raw.get("canonical_timesteps", (100, 500, 900)), "validation.canonical_timesteps"),
            dense_timesteps=_tuple_of_ints(validation_raw.get("dense_timesteps", tuple(range(0, 1001, 50))), "validation.dense_timesteps"),
            evaluate_universal_only=_strict_bool(validation_raw.get("evaluate_universal_only", True), "validation.evaluate_universal_only"),
            evaluate_full_router=_strict_bool(validation_raw.get("evaluate_full_router", True), "validation.evaluate_full_router"),
            evaluate_context_shuffle=_strict_bool(validation_raw.get("evaluate_context_shuffle", True), "validation.evaluate_context_shuffle"),
            log_temporal_svd=_strict_bool(validation_raw.get("log_temporal_svd", True), "validation.log_temporal_svd"),
            log_cross_timestep_cosine=_strict_bool(validation_raw.get("log_cross_timestep_cosine", True), "validation.log_cross_timestep_cosine"),
            log_cross_content_cosine=_strict_bool(validation_raw.get("log_cross_content_cosine", True), "validation.log_cross_content_cosine"),
        ),
        artifacts=ArtifactConfig(**artifacts_raw),
    )
    validate_config(config)
    return config


def validate_config(config: PhaseCRouterConfig) -> None:
    missing = sorted(name for name in ("a2_contract", "residual_manifest", "registry", "dataset_root", "split_manifest", "output_root") if not getattr(config, name).strip())
    if missing:
        raise ValueError("Phase C V2 configuration is missing paths: " + ", ".join(missing))
    if config.conditioning_source != "projected_private_activator_states":
        raise ValueError("Phase C V2 requires projected_private_activator_states")
    if config.activator_token_count != 4:
        raise ValueError("current A1/A2 contract requires exactly four activator tokens")
    if not 1 <= config.activator_token_dim <= 8 or config.activator_token_count * config.activator_token_dim > 32:
        raise ValueError("activator code must be positive and no larger than 32 dimensions")
    if config.temporal_anchor_count < 2 or config.temporal_interpolation != "linear":
        raise ValueError("Phase C V2 requires at least two linearly interpolated temporal anchors")
    if not config.universal_branch or not config.contextual_branch or config.contextual_rank <= 0:
        raise ValueError("Phase C V2 requires universal and contextual branches with positive rank")
    numeric = {
        "q_max": config.q_max, "learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
        "lambda_universal": config.lambda_universal, "lambda_contextual": config.lambda_contextual,
        "contextual_mean_multiplier": config.contextual_mean_multiplier,
        "temporal_smoothness.weight": config.temporal_smoothness.weight,
    }
    bad = sorted(name for name, value in numeric.items() if not math.isfinite(float(value)))
    if bad:
        raise ValueError("Phase C V2 numeric fields must be finite: " + ", ".join(bad))
    if not 0.0 < config.q_max <= 0.5 or config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("invalid q_max, learning_rate, or weight_decay")
    if min(config.lambda_universal, config.lambda_contextual, config.contextual_mean_multiplier, config.temporal_smoothness.weight) < 0:
        raise ValueError("regularization weights must be non-negative")
    strength = config.style_strength
    if (strength.min, strength.max, strength.default, strength.step) != (0.0, 1.0, 0.5, 0.01):
        raise ValueError("style_strength contract must remain [0,1], default 0.5, step 0.01")
    if config.same_timestep_content_batch <= 1 or not config.distinct_items_per_update:
        raise ValueError("Phase C V2 requires multiple distinct contents per same-timestep update")
    if config.timestep_sampling != "stratified_uniform" or not 2 <= config.timestep_bins <= 100:
        raise ValueError("timestep_bins must lie in [2,100] for stratified_uniform sampling")
    if config.steps <= 0 or config.checkpoint_every <= 0 or config.checkpoint_every > config.steps:
        raise ValueError("steps/checkpoint_every are invalid")
    if config.validation.every <= 0 or config.validation.every > config.steps:
        raise ValueError("validation.every must lie in [1, steps]")
    required_validation_modes = {
        "evaluate_universal_only": config.validation.evaluate_universal_only,
        "evaluate_full_router": config.validation.evaluate_full_router,
        "log_temporal_svd": config.validation.log_temporal_svd,
        "log_cross_timestep_cosine": config.validation.log_cross_timestep_cosine,
        "log_cross_content_cosine": config.validation.log_cross_content_cosine,
    }
    disabled_modes = sorted(name for name, enabled in required_validation_modes.items() if not enabled)
    if disabled_modes:
        raise ValueError(
            "Phase C V2 scientific validation contract requires these modes/logs: " + ", ".join(disabled_modes)
        )
    if config.temporal_smoothness.enabled and not config.temporal_smoothness.universal_only:
        raise ValueError("initial Phase C V2 only supports universal temporal smoothness")
    for name, limit in (("train_item_limit", config.train_item_limit), ("validation_item_limit", config.validation_item_limit)):
        if limit is not None and limit <= 0:
            raise ValueError(f"{name} must be positive when configured")
    if config.train_item_limit is not None and config.train_item_limit < config.same_timestep_content_batch:
        raise ValueError("train_item_limit must cover same_timestep_content_batch distinct items")
    if config.validation.evaluate_context_shuffle and config.validation_item_limit == 1:
        raise ValueError("context-shuffled validation requires at least two validation items")
    for name, values in (("seeds", config.validation.seeds), ("canonical_timesteps", config.validation.canonical_timesteps), ("dense_timesteps", config.validation.dense_timesteps)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"validation.{name} must be non-empty and unique")
    if any(value < 0 or value > 1000 for value in config.validation.canonical_timesteps + config.validation.dense_timesteps):
        raise ValueError("validation timesteps must lie in [0,1000]")
    supported = {"adam", "adamw", "adam8", "adamw8", "adam8bit", "adamw8bit", "ademamix8bit", "lion", "lion8bit", "adagrad", "adafactor", "prodigy", "prodigy8bit", "dadaptation", "dadaptationadam", "dadaptationlion", "automagic", "automagic2", "automagic3", "automagicexperiment"}
    if config.optimizer not in supported:
        raise ValueError(f"unsupported Phase C optimizer: {config.optimizer}")
    if set(config.optimizer_params) & {"lr", "learning_rate", "params", "weight_decay"}:
        raise ValueError("optimizer_params must not override learning rate or params")
    try:
        json.dumps(dict(config.optimizer_params), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("optimizer_params must be finite JSON-compatible values") from error
    artifact_values = [getattr(config.artifacts, item.name) for item in fields(ArtifactConfig)]
    if any(not value or PurePath(value).is_absolute() or ".." in PurePath(value).parts for value in artifact_values):
        raise ValueError("Phase C artifact paths must be non-empty relative paths without '..'")
    if len(set(artifact_values)) != len(artifact_values):
        raise ValueError("Phase C artifact paths must be unique")
