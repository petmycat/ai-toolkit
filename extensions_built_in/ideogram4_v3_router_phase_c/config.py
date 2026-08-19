from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import PurePath
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class TemporalSmoothnessConfig:
    enabled: bool = True
    weight: float = 1.0e-4
    delta: float = 0.02


@dataclass(frozen=True)
class ValidationConfig:
    every: int = 25
    seeds: Tuple[int, ...] = (42, 314159, 271828)
    canonical_timesteps: Tuple[int, ...] = (100, 500, 900)
    dense_timesteps: Tuple[int, ...] = tuple(range(0, 1001, 50))


@dataclass(frozen=True)
class ArtifactConfig:
    training_metrics: str = "training_metrics.jsonl"
    validation_metrics: str = "validation_metrics.jsonl"
    gate_profiles: str = "gate_profiles.jsonl"
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
    steps: int = 400
    seed: int = 42
    learning_rate: float = 1.0e-3
    optimizer: str = "adamw"
    optimizer_params: Mapping[str, Any] = field(default_factory=dict)
    weight_decay: float = 1.0e-4
    lambda_gate: float = 1.0e-3
    timestep_bins: int = 10
    timestep_sampling: str = "stratified_uniform"
    timestep_embed_dim: int = 32
    hidden_dim: int = 64
    router_rank: int = 16
    q_max: float = 0.5
    checkpoint_every: int = 25
    resume: bool = True
    train_item_limit: Optional[int] = None
    validation_item_limit: Optional[int] = None
    temporal_smoothness: TemporalSmoothnessConfig = field(default_factory=TemporalSmoothnessConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    router_api_module: Optional[str] = None


def _tuple_of_ints(value: Any, label: str) -> Tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a sequence of integers") from error
    return result


def _reject_unknown_keys(raw: Mapping[str, Any], allowed, label: str) -> None:
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {label} fields: {unknown}")


def load_config(raw: Mapping[str, Any]) -> PhaseCRouterConfig:
    _reject_unknown_keys(raw, {field.name for field in fields(PhaseCRouterConfig)}, "phase_c_router")
    smooth_raw = dict(raw.get("temporal_smoothness", {}))
    validation_raw = dict(raw.get("validation", {}))
    artifacts_raw = dict(raw.get("artifacts", {}))
    _reject_unknown_keys(smooth_raw, {field.name for field in fields(TemporalSmoothnessConfig)}, "temporal_smoothness")
    _reject_unknown_keys(validation_raw, {field.name for field in fields(ValidationConfig)}, "validation")
    _reject_unknown_keys(artifacts_raw, {field.name for field in fields(ArtifactConfig)}, "artifacts")
    config = PhaseCRouterConfig(
        a2_contract=str(raw.get("a2_contract", "")),
        residual_manifest=str(raw.get("residual_manifest", "")),
        registry=str(raw.get("registry", "")),
        dataset_root=str(raw.get("dataset_root", "")),
        split_manifest=str(raw.get("split_manifest", "")),
        output_root=str(raw.get("output_root", "")),
        steps=int(raw.get("steps", 400)),
        seed=int(raw.get("seed", 42)),
        learning_rate=float(raw.get("learning_rate", 1.0e-3)),
        optimizer=str(raw.get("optimizer", "adamw")).strip().lower(),
        optimizer_params=dict(raw.get("optimizer_params", {})),
        weight_decay=float(raw.get("weight_decay", 1.0e-4)),
        lambda_gate=float(raw.get("lambda_gate", 1.0e-3)),
        timestep_bins=int(raw.get("timestep_bins", 10)),
        timestep_sampling=str(raw.get("timestep_sampling", "stratified_uniform")),
        timestep_embed_dim=int(raw.get("timestep_embed_dim", 32)),
        hidden_dim=int(raw.get("hidden_dim", 64)),
        router_rank=int(raw.get("router_rank", 16)),
        q_max=float(raw.get("q_max", 0.5)),
        checkpoint_every=int(raw.get("checkpoint_every", 25)),
        resume=bool(raw.get("resume", True)),
        train_item_limit=(int(raw["train_item_limit"]) if raw.get("train_item_limit") is not None else None),
        validation_item_limit=(
            int(raw["validation_item_limit"])
            if raw.get("validation_item_limit") is not None
            else None
        ),
        temporal_smoothness=TemporalSmoothnessConfig(
            enabled=bool(smooth_raw.get("enabled", True)),
            weight=float(smooth_raw.get("weight", 1.0e-4)),
            delta=float(smooth_raw.get("delta", 0.02)),
        ),
        validation=ValidationConfig(
            every=int(validation_raw.get("every", 25)),
            seeds=_tuple_of_ints(validation_raw.get("seeds", (42, 314159, 271828)), "validation.seeds"),
            canonical_timesteps=_tuple_of_ints(
                validation_raw.get("canonical_timesteps", (100, 500, 900)),
                "validation.canonical_timesteps",
            ),
            dense_timesteps=_tuple_of_ints(
                validation_raw.get("dense_timesteps", tuple(range(0, 1001, 50))),
                "validation.dense_timesteps",
            ),
        ),
        artifacts=ArtifactConfig(**artifacts_raw),
        router_api_module=(str(raw["router_api_module"]) if raw.get("router_api_module") else None),
    )
    validate_config(config)
    return config


def validate_config(config: PhaseCRouterConfig) -> None:
    required_paths = {
        "a2_contract": config.a2_contract,
        "residual_manifest": config.residual_manifest,
        "registry": config.registry,
        "dataset_root": config.dataset_root,
        "split_manifest": config.split_manifest,
        "output_root": config.output_root,
    }
    missing = sorted(name for name, value in required_paths.items() if not value.strip())
    if missing:
        raise ValueError("Phase C router configuration is missing paths: " + ", ".join(missing))
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.seed < 0:
        raise ValueError("seed must be non-negative")
    numeric_values = {
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "lambda_gate": config.lambda_gate,
        "q_max": config.q_max,
        "temporal_smoothness.weight": config.temporal_smoothness.weight,
        "temporal_smoothness.delta": config.temporal_smoothness.delta,
    }
    non_finite = sorted(name for name, value in numeric_values.items() if not math.isfinite(float(value)))
    if non_finite:
        raise ValueError("Phase C numeric fields must be finite: " + ", ".join(non_finite))
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    supported_optimizers = {
        "adam",
        "adamw",
        "adam8",
        "adamw8",
        "adam8bit",
        "adamw8bit",
        "ademamix8bit",
        "lion",
        "lion8bit",
        "adagrad",
        "adafactor",
        "prodigy",
        "prodigy8bit",
        "dadaptation",
        "dadaptationadam",
        "dadaptationlion",
        "automagic",
        "automagic2",
        "automagic3",
        "automagicexperiment",
    }
    if config.optimizer not in supported_optimizers:
        raise ValueError(f"unsupported Phase C optimizer: {config.optimizer}")
    if not isinstance(config.optimizer_params, Mapping):
        raise ValueError("optimizer_params must be a mapping")
    forbidden_optimizer_params = sorted(
        set(config.optimizer_params) & {"lr", "learning_rate", "params", "weight_decay"}
    )
    if forbidden_optimizer_params:
        raise ValueError(
            "optimizer_params must not override learning rate or params: "
            + ", ".join(forbidden_optimizer_params)
        )
    try:
        json.dumps(dict(config.optimizer_params), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("optimizer_params must be finite JSON-compatible values") from error
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if config.lambda_gate < 0.0 or not config.lambda_gate < 1.0:
        raise ValueError("lambda_gate must lie in [0, 1)")
    if config.timestep_sampling != "stratified_uniform":
        raise ValueError("Phase C requires deterministic stratified_uniform timestep sampling")
    if config.timestep_bins <= 1 or config.timestep_bins > 100:
        raise ValueError("timestep_bins must lie in [2, 100]")
    if 1000 % config.timestep_bins:
        raise ValueError("timestep_bins must evenly divide 1000")
    for name, limit in (
        ("train_item_limit", config.train_item_limit),
        ("validation_item_limit", config.validation_item_limit),
    ):
        if limit is not None and limit <= 0:
            raise ValueError(f"{name} must be positive when configured")
    if config.timestep_embed_dim <= 0 or config.timestep_embed_dim % 2:
        raise ValueError("timestep_embed_dim must be a positive even integer")
    if config.hidden_dim <= 0 or config.router_rank <= 0:
        raise ValueError("router hidden_dim and router_rank must be positive")
    if config.router_rank > min(config.timestep_embed_dim, config.hidden_dim):
        raise ValueError("router_rank must not exceed timestep_embed_dim or hidden_dim")
    if not 0.0 < config.q_max <= 0.5:
        raise ValueError("q_max must lie in (0, 0.5]")
    if config.checkpoint_every <= 0 or config.checkpoint_every > config.steps:
        raise ValueError("checkpoint_every must lie in [1, steps]")
    if config.validation.every <= 0 or config.validation.every > config.steps:
        raise ValueError("validation.every must lie in [1, steps]")
    if not config.validation.seeds or len(set(config.validation.seeds)) != len(config.validation.seeds):
        raise ValueError("validation.seeds must be non-empty and unique")
    for name, values in (
        ("canonical_timesteps", config.validation.canonical_timesteps),
        ("dense_timesteps", config.validation.dense_timesteps),
    ):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"validation.{name} must be non-empty and unique")
        if any(value < 0 or value > 1000 for value in values):
            raise ValueError(f"validation.{name} must lie in [0, 1000]")
    smooth = config.temporal_smoothness
    if smooth.weight < 0.0 or smooth.weight >= 1.0:
        raise ValueError("temporal_smoothness.weight must lie in [0, 1)")
    if not 0.0 < smooth.delta <= 0.5:
        raise ValueError("temporal_smoothness.delta must lie in (0, 0.5]")
    artifact_values = [getattr(config.artifacts, field.name) for field in fields(ArtifactConfig)]
    if any(not value or PurePath(value).is_absolute() or ".." in PurePath(value).parts for value in artifact_values):
        raise ValueError("Phase C artifact paths must be non-empty relative paths without '..'")
    if len(set(artifact_values)) != len(artifact_values):
        raise ValueError("Phase C artifact paths must be unique")
