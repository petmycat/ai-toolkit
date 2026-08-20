from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import PurePath
from typing import Any, Mapping, Optional, Tuple


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a YAML boolean")
    return value


def _tuple_ints(value: Any, label: str) -> Tuple[int, ...]:
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a sequence of integers") from error


def _tuple_floats(value: Any, label: str) -> Tuple[float, ...]:
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a sequence of numbers") from error


def _reject_unknown(raw: Mapping[str, Any], dataclass_type, label: str) -> None:
    allowed = {item.name for item in fields(dataclass_type)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {unknown}")


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1.0e-2
    steps: int = 100
    noise_seeds: Tuple[int, ...] = (42, 43, 44)
    snapshot_steps: Tuple[int, ...] = (0, 5, 10, 25, 50, 75, 100)


@dataclass(frozen=True)
class ValidationConfig:
    noise_seeds: Tuple[int, ...] = (314159, 271828, 161803)
    global_scales: Tuple[float, ...] = (0.75, 0.875, 1.0, 1.125, 1.25)
    gradient_sign_scales: Tuple[float, ...] = (0.125, 0.25)
    topk: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 102)
    magnitude_alphas: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class NovelPromptConfig:
    enabled: bool = True
    prompts: Tuple[str, ...] = ()
    seeds: Tuple[int, ...] = (42,)
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance_scale: float = 7.0
    include_phase_c_v2: bool = True
    phase_c_v2_handoff: Optional[str] = None
    phase_c_v2_checkpoint: str = "best"


@dataclass(frozen=True)
class ArtifactConfig:
    config: str = "config.json"
    source_manifest: str = "source_manifest.json"
    group_registry: str = "group_registry.json"
    canonical_summary: str = "summaries/canonical_summary.json"
    gate_comparison: str = "summaries/gate_comparison.json"
    family_summary: str = "summaries/family_summary.json"
    completion_manifest: str = "completion_manifest.json"
    visual_manifest: str = "visuals/manifest.json"


@dataclass(frozen=True)
class C0Config:
    a2_contract: str
    residual_manifest: str
    registry: str
    residual_records: str
    dataset_root: str
    split_manifest: str
    output_root: str
    timesteps: Tuple[int, ...] = (100, 500, 900)
    same_timestep_content_batch: int = 4
    train_item_limit: Optional[int] = None
    validation_item_limit: Optional[int] = None
    seed: int = 42
    q_bound: float = 0.25
    gradient_checkpointing_every_n_blocks: int = 1
    resume: bool = False
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    novel_visuals: NovelPromptConfig = field(default_factory=NovelPromptConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)


def load_config(raw: Mapping[str, Any]) -> C0Config:
    _reject_unknown(raw, C0Config, "c0_joint_gate_oracle")
    optimizer_raw = dict(raw.get("optimizer", {}))
    validation_raw = dict(raw.get("validation", {}))
    visual_raw = dict(raw.get("novel_visuals", {}))
    artifacts_raw = dict(raw.get("artifacts", {}))
    _reject_unknown(optimizer_raw, OptimizerConfig, "optimizer")
    _reject_unknown(validation_raw, ValidationConfig, "validation")
    _reject_unknown(visual_raw, NovelPromptConfig, "novel_visuals")
    _reject_unknown(artifacts_raw, ArtifactConfig, "artifacts")
    required = {
        name: str(raw.get(name, ""))
        for name in (
            "a2_contract", "residual_manifest", "registry", "residual_records",
            "dataset_root", "split_manifest", "output_root",
        )
    }
    config = C0Config(
        **required,
        timesteps=_tuple_ints(raw.get("timesteps", (100, 500, 900)), "timesteps"),
        same_timestep_content_batch=int(raw.get("same_timestep_content_batch", 4)),
        train_item_limit=int(raw["train_item_limit"]) if raw.get("train_item_limit") is not None else None,
        validation_item_limit=int(raw["validation_item_limit"]) if raw.get("validation_item_limit") is not None else None,
        seed=int(raw.get("seed", 42)),
        q_bound=float(raw.get("q_bound", 0.25)),
        gradient_checkpointing_every_n_blocks=int(raw.get("gradient_checkpointing_every_n_blocks", 1)),
        resume=_strict_bool(raw.get("resume", False), "resume"),
        optimizer=OptimizerConfig(
            learning_rate=float(optimizer_raw.get("learning_rate", 1.0e-2)),
            steps=int(optimizer_raw.get("steps", 100)),
            noise_seeds=_tuple_ints(optimizer_raw.get("noise_seeds", (42, 43, 44)), "optimizer.noise_seeds"),
            snapshot_steps=_tuple_ints(optimizer_raw.get("snapshot_steps", (0, 5, 10, 25, 50, 75, 100)), "optimizer.snapshot_steps"),
        ),
        validation=ValidationConfig(
            noise_seeds=_tuple_ints(validation_raw.get("noise_seeds", (314159, 271828, 161803)), "validation.noise_seeds"),
            global_scales=_tuple_floats(validation_raw.get("global_scales", (0.75, 0.875, 1.0, 1.125, 1.25)), "validation.global_scales"),
            gradient_sign_scales=_tuple_floats(validation_raw.get("gradient_sign_scales", (0.125, 0.25)), "validation.gradient_sign_scales"),
            topk=_tuple_ints(validation_raw.get("topk", (1, 2, 4, 8, 16, 32, 64, 102)), "validation.topk"),
            magnitude_alphas=_tuple_floats(validation_raw.get("magnitude_alphas", (0.0, 0.25, 0.5, 0.75, 1.0)), "validation.magnitude_alphas"),
        ),
        novel_visuals=NovelPromptConfig(
            enabled=_strict_bool(visual_raw.get("enabled", True), "novel_visuals.enabled"),
            prompts=tuple(str(item) for item in visual_raw.get("prompts", ())),
            seeds=_tuple_ints(visual_raw.get("seeds", (42,)), "novel_visuals.seeds"),
            width=int(visual_raw.get("width", 1024)),
            height=int(visual_raw.get("height", 1024)),
            steps=int(visual_raw.get("steps", 30)),
            guidance_scale=float(visual_raw.get("guidance_scale", 7.0)),
            include_phase_c_v2=_strict_bool(visual_raw.get("include_phase_c_v2", True), "novel_visuals.include_phase_c_v2"),
            phase_c_v2_handoff=str(visual_raw["phase_c_v2_handoff"]) if visual_raw.get("phase_c_v2_handoff") else None,
            phase_c_v2_checkpoint=str(visual_raw.get("phase_c_v2_checkpoint", "best")),
        ),
        artifacts=ArtifactConfig(**artifacts_raw),
    )
    validate_config(config)
    return config


def validate_config(config: C0Config) -> None:
    missing = [name for name in (
        "a2_contract", "residual_manifest", "registry", "residual_records",
        "dataset_root", "split_manifest", "output_root",
    ) if not getattr(config, name).strip()]
    if missing:
        raise ValueError(f"C0 configuration is missing paths: {missing}")
    if config.timesteps != (100, 500, 900):
        raise ValueError("C0 Stage 1 requires exactly timesteps [100,500,900]")
    if config.resume:
        raise ValueError("C0 resume is not implemented; set resume: false to avoid silently restarting")
    if config.same_timestep_content_batch < 2:
        raise ValueError("same_timestep_content_batch must be at least 2")
    if config.train_item_limit is not None and config.train_item_limit < config.same_timestep_content_batch:
        raise ValueError("train_item_limit must cover one distinct same-timestep content batch")
    if config.validation_item_limit is not None and config.validation_item_limit <= 0:
        raise ValueError("validation_item_limit must be positive")
    if config.q_bound != 0.25:
        raise ValueError("primary C0 oracle requires q_bound=0.25")
    if not 1 <= config.gradient_checkpointing_every_n_blocks <= 34:
        raise ValueError("gradient_checkpointing_every_n_blocks must lie in [1,34]")
    if config.optimizer.steps <= 0 or config.optimizer.learning_rate <= 0 or not math.isfinite(config.optimizer.learning_rate):
        raise ValueError("optimizer steps and learning_rate must be positive and finite")
    if not config.optimizer.noise_seeds or len(set(config.optimizer.noise_seeds)) != len(config.optimizer.noise_seeds):
        raise ValueError("optimizer.noise_seeds must be non-empty and unique")
    if config.optimizer.steps < config.optimizer.noise_seeds.__len__():
        raise ValueError("optimizer.steps must allow every optimization noise seed to be used")
    if sorted(set(config.optimizer.snapshot_steps)) != list(config.optimizer.snapshot_steps):
        raise ValueError("optimizer.snapshot_steps must be sorted and unique")
    if not config.optimizer.snapshot_steps:
        raise ValueError("optimizer.snapshot_steps must be non-empty")
    if config.optimizer.snapshot_steps[0] != 0 or config.optimizer.snapshot_steps[-1] != config.optimizer.steps:
        raise ValueError("optimizer.snapshot_steps must include 0 and the final optimizer step")
    if any(step < 0 or step > config.optimizer.steps for step in config.optimizer.snapshot_steps):
        raise ValueError("optimizer.snapshot_steps are outside the configured run")
    if not config.validation.noise_seeds or len(set(config.validation.noise_seeds)) != len(config.validation.noise_seeds):
        raise ValueError("validation.noise_seeds must be non-empty and unique")
    overlap = sorted(set(config.optimizer.noise_seeds).intersection(config.validation.noise_seeds))
    if overlap:
        raise ValueError(f"optimization and validation noise seeds must be disjoint: {overlap}")
    if config.validation.global_scales != (0.75, 0.875, 1.0, 1.125, 1.25):
        raise ValueError("C0 global scales must be [0.75,0.875,1.0,1.125,1.25]")
    if any(value not in (0.125, 0.25) for value in config.validation.gradient_sign_scales):
        raise ValueError("gradient-sign scales may only be 0.125 or 0.25")
    if any(value <= 0 or value > 102 for value in config.validation.topk):
        raise ValueError("validation.topk values must lie in [1,102]")
    if config.validation.magnitude_alphas != (0.0, 0.25, 0.5, 0.75, 1.0):
        raise ValueError("C0 magnitude path must be [0,0.25,0.5,0.75,1]")
    visual = config.novel_visuals
    if visual.phase_c_v2_checkpoint not in {"best", "final"}:
        raise ValueError("novel_visuals.phase_c_v2_checkpoint must be best or final")
    if visual.width <= 0 or visual.height <= 0 or visual.steps <= 0 or visual.guidance_scale <= 0:
        raise ValueError("novel visual dimensions, steps and guidance_scale must be positive")
    if len(set(visual.seeds)) != len(visual.seeds):
        raise ValueError("novel_visuals.seeds must be unique")
    if visual.enabled and visual.prompts and not visual.seeds:
        raise ValueError("novel_visuals.seeds must be non-empty when prompts are configured")
    if visual.include_phase_c_v2 and visual.prompts and not visual.phase_c_v2_handoff:
        raise ValueError("Phase C V2 visual comparison requires phase_c_v2_handoff")
    if any(not prompt.strip() for prompt in visual.prompts):
        raise ValueError("novel_visuals.prompts must not contain empty strings; use [] for zero prompts")
    artifact_values = [getattr(config.artifacts, item.name) for item in fields(ArtifactConfig)]
    if any(not value or PurePath(value).is_absolute() or ".." in PurePath(value).parts for value in artifact_values):
        raise ValueError("C0 artifact paths must be safe non-empty relative paths")
    if len(set(artifact_values)) != len(artifact_values):
        raise ValueError("C0 artifact paths must be unique")
