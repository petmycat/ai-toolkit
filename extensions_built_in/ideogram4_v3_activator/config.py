from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


STAGE_NAMES = ("semantic_activator", "te_calibration")


@dataclass(frozen=True)
class StageArtifactsConfig:
    metrics_file: str = "metrics.jsonl"
    validation_file: str = "heldout_validation.jsonl"
    checkpoint_dir: str = "checkpoints"
    final_dir: str = "final"
    best_dir: str = "best"
    embedding_filename: str = "trigger_embedding.safetensors"
    te_adapter_filename: str = "te_adapter.safetensors"


@dataclass(frozen=True)
class StageConfig:
    name: str
    steps: int = 80
    optimizer: str = "adamw"
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    learning_rates: Dict[str, float] = field(default_factory=dict)
    save_steps: List[int] = field(default_factory=lambda: [20, 40, 60, 80])
    validation_steps: List[int] = field(default_factory=lambda: [20, 40, 60, 80])
    artifacts: StageArtifactsConfig = field(default_factory=StageArtifactsConfig)


@dataclass(frozen=True)
class Ideogram4V3ActivatorConfig:
    enabled: bool
    smoke_test: bool
    output_root: Optional[str]
    v3_weights: str
    literal_initialization: Dict[str, Any]
    helper_schedule: Dict[str, Any]
    dataset_schedule: Dict[str, Any]
    fixed_validation: Dict[str, Any]
    terminal_manual_ablation: Dict[str, Any]
    semantic_activator: StageConfig
    te_calibration: StageConfig


def _stage_config(name: str, raw: Mapping[str, Any]) -> StageConfig:
    artifacts = StageArtifactsConfig(**dict(raw.get("artifacts", {})))
    return StageConfig(
        name=name,
        steps=int(raw.get("steps", 80)),
        optimizer=str(raw.get("optimizer", "adamw")),
        optimizer_params=dict(raw.get("optimizer_params", {})),
        learning_rates={key: float(value) for key, value in raw.get("learning_rates", {}).items()},
        save_steps=[int(value) for value in raw.get("save_steps", [20, 40, 60, 80])],
        validation_steps=[int(value) for value in raw.get(
            "validation_steps",
            [0, 20, 40, 60, 80] if name == "te_calibration" else [20, 40, 60, 80],
        )],
        artifacts=artifacts,
    )


def load_config(raw: Mapping[str, Any]) -> Ideogram4V3ActivatorConfig:
    config = Ideogram4V3ActivatorConfig(
        enabled=bool(raw.get("enabled", True)),
        smoke_test=bool(raw.get("smoke_test", False)),
        output_root=raw.get("output_root"),
        v3_weights=str(raw.get("v3_weights", "")),
        literal_initialization=dict(raw.get("literal_initialization", {})),
        helper_schedule=dict(raw.get("helper_schedule", {})),
        dataset_schedule=dict(raw.get("dataset_schedule", {})),
        fixed_validation=dict(raw.get("fixed_validation", {})),
        terminal_manual_ablation=dict(raw.get("terminal_manual_ablation", {})),
        semantic_activator=_stage_config("semantic_activator", raw.get("semantic_activator", {})),
        te_calibration=_stage_config("te_calibration", raw.get("te_calibration", {})),
    )
    validate_config(config)
    return config


def validate_config(config: Ideogram4V3ActivatorConfig) -> None:
    if not config.enabled:
        return
    if not config.v3_weights.strip():
        raise ValueError("ideogram4_v3_activator.v3_weights must identify the frozen V3 snapshot/weights")
    expected_a1_steps = 5 if config.smoke_test else 80
    if config.semantic_activator.steps != expected_a1_steps:
        raise ValueError(
            f"Ideogram4 V3 semantic_activator must use exactly {expected_a1_steps} steps "
            f"when smoke_test={config.smoke_test}"
        )
    if config.te_calibration.steps <= 0:
        raise ValueError("Ideogram4 V3 te_calibration.steps must be positive")
    for stage in (config.semantic_activator, config.te_calibration):
        if not stage.learning_rates or any(value < 0 for value in stage.learning_rates.values()):
            raise ValueError(f"{stage.name}.learning_rates must contain non-negative values")
        if sorted(set(stage.save_steps)) != sorted(stage.save_steps):
            raise ValueError(f"{stage.name}.save_steps must be unique")
        if sorted(set(stage.validation_steps)) != sorted(stage.validation_steps):
            raise ValueError(f"{stage.name}.validation_steps must be unique")
        if any(step <= 0 or step > stage.steps for step in stage.save_steps):
            raise ValueError(f"{stage.name} save steps must lie in [1, {stage.steps}]")
        minimum_validation = 0 if stage.name == "te_calibration" else 1
        if any(step < minimum_validation or step > stage.steps for step in stage.validation_steps):
            raise ValueError(
                f"{stage.name} validation steps must lie in [{minimum_validation}, {stage.steps}]"
            )
    expected_a1_validation = [5] if config.smoke_test else [20, 40, 60, 80]
    if config.semantic_activator.validation_steps != expected_a1_validation:
        raise ValueError(
            "semantic_activator validation_steps must be "
            + "/".join(str(step) for step in expected_a1_validation)
        )
    if 0 not in config.te_calibration.validation_steps:
        raise ValueError("te_calibration validation_steps must include step 0 preflight")
    if config.te_calibration.steps not in config.te_calibration.validation_steps:
        raise ValueError("te_calibration validation_steps must include the final step")
    if config.te_calibration.steps not in config.te_calibration.save_steps:
        raise ValueError("te_calibration save_steps must include the final step")
    if int(config.literal_initialization.get("target_vectors", 4)) != 4:
        raise ValueError("literal_initialization.target_vectors must be exactly 4")
    required_validation = {"seed", "fixed_timesteps", "data_split_manifest"}
    missing = required_validation - set(config.fixed_validation)
    if missing:
        raise ValueError("fixed_validation is missing required fields: " + ", ".join(sorted(missing)))
    if not config.fixed_validation.get("fixed_timesteps"):
        raise ValueError("fixed_validation.fixed_timesteps must be non-empty")
    if any(
        int(timestep) < 0 or int(timestep) > 1000
        for timestep in config.fixed_validation["fixed_timesteps"]
    ):
        raise ValueError("fixed_validation.fixed_timesteps must lie in [0, 1000]")
    if not str(config.fixed_validation.get("data_split_manifest", "")).strip():
        raise ValueError("fixed_validation.data_split_manifest must be a non-empty path")
    if not config.helper_schedule:
        raise ValueError("helper_schedule must be explicitly fixed")
    helpers = [str(value).strip() for value in config.helper_schedule.get("helpers", [])]
    helper_weights = config.helper_schedule.get("weights", [])
    if not helpers or any(not helper for helper in helpers):
        raise ValueError("helper_schedule.helpers must contain non-empty helper phrases")
    if len(helper_weights) != len(helpers):
        raise ValueError("helper_schedule.weights must match helper_schedule.helpers")
    if any(float(weight) < 0 for weight in helper_weights) or sum(float(weight) for weight in helper_weights) <= 0:
        raise ValueError("helper_schedule.weights must be non-negative and sum to a positive value")
    if not config.dataset_schedule:
        raise ValueError("dataset_schedule must be explicitly fixed")
    sources = config.dataset_schedule.get("sources", [])
    source_names = [str(source.get("name", "")).lower() for source in sources]
    if len(source_names) != 1 or source_names[0] not in {"structured", "json"}:
        raise ValueError("A1 dataset_schedule must contain exactly one structured/json caption source")
    source_format = str(sources[0].get("format", "text")).lower()
    if source_format not in {"text", "json"}:
        raise ValueError("A1 structured caption source format must be text or json")
    if source_format == "json" and not str(sources[0].get("caption_field", "")).strip():
        raise ValueError(
            "A1 JSON-object captions require dataset_schedule.sources[0].caption_field; "
            "use format: text when .json sidecars contain raw caption text"
        )
    if config.terminal_manual_ablation.get("mode") not in {"manual", "external"}:
        raise ValueError("terminal_manual_ablation.mode must declare manual or external")
    if config.terminal_manual_ablation.get("automatic", False):
        raise ValueError("V3 residual ablation is terminal/manual and must not be automatic")
