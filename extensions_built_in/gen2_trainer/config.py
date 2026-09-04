from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping


_ALLOWED_MODES = {"auto", "phase_a_only", "phase_b_from_activator", "resume_phase_b"}


@dataclass(frozen=True)
class Gen2RuntimeConfig:
    mode: str
    placeholder: str
    helpers: tuple[dict[str, str], ...]
    phase_a: dict[str, Any]
    phase_b: dict[str, Any]
    unconditional: dict[str, Any]
    validation_prompts: tuple[str, ...]
    seeds: tuple[int, ...]
    dataset_split: dict[str, Any]
    prototype: dict[str, Any]
    controller: dict[str, Any]
    helpers_per_step: int
    qwen: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _forbidden_truthy(mapping: Mapping[str, Any], key: str) -> bool:
    value: Any = mapping
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return False
        value = value[part]
    return bool(value)


def _mapping_or_default(mapping: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key, {})
    _require(value is None or isinstance(value, Mapping), f"{key} must be a mapping")
    return dict(value or {})


_LEGACY_FIELDS = {
    "calibration", "effect_geometry", "mixture", "mixture_iterations", "mixture_weights", "positive_cone", "cone", "cone_iterations",
    "teacher", "teacher_weight", "content_preserve", "content_preserve_weight", "cross_content", "cross_content_weight",
    "trust_region", "trust_region_weight", "token_diversity", "token-diversity", "independent_tail",
}


def _reject_legacy_fields(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).replace("-", "_")
            if normalized in _LEGACY_FIELDS or str(key) in _LEGACY_FIELDS:
                raise ValueError(f"{path + '.' if path else ''}{key} is deprecated")
            _reject_legacy_fields(child, f"{path + '.' if path else ''}{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_legacy_fields(child, f"{path}[{index}]")


def validate_gen2_config(raw: Mapping[str, Any]) -> Gen2RuntimeConfig:
    _require(isinstance(raw, Mapping), "gen2 config must be a mapping")
    config = copy.deepcopy(dict(raw))
    _reject_legacy_fields(config)
    mode = str(config.get("mode", "auto"))
    _require(mode in _ALLOWED_MODES, f"gen2 mode must be one of {sorted(_ALLOWED_MODES)}")

    model = _mapping_or_default(config, "model")
    _require(model.get("arch") == "ideogram4", "gen2_trainer requires model.arch=ideogram4")
    _require(not model.get("unconditional_lora_path"), "gen2_trainer forbids model.unconditional_lora_path")
    for key in ("train_unconditional", "proxy_cfg"):
        _require(not _forbidden_truthy(config, key), f"gen2_trainer forbids {key}=true")
        _require(not _forbidden_truthy(model, key), f"gen2_trainer forbids model.{key}=true")
    _require(not _forbidden_truthy(config, "model.unconditional_lora_path"), "gen2_trainer forbids model.unconditional_lora_path")
    _require(not _forbidden_truthy(config, "network.unconditional_lora_path"), "gen2_trainer forbids network.unconditional_lora_path")

    unconditional = _mapping_or_default(model, "unconditional")
    _require(unconditional.get("enabled", True), "official unconditional transformer must be enabled")
    _require(unconditional.get("source", "official") == "official", "unconditional.source must be official")
    for key in ("dtype", "quantize", "qtype"):
        expected = model.get(key)
        actual = unconditional.get(key, expected)
        _require(actual == expected, f"unconditional.{key} must match conditional model.{key}")

    datasets = config.get("datasets") or []
    _require(isinstance(datasets, list) and len(datasets) == 1, "gen2_trainer requires exactly one dataset")
    _require(not config.get("datasets_reg"), "gen2_trainer forbids datasets_reg")
    helpers: list[dict[str, str]] = []
    placeholder = "[trigger]"
    for index, dataset in enumerate(datasets):
        _require(not dataset.get("trigger_word"), f"datasets[{index}].trigger_word is not supported")
        _require(not dataset.get("cache_text_embeddings", False), "gen2 uses its own conditioning cache")
        _require(dataset.get("caption_ext", "json").lstrip(".") == "json", "gen2 captions must use JSON sidecars")
        current_placeholder = dataset.get("activator_placeholder", "[trigger]")
        _require(current_placeholder == "[trigger]", "V1 activator_placeholder must be [trigger]")
        current_helpers = dataset.get("helpers")
        if current_helpers is None:
            current_helpers = config.get("helpers")
            if isinstance(current_helpers, Mapping):
                current_helpers = current_helpers.get("helpers")
        _require(current_helpers is not None, f"datasets[{index}].helpers must be inline; helper_bank paths are not supported")
        _require(isinstance(current_helpers, list) and bool(current_helpers), f"datasets[{index}].helpers must be a non-empty list")
        for helper_index, helper in enumerate(current_helpers):
            _require(isinstance(helper, Mapping), f"datasets[{index}].helpers[{helper_index}] must be an object")
            _require(bool(helper.get("id")), f"datasets[{index}].helpers[{helper_index}].id is required")
            _require(isinstance(helper.get("replacement"), str) and helper["replacement"], f"datasets[{index}].helpers[{helper_index}].replacement must be a non-empty string")
        if not helpers:
            helpers = [dict(item) for item in current_helpers]
        else:
            _require(current_helpers == helpers, "V1 requires one shared inline helper list across datasets")

    activator = _mapping_or_default(config, "activator")
    _require(activator.get("type", "soft_tokens") == "soft_tokens", "activator.type must be soft_tokens")
    _require(activator.get("insertion", "pre_qwen_inputs_embeds") == "pre_qwen_inputs_embeds", "activator.insertion must be pre_qwen_inputs_embeds")
    _require(activator.get("share_across_occurrences", True) is True, "V1 requires share_across_occurrences=true")
    _require(activator.get("occurrence_mode", "all_placeholders") == "all_placeholders", "V1 requires occurrence_mode=all_placeholders")
    initialization = activator.get("initialization") or {}
    _require(initialization.get("strategy", "literal_trigger_resampled") == "literal_trigger_resampled", "unsupported activator initialization strategy")
    _require(isinstance(initialization.get("literal", ""), str) and initialization.get("literal", "").strip(), "activator.initialization.literal is required")
    _require(float(initialization.get("perturbation_std", 0.0)) == 0.0, "literal initialization must be deterministic")
    _require(int(activator.get("tokens", 24)) == 24, "Gen2 activator.tokens must be exactly 24")
    if "trigger_local_adapter" in activator:
        raise ValueError("activator.trigger_local_adapter is deprecated; use qwen.per_layer_adapter")
    if mode in {"phase_b_from_activator", "resume_phase_b"}:
        _require(bool(activator.get("artifact_path")), f"{mode} requires activator.artifact_path")

    network = _mapping_or_default(config, "network")
    _require(network.get("type") == "temporal_rank_field_lora", "network.type must be temporal_rank_field_lora")
    _require(float(network.get("dropout", 0.0)) == 0.0, "V1 currently requires network.dropout=0")
    _require(network.get("target_modules", {}).get("attention", {}).get("q", True), "Gen2 requires attention q target")
    _require(network.get("target_modules", {}).get("attention", {}).get("k", True), "Gen2 requires attention k target")
    _require(network.get("target_modules", {}).get("attention", {}).get("v", True), "Gen2 requires attention v target")
    _require(network.get("training_scope", "conditional_only") == "conditional_only", "network.training_scope must be conditional_only")
    _require(network.get("image_token_only", True) is True, "network.image_token_only must be true")
    _require(network.get("blocks", "all") == "all", "V1 requires network.blocks=all")
    _require(int(network.get("rank", 32)) > 0, "network.rank must be positive")
    temporal = network.get("temporal") or {}
    _require(int(temporal.get("knots", 8)) >= 2, "network.temporal.knots must be at least 2")
    _require(float(temporal.get("delta_max", 1.0)) >= 0.0, "network.temporal.delta_max must be non-negative")
    _require(temporal.get("interpolation", "linear") == "linear", "only linear temporal interpolation is supported")
    _require(temporal.get("parameterization", "bounded_tanh") == "bounded_tanh", "only bounded_tanh temporal parameterization is supported")
    _require(temporal.get("init", 0.0) == 0.0, "temporal field must initialize at zero")

    sample = _mapping_or_default(config, "sample")
    _require(not bool(sample.get("disable_sampling", False)), "Gen2 smoke requires sample.disable_sampling=false")
    phase_b_steps = int(((config.get("train") or {}).get("phase_b") or {}).get("steps", 0))
    sample_every = int(sample.get("sample_every", phase_b_steps or 1))
    _require(sample_every == phase_b_steps or phase_b_steps == 0, "Gen2 V1 runs official validation after Phase B; sample.sample_every must equal phase_b.steps")
    prompts = tuple(sample.get("validation_prompts") or ())
    _require(bool(prompts), "sample.validation_prompts must contain inline JSON prompts")
    _require(len(set(prompts)) == len(prompts), "sample.validation_prompts must not contain duplicates")
    for index, prompt in enumerate(prompts):
        _require(isinstance(prompt, str), f"validation prompt {index} must be a JSON string")
        try:
            parsed = json.loads(prompt)
        except json.JSONDecodeError as error:
            raise ValueError(f"validation prompt {index} must be valid JSON") from error
        _require(isinstance(parsed, dict), f"validation prompt {index} must decode to a JSON object")
        _require("[trigger]" in prompt, f"validation prompt {index} is missing [trigger]")
    designated_helper_id = sample.get("designated_helper_id")
    _require(isinstance(designated_helper_id, str) and designated_helper_id.strip(), "sample.designated_helper_id is required")
    helper_ids = {str(helper["id"]) for helper in helpers}
    _require(designated_helper_id in helper_ids, "sample.designated_helper_id must identify an inline helper")
    seeds = tuple(sample.get("seeds") or ())
    _require(bool(seeds) and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds), "sample.seeds must be a non-empty integer list")
    scale_sweep = sample.get("unconditional_scale_sweep") or []
    _require(isinstance(scale_sweep, (list, tuple)) and len(scale_sweep) == 2, "sample.unconditional_scale_sweep must contain exactly two values")
    _require(all(float(scale) > 0.0 for scale in scale_sweep), "sample.unconditional_scale_sweep must contain exactly two positive values")
    _require(float(scale_sweep[0]) < float(scale_sweep[1]), "sample.unconditional_scale_sweep must be strictly increasing")
    _require(bool(sample.get("require_official_unconditional", True)), "Gen2 requires sample.require_official_unconditional=true")
    style_adapter = sample.get("style_adapter") or {}
    for key in ("conditional_scale", "unconditional_scale"):
        _require(float(style_adapter.get(key, 1.0 if key == "conditional_scale" else 0.0)) >= 0.0, f"style_adapter.{key} must be non-negative")

    dataset_split = _mapping_or_default(config, "dataset_split")
    dataset_split.setdefault("enabled", True)
    dataset_split.setdefault("artifact_path", "gen2/dataset_split.json")
    dataset_split.setdefault("seed", 1234)
    dataset_split.setdefault("heldout_count", 2)
    _require(dataset_split["enabled"] is True, "dataset_split.enabled must be true")
    _require(isinstance(dataset_split["artifact_path"], str) and bool(dataset_split["artifact_path"].strip()), "dataset_split.artifact_path must be a non-empty string")
    _require(isinstance(dataset_split["seed"], int) and not isinstance(dataset_split["seed"], bool) and dataset_split["seed"] >= 0, "dataset_split.seed must be a non-negative integer")
    _require(isinstance(dataset_split["heldout_count"], int) and not isinstance(dataset_split["heldout_count"], bool) and dataset_split["heldout_count"] >= 0, "dataset_split.heldout_count must be non-negative")

    prototype = _mapping_or_default(config, "prototype")
    prototype.setdefault("enabled", True)
    prototype.setdefault("ema_decay", 0.95)
    prototype.setdefault("weight", 0.1)
    prototype.setdefault("response_floor", 1e-8)
    _require(prototype["enabled"] is True, "prototype.enabled must be true")
    _require(0.0 <= float(prototype["ema_decay"]) < 1.0, "prototype.ema_decay must be in [0, 1)")
    _require(math.isfinite(float(prototype["weight"])) and float(prototype["weight"]) >= 0.0, "prototype.weight must be finite and non-negative")
    _require(math.isfinite(float(prototype["response_floor"])) and float(prototype["response_floor"]) > 0.0, "prototype.response_floor must be finite and positive")

    controller = _mapping_or_default(config, "controller")
    controller_defaults = {
        "enabled": True, "total_semantic_budget": 1.0, "initial_fraction": 0.75,
        "min_fraction": 0.0, "max_fraction": 0.75, "evaluation_every": 5,
        "ema_decay": 0.8, "conflict_threshold": -0.1, "recovery_threshold": 0.0,
        "release_patience": 2, "recovery_patience": 2, "release_step": 0.1, "recovery_step": 0.05,
        "min_hold_steps": 1,
    }
    for key, default in controller_defaults.items():
        controller.setdefault(key, default)
    _require(controller["enabled"] is True, "controller.enabled must be true")
    _require(0.0 <= float(controller["min_fraction"]) <= float(controller["initial_fraction"]) <= float(controller["max_fraction"]) <= 1.0, "controller fraction bounds must be within [0, 1]")
    _require(float(controller["total_semantic_budget"]) >= 0.0, "controller.total_semantic_budget must be non-negative")
    _require(isinstance(controller["evaluation_every"], int) and controller["evaluation_every"] >= 1, "controller.evaluation_every must be positive")
    _require(0.0 <= float(controller["ema_decay"]) < 1.0, "controller.ema_decay must be in [0, 1)")
    _require(isinstance(controller["release_patience"], int) and not isinstance(controller["release_patience"], bool) and controller["release_patience"] >= 1, "controller.release_patience must be a positive integer")
    _require(isinstance(controller["recovery_patience"], int) and not isinstance(controller["recovery_patience"], bool) and controller["recovery_patience"] >= 1, "controller.recovery_patience must be a positive integer")
    _require(float(controller["conflict_threshold"]) < float(controller["recovery_threshold"]), "controller conflict/recovery thresholds must define hysteresis")
    _require(float(controller["release_step"]) > 0.0 and float(controller["recovery_step"]) > 0.0, "controller steps must be positive")

    helpers_per_step = config.get("helpers_per_step", 1)
    _require(isinstance(helpers_per_step, int) and not isinstance(helpers_per_step, bool) and 1 <= helpers_per_step <= 5, "helpers_per_step must be an integer from 1 to 5")
    _require(helpers_per_step <= len(helpers), "helpers_per_step cannot exceed the configured helper count")

    qwen = _mapping_or_default(config, "qwen")
    qwen_adapter = _mapping_or_default(qwen, "per_layer_adapter")
    qwen_adapter.setdefault("enabled", True)
    qwen_adapter.setdefault("layers", "all")
    qwen_adapter.setdefault("rank", 4)
    qwen_adapter.setdefault("alpha", 4.0)
    qwen_adapter.setdefault("weight_decay", 0.0)
    _require(isinstance(qwen_adapter["enabled"], bool), "qwen.per_layer_adapter.enabled must be boolean")
    _require(qwen_adapter["layers"] == "all" or isinstance(qwen_adapter["layers"], (list, tuple)), "qwen.per_layer_adapter.layers must be a list or 'all'")
    if qwen_adapter["enabled"]:
        _require(qwen_adapter["layers"] == "all", "enabled Qwen adapter must cover all decoder layers")
        _require(qwen_adapter["rank"] == 4, "enabled Qwen adapter rank must be exactly 4")
        _require(float(qwen_adapter["alpha"]) == 4.0, "enabled Qwen adapter alpha must be exactly 4")
    else:
        _require(qwen_adapter["layers"] == "all", "Qwen adapter layers must be 'all' even when disabled")
    _require(isinstance(qwen_adapter["rank"], int) and not isinstance(qwen_adapter["rank"], bool) and qwen_adapter["rank"] >= 1, "qwen.per_layer_adapter.rank must be a positive integer")
    _require(float(qwen_adapter["alpha"]) >= 0.0, "qwen.per_layer_adapter.alpha must be non-negative")
    _require("lr" in qwen_adapter and float(qwen_adapter["lr"]) > 0.0 and math.isfinite(float(qwen_adapter["lr"])), "qwen.per_layer_adapter.lr must be finite and positive")
    _require(float(qwen_adapter["weight_decay"]) >= 0.0, "qwen.per_layer_adapter.weight_decay must be non-negative")
    qwen["per_layer_adapter"] = qwen_adapter

    train = config.get("train") or {}
    phase_a = train.get("phase_a") or {}
    phase_b = train.get("phase_b") or {}
    _require(int(phase_a.get("steps", 0)) >= 0 and int(phase_b.get("steps", 0)) >= 1, "Phase A steps must be non-negative and Phase B must contain at least one step")
    _require(phase_b.get("enabled", True) is True, "Phase B must remain enabled")
    _require(int(phase_b.get("batch_size", train.get("batch_size", 1))) == 2, "Phase B batch size must be 2")
    _require(int(train.get("batch_size", 1)) >= 1, "train.batch_size must be positive")
    _require(int(phase_a.get("batch_size", train.get("batch_size", 1))) >= 1, "phase_a.batch_size must be positive")
    _require(int(phase_b.get("batch_size", train.get("batch_size", 1))) >= 1, "phase_b.batch_size must be positive")
    for phase_name, phase in (("phase_a", phase_a), ("phase_b", phase_b)):
        optimizer = phase.get("optimizer", config.get("train", {}).get("optimizer", "adamw"))
        _require(isinstance(optimizer, str) and bool(optimizer.strip()), f"{phase_name}.optimizer must be a non-empty string supported by ai-toolkit")
        optimizer_params = phase.get("optimizer_params", {})
        _require(isinstance(optimizer_params, Mapping), f"{phase_name}.optimizer_params must be a mapping")
    for phase_name, phase in (("phase_a", phase_a), ("phase_b", phase_b)):
        _require(phase.get("timestep_sampling", "stratified_uniform") == "stratified_uniform", f"{phase_name}.timestep_sampling must be stratified_uniform")
        _require(int(phase.get("timestep_bins", 10)) >= 2, f"{phase_name}.timestep_bins must be at least 2")
    probes = phase_a.get("probes") or {}
    _require(isinstance(probes, Mapping), "phase_a.probes must be a mapping")
    allowed_probe_keys = {"enabled", "probe_seed", "timestep_count", "regions", "json_path", "tensor_path", "artifact_path"}
    unknown_probe_keys = set(probes) - allowed_probe_keys
    _require(not unknown_probe_keys, f"phase_a.probes contains unsupported fields: {sorted(unknown_probe_keys)}")
    _require(bool(probes.get("enabled", True)), "phase_a.probes.enabled must be true")
    _require("probe_seed" in probes and isinstance(probes["probe_seed"], int) and not isinstance(probes["probe_seed"], bool) and probes["probe_seed"] >= 0, "phase_a.probes.probe_seed must be a non-negative integer")
    _require(isinstance(probes.get("timestep_count", 1), int) and not isinstance(probes.get("timestep_count", 1), bool) and int(probes.get("timestep_count", 1)) >= 1, "phase_a.probes.timestep_count must be positive")
    _require(isinstance(probes.get("regions", {}), Mapping), "phase_a.probes.regions must be a mapping")
    from .probes import validate_regions as _validate_probe_regions
    normalized_regions = {}
    for name, region in probes["regions"].items():
        _require(isinstance(region, Mapping), f"phase_a.probes.regions.{name} must be a mapping")
        _require(set(region) <= {"min", "max"}, f"phase_a.probes.regions.{name} contains unsupported fields")
        normalized_regions[name] = {**region, "timestep_count": int(probes["timestep_count"]), "pair_count": 0}
    _validate_probe_regions(normalized_regions)
    _require(float(phase_b.get("temporal_mean_weight", 0.0)) == 0.0, "temporal_mean_weight is diagnostic-only and must be zero")
    save = config.get("save") or {}
    for phase_name in ("phase_a", "phase_b"):
        phase_save = save.get(phase_name) or {}
        _require(int(phase_save.get("save_every", 0)) >= 0, f"save.{phase_name}.save_every cannot be negative")
        _require(int(phase_save.get("max_step_saves_to_keep", 0)) >= 0, f"save.{phase_name}.max_step_saves_to_keep cannot be negative")
    sample["designated_helper_id"] = designated_helper_id
    sample["unconditional_scale_sweep"] = list(scale_sweep)
    return Gen2RuntimeConfig(
        mode=mode,
        placeholder=placeholder,
        helpers=tuple(helpers),
        phase_a=dict(phase_a),
        phase_b=dict(phase_b),
        unconditional=dict(unconditional),
        validation_prompts=prompts,
        seeds=seeds,
        dataset_split=dict(dataset_split),
        prototype=dict(prototype),
        controller=dict(controller),
        helpers_per_step=helpers_per_step,
        qwen=dict(qwen),
    )
