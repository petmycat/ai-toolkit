from __future__ import annotations

import copy
import json
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


def validate_gen2_config(raw: Mapping[str, Any]) -> Gen2RuntimeConfig:
    config = copy.deepcopy(dict(raw))
    mode = str(config.get("mode", "auto"))
    _require(mode in _ALLOWED_MODES, f"gen2 mode must be one of {sorted(_ALLOWED_MODES)}")

    model = config.get("model") or {}
    _require(model.get("arch") == "ideogram4", "gen2_trainer requires model.arch=ideogram4")
    _require(not model.get("unconditional_lora_path"), "gen2_trainer forbids model.unconditional_lora_path")
    for key in ("train_unconditional", "proxy_cfg"):
        _require(not _forbidden_truthy(config, key), f"gen2_trainer forbids {key}=true")
        _require(not _forbidden_truthy(model, key), f"gen2_trainer forbids model.{key}=true")
    _require(not _forbidden_truthy(config, "model.unconditional_lora_path"), "gen2_trainer forbids model.unconditional_lora_path")
    _require(not _forbidden_truthy(config, "network.unconditional_lora_path"), "gen2_trainer forbids network.unconditional_lora_path")

    unconditional = model.get("unconditional") or {}
    _require(unconditional.get("enabled", True), "official unconditional transformer must be enabled")
    _require(unconditional.get("source", "official") == "official", "unconditional.source must be official")
    for key in ("dtype", "quantize", "qtype"):
        expected = model.get(key)
        actual = unconditional.get(key, expected)
        _require(actual == expected, f"unconditional.{key} must match conditional model.{key}")

    datasets = config.get("datasets") or []
    _require(bool(datasets), "gen2_trainer requires at least one dataset")
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

    activator = config.get("activator") or {}
    _require(activator.get("type", "soft_tokens") == "soft_tokens", "activator.type must be soft_tokens")
    _require(activator.get("insertion", "pre_qwen_inputs_embeds") == "pre_qwen_inputs_embeds", "activator.insertion must be pre_qwen_inputs_embeds")
    _require(activator.get("share_across_occurrences", True) is True, "V1 requires share_across_occurrences=true")
    _require(activator.get("occurrence_mode", "all_placeholders") == "all_placeholders", "V1 requires occurrence_mode=all_placeholders")
    initialization = activator.get("initialization") or {}
    _require(initialization.get("strategy", "helper_centroid_resampled") == "helper_centroid_resampled", "unsupported activator initialization strategy")
    _require(float(initialization.get("perturbation_std", 0.0)) == 0.0, "V1 currently requires activator.initialization.perturbation_std=0")
    _require(int(activator.get("tokens", 8)) >= 1, "activator.tokens must be at least 1")
    if mode in {"phase_b_from_activator", "resume_phase_b"}:
        _require(bool(activator.get("artifact_path")), f"{mode} requires activator.artifact_path")

    network = config.get("network") or {}
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

    sample = config.get("sample") or {}
    _require(not bool(sample.get("disable_sampling", False)), "Gen2 smoke requires sample.disable_sampling=false")
    phase_b_steps = int(((config.get("train") or {}).get("phase_b") or {}).get("steps", 0))
    sample_every = int(sample.get("sample_every", phase_b_steps or 1))
    _require(sample_every == phase_b_steps or phase_b_steps == 0, "Gen2 V1 runs official validation after Phase B; sample.sample_every must equal phase_b.steps")
    prompts = tuple(sample.get("validation_prompts") or ())
    _require(bool(prompts), "sample.validation_prompts must contain inline JSON prompts")
    for index, prompt in enumerate(prompts):
        _require(isinstance(prompt, str), f"validation prompt {index} must be a JSON string")
        parsed = json.loads(prompt)
        _require(isinstance(parsed, dict), f"validation prompt {index} must decode to a JSON object")
        _require("[trigger]" in prompt, f"validation prompt {index} is missing [trigger]")
    seeds = tuple(sample.get("seeds") or ())
    _require(bool(seeds) and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds), "sample.seeds must be a non-empty integer list")
    scale_sweep = sample.get("unconditional_scale_sweep") or [0.0]
    _require(all(float(scale) >= 0.0 for scale in scale_sweep), "sample.unconditional_scale_sweep must be non-negative")
    _require(bool(sample.get("require_official_unconditional", True)), "Gen2 requires sample.require_official_unconditional=true")
    style_adapter = sample.get("style_adapter") or {}
    for key in ("conditional_scale", "unconditional_scale"):
        _require(float(style_adapter.get(key, 1.0 if key == "conditional_scale" else 0.0)) >= 0.0, f"style_adapter.{key} must be non-negative")

    train = config.get("train") or {}
    phase_a = train.get("phase_a") or {}
    phase_b = train.get("phase_b") or {}
    _require(int(phase_a.get("steps", 0)) >= 0 and int(phase_b.get("steps", 0)) >= 0, "phase steps must be non-negative")
    _require(int(phase_a.get("batch_size", train.get("batch_size", 1))) == int(train.get("batch_size", 1)), "V1 requires phase_a.batch_size to equal train.batch_size")
    _require(int(phase_b.get("batch_size", train.get("batch_size", 1))) == int(train.get("batch_size", 1)), "V1 requires phase_b.batch_size to equal train.batch_size")
    for phase_name, phase in (("phase_a", phase_a), ("phase_b", phase_b)):
        optimizer = phase.get("optimizer", config.get("train", {}).get("optimizer", "adamw"))
        _require(isinstance(optimizer, str) and bool(optimizer.strip()), f"{phase_name}.optimizer must be a non-empty string supported by ai-toolkit")
        optimizer_params = phase.get("optimizer_params", {})
        _require(isinstance(optimizer_params, Mapping), f"{phase_name}.optimizer_params must be a mapping")
    for phase_name, phase in (("phase_a", phase_a), ("phase_b", phase_b)):
        _require(phase.get("timestep_sampling", "stratified_uniform") == "stratified_uniform", f"{phase_name}.timestep_sampling must be stratified_uniform")
        _require(int(phase.get("timestep_bins", 10)) >= 2, f"{phase_name}.timestep_bins must be at least 2")
    margin = phase_a.get("helper_margin") or {}
    _require(int(margin.get("every", 1)) >= 1, "phase_a.helper_margin.every must be at least 1")
    _require(margin.get("mode", "absolute") in {"absolute", "relative"}, "phase_a.helper_margin.mode must be absolute or relative")
    diversity = phase_a.get("token_diversity") or {}
    _require(not diversity.get("enabled", False), "token_diversity is not implemented in Gen2 V1")
    _require(float(phase_b.get("temporal_mean_weight", 0.0)) == 0.0, "temporal_mean_weight is diagnostic-only and must be zero")
    return Gen2RuntimeConfig(
        mode=mode,
        placeholder=placeholder,
        helpers=tuple(helpers),
        phase_a=dict(phase_a),
        phase_b=dict(phase_b),
        unconditional=dict(unconditional),
        validation_prompts=prompts,
        seeds=seeds,
    )
