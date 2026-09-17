"""Strict, dependency-free configuration for the standalone v2 activator.

Only accepted native options are forwarded. Rejecting unused options is
deliberate: a stock trainer flag must never appear to affect this owned loop.
"""
from __future__ import annotations

from copy import deepcopy
import math
import os
import re

from ..config import ConfigError, L, _resolve_schema

SCHEMA_VERSION = "2.0.0"
SPEC_SHA256 = "ec3e254c6638f1781faa449114acc17c1cc8aa454e6a5e79a4032b04ec4c7ae2"

SCHEMA = {
    "schema_version": L(SCHEMA_VERSION, choices=(SCHEMA_VERSION,)),
    "spec_path": L("docs/specs/gen2_trainer_v2.md"),
    "expected_spec_sha256": L(SPEC_SHA256, "digest"),
    "execution": {
        "training_seed": L(20260917, minimum=0),
        "encoder_gradient_checkpointing": L(True),
        "dit_attention_backend": L("native", choices=("native", "flash")),
        "deterministic_algorithms": L(False),
    },
    "conditioning": {
        "num_tokens": L(4, minimum=1, maximum=32),
        "initializer_seed": L(271828, minimum=0),
        "initialization_sample_size": L(4096, minimum=2),
        "overflow_policy": L("truncate", choices=("truncate", "error")),
    },
    "optimizer": {
        "type": L("adamw", choices=("adamw", "adamw8bit")),
        "lr": L(5e-4, "positive"), "eps": L(1e-8, "positive"),
        # List shape/range is validated explicitly after schema resolution.
        "betas": L([0.9, 0.999], "betas"),
        "weight_decay": L(0.0, minimum=0),
    },
    "inference": {"unconditional_model_path": L(None, "str", nullable=True)},
    "data": {"content_groups_file": L(None, "str", nullable=True),
             "reject_train_validation_duplicates": L(True)},
    "diagnostics": {
        "time_bin_edges": L([0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.], "edges"),
        "mechanical_probes": L(True),
    },
    "recording": {
        "flush_every_updates": L(10, minimum=1), "rotate_mb": L(64, "positive"),
        "compression": L("none", choices=("none", "gzip")),
        "writer_queue_records": L(4096, minimum=1),
        "max_core_recording_mb": L(4096, "positive"),
    },
    "evaluation": {
        "named_phrase": L(None, "str", nullable=True),
        "seeds": L([42, 43], "seeds"), "make_contact_sheets": L(True),
    },
    "checkpoint": {"resume_from": L(None, "str", nullable=True)},
}

TRAIN_DEFAULTS = {
    "steps": 500, "batch_size": 1, "gradient_accumulation_steps": 4,
    "gradient_accumulation": 1, "train_unet": False, "train_text_encoder": False,
    "gradient_checkpointing": True, "dtype": "bf16", "noise_scheduler": "flowmatch",
    "timestep_type": "linear", "num_train_timesteps": 1000,
    "content_or_style": "balanced", "loss_type": "mse", "cfg_scale": 1.0,
    "do_cfg": False, "pred_scaler": 1.0, "noise_offset": 0.0, "noise_multiplier": 1.0,
    "cache_text_embeddings": False, "unload_text_encoder": False,
    "max_grad_norm": 0.0, "skip_first_sample": False, "disable_sampling": False,
    "ema_config": {"use_ema": False},
}
MODEL_DEFAULTS = {
    "name_or_path": None, "arch": "ideogram4", "dtype": "bf16",
    "quantize": True, "qtype": "qfloat8", "quantize_te": True, "qtype_te": "qfloat8",
    "low_vram": False, "layer_offloading": False, "compile": False,
    "block_compile": False, "unconditional_lora_path": None,
    "model_kwargs": {"text_encoder_path": "Qwen/Qwen3-VL-8B-Instruct", "max_text_length": 3072},
}
DATASET_DEFAULTS = {
    "folder_path": None, "caption_ext": "txt", "resolution": [768, 1280],
    "num_repeats": 1, "num_workers": 0, "cache_latents_to_disk": True,
    "caption_dropout_rate": 0., "token_dropout_rate": 0., "shuffle_tokens": False,
    "cache_text_embeddings": False, "is_reg": False, "prior_reg": False,
    "loss_multiplier": 1., "network_weight": 1., "flip_x": False, "flip_y": False,
}
SAMPLE_DEFAULTS = {
    "sampler": "flowmatch", "sample_every": 100, "sample_start_step": 0,
    "width": 1536, "height": 1024, "prompts": [], "neg": "", "seed": 42,
    "walk_seed": False, "guidance_scale": 3.0, "sample_steps": 28,
}
SAVE_DEFAULTS = {"dtype": "float32", "save_every": 100,
                 "max_step_saves_to_keep": 1, "push_to_hub": False}
LOGGING_DEFAULTS = {"log_every": 1, "verbose": False, "use_wandb": False,
                    "use_ui_logger": False, "project_name": "ai-toolkit", "run_name": None}


def _mapping(raw, defaults, path):
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected mapping")
    unknown = sorted(set(raw) - set(defaults))
    if unknown:
        raise ConfigError(f"{path}: unsupported or unknown options {unknown}; v2 only uses explicitly supported options")
    result = deepcopy(defaults)
    for key, value in raw.items():
        result[key] = (_mapping(value, defaults[key], f"{path}.{key}")
                       if isinstance(defaults[key], dict) else deepcopy(value))
    return result


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive_integer(value, path, minimum=1):
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{path}: expected integer >= {minimum}")


def _nonnegative(value, path):
    if not _number(value) or value < 0:
        raise ConfigError(f"{path}: expected finite nonnegative number")


def _bool(value, path):
    if type(value) is not bool:
        raise ConfigError(f"{path}: expected boolean")


def _text(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}: expected nonempty string")


def _fixed(section, expected, path):
    for key, value in expected.items():
        actual = section[key]
        if isinstance(value, bool):
            _bool(actual, f"{path}.{key}")
        elif isinstance(value, (int, float)) and not _number(actual):
            raise ConfigError(f"{path}.{key}: expected finite number, not a boolean or string")
        if actual != value:
            raise ConfigError(f"{path}.{key} must be {value!r} for the standalone token-only v2 experiment")


def resolve_process_config(raw):
    """Resolve one process mapping without torch, tokenizer, or model imports.

    Optimizer settings have one source: ``gen2.optimizer``. All other returned
    sections are plain mappings suitable for the supported native constructors.
    The function is idempotent, and does not mutate the caller's mapping.
    """
    if not isinstance(raw, dict):
        raise ConfigError("process: expected mapping")
    raw = deepcopy(raw)
    raw.pop("_gen2_resolved", None)
    allowed = {"type", "device", "training_folder", "trigger_word", "datasets",
               "train", "model", "save", "sample", "logging", "gen2"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"process: unsupported or unknown options {unknown}; no network/adapter/embedding config is accepted in v2")
    cfg = {"type": raw.get("type", "gen2_trainer"), "device": raw.get("device", "cuda:0"),
           "training_folder": raw.get("training_folder", "output"),
           "trigger_word": raw.get("trigger_word", "<gen2style>")}
    if cfg["type"] != "gen2_trainer":
        raise ConfigError("process.type must be gen2_trainer")
    for key in ("training_folder", "trigger_word", "device"):
        _text(cfg[key], key)
    if not re.fullmatch(r"cuda(?::\d+)?", cfg["device"]):
        raise ConfigError("device: v2 production runs require one CUDA device")
    if cfg["trigger_word"] == "[trigger]":
        raise ConfigError("trigger_word must be a distinct reserved literal, not the [trigger] alias")
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        raise ConfigError("WORLD_SIZE must be an integer") from None
    if world_size != 1:
        raise ConfigError("Gen2 v2 supports one process/GPU; WORLD_SIZE must be 1")
    gen = _resolve_schema(raw.get("gen2", {}), SCHEMA)
    cfg["gen2"] = gen
    if gen["expected_spec_sha256"].lower() != SPEC_SHA256:
        raise ConfigError("gen2.expected_spec_sha256 does not identify the approved v2 specification")
    gen["expected_spec_sha256"] = SPEC_SHA256
    _text(gen["inference"]["unconditional_model_path"], "gen2.inference.unconditional_model_path")
    betas = gen["optimizer"]["betas"]
    if not isinstance(betas, (list, tuple)) or len(betas) != 2 or not all(_number(v) and 0 <= v < 1 for v in betas):
        raise ConfigError("gen2.optimizer.betas must contain two finite values in [0, 1)")
    gen["optimizer"]["betas"] = list(betas)
    seeds = gen["evaluation"]["seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ConfigError("gen2.evaluation.seeds must be nonempty and unique")
    phrase = gen["evaluation"]["named_phrase"]
    if phrase is not None and ("[trigger]" in phrase or cfg["trigger_word"] in phrase):
        raise ConfigError("gen2.evaluation.named_phrase must not contain a trigger marker")

    for key, defaults in (("train", TRAIN_DEFAULTS), ("model", MODEL_DEFAULTS),
                          ("sample", SAMPLE_DEFAULTS), ("save", SAVE_DEFAULTS),
                          ("logging", LOGGING_DEFAULTS)):
        cfg[key] = _mapping(raw.get(key, {}), defaults, key)
    train, model = cfg["train"], cfg["model"]
    for key in ("steps", "batch_size", "gradient_accumulation_steps"):
        _positive_integer(train[key], f"train.{key}")
    _positive_integer(train["num_train_timesteps"], "train.num_train_timesteps", 2)
    _nonnegative(train["max_grad_norm"], "train.max_grad_norm")
    for key in ("gradient_checkpointing", "disable_sampling"):
        _bool(train[key], f"train.{key}")
    _fixed(train, {key: value for key, value in TRAIN_DEFAULTS.items() if key in {
        "train_unet", "train_text_encoder", "gradient_accumulation", "noise_scheduler",
        "timestep_type", "content_or_style", "loss_type", "cfg_scale", "do_cfg",
        "pred_scaler", "noise_offset", "noise_multiplier", "cache_text_embeddings",
        "unload_text_encoder", "skip_first_sample", "ema_config"}}, "train")
    _fixed(train["ema_config"], {"use_ema": False}, "train.ema_config")
    dtype_aliases = {"bf16": "bf16", "bfloat16": "bf16", "fp32": "float32", "float32": "float32"}
    for path, section in (("train", train), ("model", model)):
        dtype = section["dtype"]
        if not isinstance(dtype, str) or dtype not in dtype_aliases:
            raise ConfigError(f"{path}.dtype must be bf16 or float32; fp16 AMP is not implemented in v2")
        section["dtype"] = dtype_aliases[dtype]
    if model["dtype"] != train["dtype"]:
        raise ConfigError("model.dtype and train.dtype must agree")
    _text(model["name_or_path"], "model.name_or_path")
    _fixed(model, {"arch": "ideogram4", "low_vram": False, "layer_offloading": False,
                   "compile": False, "block_compile": False, "unconditional_lora_path": None}, "model")
    for key in ("quantize", "quantize_te"):
        _bool(model[key], f"model.{key}")
    for key in ("qtype", "qtype_te"):
        _text(model[key], f"model.{key}")
        if "|" in model[key]:
            raise ConfigError(f"model.{key} cannot load a recovery adapter in the frozen-base experiment")
    _text(model["model_kwargs"]["text_encoder_path"], "model.model_kwargs.text_encoder_path")
    limit = model["model_kwargs"]["max_text_length"]
    _positive_integer(limit, "model.model_kwargs.max_text_length", 2)
    if limit > 3072 or gen["conditioning"]["num_tokens"] >= limit:
        raise ConfigError("model.model_kwargs.max_text_length must be <= 3072 and greater than the learned bank size")

    datasets = raw.get("datasets", [])
    if not isinstance(datasets, list) or not datasets:
        raise ConfigError("datasets must be a nonempty list")
    cfg["datasets"] = []
    for index, item in enumerate(datasets):
        path = f"datasets[{index}]"
        data = _mapping(item, DATASET_DEFAULTS, path)
        _text(data["folder_path"], f"{path}.folder_path")
        _text(data["caption_ext"], f"{path}.caption_ext")
        _positive_integer(data["num_repeats"], f"{path}.num_repeats")
        _bool(data["cache_latents_to_disk"], f"{path}.cache_latents_to_disk")
        _fixed(data, {key: value for key, value in DATASET_DEFAULTS.items() if key in {
            "num_workers", "caption_dropout_rate", "token_dropout_rate", "shuffle_tokens",
            "cache_text_embeddings", "is_reg", "prior_reg", "loss_multiplier", "network_weight",
            "flip_x", "flip_y"}}, path)
        _positive_integer(data["num_workers"], f"{path}.num_workers", 0)
        sizes = data["resolution"]
        if type(sizes) is int:
            sizes = [sizes]
        if not isinstance(sizes, list) or not sizes:
            raise ConfigError(f"{path}.resolution must be a positive integer or nonempty integer list")
        for size in sizes:
            _positive_integer(size, f"{path}.resolution")
            if size % 16:
                raise ConfigError(f"{path}.resolution values must be multiples of 16 for native Ideogram latents")
        if len(sizes) != len(set(sizes)):
            raise ConfigError(f"{path}.resolution must not repeat a size")
        data["resolution"] = sizes
        cfg["datasets"].append(data)

    sample, save, logging = cfg["sample"], cfg["save"], cfg["logging"]
    for key in ("sample_every", "sample_steps", "width", "height"):
        _positive_integer(sample[key], f"sample.{key}")
    if sample["width"] % 16 or sample["height"] % 16:
        raise ConfigError("sample.width and sample.height must be multiples of 16")
    _positive_integer(sample["seed"], "sample.seed", 0)
    if "seed" not in raw.get("sample", {}):
        sample["seed"] = seeds[0]
    elif sample["seed"] != seeds[0]:
        raise ConfigError("sample.seed must equal gen2.evaluation.seeds[0]; v2 samples the explicit seeds list")
    _nonnegative(sample["guidance_scale"], "sample.guidance_scale")
    _fixed(sample, {"sampler": "flowmatch", "sample_start_step": 0, "walk_seed": False, "neg": ""}, "sample")
    if not isinstance(sample["prompts"], list) or any(not isinstance(v, str) or not v.strip() for v in sample["prompts"]):
        raise ConfigError("sample.prompts must be a list of nonempty prompt strings")
    if not train["disable_sampling"] and not sample["prompts"]:
        raise ConfigError("sample.prompts must not be empty when sampling is enabled")
    for index, prompt in enumerate(sample["prompts"]):
        if "[trigger]" not in prompt and cfg["trigger_word"] not in prompt:
            raise ConfigError(f"sample.prompts[{index}] must contain the activator marker for controlled comparisons")
    _positive_integer(save["save_every"], "save.save_every")
    _fixed(save, {"dtype": "float32", "max_step_saves_to_keep": 1, "push_to_hub": False}, "save")
    _positive_integer(save["max_step_saves_to_keep"], "save.max_step_saves_to_keep")
    _positive_integer(logging["log_every"], "logging.log_every")
    _bool(logging["verbose"], "logging.verbose")
    _fixed(logging, {"use_wandb": False, "use_ui_logger": False}, "logging")
    _text(logging["project_name"], "logging.project_name")
    if logging["run_name"] is not None:
        _text(logging["run_name"], "logging.run_name")
    cfg["_gen2_resolved"] = {
        "effective_batch_size": train["batch_size"] * train["gradient_accumulation_steps"],
        "trainable_parameter_family": "embedding", "trainable_master_dtype": "float32",
        "scheduler": "constant", "optimizer_horizon": train["steps"],
        "text_token_limit_policy": "in-place repeated bank; common retained content across evaluation modes; protected chat boundaries",
        "checkpoint_retention": "one rolling resume checkpoint and final inference export",
        "native_schema_validation": "only explicitly supported options are accepted; model loading and GPU validation remain runtime checks",
    }
    return cfg
