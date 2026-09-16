"""Dependency-free Gen2 v1 configuration validation and resolution.

The native process constructs ai-toolkit's configuration objects after this
validation. Keeping this module free of model/torch imports makes --validate
usable without model weights or the optional GPU packages.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import os
import re
from typing import Any

SCHEMA_VERSION = "1.0.0"
SPEC_SHA256 = "6880f48649280c02fe7e73bf9b3b244eef4da6ca63da02226bc876cae5df03b8"
ROLES = ("diffusion", "embedding", "text_adapter", "gates")
# Explicit initial-delivery boundary approved by the user after the v1 review.
# These are names forwarded unchanged to the native factory, not implementations.
STANDARD_OPTIMIZERS = frozenset({"adam", "adamw", "adagrad", "lion", "prodigy",
    "adafactor", "dadaptation", "adam8bit", "adamw8bit", "lion8bit", "ademamix8bit"})
DEFAULT_MILESTONE_MODES = ("full", "neutral_lora_on", "base", "base_with_conditioning",
                           "conditioning_init", "encoder_adapter_off", "tokens_init",
                           "gates_one", "gates_time_mean")
# Additional user-requested visual experiments are explicit opt-ins. Preserve
# the original defaults and production unconditional route for existing configs.
MODES = DEFAULT_MILESTONE_MODES + ("base_with_tokens", "full_uncond_half", "full_uncond_full")


class ConfigError(ValueError):
    """A configuration contradicts the supported, recorded v1 contract."""


@dataclass(frozen=True)
class Leaf:
    default: Any
    kind: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple | None = None
    nullable: bool = False


def L(default, kind=None, minimum=None, maximum=None, choices=None, nullable=False):
    if kind is None:
        kind = "bool" if isinstance(default, bool) else "int" if isinstance(default, int) else "float" if isinstance(default, float) else "str"
    return Leaf(default, kind, minimum, maximum, choices, nullable)


OPTIMIZER_SCHEMA = {
    "optimizer": L(None, "str", nullable=True), "lr": L(None, "float", 0, nullable=True),
    "optimizer_params": L({}, "map"), "lr_scheduler": L(None, "str", nullable=True),
    "lr_scheduler_params": L({}, "map"),
}
SCHEMA = {
    "schema_version": L(SCHEMA_VERSION, choices=(SCHEMA_VERSION,)),
    "spec_path": L("docs/specs/gen2_trainer_v1.md"),
    "expected_spec_sha256": L(None, "digest", nullable=True),
    "execution": {
        "training_seed": L(20260911, minimum=0), "diagnostic_seed": L(314159, minimum=0),
        "deterministic_algorithms": L(False), "encoder_gradient_checkpointing": L(True),
        "dit_attention_backend": L("native", choices=("native", "flash")),
        "error_policy": L("abort", choices=("abort",)),
    },
    "conditioning": {
        "num_tokens": L(4, minimum=1, maximum=32), "initializer_text": L("style"),
        "initializer_jitter": L(0.01, minimum=0, maximum=1), "initializer_seed": L(271828, minimum=0),
        "adapter_rank": L(4, minimum=1), "adapter_alpha": L(4.0, "positive"),
        "overflow_policy": L("error", choices=("error", "truncate")),
    },
    "losses": {"neutral_weight": L(1.0, minimum=0), "text_adapter_weight": L(1e-4, minimum=0),
               "gate_center_weight": L(1e-3, minimum=0), "gate_smoothness_weight": L(1e-4, minimum=0)},
    "phases": {"warmup_updates": L(450, minimum=0), "refinement_updates": L(2100, minimum=0),
               "calibration_updates": L(450, minimum=0), "diffusion_updates_per_cycle": L(4, minimum=1),
               "conditioning_updates_per_cycle": L(1, minimum=1)},
    "optimizers": {
        "diffusion": OPTIMIZER_SCHEMA,
        "embedding": {**OPTIMIZER_SCHEMA, "lr": L(1e-3, minimum=0, nullable=True)},
        "text_adapter": {**OPTIMIZER_SCHEMA, "lr": L(1e-5, minimum=0, nullable=True)},
        "gates": {**OPTIMIZER_SCHEMA, "lr": L(1e-3, minimum=0, nullable=True)},
    },
    "gates": {"amplitude": L(0.5, "unit_open"), "regularization_grid_points": L(65, minimum=5)},
    "inference": {"missing_trigger_policy": L("learned_neutral", choices=("learned_neutral", "base_bypass")),
                  "lora_strength": L(1.0, minimum=0),
                  "unconditional_model_path": L(None, "str", nullable=True)},
    "data": {"content_groups_file": L(None, "str", nullable=True), "reject_train_validation_duplicates": L(True)},
    "diagnostics": {
        "enabled": L(True, choices=(True,)), "activation_every": L(100, minimum=1),
        "gradient_probe_every": L(250, minimum=1), "gradient_probe_examples": L(1, minimum=1),
        "gradient_probe_taus": L([0.25, 0.75], "taus"), "gradient_probe_max_coordinates": L(65536, minimum=0),
        "gate_log_every": L(100, minimum=1), "spectra_every": L(500, minimum=1),
        "full_update_norm_every": L(0, minimum=0), "parameter_sample_elements_per_family": L(65536, minimum=1),
        "tensor_memory_budget_mb": L(512, "positive"),
        "time_bin_edges": L([0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.], "edges"),
        "prefix_atol": L(0.001, "positive"), "prefix_rtol": L(0.01, "positive"),
        "probes": {"every": L(250, minimum=1), "num_examples": L(4, minimum=1),
                   "source": L("training", choices=("training", "validation")),
                   "taus": L([0.05, 0.25, 0.5, 0.75, 1.], "taus"), "save_fixed_latent_packet": L(True, choices=(True,))},
        "tensor_dumps": {"enabled": L(False), "names": L(["suffix_features", "probe_velocities"], "dump_names"),
                         "max_packets": L(4, minimum=0), "max_total_mb": L(256, "positive")},
    },
    "recording": {"flush_every_updates": L(10, minimum=1), "rotate_mb": L(64, "positive"),
                  "compression": L("none", choices=("none", "gzip")), "writer_queue_records": L(4096, minimum=1),
                  "max_core_recording_mb": L(4096, "positive")},
    "evaluation": {"milestone_every": L(1000, minimum=1), "additional_seeds": L([43], "seeds"),
                   "prompt_groups": L({}, "prompt_groups"), "preview_modes": L(["full", "neutral_lora_on", "base"], "modes"),
                   "milestone_modes": L(list(DEFAULT_MILESTONE_MODES), "modes"), "make_contact_sheets": L(True)},
    "checkpoint": {"resume_from": L(None, "str", nullable=True), "strict_resume": L(True, choices=(True,)),
                   "save_at_stage_boundaries": L(True, choices=(True,)), "protect_stage_checkpoints": L(True, choices=(True,)),
                   "save_initial_state": L(True, choices=(True,))},
}


def schema_leaves(schema=SCHEMA, prefix="gen2"):
    """Enumerate every configurable leaf for documentation/coverage checks."""
    result = {}
    for key, entry in schema.items():
        path = f"{prefix}.{key}"
        if isinstance(entry, Leaf):
            result[path] = entry
        else:
            result.update(schema_leaves(entry, path))
    return result


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_leaf(value, spec, path):
    if value is None and spec.nullable:
        return None
    kind = spec.kind
    valid = True
    if kind == "bool": valid = type(value) is bool
    elif kind == "int": valid = type(value) is int
    elif kind in ("float", "positive", "unit_open"): valid = _number(value)
    elif kind == "str": valid = isinstance(value, str) and bool(value.strip())
    elif kind == "digest": valid = isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None
    elif kind in ("map", "prompt_groups"):
        valid = isinstance(value, dict) and all(isinstance(k, str) for k in value)
        if valid and kind == "prompt_groups":
            valid = all(re.fullmatch(r"p\d{3,}", k) and isinstance(v, str) and v.strip() for k, v in value.items())
    elif kind in ("taus", "edges", "seeds", "modes", "dump_names"):
        valid = isinstance(value, list)
        if valid:
            if kind in ("taus", "edges"):
                valid = bool(value) and all(_number(v) and (0 < v <= 1 if kind == "taus" else 0 <= v <= 1) for v in value)
                valid = valid and len(set(value)) == len(value)
                if valid and kind == "edges": valid = len(value) >= 2 and value[0] == 0 and value[-1] == 1 and value == sorted(value)
            elif kind == "seeds": valid = all(type(v) is int and v >= 0 for v in value)
            else:
                allowed = MODES if kind == "modes" else ("suffix_features", "probe_velocities")
                valid = bool(value) and all(isinstance(v, str) and v in allowed for v in value) and len(set(value)) == len(value)
    if not valid:
        raise ConfigError(f"{path}: expected {kind}; got {value!r}")
    if kind == "positive" and value <= 0: raise ConfigError(f"{path}: must be > 0")
    if kind == "unit_open" and not 0 < value < 1: raise ConfigError(f"{path}: must be strictly between 0 and 1")
    if spec.minimum is not None and value < spec.minimum: raise ConfigError(f"{path}: must be >= {spec.minimum}")
    if spec.maximum is not None and value > spec.maximum: raise ConfigError(f"{path}: must be <= {spec.maximum}")
    if spec.choices is not None and value not in spec.choices: raise ConfigError(f"{path}: must be one of {spec.choices}")
    return deepcopy(value)


def _resolve_schema(raw, schema=SCHEMA, path="gen2"):
    if not isinstance(raw, dict): raise ConfigError(f"{path}: expected mapping")
    unknown = sorted(set(raw) - set(schema))
    if unknown: raise ConfigError(f"{path}: unknown keys {unknown}")
    return {key: _validate_leaf(raw.get(key, entry.default), entry, f"{path}.{key}")
            if isinstance(entry, Leaf) else _resolve_schema(raw.get(key, {}), entry, f"{path}.{key}")
            for key, entry in schema.items()}


def family_horizons(phases):
    d, a = phases["diffusion_updates_per_cycle"], phases["conditioning_updates_per_cycle"]
    cycles, remainder = divmod(phases["refinement_updates"], d + a)
    na = cycles * a + max(0, remainder - d)
    return {"diffusion": phases["warmup_updates"] + cycles * d + min(d, remainder),
            "embedding": na, "text_adapter": na, "gates": phases["calibration_updates"]}


def scheduler_kwargs(name, requested, horizon):
    """Translate the native factory's horizon spellings without rewriting schedules.

    A step scheduler requires an explicit period. A restart period in v1 is the
    optimizer's horizon unless supplied identically. Unknown diffusers schedules
    are resolved by constructor introspection in engine.py.
    """
    kwargs = deepcopy(requested)
    if not isinstance(kwargs, dict): raise ConfigError("lr_scheduler_params must be a mapping")
    if "optimizer" in kwargs: raise ConfigError("lr_scheduler_params.optimizer is owned by the Gen2 engine")
    if kwargs.get("last_epoch", -1) != -1:
        raise ConfigError("lr_scheduler_params.last_epoch must be -1; use complete checkpoint resume")
    aliases = {"total_iters", "num_training_steps", "T_max", "T_0", "total_steps"}
    for key in aliases & kwargs.keys():
        if type(kwargs[key]) is not int or kwargs[key] != horizon:
            raise ConfigError(f"lr_scheduler_params.{key}={kwargs[key]!r} contradicts optimizer-local horizon {horizon}")
    if horizon == 0:
        return kwargs
    if name in ("cosine", "cosine_with_restarts", "constant", "linear", "constant_with_warmup"):
        accepted_alias = "T_max" if name == "cosine" else "T_0" if name == "cosine_with_restarts" else "total_iters"
        for key in aliases & kwargs.keys():
            if key not in ("total_iters", accepted_alias):
                raise ConfigError(f"{name} does not accept horizon keyword {key}")
        if accepted_alias != "total_iters" and accepted_alias in kwargs:
            if "total_iters" in kwargs: raise ConfigError(f"Specify only one of total_iters and {accepted_alias}")
        else:
            kwargs["total_iters"] = horizon
        if name == "constant": kwargs.setdefault("factor", 1.0)
        if name == "constant_with_warmup": kwargs.setdefault("num_warmup_steps", 1000)
    elif name == "step":
        if aliases & kwargs.keys(): raise ConfigError("step scheduler takes step_size, not a total-horizon constructor argument")
        if type(kwargs.get("step_size")) is not int or kwargs["step_size"] < 1:
            raise ConfigError("step scheduler requires an explicit positive integer step_size")
    return kwargs


def _native_defaults():
    return {
        "type": "gen2_trainer", "device": "cuda:0", "training_folder": "output", "trigger_word": "<gen2style>",
        "network": {"type": "lora", "linear": 32, "linear_alpha": 32.0, "dropout": 0.0},
        "train": {"steps": 3000, "batch_size": 1, "gradient_accumulation_steps": 1, "gradient_accumulation": 1,
                  "train_unet": True, "train_text_encoder": False, "gradient_checkpointing": True,
                  "dtype": "bf16", "noise_scheduler": "flowmatch", "timestep_type": "linear", "num_train_timesteps": 1000,
                  "content_or_style": "balanced", "loss_type": "mse", "cfg_scale": 1.0, "do_cfg": False,
                  "pred_scaler": 1.0, "noise_offset": 0.0, "noise_multiplier": 1.0,
                  "cache_text_embeddings": False, "unload_text_encoder": False, "optimizer": "adamw8bit", "lr": 1e-4,
                  "optimizer_params": {"weight_decay": 0.0}, "lr_scheduler": "constant", "lr_scheduler_params": {},
                  "max_grad_norm": 1.0, "skip_first_sample": False, "disable_sampling": False, "ema_config": {"use_ema": False}},
        "model": {"arch": "ideogram4", "dtype": "bf16", "quantize": True, "qtype": "qfloat8", "quantize_te": False,
                  "qtype_te": "qfloat8", "low_vram": False, "layer_offloading": False, "compile": False,
                  "unconditional_lora_path": None, "model_kwargs": {"text_encoder_path": "Qwen/Qwen3-VL-8B-Instruct", "max_text_length": 2048}},
        "save": {"dtype": "float32", "save_every": 250, "max_step_saves_to_keep": 4, "push_to_hub": False},
        "logging": {"log_every": 100, "verbose": False, "use_wandb": False, "use_ui_logger": False,
                    "project_name": "ai-toolkit", "run_name": None},
        "sample": {"sampler": "flowmatch", "sample_every": 250, "sample_start_step": 0, "width": 1024, "height": 1024,
                   "prompts": [], "neg": "", "seed": 42, "walk_seed": False, "guidance_scale": 7.0, "sample_steps": 30},
    }


def _merge_defaults(defaults, raw):
    result = deepcopy(defaults)
    for key, value in raw.items():
        result[key] = _merge_defaults(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def _must(section, key, expected, path, errors):
    if key in section and section[key] != expected:
        errors.append(f"{path}.{key} must be {expected!r}; it changes v1 gradient, branch, loss, or data semantics")


def _native_compatibility(cfg):
    errors = []
    train, model, network = cfg["train"], cfg["model"], cfg["network"]
    fixed_train = {
        "train_unet": True, "train_text_encoder": False, "noise_scheduler": "flowmatch", "timestep_type": "linear",
        "content_or_style": "balanced", "loss_type": "mse", "loss_target": "noise", "cfg_scale": 1.0,
        "max_cfg_scale": 1.0, "cfg_rescale": 1.0, "pred_scaler": 1.0, "noise_offset": 0., "noise_multiplier": 1.,
        "target_noise_multiplier": 1., "random_noise_multiplier": 0., "random_noise_shift": 0., "weight_jitter": 0.,
        "img_multiplier": 1., "noisy_latent_multiplier": 1., "prompt_dropout_prob": 0., "first_timestep_chance": 0.,
        "prompt_saturation_chance": 0., "optimal_noise_pairing_samples": 1, "gradient_accumulation": 1,
        "min_snr_gamma": None, "snr_gamma": None, "target_norm_std": None, "max_loss": None, "start_step": None,
        "adapter_assist_name_or_path": None, "latent_feature_extractor_path": None, "diffusion_feature_extractor_path": None,
        "min_denoising_steps": 0, "max_denoising_steps": train["num_train_timesteps"] - 1,
    }
    for key in ("do_cfg", "do_random_cfg", "do_signal_correction_noise", "do_batch_noise_correction", "do_signal_amplification",
                "cache_text_embeddings", "unload_text_encoder", "learnable_snr_gos", "dynamic_noise_offset", "match_noise_norm",
                "adaptive_scaling_factor", "correct_pred_norm", "do_fft_loss", "do_fft_velocity_equiv_weight", "t0_loss_target",
                "t0_velocity_equiv_weight", "do_prior_divergence", "do_paramiter_swapping", "merge_network_on_save", "train_turbo",
                "free_u", "short_and_long_captions", "short_and_long_captions_encoder_split", "single_item_batching",
                "inverted_mask_prior", "diff_output_preservation", "blank_prompt_preservation", "standardize_images",
                "force_consistent_noise", "blended_blur_noise", "do_guidance_loss", "do_guidance_loss_cfg_zero",
                "do_differential_guidance", "do_blank_stabilization", "linear_timesteps", "linear_timesteps2", "xformers", "sdp"):
        fixed_train[key] = False
    for key, value in fixed_train.items(): _must(train, key, value, "train", errors)
    if "attention_backend" in train and train["attention_backend"] != cfg["gen2"]["execution"]["dit_attention_backend"]:
        errors.append("train.attention_backend conflicts with gen2.execution.dit_attention_backend")
    for key in ("compile", "block_compile"): _must(model, key, False, "model", errors)
    if (cfg["gen2"]["inference"]["unconditional_model_path"] is not None
            and model.get("unconditional_lora_path") is not None):
        errors.append("gen2.inference.unconditional_model_path and model.unconditional_lora_path are mutually exclusive; choose the original unconditional transformer or the native correction adapter")
    for key in ("is_v2", "is_xl", "is_pixart", "is_pixart_sigma", "is_auraflow", "is_v3", "is_flux", "is_lumina2",
                "is_ssd", "is_vega", "is_v_pred", "use_flux_cfg", "experimental_xl", "attn_masking", "split_model_over_gpus", "in_context"):
        _must(model, key, False, "model", errors)
    for key in ("lora_path", "assistant_lora_path", "inference_lora_path", "accuracy_recovery_adapter", "refiner_name_or_path",
                "ignore_if_contains", "only_if_contains"):
        _must(model, key, None, "model", errors)
    if isinstance(model.get("qtype"), str) and "|" in model["qtype"]:
        errors.append("model.qtype must not include an accuracy-recovery adapter; only the frozen CFG unconditional adapter is allowed")
    _must(model, "arch", "ideogram4", "model", errors)
    for key, value in {"type": "lora", "dropout": 0., "rank_dropout": 0., "module_dropout": 0.,
                       "pretrained_lora_path": None, "all_layers": False, "layer_offloading": False}.items():
        _must(network, key, value, "network", errors)
    for key, value in network.get("network_kwargs", {}).items():
        if key in ("dropout", "rank_dropout", "module_dropout") and value == 0: continue
        errors.append(f"network.network_kwargs.{key} changes the fixed v1 LoRA target/residual implementation")
    if network.get("rank", network["linear"]) != network["linear"]: errors.append("network.rank conflicts with network.linear")
    for key in ("embedding", "adapter", "embedding_config", "adapter_config"):
        if cfg.get(key) is not None: errors.append(f"{key} is incompatible with Gen2-owned conditioning")
    _must(train.get("ema_config", {}), "use_ema", False, "train.ema_config", errors)
    _must(cfg["sample"], "walk_seed", False, "sample", errors)
    _must(cfg["sample"], "neg", "", "sample", errors)
    _must(cfg["sample"], "sampler", "flowmatch", "sample", errors)
    if cfg["save"].get("push_to_hub") is not False:
        errors.append("save.push_to_hub must be false: complete Gen2 package publishing is outside the initial implementation")
    for key, expected in (("network_multiplier", 1.), ("guidance_rescale", 0.), ("do_cfg_norm", False)):
        _must(cfg["sample"], key, expected, "sample", errors)
    if isinstance(cfg["sample"].get("prompts"), list):
        for i, prompt in enumerate(cfg["sample"]["prompts"]):
            if not isinstance(prompt, str): errors.append(f"sample.prompts[{i}] must be a content prompt string in v1")
    requested_modes = set(cfg["gen2"]["evaluation"]["preview_modes"] + cfg["gen2"]["evaluation"]["milestone_modes"])
    if requested_modes & {"full_uncond_half", "full_uncond_full"} and cfg["sample"]["guidance_scale"] <= 1:
        errors.append("sample.guidance_scale must be > 1 for full_uncond_half/full_uncond_full; native sampling otherwise skips the unconditional pass")
    for i, data in enumerate(cfg.get("datasets", [])):
        if not isinstance(data, dict): raise ConfigError(f"datasets[{i}] must be a mapping")
        for key, value in {"caption_dropout_rate": 0., "token_dropout_rate": 0., "shuffle_tokens": False,
                           "cache_text_embeddings": False, "is_reg": False, "prior_reg": False, "loss_multiplier": 1.,
                           "network_weight": 1., "alpha_mask": False, "control_from_same_folder": False,
                           "clip_image_from_same_folder": False, "standardize_images": False}.items():
            _must(data, key, value, f"datasets[{i}]", errors)
        # Gen2 deliberately reads raw captions to preserve exact content spacing.
        # Native postprocessing features would otherwise be silently ignored.
        for key in ("random_triggers", "replacements"):
            if data.get(key): errors.append(f"datasets[{i}].{key} rewrites captions outside the Gen2 canonical-caption contract")
        if data.get("trigger_word") not in (None, cfg["trigger_word"]):
            errors.append(f"datasets[{i}].trigger_word conflicts with the process.trigger_word source")
        for key in ("mask_path", "control_path", "control_path_1", "control_path_2", "control_path_3", "inpaint_path", "clip_image_path", "unconditional_path"):
            if data.get(key): errors.append(f"datasets[{i}].{key} introduces an unsupported extra condition or loss mask")
    if errors: raise ConfigError("Incompatible v1 settings:\n- " + "\n- ".join(errors))


def resolve_process_config(raw):
    """Return a fully expanded plain process mapping, without loading models."""
    if not isinstance(raw, dict): raise ConfigError("process must be a mapping")
    raw = deepcopy(raw)
    raw.pop("_gen2_resolved", None)
    cfg = _merge_defaults(_native_defaults(), raw)
    for section in ("network", "train", "model", "save", "sample", "logging"):
        if not isinstance(cfg[section], dict): raise ConfigError(f"{section} must be a mapping")
    if cfg.get("type") != "gen2_trainer": raise ConfigError("process.type must be gen2_trainer")
    if not isinstance(cfg.get("trigger_word"), str) or not cfg["trigger_word"].strip(): raise ConfigError("trigger_word must be a nonempty reserved literal")
    if not isinstance(cfg.get("datasets", []), list): raise ConfigError("datasets must be a list")
    dataset_defaults = {"caption_ext": "txt", "caption_dropout_rate": 0., "token_dropout_rate": 0.,
        "shuffle_tokens": False, "cache_latents_to_disk": True, "cache_text_embeddings": False,
        "resolution": [1024], "num_repeats": 1, "num_workers": 0}
    cfg["datasets"] = [_merge_defaults(dataset_defaults, item) if isinstance(item, dict) else item for item in cfg.get("datasets", [])]
    raw_network = raw.get("network", {})
    if "rank" in raw_network and "linear" not in raw_network: cfg["network"]["linear"] = raw_network["rank"]
    if "alpha" in raw_network and "linear_alpha" not in raw_network: cfg["network"]["linear_alpha"] = raw_network["alpha"]
    try: world = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError: raise ConfigError("WORLD_SIZE must be an integer") from None
    if world != 1: raise ConfigError("Gen2 v1 supports one process/GPU; WORLD_SIZE must be 1")
    gen = _resolve_schema(raw.get("gen2", {}))
    cfg["gen2"] = gen
    train = cfg["train"]
    for key, minimum in (("steps", 1), ("batch_size", 1), ("gradient_accumulation_steps", 1), ("num_train_timesteps", 2)):
        _validate_leaf(train[key], L(0, "int", minimum), f"train.{key}")
    if sum(gen["phases"][key] for key in ("warmup_updates", "refinement_updates", "calibration_updates")) != train["steps"]:
        raise ConfigError("gen2.phases warmup + refinement + calibration must equal train.steps")
    for path, value in (("train.lr", train["lr"]), ("network.linear_alpha", cfg["network"]["linear_alpha"])):
        _validate_leaf(value, L(1., "positive"), path)
    _validate_leaf(cfg["network"]["linear"], L(1, minimum=1), "network.linear")
    if not _number(train["max_grad_norm"]) or train["max_grad_norm"] <= 0:
        raise ConfigError("train.max_grad_norm must be finite and > 0; this native trainer has no nonpositive disable setting (Adafactor retains its native clipping exemption)")
    for key in ("save_every", "max_step_saves_to_keep"):
        _validate_leaf(cfg["save"][key], L(1, minimum=1), f"save.{key}")
    _validate_leaf(cfg["logging"]["log_every"], L(100, minimum=1), "logging.log_every")
    if cfg["save"]["dtype"] not in ("float32", "fp32", "float16", "fp16", "bf16", "bfloat16"):
        raise ConfigError("save.dtype must be float32, float16, or bfloat16 (native aliases allowed)")
    for key in ("sample_every", "sample_steps", "width", "height"):
        _validate_leaf(cfg["sample"][key], L(1, minimum=1), f"sample.{key}")
    for key in ("sample_start_step", "seed"):
        _validate_leaf(cfg["sample"][key], L(0, minimum=0), f"sample.{key}")
    _validate_leaf(cfg["sample"]["guidance_scale"], L(1., minimum=0), "sample.guidance_scale")
    if train["dtype"] not in ("bf16", "bfloat16", "fp16", "float16", "fp32", "float32"):
        raise ConfigError("train.dtype must be bf16, fp16, or float32 (native aliases allowed)")
    token_limit = cfg["model"]["model_kwargs"]["max_text_length"]
    # User-approved 2026-09-16 override: allow the native Ideogram cap while
    # retaining the shared suffix reservation under either overflow policy.
    _validate_leaf(token_limit, L(2048, minimum=2, maximum=3072), "model.model_kwargs.max_text_length")
    if gen["conditioning"]["num_tokens"] >= token_limit: raise ConfigError("num_tokens must be less than the total max_text_length")
    dg = gen["diagnostics"]
    if dg["gradient_probe_examples"] > dg["probes"]["num_examples"]:
        raise ConfigError("gradient_probe_examples must not exceed probes.num_examples")
    if dg["probes"]["source"] == "validation" and not (train.get("validation_config") or {}).get("validation_items"):
        raise ConfigError("probes.source=validation requires real train.validation_config.validation_items")
    for key in ("optimizer_params", "lr_scheduler_params"):
        if not isinstance(train[key], dict): raise ConfigError(f"train.{key} must be a mapping")
    for index, dataset in enumerate(cfg["datasets"]):
        if not isinstance(dataset, dict): raise ConfigError(f"datasets[{index}] must be a mapping")
        for key, minimum in (("num_repeats", 1), ("num_workers", 0)):
            _validate_leaf(dataset[key], L(1, minimum=minimum), f"datasets[{index}].{key}")
    horizons = family_horizons(gen["phases"])
    scheduler_args = {}
    for role in ROLES:
        override = gen["optimizers"][role]
        for key in ("optimizer", "lr", "lr_scheduler"):
            if override[key] is None: override[key] = deepcopy(train[key])
        for key in ("optimizer_params", "lr_scheduler_params"):
            override[key] = {**deepcopy(train[key]), **override[key]}
        name = override["optimizer"]
        if not isinstance(name, str) or name.lower() not in STANDARD_OPTIMIZERS:
            raise ConfigError(f"gen2.optimizers.{role}.optimizer={name!r}: initial delivery supports standard native optimizers only; custom/experimental variants and broken DAdapt aliases are deferred")
        override["optimizer"] = name.lower()
        _validate_leaf(override["lr"], L(1., minimum=0), f"gen2.optimizers.{role}.lr")
        _validate_leaf(override["lr_scheduler"], L("constant"), f"gen2.optimizers.{role}.lr_scheduler")
        params = override["optimizer_params"]
        if params.get("do_paramiter_swapping", False): raise ConfigError(f"{role}: optimizer parameter swapping conflicts with Gen2 ownership")
        if name.lower() == "adafactor" and (params.get("relative_step", False) or params.get("warmup_init", False)):
            raise ConfigError(f"{role}: native Adafactor factory supplies a numeric LR; relative_step/warmup_init must be false")
        duplicates = {"params", "lr"} & params.keys()
        if name.lower() in {"adam", "adamw", "dadaptation", "prodigy", "adam8bit", "adamw8bit", "ademamix8bit"} and "eps" in params: duplicates.add("eps")
        if duplicates: raise ConfigError(f"{role}: optimizer_params duplicates native factory arguments {sorted(duplicates)}")
        scheduler_args[role] = scheduler_kwargs(override["lr_scheduler"], override["lr_scheduler_params"], horizons[role])
    _native_compatibility(cfg)
    cfg["_gen2_resolved"] = {"family_horizons": horizons, "scheduler_kwargs": scheduler_args,
        "optimizer_support": "user-approved initial standard-native subset", "trainable_master_dtype": "float32",
        "text_token_limit_policy": f"user-approved maximum 3072; default 2048; overflow={gen['conditioning']['overflow_policy']}; learned suffix reserved",
        "ablations": [f"losses.{key}=0" for key, value in gen["losses"].items() if value == 0]
                     + [f"phases.{key}=0" for key, value in gen["phases"].items() if key.endswith("_updates") and value == 0],
        "native_schema_validation": "deferred to the native process configuration constructors; no model imports during config-only validation"}
    return cfg
