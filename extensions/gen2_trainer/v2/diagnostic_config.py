"""Strict, lightweight configuration for an independent post-training job."""
from copy import deepcopy
import math
import re
from pathlib import Path


DEFAULTS = {
    "seed": 20260918,
    "examples_per_resolution": 4,
    "noise_seeds": [1729, 1730],
    "noise_fractions": [.1, .3, .5, .7, .9],
    "named_phrase": None,  # None inherits the saved evaluation reference.
    "descent_steps": 12,
    "descent_packet_count": 4,
    "gradient_cosine_min": .999,
    "gradient_relative_l2_max": .02,
    "prediction_relative_l2_max": .02,
    "loss_relative_error_max": .02,
    "linear_modules_per_model": 2,
    "linear_input_rows": 4,
    "linear_seed": 1729,
    "max_packet_memory_mb": 512,
}


def resolve_diagnostic_config(raw):
    if not isinstance(raw, dict):
        raise ValueError("Diagnostic process must be a mapping")
    allowed = {"type", "source_checkpoint", "training_folder", "device", "diagnostic"}
    if set(raw) - allowed:
        raise ValueError(f"Unknown diagnostic process keys: {sorted(set(raw)-allowed)}")
    result = deepcopy(raw)
    if result.get("type") != "gen2_v2_diagnostic":
        raise ValueError("process.type must be gen2_v2_diagnostic")
    for name in ("source_checkpoint", "training_folder"):
        if not isinstance(result.get(name), str) or not result[name].strip():
            raise ValueError(f"{name} must be a nonempty path")
    result.setdefault("device", "cuda:0")
    if not isinstance(result["device"], str) or not re.fullmatch(r"cuda(?::\d+)?", result["device"]):
        raise ValueError("The real diagnostic requires a CUDA device")
    supplied = result.get("diagnostic", {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError("Unknown diagnostic settings or diagnostic is not a mapping")
    settings = {**deepcopy(DEFAULTS), **supplied}
    for name, lower, upper in (
        ("seed", 0, 2**32-1), ("linear_seed", 0, 2**32-1),
        ("examples_per_resolution", 1, 40), ("descent_steps", 1, 50),
        ("descent_packet_count", 1, 16), ("linear_modules_per_model", 1, 2),
        ("linear_input_rows", 1, 32), ("max_packet_memory_mb", 1, 4096),
    ):
        value = settings[name]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"diagnostic.{name} must be an integer in [{lower}, {upper}]")
    seeds = settings["noise_seeds"]
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 4 or
            any(type(v) is not int or not 0 <= v < 2**32 for v in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("noise_seeds must contain 1-4 distinct nonnegative 32-bit integers")
    fractions = settings["noise_fractions"]
    if (not isinstance(fractions, list) or not 1 <= len(fractions) <= 10 or
            any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v < 1 for v in fractions) or
            len(set(fractions)) != len(fractions)):
        raise ValueError("noise_fractions must contain 1-10 distinct finite fractions strictly between zero and one")
    for name in ("gradient_cosine_min", "gradient_relative_l2_max", "prediction_relative_l2_max", "loss_relative_error_max"):
        value = settings[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"diagnostic.{name} must be in (0,1]")
    if settings["named_phrase"] is not None and (
            not isinstance(settings["named_phrase"], str) or not settings["named_phrase"].strip()):
        raise ValueError("named_phrase must be a nonempty string or null to inherit the source value")
    result["diagnostic"] = settings
    return result


def checked_output_path(config, name, source_run=None):
    from ..data import assert_writable_path
    if not isinstance(name, str) or name in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Diagnostic job name must be a simple directory name")
    output = assert_writable_path(Path(config["training_folder"]) / name)
    source = Path(config["source_checkpoint"]).expanduser().resolve()
    protected = [source]
    if source_run is not None:
        protected.append(Path(source_run).expanduser().resolve())
    for path in protected:
        if output == path or path in output.parents or output in path.parents:
            raise ValueError("Diagnostic output must be separate from the source package/run")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Diagnostic output already exists; choose a fresh job name")
    return output
