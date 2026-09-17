"""Opt-in real-VM continuation and fresh inference-package acceptance."""
from __future__ import annotations

import copy
from pathlib import Path

import torch

from ..acceptance import compare_trees
from ..data import assert_writable_path
from ..recording import write_json
from .config import resolve_process_config
from .process import V2Runner


def _snapshot(root):
    from safetensors.torch import load_file
    root = Path(root)
    state = torch.load(root / "training_state.pt", map_location="cpu", weights_only=True)
    return {"tokens": load_file(str(root / "tokens.safetensors")), "engine": state["engine"],
            "rng": state["runtime"]["rng"], "data": state["runtime"]["data"]}


def run_acceptance(raw, output, *, split_after=None, atol=0., rtol=0.):
    """Runs the supplied real smoke three times (full, partial, continuation).

    This is intentionally explicit and expensive. It never substitutes a toy
    backend for actual Ideogram, quantization or the native data pipeline.
    """
    from .inference import generate, load_package
    config = resolve_process_config(raw)
    total = config["train"]["steps"]
    split = total // 2 if split_after is None else split_after
    if not 0 < split < total:
        raise ValueError("Acceptance split must be strictly inside the configured update budget")
    output = assert_writable_path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Acceptance requires a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    config["training_folder"] = str(output)
    config["gen2"]["checkpoint"]["resume_from"] = None
    report = {"trainer_version": "2.0.0", "total_updates": total, "split_after": split,
              "backend": "real_native_ideogram", "visual_acceptance": "pending_user_review"}
    runner = package = None
    try:
        runner = V2Runner(copy.deepcopy(config), "continuous")
        final_package = runner.run()
        expected = _snapshot(runner.checkpoints.resume_path)
        options = config["sample"]
        if not options["prompts"]:
            raise ValueError("Real package acceptance needs a sampling prompt")
        prompt = options["prompts"][0]
        settings = {"mode": "learned", "width": options["width"], "height": options["height"],
            "seed": config["gen2"]["evaluation"]["seeds"][0], "steps": options["sample_steps"],
            "guidance": options["guidance_scale"]}
        old = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(config["gen2"]["execution"]["deterministic_algorithms"])
            image, _ = generate(runner.backend, prompt, **settings)
        finally:
            torch.use_deterministic_algorithms(old, warn_only=warn)
        expected_pixels = bytes(image.tobytes())
        image.save(output / "live_final.png")
        runner.release(); runner = None

        runner = V2Runner(copy.deepcopy(config), "split")
        runner.run(stop_after=split)
        resume_path = runner.checkpoints.resume_path
        runner.release(); runner = None
        resumed_config = copy.deepcopy(config)
        resumed_config["gen2"]["checkpoint"]["resume_from"] = str(resume_path)
        runner = V2Runner(resumed_config, "split")
        runner.run()
        actual = _snapshot(runner.checkpoints.resume_path)
        report["continuous_vs_resumed"] = compare_trees(expected, actual, atol=atol, rtol=rtol)
        runner.release(); runner = None
        package = load_package(final_package)
        restored, _ = package.generate(prompt, **settings)
        restored.save(output / "reloaded_final.png")
        report["package_image_exact"] = bytes(restored.tobytes()) == expected_pixels
        report["passed"] = report["continuous_vs_resumed"]["equal"] and report["package_image_exact"]
        write_json(output / "acceptance_report.json", report)
        if not report["passed"]:
            raise RuntimeError(f"V2 real-backend acceptance failed; see {output / 'acceptance_report.json'}")
        return report
    except BaseException as error:
        report.update(passed=False, error_type=type(error).__name__, error=str(error))
        write_json(output / "acceptance_report.json", report)
        raise
    finally:
        if runner is not None:
            runner.release()
        if package is not None:
            package.release()
