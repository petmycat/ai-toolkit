"""Model-free validation and explicit extension utilities.

Normal training continues to use ``python run.py CONFIG.yaml``.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys


def read_config(path, process_index=0):
    from toolkit.config import get_config
    # Native lightweight parser owns ${ENV}, [name] and exponent handling.
    document = json.loads(json.dumps(get_config(str(path))))
    if not isinstance(document, dict) or document.get("job") != "extension":
        raise ValueError("Expected the ordinary ai-toolkit job: extension YAML document")
    config = document.get("config", {})
    processes = config.get("process", [])
    if not isinstance(processes, list) or not 0 <= process_index < len(processes):
        raise ValueError("--process-index does not select an existing config.process entry")
    selected = processes[process_index]
    if not isinstance(selected, dict) or selected.get("type") != "gen2_trainer":
        raise ValueError("The selected process must have type: gen2_trainer")
    name = selected.get("name", config.get("name"))
    if not isinstance(name, str) or not name:
        raise ValueError("config.name must name the run")
    return copy.deepcopy(selected), name


def parser():
    from .config import MODES
    result = argparse.ArgumentParser(prog="python -m extensions.gen2_trainer",
                                     description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Resolve and validate Gen2 config without loading model dependencies")
    validate.add_argument("config")
    validate.add_argument("--process-index", type=int, default=0)
    validate.add_argument("--output", help="Optional destination for the complete resolved process YAML")
    inspect = commands.add_parser("inspect", help="Validate all checksums and inspect a complete package")
    inspect.add_argument("checkpoint")
    infer = commands.add_parser("infer", help="Load a complete package and generate one image on CUDA")
    infer.add_argument("checkpoint")
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--output", required=True, help="PNG output; JSON metadata is saved beside it")
    infer.add_argument("--device")
    infer.add_argument("--mode", choices=MODES,
        help="Omit for production literal-trigger routing")
    infer.add_argument("--width", type=int, default=1024)
    infer.add_argument("--height", type=int, default=1024)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--steps", type=int, default=30)
    infer.add_argument("--guidance", type=float, default=7.)
    infer.add_argument("--strength", type=float)
    summarize = commands.add_parser("summarize", help="Regenerate descriptive JSON/Markdown summaries")
    summarize.add_argument("run_dir")
    export = commands.add_parser("export", help="Export diagnostic evidence, excluding model weights/dataset")
    export.add_argument("run_dir")
    export.add_argument("--output", required=True)
    export.add_argument("--include-images", action="store_true")
    acceptance = commands.add_parser("acceptance", help="Run actual CUDA continuous/split/resume/package acceptance")
    acceptance.add_argument("config", help="Use the small VM smoke configuration with real model/dataset paths")
    acceptance.add_argument("--process-index", type=int, default=0)
    acceptance.add_argument("--output", required=True, help="A new directory outside the protected repository gen2 folder")
    acceptance.add_argument("--split-after", type=int, help="Save/reload at this committed update; default is the middle of the run")
    acceptance.add_argument("--atol", type=float, default=0., help="Explicit floating-tensor comparison tolerance; default exact")
    acceptance.add_argument("--rtol", type=float, default=0., help="Explicit floating-tensor relative tolerance; default exact")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.command == "validate":
        from .config import resolve_process_config, SPEC_SHA256
        from .recording import assert_writable_path
        import yaml
        raw, name = read_config(arguments.config, arguments.process_index)
        resolved = resolve_process_config(raw)
        if arguments.output:
            output = assert_writable_path(arguments.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(json.dumps({"valid": True, "name": name, "spec_sha256": SPEC_SHA256,
            "resolved": resolved, "models_loaded": False}, indent=2, ensure_ascii=False))
    elif arguments.command == "inspect":
        from .checkpointing import load_manifest
        from .config import SPEC_SHA256
        print(json.dumps(load_manifest(arguments.checkpoint, expected_spec_sha256=SPEC_SHA256), indent=2, ensure_ascii=False))
    elif arguments.command == "infer":
        from .package import load_package
        from .recording import assert_writable_path, write_json
        output = assert_writable_path(arguments.output)
        if output.suffix.lower() != ".png":
            raise ValueError("--output must end in .png")
        metadata_path = output.with_suffix(".json")
        if output.exists() or metadata_path.exists():
            raise ValueError("Image or metadata output already exists; choose a new output path")
        package = load_package(arguments.checkpoint, device=arguments.device)
        try:
            image, metadata = package.generate(arguments.prompt, mode=arguments.mode,
                width=arguments.width, height=arguments.height, seed=arguments.seed,
                steps=arguments.steps, guidance=arguments.guidance, strength=arguments.strength)
            output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output)
            write_json(metadata_path, metadata)
        finally:
            package.release()
        print(json.dumps({"image": str(output), "metadata": str(metadata_path)}, indent=2))
    elif arguments.command == "summarize":
        from .diagnostics import summarize_run
        print(json.dumps(summarize_run(arguments.run_dir), indent=2, ensure_ascii=False))
    elif arguments.command == "export":
        from .diagnostics import export_diagnostics
        output = export_diagnostics(arguments.run_dir, arguments.output, include_images=arguments.include_images)
        print(str(output))
    elif arguments.command == "acceptance":
        from .acceptance import run_acceptance
        raw, name = read_config(arguments.config, arguments.process_index)
        report = run_acceptance(raw, arguments.output, split_after=arguments.split_after,
                                atol=arguments.atol, rtol=arguments.rtol)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Gen2 error: {error}", file=sys.stderr)
        raise SystemExit(1)
