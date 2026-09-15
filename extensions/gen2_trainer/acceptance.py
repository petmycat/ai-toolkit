"""Real CUDA acceptance: native D/A/G, exact continuation and package inference.

No synthetic model is selected by this entry point. The user supplies actual
Ideogram/Qwen weights and captioned VM data through the ordinary smoke config.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import torch

from .process import Gen2Runner


def tree_hash(value):
    """Hash complete optimizer/scheduler structures, including every tensor byte."""
    from .diagnostics import tensor_hash
    digest = hashlib.sha256()
    def visit(node):
        digest.update(type(node).__name__.encode())
        if torch.is_tensor(node):
            digest.update(tensor_hash(node).encode())
        elif isinstance(node, dict):
            for key in sorted(node, key=lambda item: (type(item).__name__, repr(item))):
                visit(key); visit(node[key])
        elif isinstance(node, (tuple, list)):
            for item in node:
                visit(item)
        elif node is None or isinstance(node, (str, int, float, bool)):
            digest.update(repr(node).encode())
        else:
            raise TypeError(f"Unsupported native checkpoint object in acceptance hash: {type(node).__name__}")
    visit(value)
    return digest.hexdigest()


def compare_trees(expected, actual, *, atol=0., rtol=0.):
    """Compare all recursive states; tolerances apply only to floating tensors.

    Integers, structure, dtypes, counters, scalar settings and RNG bytes remain
    exact. Differences are bounded in the report, comparison coverage is not.
    """
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("Comparison tolerances must be finite and nonnegative")
    report = {"equal": True, "atol": atol, "rtol": rtol, "compared_tensors": 0,
              "compared_tensor_elements": 0, "difference_count": 0, "differences": []}
    def fail(path, reason, **detail):
        report["equal"] = False
        report["difference_count"] += 1
        if len(report["differences"]) < 100:
            report["differences"].append({"path": path, "reason": reason, **detail})
    def visit(first, second, path):
        if torch.is_tensor(first):
            if not torch.is_tensor(second):
                fail(path, "tensor_type"); return
            if first.dtype != second.dtype or first.shape != second.shape:
                fail(path, "tensor_shape_or_dtype", expected=str((first.shape, first.dtype)), actual=str((second.shape, second.dtype))); return
            report["compared_tensors"] += 1
            report["compared_tensor_elements"] += first.numel()
            a, b = first.detach().cpu().reshape(-1), second.detach().cpu().reshape(-1)
            largest, mismatch = 0., False
            for offset in range(0, a.numel(), 1048576):
                x, y = a[offset:offset+1048576], b[offset:offset+1048576]
                if x.is_floating_point():
                    if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
                        fail(path, "nonfinite_tensor"); return
                    error = (x.double()-y.double()).abs()
                    largest = max(largest, float(error.max()) if error.numel() else 0.)
                    mismatch |= not bool((error <= atol+rtol*x.double().abs()).all())
                else:
                    mismatch |= not torch.equal(x, y)
            if mismatch:
                fail(path, "tensor_values", max_abs=largest if first.is_floating_point() else None)
        elif isinstance(first, dict):
            if not isinstance(second, dict) or first.keys() != second.keys():
                fail(path, "mapping_keys"); return
            for key in first:
                visit(first[key], second[key], f"{path}.{key}")
        elif isinstance(first, (tuple, list)):
            if type(first) is not type(second) or len(first) != len(second):
                fail(path, "sequence_structure"); return
            for index, (x, y) in enumerate(zip(first, second)):
                visit(x, y, f"{path}[{index}]")
        elif type(first) is not type(second) or first != second:
            fail(path, "scalar_value", expected=repr(first), actual=repr(second))
    visit(expected, actual, "state")
    return report


class AcceptanceRunner(Gen2Runner):
    def __init__(self, *args, evidence, **kwargs):
        self.acceptance_evidence = evidence
        super().__init__(*args, **kwargs)

    def _load(self):
        from .diagnostics import module_hash
        super()._load()
        if not self.evaluation.examples:
            raise RuntimeError("Real acceptance requires the mandatory real-data probe packet")
        options = self.config["gen2"]["diagnostics"]
        prefix = self.backend.verify_prefix([self.evaluation.examples[0]["q"]],
            atol=options["prefix_atol"], rtol=options["prefix_rtol"])
        self.acceptance_evidence["prefix_checks"].append({"load_update": self.engine.logical_update, "rows": prefix})
        native_step = self.engine.step
        def observe_step(window):
            kind = self.engine.schedule.kind_at(self.engine.logical_update)
            a_before = _conditioning_effect_probe(self.backend, window[0]) if kind == "A" else None
            before_components = {role: module_hash(module) for role, module in self.backend.components().items()}
            before_optimizers = {role: tree_hash(optimizer.state_dict()) for role, optimizer in self.engine.optimizers.items()}
            before_steps = copy.deepcopy(self.engine.family_steps)
            result = native_step(window)
            after_components = {role: module_hash(module) for role, module in self.backend.components().items()}
            after_optimizers = {role: tree_hash(optimizer.state_dict()) for role, optimizer in self.engine.optimizers.items()}
            active = set(result["active_families"])
            evidence = {"logical_update": result["logical_update"], "update_kind": result["update_kind"],
                "update_attempt_id": result["update_attempt_id"], "active_families": sorted(active),
                "component_before": before_components, "component_after": after_components,
                "optimizer_before": before_optimizers, "optimizer_after": after_optimizers,
                "steps_before": before_steps, "steps_after": copy.deepcopy(self.engine.family_steps),
                "component_changed": {role: before_components[role] != after_components[role] for role in before_components},
                "gradients": result["gradients"]}
            if a_before is not None:
                a_after = _conditioning_effect_probe(self.backend, window[0])
                evidence["actual_A_effect"] = {key: {
                    "changed": not torch.equal(a_before[key], a_after[key]),
                    "max_abs": float((a_before[key].float()-a_after[key].float()).abs().max())}
                    for key in a_before}
            self.acceptance_evidence["updates"].append(evidence)
            for role in before_components:
                if role not in active and (before_components[role] != after_components[role] or before_optimizers[role] != after_optimizers[role]):
                    raise AssertionError(f"Inactive {role} parameters or native optimizer state changed on {result['update_kind']}")
                if self.engine.family_steps[role] != before_steps[role]+int(role in active):
                    raise AssertionError(f"Incorrect successful optimizer counter for {role}")
            return result
        self.engine.step = observe_step


@torch.no_grad()
def _conditioning_effect_probe(backend, batch):
    """Observe actually consumed C+ and velocity before/after a real A step."""
    tau = batch["tau"][:1]
    # The harness step may be inside scheduled training activation recording.
    # Keep this extra diagnostic isolated from those required training rows.
    with backend.diagnostics(None), backend.branch(tau, gate_mode="one", name="acceptance_A_effect"):
        condition = backend.encode(batch["qs"][:1], styled=True)
        velocity = backend.predict(batch["zt"][:1], tau, condition)
    return {"consumed_features": condition.features[0].detach().cpu(), "velocity": velocity.detach().cpu()}


@torch.no_grad()
def prediction_packet(backend, example, gate_mode):
    """Real native consumed features and 2x2 velocities on one fixed latent state."""
    model = backend.model
    z0 = example["z0"].to(model.device_torch, model.torch_dtype)
    noise = example["noise"].to(model.device_torch, model.torch_dtype)
    tau = torch.full((z0.shape[0],), .5, device=model.device_torch, dtype=torch.float32)
    zt = model.add_noise(z0, noise, tau*1000)
    result = {"tau": tau.cpu(), "zt": zt.cpu(), "q": example["q"]}
    neutral = backend.encode([example["q"]], styled=False)
    styled = backend.encode([example["q"]], styled=True)
    result["neutral_features"] = [value.cpu() for value in neutral.features]
    result["styled_features"] = [value.cpu() for value in styled.features]
    for name, enabled, condition in (("v11", True, styled), ("v10", True, neutral),
                                     ("v01", False, styled), ("v00", False, neutral)):
        with backend.branch(tau, lora_enabled=enabled, gate_mode=gate_mode, name="acceptance_"+name):
            result[name] = backend.predict(zt, tau, condition).cpu()
    return result


def _validate_evidence(evidence, schedule):
    expected_kinds = [schedule.kind_at(update) for update in range(schedule.total)]
    actual_kinds = [row["update_kind"] for row in evidence["updates"]]
    if actual_kinds != expected_kinds:
        raise AssertionError("Recorded real updates differ from the specified D/A/G sequence")
    if set(actual_kinds) != {"D", "A", "G"}:
        raise AssertionError("Real smoke acceptance must execute D, A and G updates")
    changes = {role: any(row["component_changed"][role] for row in evidence["updates"] if role in row["active_families"])
               for role in ("diffusion", "embedding", "text_adapter", "gates")}
    if not all(changes.values()):
        raise AssertionError(f"One or more real trainable families never changed: {changes}")
    conditioning_effect = {key: any(row.get("actual_A_effect", {}).get(key, {}).get("changed", False)
                                   for row in evidence["updates"])
                           for key in ("consumed_features", "velocity")}
    if not all(conditioning_effect.values()):
        raise AssertionError(f"Real A updates did not visibly change consumed features and velocity: {conditioning_effect}")
    return {"update_sequence": actual_kinds, "family_changed_when_active": changes,
            "real_A_conditioning_effect": conditioning_effect,
            "inactive_component_and_optimizer_states_unchanged": True, "all_hashes_cover_full_state": True}


def run_acceptance(config, output, *, split_after=None, atol=0., rtol=0.):
    """Run real training three times and reload the final complete package.

    The continuous and interrupted paths each perform exactly the configured
    number of updates; the interrupted path is loaded twice. Outputs are kept.
    """
    from .checkpointing import CheckpointManager
    from .config import resolve_process_config, SPEC_SHA256
    from .diagnostics import export_diagnostics
    from .engine import PhaseSchedule
    from .inference import generate
    from .package import load_package
    from .provenance import environment_manifest
    from .recording import assert_writable_path, write_json
    config = resolve_process_config(config)
    schedule = PhaseSchedule(**config["gen2"]["phases"])
    if {schedule.kind_at(i) for i in range(schedule.total)} != {"D", "A", "G"}:
        raise ValueError("Acceptance config must include real D, A and G updates")
    if any(dataset["num_workers"] != 0 for dataset in config["datasets"]):
        raise ValueError("Reference continuation acceptance requires native dataset num_workers=0")
    if config["gen2"]["checkpoint"]["resume_from"]:
        raise ValueError("Acceptance begins from fresh initialization; clear resume_from")
    if not torch.cuda.is_available():
        raise RuntimeError("Real acceptance requires the configured CUDA VM; no mock or CPU substitute is run")
    if atol < 0 or rtol < 0 or not math.isfinite(atol) or not math.isfinite(rtol):
        raise ValueError("Acceptance tolerances must be finite and nonnegative")
    split_after = schedule.total//2 if split_after is None else split_after
    if not 0 < split_after < schedule.total:
        raise ValueError("--split-after must be strictly between zero and train.steps")
    output = assert_writable_path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Acceptance output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    config["training_folder"] = str(output/"runs")
    report = {"schema_version": "1.0.0", "status": "running", "environment": environment_manifest(),
        "spec_sha256": SPEC_SHA256, "split_after": split_after, "atol": atol, "rtol": rtol,
        "comparison_policy": "floating tensors use explicit tolerance; all structures, scalar settings, counters and RNG bytes exact",
        "continuous": {"updates": [], "prefix_checks": []}, "interrupted": {"updates": [], "prefix_checks": []}}
    write_json(output/"acceptance.json", report)
    started = time.perf_counter()
    runners = []
    package = None
    try:
        print("Gen2 acceptance: continuous real training", flush=True)
        continuous = AcceptanceRunner(copy.deepcopy(config), "continuous", evidence=report["continuous"])
        runners.append(continuous)
        continuous_path = continuous.run()
        report["continuous"]["verified"] = _validate_evidence(report["continuous"], schedule)
        report["continuous"]["checkpoint"] = str(continuous_path)
        continuous.release()

        print(f"Gen2 acceptance: real training to split update {split_after}", flush=True)
        interrupted = AcceptanceRunner(copy.deepcopy(config), "interrupted", evidence=report["interrupted"])
        runners.append(interrupted)
        split_path = interrupted.run(stop_after=split_after)
        report["split_checkpoint"] = str(split_path)
        interrupted.release()

        print("Gen2 acceptance: strict native resume and remaining updates", flush=True)
        resumed_config = copy.deepcopy(config)
        resumed_config["gen2"]["checkpoint"]["resume_from"] = str(split_path)
        resumed = AcceptanceRunner(resumed_config, "interrupted", evidence=report["interrupted"])
        runners.append(resumed)
        resumed_path = resumed.run()
        report["interrupted"]["verified"] = _validate_evidence(report["interrupted"], schedule)
        report["interrupted"]["checkpoint"] = str(resumed_path)
        example = copy.deepcopy(resumed.evaluation.examples[0])
        gate_mode = resumed.evaluation.gate_mode(schedule.total)
        live_packet = prediction_packet(resumed.backend, example, gate_mode)
        sample = config["sample"]
        live_image, live_image_metadata = generate(resumed.backend, example["q"], mode="full",
            width=sample["width"], height=sample["height"], seed=sample["seed"], steps=sample["sample_steps"],
            guidance=sample["guidance_scale"], gate_mode=gate_mode)
        live_image.save(output/"live_preview.png")
        write_json(output/"live_preview.json", live_image_metadata)
        resumed.release()

        print("Gen2 acceptance: compare every saved master, optimizer and continuation state", flush=True)
        manager = CheckpointManager(continuous_path.parent, continuous_path/"specification.md", SPEC_SHA256, read_only=True)
        first = manager.load(continuous_path)
        second = manager.load(resumed_path)
        report["continuation_components"] = compare_trees(first["components"], second["components"], atol=atol, rtol=rtol)
        report["continuation_engine"] = compare_trees(first["engine_state"], second["engine_state"], atol=atol, rtol=rtol)
        report["continuation_rng"] = compare_trees(first["rng_state"], second["rng_state"])
        report["continuation_update_ids"] = compare_trees(
            [row["update_attempt_id"] for row in report["continuous"]["updates"]],
            [row["update_attempt_id"] for row in report["interrupted"]["updates"]])
        del first, second

        print("Gen2 acceptance: independent complete-package native reload", flush=True)
        package = load_package(resumed_path)
        reloaded_packet = prediction_packet(package.backend, example, package.gate_mode)
        report["reloaded_native_predictions"] = compare_trees(live_packet, reloaded_packet, atol=atol, rtol=rtol)
        # The actual sampled PNG pixels are compared exactly, irrespective of
        # tensor tolerances. Both routes call the same shared sampler function.
        reloaded_image, image_metadata = package.generate(example["q"], mode="full", width=sample["width"],
            height=sample["height"], seed=sample["seed"], steps=sample["sample_steps"], guidance=sample["guidance_scale"])
        reloaded_image.save(output/"package_inference.png")
        write_json(output/"package_inference.json", image_metadata)
        report["preview_package_image_parity"] = {"equal": live_image.mode == reloaded_image.mode
            and live_image.size == reloaded_image.size and live_image.tobytes() == reloaded_image.tobytes(),
            "policy": "exact PNG pixel equality"}
        torch.save({"live": live_packet, "reloaded": reloaded_packet, "example": example}, output/"inference_probe.pt")
        checks = ("continuation_components", "continuation_engine", "continuation_rng", "continuation_update_ids",
                  "reloaded_native_predictions", "preview_package_image_parity")
        report["status"] = "passed" if all(report[key]["equal"] for key in checks) else "failed"
        report["checks"] = list(checks)
        export_diagnostics(output/"runs"/"continuous"/"gen2", output/"continuous_diagnostics.zip")
        export_diagnostics(output/"runs"/"interrupted"/"gen2", output/"interrupted_diagnostics.zip")
        report["seconds"] = time.perf_counter()-started
        write_json(output/"acceptance.json", report)
        if report["status"] != "passed":
            raise AssertionError(f"Real acceptance differences recorded in {output/'acceptance.json'}")
        return report
    except BaseException as error:
        report["status"] = "failed"
        report["seconds"] = time.perf_counter()-started
        report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        write_json(output/"acceptance.json", report)
        raise
    finally:
        if package is not None:
            package.release()
        for runner in runners:
            runner.release()
