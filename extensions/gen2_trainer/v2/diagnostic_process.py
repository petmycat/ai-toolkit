"""Run post-training measurements through the ordinary ``run.py`` job entry."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

import torch

from ..diagnostics import capture_rng_state, restore_rng_state
from ..recording import json_safe, write_json
from .checkpoint import load_manifest
from .config import SPEC_SHA256, resolve_process_config
from .diagnostic_config import checked_output_path, resolve_diagnostic_config
from .diagnostic_data import build_fixed_packets, tensor_digest
from .process import V2Runner


class DiagnosticRecorder:
    """Small synchronous scalar logs, durable after every completed measurement."""
    STREAMS = {"paired_losses", "descent", "gradient_fidelity"}

    def __init__(self, root, events, budget_mb=256):
        self.root, self.events = Path(root), events
        self.budget = budget_mb * 1024**2
        self.bytes = 0

    def event(self, name, **values):
        self.events.event(name, **values)
        self.events.flush()

    def record(self, stream, value):
        if stream not in self.STREAMS:
            raise ValueError(f"Unsupported diagnostic stream: {stream}")
        record = json_safe({**value, "recorded_at": datetime.now(timezone.utc).isoformat()}, redact=True)
        raw = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        if self.bytes + len(raw) > self.budget:
            raise RuntimeError("Diagnostic scalar recording budget exhausted")
        with (self.root / f"{stream}.jsonl").open("ab") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        self.bytes += len(raw)


def source_run_path(source, manifest):
    source = Path(source).expanduser().resolve()
    if source.parent.name == "checkpoints" and source.parent.parent.name == "gen2_v2":
        return source.parents[2]
    config = manifest["metadata"]["resolved_config"]
    return Path(config["training_folder"]) / manifest["metadata"]["name"]


def inherited_config(manifest, options):
    source = deepcopy(manifest["metadata"]["resolved_config"])
    source.pop("_gen2_resolved", None)
    source["device"] = options["device"]
    source["training_folder"] = options["training_folder"]
    source["train"].update(batch_size=1, disable_sampling=True)
    source["gen2"]["checkpoint"]["resume_from"] = None
    source["gen2"]["diagnostics"]["mechanical_probes"] = False
    phrase = options["diagnostic"]["named_phrase"]
    if phrase is None:
        phrase = source["gen2"]["evaluation"]["named_phrase"]
    if not isinstance(phrase, str) or not phrase.strip():
        raise ValueError("This four-way diagnostic requires a named_phrase benchmark")
    source["gen2"]["evaluation"]["named_phrase"] = phrase
    # Fresh diagnostic AdamW state is deliberate, never an implicit optimizer
    # substitution when inspecting an export produced by another optimizer.
    if source["gen2"]["optimizer"]["type"] != "adamw":
        raise ValueError("Controlled descent currently requires a source trained with adamw")
    return source


class V2DiagnosticRunner:
    def __init__(self, raw, name):
        self.options = resolve_diagnostic_config(raw)
        self.name = name
        self.source = Path(self.options["source_checkpoint"]).expanduser().resolve()
        self.source_manifest = load_manifest(self.source, expected_sha=SPEC_SHA256)
        self.save_root = checked_output_path(self.options, name,
            source_run_path(self.source, self.source_manifest))
        self.config = resolve_process_config(inherited_config(self.source_manifest, self.options))
        self.root = self.save_root / "gen2_v2_diagnostic"
        self.runner = None

    def run(self):
        from .diagnostic_objective import run_objective_diagnostics
        from .diagnostic_gradients import run_gradient_diagnostics
        import yaml

        # Validate again immediately before the first write (including repeated
        # run() calls on the same runner object).
        checked_output_path(self.options, self.name, source_run_path(self.source, self.source_manifest))
        self.root.mkdir(parents=True, exist_ok=True)
        (self.save_root / "config.yaml").write_text(yaml.safe_dump({
            "job": "extension", "config": {"name": self.name, "process": [self.options]}},
            sort_keys=False, allow_unicode=True), encoding="utf-8")
        summary = {"diagnostic_version": "1.0.0", "status": "running", "execution_completed": False,
            "source_checkpoint": str(self.source), "source_package_hash": self.source_manifest["package_hash"],
            "source_update": self.source_manifest["logical_update"],
            "style_acceptance": "not_measured",
            "limitations": ["Training-image reconstruction loss is not a perceptual style score.",
                "Selected images/noise levels are a bounded subset, not a generalization test.",
                "Controlled descent uses fresh temporary optimizer moments; it does not resume training.",
                "Low-precision gradient tolerances are diagnostic thresholds, not proof of exact arithmetic."]}
        summary_path = self.root / "diagnostic_summary.json"
        write_json(summary_path, summary)
        rng = capture_rng_state()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        started = time.perf_counter()
        try:
            print(f"[Gen2 v2 diagnostic] source update {summary['source_update']} | "
                  f"report folder {self.root}", flush=True)
            settings = deepcopy(self.options["diagnostic"])
            settings["named_phrase"] = self.config["gen2"]["evaluation"]["named_phrase"]
            settings["optimizer"] = deepcopy(self.config["gen2"]["optimizer"])
            expanded = sum(len(ds["resolution"]) if isinstance(ds.get("resolution"), list) else 1
                           for ds in self.config["datasets"])
            upper_packets = expanded * settings["examples_per_resolution"] * len(settings["noise_seeds"]) * len(settings["noise_fractions"])
            descent_forwards = 2 * min(settings["descent_packet_count"], upper_packets) * (1 + 2 * settings["descent_steps"])
            print(f"[Gen2 v2 diagnostic] planned maximum {upper_packets} fixed packets / "
                  f"{4 * upper_packets} paired forwards; descent up to {descent_forwards} forwards "
                  f"including {settings['descent_steps']} updates x {settings['descent_packet_count']} packets "
                  "x 2 starting banks with backward; "
                  "five full backward checks plus selected linear checks", flush=True)
            write_json(self.root / "diagnostic_settings.json", settings)
            self.runner = V2Runner(self.config, self.name, requested_config=self.config)
            # _load is shared to preserve native quantization/data/model setup.
            # Never invoke its training loop, events(), or checkpoint publication.
            self.runner.root = self.root
            self.runner._load()
            expected = {key: self.runner.metadata[key] for key in
                        ("model_identities", "tokenizer_identity", "dataset_identity")}
            loaded = self.runner.checkpoints.load_inference(self.source, self.runner.backend,
                                                            expected_metadata=expected)
            if loaded["package_hash"] != self.source_manifest["package_hash"]:
                raise RuntimeError("Source package changed while loading the diagnostic")
            backend = self.runner.backend
            bank_before = {key: tensor_digest(value) for key, value in backend.tokens.state_dict().items()}
            recorder = DiagnosticRecorder(self.root, self.runner.recorder)
            recorder.event("source_package_verified", package_hash=loaded["package_hash"],
                           update=loaded["logical_update"], tokens=bank_before)
            packets, fidelity_packet, packet_manifest = build_fixed_packets(self.runner, settings, recorder)
            write_json(self.root / "packet_manifest.json", packet_manifest)
            summary["packet_count"] = len(packets)
            summary["gradient_fidelity"] = run_gradient_diagnostics(backend, fidelity_packet, settings, recorder)
            write_json(summary_path, summary)
            summary["objective"] = run_objective_diagnostics(backend, packets, settings, recorder)
            if {key: tensor_digest(value) for key, value in backend.tokens.state_dict().items()} != bank_before:
                raise RuntimeError("Diagnostic did not restore the loaded token bank exactly")
            if self.runner.engine.logical_update != 0 or self.runner.engine.optimizer.state:
                raise RuntimeError("Diagnostic unexpectedly advanced the training optimizer")
            self.runner._verify_frozen()
            if load_manifest(self.source, expected_sha=SPEC_SHA256) != self.source_manifest:
                raise RuntimeError("Source checkpoint changed during the diagnostic")
            gradient_status = summary["gradient_fidelity"].get("status", "inconclusive")
            checks_complete = (summary["gradient_fidelity"].get("complete", False)
                               and summary["objective"].get("completed", False))
            summary.update(execution_completed=True, source_package_unchanged=True,
                loaded_tokens_restored=True, frozen_weights_verified=True, training_optimizer_updates=0,
                checks_complete=checks_complete, has_failed_gradient_checks=gradient_status == "failed",
                status=("incomplete_checks" if not checks_complete else
                        "completed_with_findings" if gradient_status == "failed" else "completed"))
            recorder.event("diagnostic_complete", status=summary["status"], gradient_status=gradient_status,
                           visual_acceptance="not_measured")
            return summary
        except BaseException as error:
            summary.update(status="aborted", error_type=type(error).__name__, error=str(error),
                           traceback=traceback.format_exc())
            raise
        finally:
            summary["seconds"] = time.perf_counter() - started
            try:
                write_json(summary_path, summary)
                print(f"[Gen2 v2 diagnostic] {summary['status']} | summary: {summary_path}", flush=True)
            finally:
                try:
                    if self.runner is not None:
                        try:
                            if self.runner.logger_started:
                                self.runner.logger.finish()
                        finally:
                            try:
                                if self.runner.recorder is not None:
                                    self.runner.recorder.close()
                            finally:
                                self.runner.release()
                finally:
                    restore_rng_state(rng)
                    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


class V2DiagnosticProcess:
    def __init__(self, process_id, job, config):
        # Validate paths and source package before native process setup can write.
        self.runner = V2DiagnosticRunner(config, job.name)
        from jobs.process.BaseExtensionProcess import BaseExtensionProcess
        self.native = BaseExtensionProcess(process_id, job, deepcopy(config))

    def __getattr__(self, name):
        return getattr(self.native, name)

    def run(self):
        self.native.run()
        return self.runner.run()

    def on_error(self, error):
        pass
