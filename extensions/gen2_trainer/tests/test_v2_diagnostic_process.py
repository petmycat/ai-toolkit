"""Job lifecycle with real package checksums/recording and a CPU model fixture."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from extensions.gen2_trainer.recording import Recorder
from extensions.gen2_trainer.tests.test_v2_diagnostic_objective import BackendFixture, packet
from extensions.gen2_trainer.v2.checkpoint import V2CheckpointManager, load_manifest
from extensions.gen2_trainer.v2.config import resolve_process_config, SPEC_SHA256
from extensions.gen2_trainer.v2 import diagnostic_process as module
from extensions.gen2_trainer.v2.process import specification_path


@pytest.fixture
def job(tmp_path, monkeypatch):
    source_config = resolve_process_config({"training_folder": str(tmp_path), "trigger_word": "<test>",
        "model": {"name_or_path": "fixture"}, "datasets": [{"folder_path": "fixture"}],
        "train": {"steps": 500}, "sample": {"prompts": ["[trigger] tiger"]},
        "gen2": {"schema_version": "2.0.0", "inference": {"unconditional_model_path": "fixture"},
                 "evaluation": {"named_phrase": "reference"}}})
    identities = {"model_identities": {"model": "frozen_fixture_hash"},
                  "tokenizer_identity": {"tokenizer": "fixture_hash"}, "dataset_identity": [{"hash": "fixture"}]}
    source_root = tmp_path / "source" / "gen2_v2" / "checkpoints"
    manager = V2CheckpointManager(source_root, specification_path(source_config), SPEC_SHA256)
    metadata = {"name": "source", "resolved_config": source_config, **identities}
    export = manager.export(BackendFixture(), metadata, logical_update=500)
    source_bytes = {path.name: path.read_bytes() for path in export.iterdir()}
    loaded = []

    def fake_load(runner):
        runner.backend = BackendFixture()
        runner.recorder = Recorder(runner.root, "fixture-diagnostic")
        runner.engine = SimpleNamespace(logical_update=0, optimizer=SimpleNamespace(state={}))
        runner.metadata = deepcopy(identities)
        runner.checkpoints = V2CheckpointManager(runner.root / "checkpoints",
                                                specification_path(runner.config), SPEC_SHA256)
        loaded.append(runner.backend)

    def no_training(*args, **kwargs):
        raise AssertionError("Diagnostic must not call the training loop or publish checkpoints")

    monkeypatch.setattr(module.V2Runner, "_load", fake_load)
    monkeypatch.setattr(module.V2Runner, "_verify_frozen", lambda runner: runner.backend.assert_frozen())
    monkeypatch.setattr(module.V2Runner, "run", no_training)
    monkeypatch.setattr(module.V2Runner, "save", no_training)
    monkeypatch.setattr(module, "build_fixed_packets", lambda *args: ([packet()], packet(), {"packets": ["fixture"]}))
    monkeypatch.setattr("extensions.gen2_trainer.v2.diagnostic_gradients.run_gradient_diagnostics",
                        lambda *args: {"status": "passed", "complete": True})
    options = {"type": "gen2_v2_diagnostic", "source_checkpoint": str(export),
               "training_folder": str(tmp_path), "diagnostic": {"descent_steps": 1}}
    return options, export, source_bytes, loaded


@pytest.mark.parametrize("gradient_result, expected_status", [
    ({"status": "passed", "complete": True}, "completed"),
    ({"status": "failed", "complete": True}, "completed_with_findings"),
    ({"status": "failed", "complete": False}, "incomplete_checks"),
    ({"status": "inconclusive", "complete": False}, "incomplete_checks"),
])
def test_diagnostic_loads_export_restores_tokens_and_leaves_source_untouched(job, monkeypatch, gradient_result, expected_status):
    options, export, source_bytes, loaded = job
    monkeypatch.setattr("extensions.gen2_trainer.v2.diagnostic_gradients.run_gradient_diagnostics",
                        lambda *args: gradient_result)
    runner = module.V2DiagnosticRunner(options, "diagnostic")
    rng = torch.get_rng_state().clone()
    result = runner.run()
    assert torch.equal(rng, torch.get_rng_state())
    assert result["status"] == expected_status
    assert result["execution_completed"] is True
    assert result["training_optimizer_updates"] == 0
    assert result["source_update"] == 500
    assert result["style_acceptance"] == "not_measured"
    assert result["objective"]["completed"] is True
    assert {p.name: p.read_bytes() for p in export.iterdir()} == source_bytes
    assert load_manifest(export)["logical_update"] == 500
    assert not (runner.root / "checkpoints").exists()
    torch.testing.assert_close(loaded[0].tokens.E, BackendFixture().tokens.E, rtol=0, atol=0)
    saved = json.loads((runner.root / "diagnostic_summary.json").read_text())
    assert saved == module.json_safe(result)
    assert len((runner.root / "paired_losses.jsonl").read_text().splitlines()) == 1
    with pytest.raises(ValueError, match="already exists"):
        module.V2DiagnosticRunner(options, "diagnostic")


def test_failure_preserves_completed_gradient_evidence_and_writes_aborted_report(job, monkeypatch):
    options, export, source_bytes, _ = job

    def fail(*args):
        raise RuntimeError("injected objective failure")

    monkeypatch.setattr("extensions.gen2_trainer.v2.diagnostic_objective.run_objective_diagnostics", fail)
    runner = module.V2DiagnosticRunner(options, "failure")
    with pytest.raises(RuntimeError, match="injected objective failure"):
        runner.run()
    summary = json.loads((runner.root / "diagnostic_summary.json").read_text())
    assert summary["status"] == "aborted"
    assert summary["execution_completed"] is False
    assert summary["gradient_fidelity"]["status"] == "passed"
    assert summary["error_type"] == "RuntimeError"
    assert {p.name: p.read_bytes() for p in export.iterdir()} == source_bytes
    assert not (runner.root / "checkpoints").exists()


def test_extension_registration_is_lazy_and_has_distinct_uid():
    from extensions.gen2_trainer import AI_TOOLKIT_EXTENSIONS
    extensions = {cls.uid: cls for cls in AI_TOOLKIT_EXTENSIONS}
    assert set(extensions) == {"gen2_trainer", "gen2_v2_diagnostic"}
    assert extensions["gen2_v2_diagnostic"].get_process() is module.V2DiagnosticProcess
