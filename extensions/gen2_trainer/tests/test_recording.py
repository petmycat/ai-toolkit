"""Acceptance tests for recording, diagnostic isolation and portable evidence."""
import csv
import json
import math
import random
import zipfile
from contextlib import contextmanager

import pytest
import torch

from extensions.gen2_trainer.recording import (
    Recorder, RecordingBudgetExceeded, RecordingError, append_rating_template,
    assert_writable_path, iter_records, json_safe, write_json,
)
from extensions.gen2_trainer.diagnostics import (
    TensorDumpBudget, capture_rng_state, coordinate_indices, export_diagnostics,
    gate_statistics, gradient_statistics, interaction_metrics, isolated_gradient_probe,
    isolated_rng, load_fixed_probe_packet, low_rank_spectrum, module_hash,
    prefix_comparison, region_summaries, restore_rng_state, sample_coordinates,
    sampled_update_statistics, save_fixed_probe_packet, summarize_run, tensor_hash,
)


def test_append_rotation_resume_and_unscored_ratings(tmp_path):
    config = {"rotate_mb": .001, "compression": "gzip", "writer_queue_records": 1,
              "max_core_recording_mb": 1, "flush_every_updates": 2}
    recorder = Recorder(tmp_path, "run", config)
    for update in range(12):
        recorder.record("updates", {"losses": {"L_S": float(update), "L_N": None},
                                    "undefined_reasons": {"L_N": "not_in_objective"}},
                        logical_update=update, update_attempt_id=update, stage="refinement", update_kind="A")
    saved = recorder.state_dict()
    recorder.close()
    rows = list(iter_records(tmp_path, "updates"))
    assert len(rows) == 12
    assert [row["logical_update"] for row in rows] == list(range(12))
    assert len({row["event_id"] for row in rows}) == 12
    assert list(tmp_path.glob("updates.*.jsonl.gz"))
    recorder = Recorder(tmp_path, "run", config)
    recorder.event("initializing")  # Native runner logs this before resume().
    recorder.resume(saved)
    recorder.record("updates", {"value": 12}, logical_update=12, update_attempt_id="new-attempt")
    recorder.close()
    assert len(list(iter_records(tmp_path, "updates"))) == 13
    assert list(iter_records(tmp_path, "events"))[-1]["event"] == "resume"
    rating_path = append_rating_template(tmp_path / "human_ratings.csv", [{"image_id": "one"}])
    append_rating_template(rating_path, [{"image_id": "one"}, {"image_id": "two"}])
    with rating_path.open(newline="", encoding="utf-8") as handle:
        ratings = list(csv.DictReader(handle))
    assert len(ratings) == 2
    assert all(row["target_style_fidelity"] == "" for row in ratings)


def test_failures_never_silently_drop_or_emit_nonfinite_json(tmp_path):
    with Recorder(tmp_path, "run") as recorder:
        with pytest.raises(RecordingError, match="Nonfinite"):
            recorder.record("updates", {"loss": float("nan")})
        with pytest.raises(RecordingError, match="reduced JSON"):
            recorder.record("updates", {"loss": torch.tensor(1., requires_grad=True)})
        with pytest.raises(RecordingError, match="accumulation_index"):
            recorder.record("microbatches", {"loss": 1.})
        recorder.record("microbatches", {"accumulation_index": 0, "sample_id": "one", "loss": 1.})
    assert len(list(iter_records(tmp_path, "events"))) == 2
    assert list(iter_records(tmp_path, "microbatches"))[0]["example_ids"] == ["one"]
    assert "NaN" not in (tmp_path / "events.jsonl").read_text()


def test_hard_budget_aborts_and_flushes_accepted_records(tmp_path):
    recorder = Recorder(tmp_path, "run", {"max_core_recording_mb": .002, "writer_queue_records": 1})
    recorder.event("accepted")
    with pytest.raises(RecordingBudgetExceeded):
        recorder.record("updates", {"huge": "x" * 10000})
    with pytest.raises(RecordingBudgetExceeded):
        recorder.event("cannot_continue")
    recorder.close()
    rows = list(iter_records(tmp_path, "events"))
    assert rows[0]["event"] == "accepted"
    assert any(row["event"] == "recording_budget_exceeded" for row in rows)
    assert sum(path.stat().st_size for path in tmp_path.iterdir()) <= int(.002 * 1024 * 1024)


def test_external_required_packet_counts_against_core_budget(tmp_path):
    recorder = Recorder(tmp_path, "run", {"max_core_recording_mb": .001})
    (tmp_path / "fixed_probe_packet.pt").write_bytes(b"x" * 2048)
    with pytest.raises(RecordingBudgetExceeded, match="metadata/probe"):
        recorder.flush()
    recorder.close()


def test_status_snapshot_does_not_flush_or_scan_disk(tmp_path, monkeypatch):
    recorder = Recorder(tmp_path, "run")
    recorder.event("accepted")
    monkeypatch.setattr(recorder, "flush", lambda: pytest.fail("status must not flush"))
    monkeypatch.setattr(recorder, "enforce_core_budget", lambda: pytest.fail("status must not scan disk"))
    status = recorder.status()
    assert status["record_sequence"] == 1
    assert 0 < status["core_reserved_bytes"] <= status["core_budget_bytes"]
    assert 0 <= status["queue_size"] <= status["queue_capacity"]
    recorder.close()


def test_failed_writer_reaches_caller(tmp_path):
    recorder = Recorder(tmp_path, "run")
    def fail_open(stream):
        raise OSError("disk full fixture")
    recorder._open = fail_open
    recorder.event("will_fail")
    with pytest.raises(RecordingError, match="disk full"):
        recorder.flush()
    with pytest.raises(RecordingError, match="disk full"):
        recorder.close()


def test_incomplete_append_rejected_without_truncation(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"run_id":"run"}', encoding="utf-8")
    with pytest.raises(RecordingError, match="Incomplete"):
        Recorder(tmp_path, "run")
    assert path.read_text() == '{"run_id":"run"}'


def test_readonly_source_and_secret_redaction():
    from pathlib import Path
    source = Path(__file__).resolve().parents[3] / "gen2"
    with pytest.raises(PermissionError):
        assert_writable_path(source / "must_not_create.json")
    result = json_safe({"hf_token": "secret", "token": "credential", "initializer_token_ids": [2]}, redact=True)
    assert result == {"hf_token": "[redacted]", "token": "[redacted]", "initializer_token_ids": [2]}


def test_rng_isolation_restores_all_streams_on_exception():
    import numpy as np
    generator = torch.Generator().manual_seed(25)
    state = capture_rng_state({"diagnostic": generator})
    expected = (random.random(), np.random.rand(), torch.rand(3), torch.rand(3, generator=generator))
    restore_rng_state(state, {"diagnostic": generator})
    with pytest.raises(RuntimeError):
        with isolated_rng(197, {"diagnostic": generator}):
            random.random(), np.random.rand(), torch.rand(9), torch.rand(9, generator=generator)
            raise RuntimeError("probe failed")
    actual = (random.random(), np.random.rand(), torch.rand(3), torch.rand(3, generator=generator))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2]) and torch.equal(actual[3], expected[3])


def test_coordinate_samples_and_full_gradient_statistics():
    first = torch.nn.Parameter(torch.arange(5, dtype=torch.float32))
    second = torch.nn.Parameter(torch.arange(5, 10, dtype=torch.float32))
    indices = coordinate_indices(10, 4, 99)
    assert len(indices.unique()) == 4
    assert torch.equal(indices, coordinate_indices(10, 4, 99))
    assert torch.equal(sample_coordinates([first, second], indices), indices.float())
    first.grad = torch.ones_like(first)
    result = gradient_statistics([first, second])
    assert result["l2"] == pytest.approx(math.sqrt(5))
    assert result["missing_gradient_tensors"] == 1
    assert result["nonzero_fraction"] == .5
    update = sampled_update_statistics(torch.zeros(4), torch.ones(4), 10)
    assert update["method"] == "sampled"
    assert update["l2"] == pytest.approx(math.sqrt(10))


def test_probe_gradients_restore_context_ownership_and_training_grads():
    parameter = torch.nn.Parameter(torch.tensor([2., -1.]), requires_grad=False)
    original_grad = torch.tensor([7., 8.])
    parameter.grad = original_grad
    context = {"active": False}
    def objective(sign):
        @contextmanager
        def scope():
            context["active"] = True
            try:
                loss = sign * parameter.square().sum()
                loss.register_hook(lambda gradient: gradient if context["active"] else gradient * float("nan"))
                yield loss
            finally:
                context["active"] = False
        return scope
    result = isolated_gradient_probe({"positive": objective(1), "negative": objective(-1)},
                                     {"embedding": [parameter]}, max_coordinates=0)
    pair = result["families"]["embedding"]["pairs"]["positive:negative"]
    assert pair["cosine"] == pytest.approx(-1.)
    assert pair["first_l2"] == pytest.approx(math.sqrt(20))
    assert parameter.grad is original_grad and torch.equal(parameter.grad, torch.tensor([7., 8.]))
    assert not parameter.requires_grad and not context["active"]


def test_gradient_probe_chunks_one_graph_with_small_diagnostic_budget():
    first = torch.nn.Parameter(torch.arange(64, dtype=torch.float32) + 1)
    second = torch.nn.Parameter(torch.arange(64, dtype=torch.float32) + 1)
    result = isolated_gradient_probe(
        {"first": lambda: first.square().sum() + second.square().sum(),
         "second": lambda: -first.square().sum() - second.square().sum()},
        {"diffusion": [first, second]}, max_coordinates=0, memory_budget_mb=.001)
    assert result["working_gradient_chunks"] == 2
    assert result["families"]["diffusion"]["method"] == "sampled"
    assert result["families"]["diffusion"]["pairs"]["first:second"]["cosine"] == pytest.approx(-1)
    assert first.grad is None and second.grad is None


def test_low_rank_spectra_packing_regions_and_gate_statistics():
    generator = torch.Generator().manual_seed(13)
    A = torch.randn((3, 7), generator=generator)
    B = torch.randn((6, 3), generator=generator)
    result = low_rank_spectrum(A, B, scale=.5)
    expected = torch.linalg.svdvals(.5 * B @ A)
    assert torch.allclose(torch.tensor(result["singular_values"]), expected[:3], atol=2e-6, rtol=2e-6)
    assert low_rank_spectrum(A, B * 0)["entropy_effective_rank"] is None
    x = torch.ones(1, 3, 2)
    masks = {"original_text": torch.tensor([[1, 0, 0]]), "suffix": torch.tensor([[0, 1, 0]]),
             "image": torch.tensor([[0, 0, 1]]), "padding": torch.zeros(1, 3)}
    rows = region_summaries(x, x * 2, x, x * 3, masks)
    assert {row["token_region"] for row in rows} == {"original_text", "suffix", "image"}
    assert rows[0]["residual_base_ratio"] == .5
    assert rows[0]["outside_suffix_applied_max_abs"] == 3
    correct = region_summaries(x, x * 2, x, x * masks["suffix"].unsqueeze(-1), masks)
    assert all(row["outside_suffix_applied_max_abs"] == 0 for row in correct)
    gates = gate_statistics(torch.zeros(2, 4), torch.ones(5, 2), torch.zeros(5, 2), .5)
    assert len(gates["blocks"]) == 2 and gates["R_C"] == gates["R_H"] == 0


def test_interventions_prefix_and_hashes():
    base = torch.ones(2, 3)
    rows = interaction_metrics(base + 7, base + 2, base + 3, base, target=base, reference_v00=base)
    assert all(row["difference_rms"]["interaction"] == 2 for row in rows)
    assert all(row["base_drift"]["rms"] == 0 for row in rows)
    assert not prefix_comparison(base, base + 1)["within_tolerance"]
    assert prefix_comparison(base, base, baseline=base)["baseline_noise"]["rms"] == 0
    assert tensor_hash(base) != tensor_hash(base + .01)
    assert tensor_hash(base.to(torch.bfloat16)) == tensor_hash(base.to(torch.bfloat16).clone())
    assert module_hash({"a": base}) == module_hash({"a": base.clone()})


def test_summary_export_excludes_weights_and_preserves_mandatory_probes(tmp_path):
    run = tmp_path / "run"
    with Recorder(run, "test") as recorder:
        recorder.record("updates", {"loss": 1.}, stage="refinement", update_kind="D")
        recorder.record("updates", {"loss": 10.}, stage="refinement", update_kind="A")
    save_fixed_probe_packet(run / "fixed_probe", [{"z0": torch.ones(1, 2), "sample_id": "x"}], [.25, .75], 18)
    packet = load_fixed_probe_packet(run / "fixed_probe")
    assert packet["manifest"]["examples"][0]["sample_id"] == "x"
    (run / "fixed_probe_packet.pt").write_bytes(b"legacy-native-packet")
    (run / "caption_token_report.json").write_text('{"passed":false,"failures":[{"over_by":11}]}')
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "weights.safetensors").write_bytes(b"exclude")
    (run / "samples").mkdir(exist_ok=True)
    (run / "samples" / "preview.png").write_bytes(b"picture")
    append_rating_template(run / "human_ratings.csv", [{"image_id": "one"}])
    summary = summarize_run(run)
    assert summary["human_score_status"] == "unscored"
    assert {group["update_kind"] for group in summary["aggregates"]} == {"D", "A"}
    before = {str(path.relative_to(run)): path.read_bytes() for path in run.rglob("*") if path.is_file()}
    archive = export_diagnostics(run, tmp_path / "diagnostics.zip")
    after = {str(path.relative_to(run)): path.read_bytes() for path in run.rglob("*") if path.is_file()}
    assert before == after  # Export may consume evidence in a read-only folder.
    with zipfile.ZipFile(archive) as zipped:
        assert "fixed_probe/latents.safetensors" in zipped.namelist()
        assert "fixed_probe_packet.pt" in zipped.namelist()
        assert json.loads(zipped.read("caption_token_report.json"))["failures"][0]["over_by"] == 11
        assert not any("weights.safetensors" in name or name.endswith("preview.png") for name in zipped.namelist())


def test_summary_aggregates_nested_interventions_and_pairs_actual_ratings(tmp_path):
    images = []
    with Recorder(tmp_path, "run") as recorder:
        recorder.record("probes", {"metrics": [{"difference_rms": {"interaction": 2.}}]},
                        stage="refinement", update_kind="A")
        for index, mode in enumerate(("full", "base", "base")):
            image = {"image_id": str(index), "prompt_id": "p000", "checkpoint_hash": "shared",
                     "seed": 42, "width": 512, "height": 512, "steps": 20,
                     "guidance_scale": 7., "group": "user_group", "ablation_mode": mode,
                     "sampler": "euler", "sigma_schedule": [1., 0.] if index < 2 else [.5, 0.]}
            recorder.record("samples/manifest", image)
            images.append(image)
    rating_path = append_rating_template(tmp_path / "human_ratings.csv", images)
    with rating_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames, ratings = reader.fieldnames, list(reader)
    for index, rating in enumerate(ratings):
        rating["target_style_fidelity"] = str(3 - index)
    with rating_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ratings)
    summary = summarize_run(tmp_path, write=False)
    group = next(row for row in summary["aggregates"] if row["stream"] == "probes")
    assert group["metrics"]["metrics.examples.difference_rms.interaction"]["mean"] == 2
    assert len(summary["paired_human_differences"]) == 1
    assert summary["paired_human_differences"][0]["second_minus_first"]["target_style_fidelity"] == -1


def test_optional_packet_budget(tmp_path):
    budget = TensorDumpBudget(tmp_path, max_packets=1, max_total_mb=.001, enabled=True)
    assert budget.save("first", {"suffix_features": torch.ones(8)})["saved"]
    assert budget.save("second", {"suffix_features": torch.ones(8)})["reason"] == "packet_count_budget"


def test_fixed_packet_metadata_is_linked_and_incomplete_packets_reject(tmp_path):
    packet = save_fixed_probe_packet(tmp_path / "packet", [{"z0": torch.ones(2), "q": "content"}], [.25], 3)
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["examples"][0]["q"] = "changed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata checksum"):
        load_fixed_probe_packet(packet)
    (packet / "COMPLETE.json").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        load_fixed_probe_packet(packet)
