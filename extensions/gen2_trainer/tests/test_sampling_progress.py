"""Sampling progress reports completed work without changing generated images."""
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from extensions.gen2_trainer.evaluation import Evaluation
from extensions.gen2_trainer.inference import generate
from extensions.gen2_trainer.tests.test_visual_modes import RecordingBackend, native_sampling_utilities


def test_generate_callback_counts_successful_steps_without_changing_pixels_or_rng(monkeypatch):
    backend = RecordingBackend()
    progress = []
    rng = torch.random.get_rng_state().clone()

    def unexpected_sync(*args, **kwargs):
        pytest.fail("Progress must not introduce explicit CUDA synchronization")

    def observe(completed, total):
        assert backend.current_branch is None
        progress.append((completed, total))

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_sync)
    with native_sampling_utilities():
        expected, metadata = generate(backend, "[trigger] cat", "full", width=2, height=2, steps=8, seed=7)
        actual, actual_metadata = generate(backend, "[trigger] cat", "full", width=2, height=2,
                                            steps=8, seed=7, progress_callback=observe)
    assert progress == [(step, 8) for step in range(1, 9)]
    assert expected.tobytes() == actual.tobytes()
    assert metadata == actual_metadata
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


def test_generate_callback_does_not_report_failed_step():
    backend, progress = RecordingBackend(fail_on="cfg_unconditional"), []
    with native_sampling_utilities(), pytest.raises(RuntimeError, match="sampling fixture failure"):
        generate(backend, "[trigger] cat", "full", width=2, height=2, steps=3,
                 progress_callback=lambda completed, total: progress.append((completed, total)))
    assert progress == []
    assert backend.current_branch is None


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    native = ModuleType("toolkit.config_modules")

    class GenerateImageConfig:
        def __init__(self, **kwargs):
            self.output_path = kwargs["output_path"]

        def save_image_atomic(self, image):
            image.save(self.output_path)

    native.GenerateImageConfig = GenerateImageConfig
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setattr("extensions.gen2_trainer.provenance.code_identity", lambda: {"fixture": True})
    rows = []
    recorder = SimpleNamespace(record=lambda kind, row: rows.append((kind, row)))
    config = {"sample": {"prompts": ["[trigger] cat"], "width": 2, "height": 2,
                         "sample_steps": 8, "guidance_scale": 3.},
              "gen2": {"diagnostics": {}, "execution": {"diagnostic_seed": 123},
                       "evaluation": {"prompt_groups": {}, "make_contact_sheets": False},
                       "inference": {"lora_strength": 1.},
                       "phases": {"warmup_updates": 2, "refinement_updates": 8}}}
    result = Evaluation(RecordingBackend(), config, recorder, tmp_path)
    result.test_rows = rows
    return result


def run_samples(evaluation, modes=("full", "base"), seeds=(42,), update=6):
    with native_sampling_utilities():
        evaluation.sample(update, modes, seeds, ["fixture"], "package-hash", {"diffusion": "hash"})


def test_console_reports_image_identity_quarter_steps_and_elapsed_completion(evaluation, capsys):
    run_samples(evaluation)
    output = capsys.readouterr().out
    assert "update 6: 2 new image(s) to generate" in output
    for number, mode in enumerate(("full", "base"), start=1):
        prefix = f"update 6 image {number}/2 | p000 | seed 42 | {mode}"
        assert f"{prefix} | starting" in output
        assert f"{prefix} | complete in " in output
        for step in (1, 2, 4, 6, 8):
            assert f"{prefix} | denoising {step}/8 | " in output
        assert f"{prefix} | denoising 3/8" not in output
    assert "update 6: complete, 2/2 new image(s) in " in output
    assert len(evaluation.sampled_requests) == len(evaluation.test_rows) == 2
    assert len(list((evaluation.root / "samples").glob("*.png"))) == 2


def test_progress_total_excludes_previous_requests_and_repeated_input_modes(evaluation, capsys):
    run_samples(evaluation, modes=("full",))
    capsys.readouterr()
    run_samples(evaluation, modes=("full", "base", "base"), seeds=(42, 42))
    output = capsys.readouterr().out
    assert "update 6: 1 new image(s) to generate" in output
    assert "image 1/1 | p000 | seed 42 | base | starting" in output
    assert "| full |" not in output
    assert output.count("| starting") == 1
    assert len(evaluation.sampled_requests) == 2
    run_samples(evaluation)
    output = capsys.readouterr().out
    assert "update 6: 0 new image(s) to generate" in output
    assert "complete, 0/0 new image(s)" in output
    assert "| starting" not in output


def test_failure_is_visible_and_failed_request_is_retryable(evaluation, capsys):
    evaluation.backend.fail_on = "cfg_unconditional"
    with pytest.raises(RuntimeError, match="sampling fixture failure"):
        run_samples(evaluation)
    output = capsys.readouterr().out
    assert "image 1/2 | p000 | seed 42 | full | FAILED after " in output
    assert "RuntimeError: sampling fixture failure" in output
    assert "| complete in" not in output
    assert "update 6: complete" not in output
    assert evaluation.sampled_requests == set()
    assert evaluation.test_rows == []
    evaluation.backend.fail_on = None
    run_samples(evaluation)
    assert "update 6: complete, 2/2 new image(s)" in capsys.readouterr().out
    assert len(evaluation.sampled_requests) == 2
