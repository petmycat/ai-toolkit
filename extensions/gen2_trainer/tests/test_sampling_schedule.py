"""Exercise real event orchestration without loading models or writing files."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from extensions.gen2_trainer.config import resolve_process_config
from extensions.gen2_trainer.engine import PhaseSchedule
from extensions.gen2_trainer.process import Gen2Runner


def runner_fixture(sample_every=3, save_every=6, milestone_every=6,
                   start=0, skip_initial=False, disabled=False, total=12):
    config = resolve_process_config({
        "train": {"steps": total, "skip_first_sample": skip_initial, "disable_sampling": disabled,
                  "validation_config": {"validate_every_n_steps": 5}},
        "save": {"save_every": save_every},
        "sample": {"sample_every": sample_every, "sample_start_step": start, "seed": 42},
        "gen2": {"phases": {"warmup_updates": 2, "refinement_updates": 8,
                            "calibration_updates": total-10},
                 "evaluation": {"milestone_every": milestone_every,
                                "preview_modes": ["full", "base"],
                                "milestone_modes": ["full", "full_uncond_half", "base"],
                                "additional_seeds": [43, 42]}}})
    runner = Gen2Runner.__new__(Gen2Runner)
    runner.config, runner.backend = config, object()
    runner.engine = SimpleNamespace(logical_update=0, update_attempt=0,
        schedule=PhaseSchedule(**config["gen2"]["phases"]), family_steps={})
    runner.evaluation, runner.recorder = Mock(), Mock()
    runner.saved_events = []
    runner.save = lambda reasons, protected: runner.saved_events.append(
        (runner.engine.logical_update, reasons, protected))
    return runner


def run_events(runner):
    with patch("extensions.gen2_trainer.evaluation.bundle_identity", return_value=("state-hash", {"diffusion": "hash"})):
        for update in range(runner.engine.schedule.total+1):
            runner.engine.logical_update = runner.engine.update_attempt = update
            runner.engine.family_steps = runner.engine.schedule.counts_at(update)
            runner.events(initial=update == 0)
    return runner.evaluation.sample.call_args_list


def updates(calls):
    return [call.args[0] for call in calls]


def test_sample_clock_is_independent_of_checkpoint_and_stage_schedules():
    runner = runner_fixture()
    calls = run_events(runner)
    assert updates(calls) == [0, 3, 6, 9, 12]
    # Existing protected checkpoints/numerical boundaries stay unchanged.
    assert [(update, protected) for update, _, protected in runner.saved_events] == [
        (0, True), (2, True), (6, False), (10, True), (12, True)]
    assert updates(runner.evaluation.numerical_probes.call_args_list) == [0, 2, 10, 12]
    assert updates(runner.evaluation.gradient_probes.call_args_list) == [0, 2, 10, 12]
    assert updates(runner.evaluation.validate.call_args_list) == [0, 2, 5, 10, 12]
    assert all(call.args[-2:] == ("state-hash", {"diffusion": "hash"}) for call in calls)
    different_saves = runner_fixture(save_every=1)
    assert updates(run_events(different_saves)) == [0, 3, 6, 9, 12]
    assert len(different_saves.saved_events) == 13


def test_milestones_expand_only_already_scheduled_samples_and_deduplicate():
    runner = runner_fixture(sample_every=4, milestone_every=6)
    calls = run_events(runner)
    assert updates(calls) == [0, 4, 8, 12]
    for call in calls:
        update, modes, seeds, reasons = call.args[:4]
        expanded = update in (0, 12)
        assert modes == (["full", "base", "full_uncond_half"] if expanded else ["full", "base"])
        assert seeds == ([42, 43] if expanded else [42])
        assert ("milestone" in reasons) == expanded
    # A phase boundary that happens to be a sample tick does not expand modes.
    runner = runner_fixture(sample_every=2, milestone_every=6)
    calls = {call.args[0]: call for call in run_events(runner)}
    assert calls[2].args[1:3] == (["full", "base"], [42])
    assert calls[10].args[1:3] == (["full", "base"], [42])


def test_selected_six_update_sample_interval_renders_zero_six_and_twelve_only():
    runner = runner_fixture(sample_every=6, save_every=6, milestone_every=6)
    calls = run_events(runner)
    assert updates(calls) == [0, 6, 12]
    assert all(call.args[1:3] == (["full", "base", "full_uncond_half"], [42, 43]) for call in calls)
    assert [update for update, _, _ in runner.saved_events] == [0, 2, 6, 10, 12]


@pytest.mark.parametrize("skip_initial,expected", [
    (False, [0, 6, 9, 12]), (True, [6, 9, 12])])
def test_initial_choice_is_separate_from_regular_start_step(skip_initial, expected):
    runner = runner_fixture(sample_every=3, start=5, skip_initial=skip_initial)
    assert updates(run_events(runner)) == expected


def test_disabled_images_do_not_disable_checkpoints_or_numerical_diagnostics():
    runner = runner_fixture(disabled=True)
    assert run_events(runner) == []
    assert [update for update, _, _ in runner.saved_events] == [0, 2, 6, 10, 12]
    assert updates(runner.evaluation.numerical_probes.call_args_list) == [0, 2, 10, 12]


def test_no_images_are_forced_at_final_boundary_or_save_when_interval_exceeds_run():
    runner = runner_fixture(sample_every=250, skip_initial=True)
    assert run_events(runner) == []
    assert [update for update, _, _ in runner.saved_events] == [0, 2, 6, 10, 12]
    runner = runner_fixture(sample_every=5, save_every=6, total=12)
    assert updates(run_events(runner)) == [0, 5, 10]
    assert runner.saved_events[-1][0] == 12


def test_forced_checkpoint_does_not_force_an_image():
    runner = runner_fixture(sample_every=250, skip_initial=True)
    runner.engine.logical_update = runner.engine.update_attempt = 7
    with patch("extensions.gen2_trainer.evaluation.bundle_identity", return_value=("state-hash", {})):
        runner.events(force_save=True)
    runner.evaluation.sample.assert_not_called()
    assert [update for update, _, _ in runner.saved_events] == [7]
