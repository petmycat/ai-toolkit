"""CPU lifecycle integration with real engine, recording and checkpoint files.

Native model construction and image/data decoding are replaced by bounded
fixtures. These checks exercise runner continuation/finalization, not GPU model
compatibility or native loader replay (covered by the explicit VM harness).
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from extensions.gen2_trainer.diagnostics import capture_rng_state, restore_rng_state
from extensions.gen2_trainer.recording import Recorder, iter_records
from extensions.gen2_trainer.tests.test_v2_engine import BackendFixture, batch
from extensions.gen2_trainer.v2.checkpoint import V2CheckpointManager, load_manifest
from extensions.gen2_trainer.v2.config import resolve_process_config, SPEC_SHA256
from extensions.gen2_trainer.v2.engine import V2Engine
from extensions.gen2_trainer.v2.process import V2Runner, specification_path, training_contract


class StreamFixture:
    def __init__(self):
        self.cursor = 0

    def next(self):
        self.cursor += 1
        return batch((float(torch.rand(()))*.2,))

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = state["cursor"]


class EvaluationFixture:
    def __init__(self):
        self.calls = []

    def sample(self, update):
        self.calls.append(update)

    def state_dict(self):
        return {"calls": list(self.calls)}

    def load_state_dict(self, state):
        self.calls = list(state["calls"])


class V2RunnerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = resolve_process_config({
            "training_folder": str(self.root), "trigger_word": "<test>",
            "model": {"name_or_path": "fixture"},
            "datasets": [{"folder_path": "fixture"}],
            "train": {"steps": 4, "gradient_accumulation_steps": 2},
            "sample": {"prompts": ["[trigger] tiger"], "sample_every": 2},
            "save": {"save_every": 3},
            "gen2": {"schema_version": "2.0.0",
                     "inference": {"unconditional_model_path": "fixture"}},
        })
        self.rng = capture_rng_state()
        self.cuda = patch("torch.cuda.is_available", return_value=False)
        self.prepare = patch("extensions.gen2_trainer.v2.process.prepare_batch",
                             side_effect=lambda prepared, *args: prepared)
        self.cuda.start()
        self.prepare.start()

    def tearDown(self):
        self.prepare.stop()
        self.cuda.stop()
        restore_rng_state(self.rng)
        self.temp.cleanup()

    def runner(self, name, resume=None):
        config = deepcopy(self.config)
        config["gen2"]["checkpoint"]["resume_from"] = str(resume) if resume else None
        runner = V2Runner(config, name)

        def load():
            torch.manual_seed(713)
            runner.root.mkdir(parents=True, exist_ok=True)
            runner.recorder = Recorder(runner.root, name, config=config["gen2"]["recording"])
            runner.backend = BackendFixture()
            runner.backend.model = SimpleNamespace()
            runner.engine = V2Engine(config, runner.backend)
            runner.stream = StreamFixture()
            runner.evaluation = EvaluationFixture()
            runner.manifest_by_path = {}
            runner.logging_config = SimpleNamespace(log_every=1)
            runner.logger = SimpleNamespace(log=lambda *args: None, commit=lambda **kw: None,
                                            finish=lambda: None)
            runner.logger_started = True
            runner.checkpoints = V2CheckpointManager(runner.root / "checkpoints",
                specification_path(config), SPEC_SHA256)
            runner.metadata = {"run_id": name, "training_contract": training_contract(config)}
            if resume:
                runtime = runner.checkpoints.load(resume, runner.backend, runner.engine,
                    {"training_contract": training_contract(config)})
                runner.stream.load_state_dict(runtime["data"])
                runner.evaluation.load_state_dict(runtime["evaluation"])
                runner.recorder.resume(runtime["recorder"])
                restore_rng_state(runtime["rng"])
                runner.last_checkpoint = Path(resume)

        def verify():
            runner.backend.assert_frozen()
            runner.recorder.event("frozen_weights_verified")

        runner._load = load
        runner._verify_frozen = verify
        return runner

    def test_runner_pause_resume_matches_continuous_tokens_optimizer_and_data(self):
        continuous = self.runner("continuous")
        final = continuous.run()
        expected = torch.load(continuous.checkpoints.resume_path / "training_state.pt",
                              weights_only=True, map_location="cpu")
        paused = self.runner("split")
        paused.run(stop_after=2)
        self.assertEqual(paused.engine.logical_update, 2)
        resumed = self.runner("split", paused.checkpoints.resume_path)
        result = resumed.run()
        actual = torch.load(resumed.checkpoints.resume_path / "training_state.pt",
                            weights_only=True, map_location="cpu")
        self.assertEqual(load_manifest(final)["kind"], "inference")
        self.assertEqual(load_manifest(result)["logical_update"], 4)
        torch.testing.assert_close(resumed.backend.tokens.E, continuous.backend.tokens.E, rtol=0, atol=0)
        self.assertEqual(expected["runtime"]["data"], actual["runtime"]["data"])
        self.assertEqual(expected["runtime"]["evaluation"], actual["runtime"]["evaluation"])
        for key, value in expected["engine"]["optimizer"]["state"][0].items():
            torch.testing.assert_close(value, actual["engine"]["optimizer"]["state"][0][key], rtol=0, atol=0)
        self.assertEqual(resumed.evaluation.calls, [0, 2, 4])

    def test_completed_resume_recovers_failed_final_inference_export(self):
        interrupted = self.runner("export_failure")
        with patch.object(V2CheckpointManager, "export", side_effect=OSError("inference export failed")):
            with self.assertRaisesRegex(OSError, "inference export failed"):
                interrupted.run()
        resume = interrupted.checkpoints.resume_path
        self.assertEqual(load_manifest(resume)["logical_update"], 4)
        self.assertFalse(interrupted.checkpoints.inference_path.exists())
        final_tokens = interrupted.backend.tokens.E.detach().clone()
        resumed = self.runner("export_failure", resume)
        result = resumed.run()
        self.assertEqual(result, resumed.checkpoints.inference_path)
        self.assertEqual(load_manifest(result)["kind"], "inference")
        self.assertEqual(resumed.engine.logical_update, 4)
        self.assertEqual(resumed.stream.cursor, 8)
        self.assertEqual(resumed.evaluation.calls, [0, 2, 4])
        torch.testing.assert_close(resumed.backend.tokens.E, final_tokens, rtol=0, atol=0)
        events = list(iter_records(resumed.root, "events"))
        self.assertTrue(any(row.get("event") == "frozen_weights_verified" for row in events))


if __name__ == "__main__":
    unittest.main()
