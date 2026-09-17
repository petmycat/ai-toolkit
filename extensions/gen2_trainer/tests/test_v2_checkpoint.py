"""Token-only package integrity, publication, strict resume and predictions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from extensions.gen2_trainer.v2.checkpoint import V2CheckpointManager, V2CheckpointError, load_manifest
from extensions.gen2_trainer.v2.engine import V2Engine
from extensions.gen2_trainer.tests.test_v2_engine import BackendFixture, batch, configuration


class V2CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.spec = self.root / "spec.md"
        self.spec.write_bytes(b"Immutable v2 test specification.\n")
        self.digest = hashlib.sha256(self.spec.read_bytes()).hexdigest()
        self.manager = V2CheckpointManager(self.root/"packages", self.spec, self.digest)
        self.backend = BackendFixture()
        self.engine = V2Engine(configuration(), self.backend)
        self.metadata = {"training_contract": {"placement": "in_place", "repeated": "shared"},
                         "model_identities": {"base": "frozen-hash"}, "tokenizer_identity": "tokenizer-hash",
                         "dataset_identity": "dataset-hash", "run_id": "one"}
        self.runtime = {"rng": {"torch_cpu": torch.get_rng_state(), "python": (3, (1, 2), None)},
                        "data": {"position": 3}, "evaluation": {"references": ["base.png"]}}

    def tearDown(self):
        self.temp.cleanup()

    def test_rolling_save_and_final_export_have_only_active_components(self):
        self.engine.step([batch()])
        path = self.manager.save(self.backend, self.engine, self.metadata, self.runtime)
        self.assertEqual(path.name, "resume_latest")
        self.engine.step([batch()])
        final = self.manager.save(self.backend, self.engine, self.metadata, self.runtime, final=True)
        self.assertEqual(final.name, "inference_final")
        self.assertEqual({p.name for p in self.manager.root.iterdir()}, {"resume_latest", "inference_final"})
        self.assertEqual(load_manifest(path)["logical_update"], 2)
        inference = load_manifest(final)
        self.assertEqual(inference["kind"], "inference")
        self.assertEqual(set(inference["files"]), {"tokens.safetensors", "specification.md"})
        self.assertLess((final/"tokens.safetensors").stat().st_size, 4096)

    def test_inference_reload_reproduces_conditioned_predictions_exactly(self):
        self.engine.step([batch()])
        saved = self.manager.export(self.backend, self.metadata, logical_update=1)
        prepared = batch()
        expected = self.backend.predict(prepared["zt"], prepared["tau"], self.backend.encode(prepared["qs"]))
        fresh = BackendFixture()
        self.manager.load_inference(saved, fresh, {"model_identities": self.metadata["model_identities"]})
        actual = fresh.predict(prepared["zt"], prepared["tau"], fresh.encode(prepared["qs"]))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(fresh.tokens.initial, self.backend.tokens.initial, rtol=0, atol=0)

    def test_resume_returns_runtime_and_exact_next_update(self):
        self.engine.step([batch()])
        saved = self.manager.save(self.backend, self.engine, self.metadata, self.runtime)
        fresh = BackendFixture()
        engine = V2Engine(configuration(), fresh)
        runtime = self.manager.load(saved, fresh, engine, {"training_contract": self.metadata["training_contract"]})
        self.assertEqual(runtime["data"], self.runtime["data"])
        self.assertEqual(runtime["evaluation"], self.runtime["evaluation"])
        torch.testing.assert_close(runtime["rng"]["torch_cpu"], self.runtime["rng"]["torch_cpu"])
        expected, actual = self.engine.step([batch()]), engine.step([batch()])
        self.assertEqual(actual["loss"], expected["loss"])
        torch.testing.assert_close(fresh.tokens.E, self.backend.tokens.E, rtol=0, atol=0)

    def test_metadata_and_integrity_fail_before_loading_parameters(self):
        self.engine.step([batch()])
        saved = self.manager.save(self.backend, self.engine, self.metadata, self.runtime)
        fresh = BackendFixture()
        old = fresh.tokens.E.detach().clone()
        engine = V2Engine(configuration(), fresh)
        with self.assertRaisesRegex(V2CheckpointError, "identity mismatch"):
            self.manager.load(saved, fresh, engine, {"model_identities": {"base": "wrong"}})
        torch.testing.assert_close(fresh.tokens.E, old, rtol=0, atol=0)
        with (saved/"tokens.safetensors").open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(V2CheckpointError, "checksum"):
            self.manager.load(saved, fresh, engine)
        torch.testing.assert_close(fresh.tokens.E, old, rtol=0, atol=0)

    def test_failed_staging_preserves_previous_rolling_checkpoint(self):
        self.engine.step([batch()])
        saved = self.manager.save(self.backend, self.engine, self.metadata, self.runtime)
        original_hash = load_manifest(saved)["package_hash"]
        self.engine.step([batch()])
        with patch("safetensors.torch.save_file", side_effect=OSError("simulated disk failure")):
            with self.assertRaises(OSError):
                self.manager.save(self.backend, self.engine, self.metadata, self.runtime)
        self.assertEqual(load_manifest(saved)["package_hash"], original_hash)
        self.assertEqual({p.name for p in self.manager.root.iterdir()}, {"resume_latest"})

    def test_inference_is_not_a_resume_checkpoint(self):
        saved = self.manager.export(self.backend, self.metadata, logical_update=0)
        with self.assertRaisesRegex(V2CheckpointError, "no training resume"):
            self.manager.load(saved, self.backend, self.engine)

    def test_failed_publication_restores_previous_directory(self):
        import os
        self.engine.step([batch()])
        saved = self.manager.save(self.backend, self.engine, self.metadata, self.runtime)
        original_hash = load_manifest(saved)["package_hash"]
        self.engine.step([batch()])
        replace = os.replace
        def fail_new_directory(source, destination):
            if ".pending-" in Path(source).name and Path(destination) == saved:
                raise OSError("simulated directory publication failure")
            return replace(source, destination)
        with patch("extensions.gen2_trainer.v2.checkpoint.os.replace", side_effect=fail_new_directory):
            with self.assertRaises(OSError):
                self.manager.save(self.backend, self.engine, self.metadata, self.runtime)
        self.assertEqual(load_manifest(saved)["package_hash"], original_hash)
        self.assertEqual({p.name for p in self.manager.root.iterdir()}, {"resume_latest"})

    def test_specification_and_missing_runtime_are_rejected(self):
        with self.assertRaisesRegex(V2CheckpointError, "SHA256"):
            V2CheckpointManager(self.root/"other", self.spec, "wrong")
        with self.assertRaisesRegex(V2CheckpointError, "rng, data and evaluation"):
            self.manager.save(self.backend, self.engine, self.metadata, {"rng": {}})

    def test_protected_reference_directory_cannot_be_output(self):
        protected = Path(__file__).resolve().parents[3] / "gen2" / "never_write_v2"
        manager = V2CheckpointManager(protected, self.spec, self.digest)
        with self.assertRaisesRegex(V2CheckpointError, "read-only"):
            manager.export(self.backend, self.metadata, logical_update=0)
        self.assertFalse(protected.exists())


if __name__ == "__main__":
    unittest.main()
