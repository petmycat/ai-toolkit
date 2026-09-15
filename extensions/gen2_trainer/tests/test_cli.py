import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from extensions.gen2_trainer.__main__ import parser, read_config
from extensions.gen2_trainer.acceptance import compare_trees, tree_hash


class CommandUtilityTests(unittest.TestCase):
    def test_help_and_parser_do_not_import_models(self):
        script = ("import sys; from extensions.gen2_trainer.__main__ import parser; "
                  "parser().parse_args(['validate','example.yaml']); "
                  "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules")
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = parser().parse_args(["infer", "package", "--prompt", "A  chair", "--output", "image.png"])
        self.assertIsNone(args.mode)
        self.assertIsNone(args.strength)

    @unittest.skipUnless(importlib.util.find_spec("oyaml"), "native toolkit.config requires optional local oyaml; VM dependencies provide it")
    def test_native_config_environment_name_exponents(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"config.yaml"
            path.write_text("job: extension\nconfig:\n  name: test\n  process:\n    - type: gen2_trainer\n      training_folder: '${GEN2_TEST_FOLDER}/[name]'\n      train:\n        lr: 1e-4\n", encoding="utf-8")
            with patch.dict(os.environ, {"GEN2_TEST_FOLDER": "local_test_output"}):
                config, name = read_config(path)
            self.assertEqual(name, "test")
            self.assertEqual(config["training_folder"], "local_test_output/test")
            self.assertEqual(config["train"]["lr"], .0001)

    def test_recursive_continuation_comparison_and_explicit_tolerance(self):
        expected = {"state": {0: {"moment": torch.tensor([.1, .2])}}, "step": 3,
                    "rng": torch.tensor([1, 2, 3], dtype=torch.uint8), "scheduler": None}
        actual = {"state": {0: {"moment": torch.tensor([.1, .20001])}}, "step": 3,
                    "rng": torch.tensor([1, 2, 3], dtype=torch.uint8), "scheduler": None}
        self.assertFalse(compare_trees(expected, actual)["equal"])
        self.assertTrue(compare_trees(expected, actual, atol=.001)["equal"])
        actual["rng"][0] = 2
        self.assertFalse(compare_trees(expected, actual, atol=100)["equal"])
        actual["rng"][0] = 1
        actual["step"] = 4
        self.assertFalse(compare_trees(expected, actual, atol=100)["equal"])

    def test_full_tree_hash_includes_optimizer_tensor_and_structure(self):
        original = {"optimizer": {"states": [torch.tensor([0., 1.])]}, "step": 1}
        other = {"optimizer": {"states": [torch.tensor([0., 1.])]}, "step": 1}
        self.assertEqual(tree_hash(original), tree_hash(other))
        other["optimizer"]["states"][0][1] += 1
        self.assertNotEqual(tree_hash(original), tree_hash(other))

    def test_incomplete_package_rejected_before_native_model_import(self):
        from extensions.gen2_trainer.package import load_package
        from extensions.gen2_trainer.checkpointing import CheckpointError
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(CheckpointError):
            load_package(temporary)


if __name__ == "__main__":
    unittest.main()
