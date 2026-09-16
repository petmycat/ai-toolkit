import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from extensions.gen2_trainer.__main__ import main, parser, read_config
from extensions.gen2_trainer.acceptance import compare_trees, tree_hash
from extensions.gen2_trainer.text_preflight import load_tokenizer as shared_tokenizer_loader


class CommandUtilityTests(unittest.TestCase):
    def test_help_and_parser_do_not_import_models(self):
        script = ("import sys; from extensions.gen2_trainer.__main__ import parser; "
                  "parser().parse_args(['validate','example.yaml']); "
                  "parser().parse_args(['check-captions','example.yaml','--local-files-only']); "
                  "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules")
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = parser().parse_args(["infer", "package", "--prompt", "A  chair", "--output", "image.png"])
        self.assertIsNone(args.mode)
        self.assertIsNone(args.strength)

    def test_caption_check_reports_pass_and_overflow_without_loading_models(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from extensions.gen2_trainer.config import resolve_process_config
        raw = {"model": {"model_kwargs": {"text_encoder_path": "local/tokenizer"}}}
        resolved = resolve_process_config(raw)
        manifest = [{"sample_id": "source1", "q": "A [trigger] image", "path": "image.png"}]
        tokenizer = object()
        for passed in (True, False):
            with self.subTest(passed=passed), tempfile.TemporaryDirectory() as temporary:
                report = {"passed": passed, "rows": [{"source": "image.png", "total_length": 12 if passed else 2050}],
                    "failures": [] if passed else [{"source": "image.png", "reason": "token_overflow"}],
                    "num_tokens": 4, "limit": 2048, "original_token_budget": 2044}
                scan = Mock(return_value=report)
                load_tokenizer = Mock(return_value=tokenizer)
                preflight = Mock(return_value=manifest)
                modules = {"transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer)),
                    "extensions.gen2_trainer.data": SimpleNamespace(preflight_datasets=preflight),
                    "extensions.gen2_trainer.text_preflight": SimpleNamespace(build_token_report=scan, load_tokenizer=shared_tokenizer_loader)}
                output = Path(temporary)/"nested"/"tokens.json"
                stdout = StringIO()
                with patch.dict(sys.modules, modules), patch.dict(os.environ, {"HF_TOKEN": "test-token"}), \
                        patch("extensions.gen2_trainer.__main__.read_config", return_value=(raw, "caption-check")) as read, \
                        redirect_stdout(stdout):
                    status = main(["check-captions", "source.yaml", "--process-index", "2", "--local-files-only", "--output", str(output)])
                self.assertEqual(status, 0 if passed else 1)
                read.assert_called_once_with("source.yaml", 2)
                preflight.assert_called_once_with(resolved)
                load_tokenizer.assert_called_once_with("local/tokenizer", token="test-token", local_files_only=True)
                scan.assert_called_once_with(resolved, manifest, tokenizer)
                emitted = json.loads(stdout.getvalue())
                self.assertEqual(json.loads(output.read_text(encoding="utf-8")), emitted)
                self.assertEqual(emitted["failures"], report["failures"])
                self.assertFalse(emitted["models_loaded"])
                self.assertNotIn("test-token", stdout.getvalue())

    def test_caption_check_rejects_protected_output_before_loading_tokenizer(self):
        protected_root = Path(__file__).resolve().parents[3]/"gen2"
        scan, preflight, load_tokenizer = Mock(), Mock(), Mock()
        modules = {"transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer)),
            "extensions.gen2_trainer.data": SimpleNamespace(preflight_datasets=preflight),
            "extensions.gen2_trainer.text_preflight": SimpleNamespace(build_token_report=scan, load_tokenizer=shared_tokenizer_loader)}
        with patch.dict(sys.modules, modules), \
                patch("extensions.gen2_trainer.__main__.read_config", return_value=({}, "check")), \
                self.assertRaises(PermissionError):
            main(["check-captions", "source.yaml", "--output", str(protected_root/"tokens.json")])
        preflight.assert_not_called()
        load_tokenizer.assert_not_called()
        scan.assert_not_called()

    def test_caption_check_default_tokenizer_and_clear_load_error(self):
        preflight, scan = Mock(return_value=[]), Mock()
        load_tokenizer = Mock(side_effect=OSError("cached tokenizer files are unavailable"))
        modules = {"transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer)),
            "extensions.gen2_trainer.data": SimpleNamespace(preflight_datasets=preflight),
            "extensions.gen2_trainer.text_preflight": SimpleNamespace(build_token_report=scan, load_tokenizer=shared_tokenizer_loader)}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=True), \
                patch("extensions.gen2_trainer.__main__.read_config", return_value=({}, "check")), \
                self.assertRaisesRegex(RuntimeError, "Could not load tokenizer.*local_files_only=True.*cached tokenizer"):
            main(["check-captions", "source.yaml", "--local-files-only"])
        load_tokenizer.assert_called_once_with("Qwen/Qwen3-VL-8B-Instruct", token=None, local_files_only=True)
        scan.assert_not_called()

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
