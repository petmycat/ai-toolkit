import json
import os
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

from extensions_built_in.ideogram4_v3_activator.config import load_config
from extensions_built_in.ideogram4_v3_activator.pipeline import Ideogram4V3ActivatorPipeline


class _Job:
    def __init__(self, process_config):
        self.name = "v3_parent"
        self.meta = OrderedDict({"purpose": "unit-test"})
        self.raw_config = OrderedDict({
            "job": "extension",
            "config": OrderedDict({"name": self.name, "process": [process_config]}),
            "meta": self.meta,
        })


def _block(v3_weights, split_manifest):
    return {
        "enabled": True,
        "v3_weights": v3_weights,
        "literal_initialization": {"target_vectors": 4},
        "helper_schedule": {"mode": "fixed", "helpers": ["illustration"], "weights": [1.0]},
        "dataset_schedule": {
            "enabled": True,
            "sources": [{"name": "structured", "use_main_dataset": True, "caption_ext": ".json", "format": "json"}],
            "schedule": {"keyframes": [{"step": 0, "structured": 1.0}, {"step": 80, "structured": 1.0}]},
        },
        "fixed_validation": {"seed": 42, "fixed_timesteps": [250, 500], "data_split_manifest": split_manifest},
        "semantic_activator": {"steps": 80, "learning_rates": {"embedding": 5e-4, "te_adapter": 1e-4}},
        "te_calibration": {"steps": 80, "learning_rates": {"embedding": 0.0, "te_adapter": 5e-5}},
        "terminal_manual_ablation": {"mode": "manual", "automatic": False, "residual_ablation_implemented": False},
    }


class Ideogram4V3ActivatorConfigTest(unittest.TestCase):
    def test_requires_exact_fixed_eighty_step_semantic_stage(self):
        raw = _block("v3.safetensors", "split.json")
        raw["semantic_activator"]["steps"] = 79
        with self.assertRaisesRegex(ValueError, "exactly 80"):
            load_config(raw)

    def test_te_calibration_supports_one_hundred_steps_with_final_checkpoints(self):
        raw = _block("v3.safetensors", "split.json")
        raw["te_calibration"].update({
            "steps": 100,
            "save_steps": [20, 40, 60, 80, 100],
            "validation_steps": [0, 20, 40, 60, 80, 100],
        })
        config = load_config(raw)
        self.assertEqual(config.te_calibration.steps, 100)

    def test_explicit_smoke_test_allows_five_steps_for_both_stages(self):
        raw = _block("v3.safetensors", "split.json")
        raw["smoke_test"] = True
        raw["semantic_activator"].update({
            "steps": 5,
            "save_steps": [5],
            "validation_steps": [5],
        })
        raw["te_calibration"].update({
            "steps": 5,
            "save_steps": [5],
            "validation_steps": [0, 5],
        })
        config = load_config(raw)
        self.assertTrue(config.smoke_test)
        self.assertEqual(config.semantic_activator.steps, 5)
        self.assertEqual(config.te_calibration.steps, 5)

    def test_rejects_non_structured_a1_schedule(self):
        raw = _block("v3.safetensors", "split.json")
        raw["dataset_schedule"]["sources"].append({"name": "natural"})
        with self.assertRaisesRegex(ValueError, "structured/json"):
            load_config(raw)

class Ideogram4V3ActivatorPipelineTest(unittest.TestCase):
    def _process(self, temp_dir):
        v3 = os.path.join(temp_dir, "v3.safetensors")
        split = os.path.join(temp_dir, "split.json")
        Path(v3).write_bytes(b"v3-weights")
        Path(split).write_text("{}", encoding="utf-8")
        process_config = OrderedDict({
            "type": "ideogram4_v3_activator",
            "name": "v3_run",
            "training_folder": temp_dir,
            "trigger_word": "<atomic-v3>",
            "model": {"arch": "ideogram4", "name_or_path": "test/model"},
            "datasets": [{"folder_path": "dataset"}],
            "train": {"dtype": "bf16"},
            "save": {},
            "three_phase_trigger_training": {
                "trigger": {"placeholder": "[trigger]", "literal": "<atomic-v3>"},
                "text_activator": {
                    "architecture_mode": "module_lora",
                    "embedding": {"enabled": True, "tokens": 4, "init_mode": "literal_initialization", "init_words": ""},
                    "te_adapter": {"enabled": True, "mode": "module_lora", "rank": 4, "alpha": 4, "target_modules": ["down_proj"]},
                    "tap_adapters": {"enabled": False},
                },
            },
            "ideogram4_v3_activator": _block(v3, split),
        })
        process = Ideogram4V3ActivatorPipeline()
        process.name = process_config["name"]
        process.raw_process_config = process_config
        process.job = SimpleNamespace(meta=OrderedDict({"purpose": "unit-test"}))
        process.v3_config = load_config(process_config["ideogram4_v3_activator"])
        process.get_conf = lambda key, default=None, required=False: (
            process_config[key] if key in process_config else default
        )
        process.initialize_pipeline()
        return process

    def test_child_configs_are_isolated_a1_then_a2(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            process = self._process(temp_dir)
            a1 = process.stages["semantic_activator"].build_child_process_config()
            a2 = process.stages["te_calibration"].build_child_process_config()
            self.assertEqual(a1["type"], "sd_trainer")
            self.assertIsNone(a1["network"])
            self.assertEqual(a1["train"]["steps"], 80)
            self.assertEqual(
                a1["three_phase_trigger_training"]["objective_mode"],
                "ideogram4_v3_activator",
            )
            self.assertEqual(
                a1["three_phase_trigger_training"]["text_activator"]["embedding"]["tokens"],
                4,
            )
            self.assertFalse(a1["three_phase_trigger_training"]["text_activator"]["tap_adapters"]["enabled"])
            self.assertTrue(a1["three_phase_trigger_training"]["phase_a1"]["train"]["embedding"])
            self.assertEqual(
                set(a1["three_phase_trigger_training"]["phase_runtime"]["losses"]),
                {"semantic_helper_consistency"},
            )
            self.assertFalse(a2["three_phase_trigger_training"]["phase_a2"]["train"]["embedding"])
            self.assertTrue(a2["three_phase_trigger_training"]["phase_a2"]["train"]["te_adapter"])
            self.assertEqual(a2["train"]["steps"], 80)
            self.assertEqual(a2["three_phase_trigger_training"]["phase_a2"]["steps"], 80)
            self.assertEqual(
                set(a2["three_phase_trigger_training"]["phase_runtime"]["losses"]),
                {"direct_v3_reconstruction"},
            )
            self.assertEqual(a2["network"]["pretrained_lora_path"], process.v3_weights_path)
            self.assertEqual(a2["save"]["save_steps"], [19, 39, 59, 79])
            self.assertEqual(
                a2["three_phase_trigger_training"]["validation"]["steps"],
                [0, 20, 40, 60, 80],
            )
            process.v3_config = load_config({
                **_block(process.v3_weights_path, process.v3_config.fixed_validation["data_split_manifest"]),
                "te_calibration": {
                    "steps": 100,
                    "learning_rates": {"embedding": 0.0, "te_adapter": 5e-5},
                    "save_steps": [20, 40, 60, 80, 100],
                    "validation_steps": [0, 20, 40, 60, 80, 100],
                },
            })
            process.initialize_pipeline()
            a2_100 = process.stages["te_calibration"].build_child_process_config()
            self.assertEqual(a2_100["train"]["steps"], 100)
            self.assertEqual(a2_100["three_phase_trigger_training"]["phase_a2"]["steps"], 100)
            self.assertEqual(a2_100["save"]["save_steps"], [19, 39, 59, 79, 99])
            sources = a2["three_phase_trigger_training"]["phase_runtime"]["sources"]
            self.assertNotEqual(sources["embedding"], sources["te_adapter"])
            a1_paths = process.a1_artifact_paths()
            self.assertTrue(a1_paths["embedding"].endswith(
                os.path.join("phase_a1", "final", "trigger_embedding.safetensors")
            ))
            self.assertTrue(a1_paths["literal_initialization"].endswith(
                os.path.join("phase_a1", "literal_initialization.safetensors")
            ))

    def test_stage_snapshots_contracts_and_best_artifacts_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            process = self._process(temp_dir)
            a1_snapshot = process.stages["semantic_activator"].write_snapshot()
            a2_snapshot = process.stages["te_calibration"].write_snapshot()
            self.assertNotEqual(a1_snapshot, a2_snapshot)
            self.assertTrue(os.path.isfile(a1_snapshot))
            process.write_contract("semantic_activator", "running")
            contract = json.loads(Path(process.stages["semantic_activator"].contract_path).read_text(encoding="utf-8"))
            self.assertEqual(contract["automatic_order"], ["semantic_activator", "te_calibration", "STOP"])
            self.assertEqual(contract["training"]["taps"], "disabled")

            artifacts = process._artifact_records("te_calibration")
            os.makedirs(os.path.dirname(artifacts["embedding"]), exist_ok=True)
            Path(artifacts["embedding"]).write_bytes(b"final-embedding")
            Path(artifacts["te_adapter"]).write_bytes(b"final-te")
            os.makedirs(os.path.dirname(artifacts["metrics"]), exist_ok=True)
            Path(artifacts["metrics"]).write_text("{}\n", encoding="utf-8")
            Path(artifacts["heldout_validation"]).write_text(
                json.dumps({"step": 0, "heldout_loss": 0.01}) + "\n"
                + json.dumps({"step": 80, "heldout_loss": 0.25}) + "\n",
                encoding="utf-8",
            )
            process._select_best_a2()
            best_manifest = json.loads(Path(artifacts["best_manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(best_manifest["step"], 80)
            self.assertEqual(best_manifest["heldout_loss"], 0.25)
            self.assertEqual(Path(artifacts["best_embedding"]).read_bytes(), b"final-embedding")
            self.assertNotEqual(artifacts["best_embedding"], process.a1_artifact_paths()["embedding"])


if __name__ == "__main__":
    unittest.main()
