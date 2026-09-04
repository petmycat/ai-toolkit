import copy
import unittest

import yaml

from extensions_built_in.gen2_trainer.config import validate_gen2_config


class Gen2ConfigSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("config/gen2_trainer_ideogram4_smoke.yaml", encoding="utf-8") as handle:
            cls.process = yaml.safe_load(handle)["config"]["process"][0]

    def test_smoke_and_actual_schema_validate(self):
        validate_gen2_config(self.process)
        with open("config/gen2_trainer_ideogram4_actual.yaml", encoding="utf-8") as handle:
            validate_gen2_config(yaml.safe_load(handle)["config"]["process"][0])

    def test_frozen_top_level_contract(self):
        self.assertEqual(self.process["train"]["phase_b"]["batch_size"], 2)
        self.assertTrue(self.process["train"]["phase_b"]["enabled"])
        self.assertEqual(self.process["helpers_per_step"], 3)
        self.assertEqual(self.process["train"]["phase_a"]["probes"]["timestep_count"], 5)
        self.assertEqual(set(self.process["train"]["phase_a"]["probes"]), {"enabled", "probe_seed", "timestep_count", "regions"})
        self.assertTrue(self.process["dataset_split"]["enabled"])
        self.assertTrue(self.process["prototype"]["enabled"])
        self.assertTrue(self.process["controller"]["enabled"])
        self.assertTrue(self.process["qwen"]["per_layer_adapter"]["enabled"])
        self.assertIn("probes", self.process["train"]["phase_a"])

    def test_deprecated_fields_are_rejected_anywhere(self):
        for key in (
            "calibration", "effect_geometry", "mixture", "mixture_iterations", "mixture_weights", "positive_cone", "cone",
            "cone_iterations", "teacher", "teacher_weight", "content_preserve", "content_preserve_weight",
            "cross_content", "cross_content_weight", "trust_region", "trust_region_weight", "token_diversity",
            "token-diversity", "independent_tail",
        ):
            candidate = copy.deepcopy(self.process)
            candidate["train"]["phase_a"][key] = 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "deprecated"):
                validate_gen2_config(candidate)

    def test_new_sections_reject_invalid_values(self):
        for section, key, value in (
            ("dataset_split", "seed", -1),
            ("prototype", "ema_decay", 1.0),
            ("controller", "max_fraction", 2.0),
            ("qwen", "per_layer_adapter", {"enabled": True, "lr": 0.0}),
        ):
            candidate = copy.deepcopy(self.process)
            if section == "qwen":
                candidate[section][key] = value
            else:
                candidate[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_gen2_config(candidate)

    def test_qwen_per_layer_adapter_contract(self):
        adapter = self.process["qwen"]["per_layer_adapter"]
        self.assertEqual((adapter["rank"], adapter["alpha"]), (4, 4))
        self.assertNotIn("trigger_local_adapter", self.process["activator"])
        candidate = copy.deepcopy(self.process)
        candidate["activator"]["trigger_local_adapter"] = {"enabled": True}
        with self.assertRaisesRegex(ValueError, "deprecated"):
            validate_gen2_config(candidate)
        candidate = copy.deepcopy(self.process)
        candidate["qwen"]["per_layer_adapter"]["enabled"] = False
        validate_gen2_config(candidate)


if __name__ == "__main__":
    unittest.main()
