"""Configuration tests run without importing torch or model loaders."""
from copy import deepcopy
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from extensions.gen2_trainer.config import (
    ConfigError, ROLES, STANDARD_OPTIMIZERS, resolve_process_config,
    scheduler_kwargs, schema_leaves,
)


class ConfigurationTests(unittest.TestCase):
    def test_default_schedule_and_inheritance(self):
        cfg = resolve_process_config({})
        self.assertEqual(cfg["_gen2_resolved"]["family_horizons"],
                         dict(diffusion=2130, embedding=420, text_adapter=420, gates=450))
        self.assertEqual(cfg["gen2"]["optimizers"]["diffusion"]["optimizer"], "adamw8bit")
        self.assertEqual(cfg["gen2"]["optimizers"]["embedding"]["lr"], 1e-3)

    def test_config_only_has_no_model_or_torch_imports(self):
        result = subprocess.run([sys.executable, "-B", "-c",
            "import sys; from extensions.gen2_trainer.config import resolve_process_config; "
            "resolve_process_config({}); assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unconditional_visual_modes_require_cfg_and_are_opt_in(self):
        from extensions.gen2_trainer.__main__ import parser
        modes = ["base_with_tokens", "base_with_conditioning", "encoder_adapter_off",
                 "full", "full_uncond_half", "full_uncond_full", "base"]
        cfg = resolve_process_config({"sample": {"guidance_scale": 3.},
            "gen2": {"evaluation": {"preview_modes": modes, "milestone_modes": modes}}})
        self.assertEqual(cfg["gen2"]["evaluation"]["milestone_modes"], modes)
        for mode in modes:
            self.assertEqual(parser().parse_args(["infer", "bundle", "--prompt", "scene",
                "--output", "image.png", "--mode", mode]).mode, mode)
        defaults = resolve_process_config({})["gen2"]["evaluation"]
        self.assertEqual(len(defaults["milestone_modes"]), 9)
        self.assertFalse({"full_uncond_half", "full_uncond_full"} & set(defaults["milestone_modes"]))
        for key in ("preview_modes", "milestone_modes"):
            for scale in (0., 1.):
                with self.subTest(key=key, scale=scale), self.assertRaisesRegex(ConfigError, "skips the unconditional"):
                    resolve_process_config({"sample": {"guidance_scale": scale},
                        "gen2": {"evaluation": {key: modes}}})
        # Existing conditional-only native sampling remains valid.
        resolve_process_config({"sample": {"guidance_scale": 1.}})

    def test_real_native_discovery_loads_only_lightweight_registration(self):
        script = """
import sys
from unittest.mock import patch
import toolkit.extension as native
original = native.pkgutil.iter_modules
def select_gen2(paths):
    # Exercise the real filesystem discovery/import implementation while
    # excluding unrelated plugins that require unavailable optional packages.
    return (item for item in original(paths) if item.name == 'gen2_trainer')
with patch.object(native.pkgutil, 'iter_modules', select_gen2):
    found = native.get_all_extensions()
assert len(found) == 1 and found[0].uid == 'gen2_trainer'
assert issubclass(found[0], native.Extension)
assert 'extensions.gen2_trainer.process' not in sys.modules
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_shallow_copied_role_kwargs_and_null_scalar(self):
        raw = {"train": {"optimizer": "adamw", "lr": .02,
            "optimizer_params": {"betas": [.8, .9], "weight_decay": .01}},
            "gen2": {"optimizers": {"embedding": {"lr": None, "optimizer_params": {"weight_decay": 0.}}}}}
        before = deepcopy(raw)
        cfg = resolve_process_config(raw)
        self.assertEqual(raw, before)
        embedding = cfg["gen2"]["optimizers"]["embedding"]
        self.assertEqual(embedding["lr"], .02)
        self.assertEqual(embedding["optimizer_params"], {"betas": [.8, .9], "weight_decay": 0.})
        embedding["optimizer_params"]["betas"][0] = .1
        self.assertEqual(cfg["gen2"]["optimizers"]["diffusion"]["optimizer_params"]["betas"][0], .8)

    def test_every_nested_namespace_rejects_unknown_keys(self):
        cases = [{"phases": {"warmup_udpates": 1}}, {"optimizers": {"embedding": {"lrr": 1}}},
                 {"diagnostics": {"probes": {"tau": [.5]}}}, {"evaluation": {"modse": []}}]
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaisesRegex(ConfigError, "unknown keys"):
                resolve_process_config({"gen2": raw})

    def test_schema_leaf_coverage_and_validation(self):
        leaves = schema_leaves()
        self.assertEqual(len(leaves), 90)
        for path in leaves:
            raw = {}
            node = raw
            keys = path.split(".")
            for part in keys[:-1]: node = node.setdefault(part, {})
            node[keys[-1]] = object()
            with self.subTest(path=path), self.assertRaises(ConfigError): resolve_process_config(raw)

    def test_zero_stage_and_empty_scheduler_horizon(self):
        cfg = resolve_process_config({"train": {"steps": 2, "lr_scheduler": "cosine_with_restarts"},
            "gen2": {"phases": {"warmup_updates": 2, "refinement_updates": 0, "calibration_updates": 0}}})
        self.assertEqual(cfg["_gen2_resolved"]["family_horizons"]["embedding"], 0)
        self.assertEqual(cfg["_gen2_resolved"]["scheduler_kwargs"]["embedding"], {})
        self.assertEqual(cfg["_gen2_resolved"]["scheduler_kwargs"]["diffusion"], {"total_iters": 2})

    def test_scheduler_horizon_alias_conflicts_and_step_period(self):
        self.assertEqual(scheduler_kwargs("cosine", {"T_max": 7}, 7), {"T_max": 7})
        self.assertEqual(scheduler_kwargs("constant_with_warmup", {}, 2), {"total_iters": 2, "num_warmup_steps": 1000})
        self.assertEqual(scheduler_kwargs("step", {"step_size": 3}, 9), {"step_size": 3})
        for name, values in (("cosine", {"total_iters": 6}), ("cosine", {"T_max": 7, "total_iters": 7}),
                             ("step", {}), ("linear", {"last_epoch": 2})):
            with self.subTest(name=name, values=values), self.assertRaises(ConfigError):
                scheduler_kwargs(name, values, 7)

    def test_standard_native_support_boundary(self):
        for name in STANDARD_OPTIMIZERS:
            with self.subTest(name=name):
                cfg = resolve_process_config({"train": {"optimizer": name}})
                self.assertTrue(all(cfg["gen2"]["optimizers"][role]["optimizer"] == name for role in ROLES))
        for name in ("automagic", "automagic2", "automagic3", "automagicexperiment", "adam8", "adamw8", "adamconvrot", "prodigy8bit", "dadaptationadam", "dadaptationlion"):
            with self.subTest(name=name), self.assertRaisesRegex(ConfigError, "initial delivery"):
                resolve_process_config({"train": {"optimizer": name}})

    def test_native_incompatible_settings_are_explicit(self):
        for raw in ({"train": {"do_cfg": True}}, {"train": {"min_snr_gamma": 5}},
                    {"network": {"rank_dropout": .1}}, {"model": {"compile": True}},
                    {"datasets": [{"caption_dropout_rate": .1}]}, {"train": {"ema_config": {"use_ema": True}}},
                    {"model": {"accuracy_recovery_adapter": "unapproved-adapter"}},
                    {"model": {"qtype": "qfloat8|unapproved-adapter"}},
                    {"datasets": [{"random_triggers": ["rewrite"]}]},
                    {"save": {"push_to_hub": True}},
                    {"logging": {"log_every": 0}},
                    {"gen2": {"optimizers": {"gates": {"optimizer_params": {"do_paramiter_swapping": True}}}}}):
            with self.subTest(raw=raw), self.assertRaises(ConfigError): resolve_process_config(raw)

    def test_dataset_reference_defaults_and_native_rank_alias(self):
        cfg = resolve_process_config({"datasets": [{"folder_path": "/dataset"}], "network": {"rank": 16, "alpha": 8}})
        self.assertEqual(cfg["datasets"][0]["num_workers"], 0)
        self.assertEqual(cfg["network"]["linear"], 16)
        self.assertEqual(cfg["network"]["linear_alpha"], 8)

    def test_invalid_numeric_constraints(self):
        for gen2 in ({"gates": {"amplitude": 1.}}, {"conditioning": {"num_tokens": True}},
                     {"losses": {"neutral_weight": float("nan")}}, {"diagnostics": {"gradient_probe_taus": [0, .5]}},
                     {"diagnostics": {"gradient_probe_taus": [.5, .5]}}, {"diagnostics": {"time_bin_edges": [0, .8, .4, 1]}},
                     {"checkpoint": {"strict_resume": False}}):
            with self.subTest(gen2=gen2), self.assertRaises(ConfigError): resolve_process_config({"gen2": gen2})
        for value in (0., -1., float("inf")):
            with self.subTest(max_grad_norm=value), self.assertRaisesRegex(ConfigError, "max_grad_norm"):
                resolve_process_config({"train": {"max_grad_norm": value}})

    def test_world_size_and_validation_source(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "2"}), self.assertRaisesRegex(ConfigError, "one process"):
            resolve_process_config({})
        with self.assertRaisesRegex(ConfigError, "validation_items"):
            resolve_process_config({"gen2": {"diagnostics": {"probes": {"source": "validation"}}}})


if __name__ == "__main__": unittest.main()
