"""Numerical contracts for matched objectives and reversible diagnostic descent."""
from copy import deepcopy
import hashlib
import json
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from extensions.gen2_trainer.diagnostics import capture_rng_state
from extensions.gen2_trainer.v2.diagnostic_objective import run_objective_diagnostics


class CompilerFixture:
    def __init__(self):
        self.mismatch = False

    def comparison(self, caption, named_phrase=None):
        modes = ["base", "init", "learned"] + (["named"] if named_phrase else [])
        digest = hashlib.sha256(caption.encode()).hexdigest()
        return {mode: SimpleNamespace(metadata={
            "mode": mode, "original_caption": caption, "resulting_length": 7,
            "shared_comparison_content": True,
            "common_ordinary_content_sha256": "wrong" if self.mismatch and mode == "base" else digest,
        }) for mode in modes}


class BackendFixture:
    def __init__(self):
        self.tokens = torch.nn.Module()
        self.tokens.E = torch.nn.Parameter(torch.tensor([[.5, -.2]]))
        self.tokens.register_buffer("initial", torch.zeros(1, 2))
        self.original = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)
        self.compiler = CompilerFixture()
        self.calls = []

    def assert_frozen(self):
        assert self.tokens.E.dtype == torch.float32

    def frozen_parameters(self):
        return [self.original]

    def encode(self, qs, mode="learned", gradients=False, compiled=None):
        assert compiled is not None and len(qs) == len(compiled)
        for caption, item in zip(qs, compiled):
            assert item.metadata["mode"] == mode and item.metadata["original_caption"] == caption
        with torch.set_grad_enabled(gradients):
            effect = (self.tokens.E.sum() if mode == "learned" else
                      self.tokens.initial.sum() if mode == "init" else
                      torch.tensor(1.) if mode == "named" else torch.tensor(0.))
            return SimpleNamespace(features=effect.expand(len(qs)), mode=mode)

    def predict(self, zt, tau, conditioning):
        # Exercise preservation of all three common RNG sources without adding
        # stochastic noise to this known quadratic objective.
        random.random(); np.random.random(); torch.rand(())
        self.calls.append((conditioning.mode, zt.detach().clone(), tau.detach().clone()))
        return self.original*zt + conditioning.features[:, None]


class RecorderFixture:
    def __init__(self):
        self.rows = []

    def record(self, stream, payload):
        # Enforce that no graphs, tensors or nonfinite JSON numbers escape.
        json.dumps(payload, allow_nan=False)
        self.rows.append((stream, deepcopy(payload)))


def settings(steps=3, count=4):
    return {"named_phrase": "benchmark", "seed": 77, "descent_steps": steps,
            "descent_packet_count": count,
            "optimizer": {"type": "adamw", "lr": .05, "eps": 1e-8,
                          "betas": [.9, .999], "weight_decay": 0.}}


def packet(identifier="one", values=(0.,), targets=None):
    zt = torch.tensor(values).reshape(-1, 1)
    target = torch.full_like(zt, 3.) if targets is None else torch.tensor(targets).reshape(-1, 1)
    return {"id": identifier, "qs": [f"caption {i}" for i in range(len(values))],
            "zt": zt, "target": target, "tau": torch.full((len(values),), .5),
            "metadata": [{"sample_id": f"{identifier}:{i}", "resolution": 768} for i in range(len(values))]}


def assert_tree(test, a, b):
    if torch.is_tensor(a):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for key in a:
            assert_tree(test, a[key], b[key])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for first, second in zip(a, b):
            assert_tree(test, first, second)
    else:
        test.assertEqual(a, b)


class V2DiagnosticObjectiveTests(unittest.TestCase):
    def test_paired_modes_share_inputs_and_exact_known_losses(self):
        backend, recorder = BackendFixture(), RecorderFixture()
        source = packet(values=(0., 1.))
        result = run_objective_diagnostics(backend, [source], settings(steps=0), recorder)
        paired = next(row for stream, row in recorder.rows if stream == "paired_losses")
        self.assertEqual(paired["mean_loss_by_mode"]["base"], 6.5)
        self.assertEqual(paired["mean_loss_by_mode"]["init"], 6.5)
        self.assertEqual(paired["mean_loss_by_mode"]["named"], 2.5)
        expected = ((source["zt"]+.3-source["target"])**2).mean().item()
        self.assertAlmostEqual(paired["mean_loss_by_mode"]["learned"], expected, places=6)
        self.assertAlmostEqual(paired["examples"][0]["prediction_difference_to_learned"]["base"]["mse"], .09, places=6)
        for _, zt, tau in backend.calls[:4]:
            torch.testing.assert_close(zt, source["zt"], rtol=0, atol=0)
            torch.testing.assert_close(tau, source["tau"], rtol=0, atol=0)
        self.assertEqual(set(result["paired"]["by_resolution"]), {"768"})
        self.assertTrue(result["completed"])
        self.assertEqual(result["style_acceptance"], "not_measured")
        json.dumps(result, allow_nan=False)

    def test_both_quadratic_descents_reduce_loss_and_restore_state_and_rng(self):
        backend, recorder = BackendFixture(), RecorderFixture()
        backend.tokens.E.requires_grad_(False)
        backend.tokens.E.grad = torch.tensor([[.12, .34]])
        original_grad = backend.tokens.E.grad
        saved = deepcopy(backend.tokens.state_dict())
        rng = capture_rng_state()
        result = run_objective_diagnostics(backend, [packet()], settings(steps=6), recorder)
        for label in ("initial", "learned"):
            observed = result["descent"][label]
            self.assertLess(observed["final_loss"], observed["initial_loss"])
            rows = [row for stream, row in recorder.rows if stream == "descent" and row["start"] == label]
            self.assertEqual([r["step"] for r in rows], list(range(7)))
            self.assertTrue(all(row["loss_delta_from_previous"] < 0 for row in rows[1:]))
        self.assertTrue(result["source_token_state_restored"])
        assert_tree(self, backend.tokens.state_dict(), saved)
        assert_tree(self, capture_rng_state(), rng)
        self.assertIs(backend.tokens.E.grad, original_grad)
        torch.testing.assert_close(original_grad, torch.tensor([[.12, .34]]), rtol=0, atol=0)
        self.assertFalse(backend.tokens.E.requires_grad)
        self.assertIsNone(backend.original.grad)

    def test_uneven_packets_use_example_weighting_for_objective_and_gradients(self):
        split, whole = BackendFixture(), BackendFixture()
        a, b = RecorderFixture(), RecorderFixture()
        divided = [packet("a", (0.,), (1.,)), packet("b", (.1, .2, .3), (2., 3., 4.))]
        combined = [packet("all", (0., .1, .2, .3), (1., 2., 3., 4.))]
        first = run_objective_diagnostics(split, divided, settings(steps=1), a)
        second = run_objective_diagnostics(whole, combined, settings(steps=1), b)
        self.assertAlmostEqual(first["paired"]["mean_loss_by_mode"]["learned"],
                               second["paired"]["mean_loss_by_mode"]["learned"], places=7)
        ga = [row["gradients"]["l2"] for stream, row in a.rows if stream == "descent" and row["step"] == 1]
        gb = [row["gradients"]["l2"] for stream, row in b.rows if stream == "descent" and row["step"] == 1]
        for x, y in zip(ga, gb):
            self.assertAlmostEqual(x, y, delta=1e-6)  # FP32 addition order differs across microbatches.
        for label in first["descent"]:
            self.assertAlmostEqual(first["descent"][label]["final_loss"], second["descent"][label]["final_loss"], places=6)

    def test_mismatched_comparison_content_is_rejected_before_forward(self):
        backend = BackendFixture()
        backend.compiler.mismatch = True
        saved = deepcopy(backend.tokens.state_dict())
        rng = capture_rng_state()
        with self.assertRaisesRegex(ValueError, "different ordinary caption content"):
            run_objective_diagnostics(backend, [packet()], settings(), RecorderFixture())
        self.assertFalse(backend.calls)
        assert_tree(self, backend.tokens.state_dict(), saved)
        assert_tree(self, capture_rng_state(), rng)

    def test_failure_after_update_restores_parameter_initial_grad_and_rng(self):
        backend = BackendFixture()
        backend.tokens.E.grad = torch.tensor([[9., 8.]])
        grad = backend.tokens.E.grad
        saved = deepcopy(backend.tokens.state_dict())
        rng = capture_rng_state()
        class FailingRecorder(RecorderFixture):
            def record(self, stream, row):
                super().record(stream, row)
                if stream == "descent" and row["step"] == 1:
                    backend.tokens.initial.add_(17.)
                    raise OSError("injected recorder failure after optimizer update")
        with self.assertRaisesRegex(OSError, "after optimizer update"):
            run_objective_diagnostics(backend, [packet()], settings(), FailingRecorder())
        assert_tree(self, backend.tokens.state_dict(), saved)
        assert_tree(self, capture_rng_state(), rng)
        self.assertIs(backend.tokens.E.grad, grad)
        torch.testing.assert_close(grad, torch.tensor([[9., 8.]]), rtol=0, atol=0)

    def test_descent_selection_spreads_across_packets_and_accepts_exact_ids(self):
        packets = [packet(str(index)) for index in range(8)]
        result = run_objective_diagnostics(BackendFixture(), packets, settings(steps=0, count=3), RecorderFixture())
        self.assertEqual(result["descent_packet_ids"], ["0", "4", "7"])
        options = settings(steps=0)
        options["descent_packet_ids"] = ["2", "6"]
        result = run_objective_diagnostics(BackendFixture(), packets, options, RecorderFixture())
        self.assertEqual(result["descent_packet_ids"], ["2", "6"])

    def test_nonfinite_gradients_abort_and_restore_source(self):
        backend = BackendFixture()
        saved = deepcopy(backend.tokens.state_dict())
        hook = backend.tokens.E.register_hook(lambda gradient: gradient*float("nan"))
        try:
            with self.assertRaisesRegex(FloatingPointError, "Nonfinite"):
                run_objective_diagnostics(backend, [packet()], settings(), RecorderFixture())
        finally:
            hook.remove()
        assert_tree(self, backend.tokens.state_dict(), saved)
        self.assertIsNone(backend.tokens.E.grad)

    def test_source_optimizer_must_explicitly_be_adamw(self):
        for kind in (None, "adamw8bit", "sgd"):
            with self.subTest(kind=kind):
                backend = BackendFixture()
                options = settings()
                options["optimizer"]["type"] = kind
                with self.assertRaisesRegex(ValueError, "optimizer.type: adamw"):
                    run_objective_diagnostics(backend, [packet()], options, RecorderFixture())
                self.assertFalse(backend.calls)

    def test_forward_autocast_and_fp32_loss_reduction(self):
        class AutocastBackend(BackendFixture):
            def encode(self, *args, **kwargs):
                if not torch.is_autocast_enabled("cpu"):
                    raise AssertionError("Encoder forward lost its configured autocast context")
                return super().encode(*args, **kwargs)

            def predict(self, *args, **kwargs):
                result = super().predict(*args, **kwargs)
                output = torch.mm(result, self.original.reshape(1, 1))
                if output.dtype != torch.bfloat16:
                    raise AssertionError("Diffusion forward lost its configured autocast context")
                return output

        source = packet(values=(.12345, .6789))
        backend, recorder = AutocastBackend(), RecorderFixture()
        with patch("extensions.gen2_trainer.v2.diagnostic_objective._autocast",
                   side_effect=lambda _: torch.autocast("cpu", dtype=torch.bfloat16)):
            result = run_objective_diagnostics(backend, [source], settings(steps=1), recorder)
        expected = ((source["zt"]+.3).bfloat16().float()-source["target"]).square().mean().item()
        self.assertAlmostEqual(result["paired"]["mean_loss_by_mode"]["learned"], expected, places=6)
        for stream, row in recorder.rows:
            if stream == "descent" and row["step"] == 1:
                self.assertTrue(row["gradients"]["finite"])


if __name__ == "__main__":
    unittest.main()
