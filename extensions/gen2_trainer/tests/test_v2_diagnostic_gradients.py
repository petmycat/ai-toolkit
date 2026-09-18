"""CPU harness tests. These fixtures do not certify real Ideogram/CUDA parity."""
from contextlib import nullcontext
import json
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from extensions.gen2_trainer.v2 import diagnostic_gradients as diagnostic


class Projection(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs)
        self.gradient_checkpointing = True
        self.stochastic = False

    def forward(self, x):
        value = self.linear(x).tanh()
        return torch.nn.functional.dropout(value, .25, training=True) if self.stochastic else value


class BrokenBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient*1.75


class FakeQuantizedLinear(nn.Linear):
    """Represented-weight seam without any optional quantization package."""
    @property
    def qweight(self):
        return self.weight


class WrongGradientLinear(FakeQuantizedLinear):
    def forward(self, value):
        return super().forward(BrokenBackward.apply(value))


class NonfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient*float("nan")


class Backend:
    def __init__(self):
        self.tokens = nn.Module()
        self.tokens.E = nn.Parameter(torch.tensor([[.2, -.3, .4], [-.6, .1, .2]]))
        self.model = SimpleNamespace(text_encoder=Projection(3, 5), transformer=Projection(5, 3))
        self.encoder_checkpointing = True
        for component in (self.model.text_encoder, self.model.transformer):
            component.eval().requires_grad_(False)
        self.behavior = None
        self.calls = 0
        self.random_draws = []

    def encode(self, captions, mode, gradients):
        self.calls += 1
        self.random_draws.append((random.random(), float(np.random.rand()), float(torch.rand(()))))
        if self.behavior == "oom" and not self.encoder_checkpointing:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory in test fixture")
        if self.behavior == "error" and not self.encoder_checkpointing:
            raise RuntimeError("fixture encoder execution failed")
        layer = self.model.text_encoder
        features = checkpoint(layer, self.tokens.E, use_reentrant=False) if self.encoder_checkpointing else layer(self.tokens.E)
        if self.behavior == "gradient_mismatch" and not self.encoder_checkpointing:
            features = BrokenBackward.apply(features)
        if self.behavior == "nonfinite_gradient" and not self.encoder_checkpointing:
            features = NonfiniteBackward.apply(features)
        return SimpleNamespace(features=[features for _ in captions])

    def predict(self, latents, tau, conditioning):
        layer = self.model.transformer
        values = torch.stack([feature.mean(0) for feature in conditioning.features])
        result = checkpoint(layer, values, use_reentrant=False) if layer.gradient_checkpointing else layer(values)
        prediction = latents*.2+result[:, :, None, None]
        if self.behavior == "nonfinite" and not layer.gradient_checkpointing:
            prediction = prediction*float("nan")
        if self.behavior == "disconnected" and not layer.gradient_checkpointing:
            prediction = prediction.detach()
        return prediction

    def assert_frozen(self):
        for component in (self.model.text_encoder, self.model.transformer):
            assert all(not parameter.requires_grad and parameter.grad is None for parameter in component.parameters())


class Recorder:
    def __init__(self):
        self.rows = []
        self.events = []

    def record(self, stream, record):
        json.dumps(record, allow_nan=False)
        self.rows.append((stream, record))

    def event(self, name, **values):
        json.dumps(values, allow_nan=False)
        self.events.append((name, values))


def packet():
    return {"id": "unchanged-real-packet-fixture", "qs": ["original caption [trigger]"],
            "zt": torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)/10,
            "tau": torch.tensor([.5]), "target": torch.full((1, 3, 2, 2), .15),
            "metadata": [{"source": "CPU fixture, not real model acceptance"}]}


class GradientDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.autocast = patch.object(diagnostic, "_autocast", side_effect=nullcontext)
        self.autocast.start()
        self.addCleanup(self.autocast.stop)
        self.progress = patch.object(diagnostic, "_progress")
        self.progress_mock = self.progress.start()
        self.addCleanup(self.progress.stop)

    def test_equal_checkpoint_gradients_and_real_linear_reference(self):
        backend, recorder = Backend(), Recorder()
        result = diagnostic.run_gradient_diagnostics(backend, packet(), {}, recorder)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["checkpoint_variants"]), 5)
        for row in result["checkpoint_variants"][1:]:
            self.assertTrue(row["passed"])
            self.assertEqual(row["gradient_difference"]["relative_l2"], 0.)
            self.assertEqual(row["prediction_difference"]["relative_l2"], 0.)
        self.assertEqual(len(result["linear_comparisons"]), 2)
        self.assertTrue(all(row["passed"] for row in result["linear_comparisons"]))
        self.assertTrue(all(value == backend.random_draws[0] for value in backend.random_draws))
        json.dumps(result, allow_nan=False)

    def test_restores_flags_bank_existing_grad_object_and_all_rng(self):
        backend = Backend()
        backend.encoder_checkpointing = False
        backend.model.transformer.gradient_checkpointing = True
        backend.tokens.E.requires_grad_(False)
        original_grad = torch.full_like(backend.tokens.E, .7)
        backend.tokens.E.grad = original_grad
        values = backend.tokens.E.detach().clone()
        original_rng = diagnostic._rng_state()
        with torch.no_grad():
            result = diagnostic.run_gradient_diagnostics(backend, packet(), {}, Recorder())
        self.assertEqual(result["status"], "passed")
        self.assertFalse(backend.encoder_checkpointing)
        self.assertTrue(backend.model.transformer.gradient_checkpointing)
        self.assertFalse(backend.tokens.E.requires_grad)
        self.assertIs(backend.tokens.E.grad, original_grad)
        torch.testing.assert_close(backend.tokens.E.grad, torch.full_like(original_grad, .7), rtol=0, atol=0)
        torch.testing.assert_close(backend.tokens.E, values, rtol=0, atol=0)
        current = diagnostic._rng_state()
        self.assertEqual(current["python"], original_rng["python"])
        np.testing.assert_array_equal(current["numpy"][1], original_rng["numpy"][1])
        torch.testing.assert_close(current["torch"], original_rng["torch"], rtol=0, atol=0)

    def test_checkpoint_recomputation_preserves_stochastic_forward_gradients(self):
        backend = Backend()
        backend.model.text_encoder.stochastic = True
        backend.model.transformer.stochastic = True
        result = diagnostic.run_gradient_diagnostics(backend, packet(), {}, None)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(row["gradient_difference"]["relative_l2"] == 0.
                            for row in result["checkpoint_variants"][1:]))

    def test_detects_backward_discrepancy_despite_identical_outputs(self):
        backend = Backend()
        backend.behavior = "gradient_mismatch"
        result = diagnostic.run_gradient_diagnostics(backend, packet(), {}, None)
        self.assertEqual(result["status"], "failed")
        row = next(row for row in result["checkpoint_variants"] if row["variant"] == "encoder_off")
        self.assertEqual(row["prediction_difference"]["relative_l2"], 0.)
        self.assertGreater(row["gradient_difference"]["relative_l2"], .7)
        self.assertFalse(row["passed"])

    def test_nonfinite_disconnected_oom_and_execution_errors_are_incomplete(self):
        for behavior, reason in (("nonfinite", "nonfinite_or_disconnected"),
                                 ("nonfinite_gradient", "nonfinite_or_disconnected"),
                                 ("disconnected", "nonfinite_or_disconnected"),
                                 ("oom", "cuda_oom"), ("error", "execution_error")):
            with self.subTest(behavior=behavior):
                backend = Backend()
                backend.behavior = behavior
                saved_rng = torch.get_rng_state().clone()
                result = diagnostic.run_gradient_diagnostics(backend, packet(), {}, Recorder())
                self.assertEqual(result["status"], "inconclusive")
                self.assertFalse(result["complete"])
                self.assertIsNone(result["passed"])
                incomplete = [row for row in result["checkpoint_variants"] if not row["complete"]]
                self.assertTrue(incomplete)
                self.assertTrue(all(row["reason"] == reason for row in incomplete))
                self.assertTrue(backend.encoder_checkpointing)
                self.assertTrue(backend.model.transformer.gradient_checkpointing)
                torch.testing.assert_close(saved_rng, torch.get_rng_state(), rtol=0, atol=0)
                self.assertTrue(all(not layer._forward_pre_hooks for component in (
                    backend.model.text_encoder, backend.model.transformer) for layer in component.modules()))

    def test_missing_baseline_never_passes_remaining_successful_variants(self):
        backend = Backend()
        original = backend.encode
        def fail_first(*args, **kwargs):
            if backend.calls == 0:
                backend.calls += 1
                raise torch.cuda.OutOfMemoryError("CUDA out of memory baseline")
            return original(*args, **kwargs)
        backend.encode = fail_first
        result = diagnostic.run_gradient_diagnostics(backend, packet(), {}, None)
        self.assertEqual(result["status"], "inconclusive")
        self.assertTrue(all(row["passed"] is None for row in result["checkpoint_variants"]))
        self.assertTrue(all(not row["complete"] for row in result["checkpoint_variants"]))

    def test_actual_selected_quantized_seam_detects_wrong_input_gradient(self):
        backend = Backend()
        wrong = WrongGradientLinear(3, 5).requires_grad_(False)
        backend.model.text_encoder.linear = wrong
        result = diagnostic.run_gradient_diagnostics(backend, packet(), {}, None)
        row = next(row for row in result["linear_comparisons"] if row["family"] == "encoder")
        self.assertTrue(row["quantized"])
        self.assertEqual(row["output"]["relative_l2"], 0.)
        self.assertGreater(row["input_gradient"]["relative_l2"], .7)
        self.assertEqual(row["status"], "failed")
        self.assertEqual(result["status"], "failed")

    def test_recorder_exception_still_restores_flags_grad_and_rng(self):
        backend = Backend()
        backend.tokens.E.grad = torch.ones_like(backend.tokens.E)
        original_bank = backend.tokens.E.detach().clone()
        original_rng = torch.get_rng_state().clone()
        recorder = Recorder()
        def broken(stream, row):
            if row.get("variant") == "encoder_off":
                with torch.no_grad():
                    backend.tokens.E.add_(1.)
                    backend.tokens.E.grad.mul_(2.)
                raise OSError("cannot record")
        recorder.record = broken
        with self.assertRaisesRegex(OSError, "cannot record"):
            diagnostic.run_gradient_diagnostics(backend, packet(), {}, recorder)
        self.assertTrue(backend.encoder_checkpointing and backend.model.transformer.gradient_checkpointing)
        torch.testing.assert_close(backend.tokens.E.grad, torch.ones_like(backend.tokens.E))
        torch.testing.assert_close(backend.tokens.E, original_bank)
        torch.testing.assert_close(torch.get_rng_state(), original_rng)
        self.assertTrue(all(not module._forward_pre_hooks for component in (
            backend.model.text_encoder, backend.model.transformer) for module in component.modules()))

    def test_rejects_unbounded_settings_and_unknown_checkpoint_contract(self):
        for settings in ({"linear_modules_per_model": 3}, {"linear_input_rows": 100},
                         {"gradient_relative_l2_max": float("nan")}, {"gradient_cosine_min": 2}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                diagnostic.run_gradient_diagnostics(Backend(), packet(), settings, None)
        backend = Backend()
        del backend.model.transformer.gradient_checkpointing
        with self.assertRaisesRegex(TypeError, "native transformer.gradient_checkpointing"):
            diagnostic.run_gradient_diagnostics(backend, packet(), {}, None)

    def test_encoder_candidates_exclude_vision_and_transformer_uses_executed_blocks(self):
        encoder = nn.Module()
        encoder.language_model = nn.Module()
        encoder.language_model.layers = nn.ModuleList([Projection(3, 3), Projection(3, 3)])
        encoder.visual = Projection(3, 3)
        encoder.lm_head = nn.Linear(3, 3)
        selected = diagnostic._select_linears(encoder, 4, "encoder")
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(name.startswith("language_model.layers.") for name, _ in selected))
        transformer = nn.Module()
        transformer.layers = nn.ModuleList([Projection(3, 3), Projection(3, 3)])
        transformer.unused_text_head = nn.Linear(3, 3)
        selected = diagnostic._select_linears(transformer, 4, "transformer")
        self.assertTrue(all(name.startswith("layers.") for name, _ in selected))

    def test_real_capture_fallback_skips_an_unexecuted_candidate(self):
        backend = Backend()
        backend.model.text_encoder.a_unused = nn.Linear(3, 5).requires_grad_(False)
        result = diagnostic.run_gradient_diagnostics(backend, packet(), {"linear_modules_per_model": 1}, None)
        self.assertEqual(result["status"], "passed")
        rows = [row for row in result["linear_comparisons"] if row["family"] == "encoder"]
        self.assertEqual([row["module"] for row in rows], ["linear"])
        unused = next(row for row in result["linear_capture_candidates"] if row["module"] == "a_unused")
        self.assertFalse(unused["captured"] or unused["selected"])

    def test_progress_covers_every_attempt_and_flushes_console(self):
        backend = Backend()
        result = diagnostic.run_gradient_diagnostics(backend, packet(), {}, None)
        messages = [call.args[0] for call in self.progress_mock.call_args_list]
        self.assertEqual(sum(": starting" in message for message in messages), 7)
        self.assertEqual(sum(" in " in message for message in messages), 7)
        self.assertTrue(any("configured_repeat: passed" in message for message in messages))
        self.assertTrue(any("linear 2/2 transformer.linear: passed" in message for message in messages))
        self.progress.stop()
        with patch("builtins.print") as output:
            diagnostic._progress("test progress")
        output.assert_called_once_with("[gen2 diagnostic gradients] test progress", flush=True)


if __name__ == "__main__":
    unittest.main()
