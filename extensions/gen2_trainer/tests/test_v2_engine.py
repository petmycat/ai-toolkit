"""CPU contracts for the real native AdamW and differentiable frozen backbones."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
import json
import unittest

import torch

from extensions.gen2_trainer.v2.engine import V2Engine, V2UpdateError


def configuration(steps=6, accumulation=1, **optimizer):
    return {"train": {"steps": steps, "gradient_accumulation_steps": accumulation, "max_grad_norm": 0.},
            "gen2": {"optimizer": {"type": "adamw", "lr": .0005, "eps": 1e-8,
                                   "betas": [.9, .999], "weight_decay": 0., **optimizer}}}


class TokenFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E = torch.nn.Parameter(torch.tensor([[.15, -.2, .3, .1], [-.4, .2, .12, -.1]]))
        self.register_buffer("initial", self.E.detach().clone())
        self.register_buffer("seed", torch.tensor(42))


class BackendFixture:
    def __init__(self):
        self.tokens = TokenFixture()
        self.encoder_weight = torch.nn.Parameter(torch.tensor([.7, -.3, .1, .4]), requires_grad=False)
        self.diffusion_weight = torch.nn.Parameter(torch.tensor([.8]), requires_grad=False)
        self.seen = []

    def frozen_parameters(self):
        return (self.encoder_weight, self.diffusion_weight)

    def assert_frozen(self):
        assert all(not parameter.requires_grad for parameter in self.frozen_parameters())

    def encode(self, qs, mode="learned", gradients=True):
        with torch.set_grad_enabled(gradients):
            bank = self.tokens.E if mode == "learned" else self.tokens.initial
            ordinary = torch.tensor([float(value) for value in qs])[:, None]
            # Both the token and ordinary-position feature are downstream of E.
            features = ((bank.mean(0)+ordinary)*self.encoder_weight).tanh().sum(1)
        return SimpleNamespace(features=features, metadata=[{"occurrences": 1} for _ in qs])

    def predict(self, zt, tau, conditioning):
        self.seen.append((torch.is_grad_enabled(), getattr(self, "active_accumulation_index", None)))
        return self.diffusion_weight*zt+conditioning.features[:, None]


def batch(values=(.1, .2, .3)):
    zt = torch.tensor(values).reshape(-1, 1)
    return {"qs": [str(value) for value in values], "zt": zt, "tau": torch.full((len(values),), .4),
            "target": torch.full_like(zt, .8), "metadata": [{"sample_id": str(value)} for value in values]}


class V2EngineTests(unittest.TestCase):
    def test_native_factory_epsilon_and_only_tokens_train(self):
        backend = BackendFixture()
        run = V2Engine(configuration(), backend)
        frozen = [p.clone() for p in backend.frozen_parameters()]
        before = backend.tokens.E.detach().clone()
        self.assertEqual(run.optimizer.defaults["eps"], 1e-6)
        self.assertEqual(run.optimizer.param_groups[0]["eps"], 1e-8)
        row = run.step([batch()])
        self.assertEqual(row["logical_update"], 1)
        self.assertEqual(row["feature_version"], 0)
        self.assertEqual(row["example_count"], 3)
        self.assertGreater(row["gradients"]["before_clip"]["nonzero_count"], 0)
        self.assertGreater(row["parameter_update"]["rms"], 0)
        self.assertFalse(row["gradients"]["clipping_applied"])
        self.assertFalse(torch.equal(before.norm(dim=1), backend.tokens.E.norm(dim=1)))
        self.assertEqual(len(run.optimizer.state), 1)
        self.assertIsNone(run.state_dict()["scheduler"])
        self.assertTrue(backend.seen[0][0])
        # Evidence contains scalar snapshots only, never retained feature graphs.
        json.dumps(row, allow_nan=False)
        self.assertLessEqual(row["examples"][0]["conditioning_features"]["sample_coordinates"], 4096)
        for old, parameter in zip(frozen, backend.frozen_parameters()):
            torch.testing.assert_close(old, parameter, rtol=0, atol=0)
            self.assertIsNone(parameter.grad)

    def test_uneven_accumulation_matches_whole_batch(self):
        whole = V2Engine(configuration(), BackendFixture())
        split = V2Engine(configuration(accumulation=2), BackendFixture())
        full = whole.step([batch()])
        parts = split.step([batch((.1,)), batch((.2, .3))])
        self.assertAlmostEqual(full["loss"], parts["loss"], places=7)
        torch.testing.assert_close(whole.parameter.grad, split.parameter.grad, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(whole.parameter, split.parameter, rtol=0, atol=1e-7)
        self.assertEqual([x[1] for x in split.backend.seen], [0, 1])
        self.assertEqual(sum(item["count"] for item in parts["noise_bins"]), 3)
        self.assertEqual([row["example_weight"] for row in parts["examples"]], [1/3]*3)

    def test_resume_is_exact_for_next_updates(self):
        continuous = V2Engine(configuration(), BackendFixture())
        for _ in range(2):
            continuous.step([batch()])
        state, tokens = deepcopy(continuous.state_dict()), deepcopy(continuous.backend.tokens.state_dict())
        resumed = V2Engine(configuration(), BackendFixture())
        resumed.backend.tokens.load_state_dict(tokens)
        resumed.load_state_dict(state)
        for _ in range(4):
            a, b = continuous.step([batch()]), resumed.step([batch()])
            self.assertEqual(a["loss"], b["loss"])
            torch.testing.assert_close(continuous.parameter, resumed.parameter, rtol=0, atol=0)
        with self.assertRaises(StopIteration):
            resumed.step([batch()])

    def test_zero_or_nonfinite_gradients_abort_without_update(self):
        for mode in ("zero", "nan"):
            with self.subTest(mode=mode):
                run = V2Engine(configuration(), BackendFixture())
                old = run.parameter.detach().clone()
                handle = run.parameter.register_hook(lambda gradient: gradient*0 if mode == "zero" else gradient*float("nan"))
                with self.assertRaises(V2UpdateError):
                    run.step([batch()])
                handle.remove()
                self.assertEqual(run.logical_update, 0)
                self.assertFalse(run.partial_commit)
                self.assertFalse(run.optimizer.state)
                torch.testing.assert_close(run.parameter, old, rtol=0, atol=0)
                with self.assertRaises(V2UpdateError):
                    run.state_dict()

    def test_partial_optimizer_failure_does_not_commit_counter(self):
        run = V2Engine(configuration(), BackendFixture())
        original = run.optimizer.step
        def fail_after_step(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("simulated post-mutation failure")
        run.optimizer.step = fail_after_step
        with self.assertRaises(V2UpdateError):
            run.step([batch()])
        self.assertTrue(run.partial_commit)
        self.assertEqual(run.logical_update, 0)

    def test_resume_rejects_epsilon_tampering_before_mutation(self):
        run = V2Engine(configuration(), BackendFixture())
        run.step([batch()])
        state = deepcopy(run.state_dict())
        state["optimizer"]["param_groups"][0]["eps"] = 1e-6
        fresh = V2Engine(configuration(), BackendFixture())
        with self.assertRaisesRegex(ValueError, "eps"):
            fresh.load_state_dict(state)
        self.assertFalse(fresh.optimizer.state)

    def test_accelerator_cannot_scale_accumulation_twice(self):
        fake = SimpleNamespace(gradient_accumulation_steps=4)
        with self.assertRaisesRegex(ValueError, "accumulation"):
            V2Engine(configuration(), BackendFixture(), fake)

    def test_cpu_scaler_matches_unscaled_update(self):
        class AcceleratorFixture:
            gradient_accumulation_steps = 1
            def __init__(self):
                self.scaler = torch.amp.GradScaler("cpu", init_scale=1024.)
            def autocast(self):
                return nullcontext()
            def backward(self, value):
                self.scaler.scale(value).backward()
        scaled = V2Engine(configuration(), BackendFixture(), AcceleratorFixture())
        plain = V2Engine(configuration(), BackendFixture())
        scaled.step([batch()]); plain.step([batch()])
        torch.testing.assert_close(scaled.parameter, plain.parameter, rtol=0, atol=0)

    def test_resolved_noise_edges_control_summary(self):
        cfg = configuration()
        cfg["gen2"]["diagnostics"] = {"time_bin_edges": [0., .25, .75, 1.]}
        run = V2Engine(cfg, BackendFixture())
        prepared = batch()
        prepared["tau"] = torch.tensor([.1, .25, 1.])
        row = run.step([prepared])
        self.assertEqual([item["bounds"] for item in row["noise_bins"]], [[0., .25], [.25, .75], [.75, 1.]])
        self.assertEqual([item["count"] for item in row["noise_bins"]], [1, 1, 1])

    def test_nonfinite_optimizer_moment_aborts_before_counter_commit(self):
        run = V2Engine(configuration(), BackendFixture())
        original = run.optimizer.step
        def corrupt_moment(*args, **kwargs):
            original(*args, **kwargs)
            run.optimizer.state[run.parameter]["exp_avg"].fill_(float("nan"))
        run.optimizer.step = corrupt_moment
        with self.assertRaisesRegex(V2UpdateError, "moment"):
            run.step([batch()])
        self.assertTrue(run.partial_commit)
        self.assertEqual(run.logical_update, 0)

    def test_resume_validates_moment_shapes_and_optimizer_counter(self):
        run = V2Engine(configuration(), BackendFixture())
        run.step([batch()])
        saved = deepcopy(run.state_dict())
        moments = next(iter(saved["optimizer"]["state"].values()))
        moments["exp_avg"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "shape"):
            run.validate_state_dict(saved)
        saved = deepcopy(run.state_dict())
        next(iter(saved["optimizer"]["state"].values()))["step"] += 1
        with self.assertRaisesRegex(ValueError, "counter"):
            run.validate_state_dict(saved)


if __name__ == "__main__":
    unittest.main()
