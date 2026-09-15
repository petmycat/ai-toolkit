"""Deterministic CPU contracts using real native Adam/AdamW/Adagrad optimizers."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import unittest
from types import SimpleNamespace

import torch

from extensions.gen2_trainer.config import ROLES, resolve_process_config
from extensions.gen2_trainer.engine import Gen2Engine, PartialUpdateError, PhaseSchedule
from extensions.gen2_trainer.gates import CubicTimeGates
from toolkit.optimizer import get_optimizer


def torch_scheduler(name, optimizer, **kwargs):
    # Fixture constructor bridge; production uses toolkit.scheduler unchanged.
    if name == "constant": return torch.optim.lr_scheduler.ConstantLR(optimizer, **kwargs)
    if name == "linear": return torch.optim.lr_scheduler.LinearLR(optimizer, **kwargs)
    if name == "cosine":
        if "total_iters" in kwargs: kwargs["T_max"] = kwargs.pop("total_iters")
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    if name == "cosine_with_restarts":
        if "total_iters" in kwargs: kwargs["T_0"] = kwargs.pop("total_iters")
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)
    raise ValueError(name)


def config(warmup=1, refinement=5, calibration=2, accumulation=1, **extra):
    return resolve_process_config({"train": {"optimizer": "adamw", "lr": .02,
        "max_grad_norm": 1e6, "steps": warmup + refinement + calibration,
        "gradient_accumulation_steps": accumulation, **extra},
        "gen2": {"phases": {"warmup_updates": warmup, "refinement_updates": refinement,
            "calibration_updates": calibration, "diffusion_updates_per_cycle": 1,
            "conditioning_updates_per_cycle": 1}}})


class FixtureBackend:
    """A frozen differentiable backbone, conditioning family pair, and real gates."""
    def __init__(self):
        self.d = torch.nn.Parameter(torch.tensor([.2]))
        self.e = torch.nn.Parameter(torch.tensor([.3]))
        self.t = torch.nn.Parameter(torch.tensor([.4]))
        self.gates = CubicTimeGates(1)
        self.base = torch.nn.Parameter(torch.tensor([.7]), requires_grad=False)
        self.current = None
        self.forward_records = []
        self.regularizer_calls = 0
        original = self.gates.regularizers

        def regularizers():
            self.regularizer_calls += 1
            return original()
        self.gates.regularizers = regularizers

    def parameter_families(self):
        return dict(diffusion=[self.d], embedding=[self.e], text_adapter=[self.t], gates=[self.gates.beta])

    def assert_frozen(self):
        assert not self.base.requires_grad

    def encode(self, qs, styled, gradients=False):
        q = torch.tensor([float(value) for value in qs])
        with torch.set_grad_enabled(gradients):
            features = q + self.e + self.t * q if styled else q
            rt = (self.t.square() / (self.base.detach().square() + 1e-6)).expand(len(q))
        return SimpleNamespace(features=features, rt_per_example=rt, metadata=[{"q": value} for value in qs])

    @contextmanager
    def branch(self, tau, lora_enabled, gate_mode, strength=1.0, name="student"):
        old = self.current
        self.current = dict(tau=tau, enabled=lora_enabled, gates=self.gates.values(tau, gate_mode), strength=strength)
        try: yield self.current
        finally: self.current = old

    def predict(self, latents, tau, conditioning):
        assert self.current is not None
        self.forward_records.append((self.d.detach().clone(), self.e.detach().clone(), self.t.detach().clone(), self.current["enabled"], torch.is_grad_enabled()))
        pred = self.base * latents
        if self.current["enabled"]:
            pred = pred + self.current["strength"] * self.d * self.current["gates"] * (conditioning.features[:, None] + latents)
        return pred

    def state_dict(self):
        return {role: [param.detach().clone() for param in params] for role, params in self.parameter_families().items()}

    def load_state_dict(self, saved):
        with torch.no_grad():
            for role, params in self.parameter_families().items():
                for p, value in zip(params, saved[role]): p.copy_(value)


def batch(values=(.1, .2, .3)):
    z0 = torch.tensor(values).reshape(-1, 1)
    noise = torch.tensor([value + .8 for value in values]).reshape(-1, 1)
    tau = torch.full((len(values),), .4)
    return {"qs": [str(value + 1) for value in values], "z0": z0, "noise": noise,
            "tau": tau, "zt": .6*z0 + .4*noise, "target": noise-z0,
            "metadata": [{"example_id": str(i)} for i in range(len(values))]}


def engine(cfg=None, backend=None, **kwargs):
    return Gen2Engine(cfg or config(), backend or FixtureBackend(), scheduler_factory=torch_scheduler, **kwargs)


def assert_nested_equal(test, left, right):
    if isinstance(left, torch.Tensor): torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left: assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right): assert_nested_equal(test, a, b)
    else: test.assertEqual(left, right)


class ScheduleTests(unittest.TestCase):
    def test_default_totals_and_partial_cycles(self):
        self.assertEqual(PhaseSchedule().horizons, dict(diffusion=2130, embedding=420, text_adapter=420, gates=450))
        schedule = PhaseSchedule(2, 7, 1, 2, 2)
        self.assertEqual([schedule.kind_at(i) for i in range(schedule.total)], list("DDDDAADDAG"))
        self.assertEqual(schedule.horizons, dict(diffusion=6, embedding=3, text_adapter=3, gates=1))
        self.assertEqual(schedule.counts_at(schedule.total), schedule.horizons)
        self.assertEqual(schedule.boundaries, {0, 2, 9, 10})
        with self.assertRaises(StopIteration): schedule.kind_at(schedule.total)

    def test_empty_stages(self):
        schedule = PhaseSchedule(0, 0, 2)
        self.assertEqual(schedule.kind_at(0), "G")
        self.assertEqual(schedule.stage_at(0), "calibration")


class EngineTests(unittest.TestCase):
    def test_native_factories_called_once_per_family(self):
        calls, scheduler_calls = [], []
        def factory(params, **kwargs):
            calls.append(deepcopy(kwargs))
            return get_optimizer(params, **kwargs)
        def scheduling(name, opt, **kwargs):
            scheduler_calls.append((name, deepcopy(kwargs)))
            return torch_scheduler(name, opt, **kwargs)
        run = Gen2Engine(config(), FixtureBackend(), optimizer_factory=factory, scheduler_factory=scheduling)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(scheduler_calls), 4)
        self.assertEqual([call[1]["total_iters"] for call in scheduler_calls], [4, 2, 2, 2])
        self.assertTrue(all(call["optimizer_type"] == "adamw" for call in calls))

    def test_zero_horizon_omits_scheduler_but_keeps_optimizer_state(self):
        run = engine(config(2, 0, 0, lr_scheduler="cosine_with_restarts"))
        self.assertEqual(len(run.optimizers), 4)
        for role in ("embedding", "text_adapter", "gates"):
            self.assertIsNone(run.schedulers[role])
            desc = run.state_dict()["schedulers"][role]
            self.assertFalse(desc["active"])
            self.assertEqual(desc["horizon"], 0)
            self.assertIsNone(desc["state"])
        saved = deepcopy(run.state_dict())
        reloaded = engine(config(2, 0, 0, lr_scheduler="cosine_with_restarts"))
        reloaded.load_state_dict(saved)
        assert_nested_equal(self, saved, reloaded.state_dict())

    def test_exact_gradient_ownership_and_inactive_state(self):
        run = engine()
        for i in range(run.schedule.total):
            kind = run.schedule.kind_at(i)
            expected = {"D": {"diffusion"}, "A": {"embedding", "text_adapter"}, "G": {"gates"}}[kind]
            before = deepcopy(run.state_dict())
            params = run.backend.state_dict()
            row = run.step([batch()])
            self.assertEqual(row["update_kind"], kind)
            self.assertEqual(row["feature_version"], i)
            self.assertTrue(row["gradient_ownership_asserted"])
            self.assertEqual(run.family_steps, run.schedule.counts_at(i+1))
            for role in ROLES:
                if role not in expected:
                    assert_nested_equal(self, params[role], run.backend.state_dict()[role])
                    assert_nested_equal(self, before["optimizers"][role], run.state_dict()["optimizers"][role])
                    assert_nested_equal(self, before["schedulers"][role], run.state_dict()["schedulers"][role])
                else:
                    self.assertTrue(any(not torch.equal(a,b) for a,b in zip(params[role], run.backend.state_dict()[role])))
            self.assertIsNone(run.backend.current)

    def test_teacher_then_both_students_before_update(self):
        run = engine(config(1, 0, 0))
        original = run.backend.d.detach().clone()
        run.step([batch()])
        records = run.backend.forward_records
        self.assertEqual([record[3] for record in records], [False, True, True])
        self.assertFalse(records[0][4])
        self.assertTrue(all(torch.equal(record[0], original) for record in records))

    def test_activation_hooks_receive_each_microbatch_identity(self):
        run = engine(config(1, 0, 0, accumulation=2))
        first, second = batch((.1,)), batch((.2, .3))
        seen = []
        original = run.backend.predict
        def predict(*args, **kwargs):
            seen.append((run.backend.active_accumulation_index,
                         deepcopy(run.backend.active_batch_metadata)))
            return original(*args, **kwargs)
        run.backend.predict = predict
        run.step([first, second])
        self.assertEqual(seen, [(0, first["metadata"])] * 3 + [(1, second["metadata"])] * 3)

    def test_unequal_microbatches_equal_concatenation_for_d_a_and_g(self):
        for stages, steps in (((1, 0, 0), 1), ((0, 2, 0), 2), ((0, 0, 1), 1)):
            with self.subTest(stages=stages):
                combined = engine(config(*stages, accumulation=1))
                split = engine(config(*stages, accumulation=2))
                if stages[-1]:
                    with torch.no_grad():
                        combined.backend.gates.beta.fill_(.2)
                        split.backend.gates.beta.copy_(combined.backend.gates.beta)
                for _ in range(steps):
                    row_full = combined.step([batch((.1,.2,.3))])
                    row_split = split.step([batch((.1,)), batch((.2,.3))])
                    self.assertAlmostEqual(row_full["active_total"], row_split["active_total"], places=6)
                    for role in ROLES:
                        for a,b in zip(combined.families[role],split.families[role]):
                            torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)
                            if a.grad is not None: torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=2e-6)
                self.assertEqual(combined.backend.regularizer_calls, split.backend.regularizer_calls)
                if stages[-1]: self.assertEqual(split.backend.regularizer_calls, 1)

    def test_nonfinite_second_a_family_prevents_both_steps(self):
        run = engine(config(0, 2, 0))
        run.step([batch()])
        run.backend.t.requires_grad_(True)
        run.backend.t.register_hook(lambda grad: grad * float("nan"))
        before = run.backend.state_dict()
        counters = deepcopy(run.family_steps)
        with self.assertRaisesRegex(FloatingPointError, "text_adapter"):
            run.step([batch()])
        assert_nested_equal(self, before, run.backend.state_dict())
        self.assertEqual(run.family_steps, counters)
        self.assertEqual(run.logical_update, 1)
        self.assertFalse(run.partial_commit)
        self.assertIsNone(run.backend.current)
        with self.assertRaises(RuntimeError): run.state_dict()

    def test_second_a_optimizer_failure_is_partial_and_not_checkpointable(self):
        run = engine(config(0, 2, 0))
        run.step([batch()])
        original = run.backend.e.detach().clone()
        def fail(): raise RuntimeError("fixture optimizer failure")
        run.optimizers["text_adapter"].step = fail
        with self.assertRaises(PartialUpdateError): run.step([batch()])
        self.assertTrue(run.partial_commit)
        self.assertEqual(run.logical_update, 1)
        self.assertFalse(torch.equal(original, run.backend.e))
        with self.assertRaises(RuntimeError): run.state_dict()

    def test_native_optimizer_resume_matches_continuous_across_phase_boundary(self):
        for name in ("adam", "adamw", "adagrad"):
            with self.subTest(name=name):
                cfg = config(optimizer=name, lr_scheduler="cosine")
                continuous = engine(cfg)
                interrupted = engine(cfg)
                for _ in range(8): continuous.step([batch()])
                for _ in range(3): interrupted.step([batch()])
                state, params = deepcopy(interrupted.state_dict()), interrupted.backend.state_dict()
                resumed = engine(cfg)
                resumed.backend.load_state_dict(params)
                resumed.load_state_dict(state)
                for _ in range(5): resumed.step([batch()])
                assert_nested_equal(self, continuous.backend.state_dict(), resumed.backend.state_dict())
                assert_nested_equal(self, continuous.state_dict(), resumed.state_dict())

    def test_resume_rejects_missing_or_mismatched_state(self):
        run = engine()
        saved = deepcopy(run.state_dict())
        for mutate in (lambda s: s["optimizers"].pop("gates"),
                       lambda s: s["family_steps"].update(embedding=1),
                       lambda s: s["schedule"].update(warmup_updates=3),
                       lambda s: s["schedulers"]["embedding"].update(state=None)):
            damaged = deepcopy(saved)
            mutate(damaged)
            with self.assertRaises(ValueError): engine().load_state_dict(damaged)

    def test_accelerator_may_not_divide_accumulation_twice(self):
        with self.assertRaisesRegex(ValueError, "accumulation_steps must be 1"):
            engine(accelerator=SimpleNamespace(gradient_accumulation_steps=2))

    def test_clipping_before_actual_optimizer_step(self):
        run = engine(config(1, 0, 0, max_grad_norm=.001))
        row = run.step([batch()])
        stats = row["gradients"]["diffusion"]
        self.assertTrue(stats["clipping_applied"])
        self.assertLessEqual(stats["post_clip"]["l2_norm"], .001001)
        self.assertGreater(stats["pre_clip"]["l2_norm"], .001)

    def test_shared_diagnostic_budget_aborts_before_optimizer(self):
        run = engine(config(1, 0, 0))
        original = run.backend.state_dict()
        budget = run.gen2["diagnostics"]["tensor_memory_budget_mb"] * 1024 ** 2
        run.external_diagnostic_bytes = lambda: budget - run.retained_diagnostic_bytes()
        with self.assertRaisesRegex(RuntimeError, "Diagnostic tensor budget"):
            run.step([batch()])
        self.assertFalse(run.partial_commit)
        self.assertEqual(run.logical_update, 0)
        assert_nested_equal(self, original, run.backend.state_dict())

    def test_optional_full_snapshots_use_only_remaining_budget(self):
        cfg = config(1, 0, 0)
        cfg["gen2"]["diagnostics"]["full_update_norm_every"] = 1
        run = engine(cfg)
        budget = run.gen2["diagnostics"]["tensor_memory_budget_mb"] * 1024 ** 2
        run.external_diagnostic_bytes = budget - run.retained_diagnostic_bytes() - 12
        row = run.step([batch()])
        self.assertEqual(row["parameter_updates"]["diffusion"]["full_update_reason"], "tensor_memory_budget_exceeded")

    def test_real_grad_scaler_unscales_both_a_optimizers_and_updates_once(self):
        class CPUAccelerator:
            gradient_accumulation_steps = 1
            def __init__(self):
                self.scaler = torch.amp.GradScaler("cpu", init_scale=1024., growth_interval=1)
            def backward(self, loss): self.scaler.scale(loss).backward()
            def autocast(self): return nullcontext()
        cfg = config(0, 2, 0)
        scaled = engine(cfg, accelerator=CPUAccelerator())
        reference = engine(cfg)
        for _ in range(2):
            scaled.step([batch()])
            reference.step([batch()])
        assert_nested_equal(self, scaled.backend.state_dict(), reference.backend.state_dict())
        self.assertEqual(scaled.scaler.get_scale(), 4096.)


if __name__ == "__main__": unittest.main()
