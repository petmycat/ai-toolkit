"""Checkpoint replay must carry routing into a fresh autograd worker context.

The CPU fixtures execute the checked-in native transformer and LoRA code. They
exercise the same Python-context boundary as CUDA's autograd worker without
requiring the production weights or optional quantization packages.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import Context, ContextVar, copy_context
from dataclasses import dataclass
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from extensions.gen2_trainer.backend_ideogram4 import (
    DIFFUSION_PROJECTIONS, DiffusionBranch, Ideogram4Backend, RECOMPUTING,
    differentiable_qwen_features, make_gated_lora_class, recomputation_context,
)
from extensions.gen2_trainer.gates import CubicTimeGates
from extensions.gen2_trainer.diagnostics import isolated_gradient_probe
from extensions.gen2_trainer.tests.test_backend_contract import (
    native_definitions, native_lora_class,
)
from extensions.gen2_trainer.tests.test_conditioning import fixture_encoder


class FixtureNetwork:
    network_type = "lora"
    is_active = True


def native_transformer():
    class Mixin:
        pass
    scope = dict(torch=torch, nn=nn, F=F, math=math, dataclass=dataclass,
                 checkpoint=checkpoint, OstrisModelMixin=Mixin,
                 QWEN3_VL_ACTIVATION_LAYERS=(0,), __name__=__name__,
                 LLM_TOKEN_INDICATOR=3, OUTPUT_IMAGE_INDICATOR=2,
                 SEQUENCE_PADDING_INDICATOR=-1)
    names = ["Ideogram4Config", "_rotate_half", "_apply_rotary_pos_emb",
             "Ideogram4MRoPE", "Ideogram4RMSNorm", "_build_flash_meta",
             "Ideogram4Attention", "Ideogram4MLP", "Ideogram4TransformerBlock",
             "_sinusoidal_embedding", "Ideogram4EmbedScalar",
             "Ideogram4FinalLayer", "Ideogram4Transformer2DModel"]
    native_definitions("extensions_built_in/diffusion_models/ideogram4/src/transformer.py",
                       names, scope)
    config = scope["Ideogram4Config"](
        emb_dim=12, num_layers=2, num_heads=2, intermediate_size=16,
        adanln_dim=8, in_channels=4, llm_features_dim=6, mrope_section=(1, 1, 1))
    return scope["Ideogram4Transformer2DModel"](config)


def fixture_backend():
    backend = Ideogram4Backend.__new__(Ideogram4Backend)
    backend._branch = ContextVar("checkpoint_test_branch", default=None)
    backend._diagnostic = ContextVar("checkpoint_test_diagnostic", default=None)
    backend.diffusion_network = FixtureNetwork()
    backend.gates = CubicTimeGates(2)
    transformer = native_transformer().eval().requires_grad_(False)
    backend.model = SimpleNamespace(transformer=transformer,
        unconditional_lora=SimpleNamespace(is_active=False))
    adapter_class = make_gated_lora_class(native_lora_class(), backend._branch, backend._diagnostic)
    adapters = []
    for block_id, block in enumerate(transformer.layers):
        for projection in DIFFUSION_PROJECTIONS:
            adapter = adapter_class(f"b{block_id}_{projection}", block.get_submodule(projection),
                                    lora_dim=2, alpha=4, network=backend.diffusion_network)
            adapter.gen2_block_id = block_id
            adapter.apply_to()
            with torch.no_grad():
                adapter.lora_up.weight.normal_(std=.1)
            adapters.append(adapter)
    return backend, transformer, adapters


def worker_grad(loss, params, *, retain_graph=False):
    """Do not copy the caller's Context: CUDA replay cannot rely on it either."""
    def run():
        return torch.autograd.grad(loss, params, retain_graph=retain_graph)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(Context().run, run).result()


class CheckpointContextTests(unittest.TestCase):
    def test_disposable_replay_contexts_leave_no_worker_bindings_after_reuse_or_failure(self):
        def worker():
            ambient = ContextVar("preserved_worker_binding")
            ambient.set("unchanged")
            before = dict(copy_context().items())
            contexts = []
            for _ in range(100):
                context = recomputation_context()
                contexts.append(context)
                with context:
                    self.assertTrue(RECOMPUTING.get())
                    with context:
                        self.assertTrue(RECOMPUTING.get())
                    with self.assertRaisesRegex(RuntimeError, "nested replay failed"):
                        with context:
                            raise RuntimeError("nested replay failed")
                    self.assertTrue(RECOMPUTING.get())
                self.assertFalse(RECOMPUTING.get())
                with context:
                    self.assertTrue(RECOMPUTING.get())
                self.assertFalse(RECOMPUTING.get())
                # Retain every context object to avoid relying on collection;
                # completed replay must immediately remove its private key.
                self.assertEqual(dict(copy_context().items()), before)
            self.assertEqual(len(contexts), 100)
        Context().run(worker)

    def test_native_default_checkpoint_fallback_still_matches_eager(self):
        torch.manual_seed(7)
        model = native_transformer().eval().requires_grad_(False)
        features = torch.randn(2, 4, 6, requires_grad=True)
        arguments = dict(llm_features=features, x=torch.randn(2, 4, 4),
            t=torch.tensor([.8, .2]), position_ids=torch.arange(4).reshape(1, 4, 1).expand(2, 4, 3),
            segment_ids=torch.ones(2, 4, dtype=torch.long),
            indicator=torch.tensor([[3, 3, 2, 2], [3, 3, 2, 2]]))
        expected = model(**arguments)
        reference = torch.autograd.grad(expected.square().mean(), features)
        model.enable_gradient_checkpointing()
        self.assertFalse(hasattr(model, "_gradient_checkpointing_func"))
        actual = model(**arguments)
        gradients = worker_grad(actual.square().mean(), (features,))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(gradients, reference)

    def test_raw_checkpoint_reproduces_missing_branch_with_parent_scope_open(self):
        branch = ContextVar("unwrapped_repro", default=None)
        diagnostic = ContextVar("unwrapped_diag", default=None)
        network = FixtureNetwork()
        linear = nn.Linear(3, 2, bias=False).requires_grad_(False)
        adapter = make_gated_lora_class(native_lora_class(), branch, diagnostic)(
            "repro", linear, lora_dim=2, alpha=2, network=network)
        adapter.gen2_block_id = 0
        adapter.apply_to()
        x = torch.randn(2, 4, 3, requires_grad=True)
        state = DiffusionBranch(torch.tensor([.2, .8]), torch.ones(2, 1), True, 1., False, "styled")
        token = branch.set(state)
        try:
            loss = checkpoint(linear, x, use_reentrant=False).square().mean()
            with self.assertRaisesRegex(RuntimeError, "requires an explicit branch context"):
                worker_grad(loss, (x,))
            self.assertIs(branch.get(), state)
        finally:
            branch.reset(token)

    def test_native_checkpoint_matches_eager_d_a_g_gradients_in_fresh_workers(self):
        for phase in ("D", "A", "G"):
            with self.subTest(phase=phase):
                torch.manual_seed(91)
                backend, model, adapters = fixture_backend()
                parameters = [p for adapter in adapters for p in adapter.parameters()]
                for parameter in parameters:
                    parameter.requires_grad_(phase == "D")
                backend.gates.beta.requires_grad_(phase == "G")
                with torch.no_grad():
                    backend.gates.beta.copy_(torch.tensor([[.2, -.3, .4, .1], [-.4, .1, .3, -.2]]))
                features = torch.randn(2, 4, 6, requires_grad=phase == "A")
                inputs = dict(llm_features=features, x=torch.randn(2, 4, 4),
                    t=torch.tensor([.8, .2]), position_ids=torch.arange(4).reshape(1, 4, 1).expand(2, 4, 3),
                    segment_ids=torch.ones(2, 4, dtype=torch.long),
                    indicator=torch.tensor([[3, 3, 2, 2], [3, 3, 2, 2]]))
                targets = parameters if phase == "D" else [features] if phase == "A" else [backend.gates.beta]
                tau = torch.tensor([.2, .8])
                gate_mode = "learned" if phase == "G" else "one"
                with backend.branch(tau, gate_mode=gate_mode):
                    expected = model(**inputs)
                    reference = torch.autograd.grad(expected.square().mean(), targets)
                self.assertGreater(sum(g.abs().sum().item() for g in reference), 0.)
                model.enable_gradient_checkpointing()
                model._gradient_checkpointing_func = backend._checkpoint
                seen = []
                with backend.diagnostics(lambda *args: seen.append(1)):
                    with backend.branch(tau, gate_mode=gate_mode):
                        actual = model(**inputs)
                        loss = actual.square().mean()
                    # Capture the original routing, not these ambient flags.
                    backend.diffusion_network.is_active = False
                    backend.model.unconditional_lora.is_active = True
                    if len(targets) == 1:
                        # A and G diagnostic chunks can revisit the same graph.
                        first = worker_grad(loss, targets, retain_graph=True)
                        gradients = worker_grad(loss, targets)
                        torch.testing.assert_close(first, gradients)
                    else:
                        split = len(targets)//2
                        gradients = (worker_grad(loss, targets[:split], retain_graph=True)
                                     + worker_grad(loss, targets[split:]))
                    self.assertFalse(backend.diffusion_network.is_active)
                    self.assertTrue(backend.model.unconditional_lora.is_active)
                    self.assertIsNone(backend.current_branch)
                    self.assertFalse(RECOMPUTING.get())
                torch.testing.assert_close(actual, expected)
                for expected_grad, actual_grad in zip(reference, gradients):
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-6)
                self.assertEqual(len(seen), len(adapters))
                # Ordinary training uses backward(), not autograd.grad(). It
                # must accumulate the same gradients through the worker replay.
                with backend.branch(tau, gate_mode=gate_mode):
                    training_loss = model(**inputs).square().mean()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(Context().run, training_loss.backward).result()
                for expected_grad, target in zip(reference, targets):
                    torch.testing.assert_close(target.grad, expected_grad, rtol=2e-5, atol=2e-6)
                self.assertTrue(all(parameter.grad is None for parameter in parameters
                                    if not parameter.requires_grad))

    def test_frozen_teacher_replay_keeps_adapters_off_under_styled_ambient_scope(self):
        torch.manual_seed(23)
        backend, _, adapters = fixture_backend()
        linear = adapters[0].orig_module_ref()
        x = torch.randn(2, 4, linear.in_features, requires_grad=True)
        expected = adapters[0].org_forward(x)
        reference = torch.autograd.grad(expected.square().mean(), x)
        seen = []
        tau = torch.tensor([.2, .8])
        with backend.diagnostics(lambda *args: seen.append(1)):
            with backend.branch(tau, lora_enabled=False, name="teacher"):
                actual = backend._checkpoint(linear, x, use_reentrant=False)
            with backend.branch(tau, gate_mode="learned", name="ambient_styled") as ambient:
                gradients = worker_grad(actual.square().mean(), (x,))
                self.assertIs(backend.current_branch, ambient)
                self.assertTrue(backend.diffusion_network.is_active)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(gradients, reference)
        self.assertEqual(seen, [])
        self.assertTrue(all(parameter.grad is None for adapter in adapters for parameter in adapter.parameters()))
        self.assertIsNone(backend.gates.beta.grad)

    def test_replay_exception_restores_worker_context_and_network_flags(self):
        backend, _, _ = fixture_backend()
        x = torch.randn(2, 3, requires_grad=True)
        tau = torch.tensor([.2, .8])
        captured = None
        def operation(value):
            self.assertIs(backend.current_branch, captured)
            self.assertTrue(backend.diffusion_network.is_active)
            self.assertFalse(backend.model.unconditional_lora.is_active)
            if RECOMPUTING.get():
                self.assertIsNone(backend._diagnostic.get())
                raise RuntimeError("intentional replay failure")
            return value.sin()*value
        with backend.branch(tau, name="original_styled") as captured:
            loss = backend._checkpoint(operation, x).sum()
        def worker():
            observer = lambda *args: None
            with backend.branch(tau, lora_enabled=False, name="worker_ambient") as ambient:
                backend.model.unconditional_lora.is_active = True
                with backend.diagnostics(observer):
                    diagnostic = backend._diagnostic.get()
                    with self.assertRaisesRegex(RuntimeError, "intentional replay failure"):
                        torch.autograd.grad(loss, x)
                    self.assertIs(backend.current_branch, ambient)
                    self.assertIs(backend._diagnostic.get(), diagnostic)
                    self.assertFalse(RECOMPUTING.get())
                    self.assertFalse(backend.diffusion_network.is_active)
                    self.assertTrue(backend.model.unconditional_lora.is_active)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(Context().run, worker).result()
        self.assertIsNone(backend.current_branch)
        self.assertIsNone(backend._diagnostic.get())
        self.assertFalse(RECOMPUTING.get())
        self.assertTrue(backend.diffusion_network.is_active)
        self.assertFalse(backend.model.unconditional_lora.is_active)

    def test_isolated_gradient_probe_chunks_replay_in_workers_without_state_mutation(self):
        torch.manual_seed(39)
        backend, _, adapters = fixture_backend()
        adapter = adapters[0]
        linear = adapter.orig_module_ref()
        parameters = list(adapter.parameters())
        x = torch.randn(2, 4, linear.in_features)
        tau = torch.tensor([.2, .8])
        for parameter in parameters:
            parameter.requires_grad_(False)
            parameter.grad = torch.full_like(parameter, .125)
        saved_gradients = [parameter.grad for parameter in parameters]
        saved_parameters = [parameter.detach().clone() for parameter in parameters]
        def objective(scale):
            @contextmanager
            def context():
                with backend.branch(tau, name="probe"):
                    yield backend._checkpoint(linear, x).square().mean()*scale
            return context
        # Enough space for exact coordinates plus one full parameter gradient,
        # but not both. This exercises the real diagnostic chunk planner.
        coordinate_bytes = sum(parameter.numel() for parameter in parameters)*40
        largest_gradient = max(parameter.numel()*parameter.element_size() for parameter in parameters)
        budget_mb = (coordinate_bytes+largest_gradient)/(1024*1024)
        real_grad = torch.autograd.grad
        def threaded_grad(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(Context().run, lambda: real_grad(*args, **kwargs)).result()
        with patch("torch.autograd.grad", side_effect=threaded_grad):
            report = isolated_gradient_probe({"first": objective(1.), "second": objective(2.)},
                {"diffusion": parameters}, max_coordinates=0, memory_budget_mb=budget_mb)
        self.assertGreater(report["working_gradient_chunks"], 1)
        pair = report["families"]["diffusion"]["pairs"]["first:second"]
        self.assertAlmostEqual(pair["cosine"], 1., places=6)
        self.assertAlmostEqual(pair["norm_ratio_second_first"], 2., places=6)
        for parameter, saved_gradient, saved_parameter in zip(parameters, saved_gradients, saved_parameters):
            self.assertFalse(parameter.requires_grad)
            self.assertIs(parameter.grad, saved_gradient)
            torch.testing.assert_close(parameter, saved_parameter)
        self.assertIsNone(backend.current_branch)

    def test_encoder_checkpoint_replay_can_be_reentered_for_probe_chunks(self):
        torch.manual_seed(19)
        encoder = fixture_encoder()
        inputs = torch.randn(1, 6, 4, requires_grad=True)
        mask = torch.ones(1, 6, dtype=torch.long)
        positions = mask.cumsum(-1)-1
        suffix = positions >= 4
        parameters = [inputs]+list(encoder.language_model.parameters())
        def run(checkpointed, diagnostic=None):
            features, rt, _, _ = differentiable_qwen_features(
                encoder, inputs, mask, positions, suffix,
                gradient_checkpointing=checkpointed, activation_layers=(0, 1, 2),
                causal_mask_factory=lambda **kwargs: None, diagnostic=diagnostic)
            return features.square().mean()+rt.mean()
        reference = torch.autograd.grad(run(False), parameters)
        seen = []
        loss = run(True, lambda *args: seen.append(1))
        split = len(parameters)//2
        gradients = (worker_grad(loss, parameters[:split], retain_graph=True)
                     + worker_grad(loss, parameters[split:]))
        for expected, actual in zip(reference, gradients):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(len(seen), 3)
        self.assertFalse(RECOMPUTING.get())


if __name__ == "__main__":
    unittest.main()
