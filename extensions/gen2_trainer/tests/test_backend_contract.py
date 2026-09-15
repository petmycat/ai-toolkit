"""Source-isolated native helper tests; real model/CUDA acceptance runs on the VM."""
import ast
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref
import math
from typing import List, Optional, Union, Dict, Type

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from extensions.gen2_trainer.backend_ideogram4 import (DiffusionBranch, Ideogram4Backend,
    _resolved_targets, make_gated_lora_class)
from extensions.gen2_trainer.gates import CubicTimeGates
from extensions.gen2_trainer.inference import resolve_route
from extensions.gen2_trainer.objectives import per_example_mse

ROOT = Path(__file__).resolve().parents[3]


def native_definitions(relative_path, names, namespace):
    """Execute exact named native definitions without importing unused optional deps."""
    source = ast.parse((ROOT/relative_path).read_text(encoding="utf-8"))
    definitions = [node for node in source.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in definitions} != set(names):
        raise AssertionError("The tested native helper/class moved or changed its public name")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future]+definitions, type_ignores=[]))
    exec(compile(tree, str(ROOT/relative_path), "exec"), namespace)
    return namespace


def native_lora_class():
    scope = {"torch": torch, "nn": nn, "weakref": weakref, "math": math,
             "CONV_MODULES": ["Conv2d"], "Optional": object, "List": list}
    native_definitions("toolkit/network_mixins.py", ["ToolkitModuleMixin", "ExtractableModuleMixin"], scope)
    native_definitions("toolkit/lora_special.py", ["LoRAModule"], scope)
    return scope["LoRAModule"]


class BackendContractTests(unittest.TestCase):
    def test_nested_diagnostic_suppression_restores_observer_after_exception(self):
        backend = Ideogram4Backend.__new__(Ideogram4Backend)
        backend._diagnostic = ContextVar("nested_diagnostics_test", default=None)
        observer = lambda *args: None
        with backend.diagnostics(observer):
            before = backend._diagnostic.get()
            with self.assertRaisesRegex(RuntimeError, "probe failed"):
                with backend.diagnostics(None):
                    self.assertIsNone(backend._diagnostic.get())
                    raise RuntimeError("probe failed")
            self.assertIs(backend._diagnostic.get(), before)
        self.assertIsNone(backend._diagnostic.get())

    def test_native_low_vram_placement_guard_and_graph_safety(self):
        class PlacementFixture:
            device = torch.device("cpu")
            def __init__(self): self.moves = []
            def to(self, target):
                self.moves.append(target)
                self.device = target
                return self
        backend = Ideogram4Backend.__new__(Ideogram4Backend)
        backend._branch = ContextVar("placement_branch", default=None)
        backend.model = SimpleNamespace(device_torch=torch.device("cuda:0"))
        component = PlacementFixture()
        backend._ensure_native_device(component, "transformer")
        self.assertEqual(component.moves, [torch.device("cuda:0")])
        backend._ensure_native_device(component, "transformer")
        self.assertEqual(len(component.moves), 1)
        state = DiffusionBranch(torch.tensor([.5]), torch.ones(1, 1), True, 1., False, "styled")
        state.started_components.add("transformer")
        backend._branch.set(state)
        component.device = torch.device("cpu")
        with self.assertRaisesRegex(RuntimeError, "may still recompute"):
            backend._ensure_native_device(component, "transformer")

    def test_native_network_target_and_configurable_alpha_seam(self):
        scope = native_lora_class().__init__.__globals__
        scope.update(List=List, Optional=Optional, Union=Union, Dict=Dict, Type=Type,
                     LINEAR_MODULES=["Linear"], CONV_MODULES=["Conv2d"])
        native_definitions("toolkit/network_mixins.py", ["ToolkitNetworkMixin"], scope)
        native_definitions("toolkit/kohya_lora.py", ["LoRANetwork"], scope)
        native_definitions("toolkit/lorm.py", ["count_parameters"], scope)
        native_definitions("toolkit/lora_special.py", ["LoRASpecialNetwork"], scope)
        class FixtureTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Module(), nn.Module()])
                for layer in self.layers:
                    layer.projection = nn.Linear(3, 4, bias=False)
                    layer.unadapted = nn.Linear(3, 4, bias=False)
        model = FixtureTransformer()
        keys = [f"transformer$$layers$${i}$$projection" for i in range(2)]
        network = scope["LoRASpecialNetwork"](None, model, lora_dim=2, alpha=6,
            train_text_encoder=False, train_unet=True, is_transformer=True,
            network_config=SimpleNamespace(all_layers=False),
            target_lin_modules=["FixtureTransformer"],
            only_if_contains=[f"layers.{i}.projection" for i in range(2)],
            modules_dim={key: 2 for key in keys}, modules_alpha={key: 6 for key in keys})
        self.assertEqual(len(network.unet_loras), 2)
        self.assertEqual({id(adapter.orig_module_ref()) for adapter in network.unet_loras},
                         {id(layer.projection) for layer in model.layers})
        for adapter in network.unet_loras:
            self.assertEqual(float(adapter.alpha), 6.)
            self.assertEqual(adapter.scale, 3.)

    def test_native_alpha_scaling_batch_gates_and_base_bypass(self):
        native = native_lora_class()
        branch_var, diagnostic_var = ContextVar("test_branch"), ContextVar("test_diag", default=None)
        class FixtureNetwork:
            network_type = "lora"
            is_active = True
        network = FixtureNetwork()
        linear = nn.Linear(3, 2, bias=False)
        module_class = make_gated_lora_class(native, branch_var, diagnostic_var)
        adapter = module_class("test", linear, lora_dim=2, alpha=6., network=network)
        adapter.gen2_block_id = 0
        adapter.apply_to()
        with torch.no_grad():
            adapter.lora_up.weight.fill_(.2)
        x = torch.randn(2, 4, 3, requires_grad=True)
        base = adapter.org_forward(x)
        native_residual = adapter._call_forward(x)
        manual = adapter.lora_up(adapter.lora_down(x))*3
        torch.testing.assert_close(native_residual, manual)
        token = branch_var.set(DiffusionBranch(torch.tensor([0., 1.]), torch.ones(2, 1), True, 1., False, "styled"))
        torch.testing.assert_close(linear(x), base+native_residual)
        branch_var.reset(token)

        # Actual PyTorch non-reentrant recomputation must preserve gates and emit
        # one activation row, even without our encoder-specific context_fn.
        calls = []
        diagnostic_var.set((lambda *args: calls.append(1), {}))
        beta = torch.tensor([[.1], [.3]], requires_grad=True)
        token = branch_var.set(DiffusionBranch(torch.tensor([0., 1.]), 1+.5*beta.tanh(), True, 1., False, "checkpointed"))
        x2 = x.detach().clone().requires_grad_(True)
        checkpoint(lambda value: linear(value), x2, use_reentrant=False).square().mean().backward()
        self.assertEqual(len(calls), 1)
        self.assertGreater(beta.grad.abs().sum().item(), 0)
        branch_var.reset(token)
        diagnostic_var.set(None)
        gates = torch.tensor([[.6], [1.4]], requires_grad=True)
        token = branch_var.set(DiffusionBranch(torch.tensor([0., 1.]), gates, True, 1., False, "styled"))
        torch.testing.assert_close(linear(x), base+native_residual*gates[:, None, :])
        linear(x).square().mean().backward()
        self.assertGreater(gates.grad.abs().sum().item(), 0.)
        self.assertGreater(x.grad.abs().sum().item(), 0.)
        branch_var.reset(token)
        token = branch_var.set(DiffusionBranch(torch.tensor([0., 1.]), gates, False, 1., False, "teacher"))
        torch.testing.assert_close(linear(x), base)
        branch_var.reset(token)

    def test_native_velocity_time_sign_and_padding(self):
        scope = {"torch": torch, "LLM_TOKEN_INDICATOR": 3, "OUTPUT_IMAGE_INDICATOR": 2,
                 "SEQUENCE_PADDING_INDICATOR": -1, "IMAGE_POSITION_OFFSET": 4096}
        native_definitions("extensions_built_in/diffusion_models/ideogram4/src/pipeline.py",
                           ["predict_velocity", "pad_text_features"], scope)
        class RecordingTransformer:
            def __call__(self, **kwargs):
                self.arguments = kwargs
                return torch.ones_like(kwargs["x"])*2
        transformer = RecordingTransformer()
        feature_list = [torch.ones(2, 5), torch.ones(3, 5)*2]
        features, mask = scope["pad_text_features"](feature_list, torch.device("cpu"), torch.float32)
        tau = torch.tensor([0., 1.])
        velocity = scope["predict_velocity"](transformer, torch.zeros(2, 128, 1, 2), tau, features, mask)
        torch.testing.assert_close(transformer.arguments["t"], 1-tau)
        torch.testing.assert_close(velocity, torch.full_like(velocity, -2.))
        self.assertEqual(mask.tolist(), [[1, 1, 0], [1, 1, 1]])
        self.assertEqual(transformer.arguments["indicator"][0].tolist(), [3, 3, 0, 2, 2])
        self.assertEqual(transformer.arguments["segment_ids"][0].tolist(), [1, 1, -1, 1, 1])
        z0, noise = torch.full_like(velocity, 3.), torch.full_like(velocity, 7.)
        target = noise-z0
        initial = noise
        final = initial+target*(0.-1.)
        torch.testing.assert_close(final, z0)

    def test_branch_exception_and_unconditional_isolation(self):
        backend = Ideogram4Backend.__new__(Ideogram4Backend)
        backend._branch = ContextVar("test_restoration", default=None)
        backend.gates = CubicTimeGates(2)
        backend.diffusion_network = SimpleNamespace(is_active=True)
        backend.model = SimpleNamespace(unconditional_lora=SimpleNamespace(is_active=False))
        with self.assertRaisesRegex(RuntimeError, "test exception"):
            with backend.branch(torch.tensor([.2]), lora_enabled=False, name="teacher"):
                self.assertFalse(backend.diffusion_network.is_active)
                self.assertFalse(backend.model.unconditional_lora.is_active)
                raise RuntimeError("test exception")
        self.assertTrue(backend.diffusion_network.is_active)
        self.assertIsNone(backend.current_branch)
        with backend.branch(torch.tensor([.4]), lora_enabled=False, unconditional=True):
            self.assertTrue(backend.model.unconditional_lora.is_active)
            self.assertFalse(backend.diffusion_network.is_active)
        self.assertFalse(backend.model.unconditional_lora.is_active)
        with self.assertRaises(ValueError):
            with backend.branch(torch.tensor([.4]), lora_enabled=False, gate_mode="invalid"):
                pass
        self.assertTrue(backend.diffusion_network.is_active)

    def test_target_identity_and_extra_or_duplicate_failures(self):
        model = nn.Module()
        model.layers = nn.ModuleList([nn.Module()])
        model.layers[0].projection = nn.Linear(3, 2)
        targets = _resolved_targets(model, [(0, "layers.0.projection")], "diffusion", 2, 2)
        self.assertEqual(list(targets.values())[0]["block_id"], 0)
        with self.assertRaises(ValueError):
            _resolved_targets(model, [(0, "layers.0.projection")]*2, "diffusion", 2, 2)
        with self.assertRaises(ValueError):
            _resolved_targets(model, [(0, "layers.0.projection")], "diffusion", 3, 3)

    def test_modes_do_not_mutate_components(self):
        full = resolve_route(None, True)
        self.assertTrue(full.styled and full.lora_enabled)
        neutral = resolve_route(None, False)
        self.assertFalse(neutral.styled)
        self.assertTrue(neutral.lora_enabled)
        self.assertFalse(resolve_route(None, False, "base_bypass").lora_enabled)
        self.assertTrue(resolve_route("neutral_lora_on", False, "base_bypass").lora_enabled)
        self.assertEqual(resolve_route("conditioning_init", True).token_mode, "init")
        self.assertFalse(resolve_route("conditioning_init", True).adapter_enabled)
        self.assertTrue(resolve_route("tokens_init", True).adapter_enabled)
        self.assertTrue(resolve_route("base_with_conditioning", True).styled)
        self.assertFalse(resolve_route("base_with_conditioning", True).lora_enabled)
        self.assertEqual(resolve_route("gates_time_mean", True).gate_mode, "time_mean")

    def test_fp32_equal_example_mse(self):
        pred = torch.tensor([[1., 5., 9.], [3., 5., 8.]], dtype=torch.bfloat16)
        target = torch.zeros_like(pred)
        mask = torch.tensor([[True, False, False], [True, True, False]])
        losses = per_example_mse(pred, target, mask)
        self.assertEqual(losses.dtype, torch.float32)
        torch.testing.assert_close(losses, torch.tensor([1., 17.]))


if __name__ == "__main__":
    unittest.main()
