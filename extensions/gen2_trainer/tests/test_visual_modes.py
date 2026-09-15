"""Exercise real Gen2 sampling orchestration with a tiny recording backend.

These CPU tests source-isolate the native sigma schedule and replace only the
unavailable diffusers random-tensor utility. They do not establish real-model
visual quality or production CUDA/quantization acceptance.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from extensions.gen2_trainer.backend_ideogram4 import Ideogram4Backend
from extensions.gen2_trainer.conditioning import Conditioning, LearnedTokenBank, compile_trigger
from extensions.gen2_trainer.gates import CubicTimeGates
from extensions.gen2_trainer.inference import generate, resolve_route
from extensions.gen2_trainer.tests.test_backend_contract import native_definitions


SIX_MODES = ("base_with_tokens", "base_with_conditioning", "encoder_adapter_off",
             "full", "full_uncond_half", "full_uncond_full")
EXPECTED = ((False, False, 0.), (False, True, 0.), (True, False, 0.),
            (True, True, 0.), (True, True, .5), (True, True, 1.))


@contextmanager
def native_sampling_utilities():
    scope = {"torch": torch, "math": math, "_LOGSNR_MIN": -15., "_LOGSNR_MAX": 18.}
    native_definitions("extensions_built_in/diffusion_models/ideogram4/src/pipeline.py",
                       ["_logit_normal_schedule", "get_ideogram4_sigmas"], scope)
    pipeline = ModuleType("extensions_built_in.diffusion_models.ideogram4.src.pipeline")
    pipeline.get_ideogram4_sigmas = scope["get_ideogram4_sigmas"]
    random_utils = ModuleType("diffusers.utils.torch_utils")
    random_utils.randn_tensor = lambda shape, *, generator, device, dtype: torch.randn(
        shape, generator=generator, device="cpu", dtype=dtype).to(device)
    with patch.dict(sys.modules, {pipeline.__name__: pipeline, random_utils.__name__: random_utils}):
        yield


class RecordingBackend(Ideogram4Backend):
    """Real branch contexts and gates around a deterministic scalar predictor."""
    def __init__(self, fail_on=None):
        self._branch = ContextVar(f"visual_branch_{id(self)}", default=None)
        self._diagnostic = ContextVar(f"visual_diagnostic_{id(self)}", default=None)
        self.gates = CubicTimeGates(1)
        self.tokens = LearnedTokenBank(torch.tensor([[.3, .4]]), [1], num_tokens=2)
        self.diffusion_network = nn.Linear(1, 1, bias=False)
        self.diffusion_network.is_active = True
        self.text_network = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.diffusion_network.weight.fill_(.09)
            self.text_network.weight.fill_(.03)
            self.gates.beta.copy_(torch.tensor([[.1, .2, .3, .4]]))
        self.model = SimpleNamespace(vae_scale_factor=1, patch_size=1,
            transformer=SimpleNamespace(config=SimpleNamespace(in_channels=3)),
            model_config=SimpleNamespace(model_kwargs={}, unconditional_lora_path="fixed-unconditional.safetensors"),
            device_torch=torch.device("cpu"), torch_dtype=torch.float32,
            unconditional_lora=SimpleNamespace(is_active=False),
            decode_latents=lambda latents, **kwargs: latents)
        self.config = {"inference": {"lora_strength": 1., "missing_trigger_policy": "learned_neutral"}}
        self.trigger_word = "<style>"
        self.calls, self.encodes = [], []
        self.fail_on = fail_on

    def encode(self, qs, styled=True, gradients=False, token_mode="learned", adapter_enabled=True):
        self.encodes.append(dict(qs=qs, styled=styled, gradients=gradients,
                                token_mode=token_mode, adapter_enabled=adapter_enabled))
        effect = self.tokens(token_mode).sum()*.02 if styled else torch.tensor(0.)
        if styled and adapter_enabled:
            effect = effect+self.text_network.weight.sum()
        item = compile_trigger(qs[0], self.trigger_word)
        item.update(original_length=1, total_length=3 if styled else 1)
        return Conditioning([effect.expand(item["total_length"], 2)], [item], torch.zeros(1))

    def predict(self, latents, tau, conditioning):
        branch = self.current_branch
        if branch is None:
            raise AssertionError("Sampling escaped its branch context")
        self.calls.append(dict(name=branch.name, enabled=branch.enabled, strength=branch.strength,
            unconditional=branch.unconditional, tau=tau.clone(), gates=branch.gate_values.clone(),
            latents=latents.clone(), text_tokens=conditioning.features[0].shape[0],
            native_unconditional_adapter=self.model.unconditional_lora.is_active,
            requires_grad=torch.is_grad_enabled()))
        if branch.name == self.fail_on:
            raise RuntimeError("sampling fixture failure")
        value = latents*.01
        if conditioning.features[0].numel():
            value = value+conditioning.features[0].mean()
        if branch.enabled:
            value = value+self.diffusion_network.weight.sum()*branch.gate_values.mean()*branch.strength
        if self.model.unconditional_lora.is_active:
            value = value+.01
        return value


class VisualModesTests(unittest.TestCase):
    def test_six_routes_use_current_tokens_and_independent_unconditional_strengths(self):
        for mode, (diffusion, text_adapter, uncond_strength) in zip(SIX_MODES, EXPECTED):
            with self.subTest(mode=mode):
                route = resolve_route(mode, True)
                self.assertTrue(route.styled)
                self.assertEqual(route.token_mode, "learned")
                self.assertEqual(route.lora_enabled, diffusion)
                self.assertEqual(route.adapter_enabled, text_adapter)
                self.assertEqual(route.unconditional_lora_strength, uncond_strength)
        # Production trigger routing remains the established conditional-only route.
        self.assertEqual(resolve_route(None, True).mode, "full")
        self.assertEqual(resolve_route(None, True).unconditional_lora_strength, 0.)

    def test_sampler_six_way_comparison_preserves_state_noise_and_empty_unconditional(self):
        backend = RecordingBackend()
        snapshot = {family: {name: tensor.clone() for name, tensor in component.state_dict().items()}
                    for family, component in backend.components().items()}
        for component in backend.components().values():
            for parameter in component.parameters():
                parameter.grad = torch.full_like(parameter, .123)
        rng = torch.random.get_rng_state().clone()
        initial_latents, pixels = [], []
        with native_sampling_utilities():
            for mode, (diffusion, text_adapter, uncond_strength) in zip(SIX_MODES, EXPECTED):
                with self.subTest(mode=mode):
                    backend.calls.clear()
                    image, metadata = generate(backend, "a [trigger] cat [trigger]", mode,
                        width=2, height=2, seed=43, steps=3, guidance=4., strength=1.7)
                    pixels.append(image.tobytes())
                    self.assertEqual(metadata["compiler"]["q"], "a  cat")
                    self.assertEqual(len(backend.calls), 6)
                    initial_latents.append(backend.calls[0]["latents"])
                    for conditional, unconditional in zip(backend.calls[::2], backend.calls[1::2]):
                        self.assertEqual(conditional["enabled"], diffusion)
                        self.assertEqual(conditional["strength"], 1.7)
                        self.assertEqual(conditional["text_tokens"], 3)
                        self.assertFalse(conditional["native_unconditional_adapter"])
                        self.assertEqual(unconditional["enabled"], uncond_strength > 0)
                        self.assertEqual(unconditional["strength"], uncond_strength)
                        self.assertEqual(unconditional["text_tokens"], 0)
                        self.assertTrue(unconditional["native_unconditional_adapter"])
                        self.assertFalse(conditional["requires_grad"] or unconditional["requires_grad"])
                        torch.testing.assert_close(conditional["tau"], unconditional["tau"])
                        torch.testing.assert_close(conditional["latents"], unconditional["latents"])
                        if uncond_strength:
                            torch.testing.assert_close(conditional["gates"], unconditional["gates"])
                    self.assertEqual(metadata["conditional_embedding_enabled"], True)
                    self.assertEqual(metadata["conditional_text_adapter_enabled"], text_adapter)
                    self.assertEqual(metadata["conditional_lora_strength"], 1.7 if diffusion else 0.)
                    self.assertEqual(metadata["unconditional_lora_strength"], uncond_strength)
                    self.assertEqual(metadata["unconditional_lora_strength_kind"], "absolute")
                    self.assertTrue(metadata["unconditional_image_only"])
                    self.assertFalse(metadata["unconditional_embedding_enabled"])
                    self.assertFalse(metadata["unconditional_text_adapter_enabled"])
                    self.assertIsNone(backend.current_branch)
                    self.assertTrue(backend.diffusion_network.is_active)
                    self.assertFalse(backend.model.unconditional_lora.is_active)
        for initial in initial_latents[1:]:
            torch.testing.assert_close(initial, initial_latents[0], rtol=0, atol=0)
        self.assertEqual(len(set(pixels)), 6)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        for family, component in backend.components().items():
            for name, tensor in component.state_dict().items():
                torch.testing.assert_close(tensor, snapshot[family][name], rtol=0, atol=0)
            for parameter in component.parameters():
                self.assertTrue(parameter.requires_grad)
                torch.testing.assert_close(parameter.grad, torch.full_like(parameter, .123), rtol=0, atol=0)

    def test_explicit_noise_reused_and_prediction_exception_restores_both_networks(self):
        noise = torch.full((1, 3, 2, 2), .2)
        before = noise.clone()
        with native_sampling_utilities():
            for branch_name in ("cfg_conditional", "cfg_unconditional"):
                backend = RecordingBackend(fail_on=branch_name)
                with self.assertRaisesRegex(RuntimeError, "sampling fixture failure"):
                    generate(backend, "[trigger] cat", "full_uncond_half", width=2, height=2,
                             initial_noise=noise, guidance=4., steps=2)
                self.assertIsNone(backend.current_branch)
                self.assertTrue(backend.diffusion_network.is_active)
                self.assertFalse(backend.model.unconditional_lora.is_active)
            backend = RecordingBackend()
            first, first_meta = generate(backend, "[trigger] cat", "full_uncond_half", width=2,
                height=2, initial_noise=noise, seed=1, guidance=4., steps=2)
            second, _ = generate(backend, "[trigger] cat", "full_uncond_half", width=2,
                height=2, initial_noise=noise, seed=99, guidance=4., steps=2)
            self.assertEqual(first.tobytes(), second.tobytes())
            self.assertEqual(first_meta["initial_noise_source"], "explicit_tensor")
        torch.testing.assert_close(noise, before, rtol=0, atol=0)

    def test_unconditional_comparisons_require_cfg_and_legacy_routes_keep_native_switch(self):
        backend = RecordingBackend()
        with native_sampling_utilities():
            for guidance in (0., 1.):
                for mode in ("full_uncond_half", "full_uncond_full"):
                    with self.assertRaisesRegex(ValueError, "guidance >1"):
                        generate(backend, "[trigger] cat", mode, width=2, height=2, guidance=guidance)
            self.assertEqual(backend.encodes, [])
            _, metadata = generate(backend, "[trigger] cat", "full", width=2, height=2,
                                   guidance=1., steps=2)
            self.assertEqual(len(backend.calls), 2)
            self.assertFalse(metadata["unconditional_branch_executed"])
            self.assertTrue(all(call["name"] == "cfg_conditional" for call in backend.calls))


if __name__ == "__main__":
    unittest.main()
