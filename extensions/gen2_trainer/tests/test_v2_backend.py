"""CPU contract checks; these fixtures do not certify actual model/CUDA acceptance."""
from contextlib import contextmanager
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
import unittest

import torch
from torch import nn

from extensions.gen2_trainer.v2.backend import V2Backend, qwen_features, EXPECTED_TAPS
from extensions.gen2_trainer.tests.test_v2_text import FixtureTokenizer
from extensions.gen2_trainer.tests.test_backend_contract import native_definitions


def causal_mask(**kwargs):
    return kwargs["attention_mask"]


class CausalLayer(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.01 + index*.0001))

    def forward(self, x, **kwargs):
        positions = torch.arange(1, x.shape[1]+1, device=x.device)[None, :, None]
        return x + self.scale*torch.tanh(x.cumsum(1)/positions)


class EncoderFixture(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.embed_tokens = nn.Embedding(len(tokenizer.get_vocab()), 6)
        self.language_model.layers = nn.ModuleList([CausalLayer(i) for i in range(36)])
        self.language_model.config = SimpleNamespace()
        self.language_model.rotary_emb = lambda inputs, positions: (positions, positions)


class TransformerFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.5))
        self.checkpointing = True
        self.calls = []

    def enable_gradient_checkpointing(self): self.checkpointing = True
    def disable_gradient_checkpointing(self): self.checkpointing = False
    def set_attention_backend(self, name): self.attention_backend = name


@contextmanager
def fixture_backend(modify_model=None):
    tokenizer = FixtureTokenizer()
    scope = {"torch": torch, "QWEN3_VL_ACTIVATION_LAYERS": EXPECTED_TAPS, "create_causal_mask": causal_mask}
    native_definitions("extensions_built_in/diffusion_models/ideogram4/src/pipeline.py",
                       ["get_qwen3_vl_features", "pad_text_features"], scope)
    def velocity(transformer, latents, tau, features, mask):
        transformer.calls.append((features.detach().clone(), mask.detach().clone(), torch.is_grad_enabled()))
        return latents*transformer.weight + (features.sum()/max(1, features.numel()))
    pipeline = ModuleType("pipeline")
    pipeline.get_qwen3_vl_features = scope["get_qwen3_vl_features"]
    pipeline.pad_text_features = scope["pad_text_features"]
    pipeline.predict_velocity = velocity
    transformer_module = ModuleType("transformer")
    transformer_module.QWEN3_VL_ACTIVATION_LAYERS = EXPECTED_TAPS
    model = SimpleNamespace(arch="ideogram4", network=None, unconditional_lora=None,
        transformer=TransformerFixture(), unconditional_transformer=TransformerFixture(),
        text_encoder=EncoderFixture(tokenizer), vae=nn.Linear(1, 1), tokenizer=tokenizer,
        device_torch=torch.device("cpu"), torch_dtype=torch.float32)
    if modify_model is not None:
        modify_model(model)
    config = {"trigger_word": "<s>", "model": {"model_kwargs": {"max_text_length": 3072}},
        "train": {"gradient_checkpointing": True}, "gen2": {
            "execution": {"encoder_gradient_checkpointing": True, "dit_attention_backend": "torch"},
            "conditioning": {"num_tokens": 4, "initializer_seed": 7,
                             "initialization_sample_size": 32, "overflow_policy": "truncate"},
            "evaluation": {"named_phrase": "named style"}}}
    def extraction(*args, **kwargs):
        kwargs["causal_mask_factory"] = causal_mask
        return qwen_features(*args, **kwargs)
    with patch.dict("sys.modules", {
        "extensions_built_in.diffusion_models.ideogram4.src.pipeline": pipeline,
        "extensions_built_in.diffusion_models.ideogram4.src.transformer": transformer_module,
    }), patch("extensions.gen2_trainer.v2.backend.qwen_features", side_effect=extraction):
        yield V2Backend(model, config)


class V2BackendTests(unittest.TestCase):
    def test_rejects_correction_adapter_and_missing_original_unconditional(self):
        cases = (
            (lambda model: setattr(model, "unconditional_lora", nn.Linear(1, 1)), "correction adapter"),
            (lambda model: setattr(model, "unconditional_transformer", None), "separately loaded"),
            (lambda model: setattr(model, "unconditional_transformer", model.transformer), "separately loaded"),
        )
        for change, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                with fixture_backend(change):
                    self.fail("Unsupported unconditional source was accepted")

    def test_original_models_frozen_and_only_free_token_parameter(self):
        with fixture_backend() as backend:
            self.assertEqual(backend.trigger_word, "<s>")
            self.assertFalse(any(p.requires_grad for p in backend.frozen_parameters()))
            self.assertIn(id(backend.unconditional.weight), {id(p) for p in backend.frozen_parameters()})
            self.assertTrue(backend.model.transformer.checkpointing)
            self.assertFalse(backend.unconditional.checkpointing)
            self.assertEqual([name for name, p in backend.tokens.named_parameters()], ["E"])
            backend.assert_frozen()
            backend.unconditional.weight.requires_grad_(True)
            with self.assertRaisesRegex(RuntimeError, "became trainable"):
                backend.assert_frozen()

    def test_separate_original_unconditional_empty_text_without_conditional_calls(self):
        with fixture_backend() as backend:
            latent = torch.ones(2, 3, 2, 2, requires_grad=True)
            output = backend.predict_unconditional(latent, torch.tensor([.3, .7]))
            self.assertFalse(output.requires_grad)
            self.assertEqual(len(backend.model.transformer.calls), 0)
            self.assertEqual(len(backend.unconditional.calls), 1)
            features, mask, grad = backend.unconditional.calls[0]
            self.assertEqual(features.shape, (2, 0, 6*13))
            self.assertEqual(mask.shape, (2, 0))
            self.assertFalse(grad)
            with torch.no_grad(): backend.tokens.E.mul_(9)
            torch.testing.assert_close(output, backend.predict_unconditional(latent, torch.tensor([.3, .7])))

    def test_all_thirteen_taps_match_native_packing_and_checkpoint_gradients(self):
        with fixture_backend() as backend:
            records = backend.verify_native(["ordinary scene", "x[trigger]y"], atol=0., rtol=0.)
            self.assertTrue(all(record["passed"] for record in records))
            caption = "front [trigger] tiger [trigger] room"
            results = []
            for enabled in (False, True):
                backend.encoder_checkpointing = enabled
                features = backend.encode([caption], gradients=True).features[0]
                gradient, = torch.autograd.grad(features[-5:].square().mean(), backend.tokens.E)
                results.append((features.detach(), gradient))
            for off, on in zip(results[0], results[1]):
                torch.testing.assert_close(off, on, atol=0., rtol=0.)
            self.assertGreater(results[1][1].abs().sum().item(), 0.)

    def test_conditional_gradient_reaches_tokens_through_later_ordinary_tokens(self):
        with fixture_backend() as backend:
            prior_grad = torch.randn_like(backend.tokens.E)
            backend.tokens.E.grad = prior_grad.clone()
            record = backend.probe_conditioning("[trigger] tiger [trigger] room")
            self.assertTrue(record["gradient_nonzero"] and record["gradient_finite"] and record["finite_features"])
            self.assertFalse(set(record["ordinary_positions"]) & set(record["excluded_soft_positions"]))
            torch.testing.assert_close(backend.tokens.E.grad, prior_grad)
            encoded = backend.encode(["[trigger] tiger"], gradients=True)
            value = backend.predict(torch.ones(1, 3, 2, 2), torch.tensor([.5]), encoded)
            gradient, = torch.autograd.grad(value.square().mean(), backend.tokens.E)
            self.assertGreater(gradient.abs().sum().item(), 0.)
            self.assertFalse(backend.unconditional.calls)
            self.assertTrue(all(p.grad is None for p in backend.frozen_parameters()))

    def test_marker_absent_route_exact_native_and_init_is_fixed(self):
        with fixture_backend() as backend:
            text = "ordinary scene"
            item = backend.compiler.compile(text)
            ids = torch.tensor([item.ids]); mask = torch.ones_like(ids)
            native = backend.native_features(backend.model.text_encoder, ids, mask, mask.cumsum(-1)-1)
            encoded = backend.encode([text])
            torch.testing.assert_close(native[0], encoded.features[0], atol=0., rtol=0.)
            initialized = backend.encode(["[trigger] tiger"], mode="init").features[0]
            with torch.no_grad(): backend.tokens.E.add_(1)
            torch.testing.assert_close(initialized, backend.encode(["[trigger] tiger"], mode="init").features[0])
            self.assertFalse(torch.equal(initialized, backend.encode(["[trigger] tiger"]).features[0]))


if __name__ == "__main__":
    unittest.main()
