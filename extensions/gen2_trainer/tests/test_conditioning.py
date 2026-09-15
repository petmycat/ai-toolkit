import copy
import unittest
from types import SimpleNamespace
from contextvars import ContextVar
from unittest.mock import patch

import torch
from torch import nn

from extensions.gen2_trainer.conditioning import (LearnedTokenBank, compile_trigger,
    normalized_vectors, make_masked_lora_class, native_lora_residual)
from extensions.gen2_trainer.backend_ideogram4 import Ideogram4Backend, differentiable_qwen_features, pack_activation_taps
from extensions.gen2_trainer.objectives import text_adapter_ratio


class FixtureNativeLoRA(nn.Module):
    """Small mathematical fixture, not a substitute for native backend acceptance."""
    def __init__(self, linear, rank=2, alpha=2):
        super().__init__()
        self.org_forward = linear.forward
        self.lora_down = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, linear.out_features, bias=False)
        self.scale = alpha/rank
        self.network = SimpleNamespace(is_active=True)
        nn.init.normal_(self.lora_up.weight, std=.1)

    def network_ref(self):
        return self.network

    def _call_forward(self, x):
        return self.lora_up(self.lora_down(x))*self.scale


class FixtureLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(d, d, bias=False)
        self.adapter = make_masked_lora_class(FixtureNativeLoRA)(self.mlp.down_proj)
        self.mlp.down_proj.forward = self.adapter.forward

    def forward(self, x, **kwargs):
        # Strict causal context: each position reads only its own prefix.
        prefix = x.cumsum(1)/torch.arange(1, x.shape[1]+1, device=x.device).reshape(1, -1, 1)
        return x+self.mlp.down_proj(prefix.tanh())


def fixture_encoder():
    language = nn.Module()
    language.layers = nn.ModuleList([FixtureLayer(4) for _ in range(3)])
    language.config = SimpleNamespace()
    language.rotary_emb = lambda inputs, positions: (positions, positions)
    return SimpleNamespace(language_model=language, device=torch.device("cpu"))


class ConditioningMathTests(unittest.TestCase):
    def test_literal_compiler_preserves_inner_whitespace(self):
        value = compile_trigger("  A  <s> blue\n chair [trigger] <S> ", "<s>")
        self.assertEqual(value["q"], "A   blue\n chair  <S>")
        self.assertTrue(value["trigger_present"])

    def test_normalization_analytic_jacobian_and_finite_difference(self):
        raw = torch.tensor([[.3, -.8, .5]], dtype=torch.float64, requires_grad=True)
        r = torch.tensor([2.3], dtype=torch.float64)
        actual = torch.autograd.functional.jacobian(lambda u: normalized_vectors(u, r), raw)[0, :, 0, :]
        u = raw.detach()[0]; norm = u.norm(); direction = u/norm
        expected = r[0]/norm*(torch.eye(3, dtype=torch.float64)-direction[:, None]*direction[None, :])
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        delta = torch.tensor([[.2, -.3, .7]], dtype=torch.float64)
        finite = (normalized_vectors(raw+1e-6*delta, r)-normalized_vectors(raw-1e-6*delta, r))/2e-6
        torch.testing.assert_close(finite[0], expected@delta[0], rtol=1e-8, atol=1e-9)
        with self.assertRaises(FloatingPointError):
            normalized_vectors(torch.zeros(1, 3), torch.ones(1))

    def test_initializer_rng_norm_and_saved_views(self):
        vocabulary = torch.randn(5, 6)
        before = vocabulary.clone(); rng = torch.random.get_rng_state().clone()
        tokens = LearnedTokenBank(vocabulary[[1, 3]], [1, 3], seed=7)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        torch.testing.assert_close(vocabulary, before)
        with torch.no_grad():
            tokens.U.add_(.5)
        torch.testing.assert_close(tokens().norm(dim=-1), tokens.r)
        clone = LearnedTokenBank(vocabulary[[1, 3]], [1, 3], seed=8)
        clone.load_state_dict(tokens.state_dict())
        torch.testing.assert_close(clone(), tokens())
        torch.testing.assert_close(clone("init"), tokens.e_init)

    def test_unique_tag_feature_packing_and_padding(self):
        taps = [torch.arange(4).reshape(1, 1, 4).expand(1, 3, 4)*100+j for j in range(3)]
        packed = pack_activation_taps(taps, torch.tensor([[1, 1, 0]]))
        self.assertEqual(packed[0, 0].tolist(), [0, 1, 2, 100, 101, 102, 200, 201, 202, 300, 301, 302])
        self.assertEqual(packed[0, 2].sum().item(), 0)

    def test_causal_prefix_and_checkpointed_rt_gradients(self):
        torch.manual_seed(4)
        encoder = fixture_encoder()
        encoder.device = torch.device("cpu")
        inputs = torch.randn(1, 6, 4, requires_grad=True)
        mask = torch.ones(1, 6, dtype=torch.long)
        positions = mask.cumsum(-1)-1
        suffix = positions >= 4
        def run(x, enabled=True, ckpt=False):
            return differentiable_qwen_features(encoder, x, mask, positions, suffix,
                adapter_enabled=enabled, gradient_checkpointing=ckpt, activation_layers=(0, 1, 2),
                causal_mask_factory=lambda **kwargs: None)
        reference = run(inputs, False)[0]
        observed, rt, _, _ = run(inputs)
        torch.testing.assert_close(observed[:, :4], reference[:, :4], rtol=0, atol=0)
        changed = inputs.detach().clone(); changed[:, 4:] += 20
        torch.testing.assert_close(run(changed)[0][:, :4], reference[:, :4], rtol=0, atol=0)
        # A deliberately incorrect unmasked adapter alters original features.
        from extensions.gen2_trainer.conditioning import encoder_projection_scope
        with encoder_projection_scope(torch.ones_like(suffix), True):
            wrong = encoder.language_model.layers[0](inputs)
        self.assertFalse(torch.equal(wrong[:, :4], reference[:, :4, 0::3]))
        params = [inputs]+list(encoder.language_model.parameters())
        g1 = torch.autograd.grad(observed.square().mean()+rt.mean(), params, allow_unused=True)
        observed2, rt2, _, _ = run(inputs, ckpt=True)
        g2 = torch.autograd.grad(observed2.square().mean()+rt2.mean(), params, allow_unused=True)
        torch.testing.assert_close(rt2, rt)
        for first, second in zip(g1, g2):
            if first is not None:
                torch.testing.assert_close(first, second)

    def test_regularizer_detaches_only_denominator(self):
        base = torch.ones(1, 3, 2, requires_grad=True)*2
        residual = torch.ones(1, 3, 2, requires_grad=True)
        ratio, numerator, denominator = text_adapter_ratio(base, residual, torch.tensor([[False, True, True]]))
        gbase, gres = torch.autograd.grad(ratio.sum(), (base, residual), allow_unused=True)
        self.assertIsNone(gbase)
        self.assertGreater(gres[:, 1:].sum().item(), 0)
        self.assertEqual(gres[:, :1].sum().item(), 0)

    def test_native_residual_compute_precision_matches_train_and_inference(self):
        adapter = FixtureNativeLoRA(nn.Linear(4, 3))
        inputs = torch.randn(1, 3, 4, dtype=torch.bfloat16, requires_grad=True)
        inference = native_lora_residual(adapter, inputs, torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            training = native_lora_residual(adapter, inputs, torch.bfloat16)
        torch.testing.assert_close(training, inference, rtol=0, atol=0)
        self.assertEqual(adapter.lora_down.weight.dtype, torch.float32)
        training.float().square().mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertIsNotNone(adapter.lora_down.weight.grad)

    def test_native_serialization_suffix_positions_and_overflow(self):
        encoder = fixture_encoder()
        encoder.language_model.embed_tokens = nn.Embedding(12, 4)
        class FixtureTokenizer:
            length = 4
            def apply_chat_template(self, messages, add_generation_prompt, tokenize):
                self.last_messages = messages
                assert add_generation_prompt and not tokenize
                return "serialized-with-generation-prompt"
            def __call__(self, text, add_special_tokens, truncation):
                assert not add_special_tokens and not truncation
                return {"input_ids": list(range(self.length))}
        tokenizer = FixtureTokenizer()
        backend = Ideogram4Backend.__new__(Ideogram4Backend)
        backend.trigger_word = "<s>"
        backend.encoder_checkpointing = False
        backend._diagnostic = ContextVar("fixture_encode_diag", default=None)
        backend._branch = ContextVar("fixture_encode_branch", default=None)
        backend.model = SimpleNamespace(text_encoder=encoder, tokenizer=tokenizer,
                                        max_text_length=6, torch_dtype=torch.float32, device_torch=torch.device("cpu"))
        backend.tokens = LearnedTokenBank(encoder.language_model.embed_tokens.weight[[1]], [1], num_tokens=2)
        saved_vocabulary = encoder.language_model.embed_tokens.weight.detach().clone()
        def fixture_bridge(*args, **kwargs):
            return differentiable_qwen_features(*args, **kwargs, activation_layers=(0, 1, 2),
                                               causal_mask_factory=lambda **_: None)
        with patch("extensions.gen2_trainer.backend_ideogram4.differentiable_qwen_features", fixture_bridge):
            condition = backend.encode([" A  <s> chair "], styled=True, gradients=True)
        meta = condition.metadata[0]
        self.assertEqual(meta["q"], "A   chair")
        self.assertEqual(meta["original_ids"], [0, 1, 2, 3])
        self.assertEqual(meta["suffix_positions"], [4, 5])
        self.assertEqual(meta["suffix_mask"], [False, False, False, False, True, True])
        self.assertEqual(condition.features[0].shape[0], 6)
        torch.testing.assert_close(saved_vocabulary, encoder.language_model.embed_tokens.weight)
        tokenizer.length = 5
        for styled in (False, True):
            with self.assertRaisesRegex(ValueError, "original_length=5, M=2, limit=6"):
                backend.encode(["caption"], styled=styled)


if __name__ == "__main__":
    unittest.main()
