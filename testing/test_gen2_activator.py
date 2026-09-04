import unittest

import torch
from torch import nn

from extensions_built_in.gen2_trainer.activator import (
    PlaceholderContract,
    SoftTokenBank,
    discover_qwen_down_proj_modules,
    install_qwen_down_proj_lora,
    load_qwen_down_proj_lora,
    pack_qwen_activation_features,
    replace_token_spans_with_soft_tokens,
    trigger_mask_context,
)


class Gen2ActivatorTest(unittest.TestCase):
    def test_multiple_occurrence_json_replacement(self):
        contract = PlaceholderContract()
        raw = '{"a":"x [trigger] y", "nested":["[trigger]"]}'
        _, occurrences = contract.parse(raw)
        self.assertEqual(len(occurrences), 2)
        replaced = contract.replace(raw, "style, with punctuation")
        self.assertNotIn("[trigger]", replaced)
        self.assertIn("style, with punctuation", replaced)

    def test_activation_feature_packing_interleaves_layers_per_hidden_dimension(self):
        first = torch.tensor([[[1.0, 2.0, 3.0]]])
        second = torch.tensor([[[10.0, 20.0, 30.0]]])
        packed = pack_qwen_activation_features({0: first, 1: second}, (0, 1))
        self.assertTrue(torch.equal(packed, torch.tensor([[[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]]])))

    def test_qwen_module_lora_discovers_and_installs_independent_layer_adapters(self):
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.down_proj = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                return self.down_proj(x)

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MLP()

            def forward(self, x):
                return self.mlp(x)

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = nn.Module()
                self.language_model.layers = nn.ModuleList([Layer(), Layer()])

        encoder = Encoder()
        modules = discover_qwen_down_proj_modules(encoder)
        bank = install_qwen_down_proj_lora(encoder, rank=2, alpha=2)
        self.assertEqual(set(modules), {0, 1})
        self.assertIsNot(bank.adapters["0"], bank.adapters["1"])
        bank.adapters["0"].up.weight.data.fill_(1.0)
        bank.adapters["1"].up.weight.data.zero_()
        hidden = torch.ones(1, 3, 4)
        with trigger_mask_context(torch.tensor([[1, 0, 1]])):
            output = encoder.language_model.layers[0].mlp.down_proj(hidden)
        self.assertTrue(torch.equal(output[:, 1], encoder.language_model.layers[0].mlp.down_proj(hidden)[:, 1]))
        self.assertTrue(torch.count_nonzero(output[:, [0, 2]] - hidden[:, [0, 2]]) > 0)
        bank.remove()

    def test_qwen_module_lora_disabled_is_strict_noop_and_rejects_load(self):
        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Module()
                self.mlp.down_proj = nn.Linear(4, 4, bias=False)

        encoder = nn.Module()
        encoder.language_model = nn.Module()
        encoder.language_model.layers = nn.ModuleList([Layer()])
        self.assertIsNone(install_qwen_down_proj_lora(encoder, enabled=False))
        with self.assertRaises(RuntimeError):
            load_qwen_down_proj_lora(None, {"unexpected": torch.ones(1)})

    def test_soft_token_span_length(self):
        bank = SoftTokenBank(3, 4, torch.ones(3, 4, dtype=torch.float32))
        embeddings = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
        expanded, spans = replace_token_spans_with_soft_tokens(embeddings, [(1, 3), (4, 5)], bank)
        self.assertEqual(expanded.shape[0], 6 - 3 + 2 * 3)
        self.assertEqual([end - start for start, end in spans], [3, 3])

    def test_soft_tokens_follow_input_dtype_and_keep_gradient(self):
        bank = SoftTokenBank(3, 4, torch.ones(3, 4, dtype=torch.float32))
        embeddings = torch.zeros(6, 4, dtype=torch.bfloat16)
        expanded, _ = replace_token_spans_with_soft_tokens(embeddings, [(1, 2)], bank)
        self.assertEqual(expanded.dtype, torch.bfloat16)
        expanded.sum().backward()
        self.assertIsNotNone(bank.A.grad)

    def test_qwen_position_embeddings_are_explicitly_dtype_aligned(self):
        source = __import__("pathlib").Path(__file__).parents[1] / "extensions_built_in" / "gen2_trainer" / "activator.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn("position_embeddings = tuple(item.to(dtype=inputs_embeds.dtype)", text)
        self.assertIn("position_embeddings = position_embeddings.to(dtype=inputs_embeds.dtype)", text)

    def test_missing_placeholder_fails(self):
        with self.assertRaises(ValueError):
            PlaceholderContract().replace('{"a":"plain"}', "style")


if __name__ == "__main__":
    unittest.main()
