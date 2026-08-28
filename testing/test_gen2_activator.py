import unittest

import torch

from extensions_built_in.gen2_trainer.activator import (
    PlaceholderContract,
    SoftTokenBank,
    replace_token_spans_with_soft_tokens,
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

    def test_soft_token_span_length(self):
        bank = SoftTokenBank(3, 4, torch.ones(3, 4))
        embeddings = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
        expanded, spans = replace_token_spans_with_soft_tokens(embeddings, [(1, 3), (4, 5)], bank)
        self.assertEqual(expanded.shape[0], 6 - 3 + 2 * 3)
        self.assertEqual([end - start for start, end in spans], [3, 3])

    def test_missing_placeholder_fails(self):
        with self.assertRaises(ValueError):
            PlaceholderContract().replace('{"a":"plain"}', "style")


if __name__ == "__main__":
    unittest.main()
