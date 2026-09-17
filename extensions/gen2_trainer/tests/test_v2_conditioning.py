import unittest

import torch
from torch import nn

from extensions.gen2_trainer.v2.conditioning import TokenBank, replace_input_embeddings
from extensions.gen2_trainer.tests.test_v2_text import FixtureTokenizer


class V2ConditioningTests(unittest.TestCase):
    def test_numerical_initialization_is_isolated_reproducible_and_not_semantic(self):
        tokenizer = FixtureTokenizer(); embedding = nn.Embedding(len(tokenizer.get_vocab()), 6)
        original = embedding.weight.detach().clone(); state = torch.random.get_rng_state().clone()
        first = TokenBank(embedding, tokenizer, 4, 27, 64)
        second = TokenBank(embedding, tokenizer, 4, 27, 64)
        torch.testing.assert_close(first.E, second.E, atol=0, rtol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), state)
        torch.testing.assert_close(original, embedding.weight)
        self.assertFalse(set(first.sampled_vocabulary_ids.tolist()) & set(tokenizer.all_special_ids))
        self.assertFalse(torch.equal(first.E[0], first.E[1]))
        torch.testing.assert_close(first.E.norm(dim=-1), first.initial_typical_norm.expand(4))
        self.assertIsNone(first.provenance()["semantic_initializer"])

    def test_magnitude_is_free_and_initial_saved(self):
        tokenizer = FixtureTokenizer(); embedding = nn.Embedding(len(tokenizer.get_vocab()), 6)
        bank = TokenBank(embedding, tokenizer, 4, 27, 32)
        initial = bank.initial.clone()
        with torch.no_grad(): bank.E.mul_(3)
        torch.testing.assert_close(bank(), initial*3)
        torch.testing.assert_close(bank("init"), initial)
        other = TokenBank(embedding, tokenizer, 4, 90, 32)
        other.load_state_dict(bank.state_dict())
        torch.testing.assert_close(other(), bank())

    def test_repeated_inplace_replacement_accumulates_into_one_bank(self):
        embedding = nn.Embedding(10, 3).requires_grad_(False)
        bank = nn.Parameter(torch.randn(2, 3))
        ids = torch.tensor([[1, 0, 0, 4, 0, 0, 7]])
        output = replace_input_embeddings(embedding, ids, [1, 2, 4, 5], [0, 1, 0, 1], bank)
        output.sum().backward()
        torch.testing.assert_close(bank.grad, torch.ones_like(bank)*2)
        torch.testing.assert_close(output[0, [0, 3, 6]], embedding(ids)[0, [0, 3, 6]])
        self.assertIsNone(embedding.weight.grad)


if __name__ == "__main__":
    unittest.main()
