import unittest

import torch

from extensions_built_in.gen2_trainer.phase_a_math import (
    dataset_mse,
    detached_ema_prototype,
    effective_sample_weighted_reduce,
    helper_direction_cosine,
    response_field_signature,
)


class PhaseAMathTest(unittest.TestCase):
    def test_dataset_mse_and_effective_sample_weighting(self):
        prediction = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
        target = torch.zeros_like(prediction)
        self.assertAlmostEqual(float(dataset_mse(prediction, target)), 5.0)
        self.assertAlmostEqual(float(dataset_mse(prediction, target, torch.tensor([1.0, 0.0]))), 1.0)
        values = torch.tensor([1.0, 3.0])
        self.assertAlmostEqual(float(effective_sample_weighted_reduce(values, torch.tensor([1.0, 3.0]))), 2.5)

    def test_helper_direction_cosine_returns_validity(self):
        activator = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        helper = torch.tensor([[2.0, 0.0], [1.0, 0.0]])
        cosine, valid = helper_direction_cosine(activator, helper, torch.zeros_like(activator))
        self.assertAlmostEqual(float(cosine[0]), 1.0)
        self.assertTrue(bool(valid[0]))
        self.assertFalse(bool(valid[1]))

    def test_response_signature_handles_variable_resolution_fields(self):
        first = response_field_signature(torch.ones(1, 2, 3, 4))
        second = response_field_signature(torch.ones(1, 5, 7, 8))
        self.assertEqual(first.shape, (1, 2))
        self.assertEqual(second.shape, (1, 2))
        self.assertTrue(torch.isfinite(torch.cat((first, second), dim=0)).all())

    def test_detached_ema_prototype_is_stable_and_detached(self):
        first = detached_ema_prototype(None, torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 0.5)
        second = detached_ema_prototype(first, torch.tensor([[5.0, 6.0]]), 0.5)
        self.assertTrue(torch.equal(first, torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.equal(second, torch.tensor([3.5, 4.5])))
        self.assertFalse(second.requires_grad)


if __name__ == "__main__":
    unittest.main()
