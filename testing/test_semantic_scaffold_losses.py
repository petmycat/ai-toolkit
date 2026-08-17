import unittest

import torch

from toolkit.trigger_binding_losses import (
    broad_band_penalty,
    disturbance_cap_penalty,
    effect_direction_cosine,
    normalized_disturbance_beta,
    normalized_gain_vs_neutral,
    relative_residual_rms,
    smooth_progress_weight,
    soft_gain_floor,
)


class SemanticScaffoldLossTest(unittest.TestCase):
    def test_gain_and_soft_floor(self):
        gain = normalized_gain_vs_neutral(torch.tensor([1.0, 0.5]), torch.tensor([2.0, 1.0]))
        torch.testing.assert_close(gain, torch.tensor([0.5, 0.5]), atol=1.0e-6, rtol=1.0e-6)
        equal_gain = normalized_gain_vs_neutral(torch.tensor([2.0]), torch.tensor([2.0]))
        torch.testing.assert_close(equal_gain, torch.zeros_like(equal_gain))
        self.assertGreater(float(soft_gain_floor(gain, 0.75, 0.1).mean()), 0.0)

    def test_effect_cosine_detaches_teacher_and_masks_invalid(self):
        private = torch.tensor([[1.0, 0.0], [0.0, 0.0]], requires_grad=True)
        helper = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
        result = effect_direction_cosine(private, helper, minimum_private_norm=0.5)
        result.loss.backward()
        self.assertIsNone(helper.grad)
        self.assertIsNotNone(private.grad)
        self.assertAlmostEqual(float(result.valid_fraction), 0.5)

    def test_relative_rms_and_disturbance_cap(self):
        delta = torch.tensor([[[2.0, 0.0]]])
        reference = torch.tensor([[[1.0, 0.0]]])
        self.assertGreater(float(relative_residual_rms(delta, reference)), 1.0)
        beta = normalized_disturbance_beta(delta, torch.zeros_like(delta), reference)
        self.assertGreater(float(beta[0]), 0.0)
        self.assertGreater(float(disturbance_cap_penalty(beta, 0.1)[0]), 0.0)
        self.assertEqual(float(broad_band_penalty(torch.tensor(1.0), 0.5, 2.0)), 0.0)

    def test_progress_is_detached(self):
        progress = torch.tensor(0.5, requires_grad=True)
        weight = smooth_progress_weight(progress, 1.0, 0.0)
        self.assertFalse(weight.requires_grad)

    def test_fixed_helper_schedule_decays_monotonically(self):
        weights = [float(smooth_progress_weight(step / 4.0, 1.0, 0.0)) for step in range(5)]
        self.assertEqual(weights[0], 1.0)
        self.assertEqual(weights[-1], 0.0)
        self.assertTrue(all(left >= right for left, right in zip(weights, weights[1:])))


if __name__ == '__main__':
    unittest.main()
