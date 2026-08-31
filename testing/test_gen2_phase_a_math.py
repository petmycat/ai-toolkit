import unittest

import torch

from extensions_built_in.gen2_trainer.phase_a_math import (
    calibrate_helper_losses,
    disturbance_projection,
    robust_competence,
    semantic_direction_loss,
    smooth_helper_weight,
)


class PhaseAMathTest(unittest.TestCase):
    def test_calibration_filters_bad_helpers_and_normalizes_weights(self):
        result = calibrate_helper_losses(
            torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor([[0.5, 0.6, 0.4], [1.2, 1.1, 1.3]]),
            temperature=0.2,
            min_gain=0.0,
            min_positive_fraction=0.5,
        )
        self.assertTrue(torch.equal(result.reliable_mask, torch.tensor([True, False])))
        self.assertAlmostEqual(float(result.weights.sum()), 1.0, places=6)
        self.assertEqual(float(result.weights[1]), 0.0)

    def test_disturbance_projection_identifies_orthogonal_effect(self):
        helper = torch.tensor([[1.0, 0.0, 0.0]])
        activator = torch.tensor([[2.0, 3.0, 0.0]])
        alpha, beta, valid = disturbance_projection(activator, helper)
        self.assertTrue(bool(valid[0]))
        self.assertAlmostEqual(float(alpha[0]), 2.0, places=6)
        self.assertAlmostEqual(float(beta[0]), 3.0 / (13.0 ** 0.5), places=6)

    def test_competence_uses_only_helper_positive_probes(self):
        competence, valid = robust_competence(
            torch.tensor([10.0, 10.0]),
            torch.tensor([7.0, 11.0]),
            torch.tensor([8.0, 12.0]),
        )
        self.assertTrue(torch.equal(valid, torch.tensor([True, False])))
        self.assertAlmostEqual(float(competence), 3.0 / 2.0, places=6)

    def test_latched_helper_weight_releases_smoothly(self):
        self.assertAlmostEqual(smooth_helper_weight(0.0, 1.0, 0.25, 0.75), 1.0)
        self.assertAlmostEqual(smooth_helper_weight(0.5, 1.0, 0.25, 0.75), 0.5)
        self.assertAlmostEqual(smooth_helper_weight(0.8, 1.0, 0.25, 0.75), 0.0)

    def test_semantic_direction_zero_effect_is_finite(self):
        loss = semantic_direction_loss(torch.zeros(2, 4), torch.ones(2, 4))
        self.assertTrue(torch.isfinite(loss))

    def test_calibration_allows_dataset_only_fallback_when_no_helper_wins(self):
        result = calibrate_helper_losses(
            torch.tensor([1.0, 1.0]),
            torch.tensor([[1.1, 1.2], [1.3, 1.4]]),
            temperature=0.1,
            min_gain=0.0,
            min_positive_fraction=0.5,
        )
        self.assertFalse(bool(result.reliable_mask.any()))
        self.assertAlmostEqual(float(result.weights.sum()), 1.0, places=6)
        self.assertTrue(torch.all(result.weights > 0))


if __name__ == "__main__":
    unittest.main()
