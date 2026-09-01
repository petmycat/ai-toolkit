import unittest

import torch

from extensions_built_in.gen2_trainer.phase_a_math import (
    effect_geometry_gate,
    effect_magnitude_loss,
    geometric_alignment_loss,
    geometric_orthogonal_loss,
    helper_effect_geometry,
)


class PhaseAMathTest(unittest.TestCase):
    def test_helper_effect_geometry_projects_orthogonal_effect(self):
        helpers = torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]]])
        activator = torch.tensor([[1.0, 3.0]], requires_grad=True)
        geometry = helper_effect_geometry(helpers, activator, rank=0, energy_threshold=0.99)
        self.assertAlmostEqual(float(geometry.alignment[0].detach()), 1.0, places=6)
        self.assertAlmostEqual(float(geometry.orthogonal_fraction[0].detach()), 0.0, places=6)
        self.assertAlmostEqual(float(geometry.magnitude_ratio[0].detach()), 3.16227766, places=6)
        (geometry.projection.square().mean() + geometry.orthogonal.square().mean()).backward()
        self.assertIsNotNone(activator.grad)

    def test_helper_effect_geometry_zero_effect_is_not_success(self):
        helpers = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        geometry = helper_effect_geometry(helpers, torch.zeros(1, 2))
        self.assertAlmostEqual(float(geometry.alignment[0]), 0.0, places=6)
        self.assertAlmostEqual(float(geometry.magnitude_ratio[0].detach()), 0.0, places=6)
        self.assertTrue(bool(geometry.valid[0]))
        self.assertTrue(torch.isfinite(effect_magnitude_loss(geometry.projection_norm, geometry.helper_norm_median)))

    def test_geometric_losses_are_finite_for_orthogonal_effect(self):
        helpers = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        activator = torch.tensor([[0.0, 2.0]])
        geometry = helper_effect_geometry(helpers, activator)
        self.assertTrue(torch.isfinite(geometric_alignment_loss(activator, geometry.projection)))
        self.assertTrue(torch.isfinite(geometric_orthogonal_loss(geometry.orthogonal, geometry.activator_norm)))

    def test_effect_geometry_gate_requires_heldout_metrics(self):
        train = {"alignment": 0.9, "orthogonal_fraction": 0.1, "magnitude_ratio": 0.8, "valid_fraction": 1.0}
        heldout = {"alignment": 0.2, "orthogonal_fraction": 0.1, "magnitude_ratio": 0.8, "valid_fraction": 1.0}
        passed, checks = effect_geometry_gate(train, heldout)
        self.assertFalse(passed)
        self.assertFalse(checks["heldout_alignment"])

if __name__ == "__main__":
    unittest.main()
