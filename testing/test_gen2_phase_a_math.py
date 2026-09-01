import unittest

import torch

from extensions_built_in.gen2_trainer.phase_a_math import (
    effect_geometry_gate,
    fit_shared_positive_helper_mixture,
    normalized_teacher_distillation_loss,
    positive_cone_geometry,
    positive_cone_projection,
    response_field_signature,
)


class PhaseAMathTest(unittest.TestCase):
    def test_response_signature_handles_variable_resolution_fields(self):
        first = response_field_signature(torch.ones(1, 2, 3, 4))
        second = response_field_signature(torch.ones(1, 5, 7, 8))
        self.assertEqual(first.shape, (1, 2))
        self.assertEqual(second.shape, (1, 2))
        self.assertTrue(torch.isfinite(torch.cat((first, second), dim=0)).all())

    def test_shared_positive_mixture_is_normalized_and_cross_content(self):
        helpers = torch.tensor([
            [[1.0, 0.0], [2.0, 0.0]],
            [[0.0, 1.0], [0.0, 2.0]],
        ])
        target = torch.tensor([[1.0, 0.5], [2.0, 1.0]])
        result = fit_shared_positive_helper_mixture(helpers, target)
        self.assertTrue(torch.all(result["weights"] >= 0))
        self.assertAlmostEqual(float(result["weights"].sum()), 1.0, places=5)
        self.assertTrue(torch.isfinite(result["relative_fit"]))

    def test_positive_cone_rejects_opposite_direction(self):
        helpers = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        projection, coefficients = positive_cone_projection(helpers, torch.tensor([[-1.0, 0.0]]))
        self.assertAlmostEqual(float(projection.norm()), 0.0, places=6)
        self.assertTrue(torch.all(coefficients >= 0))

    def test_positive_cone_geometry_accepts_positive_mixture(self):
        helpers = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        geometry = positive_cone_geometry(helpers, torch.tensor([[1.0, 2.0]]))
        self.assertTrue(bool(geometry["valid"][0]))
        self.assertGreater(float(geometry["alignment"][0]), 0.99)
        self.assertLess(float(geometry["orthogonal_fraction"][0]), 1e-5)

    def test_normalized_teacher_distillation_is_finite_and_detaches_teacher(self):
        activator = torch.tensor([[1.0, 2.0]], requires_grad=True)
        teacher = torch.tensor([[2.0, 4.0]], requires_grad=True)
        loss = normalized_teacher_distillation_loss(activator, teacher, effect_scale=2.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(activator.grad)
        self.assertIsNone(teacher.grad)

    def test_effect_geometry_gate_requires_heldout_metrics(self):
        train = {"alignment": 0.9, "orthogonal_fraction": 0.1, "magnitude_ratio": 0.8, "valid_fraction": 1.0}
        heldout = {"alignment": 0.2, "orthogonal_fraction": 0.1, "magnitude_ratio": 0.8, "valid_fraction": 1.0}
        passed, checks = effect_geometry_gate(train, heldout)
        self.assertFalse(passed)
        self.assertFalse(checks["heldout_alignment"])


if __name__ == "__main__":
    unittest.main()
