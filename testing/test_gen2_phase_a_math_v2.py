import unittest

import torch

from extensions_built_in.gen2_trainer.controller import (
    AdaptiveJointController,
    gradient_cosine,
    isolated_gradient_cosine,
)
from extensions_built_in.gen2_trainer.phase_a_math import (
    dataset_mse,
    detached_ema_prototype,
    helper_direction_cosine,
    helper_direction_loss,
    prototype_consistency_loss,
)


class PhaseAMathV2Test(unittest.TestCase):
    def test_three_phase_a_losses_and_per_sample_cosine(self):
        target = torch.zeros(2, 3)
        activator = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], requires_grad=True)
        baseline = torch.zeros_like(activator)
        helper = activator.detach().clone()
        self.assertAlmostEqual(float(dataset_mse(activator, target)), 5.0 / 6.0)
        loss, cosine = helper_direction_loss(activator, helper, baseline)
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        self.assertEqual(tuple(cosine.shape), (2,))
        loss.backward()
        self.assertIsNotNone(activator.grad)

    def test_detached_ema_prototype_per_region(self):
        first = detached_ema_prototype(None, {"early": torch.tensor([[1.0, 2.0]]), "middle": torch.tensor([[7.0, 8.0]])}, 0.5)
        second = detached_ema_prototype(first, {"early": torch.tensor([[3.0, 4.0]]), "late": torch.tensor([[5.0, 6.0]])}, 0.5)
        self.assertTrue(torch.equal(second["early"], torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.equal(second["middle"], torch.tensor([7.0, 8.0])))
        self.assertTrue(torch.equal(second["late"], torch.tensor([5.0, 6.0])))

    def test_prototype_consistency_is_cosine_and_masks_low_norm(self):
        losses, valid = prototype_consistency_loss(
            torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
            torch.tensor([1.0, 0.0]),
        )
        self.assertTrue(torch.allclose(losses, torch.tensor([0.0, 1.0, 0.0])))
        self.assertTrue(torch.equal(valid, torch.tensor([True, True, False])))

    def test_controller_unavailable_resume_and_hysteresis(self):
        controller = AdaptiveJointController({"initial_fraction": 0.75, "max_fraction": 0.75, "release_patience": 2, "recovery_patience": 2, "release_step": 0.25, "recovery_step": 0.1, "conflict_threshold": -0.1, "recovery_threshold": 0.0})
        controller.update(torch.tensor([-0.5]))
        controller.update(torch.tensor([-0.5]))
        controller.update(torch.tensor([-0.5]))
        self.assertTrue(controller.state.released)
        self.assertAlmostEqual(controller.fraction, 0.5)
        controller.update(torch.tensor([0.5]))
        controller.update(torch.tensor([0.5]))
        self.assertFalse(controller.state.released)
        self.assertAlmostEqual(controller.fraction, 0.6)
        controller.mark_unavailable("missing_probe")
        self.assertTrue(controller.state.unavailable)
        restored = AdaptiveJointController({"initial_fraction": 0.75, "max_fraction": 0.75})
        restored.load_state_dict(controller.state_dict())
        self.assertTrue(restored.state.unavailable)

    def test_gradient_cosine_utilities(self):
        left = [torch.tensor([1.0, 0.0]), None]
        right = [torch.tensor([2.0, 0.0]), None]
        self.assertAlmostEqual(float(gradient_cosine(left, right)), 1.0, places=6)
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        left_loss = (parameter * torch.tensor([1.0, 0.0])).sum()
        right_loss = (parameter * torch.tensor([2.0, 0.0])).sum()
        self.assertAlmostEqual(float(isolated_gradient_cosine(left_loss, right_loss, [parameter])), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
