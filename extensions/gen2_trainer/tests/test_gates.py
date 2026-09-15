import unittest

import torch

from extensions.gen2_trainer.gates import CubicTimeGates, bernstein_basis


class GateMathTests(unittest.TestCase):
    def test_zero_and_partition(self):
        gates = CubicTimeGates(34)
        t = torch.linspace(0, 1, 65)
        torch.testing.assert_close(bernstein_basis(t).sum(-1), torch.ones_like(t))
        torch.testing.assert_close(gates(t), torch.ones(65, 34))
        rc, rh = gates.regularizers()
        self.assertEqual(rc.item(), 0.)
        self.assertEqual(rh.item(), 0.)
        self.assertEqual(gates.beta.numel(), 136)

    def test_analytic_derivative_matches_autograd(self):
        gates = CubicTimeGates(3)
        with torch.no_grad():
            gates.beta.copy_(torch.tensor([[.3, -.8, .4, .2], [1., .2, -.4, .1], [-.1, .4, .8, -.6]]))
        t = torch.linspace(0, 1, 15, requires_grad=True)
        for block in range(3):
            grad = torch.autograd.grad(gates(t)[:, block].sum(), t)[0]
            torch.testing.assert_close(grad, gates.derivative(t)[:, block], rtol=1e-5, atol=2e-7)
        self.assertTrue(bool((gates(t) > .5).all() and (gates(t) < 1.5).all()))

    def test_per_example_and_regularizer_mean(self):
        gates = CubicTimeGates(2, grid_points=5)
        with torch.no_grad():
            gates.beta.copy_(torch.tensor([[0., 1., 2., 3.], [1., 1., 1., 1.]]))
        v = gates(torch.tensor([0., 1.]))
        self.assertNotEqual(v[0, 0].item(), v[1, 0].item())
        torch.testing.assert_close(v[:, 1], v[0, 1].expand(2))
        rc, rh = gates.regularizers()
        torch.testing.assert_close(rc, (gates(gates.grid)-1).square().sum()/10)
        torch.testing.assert_close(rh, gates.derivative(gates.grid).square().sum()/10)
        (rc+rh).backward()
        self.assertGreater(gates.beta.grad.abs().sum().item(), 0)
        torch.testing.assert_close(gates.values(torch.tensor([0., 1.]), "time_mean"), gates.time_mean().expand(2, -1))


if __name__ == "__main__":
    unittest.main()
