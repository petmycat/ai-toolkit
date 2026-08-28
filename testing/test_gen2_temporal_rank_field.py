import unittest

import torch

from extensions_built_in.gen2_trainer.temporal_rank_field import (
    TemporalField,
    TemporalRankFieldLoRA,
    time_smooth_regularizer,
)


class TemporalRankFieldTest(unittest.TestCase):
    def test_init_is_noop(self):
        module = TemporalRankFieldLoRA(5, 7, rank=3, knots=4)
        x = torch.randn(2, 6, 5)
        mask = torch.ones(2, 6, 1)
        residual = module(x, torch.tensor([0.1, 0.9]), mask, torch.ones(2), 1.0)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_text_rows_are_zero_and_gate_zero_is_noop(self):
        module = TemporalRankFieldLoRA(3, 4, rank=2, knots=4)
        module.up.data.fill_(1.0)
        x = torch.randn(2, 5, 3)
        mask = torch.zeros(2, 5, 1)
        mask[:, 3:] = 1
        residual = module(x, torch.tensor([0.2, 0.8]), mask, torch.ones(2), 1.0)
        self.assertTrue(torch.equal(residual[:, :3], torch.zeros_like(residual[:, :3])))
        gated = module(x, torch.tensor([0.2, 0.8]), mask, torch.zeros(2), 1.0)
        self.assertTrue(torch.equal(gated, torch.zeros_like(gated)))

    def test_per_sample_temporal_field(self):
        field = TemporalField(knots=3, rank=1, delta_max=1.0)
        field.values.data[:, 0] = torch.tensor([-1.0, 0.0, 1.0])
        result = field(torch.tensor([0.0, 1.0]))
        self.assertNotEqual(result[0, 0].item(), result[1, 0].item())

    def test_timestep_must_be_per_sample_vector(self):
        field = TemporalField(knots=4, rank=2)
        with self.assertRaises(ValueError):
            field(torch.zeros(2, 3))

    def test_smooth_regularizer_has_gradient(self):
        field = TemporalField(knots=4, rank=2)
        field.values.data[1].fill_(1.0)
        loss = time_smooth_regularizer([field])
        loss.backward()
        self.assertIsNotNone(field.values.grad)


if __name__ == "__main__":
    unittest.main()
