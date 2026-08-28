import unittest

import torch
from torch import nn

from extensions_built_in.gen2_trainer.registry import FusedQKVTemporalAdapter, set_adapter_context


class RegistryTest(unittest.TestCase):
    def test_fused_qkv_has_independent_adapters(self):
        adapter = FusedQKVTemporalAdapter(4, rank=2, alpha=2, knots=4, delta_max=1.0)
        self.assertIsNot(adapter.q.down, adapter.k.down)
        self.assertIsNot(adapter.k.down, adapter.v.down)
        x = torch.randn(2, 3, 4)
        mask = torch.ones(2, 3, 1)
        from extensions_built_in.gen2_trainer.registry import AdapterRuntimeContext

        context = AdapterRuntimeContext(torch.tensor([0.1, 0.9]), torch.ones(2), 1.0, mask)
        output = adapter.residual(x, context)
        self.assertEqual(output.shape, (2, 3, 12))

    def test_set_adapter_context_preserves_gate_scale_and_mask(self):
        owner = nn.Module()
        indicator = torch.tensor([[3, 0, 2], [3, 2, 2]])
        timestep = torch.tensor([0.2, 0.8])
        gate = torch.tensor([1.0, 0.0])
        set_adapter_context(owner, timestep, indicator, gate, 0.5)
        context = owner._gen2_adapter_context
        self.assertIs(context.timestep, timestep)
        self.assertIs(context.style_gate, gate)
        self.assertEqual(context.branch_scale, 0.5)
        self.assertTrue(torch.equal(context.image_token_mask, (indicator == 2).float().unsqueeze(-1)))

    def test_set_adapter_context_rejects_wrong_batch_shapes(self):
        owner = nn.Module()
        indicator = torch.zeros(2, 3, dtype=torch.long)
        with self.assertRaises(ValueError):
            set_adapter_context(owner, torch.zeros(2, 1), indicator, torch.ones(2), 1.0)
        with self.assertRaises(ValueError):
            set_adapter_context(owner, torch.zeros(2), indicator, torch.ones(2, 1), 1.0)


if __name__ == "__main__":
    unittest.main()
