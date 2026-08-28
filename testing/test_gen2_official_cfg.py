import unittest

import torch

from extensions_built_in.gen2_trainer.unconditional import _convert_split_attention_state_dict, official_asymmetric_cfg


class OfficialCfgTest(unittest.TestCase):
    def test_asymmetric_cfg_formula(self):
        v_c = torch.tensor([3.0])
        v_u = torch.tensor([1.0])
        delta_c = torch.tensor([2.0])
        delta_u = torch.tensor([4.0])
        result = official_asymmetric_cfg(v_c, v_u, 2.0, eta_c=1.0, eta_u=0.5, delta_c=delta_c, delta_u=delta_u)
        expected = torch.tensor([3.0]) + 2.0 * (torch.tensor([5.0]) - torch.tensor([3.0]))
        self.assertTrue(torch.equal(result, expected))

    def test_split_attention_weights_are_repacked(self):
        state = {
            "layers.0.attention.to_q.weight": torch.ones(2, 3),
            "layers.0.attention.to_k.weight": torch.full((2, 3), 2.0),
            "layers.0.attention.to_v.weight": torch.full((2, 3), 3.0),
            "layers.0.attention.to_out.0.weight": torch.eye(3),
        }
        converted = _convert_split_attention_state_dict(state)
        self.assertTrue(torch.equal(converted["layers.0.attention.qkv.weight"], torch.cat([state[key] for key in ("layers.0.attention.to_q.weight", "layers.0.attention.to_k.weight", "layers.0.attention.to_v.weight")], dim=0)))
        self.assertTrue(torch.equal(converted["layers.0.attention.o.weight"], state["layers.0.attention.to_out.0.weight"]))
        self.assertNotIn("layers.0.attention.to_q.weight", converted)

    def test_negative_scale_fails(self):
        with self.assertRaises(ValueError):
            official_asymmetric_cfg(torch.zeros(1), torch.zeros(1), 1.0, eta_u=-1.0)


if __name__ == "__main__":
    unittest.main()
