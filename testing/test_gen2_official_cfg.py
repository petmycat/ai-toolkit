import unittest

import torch

from extensions_built_in.gen2_trainer.unconditional import official_asymmetric_cfg


class OfficialCfgTest(unittest.TestCase):
    def test_asymmetric_cfg_formula(self):
        v_c = torch.tensor([3.0])
        v_u = torch.tensor([1.0])
        delta_c = torch.tensor([2.0])
        delta_u = torch.tensor([4.0])
        result = official_asymmetric_cfg(v_c, v_u, 2.0, eta_c=1.0, eta_u=0.5, delta_c=delta_c, delta_u=delta_u)
        expected = torch.tensor([3.0]) + 2.0 * (torch.tensor([5.0]) - torch.tensor([3.0]))
        self.assertTrue(torch.equal(result, expected))

    def test_negative_scale_fails(self):
        with self.assertRaises(ValueError):
            official_asymmetric_cfg(torch.zeros(1), torch.zeros(1), 1.0, eta_u=-1.0)


if __name__ == "__main__":
    unittest.main()
