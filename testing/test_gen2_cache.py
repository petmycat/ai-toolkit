import tempfile
import unittest
from pathlib import Path

import torch

from extensions_built_in.gen2_trainer.cache import load_conditioning_cache, save_conditioning_cache


class Gen2CacheTest(unittest.TestCase):
    def test_metadata_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conditioning.pt"
            save_conditioning_cache(path, [{"features": torch.ones(1)}], {"activator": "a"})
            self.assertEqual(len(load_conditioning_cache(path, {"activator": "a"})), 1)
            with self.assertRaises(ValueError):
                load_conditioning_cache(path, {"activator": "b"})


if __name__ == "__main__":
    unittest.main()
