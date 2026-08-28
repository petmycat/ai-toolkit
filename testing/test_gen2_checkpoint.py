import tempfile
import unittest
from pathlib import Path

import torch

from extensions_built_in.gen2_trainer.checkpoint import load_phase_checkpoint, save_phase_checkpoint


class Gen2CheckpointTest(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step_1"
            tensor = {"A": torch.ones(2, 3)}
            metadata = {"step": 1, "schema_version": 1}
            save_phase_checkpoint(path, tensor, metadata)
            restored, restored_metadata = load_phase_checkpoint(path)
            self.assertTrue(torch.equal(restored["A"], tensor["A"]))
            self.assertEqual(restored_metadata, metadata)


if __name__ == "__main__":
    unittest.main()
