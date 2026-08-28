import tempfile
import unittest
from pathlib import Path

import torch

from extensions_built_in.gen2_trainer.artifacts import load_tensor_artifact, save_tensor_artifact


class Gen2ArtifactTest(unittest.TestCase):
    def test_safetensors_round_trip_with_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.safetensors"
            expected = torch.randn(3, 4)
            save_tensor_artifact(
                path,
                {"A": expected},
                {"artifact": "gen2_activator", "helpers": [{"id": "h", "replacement": "style"}]},
            )
            tensors, metadata = load_tensor_artifact(path)
        self.assertTrue(torch.equal(tensors["A"], expected))
        self.assertEqual(metadata["artifact"], "gen2_activator")
        self.assertEqual(metadata["helpers"][0]["id"], "h")


if __name__ == "__main__":
    unittest.main()
