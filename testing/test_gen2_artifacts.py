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
                {
                    "artifact": "gen2_activator",
                    "schema_version": 4,
                    "initializer": "literal_trigger_resampled",
                    "literal": "<r1X1dOn9mA2>",
                    "placeholder": "[trigger]",
                    "tokens": 3,
                    "dimension": 4,
                    "qwen_per_layer_adapter": True,
                    "qwen_per_layer_adapter_layers": [0, 1],
                    "qwen_per_layer_adapter_rank": 4,
                    "qwen_per_layer_adapter_alpha": 4.0,
                },
            )
            tensors, metadata = load_tensor_artifact(path)
        self.assertTrue(torch.equal(tensors["A"], expected))
        self.assertEqual(metadata["artifact"], "gen2_activator")
        self.assertEqual(metadata["schema_version"], 4)
        self.assertEqual(metadata["initializer"], "literal_trigger_resampled")
        self.assertEqual(metadata["literal"], "<r1X1dOn9mA2>")
        self.assertTrue(metadata["qwen_per_layer_adapter"])
        self.assertEqual(metadata["qwen_per_layer_adapter_layers"], [0, 1])
        self.assertEqual(metadata["qwen_per_layer_adapter_rank"], 4)
        self.assertEqual(metadata["qwen_per_layer_adapter_alpha"], 4.0)
        self.assertEqual(metadata["placeholder"], "[trigger]")
        self.assertEqual(metadata["tokens"], 3)
        self.assertEqual(metadata["dimension"], 4)


if __name__ == "__main__":
    unittest.main()
