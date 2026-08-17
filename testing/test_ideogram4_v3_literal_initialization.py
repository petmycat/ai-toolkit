import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open

from extensions_built_in.ideogram4_v3_activator.literal_initialization import (
    LiteralInitializationConfig,
    atomic_save_literal_initialization,
    atomic_save_manifest,
    build_literal_initialization_manifest,
    interval_overlap_weights,
    manifest_sha256,
    map_literal_embeddings_to_four,
    tensor_sha256,
)


class Ideogram4V3LiteralInitializationTest(unittest.TestCase):
    def test_interval_overlap_mapping_supported_source_lengths(self):
        for token_count in (1, 2, 3, 4, 5, 8):
            with self.subTest(token_count=token_count):
                source = torch.arange(token_count * 3, dtype=torch.float64).reshape(token_count, 3)
                mapped = map_literal_embeddings_to_four(source)
                weights = interval_overlap_weights(token_count)

                self.assertEqual(mapped.shape, (4, 3))
                self.assertEqual(mapped.dtype, torch.float32)
                self.assertTrue(torch.isfinite(mapped).all())
                torch.testing.assert_close(weights.sum(dim=1), torch.ones(4))
                torch.testing.assert_close(mapped, weights @ source.float(), rtol=0, atol=0)
                torch.testing.assert_close(mapped, map_literal_embeddings_to_four(source), rtol=0, atol=0)

    def test_interval_overlap_has_expected_edge_cases(self):
        single = torch.tensor([[2.0, -1.0]])
        torch.testing.assert_close(
            map_literal_embeddings_to_four(single),
            single.expand(4, -1),
            rtol=0,
            atol=0,
        )
        four = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        torch.testing.assert_close(map_literal_embeddings_to_four(four), four, rtol=0, atol=0)

    def test_non_finite_source_is_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                map_literal_embeddings_to_four(torch.tensor([[value, 0.0]]))

    def test_manifest_hash_and_atomic_saves_are_deterministic(self):
        source = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        mapped = map_literal_embeddings_to_four(source)
        config = LiteralInitializationConfig(enabled=True)
        manifest = build_literal_initialization_manifest(
            literal="<trigger>",
            token_ids=[10, 11, 12, 13, 14],
            source_embeddings=source,
            mapped_embeddings=mapped,
            tokenizer={"name": "fake-tokenizer", "revision": "test"},
            config=config,
        )
        rebuilt = build_literal_initialization_manifest(
            literal="<trigger>",
            token_ids=[10, 11, 12, 13, 14],
            source_embeddings=source,
            mapped_embeddings=mapped,
            tokenizer={"revision": "test", "name": "fake-tokenizer"},
            config=config,
        )

        self.assertEqual(manifest_sha256(manifest), manifest_sha256(rebuilt))
        self.assertEqual(manifest["source_embeddings"]["sha256"], tensor_sha256(source))
        self.assertEqual(manifest["mapped_embeddings"]["sha256"], tensor_sha256(mapped))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / config.manifest_filename
            artifact_path = root / config.artifact_filename
            saved_manifest_hash = atomic_save_manifest(manifest_path, manifest)
            artifact_ref = atomic_save_literal_initialization(artifact_path, mapped, manifest)

            self.assertEqual(saved_manifest_hash, manifest_sha256(manifest))
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), manifest)
            self.assertEqual(artifact_ref["manifest_sha256"], manifest_sha256(manifest))
            with safe_open(artifact_path, framework="pt", device="cpu") as handle:
                torch.testing.assert_close(handle.get_tensor("embeddings"), mapped, rtol=0, atol=0)
                self.assertEqual(
                    handle.metadata()["literal_initialization.manifest_sha256"],
                    manifest_sha256(manifest),
                )


if __name__ == "__main__":
    unittest.main()
