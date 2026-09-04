import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from extensions_built_in.gen2_trainer.probes import deterministic_probe_noise, deterministic_probe_noise_seed, deterministic_probe_timestep, load_fixed_probes, probe_fingerprint, save_fixed_probes
from extensions_built_in.gen2_trainer.split import build_split, ensure_split, make_train_dataset_view


class _Item:
    def __init__(self, path, caption):
        self.path = str(path)
        self.raw_caption = caption

    def load_caption(self, _caption_dict=None):
        return None


class Gen2SplitProbeTest(unittest.TestCase):
    def _dataset(self, root):
        items = []
        for index in range(4):
            image = root / f"image_{index}.png"
            image.write_bytes(f"image-{index}".encode())
            items.append(_Item(image, json.dumps({"style": {"name": "[trigger]", "index": index}})))
        return SimpleNamespace(dataset_path=str(root), file_list=items, caption_dict=None, dataset_config=SimpleNamespace(num_repeats=1, buckets=False))

    def test_split_is_stable_and_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            path = root / "gen2" / "dataset_split.json"
            split = ensure_split(path, dataset, heldout_count=1, seed=7)
            self.assertEqual(split, ensure_split(path, dataset, heldout_count=1, seed=7))
            view = make_train_dataset_view(dataset, split)
            self.assertEqual(len(view.file_list), 3)
            dataset.file_list[0].raw_caption = json.dumps({"style": "changed"})
            with self.assertRaises(ValueError):
                ensure_split(path, dataset, heldout_count=1, seed=7)

    def test_fixed_probe_round_trip_and_strict_split_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed = 5
            noise_a, noise_seed = deterministic_probe_noise(torch.zeros(1, 2), seed, "train-0", "early", 0, "train")
            noise_b, noise_seed_b = deterministic_probe_noise(torch.zeros(1, 2), seed, "train-0", "early", 0, "train")
            self.assertEqual(noise_seed, deterministic_probe_noise_seed(seed, "train-0", "early", 0, "train"))
            self.assertEqual(noise_seed, noise_seed_b)
            self.assertTrue(torch.equal(noise_a, noise_b))
            regions = {"early": {"min": 0, "max": 500, "timestep_count": 1, "pair_count": 1}, "late": {"min": 500, "max": 1000, "timestep_count": 1, "pair_count": 1}}
            records = []
            for label, pair in (("train", "train-0"), ("heldout", "heldout-0")):
                for region, bounds in regions.items():
                    records.append({"region": region, "split_label": label, "pair_fingerprint": pair, "ordinal": 0, "fingerprint": probe_fingerprint(seed, pair, region, 0, label), "prompt": "{}", "noise_seed": deterministic_probe_noise_seed(seed, pair, region, 0, label), "noisy_latent": torch.ones(1, 2), "target": torch.zeros(1, 2), "timestep": torch.tensor([float(deterministic_probe_timestep(seed, pair, region, 0, bounds["min"], bounds["max"], label, 1))])})
            save_fixed_probes(root / "fixed_probes.json", root / "fixed_probes.safetensors", records, split_fingerprint="abc", seed=seed, regions=regions)
            loaded, metadata = load_fixed_probes(root / "fixed_probes.json", expected_split_fingerprint="abc", expected_regions=regions)
            self.assertEqual(metadata["seed"], seed)
            self.assertEqual(len(loaded), 4)
            self.assertEqual(loaded[0]["noisy_latent"].dtype, torch.float32)
            self.assertEqual(loaded[0]["noisy_latent"].device.type, "cpu")
            with self.assertRaises(ValueError):
                load_fixed_probes(root / "fixed_probes.json", expected_split_fingerprint="other")


if __name__ == "__main__":
    unittest.main()
