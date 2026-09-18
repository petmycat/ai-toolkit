from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from extensions.gen2_trainer.v2.diagnostic_config import (
    checked_output_path, resolve_diagnostic_config,
)
from extensions.gen2_trainer.v2.diagnostic_data import (
    packets_from_latent, select_native_batches, tensor_digest,
)
from extensions.gen2_trainer.v2.diagnostic_process import inherited_config


def minimal():
    return {"type": "gen2_v2_diagnostic", "source_checkpoint": "/vm/source/inference_final",
            "training_folder": "/vm/results"}


def test_strict_config_is_idempotent_and_rejects_typos_and_unbounded_work():
    source = minimal()
    original = deepcopy(source)
    config = resolve_diagnostic_config(source)
    assert source == original
    assert resolve_diagnostic_config(config) == config
    for key, value in (("descent_steps", 500), ("seed", True), ("noise_fractions", [0., .5]),
                       ("noise_seeds", [1, 1]), ("gradient_cosine_min", float("nan")),
                       ("examples_per_resolution", 0), ("missing", 5)):
        with pytest.raises(ValueError):
            resolve_diagnostic_config({**source, "diagnostic": {key: value}})
    with pytest.raises(ValueError):
        resolve_diagnostic_config({**source, "train": {"steps": 500}})


def test_output_cannot_overlap_source_run_existing_output_or_readonly_gen2(tmp_path):
    source = tmp_path / "source" / "gen2_v2" / "checkpoints" / "inference_final"
    source.mkdir(parents=True)
    config = {**minimal(), "source_checkpoint": str(source), "training_folder": str(tmp_path)}
    assert checked_output_path(config, "new_job", tmp_path / "source") == tmp_path / "new_job"
    with pytest.raises(ValueError):
        checked_output_path(config, "source", tmp_path / "source")
    with pytest.raises(ValueError):
        checked_output_path(config, "../source")
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("preserve")
    with pytest.raises(ValueError):
        checked_output_path(config, "existing")
    protected = Path(__file__).resolve().parents[3] / "gen2"
    with pytest.raises(ValueError, match="read-only"):
        checked_output_path({**config, "training_folder": str(protected)}, "forbidden")


def test_inherited_config_only_changes_diagnostic_runtime_fields():
    source = {"device": "cuda:0", "training_folder": "/vm/models", "train": {"batch_size": 1, "steps": 500},
        "gen2": {"checkpoint": {"resume_from": "old"}, "diagnostics": {"mechanical_probes": True},
                 "evaluation": {"named_phrase": "saved phrase"}, "optimizer": {"type": "adamw", "lr": .0005}}}
    before = deepcopy(source)
    result = inherited_config({"metadata": {"resolved_config": source}}, resolve_diagnostic_config(minimal()))
    assert source == before
    assert result["gen2"]["evaluation"]["named_phrase"] == "saved phrase"
    assert result["gen2"]["checkpoint"]["resume_from"] is None
    assert result["train"]["steps"] == 500
    assert result["train"]["disable_sampling"] is True
    assert result["gen2"]["optimizer"] == before["gen2"]["optimizer"]


def test_selection_uses_distinct_real_images_spread_over_lengths_for_each_resolution(tmp_path):
    paths = [str((tmp_path / f"{i}.png").resolve()) for i in range(7)]
    manifest = {path: {"original_caption": str(i), "sample_id": str(i)} for i, path in enumerate(paths)}
    compiler = SimpleNamespace(compile=lambda caption, **kw: SimpleNamespace(ids=list(range(int(caption)+2))))
    datasets = [SimpleNamespace(dataset_config=SimpleNamespace(resolution=r),
        file_list=[SimpleNamespace(path=p) for p in paths], batch_indices=[[i] for i in [6, 2, 0, 1, 4, 3, 5, 0]])
        for r in (768, 1280)]
    selected = select_native_batches(datasets, compiler, manifest, 3)
    assert [(x["resolution"], x["sample_id"]) for x in selected] == [
        (768, "0"), (768, "3"), (768, "6"), (1280, "0"), (1280, "3"), (1280, "6")]


class NativeModelFixture:
    device_torch = torch.device("cpu")
    torch_dtype = torch.float32

    def __init__(self):
        self.noise_calls = 0
        self.times = []

    def get_latent_noise_from_latents(self, z0, noise_offset):
        assert noise_offset == 0
        self.noise_calls += 1
        return torch.randn_like(z0)

    def add_noise(self, z0, noise, timestep):
        self.times.append(float(timestep[0]))
        t = timestep.view(-1, 1, 1, 1) / 1000
        return (1-t)*z0 + t*noise


def test_packets_reuse_one_noise_draw_per_seed_and_preserve_native_latent_and_rng():
    model = NativeModelFixture()
    z0 = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    before = z0.clone()
    rng = torch.get_rng_state().clone()
    settings = {"noise_seeds": [10, 11], "noise_fractions": [.1, .5, .9]}
    choice = {"expanded_dataset_index": 0, "sample_id": "fixture", "resolution": 768}
    packets = packets_from_latent(model, z0, ["exact original [trigger] caption"],
                                  [{"transforms": {"crop_x": 17}}], choice, settings)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(z0, before)
    assert model.noise_calls == 2
    assert model.times == [100, 500, 900, 100, 500, 900]
    assert len({p["id"] for p in packets}) == 6
    for group in (packets[:3], packets[3:]):
        assert len({p["diagnostic"]["noise_sha256"] for p in group}) == 1
        assert len({p["diagnostic"]["target_sha256"] for p in group}) == 1
        for p in group:
            # The target is the derivative of the same straight-line path.
            torch.testing.assert_close(p["zt"], z0 + p["tau"].view(1,1,1,1) * p["target"])
            assert p["qs"] == ["exact original [trigger] caption"]
            assert p["diagnostic"]["z0_sha256"] == tensor_digest(z0)
            assert p["metadata"][0]["native_transforms"] == {"crop_x": 17}
