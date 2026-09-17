"""Bounded real-batch selection and non-mutating diagnostic contracts on CPU."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from extensions.gen2_trainer.v2 import mechanical


class Compiler:
    def compile(self, caption, require_trigger):
        assert require_trigger
        return SimpleNamespace(ids=list(range(int(caption.split(":")[-1]))))


class Backend:
    def __init__(self):
        self.tokens = torch.nn.Module()
        self.tokens.E = torch.nn.Parameter(torch.tensor([[0.3, 0.5], [0.7, 0.2]], dtype=torch.float32))
        self.frozen = torch.nn.Parameter(torch.tensor(0.8), requires_grad=False)
        self.compiler = Compiler()
        self.model = SimpleNamespace(vae_scale_factor=8, patch_size=2)
        self.mode = "normal"

    def frozen_parameters(self):
        return [self.frozen]

    def assert_frozen(self):
        assert not self.frozen.requires_grad

    def encode(self, captions, mode, gradients):
        assert mode == "learned" and gradients
        if self.mode == "mutate":
            with torch.no_grad():
                self.tokens.E.add_(1)
        features = [self.tokens.E.mean() * torch.ones(int(caption.split(":")[-1]), 3)
                    for caption in captions]
        return SimpleNamespace(features=features)

    def predict(self, latent, tau, conditioning):
        values = torch.stack([item.mean() for item in conditioning.features]).reshape(-1, 1, 1, 1)
        result = self.frozen * latent + values
        if self.mode == "zero":
            result = result * 0
        elif self.mode == "nan":
            result = result * float("nan")
        elif self.mode == "detach":
            result = result.detach()
        return result


class Dataset:
    def __init__(self, resolution, index):
        self.dataset_config = SimpleNamespace(buckets=True, resolution=resolution)
        # Large-image batch and longest-caption batch deliberately differ.
        self.file_list = [SimpleNamespace(path=f"mechanical-fixture-{index}-{i}.png",
                                         crop_width=size, crop_height=size)
                          for i, size in enumerate((256, 32, 48))]
        self.batch_indices = [[0, 0], [1, 1], [2, 2]]
        self.epoch_num = 1
        self.seen = []
        self.mutate_epoch = False

    def __getitem__(self, index):
        self.seen.append(index)
        if self.mutate_epoch:
            self.epoch_num += 1
        random.random(); np.random.rand(); torch.rand(1)
        return [deepcopy(self.file_list[i]) for i in self.batch_indices[index]]


class Loader:
    def __init__(self, datasets):
        self.datasets = datasets
        self.generator = torch.Generator().manual_seed(7)
        self.sampler = SimpleNamespace(generator=torch.Generator().manual_seed(19))
        self.batch_sampler = None

    def __iter__(self):
        raise AssertionError("Mechanical probe must never iterate the training sampler")


class Recorder:
    def __init__(self):
        self.events, self.records = [], []

    def event(self, name, **kwargs):
        self.events.append((name, kwargs))

    def record(self, stream, row):
        self.records.append((stream, row))


@pytest.fixture
def setup(monkeypatch):
    datasets = [Dataset(768, 0), Dataset(1280, 1)]
    loader, backend, recorder = Loader(datasets), Backend(), Recorder()
    manifest = {}
    for dataset in datasets:
        for index, item in enumerate(dataset.file_list):
            # Character count intentionally does not rank like token count.
            caption = ["long character spelling but few tokens:20", "short:120", "medium:30"][index]
            manifest[str(Path(item.path).resolve())] = {"original_caption": caption}
    cleanups = []

    class DTO:
        def __init__(self, file_items):
            self.file_items = file_items
            self.cleaned = False

        def cleanup(self):
            assert not self.cleaned
            self.cleaned = True
            cleanups.append(self)

    def prepare(dto, backend, config, by_path):
        width, height = dto.file_items[0].crop_width, dto.file_items[0].crop_height
        count = len(dto.file_items)
        qs = [by_path[str(Path(item.path).resolve())]["original_caption"] for item in dto.file_items]
        latent = torch.randn(count, 1, height // 16, width // 16)
        dto.cleanup()
        return {"qs": qs, "zt": latent, "tau": torch.rand(count), "target": torch.ones_like(latent)}

    monkeypatch.setattr(mechanical, "_native_tools", lambda: (
        lambda value: value.datasets, lambda items: DTO(items), lambda: SimpleNamespace(autocast=nullcontext)))
    monkeypatch.setattr(mechanical, "_prepare_batch", prepare)
    config = {"gen2": {"execution": {"training_seed": 123}}}
    return backend, loader, config, manifest, recorder, cleanups


def test_chooses_actual_largest_combined_and_longest_caption_batches(setup):
    backend, loader, config, manifest, recorder, cleanups = setup
    choices = mechanical._candidate_batches(backend, loader.datasets, manifest)
    assert len(choices) == 4
    for dataset_index in range(2):
        rows = [row for row in choices if row["expanded_dataset_index"] == dataset_index]
        assert [row["native_batch_index"] for row in rows] == [0, 1]
        assert rows[1]["compiled_token_lengths"] == [120, 120]
        assert rows[0]["native_file_indices"] == [0, 0]
    assert all(not dataset.seen for dataset in loader.datasets)


def test_same_largest_and_longest_batch_runs_once(setup):
    backend, loader, config, manifest, recorder, cleanups = setup
    for dataset in loader.datasets:
        dataset.file_list[1].crop_width = 512
        dataset.file_list[1].crop_height = 512
    choices = mechanical._candidate_batches(backend, loader.datasets, manifest)
    assert len(choices) == 2
    assert all(len(row["selection_reasons"]) == 2 for row in choices)


def test_gradient_probe_preserves_rng_grad_identity_parameters_sampler_and_optimizer(setup):
    backend, loader, config, manifest, recorder, cleanups = setup
    optimizer = torch.optim.AdamW([backend.tokens.E], lr=0.001)
    backend.tokens.E.grad = torch.ones_like(backend.tokens.E)
    optimizer.step()
    optimizer_before = deepcopy(optimizer.state_dict())
    parameter_before, grad_before = backend.tokens.E.detach().clone(), backend.tokens.E.grad
    grad_values = grad_before.clone()
    random.seed(991); np.random.seed(992); torch.manual_seed(993)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    loader_rng, sampler_rng = loader.generator.get_state(), loader.sampler.generator.get_state()
    index_references = [dataset.batch_indices for dataset in loader.datasets]
    rows = mechanical.run_memory_probes(backend, loader, config, manifest, recorder)
    assert len(rows) == 4 and len(cleanups) == 4
    assert all(row["gradient_nonzero"] and row["gradient_finite"] for row in rows)
    assert all(row["optimizer_steps"] == 0 and row["actual_batch_size"] == 2 for row in rows)
    assert rows[0]["actual_latent_shape"] == [2, 1, 16, 16]
    assert rows[1]["actual_text_sequence_lengths"] == [120, 120]
    assert all("does not guarantee" in row["scope_limitation"] for row in rows)
    assert all(row["memory"]["cuda_available"] is False for row in rows)
    assert backend.tokens.E.grad is grad_before
    torch.testing.assert_close(grad_before, grad_values, rtol=0, atol=0)
    torch.testing.assert_close(backend.tokens.E, parameter_before, rtol=0, atol=0)
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert np.random.get_state()[2:] == numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert torch.equal(loader.generator.get_state(), loader_rng)
    assert torch.equal(loader.sampler.generator.get_state(), sampler_rng)
    for dataset, reference in zip(loader.datasets, index_references):
        assert dataset.batch_indices is reference and dataset.epoch_num == 1
        assert dataset.seen == [0, 1]
    current = optimizer.state_dict()
    assert current["param_groups"] == optimizer_before["param_groups"]
    for key, value in current["state"][0].items():
        torch.testing.assert_close(value, optimizer_before["state"][0][key], rtol=0, atol=0)
    assert backend.frozen.grad is None
    assert recorder.events[-1][0] == "memory_gradient_probes_complete"


@pytest.mark.parametrize("mode", ["zero", "nan", "detach"])
def test_failed_gradient_probe_aborts_without_counting_or_mutating(setup, mode):
    backend, loader, config, manifest, recorder, cleanups = setup
    backend.mode = mode
    before = backend.tokens.E.detach().clone()
    rng = torch.get_rng_state()
    with pytest.raises((FloatingPointError, RuntimeError)):
        mechanical.run_memory_probes(backend, loader, config, manifest, recorder)
    torch.testing.assert_close(backend.tokens.E, before, rtol=0, atol=0)
    assert backend.tokens.E.grad is None
    assert torch.equal(torch.get_rng_state(), rng)
    assert not recorder.records
    assert recorder.events[-1][0] == "memory_gradient_probe_failed"
    assert len(cleanups) == 1


def test_restore_and_reject_unexpected_backend_mutation(setup):
    backend, loader, config, manifest, recorder, cleanups = setup
    backend.mode = "mutate"
    before = backend.tokens.E.detach().clone()
    with pytest.raises(RuntimeError, match="mutated token values"):
        mechanical.run_memory_probes(backend, loader, config, manifest, recorder)
    torch.testing.assert_close(backend.tokens.E, before, rtol=0, atol=0)
    assert backend.tokens.E.grad is None


def test_epoch_changes_are_restored_and_reported(setup):
    backend, loader, config, manifest, recorder, cleanups = setup
    loader.datasets[0].mutate_epoch = True
    with pytest.raises(RuntimeError, match="dataset epoch"):
        mechanical.run_memory_probes(backend, loader, config, manifest, recorder)
    assert loader.datasets[0].epoch_num == 1


def test_invalid_native_batch_shape_fails_without_loading_any_images(setup):
    backend, loader, config, manifest, recorder, cleanups = setup
    loader.datasets[0].file_list[0].crop_width = 257
    with pytest.raises(ValueError, match="crop geometry"):
        mechanical.run_memory_probes(backend, loader, config, manifest, recorder)
    assert not cleanups and not loader.datasets[0].seen
