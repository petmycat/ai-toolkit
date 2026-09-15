"""Data/lifecycle seams tested without loading or impersonating Ideogram."""
import ast
import copy
import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest
import torch
from PIL import Image

from extensions.gen2_trainer.config import resolve_process_config
from extensions.gen2_trainer.data import (NativeDataStream, PROTECTED_ROOT, assert_writable_path,
    canonical_caption, preflight_datasets, prepare_batch)
from extensions.gen2_trainer.diagnostics import capture_rng_state, restore_rng_state
from extensions.gen2_trainer.engine import PhaseSchedule
from extensions.gen2_trainer.evaluation import Evaluation
from extensions.gen2_trainer.process import Gen2Runner


def native_data_stub(monkeypatch):
    module = ModuleType("toolkit.data_loader")
    module.image_extensions = [".png", ".jpg"]
    module.get_dataloader_datasets = lambda loader: [loader.dataset]
    module.trigger_dataloader_setup_epoch = lambda loader: setattr(loader.dataset, "epoch_num", loader.dataset.epoch_num + 1)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_readonly_guard_and_exact_caption_preflight(tmp_path, monkeypatch):
    native_data_stub(monkeypatch)
    for path in (PROTECTED_ROOT, PROTECTED_ROOT / "sub" / "log.txt"):
        with pytest.raises(ValueError, match="read-only"):
            assert_writable_path(path)
    Image.new("RGB", (8, 8), "blue").save(tmp_path / "example.png")
    raw = '  {"subject": "cat,  dog", "literal": "[trigger] <gen2style>"}  '
    (tmp_path / "example.txt").write_text(raw, encoding="utf-8")
    config = resolve_process_config({"datasets": [{"folder_path": str(tmp_path)}]})
    rows = preflight_datasets(config)
    assert rows[0]["original_caption"] == raw
    assert rows[0]["q"] == '{"subject": "cat,  dog", "literal": " "}'
    assert rows[0]["source_dimensions"] == [8, 8]
    config["datasets"].append(copy.deepcopy(config["datasets"][0]))
    with pytest.raises(ValueError, match="overlap"):
        preflight_datasets(config)


def test_json_caption_relative_keys_and_duplicate_validation(tmp_path, monkeypatch):
    native_data_stub(monkeypatch)
    monkeypatch.chdir(tmp_path)
    Image.new("RGB", (8, 8), "blue").save("example.png")
    Path("captions.json").write_text(json.dumps({"./example.png": {"caption": "long", "caption_short": "short"}}))
    config = resolve_process_config({"datasets": [{"dataset_path": "captions.json", "use_short_captions": True}]})
    assert preflight_datasets(config)[0]["q"] == "short"
    config["train"]["validation_config"] = {"validation_items": [{"image_path": "example.png", "prompt": "cat"}]}
    with pytest.raises(ValueError, match="duplicates"):
        preflight_datasets(config)


def test_prepared_state_uses_full_native_table_and_original_caption(tmp_path, monkeypatch):
    path = str((tmp_path / "image.png").resolve())
    class Batch:
        latents = torch.tensor([[[[2.]]], [[[3.]]]])
        file_items = [SimpleNamespace(path=path, raw_caption="native transformed"), SimpleNamespace(path=path)]
        cleaned = False
        def cleanup(self): self.cleaned = True
    model = SimpleNamespace(device_torch=torch.device("cpu"), torch_dtype=torch.float32,
        noise_scheduler=SimpleNamespace(timesteps=torch.tensor([1000., 1.])),
        get_latent_noise_from_latents=lambda z, **kw: torch.ones_like(z),
        add_noise=lambda z, e, t: (1-t[:, None, None, None]/1000)*z+t[:, None, None, None]/1000*e)
    monkeypatch.setattr(torch, "randint", lambda low, high, shape, **kw: torch.tensor([0, high-1]))
    config = resolve_process_config({})
    source = {"sample_id": "id", "dataset_index": 0, "group": "unassigned", "original_caption": "cat,  [trigger] dog"}
    dto = Batch()
    result = prepare_batch(dto, model, config, {path: source})
    assert result["qs"] == ["cat,   dog"] * 2
    assert result["metadata"][1]["time_table_index"] == 1
    torch.testing.assert_close(result["tau"], torch.tensor([1., .001]))
    torch.testing.assert_close(result["target"], torch.tensor([[[[-1.]]], [[[-2.]]]]))
    torch.testing.assert_close(result["zt"][0], torch.ones(1, 1, 1))
    assert dto.cleaned


class ReplayBatch:
    def __init__(self, value): self.value = value
    def cleanup(self): pass


class ReplayDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.file_list = [SimpleNamespace(path=f"file-{i}") for i in range(4)]
        self.epoch_num = 1
        self.batch_indices = None
    def __len__(self): return len(self.file_list)
    def __getitem__(self, index): return (index, random.random(), float(np.random.random()), float(torch.rand(())))


def replay_loader():
    return torch.utils.data.DataLoader(ReplayDataset(), shuffle=True, batch_size=1, num_workers=0,
        collate_fn=lambda values: ReplayBatch(values[0]))


def test_native_stream_replays_shuffle_and_augmentation_across_epoch(tmp_path, monkeypatch):
    native_data_stub(monkeypatch)
    random.seed(3); np.random.seed(3); torch.manual_seed(3)
    stream = NativeDataStream(replay_loader(), 14)
    stream.next(); stream.next()
    state, rng = copy.deepcopy(stream.state_dict()), capture_rng_state()
    expected = [stream.next().value for _ in range(7)]
    # Reconstruction can consume RNG; restore logic must recover the training boundary.
    random.random(); torch.rand(5)
    resumed = NativeDataStream(replay_loader(), 14)
    resumed.load_state_dict(state)
    restore_rng_state(rng)
    assert [resumed.next().value for _ in range(7)] == expected


def test_optional_native_factory_hook_preserves_default():
    source = Path(__file__).resolve().parents[3] / "toolkit" / "data_loader.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_dataloader_from_datasets")
    assert function.args.args[-1].arg == "dataset_class"
    assert isinstance(function.args.defaults[-1], ast.Constant) and function.args.defaults[-1].value is None
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.BoolOp)]
    assert len(calls) == 1 and isinstance(calls[0].func.op, ast.Or)
    assert [n.id for n in calls[0].func.values] == ["dataset_class", "AiToolkitDataset"]


class CollectRecorder:
    def __init__(self): self.rows = []
    def record(self, stream, row): self.rows.append((stream, row))
    def event(self, *args, **kwargs): pass
    def set_context(self, **kwargs): pass
    def flush(self): pass


def test_activation_statistics_keep_per_example_time_bins(tmp_path):
    @contextmanager
    def diagnostics(callback):
        backend.callback = callback
        yield
    backend = SimpleNamespace(current_branch=SimpleNamespace(tau=torch.tensor([.05, 1.]), name="styled_student"),
        diagnostics=diagnostics)
    recorder = CollectRecorder()
    evaluation = Evaluation(backend, resolve_process_config({}), recorder, tmp_path)
    adapter = SimpleNamespace(gen2_original_path="blocks.0.attention.qkv", gen2_block_id=0)
    x = torch.tensor([[[1., 1.]], [[4., 4.]]])
    with evaluation.activation_context(True):
        backend.callback(adapter, x, x, x/2, x/2, {"image": torch.ones(2, 1, dtype=torch.bool)})
    assert [row["time_bin"] for _, row in recorder.rows] == [0, 5]
    assert [row["regions"][0]["base_rms"] for _, row in recorder.rows] == [1., 4.]


def test_gate_statistics_also_run_at_nonboundary_calibration_save(tmp_path):
    config = resolve_process_config({"training_folder": str(tmp_path), "train": {"steps": 7, "disable_sampling": True},
        "save": {"save_every": 3}, "gen2": {"phases": {"warmup_updates": 1, "refinement_updates": 2,
            "calibration_updates": 4, "diffusion_updates_per_cycle": 1, "conditioning_updates_per_cycle": 1}}})
    runner = Gen2Runner(config, "test")
    runner.engine = SimpleNamespace(logical_update=6, update_attempt=6, schedule=PhaseSchedule(**config["gen2"]["phases"]))
    calls = []
    runner.recorder = CollectRecorder()
    runner.backend = SimpleNamespace(components=lambda: {"gates": torch.nn.Linear(1, 1)})
    runner.evaluation = SimpleNamespace(representations=lambda **kwargs: calls.append(kwargs))
    runner.save = lambda *args, **kwargs: None
    runner.events()
    assert calls == [{"spectra": False}]


def test_retained_probe_tensors_obey_memory_budget(tmp_path):
    config = resolve_process_config({"gen2": {"diagnostics": {"tensor_memory_budget_mb": .0001}}})
    evaluation = Evaluation(None, config, CollectRecorder(), tmp_path)
    evaluation.examples = [{"z0": torch.zeros(20)}]
    assert evaluation.retained_bytes() == 80
    with pytest.raises(MemoryError): evaluation._check_memory(40)


def test_examples_cover_every_schema_leaf_and_pin_unchanged_reference():
    import hashlib
    import yaml
    from extensions.gen2_trainer.config import schema_leaves, SPEC_SHA256
    extension = Path(__file__).resolve().parents[1]
    assert hashlib.sha256((extension / "docs/specs/gen2_trainer_v1.md").read_bytes()).hexdigest() == SPEC_SHA256
    for name, expected in (("train_gen2_ideogram4.example.yaml", (2130, 420, 420, 450)),
                           ("smoke_gen2_ideogram4.example.yaml", (3, 2, 2, 1))):
        document = yaml.safe_load((extension / "config" / name).read_text(encoding="utf-8"))
        assert document["job"] == "extension"
        raw = document["config"]["process"][0]
        for key in schema_leaves():
            cursor = raw
            for part in key.split("."):
                cursor = cursor[part]
        resolved = resolve_process_config(raw)
        assert tuple(resolved["_gen2_resolved"]["family_horizons"].values()) == expected


def test_fixed_packet_manifest_links_serialized_tensor_bytes(tmp_path):
    from extensions.gen2_trainer.recording import sha256
    evaluation = Evaluation(None, resolve_process_config({}), CollectRecorder(), tmp_path)
    evaluation.examples = [{"sample_id": "p", "q": "cat", "z0": torch.ones(1, 1), "noise": torch.zeros(1, 1)}]
    evaluation._write_probe_packet()
    manifest = json.loads((tmp_path / "probe_manifest.json").read_text())
    assert manifest["packet_sha256"] == sha256(tmp_path / manifest["packet"])
    loaded = torch.load(tmp_path / manifest["packet"], weights_only=True)
    torch.testing.assert_close(loaded["examples"][0]["z0"], torch.ones(1, 1))
