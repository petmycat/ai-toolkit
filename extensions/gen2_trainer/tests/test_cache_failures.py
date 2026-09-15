"""Native cache failure policy and membership checks across resolutions."""
import ast
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import copy
import importlib.util
import itertools
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from extensions.gen2_trainer.data import make_native_loader


ROOT = Path(__file__).resolve().parents[3]


def native_definition(path, name, scope):
    """Execute the actual native method bodies without unused optional deps."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    selected = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(selected, str(ROOT / path), "exec"), scope)
    return scope[name]


def load_abort_dataset(monkeypatch, native_class):
    native = ModuleType("toolkit.data_loader")
    native.AiToolkitDataset = native_class
    monkeypatch.setitem(sys.modules, native.__name__, native)
    # A private module avoids retaining the isolated native base in sys.modules.
    spec = importlib.util.spec_from_file_location("gen2_cache_failure_fixture", ROOT / "extensions/gen2_trainer/native_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Gen2AbortDataset, native


def test_native_cache_cannot_drop_one_resolution_copy(monkeypatch, tmp_path):
    messages = []
    scope = {"accelerator": SimpleNamespace(main_process_first=nullcontext),
             "print_acc": messages.append, "os": os, "itertools": itertools,
             "deque": deque, "ThreadPoolExecutor": ThreadPoolExecutor,
             "tqdm": lambda **kwargs: SimpleNamespace(update=lambda *args: None, close=lambda: None)}
    native = native_definition("toolkit/dataloader_mixins.py", "LatentCachingMixin", scope)
    abort_class, _ = load_abort_dataset(monkeypatch, native)
    path = str(tmp_path / "image.png")

    def fail_decode(*args, **kwargs):
        raise OSError("fixture image decode failed")

    bad = SimpleNamespace(path=path, get_latent_path=lambda **kwargs: str(tmp_path / "missing.safetensors"),
                          load_and_process_image=fail_decode)
    dataset = abort_class.__new__(abort_class)
    dataset.dataset_config = SimpleNamespace(resolution=1280, cache_latents_num_workers=1, buckets=True)
    dataset.dataset_path = str(tmp_path)
    dataset.is_caching_latents_to_disk = True
    dataset.is_caching_latents_to_memory = False
    dataset.sd = SimpleNamespace(device="cpu")
    dataset.transform = None
    dataset.file_list = [bad]
    dataset.buckets = {"1280x1280": SimpleNamespace(file_list_idx=[0])}
    lower_resolution = SimpleNamespace(file_list=[SimpleNamespace(path=path)])

    # A union-only membership check would still see this source after removing
    # the high-resolution entry. Exercise native decode/catch/removal dispatch.
    assert {item.path for item in lower_resolution.file_list} == {path}
    with pytest.raises(RuntimeError, match="resolution 1280.*image.png.*removing training examples is forbidden"):
        dataset.cache_latents_all_latents()
    assert dataset.file_list == [bad]
    assert dataset.buckets["1280x1280"].file_list_idx == [0]
    assert any("fixture image decode failed" in message for message in messages)


@pytest.mark.parametrize("missing_resolution", [None, 768])
def test_loader_checks_membership_for_each_source_and_resolution(monkeypatch, tmp_path, missing_resolution):
    _, native = load_abort_dataset(monkeypatch, object)
    # make_native_loader imports the normal module name lazily.
    abort_module = ModuleType("extensions.gen2_trainer.native_data")
    abort_module.Gen2AbortDataset = native.AiToolkitDataset
    monkeypatch.setitem(sys.modules, abort_module.__name__, abort_module)
    config_module = ModuleType("toolkit.config_modules")
    config_module.DatasetConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    config_module.preprocess_dataset_raw_config = native_definition(
        "toolkit/config_modules.py", "preprocess_dataset_raw_config", {})
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    first = str((tmp_path / "first.png").resolve())
    second = str((tmp_path / "second.png").resolve())
    seen = []

    def create_loader(options, **kwargs):
        seen.extend(options)
        datasets = []
        for item in options:
            path = first if item.folder_path == "source-one" else second
            files = [] if item.resolution == missing_resolution else [SimpleNamespace(path=path)]
            # Repeats/flips legitimately duplicate the path within a dataset.
            datasets.append(SimpleNamespace(file_list=files * 2))
        return SimpleNamespace(datasets=datasets)

    native.get_dataloader_from_datasets = create_loader
    native.get_dataloader_datasets = lambda loader: loader.datasets
    config = {"datasets": [{"folder_path": "source-one", "resolution": [256, 768, 1280]},
                           {"folder_path": "source-two", "resolution": [512]}],
              "train": {"batch_size": 2}, "gen2": {"execution": {"training_seed": 71}}}
    original = copy.deepcopy(config)
    manifest = [{"path": first, "dataset_index": 0}, {"path": second, "dataset_index": 1}]
    if missing_resolution is not None:
        with pytest.raises(RuntimeError, match="source 0 at resolution 768.*first.png"):
            make_native_loader(config, object(), manifest)
    else:
        assert len(make_native_loader(config, object(), manifest).datasets) == 4
    assert [item.gen2_initialization_seed for item in seen] == [71, 72, 73, 74]
    assert config == original
