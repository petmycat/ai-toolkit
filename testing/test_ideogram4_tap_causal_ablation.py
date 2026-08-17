import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from extensions_built_in.ideogram4_tap_ablation.helpers import (
    aggregate_records,
    completed_case_keys,
    compute_layer_metrics,
    condition_specs,
    resume_key,
    select_probe_items,
    tap_mask_context,
)


TAP_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)


class FakeAdapter:
    def __init__(self, active=True):
        self.active = active


class FakeActivator:
    def __init__(self):
        self.tap_adapters = {str(layer): FakeAdapter() for layer in TAP_LAYERS}
        self.component_active = {"embedding": True, "te_adapter": True, "tap_adapters": True}
        self.runtime_mode = "full"

    def set_runtime_mode(self, mode):
        self.runtime_mode = mode
        enabled = mode != "semantic_only"
        self.component_active["tap_adapters"] = enabled
        for adapter in self.tap_adapters.values():
            adapter.active = enabled


def active_layers(activator):
    return {int(layer) for layer, adapter in activator.tap_adapters.items() if adapter.active}


def test_extension_registration_and_non_training_base(monkeypatch):
    class FakeBaseExtensionProcess:
        pass

    jobs_module = types.ModuleType("jobs")
    process_module = types.ModuleType("jobs.process")
    process_module.BaseExtensionProcess = FakeBaseExtensionProcess
    jobs_module.process = process_module
    monkeypatch.setitem(sys.modules, "jobs", jobs_module)
    monkeypatch.setitem(sys.modules, "jobs.process", process_module)
    sys.modules.pop("extensions_built_in.ideogram4_tap_ablation.process", None)

    extension_module = importlib.import_module("extensions_built_in.ideogram4_tap_ablation")
    extension = extension_module.AI_TOOLKIT_EXTENSIONS[0]
    assert extension.uid == "ideogram4_tap_causal_ablation"
    process_class = extension.get_process()
    assert issubclass(process_class, FakeBaseExtensionProcess)
    assert process_class.__mro__[1] is FakeBaseExtensionProcess
    assert not hasattr(process_class, "train")


def test_condition_generator_has_exact_28_conditions():
    conditions = condition_specs(TAP_LAYERS)
    assert len(conditions) == 28
    assert conditions[0] == {"name": "none", "kind": "none", "active_layers": []}
    assert conditions[1]["name"] == "all"
    assert len([condition for condition in conditions if condition["kind"] == "solo"]) == 13
    assert len([condition for condition in conditions if condition["kind"] == "leave_one_out"]) == 13


def test_tap_mask_resists_runtime_reset_and_restores_nested_exception_state():
    activator = FakeActivator()
    original_setter = activator.set_runtime_mode
    with tap_mask_context(activator, [3, 9]):
        assert activator.component_active["tap_adapters"] is True
        assert active_layers(activator) == {3, 9}
        activator.set_runtime_mode("full")
        assert active_layers(activator) == {3, 9}
        with pytest.raises(RuntimeError):
            with tap_mask_context(activator, [35]):
                activator.set_runtime_mode("full")
                assert activator.component_active["tap_adapters"] is True
                assert active_layers(activator) == {35}
                raise RuntimeError("boom")
        assert active_layers(activator) == {3, 9}
    assert activator.component_active == {"embedding": True, "te_adapter": True, "tap_adapters": True}
    assert active_layers(activator) == set(TAP_LAYERS)
    assert activator.set_runtime_mode == original_setter


def test_metrics_formulas_and_zero_vector_cosine_are_json_safe():
    none = torch.tensor([0.0, 0.0])
    solo = torch.tensor([1.0, 0.0])
    leave = torch.tensor([0.0, 1.0])
    full = torch.tensor([1.0, 1.0])
    target = torch.tensor([1.0, 1.0])
    metrics = compute_layer_metrics(none, full, solo, leave, target)
    assert metrics["loss_none"] == pytest.approx(1.0)
    assert metrics["loss_full"] == pytest.approx(0.0)
    assert metrics["tap_gain_full_vs_none"] == pytest.approx(1.0)
    assert metrics["interaction_gain"] == pytest.approx(0.0)
    assert metrics["marginal_gain_in_full"] == pytest.approx(0.5)
    assert metrics["marginal_gain_vs_leave_one_out"] == pytest.approx(1.0)
    assert metrics["marginal_full_cosine"] == pytest.approx(2 ** -0.5)
    zero = compute_layer_metrics(none, none, none, none, none)
    for key in ("solo_full_cosine", "marginal_full_cosine", "full_target_cosine"):
        assert zero[key] is None
        assert zero[f"{key}_valid"] is False
    json.dumps(zero, allow_nan=False)


def _write_probe(root: Path, item_id: str, caption, suffix=".json"):
    image_path = root / item_id
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake-image")
    sidecar = image_path.with_suffix(suffix)
    sidecar.write_text(json.dumps({"text": caption}) if suffix == ".json" else caption, encoding="utf-8")


def test_probe_selection_is_sorted_uses_all_heldout_and_requires_placeholder(tmp_path):
    train = [f"train/{index:02d}.png" for index in range(10)]
    heldout = [f"heldout/{index:02d}.png" for index in range(4)]
    for item_id in train + heldout:
        _write_probe(tmp_path, item_id, f"subject [trigger] {item_id}")
    manifest = {"train_item_ids": list(reversed(train)), "heldout_item_ids": list(reversed(heldout))}
    selected = select_probe_items(tmp_path, manifest, train_limit=8, expected_heldout=4)
    assert [item["dataset_relative_item_id"] for item in selected[:8]] == sorted(train)[:8]
    assert [item["dataset_relative_item_id"] for item in selected[8:]] == sorted(heldout)
    assert {item["split"] for item in selected[8:]} == {"heldout"}

    broken = heldout[0]
    (tmp_path / broken).with_suffix(".json").write_text(json.dumps({"text": "no placeholder"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain"):
        select_probe_items(tmp_path, manifest, train_limit=8, expected_heldout=4)


def _record(case_id, layer, value, timestep=100):
    return {
        "run_id": "run",
        "checkpoint_step": 125,
        "probe_case_id": case_id,
        "split": "train",
        "timestep": timestep,
        "tap_layer": layer,
        "metrics": {"tap_gain_full_vs_none": value, "nullable": None},
    }


def test_resume_key_complete_case_count_and_aggregation():
    records = [_record("case-a", layer, float(index - 6)) for index, layer in enumerate(TAP_LAYERS)]
    records.extend(_record("case-b", layer, 1.0, timestep=500) for layer in TAP_LAYERS[:-1])
    assert resume_key(records[0]) == ("run", 125, "case-a", 0)
    assert completed_case_keys(records, TAP_LAYERS) == {("run", 125, "case-a")}

    aggregates = aggregate_records(records)
    broad = next(
        row for row in aggregates
        if row["group"] == "checkpoint_split_layer" and row["tap_layer"] == TAP_LAYERS[0]
    )
    stats = broad["metrics"]["tap_gain_full_vs_none"]
    assert stats["count"] == 2
    assert stats["mean"] == pytest.approx((-6.0 + 1.0) / 2.0)
    assert stats["positive_fraction"] == pytest.approx(0.5)
    assert broad["metrics"]["nullable"]["count"] == 0
    detailed = [row for row in aggregates if row["group"] == "checkpoint_split_timestep_layer"]
    assert detailed
