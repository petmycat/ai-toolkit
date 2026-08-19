import importlib
import sys
import types

import pytest
import torch

from extensions_built_in.ideogram4_v3_router_phase_c.config import load_config
from extensions_built_in.ideogram4_v3_router_phase_c.helpers import (
    DeterministicStratifiedTimestepSampler,
    detect_active_groups,
    gate_regularization,
    load_jsonl,
    rewrite_jsonl,
    select_best_validation,
    validate_registry_contract,
    validation_grid,
)


def _config():
    return {
        "a2_contract": "a2.json",
        "residual_manifest": "ablation.json",
        "registry": "registry.json",
        "dataset_root": "dataset",
        "split_manifest": "split.json",
        "output_root": "phase_c",
    }


def _registry(norm=1.0):
    return [{
        "module_index": 0,
        "module_name": "transformer.layers.0.attention.qkv",
        "block_index": 0,
        "kind": "attention",
        "group_id": "block:0:attention",
        "rank": 4,
        "down_shape": [4, 8],
        "up_shape": [8, 4],
        "down_fp32_norm": norm,
        "up_fp32_norm": norm,
        "down_finite": True,
        "up_finite": True,
    }]


def test_phase_c_defaults_match_confirmed_contract():
    config = load_config(_config())
    assert config.steps == 400
    assert config.timestep_sampling == "stratified_uniform"
    assert config.timestep_bins == 10
    assert config.timestep_embed_dim == 32
    assert config.hidden_dim == 64
    assert config.router_rank == 16
    assert config.q_max == 0.5
    assert config.optimizer == "adamw"
    assert config.optimizer_params == {}
    assert config.weight_decay == pytest.approx(1.0e-4)
    assert config.lambda_gate == pytest.approx(1.0e-3)
    assert config.temporal_smoothness.enabled
    assert config.temporal_smoothness.weight == pytest.approx(1.0e-4)
    assert config.temporal_smoothness.delta == pytest.approx(0.02)
    assert config.validation.every == 25


def test_config_allows_reasonable_overrides_but_rejects_invalid_ranges():
    raw = _config()
    raw.update({"lambda_gate": 2.0e-3, "validation": {"every": 20}, "temporal_smoothness": {"weight": 2.0e-4, "delta": 0.03}})
    config = load_config(raw)
    assert config.lambda_gate == pytest.approx(2.0e-3)
    assert config.validation.every == 20
    assert config.temporal_smoothness.weight == pytest.approx(2.0e-4)

    raw = _config()
    raw["q_max"] = 0.75
    with pytest.raises(ValueError, match="q_max"):
        load_config(raw)
    raw = _config()
    raw["router_rank"] = 128
    with pytest.raises(ValueError, match="router_rank"):
        load_config(raw)


def test_config_rejects_unknown_nonfinite_odd_and_unsafe_artifact_values():
    raw = _config()
    raw.update({"optimizer": "adamw8bit", "optimizer_params": {"betas": [0.9, 0.99]}})
    config = load_config(raw)
    assert config.optimizer == "adamw8bit"
    assert config.optimizer_params == {"betas": [0.9, 0.99]}

    raw = _config()
    raw["optimizer"] = "made_up_optimizer"
    with pytest.raises(ValueError, match="unsupported"):
        load_config(raw)

    raw = _config()
    raw["optimizer_params"] = {"lr": 0.1}
    with pytest.raises(ValueError, match="must not override"):
        load_config(raw)

    raw = _config()
    raw["learning_rate"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        load_config(raw)

    raw = _config()
    raw["timestep_embed_dim"] = 31
    with pytest.raises(ValueError, match="even"):
        load_config(raw)

    raw = _config()
    raw["timestep_bins"] = 6
    with pytest.raises(ValueError, match="divide 1000"):
        load_config(raw)

    raw = _config()
    raw["validation_item_limit"] = 0
    with pytest.raises(ValueError, match="positive"):
        load_config(raw)

    raw = _config()
    raw["typo_learning_rate"] = 1.0
    with pytest.raises(ValueError, match="unknown phase_c_router"):
        load_config(raw)

    raw = _config()
    raw["artifacts"] = {"router_filename": "../router.safetensors"}
    with pytest.raises(ValueError, match="artifact paths"):
        load_config(raw)


def test_jsonl_rewrite_supports_resume_truncation(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rewrite_jsonl(path, [{"step": 1}, {"step": 2}])
    assert load_jsonl(path) == [{"step": 1}, {"step": 2}]
    rewrite_jsonl(path, [{"step": 1}])
    assert load_jsonl(path) == [{"step": 1}]


def test_stratified_sampler_is_deterministic_and_cycles_all_bins():
    first = DeterministicStratifiedTimestepSampler(42, 10)
    second = DeterministicStratifiedTimestepSampler(42, 10)
    samples = [first.sample(step) for step in range(1, 21)]
    assert samples == [second.sample(step) for step in range(1, 21)]
    assert [bin_index for _, bin_index in samples[:10]] == list(range(10))
    assert all(bin_index * 100 <= timestep <= (1000 if bin_index == 9 else (bin_index + 1) * 100 - 1) for timestep, bin_index in samples)


def test_active_groups_come_from_actual_v3_norms_and_registry_contract():
    actual = _registry()
    recorded = {"modules": [{key: value for key, value in actual[0].items() if not key.endswith("norm") and not key.endswith("finite")} ]}
    validate_registry_contract(actual, recorded)
    assert detect_active_groups(actual) == ("block:0:attention",)
    with pytest.raises(RuntimeError, match="no active"):
        detect_active_groups(_registry(0.0))
    mismatched = {"modules": [{**recorded["modules"][0], "rank": 8}]}
    with pytest.raises(RuntimeError, match="rank"):
        validate_registry_contract(actual, mismatched)


def test_gate_contract_is_fixed_style_one_plus_q_and_regularizes_q():
    class CanonicalRouter(torch.nn.Module):
        def forward(self, timestep):
            return torch.tensor([[0.0, 0.25, 0.5]])

    q = CanonicalRouter()(torch.tensor([0.5]))
    gates = 1.0 + q
    assert gates.tolist() == [[1.0, 1.25, 1.5]]
    assert gate_regularization(q).item() == pytest.approx((0.0 + 0.25 ** 2 + 0.5 ** 2) / 3.0)


def test_validation_grid_and_best_selection_use_validation_name_only():
    grid = validation_grid([100, 500], [0, 1000], [7, 8])
    assert len(grid) == 8
    records = [
        {"step": 25, "split": "validation", "grid": "canonical", "loss": 0.4},
        {"step": 25, "split": "validation", "grid": "canonical", "loss": 0.2},
        {"step": 50, "split": "validation", "grid": "canonical", "loss": 0.1},
        {"step": 50, "split": "validation", "grid": "dense", "loss": 99.0},
        {"step": 75, "split": "heldout", "grid": "canonical", "loss": 0.0},
    ]
    assert select_best_validation(records) == pytest.approx((50, 0.1))


def test_phase_c_process_exposes_timestamped_progress_output(monkeypatch):
    class FakeBaseExtensionProcess:
        pass

    jobs_module = types.ModuleType("jobs")
    process_module = types.ModuleType("jobs.process")
    process_module.BaseExtensionProcess = FakeBaseExtensionProcess
    jobs_module.process = process_module
    monkeypatch.setitem(sys.modules, "jobs", jobs_module)
    monkeypatch.setitem(sys.modules, "jobs.process", process_module)
    sys.modules.pop("extensions_built_in.ideogram4_v3_router_phase_c.process", None)
    process_class = importlib.import_module(
        "extensions_built_in.ideogram4_v3_router_phase_c.process"
    ).Ideogram4V3RouterPhaseCProcess
    messages = []
    print_module = types.ModuleType("toolkit.print")
    print_module.print_acc = messages.append
    monkeypatch.setitem(sys.modules, "toolkit.print", print_module)
    instance = object.__new__(process_class)
    instance._run_started_at = None
    instance._phase_c_progress("training ready")
    assert messages == ["[Phase C +0s] training ready"]


def test_extension_registers_disposable_process_without_sdtrainer_semantics(monkeypatch):
    class FakeBaseExtensionProcess:
        pass

    jobs_module = types.ModuleType("jobs")
    process_module = types.ModuleType("jobs.process")
    process_module.BaseExtensionProcess = FakeBaseExtensionProcess
    jobs_module.process = process_module
    monkeypatch.setitem(sys.modules, "jobs", jobs_module)
    monkeypatch.setitem(sys.modules, "jobs.process", process_module)
    sys.modules.pop("extensions_built_in.ideogram4_v3_router_phase_c.process", None)
    extension_module = importlib.import_module("extensions_built_in.ideogram4_v3_router_phase_c")
    extension = extension_module.AI_TOOLKIT_EXTENSIONS[0]
    assert extension.uid == "ideogram4_v3_router_phase_c"
    process_class = extension.get_process()
    assert not any(base.__name__ == "SDTrainer" for base in process_class.__mro__)
