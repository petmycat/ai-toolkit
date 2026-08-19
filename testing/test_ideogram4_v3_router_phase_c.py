import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from extensions_built_in.ideogram4_v3_router_phase_c.config import load_config
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
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
    assert config.steps == 500
    assert config.conditioning_source == "projected_private_activator_states"
    assert config.activator_token_count == 4
    assert config.activator_occurrence_count == 3
    assert config.activator_occurrence_mode == "additive"
    assert config.activator_token_dim == 4
    assert config.temporal_anchor_count == 16
    assert config.contextual_rank == 4
    assert config.same_timestep_content_batch == 4
    assert config.gradient_checkpointing.enabled
    assert config.gradient_checkpointing.every_n_blocks == 1
    assert config.timestep_sampling == "stratified_uniform"
    assert config.timestep_bins == 16
    assert config.q_max == 0.5
    assert config.optimizer == "adamw8bit"
    assert config.lambda_universal == pytest.approx(1.0e-3)
    assert config.lambda_contextual == pytest.approx(2.0e-3)
    assert config.temporal_smoothness.universal_only
    assert config.validation.evaluate_context_shuffle


def test_config_allows_reasonable_overrides_but_rejects_invalid_ranges():
    raw = _config()
    raw.update({"lambda_contextual": 3.0e-3, "validation": {"every": 20}, "temporal_smoothness": {"weight": 2.0e-4}})
    config = load_config(raw)
    assert config.lambda_contextual == pytest.approx(3.0e-3)
    assert config.validation.every == 20
    assert config.temporal_smoothness.weight == pytest.approx(2.0e-4)

    raw = _config()
    raw["q_max"] = 0.75
    with pytest.raises(ValueError, match="q_max"):
        load_config(raw)
    raw = _config()
    raw["activator_occurrence_count"] = 0
    with pytest.raises(ValueError, match="occurrence_count"):
        load_config(raw)
    raw = _config()
    raw["activator_occurrence_mode"] = "mean"
    with pytest.raises(ValueError, match="additive"):
        load_config(raw)
    raw = _config()
    raw["activator_token_dim"] = 16
    with pytest.raises(ValueError, match="32 dimensions"):
        load_config(raw)

    raw = _config()
    raw["gradient_checkpointing"] = {"enabled": True, "every_n_blocks": 2}
    assert load_config(raw).gradient_checkpointing.every_n_blocks == 2
    raw["gradient_checkpointing"]["every_n_blocks"] = 0
    with pytest.raises(ValueError, match="every_n_blocks"):
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
    raw["timestep_bins"] = 1
    with pytest.raises(ValueError, match=r"\[2,100\]"):
        load_config(raw)

    raw = _config()
    raw["validation_item_limit"] = 0
    with pytest.raises(ValueError, match="positive"):
        load_config(raw)

    raw = _config()
    raw["validation_item_limit"] = 1
    with pytest.raises(ValueError, match="context-shuffled"):
        load_config(raw)

    raw = _config()
    raw["resume"] = "false"
    with pytest.raises(ValueError, match="YAML boolean"):
        load_config(raw)

    raw = _config()
    raw["validation"] = {"log_temporal_svd": False}
    with pytest.raises(ValueError, match="scientific validation contract"):
        load_config(raw)

    raw = _config()
    raw["typo_learning_rate"] = 1.0
    with pytest.raises(ValueError, match="unknown phase_c_v2"):
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
    assert sorted(bin_index for _, bin_index in samples[:10]) == list(range(10))
    assert sorted(bin_index for _, bin_index in samples[10:]) == list(range(10))
    assert [bin_index for _, bin_index in samples[:10]] != [bin_index for _, bin_index in samples[10:]]
    for timestep, bin_index in samples:
        lower = (bin_index * 1001) // 10
        upper = ((bin_index + 1) * 1001) // 10 - 1
        assert lower <= timestep <= upper


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
    assert len(grid) == 6
    assert sum(record["grid"] == "canonical" for record in grid) == 4
    assert sum(record["grid"] == "dense" for record in grid) == 2
    records = [
        {"step": 25, "split": "validation", "grid": "canonical", "loss": 0.4},
        {"step": 25, "split": "validation", "grid": "canonical", "loss": 0.2},
        {"step": 50, "split": "validation", "grid": "canonical", "loss": 0.1},
        {"step": 50, "split": "validation", "grid": "dense", "loss": 99.0},
        {"step": 75, "split": "heldout", "grid": "canonical", "loss": 0.0},
    ]
    assert select_best_validation(records) == pytest.approx((50, 0.1))


def test_checkpoint_stride_two_preserves_gate_gradients_across_all_blocks(monkeypatch):
    module_path = (
        Path(__file__).parents[1]
        / "extensions_built_in"
        / "diffusion_models"
        / "ideogram4"
        / "src"
        / "transformer.py"
    )
    spec = importlib.util.spec_from_file_location("phase_c_test_transformer", module_path)
    transformer_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = transformer_module
    spec.loader.exec_module(transformer_module)
    model_class = transformer_module.Ideogram4Transformer2DModel
    model = object.__new__(model_class)
    torch.nn.Module.__init__(model)
    model.gradient_checkpointing = False
    model.gradient_checkpointing_every_n_blocks = 1

    calls = []

    class GateLayer(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, hidden, *args):
            from toolkit.residual_gating import current_residual_gates

            gates = current_residual_gates()
            return hidden + gates[:, :1] * float(self.index + 1)

    model.layers = torch.nn.ModuleList([GateLayer(index) for index in range(4)])
    original_checkpoint = transformer_module.checkpoint

    def recording_checkpoint(function, *args, **kwargs):
        calls.append(function)
        return original_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(transformer_module, "checkpoint", recording_checkpoint)
    model.enable_gradient_checkpointing(every_n_blocks=2)
    gates = torch.tensor([[1.0]], requires_grad=True)
    hidden = torch.zeros(1, 1, requires_grad=True)
    from toolkit.residual_gating import residual_gate_tensor_context

    with residual_gate_tensor_context(gates):
        residual_gates = transformer_module.current_residual_gates()
        for layer_index, layer in enumerate(model.layers):
            checkpoint_layer = (
                model.gradient_checkpointing
                and torch.is_grad_enabled()
                and layer_index % model.gradient_checkpointing_every_n_blocks == 0
            )
            if checkpoint_layer:
                def checkpointed_layer(hidden_states, gate_tensor, current_layer=layer):
                    with residual_gate_tensor_context(gate_tensor):
                        return current_layer(hidden_states, None, None, None, None, None)

                hidden = transformer_module.checkpoint(
                    checkpointed_layer, hidden, residual_gates, use_reentrant=False
                )
            else:
                hidden = layer(hidden, None, None, None, None, None)
    hidden.sum().backward()
    assert len(calls) == 2
    assert gates.grad.item() == pytest.approx(10.0)


def _phase_c_process_class(monkeypatch):
    class FakeBaseExtensionProcess:
        pass

    jobs_module = types.ModuleType("jobs")
    process_module = types.ModuleType("jobs.process")
    process_module.BaseExtensionProcess = FakeBaseExtensionProcess
    jobs_module.process = process_module
    monkeypatch.setitem(sys.modules, "jobs", jobs_module)
    monkeypatch.setitem(sys.modules, "jobs.process", process_module)
    sys.modules.pop("extensions_built_in.ideogram4_v3_router_phase_c.process", None)
    return importlib.import_module(
        "extensions_built_in.ideogram4_v3_router_phase_c.process"
    ).Ideogram4V3RouterPhaseCProcess


def test_batched_validation_modes_preserve_mode_and_gate_order(monkeypatch):
    process_class = _phase_c_process_class(monkeypatch)
    from toolkit.residual_gating import current_residual_gates

    instance = object.__new__(process_class)
    instance.network = types.SimpleNamespace(_residual_gate_runtime=None)

    def fake_predict(prepared, prompt, runtime_mode):
        assert prepared["latent"].shape[0] == 3
        assert len(prepared["calibrated_embeds"].text_embeds) == 3
        return current_residual_gates()[:, :1].clone()

    instance._predict = fake_predict
    prepared = {
        "latent": torch.zeros(1, 1),
        "noise": torch.zeros(1, 1),
        "noisy_latents": torch.zeros(1, 1),
        "timesteps": torch.tensor([500.0]),
        "target": torch.zeros(1, 1),
        "prompt_template": "prompt",
        "calibrated_embeds": AdvancedPromptEmbeds(text_embeds=[torch.zeros(2, 4)]),
    }
    predictions = instance._predict_validation_modes(
        prepared,
        {
            "universal_only": torch.tensor([[0.1]]),
            "full_router": torch.tensor([[0.2]]),
            "context_shuffled": torch.tensor([[0.3]]),
        },
        "registry",
        torch,
    )
    assert list(predictions) == ["universal_only", "full_router", "context_shuffled"]
    assert predictions["universal_only"].item() == pytest.approx(1.1)
    assert predictions["full_router"].item() == pytest.approx(1.2)
    assert predictions["context_shuffled"].item() == pytest.approx(1.3)


def test_normal_v3_validation_cache_reuses_and_finally_verifies(monkeypatch):
    process_class = _phase_c_process_class(monkeypatch)

    instance = object.__new__(process_class)
    instance._normal_v3_validation_cache = {}
    calls = []
    instance._validation_q = lambda **kwargs: (
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
    )

    def fake_modes(prepared, q_by_mode, registry_fingerprint, torch_module):
        calls.append(True)
        return {"normal_v3": torch.tensor([[2.0]])}

    instance._predict_validation_modes = fake_modes
    prepared = {"target": torch.tensor([[1.0]])}
    router = object()
    first = instance._normal_v3_validation_loss(
        router, prepared, "item", 500, 42, "registry", torch, verify=False
    )
    second = instance._normal_v3_validation_loss(
        router, prepared, "item", 500, 42, "registry", torch, verify=False
    )
    verified = instance._normal_v3_validation_loss(
        router, prepared, "item", 500, 42, "registry", torch, verify=True
    )
    assert first.item() == pytest.approx(1.0)
    assert second.item() == pytest.approx(1.0)
    assert verified.item() == pytest.approx(1.0)
    assert len(calls) == 2

    instance._predict_validation_modes = lambda *args, **kwargs: {
        "normal_v3": torch.tensor([[3.0]])
    }
    with pytest.raises(RuntimeError, match="cache verification failed"):
        instance._normal_v3_validation_loss(
            router, prepared, "item", 500, 42, "registry", torch, verify=True
        )


def test_phase_c_process_exposes_timestamped_progress_output(monkeypatch):
    process_class = _phase_c_process_class(monkeypatch)
    messages = []
    print_module = types.ModuleType("toolkit.print")
    print_module.print_acc = messages.append
    monkeypatch.setitem(sys.modules, "toolkit.print", print_module)
    instance = object.__new__(process_class)
    instance._run_started_at = None
    instance._phase_c_progress("training ready")
    assert messages == ["[Phase C +0s] training ready"]


def test_fresh_run_cleanup_removes_stale_checkpoints_and_metrics(tmp_path):
    from extensions_built_in.ideogram4_v3_router_phase_c.process import Ideogram4V3RouterPhaseCProcess

    instance = object.__new__(Ideogram4V3RouterPhaseCProcess)
    instance.output_root = tmp_path
    instance.phase_c = load_config({**_config(), "output_root": str(tmp_path)})
    checkpoint = tmp_path / instance.phase_c.artifacts.checkpoints_dir / "step_000025"
    checkpoint.mkdir(parents=True)
    (checkpoint / "stale.txt").write_text("stale", encoding="utf-8")
    paths = instance._artifact_paths()
    rewrite_jsonl(paths["training_metrics"], [{"step": 25}])

    instance._sanitize_resume_artifacts(paths, 0)

    assert not (tmp_path / instance.phase_c.artifacts.checkpoints_dir).exists()
    assert load_jsonl(paths["training_metrics"]) == []


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
