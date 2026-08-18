import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

try:
    import optimum.quanto  # noqa: F401
except ImportError:
    optimum_module = types.ModuleType("optimum")
    quanto_module = types.ModuleType("optimum.quanto")

    class FakeQTensor(torch.Tensor):
        pass

    quanto_module.QTensor = FakeQTensor
    quanto_module.QBytesTensor = FakeQTensor
    optimum_module.quanto = quanto_module
    sys.modules["optimum"] = optimum_module
    sys.modules["optimum.quanto"] = quanto_module

try:
    import diffusers  # noqa: F401
except ImportError:
    diffusers_module = types.ModuleType("diffusers")
    for class_name in (
        "AutoencoderKL",
        "UNet2DConditionModel",
        "PixArtTransformer2DModel",
        "AuraFlowTransformer2DModel",
        "WanTransformer3DModel",
    ):
        setattr(diffusers_module, class_name, type(class_name, (), {}))
    sys.modules["diffusers"] = diffusers_module

from extensions_built_in.ideogram4_v3_residual_ablation.helpers import (
    REPORT_METRIC_FIELDS,
    ResidualAblationRuntime,
    aggregate_records,
    annotate_viability_metrics,
    build_helper_response_bases,
    build_module_registry,
    build_report,
    classify_lora_module,
    finite_difference_metrics,
    gate_gradients,
    helper_response_spectrum,
    orthonormal_basis,
    parse_activator_a2_contract,
    projection_metrics,
    registry_maps,
    residual_runtime_context,
    validate_complete_partition,
)

class FakeNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_active = True
        self.is_merged_in = False
        self.is_lorm = False
        self._multiplier = 1.0
        self.torch_multiplier = torch.ones(1)
        self.network_type = "lora"
        self._residual_ablation_runtime = None


class FakeRegistryModule:
    def __init__(self, name, rank=2):
        self.lora_name = name
        self.lora_dim = rank
        self.lora_down = SimpleNamespace(weight=torch.empty(rank, 4))
        self.lora_up = SimpleNamespace(weight=torch.empty(4, rank))


class FakeLoRAModule(torch.nn.Module):
    def __init__(self, name, base, network):
        super().__init__()
        self.lora_name = name
        self.lora_dim = 1
        self.lora_down = torch.nn.Linear(2, 1, bias=False)
        self.lora_up = torch.nn.Linear(1, 2, bias=False)
        self.network = network
        self.org_module = [base]

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def forward(self, x):
        original = self.org_forward(x)
        residual = self.lora_up(self.lora_down(x)) * self.network.torch_multiplier
        runtime = self.network._residual_ablation_runtime
        if runtime is not None:
            residual = runtime.apply(self, residual)
        return original + residual


def _new_a2_contract():
    return {
        "schema": "ai-toolkit.ideogram4-v3-activator-stage-contract",
        "schema_version": 1,
        "pipeline": "ideogram4_v3_activator",
        "stage": "te_calibration",
        "status": "completed",
        "return_code": 0,
        "config_snapshot": "te_calibration.yaml",
        "config_snapshot_sha256": "a" * 64,
        "sources": {"v3_weights": {"path": "v3.safetensors", "sha256": "b" * 64}},
        "artifacts": {
            "best_embedding": "best/embedding.safetensors",
            "best_te_adapter": "best/te_adapter.safetensors",
            "best_manifest": "best/best_heldout.json",
        },
        "training": {"network": "V3 active/frozen"},
    }


def test_new_activator_a2_contract_is_primary_and_requires_best_artifacts():
    parsed = parse_activator_a2_contract(_new_a2_contract())
    assert parsed["v3_weights"] == "v3.safetensors"
    assert parsed["best_embedding"].endswith("embedding.safetensors")
    assert parsed["best_manifest_sha256"] is None

    structured = _new_a2_contract()
    structured["artifacts"] = {
        "best_embedding": {"path": "best/embedding.safetensors", "sha256": "c" * 64},
        "best_te_adapter": {"path": "best/te_adapter.safetensors", "sha256": "d" * 64},
        "best_manifest": {"path": "best/best_heldout.json", "sha256": "e" * 64},
    }
    structured_parsed = parse_activator_a2_contract(structured)
    assert structured_parsed["best_embedding"] == "best/embedding.safetensors"
    assert structured_parsed["best_embedding_sha256"] == "c" * 64
    assert structured_parsed["best_manifest"] == "best/best_heldout.json"
    assert structured_parsed["best_manifest_sha256"] == "e" * 64

    invalid = _new_a2_contract()
    invalid["artifacts"].pop("best_te_adapter")
    with pytest.raises(RuntimeError, match="best_te_adapter"):
        parse_activator_a2_contract(invalid)
    legacy = {"phase": "a2", "training_schema_version": 8, "status": "completed", "return_code": 0}
    with pytest.raises(RuntimeError, match="schema"):
        parse_activator_a2_contract(legacy)


def test_extension_registration_is_diagnostics_only(monkeypatch):
    class FakeBaseExtensionProcess:
        pass

    jobs_module = types.ModuleType("jobs")
    process_module = types.ModuleType("jobs.process")
    process_module.BaseExtensionProcess = FakeBaseExtensionProcess
    jobs_module.process = process_module
    monkeypatch.setitem(sys.modules, "jobs", jobs_module)
    monkeypatch.setitem(sys.modules, "jobs.process", process_module)
    sys.modules.pop("extensions_built_in.ideogram4_v3_residual_ablation.process", None)

    extension_module = importlib.import_module("extensions_built_in.ideogram4_v3_residual_ablation")
    extension = extension_module.AI_TOOLKIT_EXTENSIONS[0]
    assert extension.uid == "ideogram4_v3_residual_ablation"
    process_class = extension.get_process()
    assert process_class.__mro__[1] is FakeBaseExtensionProcess
    assert not hasattr(process_class, "train")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("transformer$$layers$$0$$attention$$qkv", (0, "attention")),
        ("transformer$$layers$$17$$feed_forward$$w2", (17, "mlp")),
        ("transformer$$layers$$3$$adaln_modulation", (3, "adaln")),
        ("transformer$$input_proj", (None, "other")),
    ],
)
def test_classification_is_complete_without_fixed_group_count(name, expected):
    assert classify_lora_module(name) == expected


def test_registry_partitions_all_modules_and_keeps_dynamic_groups():
    modules = [
        FakeRegistryModule("transformer$$layers$$0$$attention$$qkv"),
        FakeRegistryModule("transformer$$layers$$0$$attention$$o"),
        FakeRegistryModule("transformer$$layers$$0$$feed_forward$$w1"),
        FakeRegistryModule("transformer$$layers$$2$$adaln_modulation"),
        FakeRegistryModule("transformer$$final_layer$$linear"),
    ]
    registry = build_module_registry(modules)
    summary = validate_complete_partition(registry)
    module_to_group, group_to_modules = registry_maps(registry)
    assert summary["module_count"] == 5
    assert summary["group_count"] == 4
    assert summary["kind_counts"] == {"attention": 2, "mlp": 1, "adaln": 1, "other": 1}
    assert module_to_group["transformer.layers.0.attention.qkv"] == "block:0:attention"
    assert len(group_to_modules["block:0:attention"]) == 2


def _fake_lora(name="transformer$$layers$$0$$attention$$qkv"):
    network = FakeNetwork()
    base = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.eye(2))
    lora = FakeLoRAModule(name, base, network)
    with torch.no_grad():
        lora.lora_down.weight.copy_(torch.tensor([[1.0, 0.0]]))
        lora.lora_up.weight.copy_(torch.tensor([[2.0], [0.0]]))
    lora.apply_to()
    return network, base, lora


def test_forward_hook_is_default_off_and_backward_compatible():
    network, base, _ = _fake_lora()
    x = torch.tensor([[3.0, 4.0]])
    expected = torch.tensor([[9.0, 4.0]])
    assert torch.equal(base(x), expected)
    assert network._residual_ablation_runtime is None


def test_running_stats_chunk_large_residual_without_duplicate_group_scan():
    group = "block:0:attention"
    module = FakeRegistryModule("transformer$$layers$$0$$attention$$qkv")
    runtime = ResidualAblationRuntime(
        {"transformer.layers.0.attention.qkv": group}, capture_vectors=True, sketch_size=4
    )
    residual = torch.arange(600000, dtype=torch.float16)
    runtime.apply(module, residual)
    summary = runtime.summary()
    assert summary["modules"]["transformer.layers.0.attention.qkv"]["numel"] == residual.numel()
    assert summary["groups"][group]["numel"] == residual.numel()
    assert summary["groups"][group]["call_count"] == 1
    assert runtime.captured_group_vectors()[group].numel() == 4


def test_runtime_streams_stats_and_applies_temporary_group_gate():
    network, base, lora = _fake_lora()
    runtime = ResidualAblationRuntime({"transformer.layers.0.attention.qkv": "block:0:attention"})
    runtime.set_gates({"block:0:attention": 0.5})
    with residual_runtime_context(network, runtime):
        result = base(torch.tensor([[3.0, 4.0]]))
    assert torch.equal(result, torch.tensor([[6.0, 4.0]]))
    assert network._residual_ablation_runtime is None
    summary = runtime.summary()
    stats = summary["groups"]["block:0:attention"]
    assert stats["call_count"] == 1
    assert stats["rms"] == pytest.approx((36.0 / 2.0) ** 0.5)
    assert summary["modules"][lora.lora_name.replace("$$", ".")]["max_abs"] == pytest.approx(6.0)


def test_gate_gradient_matches_quadratic_loss_without_training_lora_parameters():
    network, base, lora = _fake_lora()
    lora.requires_grad_(False)
    runtime = ResidualAblationRuntime({"transformer.layers.0.attention.qkv": "block:0:attention"}, capture=False)
    gate = torch.tensor(1.0, requires_grad=True)
    runtime.set_gates({"block:0:attention": gate})
    with residual_runtime_context(network, runtime):
        output = base(torch.tensor([[1.0, 0.0]]))
        loss = output.square().mean()
        gradient = gate_gradients(loss, {"block:0:attention": gate})
    assert gradient["block:0:attention"] == pytest.approx(6.0)
    assert all(parameter.grad is None for parameter in lora.parameters())


def test_helper_subspace_is_built_from_captured_runtime_responses():
    group = "block:0:attention"
    module = FakeRegistryModule("transformer$$layers$$0$$attention$$qkv")
    runtime = ResidualAblationRuntime(
        {"transformer.layers.0.attention.qkv": group}, capture_vectors=True, sketch_size=2
    )
    runtime.apply(module, torch.tensor([1.0, 0.0]))
    illustration = runtime.captured_group_vectors()
    runtime.reset()
    runtime.apply(module, torch.tensor([0.0, 2.0]))
    poster = runtime.captured_group_vectors()
    bases, mean_norms, spectra = build_helper_response_bases({"illustration": illustration, "poster": poster})
    assert bases[group].shape == (2, 2)
    assert mean_norms[group] == pytest.approx(1.5)
    assert spectra[group]["top_energy_fraction"] == pytest.approx(0.8)
    expected_effective_rank = torch.exp(
        -(torch.tensor(0.8) * torch.log(torch.tensor(0.8)) + torch.tensor(0.2) * torch.log(torch.tensor(0.2)))
    ).item()
    assert spectra[group]["effective_rank"] == pytest.approx(expected_effective_rank)
    assert spectra[group]["numerical_rank"] == 2
    assert spectra[group]["energy_fractions"] == pytest.approx([0.8, 0.2])

    runtime.reset()
    runtime.set_projection_basis(bases)
    runtime.apply(module, torch.tensor([3.0, 4.0]))
    projection = runtime.summary()["helper_subspace_projection"][group]
    assert projection["energy_fraction"] == pytest.approx(1.0)

    basis = orthonormal_basis([torch.tensor([1.0, 0.0])])
    metrics = projection_metrics(torch.tensor([3.0, 4.0]), basis)
    assert metrics["energy_fraction"] == pytest.approx(9.0 / 25.0)


def test_helper_response_spectrum_distinguishes_rank_one_and_diverse_banks():
    rank_one = helper_response_spectrum([
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([2.0, 0.0, 0.0]),
        torch.tensor([-3.0, 0.0, 0.0]),
    ])
    assert rank_one["top_energy_fraction"] == pytest.approx(1.0)
    assert rank_one["effective_rank"] == pytest.approx(1.0)
    assert rank_one["numerical_rank"] == 1

    diverse = helper_response_spectrum([
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
    ])
    assert diverse["top_energy_fraction"] == pytest.approx(1.0 / 3.0)
    assert diverse["effective_rank"] == pytest.approx(3.0)
    assert diverse["numerical_rank"] == 3


def test_finite_differences_use_exact_candidate_scales():
    metrics = finite_difference_metrics({0.75: 0.0625, 1.0: 0.0, 1.25: 0.0625})
    assert metrics["left_slope"] == pytest.approx(-0.25)
    assert metrics["right_slope"] == pytest.approx(0.25)
    assert metrics["central_secant"] == pytest.approx(0.0)
    assert metrics["curvature"] == pytest.approx(2.0)
    assert metrics["best_scale"] == 1.0


def test_aggregation_and_report_include_required_viability_fields_without_68_groups():
    records = []
    for split, gradient in (("train", 1.0), ("heldout", 2.0)):
        records.append({
            "run_id": "run",
            "probe_case_id": split,
            "split": split,
            "group_id": "block:0:attention",
            "kind": "attention",
            "family": "attention",
            "block_index": 0,
            "reference_type": "calibrated_activator",
            "helper_response_spectrum": {
                "top_energy_fraction": 0.9,
                "effective_rank": 1.4,
                "numerical_rank": 2,
            },
            "metrics": {
                "dL_dg": gradient,
                "residual_rms": 0.5,
                "normalized_rms": 0.25,
                "projection_p_j": 0.8,
                "magnitude_ratio_m_j": 1.1,
                "helper_top_energy_fraction": 0.9,
                "helper_effective_rank": 1.4,
                "helper_numerical_rank": 2,
                "best_scale": 1.0,
                "fd_result": "prefer_1.0",
            },
        })
    annotated = annotate_viability_metrics(records)
    assert annotated[0]["metrics"]["mean_grad"] == pytest.approx(1.5)
    assert annotated[0]["metrics"]["gradient_sign_consistency"] == pytest.approx(1.0)
    assert annotated[0]["metrics"]["heldout_agreement"] == pytest.approx(1.0)
    assert annotated[0]["metrics"]["candidate_score"] > 0
    aggregates = aggregate_records(annotated)
    assert any(row["group"] == "split_group" and row["group_id"] == "block:0:attention" for row in aggregates)
    report = build_report(annotated, {
        "module_count": 1,
        "group_count": 1,
        "kind_counts": {"attention": 1, "mlp": 0, "adaln": 0, "other": 0},
    })
    for field in REPORT_METRIC_FIELDS:
        label = field.replace("residual_rms", "RMS").replace("normalized_rms", "normalized RMS")
        if field in {"residual_rms", "normalized_rms"}:
            assert label in report
    assert "Projection `p_j`" in report
    assert "magnitude ratio `m_j`" in report
    assert "Helper response spectrum" in report
    assert "top-mode energy=0.900000" in report
    assert "effective rank=1.4000" in report
    assert "candidate score" in report
    assert "不会启动 Phase C" in report
