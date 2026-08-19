from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from toolkit.residual_gating import (
    ResidualGateRouter,
    active_registry_fingerprint,
    aggregate_activator_occurrences,
    bind_active_registry,
    build_module_registry,
    current_residual_gates,
    effective_gates,
    filter_active_registry,
    normalize_canonical_timestep,
    residual_gate_tensor_context,
    serialize_active_registry,
    timestep_fourier_features,
)


class FakeLoRA:
    def __init__(self, name, down, up):
        self.lora_name = name
        self.lora_dim = down.shape[0]
        self.lora_down = SimpleNamespace(weight=torch.nn.Parameter(down))
        self.lora_up = SimpleNamespace(weight=torch.nn.Parameter(up))


def test_router_contract_and_zero_initialized_q():
    router = ResidualGateRouter(
        group_count=3,
        conditioning_dim=8,
        activator_token_count=4,
        activator_token_dim=2,
        temporal_anchor_count=4,
        contextual_rank=2,
    )
    tau = torch.tensor([0.0, 0.25, 1.0])
    states = torch.randn(3, 4, 8)
    code = router.encode_activator(states)
    q = router(tau, code)

    assert code.shape == (3, 8)
    assert q.shape == (3, 3)
    assert torch.equal(q, torch.zeros_like(q))
    assert torch.equal(router.universal_anchors, torch.zeros_like(router.universal_anchors))
    assert torch.equal(router.context_out, torch.zeros_like(router.context_out))
    assert router.context_in.abs().sum() > 0


def test_additive_multi_occurrence_aggregation_preserves_virtual_token_slots():
    projected = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(12, 2)
    aggregated = aggregate_activator_occurrences(
        projected, occurrence_count=3, token_count=4, mode="additive"
    )
    expected = projected.reshape(3, 4, 2).sum(dim=0, keepdim=True)
    assert aggregated.shape == (1, 4, 2)
    assert torch.equal(aggregated, expected)


def test_occurrence_aggregation_rejects_wrong_position_count_and_mode():
    with pytest.raises(ValueError, match="expected 12"):
        aggregate_activator_occurrences(
            torch.zeros(8, 2), occurrence_count=3, token_count=4, mode="additive"
        )
    with pytest.raises(ValueError, match="only additive"):
        aggregate_activator_occurrences(
            torch.zeros(12, 2), occurrence_count=3, token_count=4, mode="mean"
        )


def test_effective_gate_anchors_are_exact_and_lower_branch_ignores_q():
    q = torch.tensor([[-0.5, 0.0, 0.5], [0.3, -0.2, 0.1]], requires_grad=True)
    assert torch.equal(effective_gates(q, 0.0), torch.zeros_like(q))
    assert torch.equal(effective_gates(q, 0.5), torch.ones_like(q))
    assert torch.equal(effective_gates(q, 1.0), 1.0 + q)

    lower = effective_gates(q, torch.tensor([0.25, 0.5]))
    assert torch.equal(lower[0], torch.full_like(lower[0], 0.5))
    assert torch.equal(lower[1], torch.ones_like(lower[1]))
    lower.sum().backward()
    assert torch.equal(q.grad, torch.zeros_like(q))


def test_initial_style_one_is_identity_and_only_router_receives_gradient():
    router = ResidualGateRouter(group_count=2, conditioning_dim=8)
    frozen_lora = torch.nn.Linear(2, 2, bias=False)
    frozen_lora.requires_grad_(False)
    residual = frozen_lora(torch.tensor([[1.0, 2.0]])).detach()

    code = router.encode_activator(torch.randn(1, 4, 8))
    q = router(torch.tensor([0.4]), code)
    gates = effective_gates(q, 1.0)
    assert torch.equal(gates, torch.ones_like(gates))
    loss = (residual * gates[:, :1]).square().sum()
    loss.backward()

    assert all(parameter.grad is None for parameter in frozen_lora.parameters())
    assert router.universal_anchors.grad is not None
    assert router.universal_anchors.grad.abs().sum() > 0
    assert router.context_out.grad is not None
    assert router.context_out.grad.abs().sum() > 0


def test_bound_and_applied_contextual_increment_match_full_field_math():
    router = ResidualGateRouter(group_count=2, conditioning_dim=8, temporal_anchor_count=2)
    with torch.no_grad():
        router.universal_anchors.fill_(0.2)
        router.context_out.fill_(0.1)
    code = router.encode_activator(torch.randn(1, 4, 8))
    universal_raw, contextual_raw, full = router.components(torch.tensor([0.5]), code)
    universal_applied = router.bound(universal_raw)
    contextual_increment = full - universal_applied
    assert torch.allclose(full, router.bound(universal_raw + contextual_raw))
    assert torch.allclose(universal_applied + contextual_increment, full)
    assert not torch.allclose(contextual_increment, router.bound(contextual_raw))


def test_temporal_smoothness_is_anchor_spacing_normalized():
    coarse = ResidualGateRouter(group_count=1, conditioning_dim=8, temporal_anchor_count=2)
    fine = ResidualGateRouter(group_count=1, conditioning_dim=8, temporal_anchor_count=5)
    with torch.no_grad():
        coarse.universal_anchors[:, 0] = torch.linspace(0.0, 1.0, 2)
        fine.universal_anchors[:, 0] = torch.linspace(0.0, 1.0, 5)
    assert coarse.universal_temporal_smoothness().item() == pytest.approx(1.0)
    assert fine.universal_temporal_smoothness().item() == pytest.approx(1.0)


def test_local_temporal_gradient_touches_only_neighboring_anchors():
    router = ResidualGateRouter(
        group_count=2,
        conditioning_dim=8,
        temporal_anchor_count=4,
        contextual_rank=2,
    )
    code = router.encode_activator(torch.randn(1, 4, 8))
    router(torch.tensor([0.5]), code).sum().backward()
    universal_grad = router.universal_anchors.grad.abs().sum(dim=1)
    contextual_grad = router.context_out.grad.abs().sum(dim=(1, 2))
    assert universal_grad.tolist()[0] == 0.0
    assert universal_grad.tolist()[3] == 0.0
    assert universal_grad[1] > 0 and universal_grad[2] > 0
    assert contextual_grad.tolist()[0] == 0.0
    assert contextual_grad.tolist()[3] == 0.0
    assert contextual_grad[1] > 0 and contextual_grad[2] > 0


def test_canonical_timestep_contract_rejects_noncanonical_inputs():
    assert normalize_canonical_timestep(torch.tensor([0.2, 0.8])).shape == (2, 1)
    assert normalize_canonical_timestep(torch.tensor([[0.2, 0.2], [0.8, 0.8]])).shape == (2, 1)
    with pytest.raises(ValueError, match="one canonical timestep"):
        normalize_canonical_timestep(torch.tensor([[0.2, 0.3]]))
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        normalize_canonical_timestep(torch.tensor([1.1]))


def test_group_gate_expansion_matches_cat_repeated_batch_layout():
    from toolkit.residual_gating import apply_group_gate

    residual = torch.ones(4, 1)
    gates = torch.tensor([[2.0], [3.0]])
    output = apply_group_gate(residual, gates, 0)
    assert output[:, 0].tolist() == [2.0, 3.0, 2.0, 3.0]


def test_active_registry_uses_fp32_finite_down_up_norms_and_binds_indices():
    modules = [
        FakeLoRA(
            "transformer$$layers$$0$$attention$$qkv",
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0], [0.0]]),
        ),
        FakeLoRA(
            "transformer$$layers$$0$$attention$$o",
            torch.tensor([[1.0, 0.0]]),
            torch.zeros(2, 1),
        ),
        FakeLoRA(
            "transformer$$layers$$1$$feed_forward$$w1",
            torch.tensor([[float("nan"), 0.0]]),
            torch.tensor([[1.0], [0.0]]),
        ),
    ]
    registry = build_module_registry(modules)
    active = filter_active_registry(registry, norm_threshold=1.0e-8)

    assert {row["group_id"] for row in active} == {"block:0:attention"}
    assert len(active) == 2
    assert all(row["group_index"] == 0 for row in active)
    assert [row["module_active"] for row in active] == [True, False]
    bound = bind_active_registry(modules, active)
    assert bound["transformer.layers.0.attention.qkv"] == 0
    assert modules[0]._residual_gate_group_index == 0
    assert modules[1]._residual_gate_group_index == 0
    assert modules[2]._residual_gate_group_index == -1

    payload = serialize_active_registry(active)
    assert payload["group_count"] == 1
    assert payload["module_count"] == 2
    assert active_registry_fingerprint(active) == active_registry_fingerprint(active)


def test_non_reentrant_checkpoint_recomputation_receives_same_gate_tensor():
    gate = torch.tensor([[0.75]], requires_grad=True)
    x = torch.tensor([[2.0]], requires_grad=True)
    seen = []

    def gated(value, explicit_gate):
        with residual_gate_tensor_context(explicit_gate):
            active = current_residual_gates()
            seen.append(active)
            return value * active

    with residual_gate_tensor_context(gate):
        output = checkpoint(gated, x, gate, use_reentrant=False)
    assert current_residual_gates() is None
    output.sum().backward()

    assert len(seen) >= 2
    assert all(item is gate for item in seen)
    assert gate.grad.item() == pytest.approx(2.0)
    assert x.grad.item() == pytest.approx(0.75)
