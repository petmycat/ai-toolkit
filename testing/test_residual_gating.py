from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from toolkit.residual_gating import (
    ResidualGateRouter,
    active_registry_fingerprint,
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
    router = ResidualGateRouter(group_count=3)
    tau = torch.tensor([0.0, 0.25, 1.0])
    features = timestep_fourier_features(tau)
    q = router(tau)

    assert features.shape == (3, 32)
    assert not torch.equal(features[0], features[-1])
    assert q.shape == (3, 3)
    assert torch.equal(q, torch.zeros_like(q))
    assert router.net[0].in_features == 32
    assert router.net[0].out_features == 64
    assert router.net[2].in_features == 64
    assert router.net[2].out_features == 16
    assert router.net[4].in_features == 16
    assert router.net[4].out_features == 3
    assert torch.equal(router.net[4].weight, torch.zeros_like(router.net[4].weight))
    assert torch.equal(router.net[4].bias, torch.zeros_like(router.net[4].bias))


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
    router = ResidualGateRouter(group_count=2)
    frozen_lora = torch.nn.Linear(2, 2, bias=False)
    frozen_lora.requires_grad_(False)
    residual = frozen_lora(torch.tensor([[1.0, 2.0]])).detach()

    q = router(torch.tensor([0.4]))
    gates = effective_gates(q, 1.0)
    assert torch.equal(gates, torch.ones_like(gates))
    loss = (residual * gates[:, :1]).square().sum()
    loss.backward()

    assert all(parameter.grad is None for parameter in frozen_lora.parameters())
    assert router.net[4].weight.grad is not None
    assert router.net[4].weight.grad.abs().sum() > 0
    assert router.net[4].bias.grad is not None
    assert router.net[4].bias.grad.abs().sum() > 0


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
