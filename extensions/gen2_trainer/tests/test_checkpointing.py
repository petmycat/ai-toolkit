"""Coherent package and deterministic optimizer/RNG continuation acceptance."""
import json

import pytest
import torch

from extensions.gen2_trainer.checkpointing import CheckpointError, CheckpointManager, load_manifest
from extensions.gen2_trainer.diagnostics import capture_rng_state, restore_rng_state
from extensions.gen2_trainer.recording import sha256


ROLES = ("diffusion", "embedding", "text_adapter", "gates")


class NativeNetworkFixture(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 2, bias=False)
        self.export_calls = []

    def save_weights(self, file, dtype, metadata):
        from safetensors.torch import save_file
        self.export_calls.append((dtype, metadata))
        save_file({"native.weight": self.weight.detach().cpu().to(dtype)}, file, metadata=metadata)


def package():
    modules = {role: NativeNetworkFixture() if role in {"diffusion", "text_adapter"}
               else torch.nn.Linear(2, 2, bias=False) for role in ROLES}
    optimizers = {role: torch.optim.AdamW(module.parameters(), lr=.01) for role, module in modules.items()}
    return modules, optimizers


def engine_state(optimizers, update=0):
    return {"committed_update": update, "update_attempt": update,
            "family_steps": {role: update for role in ROLES},
            "optimizers": {role: optimizer.state_dict() for role, optimizer in optimizers.items()},
            "schedulers": {role: {"active": False, "horizon": 0, "requested": {}, "state": None}
                           for role in ROLES},
            "accumulation_status": "boundary", "failed": False, "scaler": None,
            "schedule": {"warmup_updates": 2}}


def step(modules, optimizers):
    inputs = torch.randn(3, 2)
    for role, module in modules.items():
        optimizers[role].zero_grad(set_to_none=True)
        module(inputs).square().mean().backward()
        optimizers[role].step()


def test_complete_components_native_exports_and_deterministic_resume(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_bytes(b"immutable\r\n")
    manager = CheckpointManager(tmp_path / "checkpoints", spec)
    modules, optimizers = package()
    torch.manual_seed(818)
    step(modules, optimizers)
    state = engine_state(optimizers, 1)
    saved_rng = capture_rng_state()
    checkpoint = manager.save(1, modules, state, saved_rng,
                              {"model_identities": {"model": "fixture"}, "token_layout": "suffix"},
                              recorder_state={"last_event_id": "one"}, export_dtype=torch.float16)
    assert sha256(spec) == manager.spec_sha256
    assert modules["diffusion"].export_calls[0][0] == torch.float16
    manifest = load_manifest(checkpoint)
    assert manifest["components"]["diffusion"]["native_export"] == "diffusion.safetensors"
    step(modules, optimizers)
    expected = {role: module.weight.detach().clone() for role, module in modules.items()}
    restored, restored_optimizers = package()
    loaded = manager.load(checkpoint, expected_metadata={"model_identities": {"model": "fixture"}}, components=restored)
    for role in ROLES:
        restored_optimizers[role].load_state_dict(loaded["engine_state"]["optimizers"][role])
    restore_rng_state(loaded["rng_state"])
    step(restored, restored_optimizers)
    for role in ROLES:
        assert torch.equal(expected[role], restored[role].weight)
        assert restored_optimizers[role].state_dict()["state"][0]["step"] == 2
    assert loaded["recorder_state"]["last_event_id"] == "one"


def test_strict_validation_before_live_mutation_and_corruption(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text("immutable", encoding="utf-8")
    manager = CheckpointManager(tmp_path / "checkpoints", spec)
    modules, optimizers = package()
    saved = manager.save(0, modules, engine_state(optimizers), capture_rng_state(), {"model": "original"})
    before = modules["diffusion"].weight.detach().clone()
    with pytest.raises(CheckpointError, match="metadata mismatch"):
        manager.load(saved, expected_metadata={"model": "different"}, components=modules)
    assert torch.equal(before, modules["diffusion"].weight)
    (saved / "gates.masters.safetensors").write_bytes(b"corrupted")
    with pytest.raises(CheckpointError, match="checksum"):
        manager.load(saved)


def test_incomplete_modified_spec_and_missing_optimizer_fail(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text("immutable", encoding="utf-8")
    manager = CheckpointManager(tmp_path / "checkpoints", spec)
    modules, optimizers = package()
    state = engine_state(optimizers)
    state["optimizers"].pop("embedding")
    with pytest.raises(CheckpointError, match="four optimizers"):
        manager.save(0, modules, state, capture_rng_state(), {})
    incomplete = tmp_path / "unfinished"
    incomplete.mkdir()
    with pytest.raises(CheckpointError, match="Incomplete"):
        manager.load(incomplete)
    saved = manager.save(0, modules, engine_state(optimizers), capture_rng_state(), {})
    spec.write_text("unauthorized successor", encoding="utf-8")
    with pytest.raises(CheckpointError, match="specification changed"):
        manager.load(saved)


def test_zero_horizon_marker_partial_update_and_float32_masters(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text("immutable", encoding="utf-8")
    manager = CheckpointManager(tmp_path / "checkpoints", spec)
    modules, optimizers = package()
    state = engine_state(optimizers)
    state["failed"] = True
    with pytest.raises(CheckpointError, match="failed update"):
        manager.save(0, modules, state, capture_rng_state(), {})
    state["failed"] = False
    state["schedulers"]["gates"]["horizon"] = 10
    with pytest.raises(CheckpointError, match="horizon=0"):
        manager.save(0, modules, state, capture_rng_state(), {})
    state = engine_state(optimizers)
    modules["embedding"].to(dtype=torch.float16)
    with pytest.raises(CheckpointError, match="float32"):
        manager.save(0, modules, state, capture_rng_state(), {})
    assert not (manager.root / "update_00000000").exists()


def test_retention_protects_initial_and_boundaries(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text("immutable", encoding="utf-8")
    manager = CheckpointManager(tmp_path / "checkpoints", spec, max_to_keep=1)
    modules, optimizers = package()
    for update in range(4):
        manager.save(update, modules, engine_state(optimizers, update), capture_rng_state(), {},
                     protected=update == 1, reasons=["boundary"] if update == 1 else ["interval"])
    assert {path.name for path in manager.root.glob("update_*")} == {
        "update_00000000", "update_00000001", "update_00000003"}
    with pytest.raises(CheckpointError, match="already exists"):
        manager.save(3, modules, engine_state(optimizers, 3), capture_rng_state(), {})


def test_readonly_manager_loads_without_allowing_save_or_retention(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text("immutable", encoding="utf-8")
    manager = CheckpointManager(tmp_path / "checkpoints", spec)
    modules, optimizers = package()
    saved = manager.save(0, modules, engine_state(optimizers), capture_rng_state(), {})
    readonly = CheckpointManager(manager.root, spec, read_only=True)
    assert readonly.load(saved, load_training_state=False)["manifest"]["logical_update"] == 0
    with pytest.raises(CheckpointError, match="read-only"):
        readonly.save(1, modules, engine_state(optimizers, 1), capture_rng_state(), {})
    with pytest.raises(CheckpointError, match="read-only"):
        readonly.prune()
