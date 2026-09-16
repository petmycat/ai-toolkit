"""Original unconditional model lifecycle and identity checks using CPU fixtures."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
from unittest.mock import Mock

import pytest
import torch

from extensions.gen2_trainer import process, provenance
from extensions.gen2_trainer.backend_ideogram4 import Ideogram4Backend
from extensions.gen2_trainer.checkpointing import CheckpointError, CheckpointManager
from extensions.gen2_trainer.config import ROLES, SPEC_SHA256, resolve_process_config
from extensions.gen2_trainer.diagnostics import capture_rng_state
from extensions.gen2_trainer.package import load_package, tokenizer_identity


def frozen_linear(value):
    layer = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.fill_(value)
    layer.register_buffer("quantizer_scale", torch.tensor([.125]))
    return layer.eval().requires_grad_(False)


class FrozenModels:
    def __init__(self, trace=None):
        self.trace = trace if trace is not None else []
        self.transformer = frozen_linear(1)
        self.text_encoder = frozen_linear(2)
        self.text_encoder.config = SimpleNamespace(_commit_hash="encoder-fixture")
        self.vae = frozen_linear(3)
        self.unconditional_lora = None
        self.device_torch = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.tokenizer = SimpleNamespace(name_or_path="tokenizer-fixture", chat_template="fixture",
                                         get_vocab=lambda: {"word": 1})
        self.noise_scheduler = SimpleNamespace(set_train_timesteps=lambda *a, **kw: self.trace.append("scheduler"))

    def load_model(self):
        self.trace.append("native_load")


class BackendFixture:
    def __init__(self, model):
        self.model = model
        self.parts = {role: torch.nn.Linear(2, 2, bias=False) for role in ROLES}
        with torch.no_grad():
            for component in self.parts.values():
                component.weight.fill_(-2)

    def module_manifest(self):
        return []

    def components(self):
        return self.parts

    def parameter_families(self):
        return {role: list(module.parameters()) for role, module in self.parts.items()}

    def assert_frozen(self):
        original = getattr(self.model, "unconditional_transformer", None)
        if original is not None:
            assert not any(p.requires_grad for p in original.parameters())


def stub_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)


def stub_native_models(monkeypatch, trace, original_value=4):
    models = []

    def create_model(*args, **kwargs):
        trace.append("construct")
        model = FrozenModels(trace)
        models.append(model)
        return model

    def load_original(model, gen2_config):
        assert trace[-1] == "native_load"
        trace.append("original_load")
        if gen2_config["inference"].get("unconditional_model_path") is not None:
            model.unconditional_transformer = frozen_linear(original_value)

    stub_module(monkeypatch, "extensions_built_in.diffusion_models.ideogram4.ideogram4",
                Ideogram4Model=create_model)
    stub_module(monkeypatch, "extensions.gen2_trainer.original_unconditional",
                load_original_unconditional=load_original)
    return models


def test_full_model_hashes_preserve_legacy_keys_and_cover_original_buffers():
    model = FrozenModels()
    legacy = provenance.frozen_model_hashes(model)
    assert legacy == {"diffusion": provenance.frozen_state_hash(model.transformer),
        "text_encoder": provenance.frozen_state_hash(model.text_encoder),
        "vae": provenance.frozen_state_hash(model.vae), "unconditional_lora": None}
    model.unconditional_transformer = None
    assert provenance.frozen_model_hashes(model) == legacy
    model.unconditional_transformer = frozen_linear(4)
    initial = provenance.frozen_model_hashes(model)
    assert set(initial) == set(legacy) | {"unconditional_transformer"}
    assert {key: initial[key] for key in legacy} == legacy
    model.unconditional_transformer.quantizer_scale.add_(1)
    changed = provenance.frozen_model_hashes(model)
    assert changed["unconditional_transformer"] != initial["unconditional_transformer"]
    assert {key: changed[key] for key in legacy} == legacy


def test_resume_contract_keeps_legacy_default_but_pins_original_source():
    resolved = resolve_process_config({})
    legacy = deepcopy(resolved)
    del legacy["gen2"]["inference"]["unconditional_model_path"]
    assert process.resume_contract(resolved) == process.resume_contract(legacy)
    original = deepcopy(resolved)
    original["gen2"]["inference"]["unconditional_model_path"] = "local-original"
    contract = process.resume_contract(original)
    assert contract != process.resume_contract(legacy)
    assert contract["gen2"]["inference"]["unconditional_model_path"] == "local-original"
    assert resolved["gen2"]["inference"]["unconditional_model_path"] is None


def test_process_loads_original_before_hashes_and_adapter_construction(monkeypatch, tmp_path):
    trace = []
    models = stub_native_models(monkeypatch, trace)
    stub_module(monkeypatch, "toolkit.accelerator",
                get_accelerator=lambda: SimpleNamespace(num_processes=1, scaler=None))
    stub_module(monkeypatch, "toolkit.logging_aitk", create_logger=Mock())
    stub_module(monkeypatch, "extensions.gen2_trainer.text_preflight",
                load_tokenizer=lambda config: object(),
                build_token_report=lambda *a: {"passed": True, "captions_checked": 1,
                    "failures": [], "original_token_budget": 2044},
                require_token_report=lambda *a: None)
    monkeypatch.setattr(process, "native_configuration", lambda config: {"logging": object(), "model": object()})
    monkeypatch.setattr(process, "Recorder", Mock())
    monkeypatch.setattr(process, "preflight_datasets", lambda config: [{"sample_id": "fixture"}])
    monkeypatch.setattr(process.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(process.torch, "manual_seed", Mock())
    monkeypatch.setattr(process.torch, "use_deterministic_algorithms", Mock())
    actual_hashes = provenance.frozen_model_hashes

    def hash_models(model, *args):
        assert hasattr(model, "unconditional_transformer")
        trace.append("hashes")
        return actual_hashes(model, *args)

    class BackendReached(Exception):
        pass

    def backend_factory(model, *args, **kwargs):
        assert trace[-1] == "hashes"
        trace.append("backend")
        raise BackendReached

    monkeypatch.setattr(provenance, "frozen_model_hashes", hash_models)
    monkeypatch.setattr(Ideogram4Backend, "from_native", staticmethod(backend_factory))
    runner = process.Gen2Runner({"training_folder": str(tmp_path),
        "gen2": {"inference": {"unconditional_model_path": "local-original"}}}, "ordering")
    with pytest.raises(BackendReached):
        runner._load()
    assert trace == ["construct", "native_load", "original_load", "scheduler", "hashes", "backend"]
    assert runner.initial_frozen_hashes == actual_hashes(models[0])
    assert "unconditional_transformer" in runner.initial_frozen_hashes


def test_final_frozen_check_detects_original_unconditional_mutation(tmp_path):
    model = FrozenModels()
    model.unconditional_transformer = frozen_linear(4)
    runner = process.Gen2Runner({"training_folder": str(tmp_path)}, "frozen-check")
    runner.backend = BackendFixture(model)
    runner.engine = SimpleNamespace(logical_update=1)
    runner.recorder = Mock()
    runner.initial_frozen_hashes = provenance.frozen_model_hashes(model)
    runner._verify_frozen()
    assert runner.frozen_verified_at == 1
    assert runner.recorder.event.call_args.kwargs["hashes"]["unconditional_transformer"] == \
        runner.initial_frozen_hashes["unconditional_transformer"]
    model.unconditional_transformer.quantizer_scale.add_(1)
    runner.engine.logical_update = 2
    with pytest.raises(RuntimeError, match="Full original model state changed"):
        runner._verify_frozen()
    assert runner.frozen_verified_at == 1


def save_package_fixture(tmp_path, original_enabled):
    config = resolve_process_config({"gen2": {"inference": {
        "unconditional_model_path": "local-original" if original_enabled else None}}})
    if not original_enabled:
        # Simulate a package written before this additive field existed.
        del config["gen2"]["inference"]["unconditional_model_path"]
    model = FrozenModels()
    if original_enabled:
        model.unconditional_transformer = frozen_linear(4)
    backend = BackendFixture(model)
    with torch.no_grad():
        for component in backend.components().values():
            component.weight.fill_(7)
    metadata = {"resolved_config": config, "model_identities": provenance.frozen_model_hashes(model),
        "tokenizer_identity": tokenizer_identity(model), "module_mapping": []}
    spec = Path(__file__).resolve().parents[1]/"docs"/"specs"/"gen2_trainer_v1.md"
    manager = CheckpointManager(tmp_path/"checkpoints", spec, SPEC_SHA256)
    state = {"committed_update": 0, "accumulation_status": "boundary", "failed": False,
        "optimizers": {role: {} for role in ROLES}, "schedulers": {role: {} for role in ROLES}}
    return manager.save(0, backend.components(), state, capture_rng_state(), metadata), metadata


@pytest.mark.parametrize("original_enabled,actual_original_value,should_pass", [
    (False, 4, True), (True, 4, True), (True, 5, False)])
def test_package_original_identity_and_legacy_loading(monkeypatch, tmp_path,
        original_enabled, actual_original_value, should_pass):
    checkpoint, metadata = save_package_fixture(tmp_path, original_enabled)
    trace, backends = [], []
    models = stub_native_models(monkeypatch, trace, actual_original_value)
    monkeypatch.setattr(process, "native_configuration", lambda config: {"model": object()})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("extensions.gen2_trainer.diagnostics.isolated_rng", lambda seed: nullcontext())
    actual_hashes = provenance.frozen_model_hashes

    def hash_models(model):
        assert trace[-1] == "original_load"
        trace.append("hashes")
        return actual_hashes(model)

    def backend_factory(model, *args, **kwargs):
        assert trace[-1] == "hashes"
        trace.append("backend")
        backend = BackendFixture(model)
        backends.append(backend)
        return backend

    monkeypatch.setattr(provenance, "frozen_model_hashes", hash_models)
    monkeypatch.setattr(Ideogram4Backend, "from_native", staticmethod(backend_factory))
    if should_pass:
        loaded = load_package(checkpoint)
        assert loaded.manifest["metadata"]["model_identities"] == metadata["model_identities"]
        assert actual_hashes(models[0]) == metadata["model_identities"]
        for component in loaded.backend.components().values():
            assert torch.equal(component.weight, torch.full_like(component.weight, 7))
            assert not component.weight.requires_grad
    else:
        with pytest.raises(CheckpointError, match="metadata mismatch: model_identities"):
            load_package(checkpoint)
        # Full original model identity is checked before restoring any masters.
        for component in backends[0].components().values():
            assert torch.equal(component.weight, torch.full_like(component.weight, -2))
    assert trace == ["construct", "native_load", "original_load", "hashes", "backend"]
