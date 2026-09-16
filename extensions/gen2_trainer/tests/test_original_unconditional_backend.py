"""Native tiny-model checks for a live shared LoRA on the original CFG model."""
from contextvars import ContextVar
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from extensions.gen2_trainer.backend_ideogram4 import (
    DIFFUSION_PROJECTIONS, DiffusionBranch, Ideogram4Backend,
    bind_unconditional_lora, make_gated_lora_class,
)
from extensions.gen2_trainer.conditioning import Conditioning
from extensions.gen2_trainer.gates import CubicTimeGates
from extensions.gen2_trainer.inference import generate
from extensions.gen2_trainer.provenance import frozen_state_hash
from extensions.gen2_trainer.tests.test_backend_contract import native_definitions, native_lora_class
from extensions.gen2_trainer.tests.test_checkpoint_context import native_transformer
from extensions.gen2_trainer.tests.test_visual_modes import native_sampling_utilities


class NativeAdapterNetwork(nn.Module):
    network_type = "lora"

    def __init__(self):
        super().__init__()
        self.is_active = False
        self.unet_loras = nn.ModuleList()


def fixture(bind=True):
    torch.manual_seed(73)
    backend = Ideogram4Backend.__new__(Ideogram4Backend)
    backend._branch = ContextVar("original_cfg_test_branch", default=None)
    backend._diagnostic = ContextVar("original_cfg_test_diagnostic", default=None)
    conditional = native_transformer().eval().requires_grad_(False)
    original = native_transformer().eval().requires_grad_(False)
    pristine = copy.deepcopy(original)
    backend.model = SimpleNamespace(transformer=conditional, unconditional_transformer=original,
        unconditional_lora=None, device_torch=torch.device("cpu"), torch_dtype=torch.float32)
    backend.diffusion_network = NativeAdapterNetwork()
    backend.gates = CubicTimeGates(2)
    with torch.no_grad():
        backend.gates.beta.copy_(torch.tensor([[.6, -.2, .7, -.5], [-.7, .4, -.3, .9]]))
    cls = make_gated_lora_class(native_lora_class(), backend._branch, backend._diagnostic)
    for block, layer in enumerate(conditional.layers):
        for projection in DIFFUSION_PROJECTIONS:
            path = f"layers.{block}.{projection}"
            adapter = cls(f"transformer$${path.replace('.', '$$')}", layer.get_submodule(projection),
                          lora_dim=2, alpha=4, network=backend.diffusion_network)
            adapter.gen2_block_id, adapter.gen2_original_path = block, path
            adapter.apply_to()
            with torch.no_grad():
                adapter.lora_up.weight.normal_(std=.12)
            backend.diffusion_network.unet_loras.append(adapter)
    scope = {"torch": torch, "LLM_TOKEN_INDICATOR": 3, "OUTPUT_IMAGE_INDICATOR": 2,
             "SEQUENCE_PADDING_INDICATOR": -1, "IMAGE_POSITION_OFFSET": 65536}
    native_definitions("extensions_built_in/diffusion_models/ideogram4/src/pipeline.py",
                       ["pad_text_features", "predict_velocity"], scope)
    backend.pad_text_features, backend.predict_velocity = scope["pad_text_features"], scope["predict_velocity"]
    if bind:
        backend._unconditional_mapping = bind_unconditional_lora(
            original, backend.diffusion_network, backend._branch, backend._diagnostic)
    return backend, pristine


def unconditional_branch(backend, tau, **kwargs):
    return backend.branch(tau, unconditional=True, allow_unconditional_lora=True, **kwargs)


@pytest.mark.parametrize("strength", [0., .5, 1.])
@pytest.mark.parametrize("gate_mode", ["one", "learned", "time_mean"])
def test_all_native_projection_residuals_share_strength_gate_and_exact_block(strength, gate_mode):
    backend, pristine = fixture()
    tau = torch.tensor([.15, .85])
    with torch.no_grad(), unconditional_branch(backend, tau, strength=strength, gate_mode=gate_mode) as state:
        for adapter in backend.diffusion_network.unet_loras:
            target = backend.model.unconditional_transformer.get_submodule(adapter.gen2_original_path)
            x = torch.randn(2, 3, target.in_features)
            base = pristine.get_submodule(adapter.gen2_original_path)(x)
            residual = adapter.lora_up(adapter.lora_down(x)) * adapter.scale
            expected = base + strength * state.gate_values[:, adapter.gen2_block_id, None, None] * residual
            torch.testing.assert_close(target(x), expected, rtol=1e-5, atol=2e-6)
            # The trained factors add the same delta to a different frozen base.
            conditional = adapter.orig_module_ref()(x)
            torch.testing.assert_close(target(x)-base, conditional-adapter.org_forward(x), rtol=1e-5, atol=2e-6)
    assert backend.current_branch is None and not backend.diffusion_network.is_active


def test_binding_registers_no_factors_and_preserves_all_frozen_state():
    backend, _ = fixture(bind=False)
    original, network = backend.model.unconditional_transformer, backend.diffusion_network
    parameters = {name: id(value) for name, value in original.named_parameters()}
    modules = {name: id(value) for name, value in original.named_modules()}
    original_state = {name: value.clone() for name, value in original.state_dict().items()}
    original_hash = frozen_state_hash(original)
    factor_ids = {id(value) for value in network.parameters()}
    network_state = {name: value.clone() for name, value in network.state_dict().items()}
    records = bind_unconditional_lora(original, network, backend._branch, backend._diagnostic)
    assert frozen_state_hash(original) == original_hash
    assert {name: id(value) for name, value in original.named_parameters()} == parameters
    assert {name: id(value) for name, value in original.named_modules()} == modules
    assert {id(value) for value in network.parameters()} == factor_ids
    assert not factor_ids.intersection(parameters.values())
    assert original.state_dict().keys() == original_state.keys()
    for name, expected in original_state.items():
        torch.testing.assert_close(original.state_dict()[name], expected)
    for name, expected in network_state.items():
        torch.testing.assert_close(network.state_dict()[name], expected)
    assert len(records) == 2 * len(DIFFUSION_PROJECTIONS)
    for record, adapter in zip(records, network.unet_loras):
        assert record["parameter_count"] == 0 and record["frozen_base"]
        assert record["native_key"] == record["shared_with_native_key"] == adapter.lora_name
        assert record["block_id"] == adapter.gen2_block_id
        assert record["original_path"] == "unconditional_transformer." + adapter.gen2_original_path
        target = original.get_submodule(adapter.gen2_original_path)
        assert target._gen2_shared_lora() is adapter


@pytest.mark.parametrize("assign", [False, True])
def test_optimizer_updates_and_state_reload_are_visible_immediately(assign):
    backend, pristine = fixture()
    network = backend.diffusion_network
    adapter = network.unet_loras[-1]
    target = backend.model.unconditional_transformer.get_submodule(adapter.gen2_original_path)
    x = torch.randn(2, 3, target.in_features)
    tau = torch.tensor([.2, .8])
    latents, conditioning = torch.randn(2, 4, 2, 2), backend.empty_conditioning(2, 6)
    frozen = (backend.model.transformer, backend.model.unconditional_transformer)
    base_hashes = [frozen_state_hash(component) for component in frozen]

    def complete_outputs():
        unconditional = backend.predict(latents, tau, conditioning)
        with backend.branch(tau):
            conditional = backend.predict(latents, tau, conditioning)
        return conditional, unconditional

    with torch.no_grad(), unconditional_branch(backend, tau):
        first = target(x)
        first_complete = complete_outputs()
        optimizer = torch.optim.SGD(network.parameters(), lr=.1)
        adapter.lora_up.weight.grad = torch.ones_like(adapter.lora_up.weight)
        optimizer.step()
        changed = target(x)
        changed_complete = complete_outputs()
        assert not torch.allclose(first, changed)
        assert all(not torch.allclose(old, new) for old, new in zip(first_complete, changed_complete))
        base = pristine.get_submodule(adapter.gen2_original_path)(x)
        torch.testing.assert_close(changed, base + adapter.lora_up(adapter.lora_down(x))*adapter.scale)
        state = {name: value.clone() for name, value in network.state_dict().items()}
        for name in state:
            if name.endswith("lora_up.weight"):
                state[name].fill_(.27)
        network.load_state_dict(state, assign=assign)
        reloaded = target(x)
        reloaded_complete = complete_outputs()
        assert not torch.allclose(changed, reloaded)
        assert all(not torch.allclose(old, new) for old, new in zip(changed_complete, reloaded_complete))
        torch.testing.assert_close(reloaded, base + adapter.lora_up(adapter.lora_down(x))*adapter.scale)
        assert target._gen2_shared_lora() is adapter
    assert [frozen_state_hash(component) for component in frozen] == base_hashes


@pytest.mark.parametrize("problem", ["missing_adapter", "duplicate_adapter", "bad_block", "unknown_path",
                                      "missing_projection", "shape", "alias", "conditional_target", "already_bound"])
def test_entire_mapping_validates_before_any_forward_mutation(problem):
    backend, _ = fixture(bind=False)
    original, network = backend.model.unconditional_transformer, backend.diffusion_network
    last = network.unet_loras[-1]
    if problem == "missing_adapter":
        del network.unet_loras[-1]
    elif problem == "duplicate_adapter":
        network.unet_loras.append(last)
    elif problem == "bad_block":
        last.gen2_block_id = 0
    elif problem == "unknown_path":
        last.gen2_original_path = "layers.1.unknown"
    elif problem == "missing_projection":
        del original.layers[-1].feed_forward.w3
    elif problem == "shape":
        original.layers[-1].feed_forward.w3 = nn.Linear(7, 8, bias=False)
    elif problem == "alias":
        original.layers[-1].feed_forward.w3 = original.layers[-1].feed_forward.w1
    elif problem == "conditional_target":
        original.layers[-1].feed_forward.w3 = last.orig_module_ref()
    elif problem == "already_bound":
        original.layers[-1].feed_forward.w3._gen2_shared_lora = "existing"
    before = {name: module.forward for name, module in original.named_modules()}
    markers = {name: getattr(module, "_gen2_shared_lora", None) for name, module in original.named_modules()}
    with pytest.raises(ValueError):
        bind_unconditional_lora(original, network, backend._branch, backend._diagnostic)
    assert {name: module.forward for name, module in original.named_modules()} == before
    assert {name: getattr(module, "_gen2_shared_lora", None) for name, module in original.named_modules()} == markers


def test_real_native_velocity_uses_original_for_unconditional_and_conditional_for_positive():
    backend, pristine = fixture()
    latents, tau = torch.randn(2, 4, 2, 2), torch.tensor([.3, .7])
    empty = backend.empty_conditioning(2, 6)
    features, mask = backend.pad_text_features(empty.features, torch.device("cpu"), torch.float32)
    visited = []
    native_predict = backend.predict_velocity

    def track(transformer, *args):
        visited.append(transformer)
        return native_predict(transformer, *args)

    backend.predict_velocity = track
    with torch.no_grad():
        reference = native_predict(pristine, latents, tau, features, mask)
        with unconditional_branch(backend, tau, strength=0.):
            uncond = backend.predict(latents, tau, empty)
        with backend.branch(tau, lora_enabled=False):
            positive = backend.predict(latents, tau, empty)
            direct_positive = native_predict(backend.model.transformer, latents, tau, features, mask)
    assert visited == [backend.model.unconditional_transformer, backend.model.transformer]
    torch.testing.assert_close(uncond, reference)
    torch.testing.assert_close(positive, direct_positive)
    assert not torch.allclose(uncond, positive)


def test_unconditional_branch_requires_inference_and_explicit_personalization_opt_in():
    backend, _ = fixture()
    tau = torch.tensor([.5])
    with pytest.raises(ValueError, match="inference-only"):
        with unconditional_branch(backend, tau, lora_enabled=False):
            pass
    with torch.no_grad(), pytest.raises(ValueError, match="explicit inference opt-in"):
        with backend.branch(tau, unconditional=True):
            pass
    assert backend.current_branch is None and not backend.diffusion_network.is_active


def test_predict_rejects_nonempty_text_and_reenabled_gradients():
    backend, _ = fixture()
    tau, latents = torch.tensor([.5]), torch.randn(1, 4, 2, 2)
    nonempty = Conditioning([torch.randn(1, 6)], [{"original_length": 1, "total_length": 1}], torch.zeros(1))
    with torch.no_grad(), unconditional_branch(backend, tau):
        with pytest.raises(ValueError, match="zero text tokens"):
            backend.predict(latents, tau, nonempty)
        with torch.enable_grad(), pytest.raises(RuntimeError, match="inference-only"):
            backend.predict(latents, tau, backend.empty_conditioning(1, 6))


def test_wrapped_projection_rejects_missing_wrong_and_grad_enabled_contexts():
    backend, _ = fixture()
    target = backend.model.unconditional_transformer.layers[0].attention.qkv
    x, tau = torch.randn(1, 2, target.in_features), torch.tensor([.5])
    with torch.no_grad(), pytest.raises(RuntimeError, match="unconditional branch context"):
        target(x)
    with torch.no_grad(), backend.branch(tau), pytest.raises(RuntimeError, match="unconditional branch context"):
        target(x)
    state = DiffusionBranch(tau, torch.ones(1, 2), True, 1., True, "invalid_grad")
    token = backend._branch.set(state)
    try:
        with pytest.raises(RuntimeError, match="inference-only"):
            target(x)
    finally:
        backend._branch.reset(token)


def test_nested_failure_restores_branch_network_and_diagnostic_contexts():
    backend, _ = fixture()
    tau, latents = torch.tensor([.2, .8]), torch.randn(2, 4, 2, 2)
    seen = []

    def observer(*args):
        seen.append(args)
        raise RuntimeError("observer failure")

    with torch.no_grad(), backend.diagnostics(observer):
        diagnostic = backend._diagnostic.get()
        with backend.branch(tau, name="outer") as outer:
            with pytest.raises(RuntimeError, match="observer failure"):
                with unconditional_branch(backend, tau):
                    backend.predict(latents, tau, backend.empty_conditioning(2, 6))
            assert backend.current_branch is outer
            assert backend.diffusion_network.is_active
            assert backend._diagnostic.get() is diagnostic
        assert backend.current_branch is None and not backend.diffusion_network.is_active
    assert backend._diagnostic.get() is None
    assert len(seen) == 1
    regions = seen[0][-1]
    assert regions["image"].all() and not regions["original"].any() and not regions["suffix"].any()


def test_invalid_gate_mode_preserves_an_existing_branch_and_flags():
    backend, _ = fixture()
    tau = torch.tensor([.5])
    with torch.no_grad(), backend.branch(tau, lora_enabled=False) as outer:
        with pytest.raises(ValueError, match="Unknown gate mode"):
            with unconditional_branch(backend, tau, gate_mode="invalid"):
                pass
        assert backend.current_branch is outer
        assert not backend.diffusion_network.is_active


@pytest.mark.parametrize("kind", ["original_transformer", "conditional_transformer", "conditional_with_frozen_adapter"])
def test_generate_reports_loaded_backend_source_and_shared_parameter_source(kind):
    backend, _ = fixture()
    original = kind == "original_transformer"
    correction = "fixed-correction.safetensors" if kind == "conditional_with_frozen_adapter" else None
    source = "ideogram-ai/ideogram-4-fp8" if original else None
    if not original:
        backend.model.unconditional_transformer = None
    backend.model.unconditional_lora = SimpleNamespace(is_active=False) if correction else None
    backend.model.model_config = SimpleNamespace(model_kwargs={}, unconditional_lora_path=correction)
    backend.model.vae_scale_factor = backend.model.patch_size = 1
    backend.model.decode_latents = lambda latents, **kwargs: latents[:, :3]
    backend.config = {"inference": {"unconditional_model_path": source, "lora_strength": 1.,
                                     "missing_trigger_policy": "learned_neutral"}}
    backend.trigger_word = "<style>"
    backend.encode = lambda *args, **kwargs: Conditioning(
        [torch.zeros(1, 6)], [{"original_length": 1, "total_length": 1}], torch.zeros(1))
    frozen = [backend.model.transformer]
    if original:
        frozen.append(backend.model.unconditional_transformer)
    base_hashes = [frozen_state_hash(component) for component in frozen]
    with native_sampling_utilities():
        image, metadata = generate(backend, "[trigger] cat", mode="full_uncond_half",
            width=2, height=2, steps=2, guidance=3., initial_noise=torch.full((1, 4, 2, 2), .2))
    assert image.size == (2, 2)
    assert metadata["unconditional_backend"] == kind
    assert metadata["unconditional_model_source"] == source
    assert metadata["unconditional_model_component"] == ("unconditional_transformer" if original else "transformer")
    assert metadata["unconditional_adapter"] == correction
    assert metadata["unconditional_personalization_parameter_source"] == "diffusion"
    assert metadata["unconditional_lora_strength"] == .5
    assert metadata["unconditional_image_only"] and metadata["unconditional_text_tokens"] == 0
    assert backend.current_branch is None and not backend.diffusion_network.is_active
    assert [frozen_state_hash(component) for component in frozen] == base_hashes
