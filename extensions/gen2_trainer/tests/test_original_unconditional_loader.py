"""Exercise the native loading seam using tiny weights and no Hub requests."""
import ast
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from extensions.gen2_trainer.diagnostics import capture_rng_state
from extensions.gen2_trainer.original_unconditional import load_original_unconditional


ROOT = Path(__file__).resolve().parents[3]
NATIVE = "extensions_built_in/diffusion_models/ideogram4/ideogram4.py"


def source_definition(path, name, namespace, owner=None):
    """Execute the actual native definition without importing optional runtimes."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = tree.body if owner is None else next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner).body
    definition = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, definition], type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace[name]


class TinyTransformer(nn.Module):
    aitk_cast_on_load = True

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2)
        self.rotary_emb = nn.Module()
        self.rotary_emb.register_buffer("inv_freq", torch.empty(1, device="meta"), persistent=False)
        self.gradient_checkpointing = True
        self.post_load_kwargs = None

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        return state_dict

    @classmethod
    def aitk_from_config(cls, config):
        return cls()

    def aitk_post_load(self, **kwargs):
        self.post_load_kwargs = kwargs
        self.quantized = kwargs["qtype"] is not None
        # Native quantization/backends may consume any of these RNG streams.
        random.random()
        np.random.random()
        torch.rand(3)

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def set_attention_backend(self, backend):
        self.attention_backend = backend


TinyTransformer.load_from_state_dict = source_definition(
    "toolkit/models/v2/_mixin.py", "load_from_state_dict",
    {"torch": torch, "flush": lambda: None}, owner="OstrisModelMixin")


def fixture(*, missing=False, unexpected=False):
    calls = []

    def component_state_dict(base, component, basename):
        calls.append((base, component, basename))
        state = {"projection.weight": torch.tensor([[1., 2.], [3., 4.]]),
                 "projection.weight_scale": torch.tensor([2., 3.]),
                 "projection.bias": torch.tensor([.2, .3])}
        if missing:
            del state["projection.bias"]
        if unexpected:
            state["unknown.weight"] = torch.ones(1)
        return state

    scope = {"torch": torch, "FP8_SCALE_SUFFIX": ".weight_scale", "print_acc": lambda *args: None}
    dequantize = source_definition(NATIVE, "_dequantize_fp8_state_dict", scope)
    loader = source_definition(NATIVE, "_load_transformer", {
        "torch": torch, "flush": lambda: None,
        "Ideogram4Config": lambda: SimpleNamespace(emb_dim=4, num_heads=2, rope_theta=10),
        "_load_component_state_dict": component_state_dict,
        "_dequantize_fp8_state_dict": dequantize,
        "Ideogram4Transformer2DModel": TinyTransformer,
    }, owner="Ideogram4Model")
    policy = {"qtype": "qfloat8", "offload": .5, "dtype": torch.float32,
              "device": torch.device("cpu"), "quantize_device": torch.device("cpu")}
    requested_policies = []

    def component_load_kwargs(component):
        requested_policies.append(component)
        return policy

    model = SimpleNamespace(transformer=TinyTransformer(), torch_dtype=torch.float32,
        device_torch=torch.device("cpu"), unconditional_lora=None,
        model_config=SimpleNamespace(low_vram=True, unconditional_lora_path=None),
        print_and_status_update=lambda *args: None, component_load_kwargs=component_load_kwargs)
    model._load_transformer = loader.__get__(model)
    return model, calls, policy, requested_policies


def configuration(source="ideogram-ai/ideogram-4-fp8"):
    return {"inference": {"unconditional_model_path": source},
            "execution": {"training_seed": 123, "dit_attention_backend": "native"}}


def assert_rng_unchanged(before):
    after = capture_rng_state()
    assert before["python"] == after["python"]
    for name in before["numpy"]:
        if name == "keys":
            assert torch.equal(before["numpy"][name], after["numpy"][name])
        else:
            assert before["numpy"][name] == after["numpy"][name]
    assert torch.equal(before["torch_cpu"], after["torch_cpu"])
    assert len(before["torch_cuda"]) == len(after["torch_cuda"])
    for old, new in zip(before["torch_cuda"], after["torch_cuda"]):
        assert torch.equal(old, new)


def test_original_component_native_scales_strict_loading_and_policy():
    model, calls, policy, requested_policies = fixture()
    before = capture_rng_state()
    conditional_state = {name: parameter.detach().clone() for name, parameter in model.transformer.named_parameters()}
    result = load_original_unconditional(model, configuration())
    assert_rng_unchanged(before)
    assert calls == [("ideogram-ai/ideogram-4-fp8", "unconditional_transformer", "diffusion_pytorch_model")]
    assert requested_policies == ["transformer"]
    assert result.post_load_kwargs == policy
    assert result.quantized
    assert result is model.unconditional_transformer and result is not model.transformer
    assert not result.training and not result.gradient_checkpointing
    assert result.attention_backend == "native"
    assert all(not parameter.requires_grad for parameter in result.parameters())
    assert all(not tensor.is_meta for tensor in list(result.parameters()) + list(result.buffers()))
    torch.testing.assert_close(result.projection.weight, torch.tensor([[2., 4.], [9., 12.]]))
    assert model.transformer.training and all(parameter.requires_grad for parameter in model.transformer.parameters())
    for name, parameter in model.transformer.named_parameters():
        torch.testing.assert_close(parameter, conditional_state[name])


def test_native_default_component_unchanged_and_invalid_component_fails_before_io():
    model, calls, _, _ = fixture()
    model._load_transformer("local-root")
    assert calls == [("local-root", "transformer", "diffusion_pytorch_model")]
    with pytest.raises(ValueError, match="Unsupported Ideogram4 transformer component"):
        model._load_transformer("local-root", component="text_encoder")
    assert len(calls) == 1


@pytest.mark.parametrize("bad_state", ["missing", "unexpected"])
def test_native_strict_state_failure_propagates_and_restores_rng(bad_state):
    model, calls, _, policies = fixture(**{bad_state: True})
    before = capture_rng_state()
    with pytest.raises(RuntimeError, match="state_dict"):
        load_original_unconditional(model, configuration())
    assert_rng_unchanged(before)
    assert len(calls) == 1 and not policies
    assert not hasattr(model, "unconditional_transformer")


@pytest.mark.parametrize("inference", [{}, {"unconditional_model_path": None}])
def test_disabled_is_no_op_without_inspecting_model_or_execution(inference):
    before = capture_rng_state()
    assert load_original_unconditional(object(), {"inference": inference}) is None
    assert_rng_unchanged(before)


@pytest.mark.parametrize("configured", [True, False])
def test_correction_adapter_conflict_fails_before_loading(configured):
    model, calls, _, _ = fixture()
    if configured:
        model.model_config.unconditional_lora_path = "correction.safetensors"
    else:
        model.unconditional_lora = object()
    with pytest.raises(ValueError, match="conflicts"):
        load_original_unconditional(model, configuration())
    assert not calls


@pytest.mark.parametrize("same_instance", [True, False])
def test_conditional_alias_is_rejected_before_post_load_or_freezing(same_instance):
    model, _, _, policies = fixture()
    result = model.transformer if same_instance else TinyTransformer()
    if not same_instance:
        result.projection.weight = model.transformer.projection.weight
    model._load_transformer = lambda *args, **kwargs: result
    with pytest.raises(ValueError, match="must be separate"):
        load_original_unconditional(model, configuration())
    assert not policies
    assert all(parameter.requires_grad for parameter in model.transformer.parameters())
    assert not hasattr(model, "unconditional_transformer")


@pytest.mark.parametrize("meta_kind", ["parameter", "buffer"])
def test_unloaded_meta_tensor_fails_without_publishing_component(meta_kind):
    model, _, _, _ = fixture()
    native_load = model._load_transformer

    def load(*args, **kwargs):
        result = native_load(*args, **kwargs)
        if meta_kind == "parameter":
            result.bad_parameter = nn.Parameter(torch.empty(2, device="meta"))
        else:
            result.register_buffer("bad_buffer", torch.empty(2, device="meta"))
        return result

    model._load_transformer = load
    before = capture_rng_state()
    with pytest.raises(ValueError, match="unloaded meta tensors"):
        load_original_unconditional(model, configuration())
    assert_rng_unchanged(before)
    assert not hasattr(model, "unconditional_transformer")


def test_loader_initialization_is_repeatable_without_advancing_training_rng():
    model, _, _, _ = fixture()
    observed = []
    native_load = model._load_transformer

    def load(*args, **kwargs):
        observed.append((random.random(), np.random.random(), torch.rand(1).item()))
        return native_load(*args, **kwargs)

    model._load_transformer = load
    load_original_unconditional(model, configuration())
    random.random(), np.random.random(), torch.rand(3)
    before = capture_rng_state()
    load_original_unconditional(model, configuration())
    assert_rng_unchanged(before)
    assert observed[0] == observed[1]


def test_local_repository_root_is_forwarded_and_attention_selected():
    model, calls, _, _ = fixture()
    config = configuration(Path("local-model-root"))
    config["execution"]["dit_attention_backend"] = "flash"
    result = load_original_unconditional(model, config)
    assert calls[0] == ("local-model-root", "unconditional_transformer", "diffusion_pytorch_model")
    assert result.attention_backend == "flash"
