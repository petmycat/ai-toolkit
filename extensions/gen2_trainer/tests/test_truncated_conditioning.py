"""Token truncation through the real encoder/sampler orchestration, using tiny CPU layers.

The fixture replaces Qwen weights and its attention mask factory, not Gen2's
token-bank normalization, suffix assembly, masked adapter, or prefix checks.
"""
from contextvars import ContextVar
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import extensions.gen2_trainer.backend_ideogram4 as backend_module
from extensions.gen2_trainer.conditioning import LearnedTokenBank
from extensions.gen2_trainer.gates import CubicTimeGates
from extensions.gen2_trainer.inference import generate
from extensions.gen2_trainer.tests.test_conditioning import fixture_encoder
from extensions.gen2_trainer.tests.test_visual_modes import native_sampling_utilities


TAPS = (0, 1, 2)
FULL_LENGTH, RETAINED_LENGTH, NUM_TOKENS, LIMIT = 3233, 3068, 4, 3072
PROMPT = "A [trigger] subject with a long caption [trigger]"


class TokenizerFixture:
    truncation_side = "right"

    def __init__(self, length):
        self.length = length
        self.calls = []

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt and not tokenize
        self.messages = messages
        return "<user>" + messages[0]["content"][0]["text"] + "</user><assistant>"

    def __call__(self, serialized, *, add_special_tokens, truncation):
        assert not add_special_tokens and truncation is False
        self.calls.append(serialized)
        return {"input_ids": list(range(self.length))}


class RecordingEmbedding(nn.Embedding):
    def __init__(self):
        super().__init__(4096, 4)
        self.calls = []

    def forward(self, ids):
        self.calls.append(ids.detach().clone())
        return super().forward(ids)


@pytest.fixture
def encoding_backend(monkeypatch):
    # Isolate fixture initialization from other tests' training RNG assumptions.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(842)
        encoder = fixture_encoder()
        encoder.language_model.embed_tokens = RecordingEmbedding()
        encoder.language_model.requires_grad_(False)
        for layer in encoder.language_model.layers:
            layer.adapter.requires_grad_(True)
        bank = LearnedTokenBank(encoder.language_model.embed_tokens.weight[[1, 2]], [1, 2],
                                num_tokens=NUM_TOKENS, seed=145)

    backend = backend_module.Ideogram4Backend.__new__(backend_module.Ideogram4Backend)
    backend.config = {"conditioning": {"overflow_policy": "truncate"},
        "inference": {"lora_strength": 1., "missing_trigger_policy": "learned_neutral",
                      "unconditional_model_path": None}}
    backend.trigger_word = "<style>"
    backend.encoder_checkpointing = False
    backend._diagnostic = ContextVar("truncated_encoding_diagnostic", default=None)
    backend._branch = ContextVar("truncated_encoding_branch", default=None)
    backend.model = SimpleNamespace(text_encoder=encoder, tokenizer=TokenizerFixture(FULL_LENGTH),
        max_text_length=LIMIT, torch_dtype=torch.float32, device_torch=torch.device("cpu"))
    backend.tokens = bank
    backend.bridge_calls = []
    backend.native_calls = []
    real_bridge = backend_module.differentiable_qwen_features

    def bridge(encoder, inputs, mask, positions, suffix, **kwargs):
        backend.bridge_calls.append({"inputs": inputs.detach().clone(), "mask": mask.clone(),
            "positions": positions.clone(), "suffix": suffix.clone(),
            "adapter_enabled": kwargs.get("adapter_enabled", True)})
        return real_bridge(encoder, inputs, mask, positions, suffix, **kwargs,
            activation_layers=TAPS, causal_mask_factory=lambda **unused: None)

    def native(encoder, ids, mask, positions):
        backend.native_calls.append(ids.detach().clone())
        inputs = encoder.language_model.embed_tokens(ids)
        return real_bridge(encoder, inputs, mask, positions, torch.zeros_like(mask, dtype=torch.bool),
            adapter_enabled=False, activation_layers=TAPS, causal_mask_factory=lambda **unused: None)[0]

    backend.native_features = native
    monkeypatch.setattr(backend_module, "differentiable_qwen_features", bridge)
    monkeypatch.setattr(backend_module, "EXPECTED_ACTIVATION_LAYERS", TAPS)
    return backend


def assert_truncated_metadata(metadata, styled):
    assert metadata["original_ids"] == list(range(RETAINED_LENGTH))
    assert metadata["original_length"] == RETAINED_LENGTH
    assert metadata["untruncated_length"] == FULL_LENGTH
    assert metadata["truncated"] is True
    assert metadata["truncated_tokens"] == FULL_LENGTH-RETAINED_LENGTH == 165
    assert metadata["overflow"] is True
    assert metadata["overflow_policy"] == "truncate"
    assert metadata["truncation_side"] == "right"
    assert metadata["untruncated_ids_sha256"] == hashlib.sha256(json.dumps(list(range(FULL_LENGTH))).encode()).hexdigest()
    assert metadata["encoded_ids_sha256"] == hashlib.sha256(json.dumps(list(range(RETAINED_LENGTH))).encode()).hexdigest()
    assert metadata["total_length"] == RETAINED_LENGTH + (NUM_TOKENS if styled else 0)
    assert metadata["suffix_positions"] == (list(range(RETAINED_LENGTH, LIMIT)) if styled else [])
    assert metadata["positions"] == list(range(metadata["total_length"]))
    assert metadata["suffix_mask"] == [False]*RETAINED_LENGTH + ([True]*NUM_TOKENS if styled else [])


@pytest.mark.parametrize("adapter_enabled", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_truncated_styled_and_neutral_share_prefix_keep_all_tokens_and_backprop(encoding_backend, adapter_enabled, checkpointed):
    backend = encoding_backend
    backend.encoder_checkpointing = checkpointed
    neutral = backend.encode([PROMPT], styled=False)
    styled = backend.encode([PROMPT], styled=True, gradients=True, adapter_enabled=adapter_enabled)
    assert_truncated_metadata(neutral.metadata[0], False)
    assert_truncated_metadata(styled.metadata[0], True)
    assert neutral.features[0].shape == (RETAINED_LENGTH, 4*len(TAPS))
    assert styled.features[0].shape == (LIMIT, 4*len(TAPS))
    torch.testing.assert_close(styled.features[0][:RETAINED_LENGTH], neutral.features[0], rtol=1e-6, atol=1e-7)
    call = backend.bridge_calls[-1]
    assert call["adapter_enabled"] is adapter_enabled
    torch.testing.assert_close(call["inputs"][0, -NUM_TOKENS:], backend.tokens().detach(), rtol=0, atol=0)
    assert call["suffix"][0, -NUM_TOKENS:].all() and not call["suffix"][0, :-NUM_TOKENS].any()
    assert call["positions"][0].tolist() == list(range(LIMIT))
    assert all(ids[0].tolist() == list(range(RETAINED_LENGTH))
               for ids in backend.model.text_encoder.language_model.embed_tokens.calls)
    objective = styled.features[0][-NUM_TOKENS:, 0].sum() + .0001*styled.rt_per_example.sum()
    objective.backward()
    assert backend.tokens.U.grad is not None and torch.isfinite(backend.tokens.U.grad).all()
    assert backend.tokens.U.grad.abs().sum() > 0
    assert backend.model.text_encoder.language_model.embed_tokens.weight.grad is None
    adapter_parameters = [parameter for layer in backend.model.text_encoder.language_model.layers
                          for parameter in layer.adapter.parameters()]
    if adapter_enabled:
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in adapter_parameters)
        assert sum(parameter.grad.abs().sum() for parameter in adapter_parameters) > 0
    else:
        assert all(parameter.grad is None for parameter in adapter_parameters)
        assert styled.rt_per_example.eq(0).all()


def test_prefix_acceptance_uses_retained_ids_for_native_repeat_sibling_and_styled(encoding_backend):
    backend = encoding_backend
    rows = backend.verify_prefix([PROMPT], atol=1e-6, rtol=1e-6)
    assert len(rows) == 3*len(TAPS)
    assert all(row["passed"] for row in rows)
    assert {row["comparison"] for row in rows} == {"native_repeat", "neutral_sibling", "styled_prefix"}
    assert all(ids.shape == (1, RETAINED_LENGTH) for ids in backend.native_calls)
    assert all(ids[0].tolist() == list(range(RETAINED_LENGTH))
               for ids in backend.model.text_encoder.language_model.embed_tokens.calls)
    assert [call["inputs"].shape[1] for call in backend.bridge_calls] == [LIMIT, RETAINED_LENGTH]
    assert backend.bridge_calls[0]["suffix"].sum() == NUM_TOKENS
    assert backend.bridge_calls[1]["suffix"].sum() == 0


@pytest.mark.parametrize("styled", [False, True])
def test_error_policy_rejects_before_vocabulary_lookup_or_encoder(encoding_backend, styled):
    backend = encoding_backend
    backend.config["conditioning"]["overflow_policy"] = "error"
    with pytest.raises(ValueError, match="original_length=3233, M=4, limit=3072"):
        backend.encode([PROMPT], styled=styled)
    assert backend.model.text_encoder.language_model.embed_tokens.calls == []
    assert backend.bridge_calls == backend.native_calls == []


@pytest.mark.parametrize("styled,adapter_enabled", [(False, False), (True, False), (True, True)])
def test_short_caption_ids_and_features_identical_between_policies(encoding_backend, styled, adapter_enabled):
    backend = encoding_backend
    backend.model.tokenizer.length = 12
    expected = backend.encode([PROMPT], styled=styled, adapter_enabled=adapter_enabled)
    backend.config["conditioning"]["overflow_policy"] = "error"
    actual = backend.encode([PROMPT], styled=styled, adapter_enabled=adapter_enabled)
    torch.testing.assert_close(actual.features[0], expected.features[0], rtol=0, atol=0)
    torch.testing.assert_close(actual.rt_per_example, expected.rt_per_example, rtol=0, atol=0)
    for key in ("original_ids", "original_length", "untruncated_length", "total_length", "positions", "suffix_positions", "suffix_mask"):
        assert actual.metadata[0][key] == expected.metadata[0][key]
    assert actual.metadata[0]["original_ids"] == list(range(12))
    assert actual.metadata[0]["truncated"] is expected.metadata[0]["truncated"] is False
    assert actual.metadata[0]["truncated_tokens"] == expected.metadata[0]["truncated_tokens"] == 0
    assert actual.metadata[0]["overflow"] is expected.metadata[0]["overflow"] is False


def test_sampler_metadata_contains_actual_retained_conditioning(encoding_backend):
    backend = encoding_backend
    model = backend.model
    model.vae_scale_factor = model.patch_size = 1
    model.transformer = SimpleNamespace(config=SimpleNamespace(in_channels=3))
    model.model_config = SimpleNamespace(model_kwargs={}, unconditional_lora_path=None)
    model.unconditional_lora = None
    model.decode_latents = lambda latents, **kwargs: latents
    backend.gates = CubicTimeGates(1)
    backend.diffusion_network = SimpleNamespace(is_active=True)
    calls = []

    def predict(latents, tau, conditioning):
        calls.append((backend.current_branch.name, conditioning))
        value = conditioning.features[0].mean()*.001 if conditioning.features[0].numel() else 0.
        return latents*.01 + value

    backend.predict = predict
    with native_sampling_utilities():
        image, metadata = generate(backend, PROMPT, mode="full_uncond_half",
            width=2, height=2, steps=2, guidance=3., seed=42)
    assert image.size == (2, 2)
    assert_truncated_metadata(metadata["conditioning"], True)
    assert metadata["prompt"] == PROMPT
    assert metadata["unconditional_text_tokens"] == 0
    assert metadata["unconditional_lora_strength"] == .5
    assert len(calls) == 4
    for branch, condition in calls:
        if branch == "cfg_conditional":
            assert condition.metadata[0] == metadata["conditioning"]
            assert condition.features[0].shape[0] == LIMIT
        else:
            assert branch == "cfg_unconditional" and condition.features[0].shape[0] == 0
    assert backend.tokens.U.grad is None
