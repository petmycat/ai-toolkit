from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch.utils.checkpoint as checkpoint
from typing import Any, Iterable, Mapping

import torch
from torch import nn


class TriggerLocalTEAdapter(nn.Module):
    """Low-rank residual written only to expanded activator rows."""

    def __init__(self, dimension: int, rank: int = 4, alpha: float = 4.0) -> None:
        super().__init__()
        if dimension < 1 or rank < 1 or alpha < 0:
            raise ValueError("invalid trigger-local adapter dimensions")
        self.dimension = dimension
        self.rank = rank
        self.alpha = float(alpha)
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)
        self.zero_reference: tuple[torch.Tensor, ...] | None = None

    def forward(self, hidden_states: torch.Tensor, activator_mask: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3 or activator_mask.shape != hidden_states.shape[:2]:
            raise ValueError("trigger-local adapter expects hidden states and (batch, sequence) mask")
        parameter_dtype = self.down.weight.dtype
        adapter_input = hidden_states.to(dtype=parameter_dtype)
        residual = self.up(self.down(adapter_input)) * (self.alpha / self.rank)
        mask = activator_mask.to(device=hidden_states.device, dtype=residual.dtype).unsqueeze(-1)
        return (residual * mask).to(dtype=hidden_states.dtype)


@dataclass(frozen=True)
class PlaceholderOccurrence:
    index: int
    path: tuple[str | int, ...]
    start: int
    end: int


class PlaceholderContract:
    def __init__(self, placeholder: str = "[trigger]") -> None:
        if placeholder != "[trigger]":
            raise ValueError("Gen2 V1 only supports the literal [trigger] placeholder")
        self.placeholder = placeholder

    def parse(self, raw: str | dict[str, Any]) -> tuple[dict[str, Any], list[PlaceholderOccurrence]]:
        value = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(value, dict):
            raise ValueError("Ideogram caption must be a JSON object")
        occurrences: list[PlaceholderOccurrence] = []

        def visit(node: Any, path: tuple[str | int, ...]) -> Any:
            if isinstance(node, dict):
                return {key: visit(child, path + (key,)) for key, child in node.items()}
            if isinstance(node, list):
                return [visit(child, path + (index,)) for index, child in enumerate(node)]
            if isinstance(node, str):
                cursor = 0
                while True:
                    start = node.find(self.placeholder, cursor)
                    if start < 0:
                        break
                    occurrences.append(PlaceholderOccurrence(len(occurrences), path, start, start + len(self.placeholder)))
                    cursor = start + len(self.placeholder)
            return node

        visit(value, ())
        return value, occurrences

    def replace(self, raw: str | dict[str, Any], replacement: str | Iterable[str]) -> str:
        value, occurrences = self.parse(raw)
        if not occurrences:
            raise ValueError("caption contains no [trigger] placeholder")
        replacements = [replacement] if isinstance(replacement, str) else list(replacement)
        if len(replacements) == 1:
            replacements *= len(occurrences)
        if len(replacements) != len(occurrences):
            raise ValueError("replacement count must equal placeholder occurrence count")
        counter = 0

        def visit(node: Any) -> Any:
            nonlocal counter
            if isinstance(node, dict):
                return {key: visit(child) for key, child in node.items()}
            if isinstance(node, list):
                return [visit(child) for child in node]
            if isinstance(node, str):
                while self.placeholder in node:
                    node = node.replace(self.placeholder, replacements[counter], 1)
                    counter += 1
            return node

        result = visit(value)
        if counter != len(occurrences):
            raise RuntimeError("placeholder replacement traversal diverged")
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    def fingerprint(self, raw: str | dict[str, Any]) -> str:
        normalized = self.replace(raw, self.placeholder)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class SoftTokenBank(nn.Module):
    def __init__(self, tokens: int, dimension: int, initialization: torch.Tensor | None = None) -> None:
        super().__init__()
        if tokens < 1 or dimension < 1:
            raise ValueError("tokens and dimension must be positive")
        if initialization is None:
            initialization = torch.empty(tokens, dimension)
            nn.init.normal_(initialization, std=0.02)
        if tuple(initialization.shape) != (tokens, dimension):
            raise ValueError("initialization shape must match (tokens, dimension)")
        self.tokens = tokens
        self.dimension = dimension
        self.embedding = nn.Parameter(initialization.detach().clone())

    @property
    def A(self) -> nn.Parameter:
        return self.embedding

    def expand(self, occurrence_count: int) -> torch.Tensor:
        if occurrence_count < 0:
            raise ValueError("occurrence_count cannot be negative")
        return self.embedding.unsqueeze(0).expand(occurrence_count, -1, -1)


def resample_embedding_sequence(embedding: torch.Tensor, tokens: int) -> torch.Tensor:
    if embedding.ndim != 2 or embedding.shape[0] == 0:
        raise ValueError("embedding must have shape (sequence, dimension)")
    if tokens < 1:
        raise ValueError("tokens must be positive")
    if embedding.shape[0] == tokens:
        return embedding.clone()
    positions = torch.linspace(0, embedding.shape[0] - 1, tokens, device=embedding.device, dtype=torch.float32)
    left = positions.floor().long()
    right = positions.ceil().long()
    weight = (positions - left).unsqueeze(-1).to(embedding.dtype)
    return embedding[left] * (1 - weight) + embedding[right] * weight


def replace_token_spans_with_soft_tokens(
    token_embeddings: torch.Tensor,
    spans: list[tuple[int, int]],
    bank: SoftTokenBank,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    if token_embeddings.ndim != 2:
        raise ValueError("token_embeddings must have shape (sequence, dimension)")
    if token_embeddings.shape[-1] != bank.dimension:
        raise ValueError("embedding dimension does not match soft token bank")
    ordered = sorted(spans)
    pieces: list[torch.Tensor] = []
    expanded_spans: list[tuple[int, int]] = []
    cursor = 0
    output_cursor = 0
    for start, end in ordered:
        if start < cursor or end <= start or end > token_embeddings.shape[0]:
            raise ValueError("placeholder token spans must be ordered, disjoint, and in range")
        prefix = token_embeddings[cursor:start]
        pieces.append(prefix)
        output_cursor += prefix.shape[0]
        soft_tokens = bank.A.to(device=token_embeddings.device, dtype=token_embeddings.dtype)
        pieces.append(soft_tokens)
        expanded_spans.append((output_cursor, output_cursor + bank.tokens))
        output_cursor += bank.tokens
        cursor = end
    pieces.append(token_embeddings[cursor:])
    return torch.cat(pieces, dim=0), expanded_spans


def pad_expanded_embeddings(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("at least one embedding sequence is required")
    max_length = max(item.shape[0] for item in sequences)
    dimension = sequences[0].shape[-1]
    padded = sequences[0].new_zeros(len(sequences), max_length, dimension)
    mask = torch.zeros(len(sequences), max_length, dtype=torch.long, device=sequences[0].device)
    for index, item in enumerate(sequences):
        if item.shape[-1] != dimension:
            raise ValueError("all embedding sequences must have the same dimension")
        padded[index, : item.shape[0]] = item
        mask[index, : item.shape[0]] = 1
    return padded, mask


def pack_qwen_activation_features(
    captured: Mapping[int, torch.Tensor],
    activation_layers: tuple[int, ...],
) -> torch.Tensor:
    missing = set(activation_layers).difference(captured)
    if missing:
        raise RuntimeError(f"Qwen activation layers were not captured: {sorted(missing)}")
    selected = [captured[index] for index in activation_layers]
    stacked = torch.stack(selected, dim=0)
    stacked = torch.permute(stacked, (1, 2, 3, 0))
    batch_size, sequence_length = stacked.shape[:2]
    return stacked.reshape(batch_size, sequence_length, -1)


def encode_qwen_inputs_embeds(
    text_encoder,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    pos_2d: torch.Tensor,
    activation_layers: tuple[int, ...],
    activator_mask: torch.Tensor | None = None,
    trigger_local_adapter: TriggerLocalTEAdapter | None = None,
    return_details: bool = False,
    gradient_checkpointing: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from transformers.masking_utils import create_causal_mask

    language_model = text_encoder.language_model
    embedding_weight = language_model.embed_tokens.weight
    inputs_embeds = inputs_embeds.to(device=embedding_weight.device, dtype=embedding_weight.dtype)
    if inputs_embeds.dtype != embedding_weight.dtype:
        raise RuntimeError("Gen2 Qwen inputs_embeds dtype must match the Qwen embedding dtype")
    position_ids_4d = pos_2d.to(device=inputs_embeds.device)[None, ...].expand(4, pos_2d.shape[0], -1)
    text_position_ids = position_ids_4d[0]
    mrope_position_ids = position_ids_4d[1:]
    causal_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)
    if isinstance(position_embeddings, tuple):
        position_embeddings = tuple(item.to(dtype=inputs_embeds.dtype) for item in position_embeddings)
    else:
        position_embeddings = position_embeddings.to(dtype=inputs_embeds.dtype)
    captured: dict[int, torch.Tensor] = {}
    hidden_states = inputs_embeds
    tap_set = set(activation_layers)
    for layer_index, decoder_layer in enumerate(language_model.layers):
        def layer_forward(states, layer=decoder_layer):
            output = layer(
                states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            return output[0] if isinstance(output, tuple) else output

        if gradient_checkpointing and torch.is_grad_enabled() and hidden_states.requires_grad:
            hidden_states = checkpoint.checkpoint(layer_forward, hidden_states, use_reentrant=False, preserve_rng_state=False)
        else:
            hidden_states = layer_forward(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        if trigger_local_adapter is not None and activator_mask is not None:
            hidden_states = hidden_states + trigger_local_adapter(hidden_states, activator_mask).to(hidden_states.dtype)
        if layer_index in tap_set:
            captured[layer_index] = hidden_states
    features = pack_qwen_activation_features(captured, activation_layers)
    features = features * attention_mask.to(features.dtype).unsqueeze(-1)
    if return_details:
        if activator_mask is None:
            activator_mask = torch.zeros_like(attention_mask)
        activator_float = activator_mask.to(features.dtype).unsqueeze(-1)
        pooled = (features * activator_float).sum(dim=1)
        denominator = activator_mask.sum(dim=1, keepdim=True).clamp_min(1).to(features.dtype)
        ordinary_mask = (attention_mask * (1 - activator_mask)).to(features.dtype).unsqueeze(-1)
        ordinary_pooled = (features * ordinary_mask).sum(dim=1)
        ordinary_denominator = ordinary_mask.sum(dim=1).clamp_min(1.0)
        return features, pooled / denominator, ordinary_pooled / ordinary_denominator
    return features


def normalize_inline_helpers(helpers: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    result = []
    for item in helpers:
        if not isinstance(item, Mapping) or not item.get("id") or not isinstance(item.get("replacement"), str):
            raise ValueError('each inline helper must contain "id" and string "replacement"')
        replacement = item["replacement"].strip()
        if not replacement:
            raise ValueError("inline helper replacement cannot be empty")
        result.append({"id": str(item["id"]), "replacement": replacement})
    if not result:
        raise ValueError("at least one inline helper is required")
    return result
