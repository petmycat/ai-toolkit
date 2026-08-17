from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from safetensors.torch import save_file


LITERAL_INITIALIZATION_SCHEMA = "ai-toolkit.ideogram4-v3-activator.literal-initialization"
LITERAL_INITIALIZATION_SCHEMA_VERSION = 1
TARGET_VECTOR_COUNT = 4


@dataclass(frozen=True)
class LiteralInitializationConfig:
    enabled: bool = False
    target_vectors: int = TARGET_VECTOR_COUNT
    dtype: str = "float32"
    manifest_filename: str = "literal_initialization_manifest.json"
    artifact_filename: str = "literal_initialization.safetensors"

    def __post_init__(self) -> None:
        if self.target_vectors != TARGET_VECTOR_COUNT:
            raise ValueError("Ideogram4 literal initialization requires exactly four target vectors")
        if self.dtype != "float32":
            raise ValueError("literal initialization mapping is defined in deterministic FP32")


def interval_overlap_weights(source_count: int, target_count: int = TARGET_VECTOR_COUNT) -> torch.Tensor:
    if source_count <= 0:
        raise ValueError("source_count must be positive")
    if target_count != TARGET_VECTOR_COUNT:
        raise ValueError("Ideogram4 literal initialization requires exactly four target vectors")

    weights = torch.zeros((target_count, source_count), dtype=torch.float32)
    for target_index in range(target_count):
        target_start = target_index * source_count
        target_end = (target_index + 1) * source_count
        for source_index in range(source_count):
            source_start = source_index * target_count
            source_end = (source_index + 1) * target_count
            overlap = max(0, min(target_end, source_end) - max(target_start, source_start))
            weights[target_index, source_index] = float(overlap) / float(source_count)
    return weights


def map_literal_embeddings_to_four(token_embeddings: torch.Tensor) -> torch.Tensor:
    embeddings = torch.as_tensor(token_embeddings)
    if embeddings.ndim != 2 or embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0:
        raise ValueError("token_embeddings must have shape [positive token count, positive embedding dim]")
    embeddings = embeddings.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("token_embeddings must contain only finite values")
    weights = interval_overlap_weights(int(embeddings.shape[0])).to(device=embeddings.device)
    mapped = weights.matmul(embeddings)
    if not bool(torch.isfinite(mapped).all()):
        raise ValueError("literal initialization mapping produced non-finite values")
    return mapped.contiguous()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    normalized = tensor.detach().cpu().to(dtype=torch.float32).contiguous()
    header = canonical_json_bytes({"dtype": "float32", "shape": list(normalized.shape)})
    payload = normalized.reshape(-1).view(torch.uint8).numpy().tobytes()
    return sha256_bytes(header + b"\n" + payload)


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(manifest)))


def build_literal_initialization_manifest(
    *,
    literal: str,
    token_ids: Sequence[int],
    source_embeddings: torch.Tensor,
    mapped_embeddings: Optional[torch.Tensor] = None,
    tokenizer: Optional[Mapping[str, Any]] = None,
    config: Optional[LiteralInitializationConfig] = None,
) -> Dict[str, Any]:
    if not isinstance(literal, str) or not literal:
        raise ValueError("literal must be a non-empty string")
    source = torch.as_tensor(source_embeddings).detach().to(dtype=torch.float32)
    if source.ndim != 2 or source.shape[0] <= 0 or source.shape[1] <= 0:
        raise ValueError("source_embeddings must have shape [positive token count, positive embedding dim]")
    normalized_ids = [int(token_id) for token_id in token_ids]
    if len(normalized_ids) != source.shape[0]:
        raise ValueError("token_ids length must match source_embeddings token count")
    if any(token_id < 0 for token_id in normalized_ids):
        raise ValueError("token_ids must be non-negative")
    if not bool(torch.isfinite(source).all()):
        raise ValueError("source_embeddings must contain only finite values")

    mapped = map_literal_embeddings_to_four(source) if mapped_embeddings is None else torch.as_tensor(
        mapped_embeddings
    ).detach().to(dtype=torch.float32)
    expected_shape = (TARGET_VECTOR_COUNT, int(source.shape[1]))
    if tuple(mapped.shape) != expected_shape:
        raise ValueError(f"mapped_embeddings must have shape {expected_shape}")
    if not bool(torch.isfinite(mapped).all()):
        raise ValueError("mapped_embeddings must contain only finite values")

    resolved_config = config or LiteralInitializationConfig()
    weights = interval_overlap_weights(int(source.shape[0]))
    return {
        "schema": LITERAL_INITIALIZATION_SCHEMA,
        "schema_version": LITERAL_INITIALIZATION_SCHEMA_VERSION,
        "literal": literal,
        "token_ids": normalized_ids,
        "source_token_count": int(source.shape[0]),
        "target_vector_count": TARGET_VECTOR_COUNT,
        "embedding_dim": int(source.shape[1]),
        "mapping": {
            "name": "normalized_interval_overlap",
            "compute_dtype": "float32",
            "weights": weights.tolist(),
        },
        "source_embeddings": {
            "dtype": "float32",
            "shape": list(source.shape),
            "sha256": tensor_sha256(source),
        },
        "mapped_embeddings": {
            "dtype": "float32",
            "shape": list(mapped.shape),
            "sha256": tensor_sha256(mapped),
        },
        "tokenizer": dict(tokenizer or {}),
        "config": asdict(resolved_config),
    }


def _atomic_replace(temp_path: str, destination: Path) -> None:
    try:
        with open(temp_path, "r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        pass
    os.replace(temp_path, destination)


def atomic_save_manifest(path: os.PathLike[str] | str, manifest: Mapping[str, Any]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(dict(manifest)) + b"\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return sha256_bytes(payload[:-1])


def atomic_save_literal_initialization(
    path: os.PathLike[str] | str,
    mapped_embeddings: torch.Tensor,
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    destination = Path(path)
    if destination.suffix.lower() != ".safetensors":
        raise ValueError("literal initialization artifact path must end with .safetensors")
    mapped = torch.as_tensor(mapped_embeddings).detach().cpu().to(dtype=torch.float32).contiguous()
    if tuple(mapped.shape) != (
        TARGET_VECTOR_COUNT,
        int(manifest.get("embedding_dim", -1)),
    ):
        raise ValueError("mapped_embeddings shape does not match manifest")
    if tensor_sha256(mapped) != manifest.get("mapped_embeddings", {}).get("sha256"):
        raise ValueError("mapped_embeddings hash does not match manifest")
    if not bool(torch.isfinite(mapped).all()):
        raise ValueError("mapped_embeddings must contain only finite values")

    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_json = canonical_json_bytes(dict(manifest)).decode("utf-8")
    metadata = {
        "literal_initialization.manifest": manifest_json,
        "literal_initialization.manifest_sha256": manifest_sha256(manifest),
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(descriptor)
    try:
        save_file({"embeddings": mapped}, temporary, metadata=metadata)
        _atomic_replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {
        "path": str(destination),
        "sha256": sha256_bytes(destination.read_bytes()),
        "manifest_sha256": metadata["literal_initialization.manifest_sha256"],
    }


__all__: Tuple[str, ...] = (
    "LITERAL_INITIALIZATION_SCHEMA",
    "LITERAL_INITIALIZATION_SCHEMA_VERSION",
    "LiteralInitializationConfig",
    "TARGET_VECTOR_COUNT",
    "atomic_save_literal_initialization",
    "atomic_save_manifest",
    "build_literal_initialization_manifest",
    "canonical_json_bytes",
    "interval_overlap_weights",
    "manifest_sha256",
    "map_literal_embeddings_to_four",
    "sha256_bytes",
    "tensor_sha256",
)
