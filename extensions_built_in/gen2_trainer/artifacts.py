from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


def save_tensor_artifact(path: str | Path, tensors: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
    payload = {key: value.detach().cpu().contiguous() for key, value in tensors.items()}
    encoded_metadata = {key: json.dumps(value, sort_keys=True) if not isinstance(value, str) else value for key, value in metadata.items()}
    save_file(payload, str(path), metadata=encoded_metadata)


def load_tensor_artifact(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors = load_file(str(path), device="cpu")
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        raw_metadata = handle.metadata() or {}
    metadata: dict[str, Any] = {}
    for key, value in raw_metadata.items():
        try:
            metadata[key] = json.loads(value)
        except json.JSONDecodeError:
            metadata[key] = value
    return tensors, metadata
