from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = 1


def artifact_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_conditioning_cache(path: str | Path, entries: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    root = Path(path)
    root.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "metadata": metadata, "entries": entries}, root)


def load_conditioning_cache(path: str | Path, expected_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Gen2 conditioning cache schema")
    actual = payload.get("metadata") or {}
    for key, expected in expected_metadata.items():
        if actual.get(key) != expected:
            raise ValueError(f"conditioning cache fingerprint mismatch for {key}")
    return payload["entries"]
