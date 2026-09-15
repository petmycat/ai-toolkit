"""Bounded hashing and explicit environment provenance; never dump environment secrets."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return b""


def code_identity():
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        source.update(path.relative_to(ROOT).as_posix().encode())
        source.update(path.read_bytes())
    return {"git_revision": _git("rev-parse", "HEAD").decode().strip() or None,
            "dirty_diff_sha256": hashlib.sha256(_git("diff", "--binary", "HEAD")).hexdigest(),
            "extension_source_sha256": source.hexdigest()}


def environment_manifest():
    packages = {}
    for name in ("torch", "torchvision", "transformers", "diffusers", "accelerate", "bitsandbytes",
                 "optimum-quanto", "safetensors", "numpy", "prodigyopt", "lion-pytorch", "dadaptation"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {**code_identity(), "python": sys.version, "platform": platform.platform(),
              "packages": packages, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpus": []}
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            result["gpus"].append({"index": index, "name": props.name, "memory_bytes": props.total_memory,
                                   "compute_capability": [props.major, props.minor]})
    try:
        result["driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=10).decode().strip()
    except (OSError, subprocess.SubprocessError):
        result["driver"] = None
    return result


def frozen_state_hash(module, exclude_parameters=()):
    """Hash all original state in 8 MiB transfers, including quantizer buffers.

Quantized module state_dict exposes its stored data/scales; no dense model copy
is constructed. A backend with opaque state must fail rather than claim a full
weight check after hashing only a sample.
"""
    excluded = {id(p) for p in exclude_parameters}
    omitted = {name for name, p in module.named_parameters() if id(p) in excluded}
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if name in omitted:
            continue
        digest.update(name.encode())
        if not torch.is_tensor(value):
            digest.update(json.dumps(value, sort_keys=True, default=str).encode())
            continue
        tensor = value.detach()
        # Native quantizer serialization should expose ordinary storage tensors.
        if type(tensor) is not torch.Tensor and hasattr(tensor, "dequantize"):
            raise RuntimeError(f"Cannot verify opaque frozen state {name}; native serialized quantizer tensors are required")
        digest.update(f"{tensor.dtype}:{tuple(tensor.shape)}".encode())
        flat = tensor.reshape(-1)
        elements = max(1, (8 * 1024 * 1024) // tensor.element_size())
        for offset in range(0, flat.numel(), elements):
            chunk = flat[offset:offset + elements].contiguous().cpu().view(torch.uint8)
            digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()
