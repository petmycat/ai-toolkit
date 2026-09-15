"""Coherent Gen2 packages, strict validation, and complete training resume."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .recording import SCHEMA_VERSION, assert_writable_path, json_safe, sha256, write_json


COMPONENTS = frozenset({"diffusion", "embedding", "text_adapter", "gates"})
COMPLETE_MARKER = "COMPLETE.json"


class CheckpointError(RuntimeError):
    pass


def _safe_member(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise CheckpointError(f"Checkpoint member escapes its directory: {relative!r}")
    return path


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def load_manifest(path: str | Path, *, verify_components: bool = True,
                  expected_spec_sha256: str | None = None) -> dict:
    """Read/verify an inference manifest without importing torch or loading models."""
    root = Path(path).expanduser().resolve()
    marker = root / COMPLETE_MARKER
    manifest_path = root / "manifest.json"
    if not marker.is_file() or not manifest_path.is_file():
        raise CheckpointError(f"Incomplete checkpoint (no complete marker/manifest): {root}")
    try:
        complete = json.loads(marker.read_text(encoding="utf-8"))
        manifest = json_safe(json.loads(manifest_path.read_text(encoding="utf-8")))
    except (ValueError, TypeError) as exc:
        raise CheckpointError(f"Invalid checkpoint metadata: {root}") from exc
    actual_hash = sha256(manifest_path)
    if complete.get("manifest_sha256") != actual_hash:
        raise CheckpointError("Checkpoint manifest checksum differs from its complete marker")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointError("Unsupported Gen2 checkpoint schema")
    if expected_spec_sha256 and manifest.get("spec_sha256") != expected_spec_sha256:
        raise CheckpointError("Checkpoint immutable-spec checksum mismatch")
    if set(manifest.get("components", {})) != COMPONENTS:
        raise CheckpointError("A complete package requires diffusion, embedding, text_adapter and gates")
    files = manifest.get("files", {})
    required = {entry["masters"] for entry in manifest["components"].values()}
    required.update({"training_state.pt", "specification.md"})
    if not required.issubset(files):
        raise CheckpointError("Checkpoint file manifest omits required resume/package components")
    if verify_components:
        for relative, expected in files.items():
            member = _safe_member(root, relative)
            if not member.is_file():
                raise CheckpointError(f"Missing checkpoint component: {relative}")
            if member.stat().st_size != expected["size"] or sha256(member) != expected["sha256"]:
                raise CheckpointError(f"Checkpoint component checksum mismatch: {relative}")
        if sha256(root / "specification.md") != manifest["spec_sha256"]:
            raise CheckpointError("Embedded immutable specification checksum mismatch")
    manifest = dict(manifest)
    manifest["package_hash"] = actual_hash
    return manifest


def _cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _validate_engine_state(engine_state: Mapping):
    if engine_state.get("failed", False):
        raise CheckpointError("A partially committed/failed update cannot be checkpointed")
    if engine_state.get("accumulation_status") != "boundary":
        raise CheckpointError("Gen2 checkpoints require an accumulation boundary")
    for key in ("optimizers", "schedulers"):
        if set(engine_state.get(key, {})) != COMPONENTS:
            raise CheckpointError(f"Checkpoint requires all four {key} states")
    for role, state in engine_state["schedulers"].items():
        if isinstance(state, Mapping) and state.get("active") is False:
            if state.get("horizon") != 0 or state.get("state") is not None:
                raise CheckpointError(f"Inactive scheduler {role} must explicitly have horizon=0,state=None")


class CheckpointManager:
    """Publish new directories atomically after every component has been written.

    Native networks supply their own inference serialization; a second safetensors
    file preserves exact parameter masters independent of export dtype/key mapping.
    Training pickle contains only tensors and primitive containers and is loaded
    using torch's restricted ``weights_only`` loader.
    """

    def __init__(self, root: str | Path, spec_path: str | Path,
                 expected_spec_sha256: str | None = None, max_to_keep: int = 4,
                 *, read_only: bool = False):
        self.read_only = bool(read_only)
        self.root = Path(root).expanduser().resolve() if self.read_only else assert_writable_path(root)
        self.spec_path = Path(spec_path).expanduser().resolve()
        if not self.spec_path.is_file():
            raise CheckpointError(f"Immutable specification does not exist: {self.spec_path}")
        self.spec_sha256 = sha256(self.spec_path)
        if expected_spec_sha256 and self.spec_sha256 != expected_spec_sha256:
            raise CheckpointError("Immutable specification differs from expected_spec_sha256")
        self.max_to_keep = int(max_to_keep)
        if self.max_to_keep < 1:
            raise ValueError("max_to_keep must be >= 1")

    def save(self, logical_update: int, components: Mapping[str, Any],
             engine_state: Mapping, rng_state: Mapping, metadata: Mapping,
             recorder_state: Mapping | None = None, protected: bool = False,
             reasons=(), export_dtype=None) -> Path:
        if self.read_only:
            raise CheckpointError("This checkpoint manager is read-only; saving is disabled")
        import torch
        from safetensors.torch import save_file
        if logical_update < 0 or set(components) != COMPONENTS:
            raise CheckpointError("Invalid update or incomplete component mapping")
        _validate_engine_state(engine_state)
        if int(engine_state.get("committed_update", logical_update)) != logical_update:
            raise CheckpointError("Engine counter differs from checkpoint update")
        if sha256(self.spec_path) != self.spec_sha256:
            raise CheckpointError("Immutable specification changed since manager initialization")
        if not {"python", "numpy", "torch_cpu", "torch_cuda", "generators"}.issubset(rng_state):
            raise CheckpointError("Complete checkpoint requires training/diagnostic/preview RNG state")
        metadata = json_safe(metadata, redact=True)
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / f"update_{logical_update:08d}"
        if destination.exists():
            # Existing checkpoints are immutable. The caller deduplicates events.
            raise CheckpointError(f"Checkpoint already exists: {destination}")
        temporary = self.root / f".update_{logical_update:08d}.{uuid.uuid4().hex}.incomplete"
        temporary.mkdir()
        descriptors = {}
        try:
            for name in sorted(COMPONENTS):
                component = components[name]
                raw = component.state_dict() if hasattr(component, "state_dict") else component
                if not isinstance(raw, Mapping) or not raw:
                    raise CheckpointError(f"Empty or invalid state for {name}")
                masters = {}
                for key, tensor in raw.items():
                    if not isinstance(tensor, torch.Tensor):
                        raise CheckpointError(f"Component {name}.{key} is not a tensor")
                    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                        raise CheckpointError(f"Nonfinite checkpoint tensor: {name}.{key}")
                    masters[key] = tensor.detach().cpu().contiguous().clone()
                if hasattr(component, "named_parameters"):
                    for key, parameter in component.named_parameters():
                        if parameter.is_floating_point() and parameter.dtype != torch.float32:
                            raise CheckpointError(f"Trainable master {name}.{key} must remain float32")
                master_file = f"{name}.masters.safetensors"
                save_file(masters, str(temporary / master_file), metadata={"spec_sha256": self.spec_sha256,
                          "schema_version": SCHEMA_VERSION, "parameter_family": name})
                descriptors[name] = {"masters": master_file,
                                     "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                                 for key, value in masters.items()}}
                if hasattr(component, "save_weights"):
                    native_file = f"{name}.safetensors"
                    component.save_weights(str(temporary / native_file),
                                           dtype=export_dtype or torch.float32,
                                           metadata={"gen2_spec_sha256": self.spec_sha256,
                                                     "gen2_schema_version": SCHEMA_VERSION})
                    descriptors[name]["native_export"] = native_file
                del masters
            state = {"engine_state": _cpu_tree(engine_state), "rng_state": _cpu_tree(rng_state),
                     "recorder_state": _cpu_tree(recorder_state or {})}
            torch.save(state, temporary / "training_state.pt")
            # This is an unchanged tracked/reference copy; the read-only source is
            # only read. Recording hashes does not rewrite its bytes.
            shutil.copyfile(self.spec_path, temporary / "specification.md")
            files = {}
            for file in sorted(temporary.iterdir()):
                # Windows _commit requires a writable descriptor. These are
                # newly created checkpoint copies, never the read-only source.
                with file.open("r+b") as handle:
                    os.fsync(handle.fileno())
                files[file.name] = {"sha256": sha256(file), "size": file.stat().st_size}
            manifest = {"schema_version": SCHEMA_VERSION, "spec_version": SCHEMA_VERSION,
                        "spec_sha256": self.spec_sha256, "logical_update": logical_update,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "protected": bool(protected or logical_update == 0),
                        "reasons": list(dict.fromkeys(reasons)), "components": descriptors,
                        "files": files, "metadata": metadata,
                        "compatibility_sha256": _canonical_hash(metadata),
                        "accumulation_status": "boundary"}
            write_json(temporary / "manifest.json", manifest)
            write_json(temporary / COMPLETE_MARKER,
                       {"manifest_sha256": sha256(temporary / "manifest.json"),
                        "logical_update": logical_update})
            # Same-filesystem directory rename publishes the package in one step.
            os.replace(temporary, destination)
        except BaseException:
            # Leave the incomplete directory as evidence, never label it resumable.
            raise
        self.prune()
        return destination

    def load(self, path: str | Path, expected_metadata: Mapping | None = None,
             components: Mapping[str, Any] | None = None,
             *, load_training_state: bool = True) -> dict:
        import torch
        from safetensors.torch import load_file
        if sha256(self.spec_path) != self.spec_sha256:
            raise CheckpointError("Immutable specification changed before resume")
        root = Path(path).expanduser().resolve()
        manifest = load_manifest(root, expected_spec_sha256=self.spec_sha256)
        if expected_metadata:
            actual = manifest["metadata"]
            for key, expected in json_safe(expected_metadata, redact=True).items():
                if key not in actual or actual[key] != expected:
                    raise CheckpointError(f"Strict resume metadata mismatch: {key}")
        states = {}
        for name, descriptor in manifest["components"].items():
            states[name] = load_file(str(root / descriptor["masters"]), device="cpu")
            tensors = descriptor["tensors"]
            if set(states[name]) != set(tensors):
                raise CheckpointError(f"Tensor key mismatch in {name}")
            for key, value in states[name].items():
                if list(value.shape) != tensors[key]["shape"] or str(value.dtype) != tensors[key]["dtype"]:
                    raise CheckpointError(f"Tensor shape/dtype mismatch in {name}.{key}")
        training = {}
        if load_training_state:
            try:
                training = torch.load(root / "training_state.pt", map_location="cpu", weights_only=True)
            except Exception as exc:
                raise CheckpointError("Training state cannot be loaded with restricted tensor-only deserialization") from exc
            if not {"engine_state", "rng_state", "recorder_state"}.issubset(training):
                raise CheckpointError("Training state is missing a required component")
            if not {"python", "numpy", "torch_cpu", "torch_cuda", "generators"}.issubset(training["rng_state"]):
                raise CheckpointError("Training state is missing required random streams")
            _validate_engine_state(training["engine_state"])
            if int(training["engine_state"].get("committed_update", -1)) != manifest["logical_update"]:
                raise CheckpointError("Resume engine counter differs from manifest")
        if components is not None:
            if set(components) != COMPONENTS:
                raise CheckpointError("Restore requires all four live components")
            # Validate every mapping before modifying any live component.
            for name, component in components.items():
                current = component.state_dict()
                if set(current) != set(states[name]):
                    raise CheckpointError(f"Live module mapping mismatch for {name}")
                if any(current[key].shape != states[name][key].shape for key in current):
                    raise CheckpointError(f"Live module shape mismatch for {name}")
            for name, component in components.items():
                component.load_state_dict(states[name], strict=True)
        return {"manifest": manifest, "components": states, **training}

    def prune(self) -> list[Path]:
        """Prune only old complete rolling saves inside this exact checkpoint root."""
        if self.read_only:
            raise CheckpointError("This checkpoint manager is read-only; retention edits are disabled")
        if not self.root.exists():
            return []
        rolling = []
        for path in self.root.glob("update_*"):
            if not path.is_dir() or not (path / COMPLETE_MARKER).is_file():
                continue
            manifest = load_manifest(path, verify_components=False)
            if not manifest["protected"]:
                rolling.append((manifest["logical_update"], path))
        removed = []
        for _, path in sorted(rolling)[:-self.max_to_keep]:
            resolved = assert_writable_path(path)
            if resolved.parent != self.root or not resolved.name.startswith("update_"):
                raise CheckpointError(f"Unsafe retention target: {resolved}")
            shutil.rmtree(resolved)
            removed.append(resolved)
        return removed
