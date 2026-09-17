"""Small token-only v2 packages and one verified rolling resume checkpoint."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


SCHEMA_VERSION = "2.0.0"


class V2CheckpointError(RuntimeError):
    pass


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write(path, content):
    with Path(path).open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _writable(path):
    # Keep the user's untracked gen2 reference/results directory read-only,
    # including aliases through symlinks and junctions.
    root = Path(__file__).resolve().parents[3] / "gen2"
    resolved = Path(path).expanduser().resolve()
    if resolved == root.resolve() or root.resolve() in resolved.parents:
        raise V2CheckpointError(f"Protected read-only Gen2 reference/result directory: {resolved}")
    return resolved


def _member(root, name):
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise V2CheckpointError("Invalid v2 package member")
    path = (root / name).resolve()
    if path.parent != root:
        raise V2CheckpointError("V2 package member escapes its directory")
    return path


def _match_metadata(actual, expected, prefix="metadata"):
    """Every explicitly requested identity must match; additional metadata is fine."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise V2CheckpointError(f"V2 identity mismatch: {prefix}")
        for key, value in expected.items():
            if key not in actual:
                raise V2CheckpointError(f"V2 identity missing: {prefix}.{key}")
            _match_metadata(actual[key], value, f"{prefix}.{key}")
    elif _json_bytes(actual) != _json_bytes(expected):
        raise V2CheckpointError(f"V2 identity mismatch: {prefix}")


def load_manifest(path, *, verify=True, expected_sha=None):
    """Verify a package without model construction or pickle deserialization.

    Returns the manifest fields including metadata, kind, logical_update,
    spec_sha256, files, token_schema, plus package_hash (the manifest SHA256).
    """
    root = Path(path).expanduser().resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        complete = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise V2CheckpointError(f"Missing or malformed v2 completion metadata: {root}") from error
    digest = _digest(root / "manifest.json")
    if complete.get("manifest_sha256") != digest:
        raise V2CheckpointError("V2 manifest checksum differs from completion marker")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("kind") not in ("resume", "inference"):
        raise V2CheckpointError("Unsupported v2 package schema or kind")
    if (type(manifest.get("logical_update")) is not int or manifest["logical_update"] < 0 or
            complete.get("logical_update") != manifest["logical_update"]):
        raise V2CheckpointError("Invalid v2 package logical counter")
    if expected_sha is not None and manifest.get("spec_sha256") != expected_sha:
        raise V2CheckpointError("V2 immutable specification identity mismatch")
    if not isinstance(manifest.get("metadata"), dict) or not isinstance(manifest.get("token_schema"), dict):
        raise V2CheckpointError("V2 package omits identities or token state schema")
    if not {"E", "initial"}.issubset(manifest["token_schema"]):
        raise V2CheckpointError("V2 package omits learned or initial token vectors")
    required = {"tokens.safetensors", "specification.md"}
    if manifest["kind"] == "resume":
        required.add("training_state.pt")
    if set(manifest.get("files", {})) != required:
        raise V2CheckpointError("V2 package contains missing or unexpected components")
    for name, expected in manifest["files"].items():
        member = _member(root, name)
        if not isinstance(expected, dict) or set(expected) != {"size", "sha256"}:
            raise V2CheckpointError("Malformed v2 file integrity record")
        if verify:
            if not member.is_file() or member.stat().st_size != expected["size"] or _digest(member) != expected["sha256"]:
                raise V2CheckpointError(f"V2 component checksum mismatch: {name}")
    if manifest["files"]["specification.md"]["sha256"] != manifest.get("spec_sha256"):
        raise V2CheckpointError("Embedded specification does not match v2 spec identity")
    return {**manifest, "package_hash": digest}


def _cpu_tree(value):
    """Restrict training serialization to tensors and primitive containers."""
    import torch
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise V2CheckpointError("Nonfinite tensor in v2 resume state")
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        if any(not isinstance(key, (str, int)) for key in value):
            raise V2CheckpointError("Resume dictionary keys must be strings or integers")
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if value is None or type(value) in (str, int, bool, float):
        if isinstance(value, float):
            import math
            if not math.isfinite(value):
                raise V2CheckpointError("Nonfinite scalar in v2 resume state")
        return value
    raise V2CheckpointError(f"Unsupported resume type {type(value).__name__}; use primitive RNG/data state")


def _token_state(backend):
    import torch
    state = {name: value.detach().cpu().contiguous().clone() for name, value in backend.tokens.state_dict().items()}
    if not {"E", "initial"}.issubset(state):
        raise V2CheckpointError("V2 token module must save E and initial")
    if state["E"].dtype != torch.float32 or state["initial"].dtype != torch.float32:
        raise V2CheckpointError("V2 learned and initial vector storage must be FP32")
    if state["E"].ndim != 2 or state["initial"].shape != state["E"].shape:
        raise V2CheckpointError("Invalid v2 learned/initial token shape")
    if not all(not value.is_floating_point() or bool(torch.isfinite(value).all()) for value in state.values()):
        raise V2CheckpointError("Nonfinite v2 token state")
    return state


def _schema(state):
    return {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in state.items()}


class V2CheckpointManager:
    """Stage and verify complete packages before replacing the previous save.

    Publication uses sibling directory renames, with rollback on failure. The
    preceding package is retained in a recovery directory during the switch;
    it is removed only after the new directory has been verified in place.
    """

    def __init__(self, root, spec_path, expected_sha):
        # Construction/load are read-only and may inspect user-provided results
        # under gen2/. Every publication validates its actual output path.
        self.root = Path(root).expanduser().resolve()
        self.spec_path = Path(spec_path).expanduser().resolve()
        self.spec_bytes = self.spec_path.read_bytes()
        self.spec_sha256 = hashlib.sha256(self.spec_bytes).hexdigest()
        if self.spec_sha256 != expected_sha:
            raise V2CheckpointError("Immutable v2 reference differs from expected SHA256")
        self.resume_path = self.root / "resume_latest"
        self.inference_path = self.root / "inference_final"

    def _remove_owned(self, path, parent):
        resolved = _writable(path)
        if resolved.parent != parent or not resolved.name.startswith(".v2-"):
            raise V2CheckpointError("Unsafe checkpoint staging/recovery cleanup path")
        shutil.rmtree(resolved)

    def _publish(self, destination, backend, metadata, logical_update, training=None):
        from safetensors.torch import save_file
        import torch
        destination = _writable(destination)
        if type(logical_update) is not int or logical_update < 0:
            raise V2CheckpointError("A package requires a valid committed update counter")
        metadata = json.loads(_json_bytes(metadata))
        if not isinstance(metadata, dict):
            raise V2CheckpointError("V2 metadata must be a mapping")
        backend.assert_frozen()
        tokens = _token_state(backend)
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = destination.parent / f".v2-{destination.name}.pending-{uuid.uuid4().hex}"
        backup = destination.parent / f".v2-{destination.name}.previous-{uuid.uuid4().hex}"
        stage.mkdir()
        previous_moved = published = False
        try:
            save_file(tokens, str(stage / "tokens.safetensors"), metadata={"schema_version": SCHEMA_VERSION,
                      "spec_sha256": self.spec_sha256, "parameter_family": "embedding"})
            _write(stage / "specification.md", self.spec_bytes)
            if training is not None:
                with (stage / "training_state.pt").open("xb") as stream:
                    torch.save(training, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
            names = ["tokens.safetensors", "specification.md"] + (["training_state.pt"] if training is not None else [])
            manifest = {"schema_version": SCHEMA_VERSION, "kind": "resume" if training is not None else "inference",
                "logical_update": logical_update, "spec_sha256": self.spec_sha256,
                "created_at": datetime.now(timezone.utc).isoformat(), "metadata": metadata,
                "token_schema": _schema(tokens), "files": {
                    name: {"size": (stage/name).stat().st_size, "sha256": _digest(stage/name)} for name in names}}
            _write(stage / "manifest.json", _json_bytes(manifest))
            _write(stage / "COMPLETE.json", _json_bytes({"manifest_sha256": _digest(stage/"manifest.json"),
                                                        "logical_update": logical_update}))
            load_manifest(stage, expected_sha=self.spec_sha256)
            if destination.exists():
                # Never replace an unrelated directory or an incomplete result.
                load_manifest(destination, expected_sha=self.spec_sha256)
                os.replace(destination, backup)
                previous_moved = True
            os.replace(stage, destination)
            published = True
            load_manifest(destination, expected_sha=self.spec_sha256)
            if previous_moved:
                self._remove_owned(backup, destination.parent)
            return destination
        except Exception:
            if previous_moved and backup.exists():
                if published and destination.exists():
                    failed = destination.parent / f".v2-{destination.name}.failed-{uuid.uuid4().hex}"
                    os.replace(destination, failed)
                    os.replace(backup, destination)
                    self._remove_owned(failed, destination.parent)
                elif not destination.exists():
                    os.replace(backup, destination)
            raise
        finally:
            if stage.exists():
                self._remove_owned(stage, destination.parent)

    def save(self, backend, engine, metadata, runtime_state, final=False):
        if not isinstance(runtime_state, Mapping) or not {"rng", "data", "evaluation"}.issubset(runtime_state):
            raise V2CheckpointError("Resume requires rng, data and evaluation runtime state")
        state = engine.state_dict()
        training = _cpu_tree({"engine": state, "runtime": runtime_state})
        path = self._publish(self.resume_path, backend, metadata, engine.logical_update, training)
        if final:
            return self.export(backend, metadata, logical_update=engine.logical_update)
        return path

    def export(self, backend, metadata, path=None, logical_update=None):
        if logical_update is None:
            raise V2CheckpointError("Inference export requires the committed logical_update")
        return self._publish(path or self.inference_path, backend, metadata, logical_update)

    def _read_tokens(self, path, backend, expected_metadata=None):
        from safetensors.torch import load_file
        import torch
        root = Path(path).expanduser().resolve()
        manifest = load_manifest(root, expected_sha=self.spec_sha256)
        if expected_metadata is not None:
            _match_metadata(manifest["metadata"], expected_metadata)
        tokens = load_file(str(root / "tokens.safetensors"), device="cpu")
        if _schema(tokens) != manifest["token_schema"] or _schema(tokens) != _schema(backend.tokens.state_dict()):
            raise V2CheckpointError("V2 package token schema does not match this backend")
        if any(value.is_floating_point() and not bool(torch.isfinite(value).all()) for value in tokens.values()):
            raise V2CheckpointError("Nonfinite loaded token state")
        return manifest, tokens

    def load(self, path, backend, engine, expected_metadata=None):
        import torch
        root = Path(path).expanduser().resolve()
        manifest, tokens = self._read_tokens(root, backend, expected_metadata)
        if manifest["kind"] != "resume":
            raise V2CheckpointError("An inference export has no training resume state")
        training = torch.load(root / "training_state.pt", map_location="cpu", weights_only=True)
        if not isinstance(training, dict) or set(training) != {"engine", "runtime"}:
            raise V2CheckpointError("Invalid v2 training state payload")
        runtime = training["runtime"]
        if not isinstance(runtime, dict) or not {"rng", "data", "evaluation"}.issubset(runtime):
            raise V2CheckpointError("Incomplete v2 runtime resume state")
        if training["engine"].get("logical_update") != manifest["logical_update"]:
            raise V2CheckpointError("V2 manifest and training counters disagree")
        engine.validate_state_dict(training["engine"])
        # All hashes, identities and schemas are checked before either mutable
        # component is restored. A restoration failure poisons the engine.
        try:
            backend.tokens.load_state_dict(tokens, strict=True)
            engine.load_state_dict(training["engine"])
            backend.assert_frozen()
        except Exception:
            engine.failed = True
            engine.accumulation_status = "failed"
            raise
        return runtime

    def load_inference(self, path, backend, expected_metadata=None):
        manifest, tokens = self._read_tokens(path, backend, expected_metadata)
        backend.tokens.load_state_dict(tokens, strict=True)
        backend.assert_frozen()
        return manifest
