"""Bounded, append-only recording for Gen2 runs (no model imports)."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import queue
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "1.0.0"
STREAMS = frozenset({"events", "microbatches", "updates", "activations", "probes",
                     "gates", "spectra", "samples/manifest", "dataset_manifest"})
RATING_FIELDS = ("image_id", "prompt_id", "checkpoint_hash", "rater", "date",
                 "target_style_fidelity", "content_fulfillment", "visible_artifacts",
                 "copying_suspicious_similarity", "notes")


class RecordingError(RuntimeError):
    """Recording cannot continue faithfully; the training loop must abort."""


class RecordingBudgetExceeded(RecordingError):
    pass


def assert_writable_path(path: str | Path) -> Path:
    """Enforce the user's read-only source directory even for configured outputs."""
    resolved = Path(path).expanduser().resolve()
    forbidden = Path(__file__).resolve().parents[2] / "gen2"
    if resolved == forbidden or forbidden in resolved.parents:
        raise PermissionError(f"Gen2 source folder is read-only: {resolved}")
    return resolved


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any, path: str = "$", *, redact: bool = False) -> Any:
    """Validate finite JSON, rejecting tensors so graphs never enter the queue.

    Callers reduce/detach tensors on-device and pass Python values. In particular,
    NaN must become an explicit failure, not JSON's nonstandard ``NaN`` token.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RecordingError(f"Nonfinite number at {path}: {value!r}")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RecordingError(f"JSON key must be a string at {path}")
            # Model tokenizer IDs are not secrets. Redact only credential keys.
            secret = (key.lower() in {"token", "authorization", "credentials"}
                      or bool(re.search(r"(^|_)(api_key|access_token|auth_token|bearer_token|hub_token|password|secret|secret_key|hf_token)$", key.lower())))
            result[key] = "[redacted]" if redact and secret else json_safe(item, f"{path}.{key}", redact=redact)
        return result
    if isinstance(value, (list, tuple)):
        return [json_safe(item, f"{path}[{i}]", redact=redact) for i, item in enumerate(value)]
    raise RecordingError(f"Expected reduced JSON data at {path}, received {type(value).__name__}")


def write_json(path: str | Path, value: Any) -> Path:
    """Publish a fully written UTF-8 JSON document by same-directory replacement."""
    target = assert_writable_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(json_safe(value, redact=True), ensure_ascii=False, allow_nan=False,
                      sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def metric(value: float | None, reason: str | None = None, **labels) -> dict:
    if value is None and not reason:
        raise ValueError("Undefined metrics require a reason")
    return json_safe({"value": value, "reason": reason, **labels})


def _option(config, name, default):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def iter_records(root: str | Path, stream: str):
    """Read closed segments oldest-first, followed by the active append file."""
    if stream not in STREAMS:
        raise ValueError(f"Unknown recording stream: {stream}")
    active = Path(root) / f"{stream}.jsonl"
    segments = sorted(active.parent.glob(f"{active.stem}.[0-9]*.jsonl*"))
    for path in [*segments, active]:
        if not path.exists():
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.endswith("\n"):
                    raise RecordingError(f"Incomplete append-only record in {path}:{number}")
                try:
                    record = json.loads(line)
                except ValueError as exc:
                    raise RecordingError(f"Invalid record in {path}:{number}") from exc
                yield json_safe(record)


class Recorder:
    """One bounded writer with backpressure; failures propagate to the producer.

    ``record`` reserves bytes before enqueuing, so pending rows count against the
    hard budget. ``flush`` is a barrier and performs fsync on all open streams.
    An exhausted budget permanently poisons this recorder: continuing a run after
    catching that exception is forbidden. Previously accepted rows are flushed.
    """

    def __init__(self, root: str | Path, run_id: str, config=None, context=None):
        self.root = assert_writable_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.config = config or {}
        self.rotate_bytes = int(float(_option(config, "rotate_mb", 64)) * 1024 * 1024)
        self.budget_bytes = int(float(_option(config, "max_core_recording_mb", 4096)) * 1024 * 1024)
        self.flush_every = int(_option(config, "flush_every_updates", 10))
        self.compression = _option(config, "compression", "none")
        size = int(_option(config, "writer_queue_records", 4096))
        if min(self.rotate_bytes, self.budget_bytes, self.flush_every, size) <= 0:
            raise ValueError("Recording capacities and intervals must be positive")
        if self.compression not in {"none", "gzip"}:
            raise ValueError("compression must be none or gzip")
        self.context = {"logical_update": 0, "update_attempt_id": 0,
                        "stage": "initialization", "update_kind": "initialization"}
        if context:
            self.context.update(context)
        self._queue = queue.Queue(maxsize=size)
        self._lock = threading.Lock()
        self._files = {}
        self._segments = {}
        self._offsets = {}
        self._error: BaseException | None = None
        self._fatal: BaseException | None = None
        self._closed = False
        self._sequence = 0
        self._last_event_id = None
        self._written_sequence = 0
        self._last_flushed_update = -1
        self._used_bytes = 0
        self._pending_bytes = 0
        # Count existing metadata/config/recording (weights and images excluded).
        for path in self.root.rglob("*"):
            if path.is_file() and "checkpoints" not in path.relative_to(self.root).parts:
                if self._is_core_file(path):
                    self._used_bytes += path.stat().st_size
        existing_count = existing_sequence_sum = 0
        for stream in STREAMS:
            for row in iter_records(self.root, stream):
                if row.get("run_id") != self.run_id:
                    raise RecordingError(f"Existing {stream} belongs to another run")
                if row.get("schema_version") != SCHEMA_VERSION or not isinstance(row.get("record_sequence"), int):
                    raise RecordingError(f"Existing {stream} has an incompatible recorder schema")
                sequence = row["record_sequence"]
                existing_count += 1
                existing_sequence_sum += sequence
                if sequence > self._sequence:
                    self._sequence = sequence
                    self._last_event_id = row.get("event_id")
            active = self.root / f"{stream}.jsonl"
            self._offsets[stream] = active.stat().st_size if active.exists() else 0
        if existing_count != self._sequence or existing_sequence_sum != self._sequence * (self._sequence + 1) // 2:
            raise RecordingError("Existing append-only record sequences contain missing or duplicate rows")
        self._written_sequence = self._sequence
        self._session_id = uuid.uuid4().hex
        if self._used_bytes >= self.budget_bytes:
            raise RecordingBudgetExceeded("Existing core recording meets/exceeds the configured budget")
        self._worker = threading.Thread(target=self._writer, name="gen2-recorder", daemon=True)
        self._worker.start()

    def set_context(self, **context):
        self.context.update(json_safe(context))

    def _check(self):
        if self._error is not None:
            raise RecordingError(f"Recorder writer failed: {self._error}") from self._error
        if self._fatal is not None:
            raise self._fatal
        if self._closed:
            raise RecordingError("Recorder is closed")

    def _enqueue(self, item):
        while True:
            if self._error is not None:
                raise RecordingError(f"Recorder writer failed: {self._error}") from self._error
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def record(self, stream: str, payload: Mapping | None = None, **context) -> str:
        self._check()
        if stream not in STREAMS:
            raise ValueError(f"Unknown recording stream {stream!r}; expected {sorted(STREAMS)}")
        data = dict(payload or {})
        explicit = {key: data.pop(key) for key in ("logical_update", "update_attempt_id", "update_attempt", "stage", "update_kind")
                    if key in data}
        envelope = {**self.context, **explicit, **context}
        # Engine uses either spelling; persisted schema has one unambiguous key.
        if "update_attempt" in envelope:
            envelope["update_attempt_id"] = envelope.pop("update_attempt")
        envelope.update({"schema_version": SCHEMA_VERSION, "run_id": self.run_id,
                         "timestamp": datetime.now(timezone.utc).isoformat()})
        row = {**data, **envelope}
        for key in ("logical_update", "update_attempt_id"):
            if not isinstance(row[key], (int, str)) or (isinstance(row[key], int) and row[key] < 0):
                raise RecordingError(f"Invalid {key}: {row[key]!r}")
        if stream == "microbatches":
            if "example_ids" not in row and ("example_id" in row or "sample_id" in row):
                row["example_ids"] = [row.get("example_id", row.get("sample_id"))]
            if "accumulation_index" not in row or "example_ids" not in row:
                raise RecordingError("Microbatch rows require accumulation_index and example_ids")
        try:
            row = json_safe(row, redact=True)
        except RecordingError as exc:
            # The invalid value itself cannot enter JSON. A finite explanation can.
            if stream != "events":
                self.event("nonfinite_or_invalid_record", source_stream=stream, error=str(exc))
                self.flush()
            raise
        with self._lock:
            next_sequence = self._sequence + 1
            row["record_sequence"] = next_sequence
            row["event_id"] = f"{self._session_id}:{next_sequence}"
            raw = (json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
            if self._used_bytes + len(raw) > self.budget_bytes:
                self._fatal = RecordingBudgetExceeded(
                    f"Core recording budget exhausted: used/reserved={self._used_bytes}, "
                    f"next_record={len(raw)}, limit={self.budget_bytes}; abort training")
            else:
                self._used_bytes += len(raw)
                self._pending_bytes += len(raw)
                self._sequence = next_sequence
                self._last_event_id = row["event_id"]
        if self._fatal is not None:
            # A large rejected row can leave enough room for a small, finite
            # failure record. Never exceed the hard limit even for this notice.
            notice = {**envelope, "event": "recording_budget_exceeded", "error": str(self._fatal),
                      "record_sequence": self._sequence + 1,
                      "event_id": f"{self._session_id}:{self._sequence + 1}"}
            error_raw = (json.dumps(notice, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
            with self._lock:
                fits = self._used_bytes + len(error_raw) <= self.budget_bytes
                if fits:
                    self._used_bytes += len(error_raw)
                    self._pending_bytes += len(error_raw)
                    self._sequence += 1
                    self._last_event_id = notice["event_id"]
            if fits:
                self._enqueue(("record", "events", error_raw, self._sequence))
            self._barrier()
            raise self._fatal
        self._enqueue(("record", stream, raw, next_sequence))
        update = int(row["logical_update"])
        if stream == "updates" and update % self.flush_every == 0 and update != self._last_flushed_update:
            self.flush()
            self._last_flushed_update = update
        return row["event_id"]

    def event(self, name: str, **payload):
        return self.record("events", {"event": name, **payload})

    def _open(self, stream):
        if stream not in self._files:
            path = self.root / f"{stream}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._files[stream] = path.open("ab")
        return self._files[stream]

    def _rotate(self, stream):
        handle = self._files.pop(stream)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        active = self.root / f"{stream}.jsonl"
        index = self._segments.get(stream, 0) + 1
        while True:
            closed = active.with_name(f"{active.stem}.{index:06d}.jsonl")
            if not closed.exists() and not closed.with_suffix(".jsonl.gz").exists():
                break
            index += 1
        os.replace(active, closed)
        self._segments[stream] = index
        self._offsets[stream] = 0
        if self.compression == "gzip":
            compressed = closed.with_suffix(".jsonl.gz")
            temporary = compressed.with_suffix(".gz.tmp")
            try:
                with closed.open("rb") as source, temporary.open("xb") as destination:
                    with gzip.GzipFile(fileobj=destination, mode="wb", mtime=0) as zipper:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            zipper.write(chunk)
                    destination.flush()
                    os.fsync(destination.fileno())
                # Occasionally gzip is larger than its input. Reserve that growth
                # before publishing so compression cannot exceed the hard budget.
                difference = closed.stat().st_size - temporary.stat().st_size
                with self._lock:
                    if self._used_bytes - difference > self.budget_bytes:
                        raise RecordingBudgetExceeded("Gzip segment growth would exceed the core recording budget")
                os.replace(temporary, compressed)
                # Budget tracks live bytes; rotation never drops a core row.
                closed.unlink()
                with self._lock:
                    self._used_bytes -= difference
            finally:
                if temporary.exists():
                    temporary.unlink()

    def _writer(self):
        try:
            while True:
                item = self._queue.get()
                try:
                    if item[0] in {"flush", "stop"}:
                        for handle in self._files.values():
                            handle.flush()
                            os.fsync(handle.fileno())
                        item[1].set()
                        if item[0] == "stop":
                            return
                    else:
                        _, stream, raw, sequence = item
                        if self._offsets[stream] and self._offsets[stream] + len(raw) > self.rotate_bytes:
                            self._open(stream)
                            self._rotate(stream)
                        handle = self._open(stream)
                        handle.write(raw)
                        self._offsets[stream] += len(raw)
                        self._written_sequence = sequence
                        with self._lock:
                            self._pending_bytes -= len(raw)
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._error = exc
        finally:
            for handle in self._files.values():
                try:
                    handle.close()
                except OSError:
                    pass
            self._files.clear()

    def _barrier(self, stop=False):
        barrier = threading.Event()
        self._enqueue(("stop" if stop else "flush", barrier))
        while not barrier.wait(0.1):
            if self._error is not None:
                raise RecordingError(f"Recorder writer failed: {self._error}") from self._error

    def flush(self):
        self._check()
        self._barrier()
        self.enforce_core_budget()

    def _is_core_file(self, path):
        relative = path.relative_to(self.root)
        if any(part in {"checkpoints", "weights", "tensor_dumps", "dataset", "datasets"} for part in relative.parts):
            return False
        return (path.suffix.lower() in {".json", ".jsonl", ".gz", ".yaml", ".yml", ".csv", ".md"}
                or relative.as_posix() == "fixed_probe_packet.pt"
                or path.suffix == ".safetensors" and any(part in {"fixed_probe", "fixed_probes", "probe_packet", "fixed_probe_packet"}
                                                          for part in relative.parts))

    def enforce_core_budget(self):
        """Account for manifests/fixed packets written by the orchestration layer.

        Flush invokes this at every configured update interval and checkpoint.
        Only completed, accepted streams count; optional dumps/weights/images do
        not consume this budget. A caller writing a large mandatory packet should
        call this immediately afterward as well.
        """
        disk_bytes = 0
        for path in self.root.rglob("*"):
            if path.is_file() and self._is_core_file(path):
                disk_bytes += path.stat().st_size
        # A producer may have queued further records after the flush barrier.
        with self._lock:
            self._used_bytes = disk_bytes + self._pending_bytes
            if self._used_bytes > self.budget_bytes:
                self._fatal = RecordingBudgetExceeded(
                    f"Core metadata/probe/recording budget exceeded: {self._used_bytes} > {self.budget_bytes}; abort training")
        if self._fatal is not None:
            raise self._fatal

    def state_dict(self):
        self.flush()
        return {"run_id": self.run_id, "record_sequence": self._sequence,
                "last_event_id": self._last_event_id, "offsets": dict(self._offsets),
                "core_bytes": self._used_bytes, "queue_size": self._queue.qsize(),
                "session_id": self._session_id}

    def status(self):
        """Bounded, non-flushing snapshot for ordinary update telemetry."""
        with self._lock:
            return {"queue_size": self._queue.qsize(), "queue_capacity": self._queue.maxsize,
                    "core_reserved_bytes": self._used_bytes, "core_budget_bytes": self.budget_bytes,
                    "pending_record_bytes": self._pending_bytes, "record_sequence": self._sequence,
                    "written_record_sequence": self._written_sequence,
                    "writer_failed": self._error is not None, "closed": self._closed}

    def resume(self, saved_state: Mapping):
        if saved_state.get("run_id") != self.run_id:
            raise RecordingError("Recorder resume run ID mismatch")
        if self._sequence < int(saved_state.get("record_sequence", 0)):
            raise RecordingError("Recorder is missing rows referenced by the checkpoint")
        saved_sequence = int(saved_state.get("record_sequence", 0))
        if saved_sequence:
            self.flush()
            found = False
            for stream in STREAMS:
                for row in iter_records(self.root, stream):
                    if row.get("record_sequence") == saved_sequence:
                        if row.get("event_id") != saved_state.get("last_event_id"):
                            raise RecordingError("Checkpoint recorder last-event identity does not match retained rows")
                        found = True
                        break
                if found:
                    break
            if not found:
                raise RecordingError("Checkpoint last event is missing from retained recording")
        self.event("resume", checkpoint_recorder_state=json_safe(saved_state),
                   previous_last_event_id=self._last_event_id,
                   retained_rows_after_checkpoint=self._sequence - int(saved_state.get("record_sequence", 0)))
        self.flush()

    def close(self):
        if self._closed:
            return
        try:
            if self._error is None:
                self._barrier(stop=True)
            self._worker.join(timeout=5)
        finally:
            self._closed = True
        if self._error is not None:
            raise RecordingError(f"Recorder writer failed: {self._error}") from self._error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def append_rating_template(path: str | Path, images: list[Mapping]) -> Path:
    """Append new image IDs with blank human scores; preserve existing ratings."""
    path = assert_writable_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != RATING_FIELDS:
                raise RecordingError("Human rating CSV columns differ from the v1 schema")
            existing = {row["image_id"] for row in reader}
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RATING_FIELDS)
        if needs_header:
            writer.writeheader()
        for image in images:
            image_id = str(image["image_id"])
            if image_id not in existing:
                writer.writerow({"image_id": image_id, "prompt_id": image.get("prompt_id", ""),
                                 "checkpoint_hash": image.get("checkpoint_hash", "")})
                existing.add(image_id)
        handle.flush()
        os.fsync(handle.fileno())
    return path
