from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SPLIT_SCHEMA_VERSION = 1
SPLIT_ALGORITHM = "sorted_pair_fingerprint_random_v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Gen2 captions must have string JSON object keys")
        return {key: _canonical_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical_json_value(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_canonical_json_value(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _dataset_root(dataset: Any) -> str:
    root = str(getattr(dataset, "dataset_path", ""))
    if not root:
        raise ValueError("Gen2 dataset has no dataset root")
    if not os.path.isdir(root):
        root = os.path.dirname(root)
    return os.path.abspath(root)


def _relative_path(path: str, root: str) -> str:
    try:
        relative = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    except ValueError as error:
        raise ValueError(f"dataset image path cannot be relativized: {path}") from error
    if relative == ".." or relative.startswith(".." + os.sep):
        raise ValueError(f"dataset image is outside dataset root: {path}")
    return Path(relative).as_posix()


def _digest_prompt(raw_caption: str) -> str:
    from toolkit.ideogram_caption import digest_caption_string
    return digest_caption_string(raw_caption)


@dataclass(frozen=True)
class PairRecord:
    index: int
    relative_path: str
    image_sha256: str
    raw_caption_sha256: str
    digested_prompt_sha256: str
    pair_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "relative_path": self.relative_path,
            "image_sha256": self.image_sha256,
            "raw_caption_sha256": self.raw_caption_sha256,
            "digested_prompt_sha256": self.digested_prompt_sha256,
            "pair_sha256": self.pair_sha256,
        }


@dataclass(frozen=True)
class DatasetSplit:
    schema_version: int
    algorithm: str
    dataset_root: str
    dataset_fingerprint: str
    seed: int
    heldout_count: int
    records: tuple[PairRecord, ...]
    train_pair_sha256: tuple[str, ...]
    heldout_pair_sha256: tuple[str, ...]

    @property
    def train_records(self) -> tuple[PairRecord, ...]:
        values = set(self.train_pair_sha256)
        return tuple(record for record in self.records if record.pair_sha256 in values)

    @property
    def heldout_records(self) -> tuple[PairRecord, ...]:
        values = set(self.heldout_pair_sha256)
        return tuple(record for record in self.records if record.pair_sha256 in values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "policy": {"heldout_usage": "evaluation_only", "immutable": True},
            "dataset_root": self.dataset_root,
            "dataset_fingerprint": self.dataset_fingerprint,
            "seed": self.seed,
            "heldout_count": self.heldout_count,
            "total_count": len(self.records),
            "train_count": len(self.train_pair_sha256),
            "heldout_pair_count": len(self.heldout_pair_sha256),
            "records": [record.as_dict() for record in self.records],
            "train_pair_sha256": list(self.train_pair_sha256),
            "heldout_pair_sha256": list(self.heldout_pair_sha256),
        }


def _pair_records(dataset: Any) -> tuple[PairRecord, ...]:
    root = _dataset_root(dataset)
    items = getattr(dataset, "file_list", None)
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("Gen2 dataset must contain at least one image-caption pair")
    if getattr(getattr(dataset, "dataset_config", None), "num_repeats", 1) != 1:
        raise ValueError("Gen2 dataset split requires dataset num_repeats=1")
    records: list[PairRecord] = []
    seen_paths: set[str] = set()
    seen_pairs: set[str] = set()
    for index, item in enumerate(items):
        path = getattr(item, "path", None)
        if not isinstance(path, str) or not os.path.isfile(path):
            raise ValueError(f"Gen2 split requires a readable local image file: {path!r}")
        relative_path = _relative_path(path, root)
        if relative_path in seen_paths:
            raise ValueError(f"Gen2 dataset contains duplicate relative image path: {relative_path}")
        seen_paths.add(relative_path)
        item.load_caption(getattr(dataset, "caption_dict", None))
        raw_caption = getattr(item, "raw_caption", None)
        if not isinstance(raw_caption, str) or not raw_caption.strip():
            raise ValueError(f"Gen2 dataset item has no raw caption: {path}")
        try:
            parsed = json.loads(raw_caption)
        except json.JSONDecodeError as error:
            raise ValueError(f"Gen2 caption must be valid JSON: {path}") from error
        canonical_caption = _canonical_json(parsed)
        image_hash = _sha256_bytes(Path(path).read_bytes())
        caption_hash = _sha256_bytes(canonical_caption.encode("utf-8"))
        digested_hash = _sha256_bytes(_digest_prompt(raw_caption).encode("utf-8"))
        # Keep the material explicit and version-stable: path, image, caption.
        pair_hash = _sha256_bytes("\n".join((relative_path, image_hash, caption_hash)).encode("utf-8"))
        if pair_hash in seen_pairs:
            raise ValueError(f"Gen2 dataset contains duplicate pair fingerprint: {pair_hash}")
        seen_pairs.add(pair_hash)
        records.append(PairRecord(index, relative_path, image_hash, caption_hash, digested_hash, pair_hash))
    return tuple(sorted(records, key=lambda record: record.pair_sha256))


def _dataset_fingerprint(records: Iterable[PairRecord]) -> str:
    return _sha256_bytes("\n".join(record.pair_sha256 for record in records).encode("utf-8"))


def build_split(dataset: Any, heldout_count: int, seed: int) -> DatasetSplit:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("dataset_split.seed must be a non-negative integer")
    if isinstance(heldout_count, bool) or not isinstance(heldout_count, int):
        raise ValueError("dataset_split.heldout_count must be an integer")
    records = _pair_records(dataset)
    if heldout_count < 0 or heldout_count >= len(records):
        raise ValueError("dataset_split.heldout_count must satisfy 0 <= heldout_count < total pair count")
    selected = set(random.Random(seed).sample(range(len(records)), heldout_count))
    heldout = tuple(record.pair_sha256 for index, record in enumerate(records) if index in selected)
    train = tuple(record.pair_sha256 for record in records if record.pair_sha256 not in set(heldout))
    return DatasetSplit(SPLIT_SCHEMA_VERSION, SPLIT_ALGORITHM, _dataset_root(dataset), _dataset_fingerprint(records), seed, heldout_count, records, train, heldout)


def _validate_payload(payload: dict[str, Any]) -> DatasetSplit:
    if payload.get("schema_version") != SPLIT_SCHEMA_VERSION or payload.get("algorithm") != SPLIT_ALGORITHM:
        raise ValueError("unsupported Gen2 dataset split schema")
    if payload.get("policy") != {"heldout_usage": "evaluation_only", "immutable": True}:
        raise ValueError("Gen2 dataset split has an invalid immutable policy")
    try:
        records = tuple(PairRecord(**record) for record in payload["records"])
        train = tuple(payload["train_pair_sha256"])
        heldout = tuple(payload["heldout_pair_sha256"])
        seed = payload["seed"]
        heldout_count = payload["heldout_count"]
    except (KeyError, TypeError) as error:
        raise ValueError("Gen2 dataset split metadata is incomplete") from error
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Gen2 dataset split seed is invalid")
    pairs = [record.pair_sha256 for record in records]
    if pairs != sorted(pairs) or len(set(pairs)) != len(pairs):
        raise ValueError("Gen2 split records are not canonical and unique")
    if len(set(train)) != len(train) or len(set(heldout)) != len(heldout):
        raise ValueError("Gen2 split memberships contain duplicates")
    if set(train) | set(heldout) != set(pairs) or set(train) & set(heldout):
        raise ValueError("Gen2 dataset split membership is invalid")
    if heldout_count != len(heldout) or heldout_count < 0 or heldout_count >= len(records):
        raise ValueError("Gen2 dataset split heldout count is invalid")
    if len(train) + len(heldout) != len(records):
        raise ValueError("Gen2 dataset split counts are invalid")
    if payload.get("total_count") != len(records) or payload.get("train_count") != len(train) or payload.get("heldout_pair_count") != len(heldout):
        raise ValueError("Gen2 dataset split counts are invalid")
    if _dataset_fingerprint(records) != payload.get("dataset_fingerprint"):
        raise ValueError("Gen2 dataset split fingerprint is invalid")
    return DatasetSplit(payload["schema_version"], payload["algorithm"], payload["dataset_root"], payload["dataset_fingerprint"], seed, heldout_count, records, train, heldout)


def load_split(path: Path) -> DatasetSplit:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Gen2 dataset split artifact is unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError("Gen2 dataset split artifact must be a JSON object")
    return _validate_payload(payload)


def validate_existing_split(path: Path, dataset: Any, heldout_count: int, seed: int) -> DatasetSplit:
    existing = load_split(path)
    current = build_split(dataset, heldout_count, seed)
    if existing.as_dict() != current.as_dict():
        raise ValueError("Gen2 dataset changed or dataset_split parameters do not match the existing split artifact")
    return existing


def ensure_split(path: Path, dataset: Any, heldout_count: int, seed: int) -> DatasetSplit:
    if path.exists():
        return validate_existing_split(path, dataset, heldout_count, seed)
    split = build_split(dataset, heldout_count, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return split


def make_dataset_view(dataset: Any, allowed_pair_sha256: Iterable[str], *, require_nonempty: bool = True) -> Any:
    allowed = set(allowed_pair_sha256)
    view = copy.copy(dataset)
    view.dataset_config = copy.copy(dataset.dataset_config)
    view.file_list = [item for item in dataset.file_list if _item_pair_hash(item, dataset) in allowed]
    if require_nonempty and not view.file_list:
        raise ValueError("Gen2 dataset subset is empty")
    if getattr(view.dataset_config, "num_repeats", 1) != 1:
        raise ValueError("Gen2 dataset_split requires dataset num_repeats=1")
    view.epoch_num = 0
    view.buckets = {}
    view.batch_indices = []
    if view.file_list and getattr(view.dataset_config, "buckets", False):
        view.setup_buckets(quiet=True)
    actual = {_item_pair_hash(item, dataset) for item in view.file_list}
    if actual != allowed:
        raise ValueError("Gen2 dataset view does not exactly match its fingerprint allowlist")
    return view


def make_train_dataset_view(dataset: Any, split: DatasetSplit) -> Any:
    return make_dataset_view(dataset, split.train_pair_sha256)


def _item_pair_hash(item: Any, dataset: Any) -> str:
    path = getattr(item, "path", None)
    if not isinstance(path, str) or not os.path.isfile(path):
        raise ValueError("Gen2 dataset item has no readable local image path")
    item.load_caption(getattr(dataset, "caption_dict", None))
    try:
        parsed = json.loads(item.raw_caption)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Gen2 dataset item caption must be valid JSON") from error
    image_hash = _sha256_bytes(Path(path).read_bytes())
    caption_hash = _sha256_bytes(_canonical_json(parsed).encode("utf-8"))
    material = "\n".join((_relative_path(path, _dataset_root(dataset)), image_hash, caption_hash))
    return _sha256_bytes(material.encode("utf-8"))
