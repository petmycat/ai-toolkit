from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: os.PathLike | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: os.PathLike | str, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_jsonl(path: os.PathLike | str, records: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_jsonl(path: os.PathLike | str) -> List[Dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    records = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object: {source}")
            records.append(value)
    return records


def condition_specs(tap_layers: Sequence[int]) -> List[Dict[str, Any]]:
    layers = tuple(int(layer) for layer in tap_layers)
    if len(layers) != 13 or len(set(layers)) != 13:
        raise ValueError("tap causal ablation requires exactly 13 unique tap layers")
    conditions = [
        {"name": "none", "kind": "none", "active_layers": []},
        {"name": "all", "kind": "all", "active_layers": list(layers)},
    ]
    conditions.extend(
        {"name": f"solo:{layer}", "kind": "solo", "tap_layer": layer, "active_layers": [layer]}
        for layer in layers
    )
    conditions.extend(
        {
            "name": f"leave_one_out:{layer}",
            "kind": "leave_one_out",
            "tap_layer": layer,
            "active_layers": [candidate for candidate in layers if candidate != layer],
        }
        for layer in layers
    )
    return conditions


def _tap_adapters(activator: Any) -> Mapping[str, Any]:
    adapters = getattr(activator, "tap_adapters", None)
    if adapters is None or not hasattr(adapters, "items"):
        raise TypeError("text activator does not expose tap_adapters")
    return adapters


def tap_state(activator: Any) -> Dict[str, Any]:
    adapters = _tap_adapters(activator)
    return {
        "component_active": dict(getattr(activator, "component_active", {})),
        "adapter_active": {str(key): bool(getattr(adapter, "active", True)) for key, adapter in adapters.items()},
        "set_runtime_mode": getattr(activator, "set_runtime_mode", None),
    }


def _apply_tap_mask(activator: Any, active_layers: Sequence[int]) -> None:
    active = {str(int(layer)) for layer in active_layers}
    adapters = _tap_adapters(activator)
    available = {str(key) for key in adapters}
    unknown = sorted(active.difference(available))
    if unknown:
        raise KeyError(f"unknown tap layers in mask: {unknown}")
    component_active = getattr(activator, "component_active", None)
    if isinstance(component_active, dict):
        component_active["tap_adapters"] = True
    for key, adapter in adapters.items():
        adapter.active = str(key) in active


def _restore_tap_state(activator: Any, snapshot: Mapping[str, Any]) -> None:
    original = snapshot["set_runtime_mode"]
    if original is not None:
        activator.set_runtime_mode = original
    component_active = getattr(activator, "component_active", None)
    if isinstance(component_active, dict):
        component_active.clear()
        component_active.update(snapshot["component_active"])
    for key, adapter in _tap_adapters(activator).items():
        adapter.active = snapshot["adapter_active"][str(key)]


@contextlib.contextmanager
def tap_mask_context(activator: Any, active_layers: Sequence[int]) -> Iterator[Any]:
    snapshot = tap_state(activator)
    original = snapshot["set_runtime_mode"]

    def guarded_set_runtime_mode(mode: Optional[str]) -> None:
        if callable(original):
            original(mode)
        if mode is None or mode == "full":
            _apply_tap_mask(activator, active_layers)

    try:
        if callable(original):
            activator.set_runtime_mode = guarded_set_runtime_mode
        _apply_tap_mask(activator, active_layers)
        yield activator
    finally:
        _restore_tap_state(activator, snapshot)


def assert_frozen(module: Any) -> None:
    trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"inference-only ablation found trainable parameters: {trainable[:10]}")


def assert_tap_state_equal(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if actual["component_active"] != expected["component_active"] or actual["adapter_active"] != expected["adapter_active"]:
        raise RuntimeError("tap condition leaked activator state")


def _normalize_item_id(value: Any) -> str:
    text = os.fspath(value).strip().replace("\\", "/")
    if not text or text.startswith("/") or (len(text) > 1 and text[1] == ":"):
        raise ValueError(f"invalid dataset-relative item id: {value!r}")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts or ".." in parts:
        raise ValueError(f"invalid dataset-relative item id: {value!r}")
    return PurePosixPath(*parts).as_posix()


def read_sidecar_caption(image_path: os.PathLike | str) -> Tuple[str, str]:
    stem = os.path.splitext(os.fspath(image_path))[0]
    for suffix in (".json", ".txt"):
        path = stem + suffix
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()
        if suffix == ".txt":
            caption = raw.strip()
        else:
            payload = json.loads(raw)
            if isinstance(payload, str):
                caption = payload.strip()
            elif isinstance(payload, Mapping) and ("text" in payload or "caption" in payload):
                caption = str(payload.get("text", payload.get("caption", ""))).strip()
            else:
                raise ValueError(f"caption JSON must be a string or contain text/caption: {path}")
        if not caption:
            raise ValueError(f"caption is empty: {path}")
        return caption, path
    raise FileNotFoundError(f"missing .json/.txt caption sidecar for {image_path}")


def select_probe_items(
    dataset_root: os.PathLike | str,
    split_manifest: Mapping[str, Any],
    *,
    train_limit: int = 8,
    expected_heldout: Optional[int] = 4,
    placeholder: str = "[trigger]",
) -> List[Dict[str, Any]]:
    root = os.path.abspath(os.fspath(dataset_root))
    train_ids = sorted(_normalize_item_id(value) for value in split_manifest.get("train_item_ids", ()))
    heldout_ids = sorted(_normalize_item_id(value) for value in split_manifest.get("heldout_item_ids", ()))
    if len(train_ids) < train_limit:
        raise ValueError(f"split manifest has only {len(train_ids)} train items; need {train_limit}")
    if expected_heldout is not None and len(heldout_ids) != expected_heldout:
        raise ValueError(f"split manifest heldout count is {len(heldout_ids)}; expected {expected_heldout}")
    selected = [("train", item_id) for item_id in train_ids[:train_limit]]
    selected.extend(("heldout", item_id) for item_id in heldout_ids)
    items = []
    for split, item_id in selected:
        image_path = os.path.join(root, item_id.replace("/", os.sep))
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"probe image is missing: {image_path}")
        caption, caption_path = read_sidecar_caption(image_path)
        if placeholder not in caption:
            raise ValueError(f"probe caption must contain {placeholder!r}: {caption_path}")
        items.append({
            "split": split,
            "dataset_relative_item_id": item_id,
            "image_path": image_path,
            "caption_path": caption_path,
            "caption": caption,
        })
    return items


def resume_key(record: Mapping[str, Any]) -> Tuple[str, int, str, int]:
    return (
        str(record["run_id"]),
        int(record["checkpoint_step"]),
        str(record["probe_case_id"]),
        int(record["tap_layer"]),
    )


def completed_case_keys(records: Iterable[Mapping[str, Any]], tap_layers: Sequence[int]) -> set[Tuple[str, int, str]]:
    expected = {int(layer) for layer in tap_layers}
    grouped: Dict[Tuple[str, int, str], set[int]] = defaultdict(set)
    for record in records:
        key = (str(record["run_id"]), int(record["checkpoint_step"]), str(record["probe_case_id"]))
        grouped[key].add(int(record["tap_layer"]))
    return {key for key, layers in grouped.items() if layers == expected}


def _torch():
    import torch
    return torch


def _flat_float(tensor: Any):
    return tensor.detach().float().reshape(-1)


def _rms(tensor: Any) -> float:
    torch = _torch()
    return float(torch.mean(_flat_float(tensor).square()).sqrt().item())


def _loss(prediction: Any, target: Any) -> float:
    torch = _torch()
    return float(torch.mean((_flat_float(prediction) - _flat_float(target)).square()).item())


def _cosine(left: Any, right: Any, epsilon: float) -> Dict[str, Any]:
    torch = _torch()
    left_flat, right_flat = _flat_float(left), _flat_float(right)
    left_norm, right_norm = torch.linalg.vector_norm(left_flat), torch.linalg.vector_norm(right_flat)
    valid = bool(left_norm.item() > epsilon and right_norm.item() > epsilon)
    return {"value": float(torch.dot(left_flat, right_flat).div(left_norm * right_norm).item()) if valid else None, "valid": valid}


def compute_layer_metrics(
    none_prediction: Any,
    full_prediction: Any,
    solo_prediction: Any,
    leave_one_out_prediction: Any,
    target: Any,
    *,
    epsilon: float = 1.0e-12,
) -> Dict[str, Any]:
    full_residual = full_prediction - none_prediction
    solo_residual = solo_prediction - none_prediction
    leave_residual = leave_one_out_prediction - none_prediction
    marginal_residual = full_prediction - leave_one_out_prediction
    loss_none = _loss(none_prediction, target)
    loss_full = _loss(full_prediction, target)
    loss_solo = _loss(solo_prediction, target)
    loss_leave = _loss(leave_one_out_prediction, target)
    denominator = abs(loss_none) + epsilon
    full_rms, solo_rms = _rms(full_residual), _rms(solo_residual)
    leave_rms, marginal_rms = _rms(leave_residual), _rms(marginal_residual)
    interaction = full_residual - solo_residual - leave_residual
    metrics = {
        "loss_none": loss_none,
        "loss_full": loss_full,
        "loss_solo": loss_solo,
        "loss_leave_one_out": loss_leave,
        "tap_gain_full_vs_none": (loss_none - loss_full) / denominator,
        "solo_gain_vs_none": (loss_none - loss_solo) / denominator,
        "leave_one_out_gain_vs_none": (loss_none - loss_leave) / denominator,
        "marginal_gain_in_full": (loss_leave - loss_full) / denominator,
        "marginal_gain_vs_leave_one_out": (loss_leave - loss_full) / (abs(loss_leave) + epsilon),
        "interaction_gain": (loss_solo + loss_leave - loss_full - loss_none) / denominator,
        "full_residual_rms": full_rms,
        "solo_residual_rms": solo_rms,
        "leave_one_out_residual_rms": leave_rms,
        "marginal_residual_rms": marginal_rms,
        "solo_relative_rms_error": _rms(solo_residual - full_residual) / (full_rms + epsilon),
        "leave_one_out_relative_rms_error": _rms(leave_residual - full_residual) / (full_rms + epsilon),
        "solo_rms_ratio": solo_rms / (full_rms + epsilon),
        "marginal_rms_ratio": marginal_rms / (full_rms + epsilon),
    }
    for name, left, right in (
        ("solo_full_cosine", solo_residual, full_residual),
        ("marginal_full_cosine", marginal_residual, full_residual),
        ("leave_one_out_full_cosine", leave_residual, full_residual),
        ("full_target_cosine", full_residual, target - none_prediction),
        ("solo_target_cosine", solo_residual, target - none_prediction),
        ("marginal_target_cosine", marginal_residual, target - none_prediction),
    ):
        result = _cosine(left, right, epsilon)
        metrics[name] = result["value"]
        metrics[f"{name}_valid"] = result["valid"]
    return metrics


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def aggregate_records(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    records = list(records)
    dimensions = (
        ("checkpoint_split_layer", ("checkpoint_step", "split", "tap_layer")),
        ("checkpoint_split_timestep_layer", ("checkpoint_step", "split", "timestep", "tap_layer")),
    )
    metric_names = sorted({
        key
        for record in records
        for key, value in record.get("metrics", {}).items()
        if not key.endswith("_valid") and (value is None or isinstance(value, (int, float)))
    })
    output = []
    for group_name, keys in dimensions:
        grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[tuple(record[key] for key in keys)].append(record)
        for identity, rows in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
            entry = {"group": group_name, **dict(zip(keys, identity)), "record_count": len(rows), "metrics": {}}
            for metric in metric_names:
                values = [float(row["metrics"][metric]) for row in rows if row["metrics"].get(metric) is not None]
                if not values:
                    entry["metrics"][metric] = {"count": 0, "mean": None, "median": None, "p10": None, "p90": None, "std": None, "positive_fraction": None}
                    continue
                entry["metrics"][metric] = {
                    "count": len(values),
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                    "p10": _percentile(values, 0.10),
                    "p90": _percentile(values, 0.90),
                    "std": statistics.pstdev(values),
                    "positive_fraction": sum(value > 0 for value in values) / len(values),
                }
            output.append(entry)
    return output
