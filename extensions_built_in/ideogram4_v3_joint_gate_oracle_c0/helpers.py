from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


FAMILY_ORDER = ("adaln", "attention", "mlp")
NON_GATE_DIAGNOSTIC_GROUPS = frozenset({"global:adaln", "global:other"})


def sha256_file(path: os.PathLike | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: os.PathLike | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def load_jsonl(path: os.PathLike | str) -> list[Dict[str, Any]]:
    output = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} must be an object: {path}")
            output.append(value)
    return output


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


def canonical_group_rows(registry_payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    modules = registry_payload.get("modules")
    groups = registry_payload.get("groups")
    if not isinstance(modules, Sequence) or not isinstance(groups, Mapping):
        raise RuntimeError("residual registry must contain modules and groups")
    rows: Dict[str, Dict[str, Any]] = {}
    module_groups: Dict[str, list[str]] = defaultdict(list)
    for row in modules:
        if not isinstance(row, Mapping):
            raise RuntimeError("registry module row must be an object")
        group_id = str(row.get("group_id", ""))
        kind = str(row.get("kind", ""))
        block = row.get("block_index")
        module_name = str(row.get("module_name", row.get("name", "")))
        if not group_id or not module_name:
            raise RuntimeError("registry module row lacks group_id or module name")
        module_groups[group_id].append(module_name)
        identity = rows.setdefault(group_id, {"group_id": group_id, "block_index": int(block), "family": kind})
        if identity["block_index"] != int(block) or identity["family"] != kind:
            raise RuntimeError(f"inconsistent group identity in registry: {group_id}")
    if set(rows) != set(str(key) for key in groups):
        raise RuntimeError("registry module partition disagrees with groups mapping")
    ordered = sorted(rows.values(), key=lambda row: (row["block_index"], FAMILY_ORDER.index(row["family"])))
    if len(ordered) != 102:
        raise RuntimeError(f"C0 requires exactly 102 active groups, found {len(ordered)}")
    expected = {(block, family) for block in range(34) for family in FAMILY_ORDER}
    actual = {(row["block_index"], row["family"]) for row in ordered}
    if actual != expected:
        raise RuntimeError("registry does not implement exact 34 x [AdaLN, attention, MLP] grouping")
    for index, row in enumerate(ordered):
        row["index"] = index
        row["modules"] = sorted(module_groups[row["group_id"]])
    return ordered


def group_registry_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _balanced_scalar(rows: Sequence[Mapping[str, Any]], field: str) -> tuple[float | None, int]:
    by_item_seed: Dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        value = _finite_float(row.get("metrics", {}).get(field))
        if value is None:
            continue
        by_item_seed[(str(row.get("item_id", "")), int(row.get("noise_seed", 0)))].append(value)
    by_item: Dict[str, list[float]] = defaultdict(list)
    for (item_id, _), values in by_item_seed.items():
        by_item[item_id].append(statistics.fmean(values))
    values = [statistics.fmean(item_values) for item_values in by_item.values() if item_values]
    return (statistics.fmean(values), len(values)) if values else (None, 0)


def _fd_vote(row: Mapping[str, Any]) -> tuple[float, float, float] | None:
    metrics = row.get("metrics", {})
    if not metrics.get("finite_difference_selected") or not str(metrics.get("fd_result", "")).startswith("prefer_"):
        return None
    losses = row.get("candidate_losses", {})
    values = tuple(_finite_float(losses.get(str(scale))) for scale in (0.75, 1.0, 1.25))
    if any(value is None for value in values):
        return None
    left, center, right = (float(value) for value in values)
    if any(_finite_float(metrics.get(name)) is None for name in ("left_slope", "right_slope", "central_secant", "curvature")):
        return None
    tolerance = max(1.0e-12, 1.0e-6 * abs(center))
    losses_by_scale = {0.75: left, 1.0: center, 1.25: right}
    minimum = min(losses_by_scale.values())
    winners = [scale for scale, loss in losses_by_scale.items() if abs(loss - minimum) <= tolerance]
    if 1.0 in winners:
        return 0.0, 1.0, 0.0
    if len(winners) == 2 and set(winners) == {0.75, 1.25}:
        return 0.5, 0.0, 0.5
    if len(winners) == 3:
        return 0.0, 1.0, 0.0
    winner = winners[0]
    return {0.75: (1.0, 0.0, 0.0), 1.0: (0.0, 1.0, 0.0), 1.25: (0.0, 0.0, 1.0)}[winner]


def _balanced_vote(rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[float, float, float] | None, int]:
    by_item_seed: Dict[tuple[str, int], list[tuple[float, float, float]]] = defaultdict(list)
    for row in rows:
        vote = _fd_vote(row)
        if vote is not None:
            by_item_seed[(str(row.get("item_id", "")), int(row.get("noise_seed", 0)))].append(vote)
    by_item: Dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for (item_id, _), votes in by_item_seed.items():
        by_item[item_id].append(tuple(statistics.fmean(values) for values in zip(*votes)))
    item_votes = [tuple(statistics.fmean(values) for values in zip(*votes)) for votes in by_item.values() if votes]
    if not item_votes:
        return None, 0
    return tuple(statistics.fmean(values) for values in zip(*item_votes)), len(item_votes)


def build_prior_vectors(
    records: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    timestep: int,
    *,
    fd_splits: Sequence[str] = ("train",),
) -> Dict[str, Any]:
    group_ids = [str(row["group_id"]) for row in group_rows]
    group_set = set(group_ids)
    scoped = [row for row in records if int(row.get("timestep", -1)) == int(timestep)]
    non_gate_diagnostics = sorted({
        str(row.get("group_id")) for row in scoped
        if str(row.get("group_id")) in NON_GATE_DIAGNOSTIC_GROUPS
    })
    unknown = sorted({
        str(row.get("group_id")) for row in scoped
        if str(row.get("group_id")) not in group_set
        and str(row.get("group_id")) not in NON_GATE_DIAGNOSTIC_GROUPS
    })
    if unknown:
        raise RuntimeError(f"C0 prior records contain unknown groups: {unknown[:10]}")
    duplicate_keys = set()
    by_group: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in scoped:
        if str(row.get("group_id")) in NON_GATE_DIAGNOSTIC_GROUPS:
            continue
        key = (str(row.get("run_id", "")), str(row.get("probe_case_id", "")), str(row.get("group_id", "")))
        if key in duplicate_keys:
            raise RuntimeError(f"duplicate C0 prior record key: {key}")
        duplicate_keys.add(key)
        by_group[str(row["group_id"])].append(row)
    fd_q, mean_grad, details = [], [], []
    missing_record_groups = [group for group in group_ids if not by_group[group]]
    if missing_record_groups:
        raise RuntimeError(f"C0 prior records lack all evidence for groups: {missing_record_groups[:10]}")
    for group in group_ids:
        values = by_group[group]
        split_stats = {}
        for split in ("train", "heldout"):
            split_rows = [row for row in values if row.get("split") == split]
            gradient, gradient_count = _balanced_scalar(split_rows, "dL_dg")
            vote, fd_count = _balanced_vote(split_rows)
            split_stats[split] = {"gradient": gradient, "gradient_count": gradient_count, "vote": vote, "fd_count": fd_count, "record_count": len(split_rows)}
        available_gradients = [split_stats[split]["gradient"] for split in fd_splits if split_stats[split]["gradient"] is not None]
        mean_gradient = statistics.fmean(available_gradients) if available_gradients else 0.0
        invalid_fd_splits = sorted(set(fd_splits) - {"train", "heldout"})
        if invalid_fd_splits:
            raise ValueError(f"invalid FD evidence splits: {invalid_fd_splits}")
        available_votes = [split_stats[split]["vote"] for split in fd_splits if split_stats[split]["vote"] is not None]
        if available_votes:
            composition = tuple(statistics.fmean(values) for values in zip(*available_votes))
            fd_status = "+".join(split for split in fd_splits if split_stats[split]["vote"] is not None)
            fd_q_value = -0.25 * composition[0] + 0.25 * composition[2]
        else:
            composition, fd_status, fd_q_value = None, "missing", 0.0
        fd_q.append(fd_q_value)
        mean_grad.append(mean_gradient)
        details.append({"group_id": group, "gradient_mean": mean_gradient, "gradient_status": "complete" if len(available_gradients) == 2 else ("partial" if available_gradients else "no_finite_gradient"), "fd_composition": composition, "fd_status": fd_status, "fd_q": fd_q_value, "split_stats": split_stats})
    return {"fd_q": fd_q, "mean_gradient": mean_grad, "details": details, "unknown_groups": unknown, "ignored_non_gate_diagnostic_groups": non_gate_diagnostics, "timestep": int(timestep), "fd_evidence_splits": list(fd_splits), "fd_tie_policy": "center_then_symmetric_outer", "balance_order": ["noise_seed", "item_id", "split"]}


def select_balanced_batch(items: Sequence[Any], step: int, count: int, seed: int) -> list[Any]:
    import random

    if not items or count <= 0 or count > len(items):
        raise ValueError("C0 train batch must be non-empty and no larger than the train set")
    start = (step - 1) * count
    selected = []
    cycle = start // len(items)
    offset = start % len(items)
    while len(selected) < count:
        order = list(range(len(items)))
        random.Random((seed << 32) ^ cycle ^ 0x9E3779B97F4A7C15).shuffle(order)
        for index in order[offset:]:
            if index not in selected:
                selected.append(index)
                if len(selected) == count:
                    break
        cycle += 1
        offset = 0
    return [items[index] for index in selected]


def q_statistics(q: Any, group_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    import torch

    value = q.detach().float().reshape(-1)
    absolute = value.abs()
    output = {
        "mean_q": float(value.mean().item()),
        "mean_abs_q": float(absolute.mean().item()),
        "rms_q": float(value.square().mean().sqrt().item()),
        "max_abs_q": float(absolute.max().item()),
        "fraction_abs_lt_0_01": float((absolute < 0.01).float().mean().item()),
        "fraction_abs_gt_0_05": float((absolute > 0.05).float().mean().item()),
        "fraction_abs_gt_0_10": float((absolute > 0.10).float().mean().item()),
        "fraction_abs_gt_0_20": float((absolute > 0.20).float().mean().item()),
        "boundary_fraction": float((absolute >= 0.249).float().mean().item()),
        "families": {},
    }
    for family in FAMILY_ORDER:
        indices = [int(row["index"]) for row in group_rows if row["family"] == family]
        family_value = value[torch.tensor(indices, device=value.device)]
        output["families"][family] = {
            "mean_q": float(family_value.mean().item()),
            "mean_abs_q": float(family_value.abs().mean().item()),
            "rms_q": float(family_value.square().mean().sqrt().item()),
            "max_abs_q": float(family_value.abs().max().item()),
        }
    return output


def cosine(a: Any, b: Any, epsilon: float = 1.0e-12) -> float:
    import torch

    left = a.detach().float().reshape(-1)
    right = b.detach().float().reshape(-1).to(left.device)
    denominator = left.norm() * right.norm()
    if float(denominator.item()) <= epsilon:
        return 0.0
    return float(torch.dot(left, right).div(denominator).item())


def sign_agreement(q: Any, fd_q: Any) -> Dict[str, Any]:
    import torch

    left = q.detach().float().reshape(-1)
    right = fd_q.detach().float().reshape(-1).to(left.device)
    mask = right != 0
    count = int(mask.sum().item())
    return {
        "evidence_count": count,
        "agreement_fraction": float((torch.sign(left[mask]) == torch.sign(right[mask])).float().mean().item()) if count else None,
    }


def interpolation_weights(timestep: float, anchors: Sequence[int] = (100, 500, 900)) -> Tuple[int, int, float]:
    value = float(timestep)
    if value <= anchors[0]:
        return 0, 0, 0.0
    if value >= anchors[-1]:
        last = len(anchors) - 1
        return last, last, 0.0
    for index in range(len(anchors) - 1):
        left, right = anchors[index], anchors[index + 1]
        if left <= value <= right:
            return index, index + 1, (value - left) / (right - left)
    raise AssertionError("unreachable interpolation interval")


def interpolate_oracle(timestep: float, q_by_timestep: Mapping[int, Any]):
    left, right, weight = interpolation_weights(timestep, tuple(sorted(q_by_timestep)))
    ordered = [q_by_timestep[key] for key in sorted(q_by_timestep)]
    if left == right:
        return ordered[left]
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def metric_payload(v3_loss: float, candidate_loss: float, epsilon: float = 1.0e-12) -> Dict[str, float]:
    gain = float(v3_loss) - float(candidate_loss)
    return {
        "loss": float(candidate_loss),
        "absolute_gain": gain,
        "normalized_gain": gain / (float(v3_loss) + epsilon),
        "better_than_v3": bool(candidate_loss < v3_loss),
    }


def summarize_validation(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["timestep"], record["split"], record["candidate"])].append(record)
    output = []
    for identity, values in sorted(grouped.items()):
        gains = [float(row["absolute_gain"]) for row in values]
        normalized = [float(row["normalized_gain"]) for row in values]
        output.append({
            "timestep": identity[0],
            "split": identity[1],
            "candidate": identity[2],
            "record_count": len(values),
            "mean_loss": statistics.fmean(float(row["loss"]) for row in values),
            "mean_absolute_gain": statistics.fmean(gains),
            "mean_normalized_gain": statistics.fmean(normalized),
            "positive_fraction": sum(value > 0 for value in gains) / len(gains),
            "gain_vs_phase_c_v2_reference": statistics.fmean(gains) / 4.4e-6,
        })
    return {"phase_c_v2_reference_gain": 4.4e-6, "rows": output}


def safe_prompt_slug(index: int, prompt: str) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]
    return f"prompt_{index:03d}_{digest}"
