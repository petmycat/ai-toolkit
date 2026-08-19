from __future__ import annotations

import json
import math
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from extensions_built_in.ideogram4_v3_residual_ablation.helpers import sha256_file


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


def append_jsonl(path: os.PathLike | str, record: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def rewrite_jsonl(path: os.PathLike | str, records: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
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


def load_json(path: os.PathLike | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


class DeterministicStratifiedTimestepSampler:
    def __init__(self, seed: int, bins: int = 10, maximum: int = 1000):
        if bins <= 1 or maximum <= 0 or maximum % bins:
            raise ValueError("timestep maximum must be positive and divisible by bins")
        self.seed = int(seed)
        self.bins = int(bins)
        self.maximum = int(maximum)
        self.bin_width = maximum // bins

    def sample(self, step: int) -> Tuple[int, int]:
        if step <= 0:
            raise ValueError("step must be positive")
        bin_index = (step - 1) % self.bins
        cycle = (step - 1) // self.bins
        rng = random.Random((self.seed << 32) ^ (cycle * self.bins + bin_index))
        lower = bin_index * self.bin_width
        upper = self.maximum if bin_index == self.bins - 1 else (bin_index + 1) * self.bin_width - 1
        return rng.randint(lower, upper), bin_index


def detect_active_groups(registry: Sequence[Mapping[str, Any]], epsilon: float = 0.0) -> Tuple[str, ...]:
    active = set()
    for row in registry:
        down_norm = row.get("down_fp32_norm")
        up_norm = row.get("up_fp32_norm")
        if down_norm is None or up_norm is None:
            raise RuntimeError("actual V3 registry must include down_fp32_norm and up_fp32_norm")
        if not row.get("down_finite", True) or not row.get("up_finite", True):
            raise RuntimeError(f"V3 weights contain non-finite tensors: {row.get('module_name')}")
        if float(down_norm) > epsilon and float(up_norm) > epsilon:
            active.add(str(row["group_id"]))
    if not active:
        raise RuntimeError("actual V3 weights contain no active residual groups")
    return tuple(sorted(active))


def validate_registry_contract(actual: Sequence[Mapping[str, Any]], recorded: Mapping[str, Any]) -> None:
    recorded_rows = recorded.get("modules")
    if not isinstance(recorded_rows, list):
        raise RuntimeError("residual ablation registry lacks modules")
    actual_by_name = {str(row["module_name"]): row for row in actual}
    recorded_by_name = {str(row["module_name"]): row for row in recorded_rows}
    if set(actual_by_name) != set(recorded_by_name):
        missing = sorted(set(recorded_by_name) - set(actual_by_name))
        extra = sorted(set(actual_by_name) - set(recorded_by_name))
        raise RuntimeError(f"V3 registry module mismatch; missing={missing[:5]}, extra={extra[:5]}")
    for name, row in actual_by_name.items():
        recorded_row = recorded_by_name[name]
        for field in ("group_id", "block_index", "kind", "rank", "down_shape", "up_shape"):
            if row.get(field) != recorded_row.get(field):
                raise RuntimeError(f"V3 registry field mismatch for {name}: {field}")


def gate_regularization(q_values: Any):
    return q_values.float().square().mean()


def temporal_smoothness(router: Any, normalized_timestep: Any, delta: float):
    import torch

    center = normalized_timestep.float().reshape(-1)
    left = (center - float(delta)).clamp(0.0, 1.0)
    right = (center + float(delta)).clamp(0.0, 1.0)
    left_q = router(left)
    right_q = router(right)
    return torch.mean((right_q - left_q).float().square())


def validation_grid(canonical: Sequence[int], dense: Sequence[int], seeds: Sequence[int]) -> List[Dict[str, Any]]:
    records = []
    for grid_name, timesteps in (("canonical", canonical), ("dense", dense)):
        for seed in seeds:
            for timestep in timesteps:
                records.append({"grid": grid_name, "seed": int(seed), "timestep": int(timestep)})
    return records


def select_best_validation(records: Iterable[Mapping[str, Any]]) -> Tuple[int, float]:
    by_step: Dict[int, List[float]] = {}
    for record in records:
        if record.get("split") != "validation" or record.get("grid") != "canonical":
            continue
        loss = record.get("loss")
        if isinstance(loss, (int, float)) and math.isfinite(float(loss)):
            by_step.setdefault(int(record["step"]), []).append(float(loss))
    if not by_step:
        raise RuntimeError("best router selection requires canonical validation records")
    step, losses = min(by_step.items(), key=lambda item: (sum(item[1]) / len(item[1]), item[0]))
    return step, sum(losses) / len(losses)


def router_config_payload(config: Any, active_registry: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema": "ai-toolkit.ideogram4-v3-phase-c-router-config",
        "schema_version": 1,
        "canonical_api": "toolkit.residual_gating.ResidualGateRouter",
        "runtime_api": "toolkit.residual_gating.ResidualGateRuntime",
        "timestep_feature_encoding": "sin_cos_pi_integer_v2",
        "style_strength": {
            "range": [0.0, 1.0],
            "default": 0.5,
            "step": 0.01,
            "equation_lower": "g = 2*s for 0 <= s <= 0.5",
            "equation_upper": "g = 1 + (2*s - 1)*q(t) for 0.5 < s <= 1",
            "anchors": {"0.0": "V3 diffusion residual off", "0.5": "exact ordinary V3", "1.0": "full Phase C orchestration"},
        },
        "training_gate_equation": "g = 1 + q(t)",
        "timestep_embed_dim": int(config.timestep_embed_dim),
        "hidden_dim": int(config.hidden_dim),
        "router_rank": int(config.router_rank),
        "q_max": float(config.q_max),
        "active_registry": dict(active_registry),
        "training": {
            "steps": int(config.steps),
            "optimizer": "AdamW",
            "fresh_optimizer": True,
            "learning_rate": float(config.learning_rate),
            "weight_decay": float(config.weight_decay),
            "lambda_gate": float(config.lambda_gate),
            "timestep_sampling": config.timestep_sampling,
            "train_item_limit": config.train_item_limit,
            "validation_item_limit": config.validation_item_limit,
            "timestep_bins": int(config.timestep_bins),
            "temporal_smoothness": asdict(config.temporal_smoothness),
        },
    }


def artifact_ref(path: Path, root: Path) -> Dict[str, str]:
    return {"path": os.path.relpath(path, root).replace(os.sep, "/"), "sha256": sha256_file(path)}
