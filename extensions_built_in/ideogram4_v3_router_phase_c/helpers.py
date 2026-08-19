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
    def __init__(self, seed: int, bins: int = 16, maximum: int = 1000):
        if bins <= 1 or maximum <= 0 or bins > maximum:
            raise ValueError("timestep maximum must be positive and bins must lie in [2, maximum]")
        self.seed = int(seed)
        self.bins = int(bins)
        self.maximum = int(maximum)

    def sample(self, step: int) -> Tuple[int, int]:
        if step <= 0:
            raise ValueError("step must be positive")
        cycle = (step - 1) // self.bins
        cycle_offset = (step - 1) % self.bins
        permutation = list(range(self.bins))
        random.Random((self.seed << 32) ^ cycle).shuffle(permutation)
        bin_index = permutation[cycle_offset]
        population = self.maximum + 1
        lower = (bin_index * population) // self.bins
        upper = ((bin_index + 1) * population) // self.bins - 1
        rng = random.Random((self.seed << 32) ^ (cycle * self.bins + bin_index) ^ 0x9E3779B97F4A7C15)
        return rng.randint(lower, upper), bin_index


def aggregate_projected_activator_occurrences(
    projected: Any,
    *,
    occurrence_count: int,
    token_count: int,
    mode: str = "additive",
):
    if occurrence_count <= 0 or token_count <= 0:
        raise ValueError("activator occurrence_count and token_count must be positive")
    if mode != "additive":
        raise ValueError("only additive activator occurrence aggregation is supported")
    if projected.ndim != 2 or projected.shape[0] != occurrence_count * token_count:
        raise ValueError(
            f"projected activator states must have shape "
            f"[{occurrence_count * token_count},D]"
        )
    occurrences = projected.reshape(occurrence_count, token_count, projected.shape[-1])
    return occurrences.sum(dim=0, keepdim=True)


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
    records = [
        {"grid": "canonical", "seed": int(seed), "timestep": int(timestep)}
        for seed in seeds
        for timestep in canonical
    ]
    records.extend(
        {"grid": "dense", "seed": None, "timestep": int(timestep)}
        for timestep in dense
    )
    return records


def select_best_validation(records: Iterable[Mapping[str, Any]]) -> Tuple[int, float]:
    by_step: Dict[int, List[float]] = {}
    for record in records:
        if record.get("split") != "validation" or record.get("grid") != "canonical":
            continue
        loss = record.get("full_router_loss")
        if not isinstance(loss, (int, float)):
            loss = record.get("loss")
        if isinstance(loss, (int, float)) and math.isfinite(float(loss)):
            by_step.setdefault(int(record["step"]), []).append(float(loss))
    if not by_step:
        raise RuntimeError("best router selection requires canonical validation records")
    step, losses = min(by_step.items(), key=lambda item: (sum(item[1]) / len(item[1]), item[0]))
    return step, sum(losses) / len(losses)


def router_config_payload(config: Any, active_registry: Mapping[str, Any], conditioning_dim: int) -> Dict[str, Any]:
    anchor_count = int(config.temporal_anchor_count)
    return {
        "schema": "ai-toolkit.ideogram4-v3-phase-c-v2-router-config",
        "schema_version": 2,
        "contract_revision": 4,
        "canonical_api": "toolkit.residual_gating.ResidualGateRouter",
        "runtime_api": "toolkit.residual_gating.ResidualGateRuntime",
        "conditioning_source": config.conditioning_source,
        "conditioning_dim": int(conditioning_dim),
        "conditioning_normalization": "shared_layer_norm",
        "activator_mask_schema": "a1-a2-trigger-mask-v1",
        "activator_token_count": int(config.activator_token_count),
        "activator_occurrence_count": int(config.activator_occurrence_count),
        "activator_occurrence_mode": config.activator_occurrence_mode,
        "activator_pre_router_aggregation": "sum_by_virtual_token_index",
        "activator_token_dim": int(config.activator_token_dim),
        "activator_code_dim": int(config.activator_token_count * config.activator_token_dim),
        "temporal_anchor_count": anchor_count,
        "anchor_locations": [index / float(anchor_count - 1) for index in range(anchor_count)],
        "temporal_interpolation": config.temporal_interpolation,
        "contextual_rank": int(config.contextual_rank),
        "q_max": float(config.q_max),
        "style_strength": {
            "range": [0.0, 1.0], "default": 0.5, "step": 0.01,
            "equation_lower": "g = 2*s for 0 <= s <= 0.5",
            "equation_upper": "g = 1 + (2*s - 1)*q(t,z_A) for 0.5 < s <= 1",
            "anchors": {"0.0": "V3 diffusion residual off", "0.5": "exact ordinary V3", "1.0": "full Phase C V2 orchestration"},
        },
        "training_gate_equation": "g = 1 + q(t,z_A)",
        "active_registry": dict(active_registry),
        "training": {
            "steps": int(config.steps), "seed": int(config.seed),
            "checkpoint_every": int(config.checkpoint_every), "resume": bool(config.resume),
            "optimizer": config.optimizer,
            "optimizer_params": dict(config.optimizer_params), "fresh_optimizer": True,
            "learning_rate": float(config.learning_rate), "weight_decay": float(config.weight_decay),
            "lambda_universal": float(config.lambda_universal),
            "lambda_contextual": float(config.lambda_contextual),
            "contextual_mean_multiplier": float(config.contextual_mean_multiplier),
            "same_timestep_content_batch": int(config.same_timestep_content_batch),
            "distinct_items_per_update": bool(config.distinct_items_per_update),
            "gradient_checkpointing": asdict(config.gradient_checkpointing),
            "timestep_sampling": config.timestep_sampling,
            "timestep_bins": int(config.timestep_bins),
            "train_item_limit": config.train_item_limit,
            "validation_item_limit": config.validation_item_limit,
            "temporal_smoothness": asdict(config.temporal_smoothness),
        },
        "validation": asdict(config.validation),
        "artifacts": asdict(config.artifacts),
    }


def artifact_ref(path: Path, root: Path) -> Dict[str, str]:
    return {"path": os.path.relpath(path, root).replace(os.sep, "/"), "sha256": sha256_file(path)}
