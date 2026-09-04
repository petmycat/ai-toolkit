from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file


FIXED_PROBE_SCHEMA_VERSION = 5
FIXED_PROBE_ALGORITHM = "split_label_pair_region_random_sample_v2"
FIXED_PROBE_NOISE_ALGORITHM = "cpu_torch_randn_seeded_by_probe_identity_v1"


def _region_bounds(region: Mapping[str, Any]) -> tuple[int, int, int]:
    lower_key, upper_key = ("min", "max") if "min" in region or "max" in region else ("start", "end")
    if not all(key in region for key in (lower_key, upper_key, "timestep_count")):
        raise ValueError("fixed probe regions require min, max, and timestep_count")
    start, end, count = (region[key] for key in (lower_key, upper_key, "timestep_count"))
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end, count)):
        raise ValueError("fixed probe region bounds and count must be integers")
    if start < 0 or end > 1000 or end <= start or count < 1 or count > end - start:
        raise ValueError("fixed probe regions must be non-empty half-open intervals")
    return start, end, count


def validate_regions(regions: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(regions, Mapping) or not regions:
        raise ValueError("fixed probes require at least one timestep region")
    result: dict[str, dict[str, Any]] = {}
    previous_end = 0
    timestep_count: int | None = None
    ordered_regions = sorted(regions.items(), key=lambda item: _region_bounds(item[1])[0])
    for name, metadata in ordered_regions:
        if not isinstance(name, str) or not name or not isinstance(metadata, Mapping):
            raise ValueError("fixed probe region metadata is invalid")
        start, end, count = _region_bounds(metadata)
        if start != previous_end:
            raise ValueError("fixed probe regions must be sorted, contiguous, and cover [0,1000]")
        if timestep_count is None:
            timestep_count = count
        elif count != timestep_count:
            raise ValueError("fixed probe timestep_count must be uniform across regions")
        result[name] = dict(metadata)
        previous_end = end
    if previous_end != 1000:
        raise ValueError("fixed probe regions must cover [0,1000]")
    return result


def _seed_material(seed: int, split_label: str, pair_fingerprint: str, region: str, ordinal: int) -> str:
    return f"{seed}|{split_label}|{pair_fingerprint}|{region}|{ordinal}"


def probe_fingerprint(seed: int, pair_fingerprint: str, region: str, ordinal: int, split_label: str = "") -> str:
    return hashlib.sha256(_seed_material(int(seed), split_label, pair_fingerprint, region, int(ordinal)).encode("utf-8")).hexdigest()


def deterministic_probe_noise_seed(seed: int, pair_fingerprint: str, region: str, ordinal: int, split_label: str = "") -> int:
    material = f"{_seed_material(int(seed), split_label, pair_fingerprint, region, int(ordinal))}|noise|{FIXED_PROBE_NOISE_ALGORITHM}"
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def deterministic_probe_noise(reference: torch.Tensor, seed: int, pair_fingerprint: str, region: str, ordinal: int, split_label: str = "") -> tuple[torch.Tensor, int]:
    if reference.numel() == 0:
        raise ValueError("fixed probe noise reference must be non-empty")
    derived_seed = deterministic_probe_noise_seed(seed, pair_fingerprint, region, ordinal, split_label)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derived_seed)
    noise = torch.randn(reference.shape, generator=generator, device="cpu", dtype=torch.float32)
    return noise.to(device=reference.device, dtype=reference.dtype), derived_seed


def deterministic_probe_timesteps(seed: int, split_label: str, pair_fingerprint: str, region: str, start: int, end: int, count: int) -> tuple[int, ...]:
    _region_bounds({"min": start, "max": end, "timestep_count": count})
    derived_seed = int.from_bytes(hashlib.sha256(f"{seed}|{split_label}|{pair_fingerprint}|{region}|timesteps".encode("utf-8")).digest()[:8], "big")
    return tuple(random.Random(derived_seed).sample(range(start, end), count))


def deterministic_probe_timestep(seed: int, pair_fingerprint: str, region: str, ordinal: int, start: int, end: int, split_label: str = "", count: int | None = None) -> int:
    # The legacy helper omits count; use the complete region in that case.
    sample_count = end - start if count is None else count
    values = deterministic_probe_timesteps(seed, split_label, pair_fingerprint, region, start, end, sample_count)
    if ordinal < 0 or ordinal >= len(values):
        raise ValueError("fixed probe ordinal exceeds the timestep region capacity")
    return values[ordinal]


def select_probe_pairs(pair_fingerprints: Sequence[str], count: int, seed: int, split_label: str = "train") -> tuple[str, ...]:
    values = tuple(sorted(set(str(value) for value in pair_fingerprints)))
    if isinstance(count, bool) or not isinstance(count, int) or count < 0 or count > len(values):
        raise ValueError("fixed probe pair count exceeds split capacity")
    if count == 0:
        return ()
    derived = int.from_bytes(hashlib.sha256(f"{seed}|{split_label}|pair-selection".encode("utf-8")).digest()[:8], "big")
    return tuple(sorted(random.Random(derived).sample(values, count)))


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _cpu_float32(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError(f"fixed probe tensor {name} must be finite and non-empty")
    return tensor


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def save_fixed_probes(json_path: str | Path, safetensors_path: str | Path, probes: Iterable[Mapping[str, Any]], *, split_fingerprint: str, seed: int, regions: Mapping[str, Mapping[str, Any]]) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("fixed probe seed must be an integer")
    if not split_fingerprint:
        raise ValueError("fixed probes require a dataset split fingerprint")
    region_metadata = validate_regions(regions)
    probe_list = list(probes)
    tensors: dict[str, torch.Tensor] = {}
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts: dict[tuple[str, str], int] = {}
    pair_counts: dict[tuple[str, str, str], int] = {}
    for index, probe in enumerate(probe_list):
        region = str(probe.get("region", ""))
        if region not in region_metadata:
            raise ValueError(f"fixed probe {index} has unknown region {region!r}")
        pair = str(probe.get("pair_fingerprint", ""))
        ordinal = probe.get("ordinal", -1)
        if not pair or isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise ValueError(f"fixed probe {index} metadata is invalid")
        start, end, timestep_count = _region_bounds(region_metadata[region])
        if ordinal < 0 or ordinal >= timestep_count:
            raise ValueError(f"fixed probe {index} ordinal is outside its region")
        fingerprint = str(probe.get("fingerprint", ""))
        expected_fp = probe_fingerprint(seed, pair, region, ordinal, str(probe.get("split_label", "")))
        if fingerprint != expected_fp or fingerprint in seen:
            raise ValueError(f"fixed probe {index} fingerprint is invalid or duplicated")
        timestep = probe.get("timestep")
        if not isinstance(timestep, torch.Tensor) or timestep.numel() != 1:
            raise ValueError(f"fixed probe {index} timestep must contain one value")
        label = str(probe.get("split_label", ""))
        if label not in {"train", "heldout"}:
            raise ValueError(f"fixed probe {index} has invalid split_label")
        expected_t = deterministic_probe_timestep(seed, pair, region, ordinal, start, end, label, timestep_count)
        if int(round(float(timestep.item()))) != expected_t:
            raise ValueError(f"fixed probe {index} timestep is not deterministic")
        expected_noise_seed = deterministic_probe_noise_seed(seed, pair, region, ordinal, label)
        if probe.get("noise_seed") != expected_noise_seed:
            raise ValueError(f"fixed probe {index} noise seed is not deterministic")
        noisy = _cpu_float32(probe["noisy_latent"], f"{index}.noisy_latent")
        target = _cpu_float32(probe["target"], f"{index}.target")
        if noisy.shape != target.shape:
            raise ValueError(f"fixed probe {index} noisy_latent/target shape mismatch")
        prefix = f"probe_{index:04d}"
        tensors[f"{prefix}.noisy_latent"] = noisy
        tensors[f"{prefix}.target"] = target
        entry = {"index": index, "region": region, "split_label": label, "pair_fingerprint": pair, "ordinal": ordinal, "fingerprint": fingerprint, "prompt": str(probe["prompt"]), "noise_seed": expected_noise_seed, "noise_algorithm": FIXED_PROBE_NOISE_ALGORITHM, "noisy_latent": f"{prefix}.noisy_latent", "target": f"{prefix}.target", "timestep": expected_t, "shape": list(noisy.shape), "dtype": "float32", "noisy_latent_sha256": _tensor_hash(noisy), "target_sha256": _tensor_hash(target)}
        entries.append(entry)
        seen.add(fingerprint)
        counts[(label, region)] = counts.get((label, region), 0) + 1
        pair_counts[(label, region, pair)] = pair_counts.get((label, region, pair), 0) + 1
    labels_present = {entry["split_label"] for entry in entries}
    for region, metadata in region_metadata.items():
        pair_count = metadata.get("pair_count")
        if isinstance(pair_count, bool) or not isinstance(pair_count, int) or pair_count < 0:
            raise ValueError(f"fixed probe region {region} pair_count is invalid")
        for label in labels_present:
            expected = pair_count * int(metadata["timestep_count"])
            if counts.get((label, region), 0) != expected:
                raise ValueError(f"fixed probe {label}/{region} count is incomplete")
            pairs = [key for key in pair_counts if key[0] == label and key[1] == region]
            if len(pairs) != pair_count or any(pair_counts[key] != int(metadata["timestep_count"]) for key in pairs):
                raise ValueError(f"fixed probe {label}/{region} pair timestep count is incomplete")
    safetensors_path, json_path = Path(safetensors_path), Path(json_path)
    safetensors_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(safetensors_path), metadata={"artifact": "gen2_fixed_probes", "schema_version": str(FIXED_PROBE_SCHEMA_VERSION), "algorithm": FIXED_PROBE_ALGORITHM, "noise_algorithm": FIXED_PROBE_NOISE_ALGORITHM})
    payload = {"artifact": "gen2_fixed_probes", "schema_version": FIXED_PROBE_SCHEMA_VERSION, "algorithm": FIXED_PROBE_ALGORITHM, "noise_algorithm": FIXED_PROBE_NOISE_ALGORITHM, "tensor_artifact": safetensors_path.name, "split_fingerprint": split_fingerprint, "seed": seed, "regions": json.loads(_json_value(region_metadata)), "count": len(entries), "probes": entries}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_fixed_probes(json_path: str | Path, safetensors_path: str | Path | None = None, *, expected_split_fingerprint: str | None = None, expected_regions: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    json_path = Path(json_path)
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Gen2 fixed probe artifact is unreadable") from error
    if payload.get("artifact") != "gen2_fixed_probes" or payload.get("schema_version") != FIXED_PROBE_SCHEMA_VERSION or payload.get("algorithm") != FIXED_PROBE_ALGORITHM or payload.get("noise_algorithm") != FIXED_PROBE_NOISE_ALGORITHM:
        raise ValueError("unsupported Gen2 fixed probe artifact")
    if expected_split_fingerprint is not None and payload.get("split_fingerprint") != expected_split_fingerprint:
        raise ValueError("fixed probe split fingerprint does not match the active dataset split")
    if expected_regions is not None and payload.get("regions") != json.loads(_json_value(validate_regions(expected_regions))):
        raise ValueError("fixed probe region metadata does not match")
    regions = validate_regions(payload.get("regions"))
    path = Path(safetensors_path) if safetensors_path is not None else json_path.parent / payload["tensor_artifact"]
    tensors = load_file(str(path), device="cpu")
    entries = payload.get("probes")
    if not isinstance(entries, list) or len(entries) != payload.get("count", -1):
        raise ValueError("fixed probe index metadata is invalid")
    result: list[dict[str, Any]] = []
    for expected_index, entry in enumerate(entries):
        if entry.get("index") != expected_index or entry.get("region") not in regions:
            raise ValueError("fixed probe index or region metadata is invalid")
        try:
            noisy, target = tensors[entry["noisy_latent"]], tensors[entry["target"]]
        except KeyError as error:
            raise ValueError(f"fixed probe tensor is missing: {error}") from error
        if noisy.dtype != torch.float32 or target.dtype != torch.float32 or noisy.device.type != "cpu" or target.device.type != "cpu" or noisy.shape != target.shape or not torch.isfinite(noisy).all() or not torch.isfinite(target).all():
            raise ValueError("fixed probe tensors must be CPU float32, finite, and shape aligned")
        if list(noisy.shape) != entry.get("shape") or entry.get("dtype") != "float32" or _tensor_hash(noisy) != entry.get("noisy_latent_sha256") or _tensor_hash(target) != entry.get("target_sha256"):
            raise ValueError("fixed probe tensor shape, dtype, or hash metadata is invalid")
        region, pair, ordinal, label = entry["region"], str(entry["pair_fingerprint"]), entry["ordinal"], str(entry.get("split_label", ""))
        start, end, count = _region_bounds(regions[region])
        if entry["fingerprint"] != probe_fingerprint(int(payload["seed"]), pair, region, ordinal, label) or ordinal < 0 or ordinal >= count:
            raise ValueError("fixed probe fingerprint metadata is invalid")
        if int(entry["timestep"]) != deterministic_probe_timestep(int(payload["seed"]), pair, region, ordinal, start, end, label, count):
            raise ValueError("fixed probe timestep metadata is invalid")
        expected_noise_seed = deterministic_probe_noise_seed(int(payload["seed"]), pair, region, ordinal, label)
        if entry.get("noise_seed") != expected_noise_seed or entry.get("noise_algorithm") != FIXED_PROBE_NOISE_ALGORITHM:
            raise ValueError("fixed probe noise metadata is invalid")
        result.append({"region": region, "split_label": label, "pair_fingerprint": pair, "ordinal": ordinal, "fingerprint": entry["fingerprint"], "prompt": entry["prompt"], "noise_seed": expected_noise_seed, "noise_algorithm": FIXED_PROBE_NOISE_ALGORITHM, "noisy_latent": noisy, "target": target, "timestep": torch.tensor([float(entry["timestep"])], dtype=torch.float32)})
    return result, payload
