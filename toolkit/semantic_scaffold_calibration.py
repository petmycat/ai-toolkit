"""Deterministic helper calibration for semantic scaffold A1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
import statistics
import tempfile
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from toolkit.trigger_binding_artifacts import fingerprint, sha256_file
from toolkit.trigger_data_split import normalize_dataset_relative_item_id
from toolkit.trigger_validation import isolated_rng


CALIBRATION_SCHEMA = 'ai-toolkit.semantic-scaffold-calibration'
CALIBRATION_SCHEMA_VERSION = 1
PROBE_SCHEMA = 'ai-toolkit.semantic-scaffold-probes'
PROBE_SCHEMA_VERSION = 1


class SemanticScaffoldCalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class FixedDiffusionProbeCase:
    probe_case_id: str
    split: str
    item_id: str
    caption_hash: str
    image_hash: str
    noise_seed: int
    timestep: Optional[int]
    sigma: Optional[float]
    target_mode: str
    transform: Dict[str, Any]
    latent_hash: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle, temp_path = tempfile.mkstemp(prefix='.semantic-scaffold-', suffix='.tmp', dir=directory)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def build_fixed_probe_cases(
    items: Iterable[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    *,
    noise_seeds: Sequence[int],
    fixed_timesteps: Sequence[int] = (),
    fixed_sigmas: Sequence[float] = (),
    limit: int = 0,
    target_mode: str = 'flow',
) -> List[FixedDiffusionProbeCase]:
    if bool(fixed_timesteps) == bool(fixed_sigmas):
        raise ValueError('probe cases require exactly one of fixed_timesteps or fixed_sigmas')
    if not noise_seeds or any(seed < 0 for seed in noise_seeds):
        raise ValueError('probe cases require non-negative noise seeds')
    by_id = {}
    for item in items:
        item_id = normalize_dataset_relative_item_id(item.get('dataset_relative_item_id', item.get('item_id', '')))
        if item_id in by_id:
            raise ValueError(f'duplicate calibration probe item: {item_id}')
        by_id[item_id] = dict(item)
    cases = []
    for split, ids in (
        ('train', split_manifest.get('train_item_ids', ())),
        ('heldout', split_manifest.get('heldout_item_ids', ())),
    ):
        selected_ids = [normalize_dataset_relative_item_id(value) for value in ids if normalize_dataset_relative_item_id(value) in by_id]
        if limit > 0:
            selected_ids = selected_ids[:limit]
        if not selected_ids:
            raise ValueError(f'calibration requires at least one {split} probe item')
        for item_id in selected_ids:
            item = by_id[item_id]
            caption = str(item.get('caption', ''))
            image_path = item.get('image_path')
            image_hash = sha256_file(image_path) if image_path and os.path.isfile(image_path) else fingerprint(item.get('image_identity', item_id))
            caption_hash = fingerprint(caption)
            transform = dict(item.get('transform', {}))
            schedule_values = fixed_timesteps or fixed_sigmas
            for schedule_index, schedule_value in enumerate(schedule_values):
                for noise_seed in noise_seeds:
                    case_payload = {
                        'split': split,
                        'item_id': item_id,
                        'caption_hash': caption_hash,
                        'image_hash': image_hash,
                        'noise_seed': int(noise_seed),
                        'schedule_index': schedule_index,
                        'schedule_value': schedule_value,
                        'target_mode': target_mode,
                        'transform': transform,
                    }
                    cases.append(FixedDiffusionProbeCase(
                        probe_case_id=fingerprint(case_payload),
                        split=split,
                        item_id=item_id,
                        caption_hash=caption_hash,
                        image_hash=image_hash,
                        noise_seed=int(noise_seed),
                        timestep=int(schedule_value) if fixed_timesteps else None,
                        sigma=float(schedule_value) if fixed_sigmas else None,
                        target_mode=target_mode,
                        transform=transform,
                    ))
    return cases


def helper_gain(prediction: torch.Tensor, neutral_prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1.0e-6) -> float:
    if prediction.shape != neutral_prediction.shape or prediction.shape != target.shape:
        raise ValueError('calibration predictions and target must have matching shapes')
    helper_loss = torch.mean((prediction.float() - target.float()).square())
    neutral_loss = torch.mean((neutral_prediction.float() - target.float()).square())
    value = 1.0 - helper_loss / (neutral_loss + epsilon)
    result = float(value.detach().cpu().item())
    if not math.isfinite(result):
        raise ValueError('calibration gain must be finite')
    return result


def summarize_gains(values: Sequence[float]) -> Dict[str, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError('gain summary requires finite values')
    sorted_values = sorted(float(value) for value in values)
    p10_index = max(0, math.ceil(0.1 * len(sorted_values)) - 1)
    p90_index = max(0, math.ceil(0.9 * len(sorted_values)) - 1)
    return {
        'count': len(sorted_values),
        'mean': statistics.fmean(sorted_values),
        'median': statistics.median(sorted_values),
        'p10': sorted_values[p10_index],
        'p90': sorted_values[p90_index],
        'std': statistics.pstdev(sorted_values) if len(sorted_values) > 1 else 0.0,
        'positive_fraction': sum(value > 0 for value in sorted_values) / len(sorted_values),
    }


def select_helpers(
    helper_records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    min_median_gain: float,
    min_positive_fraction: float,
    min_heldout_positive_fraction: float,
    min_p10_gain: float,
    max_train_heldout_gap: float,
    max_helpers: int,
    sampling_floor: float,
) -> Dict[str, Any]:
    candidates = {}
    eligible = []
    for phrase, records in helper_records.items():
        train = [float(record['gain']) for record in records if record['split'] == 'train']
        heldout = [float(record['gain']) for record in records if record['split'] == 'heldout']
        if not train or not heldout:
            raise ValueError(f'helper {phrase!r} requires train and heldout observations')
        train_stats = summarize_gains(train)
        heldout_stats = summarize_gains(heldout)
        combined_stats = summarize_gains(train + heldout)
        gap = abs(train_stats['mean'] - heldout_stats['mean'])
        is_eligible = (
            combined_stats['median'] >= min_median_gain
            and combined_stats['positive_fraction'] >= min_positive_fraction
            and heldout_stats['positive_fraction'] >= min_heldout_positive_fraction
            and combined_stats['p10'] >= min_p10_gain
            and gap <= max_train_heldout_gap
        )
        score = combined_stats['median'] + combined_stats['p10'] + heldout_stats['mean'] - gap
        candidates[phrase] = {
            'train': train_stats,
            'heldout': heldout_stats,
            'combined': combined_stats,
            'train_heldout_gap': gap,
            'eligible': is_eligible,
            'score': score,
        }
        if is_eligible:
            eligible.append((phrase, score))
    eligible.sort(key=lambda pair: (-pair[1], pair[0]))
    selected = eligible[:max_helpers]
    raw_weights = {phrase: max(score, 0.0) + sampling_floor for phrase, score in selected}
    total = sum(raw_weights.values())
    weights = {phrase: value / total for phrase, value in raw_weights.items()} if total else {}
    return {'candidates': candidates, 'selected_helpers': [phrase for phrase, _ in selected], 'sampling_weights': weights}


def run_calibration(
    cases: Sequence[FixedDiffusionProbeCase],
    helper_phrases: Sequence[str],
    *,
    neutral_phrase: str,
    prepare_case: Callable[[FixedDiffusionProbeCase], Mapping[str, torch.Tensor]],
    predict_phrase: Callable[[str, Mapping[str, torch.Tensor]], torch.Tensor],
    selection_kwargs: Mapping[str, Any],
    output_dir: str,
    identity: Mapping[str, Any],
    manifest_filename: str = 'semantic_scaffold_manifest.json',
    probe_manifest_filename: str = 'semantic_scaffold_probe_manifest.json',
) -> Dict[str, Any]:
    probe_payload = {
        'schema': PROBE_SCHEMA,
        'schema_version': PROBE_SCHEMA_VERSION,
        'identity': dict(identity),
        'cases': [case.as_dict() for case in cases],
    }
    probe_payload['probe_manifest_hash'] = fingerprint(probe_payload)
    os.makedirs(output_dir, exist_ok=True)
    _atomic_write_json(os.path.join(output_dir, probe_manifest_filename), probe_payload)
    records: Dict[str, List[Dict[str, Any]]] = {phrase: [] for phrase in helper_phrases}
    with torch.inference_mode(), isolated_rng(0):
        for case in cases:
            prepared = dict(prepare_case(case))
            required = {'target'}
            if not required.issubset(prepared):
                raise ValueError('prepared calibration case must contain target')
            neutral_prediction = predict_phrase(neutral_phrase, prepared).detach()
            for phrase in helper_phrases:
                prediction = predict_phrase(phrase, prepared).detach()
                records[phrase].append({
                    'probe_case_id': case.probe_case_id,
                    'split': case.split,
                    'gain': helper_gain(prediction, neutral_prediction, prepared['target']),
                })
    selection = select_helpers(records, **selection_kwargs)
    manifest = {
        'schema': CALIBRATION_SCHEMA,
        'schema_version': CALIBRATION_SCHEMA_VERSION,
        'identity': dict(identity),
        'probe_manifest_hash': probe_payload['probe_manifest_hash'],
        'neutral_phrase': neutral_phrase,
        'helper_records': records,
        **selection,
        'failure_reason': None,
    }
    if not manifest['selected_helpers']:
        manifest['failure_reason'] = 'semantic_scaffold_calibration_failed'
    manifest['manifest_hash'] = fingerprint(manifest)
    _atomic_write_json(os.path.join(output_dir, manifest_filename), manifest)
    if not manifest['selected_helpers']:
        raise SemanticScaffoldCalibrationError('semantic_scaffold_calibration_failed')
    return manifest
