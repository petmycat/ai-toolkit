"""Deterministic helper calibration for semantic scaffold A1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
import statistics
import tempfile
import time
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
    probe_scope: str = 'split',
    target_mode: str = 'flow',
) -> List[FixedDiffusionProbeCase]:
    if bool(fixed_timesteps) == bool(fixed_sigmas):
        raise ValueError('probe cases require exactly one of fixed_timesteps or fixed_sigmas')
    if probe_scope not in ('split', 'all'):
        raise ValueError('probe_scope must be split or all')
    if not noise_seeds or any(seed < 0 for seed in noise_seeds):
        raise ValueError('probe cases require non-negative noise seeds')
    by_id = {}
    for item in items:
        item_id = normalize_dataset_relative_item_id(item.get('dataset_relative_item_id', item.get('item_id', '')))
        if item_id in by_id:
            raise ValueError(f'duplicate calibration probe item: {item_id}')
        by_id[item_id] = dict(item)
    cases = []
    if probe_scope == 'all':
        split_groups = [('all', list(by_id))]
    else:
        split_groups = [
            ('train', split_manifest.get('train_item_ids', ())),
            ('heldout', split_manifest.get('heldout_item_ids', ())),
        ]
    for split, ids in split_groups:
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
    value = (neutral_loss - helper_loss) / (neutral_loss.abs() + epsilon)
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


def _pairwise_cosine(vectors: Sequence[torch.Tensor]) -> float:
    if len(vectors) < 2:
        return 0.0
    normalized = []
    for vector in vectors:
        vector = vector.float().reshape(-1)
        norm = torch.linalg.vector_norm(vector)
        if float(norm.item()) <= 1.0e-12:
            continue
        normalized.append(vector / norm)
    if len(normalized) < 2:
        return 0.0
    stacked = torch.stack(normalized)
    count = stacked.shape[0]
    resultant = stacked.sum(dim=0)
    value = (resultant.square().sum() - count) / (count * (count - 1))
    return float(value.detach().cpu().item())


def _conditioning_summary(
    records: Sequence[Mapping[str, Any]],
    conditioning_vectors: Optional[Sequence[torch.Tensor]] = None,
) -> Dict[str, float]:
    rms_values = [float(record['conditioning_relative_rms']) for record in records if 'conditioning_relative_rms' in record]
    if not rms_values:
        return {'count': 0, 'mean_relative_rms': 0.0, 'median_relative_rms': 0.0, 'pairwise_direction_consistency': 0.0}
    if conditioning_vectors:
        consistency = _pairwise_cosine(conditioning_vectors)
    else:
        consistency_values = [
            float(record['conditioning_direction_consistency'])
            for record in records
            if 'conditioning_direction_consistency' in record
        ]
        consistency = statistics.fmean(consistency_values) if consistency_values else 0.0
    return {
        'count': len(rms_values),
        'mean_relative_rms': statistics.fmean(rms_values),
        'median_relative_rms': statistics.median(rms_values),
        'pairwise_direction_consistency': consistency,
    }


def _record_stats(records: Sequence[Mapping[str, Any]], field: str) -> Dict[str, float]:
    values = [float(record[field]) for record in records if field in record]
    return summarize_gains(values) if values else {'count': 0, 'mean': 0.0, 'median': 0.0, 'p10': 0.0, 'p90': 0.0, 'std': 0.0, 'positive_fraction': 0.0}


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
    selection_mode: str = 'target_compatibility',
    min_conditioning_relative_rms: float = 0.0,
    min_conditioning_direction_consistency: float = -1.0,
    min_mean_gain: float = -1.0e9,
    max_mean_gain_regression: float = 1.0e9,
    min_pairwise_compatibility: float = -1.0,
    helper_conditioning_vectors: Optional[Mapping[str, Sequence[torch.Tensor]]] = None,
) -> Dict[str, Any]:
    candidates = {}
    eligible = []
    helper_vectors = {
        phrase: list((helper_conditioning_vectors or {}).get(phrase, ()))
        for phrase in helper_records
    }
    compatibility = {}
    for left_index, left in enumerate(helper_records):
        for right in list(helper_records)[left_index + 1:]:
            left_vectors = helper_vectors[left]
            right_vectors = helper_vectors[right]
            pair_values = []
            for left_vector, right_vector in zip(left_vectors, right_vectors):
                pair_values.append(_pairwise_cosine([left_vector, right_vector]))
            compatibility[f'{left}||{right}'] = statistics.fmean(pair_values) if pair_values else 0.0
    for phrase, records in helper_records.items():
        all_scope = all(record['split'] == 'all' for record in records)
        train = [float(record['gain']) for record in records if record['split'] == 'train']
        heldout = [float(record['gain']) for record in records if record['split'] == 'heldout']
        combined_values = [float(record['gain']) for record in records]
        if not combined_values:
            raise ValueError(f'helper {phrase!r} requires calibration observations')
        combined_stats = summarize_gains(combined_values)
        train_stats = summarize_gains(train) if train else None
        heldout_stats = summarize_gains(heldout) if heldout else None
        gap = abs(train_stats['mean'] - heldout_stats['mean']) if train_stats and heldout_stats else None
        conditioning = _conditioning_summary(records, helper_vectors[phrase])
        mean_gain = combined_stats['mean']
        conditioning_ok = (
            conditioning['median_relative_rms'] >= min_conditioning_relative_rms
            and conditioning['pairwise_direction_consistency'] >= min_conditioning_direction_consistency
        )
        target_not_harmful = (
            mean_gain >= min_mean_gain
            and mean_gain >= -max_mean_gain_regression
        )
        target_gate = (
            combined_stats['median'] >= min_median_gain
            and combined_stats['positive_fraction'] >= min_positive_fraction
            and train_stats is not None
            and heldout_stats is not None
            and heldout_stats['positive_fraction'] >= min_heldout_positive_fraction
            and combined_stats['p10'] >= min_p10_gain
            and gap <= max_train_heldout_gap
        )
        if selection_mode == 'conditioning_space':
            is_eligible = conditioning_ok and target_not_harmful
            score = conditioning['median_relative_rms'] * max(conditioning['pairwise_direction_consistency'], 0.0) + mean_gain
        else:
            is_eligible = target_gate
            score = combined_stats['median'] + combined_stats['p10'] + heldout_stats['mean'] - gap
        candidates[phrase] = {
            'train': train_stats,
            'heldout': heldout_stats,
            'combined': combined_stats,
            'split_statistics_valid': not all_scope,
            'conditioning': conditioning,
            'train_heldout_gap': gap,
            'pairwise_compatibility': {
                key: value for key, value in compatibility.items()
                if phrase in key
            },
            'target_gate': target_gate,
            'conditioning_gate': conditioning_ok,
            'target_not_harmful': target_not_harmful,
            'eligible': is_eligible,
            'score': score,
        }
        if is_eligible:
            eligible.append((phrase, score))
    eligible.sort(key=lambda pair: (-pair[1], pair[0]))
    selected = []
    for phrase, score in eligible:
        if len(selected) >= max_helpers:
            break
        if selected:
            pair_values = [
                compatibility.get(f'{other}||{phrase}', compatibility.get(f'{phrase}||{other}', 0.0))
                for other in selected
            ]
            if min(pair_values) < min_pairwise_compatibility:
                continue
        selected.append((phrase, score))
    raw_weights = {phrase: max(score, 0.0) + sampling_floor for phrase, score in selected}
    total = sum(raw_weights.values())
    weights = {phrase: value / total for phrase, value in raw_weights.items()} if total else {}
    return {
        'candidates': candidates,
        'compatibility_matrix': compatibility,
        'selected_helpers': [phrase for phrase, _ in selected],
        'sampling_weights': weights,
    }


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
    progress_filename: str = 'semantic_scaffold_calibration_progress.json',
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
    conditioning_vectors: Dict[str, List[torch.Tensor]] = {phrase: [] for phrase in helper_phrases}
    total_predictions = len(cases) * (1 + len(helper_phrases))
    progress_path = os.path.join(output_dir, progress_filename)

    def write_progress(status: str, **extra: Any) -> None:
        payload = {
            'schema': 'ai-toolkit.semantic-scaffold-calibration-progress',
            'schema_version': 1,
            'status': status,
            'case_count': len(cases),
            'helper_count': len(helper_phrases),
            'total_predictions': total_predictions,
            'completed_predictions': completed_predictions,
            'elapsed_seconds': time.perf_counter() - started_at,
            'records_completed': {phrase: len(values) for phrase, values in records.items()},
            **extra,
        }
        _atomic_write_json(progress_path, payload)

    completed_predictions = 0
    started_at = time.perf_counter()
    write_progress('started')
    print(f'  - Semantic scaffold calibration: {len(cases)} cases, {len(helper_phrases)} helpers, {total_predictions} predictions', flush=True)
    with torch.inference_mode(), isolated_rng(0):
        for case_index, case in enumerate(cases, start=1):
            case_started_at = time.perf_counter()
            print(f'  - Calibration case {case_index}/{len(cases)} [{case.split}] {case.item_id}', flush=True)
            prepared = dict(prepare_case(case))
            write_progress('case_prepared', case_index=case_index, probe_case_id=case.probe_case_id)
            required = {'target'}
            if not required.issubset(prepared):
                raise ValueError('prepared calibration case must contain target')
            template = prepared.get('prompt_template')
            placeholder = prepared.get('placeholder')
            if template is not None and placeholder is not None:
                if placeholder not in template:
                    raise ValueError(f'calibration prompt lacks placeholder {placeholder!r}: {str(template)[:200]!r}')
                print(
                    f'    prompt variants: neutral={template.replace(placeholder, neutral_phrase)[:160]!r}',
                    flush=True,
                )
                for phrase in helper_phrases:
                    print(f'      helper {phrase!r}: {template.replace(placeholder, phrase)[:160]!r}', flush=True)
            neutral_result = predict_phrase(neutral_phrase, prepared)
            neutral_prediction = neutral_result[0] if isinstance(neutral_result, tuple) else neutral_result
            neutral_conditioning = neutral_result[1] if isinstance(neutral_result, tuple) and len(neutral_result) > 1 else None
            neutral_prediction = neutral_prediction.detach()
            if neutral_conditioning is not None:
                if not isinstance(neutral_conditioning, torch.Tensor):
                    raise ValueError('neutral conditioning effect must be a tensor')
                neutral_conditioning = neutral_conditioning.detach().float().reshape(-1)
            completed_predictions += 1
            print(f'    neutral complete ({completed_predictions}/{total_predictions})', flush=True)
            write_progress('neutral_complete', case_index=case_index, probe_case_id=case.probe_case_id)
            for phrase in helper_phrases:
                helper_result = predict_phrase(phrase, prepared)
                if isinstance(helper_result, tuple):
                    prediction, helper_conditioning = helper_result
                else:
                    prediction, helper_conditioning = helper_result, None
                prediction = prediction.detach()
                completed_predictions += 1
                gain = helper_gain(prediction, neutral_prediction, prepared['target'])
                conditioning_relative_rms = 0.0
                conditioning_direction_consistency = 0.0
                if helper_conditioning is not None:
                    if not isinstance(helper_conditioning, torch.Tensor):
                        raise ValueError('helper conditioning effect must be a tensor')
                    helper_conditioning = helper_conditioning.detach().float().reshape(-1)
                    if neutral_conditioning is not None:
                        helper_conditioning = helper_conditioning - neutral_conditioning
                    conditioning_vectors[phrase].append(helper_conditioning)
                    conditioning_relative_rms = float(helper_conditioning.square().mean().sqrt().cpu().item())
                    if helper_conditioning.numel() and conditioning_vectors[phrase][:-1]:
                        prior = torch.stack(conditioning_vectors[phrase][:-1])
                        prototype = prior.mean(dim=0)
                        conditioning_direction_consistency = float(
                            torch.nn.functional.cosine_similarity(
                                helper_conditioning.unsqueeze(0), prototype.unsqueeze(0), dim=1
                            ).item()
                        )
                prediction_delta_rms = float(
                    (prediction.float() - neutral_prediction.float()).square().mean().sqrt().detach().cpu().item()
                )
                records[phrase].append({
                    'probe_case_id': case.probe_case_id,
                    'split': case.split,
                    'gain': gain,
                    'prediction_delta_rms': prediction_delta_rms,
                    'conditioning_relative_rms': conditioning_relative_rms,
                    'conditioning_direction_consistency': conditioning_direction_consistency,
                    '_conditioning_vector': helper_conditioning.cpu() if helper_conditioning is not None else None,
                })
                records[phrase][-1].pop('_conditioning_vector', None)
                print(
                    f'    helper {phrase!r} complete ({completed_predictions}/{total_predictions}), '
                    f'gain={gain:.9f}, prediction_delta_rms={prediction_delta_rms:.9e}',
                    flush=True,
                )
                write_progress(
                    'prediction_complete',
                    case_index=case_index,
                    probe_case_id=case.probe_case_id,
                    helper=phrase,
                    gain=gain,
                    prediction_delta_rms=prediction_delta_rms,
                    conditioning_relative_rms=conditioning_relative_rms,
                    conditioning_direction_consistency=conditioning_direction_consistency,
                )
            elapsed = time.perf_counter() - case_started_at
            print(f'    case complete in {elapsed:.1f}s', flush=True)
    print(f'  - Semantic scaffold calibration complete in {time.perf_counter() - started_at:.1f}s', flush=True)
    selection = select_helpers(
        records,
        helper_conditioning_vectors=conditioning_vectors,
        **selection_kwargs,
    )
    print('  - Semantic scaffold helper selection:', flush=True)
    for phrase, candidate in selection['candidates'].items():
        combined = candidate['combined']
        heldout = candidate['heldout']
        conditioning = candidate['conditioning']
        heldout_fraction = None if heldout is None else heldout['positive_fraction']
        print(
            f'    {phrase!r}: eligible={candidate["eligible"]}, '
            f'median={combined["median"]:.9f}, positive_fraction={combined["positive_fraction"]:.3f}, '
            f'heldout_positive_fraction={heldout_fraction}, p10={combined["p10"]:.9f}, '
            f'conditioning_rms={conditioning["median_relative_rms"]:.9e}, '
            f'conditioning_consistency={conditioning["pairwise_direction_consistency"]:.3f}, '
            f'train_heldout_gap={candidate["train_heldout_gap"]}, score={candidate["score"]:.9f}',
            flush=True,
        )
    if selection['selected_helpers']:
        print(f'    selected={selection["selected_helpers"]}', flush=True)
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
    write_progress(
        'complete' if manifest['selected_helpers'] else 'failed',
        selected_helpers=manifest['selected_helpers'],
        candidates=manifest['candidates'],
        failure_reason=manifest['failure_reason'],
    )
    if not manifest['selected_helpers']:
        raise SemanticScaffoldCalibrationError('semantic_scaffold_calibration_failed')
    return manifest
