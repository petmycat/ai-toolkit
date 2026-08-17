import json
import math
import os
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from toolkit.config_modules import TriggerDataSplitConfig, TriggerValidationConfig


PredictionOrLoss = Union[torch.Tensor, float, int]
EvaluationCallable = Callable[[], PredictionOrLoss]


@dataclass(frozen=True)
class TriggerValidationResult:
    trigger_gain: float
    decoy_gain: float
    raw_gap: float
    effective_gap: float
    base_trigger_loss: float
    student_trigger_loss: float
    base_decoy_loss: float
    student_decoy_loss: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _validate_filename(value: str, field_name: str):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field_name} must be a non-empty filename')
    if os.path.basename(value) != value or value in {'.', '..'}:
        raise ValueError(f'{field_name} must be a filename, not a path')


def _validate_manifest_path(value: Optional[str], field_name: str, require_exists: bool):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field_name} must be a non-empty path when validation is enabled')
    normalized = os.path.abspath(os.path.expanduser(value))
    if require_exists and not os.path.isfile(normalized):
        raise ValueError(f'{field_name} does not exist or is not a file: {value}')


def validate_trigger_data_split_config(
    config: 'TriggerDataSplitConfig',
    *,
    require_manifest_file: bool = False,
):
    if not isinstance(config.enabled, bool):
        raise ValueError('three_phase_trigger_training.data_split.enabled must be boolean')
    if not config.enabled:
        return
    if config.seed < 0:
        raise ValueError('three_phase_trigger_training.data_split.seed must be non-negative')
    if not math.isfinite(config.heldout_fraction) or not 0.0 < config.heldout_fraction < 1.0:
        raise ValueError(
            'three_phase_trigger_training.data_split.heldout_fraction must be strictly between 0 and 1'
        )
    if not isinstance(config.reuse_existing, bool):
        raise ValueError('three_phase_trigger_training.data_split.reuse_existing must be boolean')
    _validate_manifest_path(
        config.manifest_path,
        'three_phase_trigger_training.data_split.manifest_path',
        require_manifest_file,
    )


def validate_trigger_validation_config(
    config: 'TriggerValidationConfig',
    *,
    require_manifest_files: bool = True,
    data_split_config: Optional['TriggerDataSplitConfig'] = None,
):
    if not isinstance(config.enabled, bool):
        raise ValueError('three_phase_trigger_training.validation.enabled must be boolean')
    if not config.enabled:
        return
    if config.every <= 0 and not config.steps:
        if config.every == 0:
            raise ValueError('three_phase_trigger_training.validation.every must be positive')
        raise ValueError('three_phase_trigger_training.validation.every or steps must be configured')
    if any(step < 0 for step in config.steps):
        raise ValueError('validation.steps must be non-negative')
    if len(set(config.steps)) != len(config.steps):
        raise ValueError('validation.steps must be unique')
    if config.seed < 0:
        raise ValueError('three_phase_trigger_training.validation.seed must be non-negative')
    if bool(config.fixed_timesteps) == bool(config.fixed_sigmas):
        raise ValueError('validation must configure exactly one of fixed_timesteps or fixed_sigmas')
    if any(value < 0 for value in config.fixed_timesteps):
        raise ValueError('validation.fixed_timesteps must be non-negative')
    if any(not math.isfinite(value) or value < 0 for value in config.fixed_sigmas):
        raise ValueError('validation.fixed_sigmas must be finite and non-negative')
    if len(set(config.fixed_timesteps)) != len(config.fixed_timesteps):
        raise ValueError('validation.fixed_timesteps must be unique')
    if len(set(config.fixed_sigmas)) != len(config.fixed_sigmas):
        raise ValueError('validation.fixed_sigmas must be unique')

    split_manifest = getattr(config, 'data_split_manifest', None)
    managed_split_enabled = data_split_config is not None and data_split_config.enabled
    if managed_split_enabled:
        configured_manifest = split_manifest or data_split_config.manifest_path
        _validate_manifest_path(
            configured_manifest,
            'three_phase_trigger_training.data_split.manifest_path',
            False,
        )
        if config.train_probe_manifest is not None or config.heldout_manifest is not None:
            raise ValueError(
                'managed data_split cannot be combined with legacy train-probe/held-out manifests'
            )
    elif split_manifest is not None:
        _validate_manifest_path(
            split_manifest,
            'validation.data_split_manifest',
            require_manifest_files,
        )
        if config.train_probe_manifest is not None or config.heldout_manifest is not None:
            raise ValueError(
                'validation.data_split_manifest cannot be combined with legacy train-probe/held-out manifests'
            )
    else:
        _validate_manifest_path(
            config.train_probe_manifest,
            'validation.train_probe_manifest',
            require_manifest_files,
        )
        _validate_manifest_path(
            config.heldout_manifest,
            'validation.heldout_manifest',
            require_manifest_files,
        )
        if os.path.abspath(os.path.expanduser(config.train_probe_manifest)) == os.path.abspath(
            os.path.expanduser(config.heldout_manifest)
        ):
            raise ValueError('validation train-probe and held-out manifests must be different files')

    if not config.caption_sources or any(
        not isinstance(source, str) or not source.strip() for source in config.caption_sources
    ):
        raise ValueError('validation.caption_sources must contain non-empty strings')
    if len(set(config.caption_sources)) != len(config.caption_sources):
        raise ValueError('validation.caption_sources must be unique')
    if not config.negative_phrases or any(
        not isinstance(phrase, str) for phrase in config.negative_phrases
    ):
        raise ValueError('validation.negative_phrases must contain strings')
    if not math.isfinite(config.gain_epsilon) or config.gain_epsilon <= 0:
        raise ValueError('validation.gain_epsilon must be positive')

    filenames = (
        config.train_probe_output_filename,
        config.heldout_output_filename,
        config.aggregate_output_filename,
    )
    for field_name, filename in zip(
        (
            'validation.train_probe_output_filename',
            'validation.heldout_output_filename',
            'validation.aggregate_output_filename',
        ),
        filenames,
    ):
        _validate_filename(filename, field_name)
    if len(set(filenames)) != len(filenames):
        raise ValueError('validation output filenames must be unique')


@contextmanager
def isolated_rng(seed: Optional[int] = None):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def make_python_rng(seed: int) -> random.Random:
    return random.Random(seed)


def make_torch_generator(seed: int, device: Union[str, torch.device] = 'cpu') -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _to_loss(
    value: PredictionOrLoss,
    target: Optional[torch.Tensor],
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], PredictionOrLoss]],
    name: str,
) -> float:
    if target is not None:
        if not isinstance(value, torch.Tensor):
            raise ValueError(f'{name} prediction must be a tensor when target is provided')
        if value.shape != target.shape:
            raise ValueError(f'{name} prediction and target shapes must match')
        computed = loss_fn(value, target) if loss_fn is not None else F.mse_loss(
            value.float(), target.float(), reduction='mean'
        )
    else:
        computed = value
    tensor = torch.as_tensor(computed).detach().float().cpu()
    if tensor.numel() != 1:
        raise ValueError(f'{name} must produce a scalar loss for one item')
    result = float(tensor.item())
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'{name} loss must be finite and non-negative')
    return result


def evaluate_gain(
    base_trigger: EvaluationCallable,
    student_trigger: EvaluationCallable,
    base_decoy: EvaluationCallable,
    student_decoy: EvaluationCallable,
    *,
    target: Optional[torch.Tensor] = None,
    trigger_target: Optional[torch.Tensor] = None,
    decoy_target: Optional[torch.Tensor] = None,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], PredictionOrLoss]] = None,
    epsilon: float = 1.0e-6,
) -> TriggerValidationResult:
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError('epsilon must be positive')
    if target is not None and (trigger_target is not None or decoy_target is not None):
        raise ValueError('target cannot be combined with branch-specific targets')
    effective_trigger_target = target if trigger_target is None else trigger_target
    effective_decoy_target = target if decoy_target is None else decoy_target

    with torch.no_grad():
        base_trigger_loss = _to_loss(base_trigger(), effective_trigger_target, loss_fn, 'base_trigger')
        student_trigger_loss = _to_loss(student_trigger(), effective_trigger_target, loss_fn, 'student_trigger')
        base_decoy_loss = _to_loss(base_decoy(), effective_decoy_target, loss_fn, 'base_decoy')
        student_decoy_loss = _to_loss(student_decoy(), effective_decoy_target, loss_fn, 'student_decoy')

    trigger_gain = 1.0 - student_trigger_loss / (base_trigger_loss + epsilon)
    decoy_gain = 1.0 - student_decoy_loss / (base_decoy_loss + epsilon)
    return TriggerValidationResult(
        trigger_gain=trigger_gain,
        decoy_gain=decoy_gain,
        raw_gap=trigger_gain - decoy_gain,
        effective_gap=trigger_gain - max(decoy_gain, 0.0),
        base_trigger_loss=base_trigger_loss,
        student_trigger_loss=student_trigger_loss,
        base_decoy_loss=base_decoy_loss,
        student_decoy_loss=student_decoy_loss,
    )


def aggregate_results(records: Iterable[Union[TriggerValidationResult, Mapping[str, Any]]]) -> Dict[str, Any]:
    normalized = [record.as_dict() if isinstance(record, TriggerValidationResult) else dict(record) for record in records]
    if not normalized:
        raise ValueError('cannot aggregate an empty validation result collection')
    metrics = (
        'trigger_gain',
        'decoy_gain',
        'raw_gap',
        'effective_gap',
        'base_trigger_loss',
        'student_trigger_loss',
        'base_decoy_loss',
        'student_decoy_loss',
    )
    aggregate: Dict[str, Any] = {'count': len(normalized)}
    for metric in metrics:
        values = [float(record[metric]) for record in normalized]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f'cannot aggregate non-finite {metric}')
        aggregate[metric] = sum(values) / len(values)
    aggregate['trigger_gain_positive_rate'] = sum(
        float(record['trigger_gain']) > 0.0 for record in normalized
    ) / len(normalized)
    aggregate['effective_gap_positive_rate'] = sum(
        float(record['effective_gap']) > 0.0 for record in normalized
    ) / len(normalized)
    return aggregate


def should_run_validation(completed_updates: int, config: 'TriggerValidationConfig') -> bool:
    if not isinstance(completed_updates, int) or completed_updates < 0:
        raise ValueError('completed_updates must be a non-negative integer')
    steps = tuple(getattr(config, 'steps', ()) or ())
    if steps:
        return completed_updates in steps
    every = int(getattr(config, 'every', 0))
    return every > 0 and completed_updates % every == 0


def assert_probe_split_disjoint(train_item_ids: Iterable[str], heldout_item_ids: Iterable[str]) -> None:
    train = {str(item).replace('\\', '/') for item in train_item_ids}
    heldout = {str(item).replace('\\', '/') for item in heldout_item_ids}
    overlap = sorted(train & heldout)
    if overlap:
        raise ValueError(f'train/held-out probe leakage detected: {overlap}')


def build_fixed_probe_sets(
    items: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    seed: int,
    limit: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    if seed < 0:
        raise ValueError('probe seed must be non-negative')
    train_ids = {str(value).replace('\\', '/') for value in manifest.get('train_item_ids', [])}
    heldout_ids = {str(value).replace('\\', '/') for value in manifest.get('heldout_item_ids', [])}
    assert_probe_split_disjoint(train_ids, heldout_ids)
    by_id = {}
    for item in items:
        item_id = str(item.get('dataset_relative_item_id', item.get('item_id', ''))).replace('\\', '/')
        if not item_id:
            raise ValueError('probe item is missing dataset_relative_item_id')
        if item_id in by_id:
            raise ValueError(f'duplicate probe item ID: {item_id}')
        by_id[item_id] = dict(item)
    rng = random.Random(seed)
    result = {}
    for split, allowed in (('train', train_ids), ('heldout', heldout_ids)):
        selected = [by_id[item_id] for item_id in sorted(allowed) if item_id in by_id]
        rng.shuffle(selected)
        if limit is not None:
            if limit <= 0:
                raise ValueError('probe limit must be positive')
            selected = selected[:limit]
        for index, item in enumerate(selected):
            item['probe_index'] = index
            item['probe_split'] = split
        result[split] = selected
    if not result['train'] or not result['heldout']:
        raise ValueError('fixed probes require at least one item in each split')
    return result


def write_fixed_probe_results(
    output_dir: str,
    config: Any,
    *,
    step: int,
    probe_sets: Mapping[str, Iterable[Mapping[str, Any]]],
    evaluate: Callable[[Mapping[str, Any], str, int], Mapping[str, Any]],
) -> Dict[str, Any]:
    if step < 0:
        raise ValueError('validation step must be non-negative')
    aggregates = {}
    for split in ('train', 'heldout'):
        filename_field = 'train_probe_output_filename' if split == 'train' else 'heldout_output_filename'
        writer = IdempotentJSONLWriter(
            output_dir,
            getattr(config, filename_field),
            key_fields=('step', 'probe_split', 'dataset_relative_item_id', 'probe_index'),
        )
        records = []
        for item in probe_sets.get(split, []):
            record = dict(evaluate(item, split, step))
            record.update({'step': step, 'probe_split': split, 'dataset_relative_item_id': item['dataset_relative_item_id'], 'probe_index': item.get('probe_index')})
            writer.write(record)
            records.append(record)
        if not records:
            raise ValueError(f'no fixed probes available for split {split}')
        if all(SEMANTIC_METRIC_ALLOWLIST.issubset(record) for record in records):
            aggregates[split] = {'step': step, 'probe_split': split, **aggregate_semantic_metrics(records)}
        else:
            aggregates[split] = {'step': step, 'probe_split': split, 'count': len(records)}
            for key in records[0]:
                values = [float(record[key]) for record in records if isinstance(record.get(key), (int, float)) and not isinstance(record[key], bool)]
                if values and all(math.isfinite(value) for value in values):
                    aggregates[split][key] = sum(values) / len(values)
    JSONLWriter(output_dir, config.aggregate_output_filename).write({'step': step, 'probe_splits': aggregates})
    return aggregates


SEMANTIC_VALIDATION_SCHEMA = 'ai-toolkit.semantic-scaffold-validation'
SEMANTIC_VALIDATION_SCHEMA_VERSION = 1
SEMANTIC_METRIC_ALLOWLIST = frozenset({
    'gain_active', 'gain_full', 'gain_semantic_only', 'gain_helper',
    'active_helper_gap', 'tap_gain_delta', 'target_mse',
    'prediction_delta', 'tap_prediction_delta', 'tap_relative_rms',
    'disturbance_beta', 'semantic_cosine',
})


def semantic_prediction_metrics(
    *,
    neutral_prediction: torch.Tensor,
    helper_prediction: torch.Tensor,
    bypass_prediction: torch.Tensor,
    semantic_prediction: torch.Tensor,
    full_prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1.0e-6,
) -> Dict[str, float]:
    tensors = (neutral_prediction, helper_prediction, bypass_prediction, semantic_prediction, full_prediction, target)
    if any(tensor.shape != target.shape for tensor in tensors):
        raise ValueError('semantic validation predictions and target must have matching shapes')
    def mse(value):
        return float(F.mse_loss(value.float(), target.float()).detach().cpu().item())
    neutral_mse = mse(neutral_prediction)
    helper_mse = mse(helper_prediction)
    semantic_mse = mse(semantic_prediction)
    full_mse = mse(full_prediction)
    gain_full = 1.0 - full_mse / (neutral_mse + epsilon)
    gain_semantic_only = 1.0 - semantic_mse / (neutral_mse + epsilon)
    gain_helper = 1.0 - helper_mse / (neutral_mse + epsilon)
    semantic_delta = (semantic_prediction.float() - bypass_prediction.float()).flatten()
    full_delta = (full_prediction.float() - bypass_prediction.float()).flatten()
    tap_delta = (full_prediction.float() - semantic_prediction.float()).flatten()
    target_delta = (target.float() - bypass_prediction.float()).flatten()
    semantic_rms = torch.sqrt(semantic_delta.square().mean() + epsilon)
    tap_rms = torch.sqrt(tap_delta.square().mean() + epsilon)
    return {
        'gain_active': gain_full,
        'gain_full': gain_full,
        'gain_semantic_only': gain_semantic_only,
        'gain_helper': gain_helper,
        'active_helper_gap': gain_full - gain_helper,
        'tap_gain_delta': gain_full - gain_semantic_only,
        'target_mse': full_mse,
        'prediction_delta': float(torch.linalg.vector_norm(full_delta).detach().cpu().item()),
        'tap_prediction_delta': float(torch.linalg.vector_norm(tap_delta).detach().cpu().item()),
        'tap_relative_rms': float((tap_rms / semantic_rms).detach().cpu().item()),
        'disturbance_beta': float((full_delta.square().mean() / (target_delta.square().mean() + epsilon)).detach().cpu().item()),
        'semantic_cosine': float(F.cosine_similarity(full_delta[None], semantic_delta[None], dim=1, eps=epsilon).detach().cpu().item()),
    }


def aggregate_semantic_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError('semantic metric aggregation requires records')
    aggregate = {'count': len(records)}
    for metric in sorted(SEMANTIC_METRIC_ALLOWLIST):
        values = [float(record[metric]) for record in records]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f'non-finite semantic validation metric: {metric}')
        sorted_values = sorted(values)
        aggregate[metric] = {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'p10': float(np.percentile(values, 10)),
            'p90': float(np.percentile(values, 90)),
            'std': float(np.std(values)),
            'positive_fraction': sum(value > 0 for value in values) / len(values),
        }
    return aggregate


class IdempotentJSONLWriter:
    def __init__(self, output_dir: str, filename: str, *, key_fields: Tuple[str, ...]):
        _validate_filename(filename, 'filename')
        self.path = os.path.join(output_dir, filename)
        self.key_fields = tuple(key_fields)

    def write(self, record: Mapping[str, Any]) -> bool:
        payload = dict(record)
        key = tuple(payload[field] for field in self.key_fields)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        existing = set()
        if os.path.isfile(self.path):
            with open(self.path, 'r', encoding='utf-8') as handle:
                for line in handle:
                    current = json.loads(line)
                    existing.add(tuple(current[field] for field in self.key_fields))
        if key in existing:
            return False
        with open(self.path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
        return True


class JSONLWriter:
    def __init__(self, output_dir: str, filename: str):
        _validate_filename(filename, 'filename')
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError('output_dir must be a non-empty path')
        self.path = os.path.join(output_dir, filename)

    def write(self, record: Union[TriggerValidationResult, Mapping[str, Any]]):
        payload = record.as_dict() if isinstance(record, TriggerValidationResult) else dict(record)
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + '\n')
