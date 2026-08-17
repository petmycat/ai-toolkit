"""Non-gradient curriculum state for the V8 A1 semantic scaffold."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import torch
from safetensors.torch import load_file, save_file


@dataclass
class HelperLatchState:
    latched: bool = False
    streak: int = 0
    latch_step: Optional[int] = None
    last_gain: Optional[float] = None

    def advance(self, gain: float, *, margin: float, patience: int, step: int) -> bool:
        if self.latched:
            return False
        self.last_gain = float(gain)
        if gain <= margin:
            self.streak = 0
            return False
        self.streak += 1
        if self.streak >= patience:
            self.latched = True
            self.latch_step = int(step)
            return True
        return False


@dataclass
class TapCurriculumState:
    stage: str = 'a1.0'
    evidence_streak: int = 0
    unlock_step: Optional[int] = None
    ramp_progress: float = 0.0
    failure: bool = False
    failure_reason: Optional[str] = None
    forced_handoff: bool = False
    handoff_reason: Optional[str] = None
    unmet_conditions: tuple[str, ...] = ()

    def observe_maturity(
        self,
        *,
        step: int,
        semantic_gain: float,
        helper_cosine: float,
        prototype_loss: float,
        gain_drift: float,
        minimum_observations: int,
        min_step: int,
        min_semantic_gain: float,
        min_helper_cosine: float,
        max_helper_cosine: float,
        max_prototype_loss: float,
        max_gain_drift: float,
        required_observations: int,
        patience: int,
        max_wait_step: int,
    ) -> str:
        if self.unlock_step is not None:
            return 'unlocked'
        conditions = {
            'min_step': step >= min_step,
            'semantic_gain': semantic_gain >= min_semantic_gain,
            'helper_cosine_low': helper_cosine >= min_helper_cosine,
            'helper_cosine_high': helper_cosine <= max_helper_cosine,
            'prototype_loss': prototype_loss <= max_prototype_loss,
            'gain_drift': gain_drift <= max_gain_drift,
            'prototype_observations': minimum_observations >= required_observations,
        }
        self.unmet_conditions = tuple(name for name, passed in conditions.items() if not passed)
        self.evidence_streak = self.evidence_streak + 1 if not self.unmet_conditions else 0
        if self.evidence_streak >= patience:
            self.stage = 'handoff_pending'
            self.unlock_step = int(step) + 1
            self.ramp_progress = 0.0
            self.handoff_reason = 'semantic_maturity_gate_passed'
            return 'handoff_pending'
        if step >= max_wait_step:
            self.stage = 'handoff_pending'
            self.unlock_step = int(step) + 1
            self.ramp_progress = 0.0
            self.forced_handoff = True
            self.handoff_reason = 'semantic_maturity_max_wait_reached'
            return 'forced_handoff_pending'
        return 'waiting'

    def advance_ramp(self, *, step: int, ramp_steps: int) -> float:
        if self.unlock_step is None:
            self.ramp_progress = 0.0
            return self.ramp_progress
        if ramp_steps <= 0:
            self.ramp_progress = 1.0
        else:
            self.ramp_progress = min(max((step - self.unlock_step) / float(ramp_steps), 0.0), 1.0)
        if self.ramp_progress >= 1.0:
            self.stage = 'a1.1'
        return self.ramp_progress


@dataclass
class SemanticScaffoldState:
    tap_layers: tuple[int, ...]
    semantic_prototypes: Dict[str, torch.Tensor] = field(default_factory=dict)
    tap_prototypes: Dict[str, torch.Tensor] = field(default_factory=dict)
    semantic_observation_counts: Dict[str, int] = field(default_factory=dict)
    tap_observation_counts: Dict[str, int] = field(default_factory=dict)
    gain_ema: Optional[float] = None
    semantic_helper_cosine_ema: Optional[float] = None
    semantic_prototype_loss_ema: Optional[float] = None
    semantic_gain_drift_ema: Optional[float] = None
    helper_gain_ema: Dict[str, float] = field(default_factory=dict)
    helper_latch: HelperLatchState = field(default_factory=HelperLatchState)
    curriculum: TapCurriculumState = field(default_factory=TapCurriculumState)
    last_update_step: int = 0
    schema_version: int = 1

    def __post_init__(self) -> None:
        self.tap_layers = tuple(int(layer) for layer in self.tap_layers)
        if len(self.tap_layers) != len(set(self.tap_layers)):
            raise ValueError('tap_layers must be unique')
        for layer in self.tap_layers:
            key = str(layer)
            self.semantic_observation_counts.setdefault(key, 0)
            self.tap_observation_counts.setdefault(key, 0)

    def update_ema(self, name: str, value: torch.Tensor, *, decay: float, kind: str) -> torch.Tensor:
        if not 0.0 <= decay < 1.0:
            raise ValueError('EMA decay must be in [0, 1)')
        if value.ndim < 1 or not bool(torch.isfinite(value).all().item()):
            raise ValueError('EMA observation must be finite and non-empty')
        target = self.semantic_prototypes if kind == 'semantic' else self.tap_prototypes
        counts = self.semantic_observation_counts if kind == 'semantic' else self.tap_observation_counts
        detached = value.detach().clone()
        if name not in target:
            target[name] = detached
        else:
            if target[name].shape != detached.shape:
                raise ValueError(f'{kind} prototype shape changed for {name}')
            target[name] = decay * target[name].to(detached) + (1.0 - decay) * detached
        counts[name] = counts.get(name, 0) + 1
        return target[name].detach().clone()

    def snapshot(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            'semantic': {key: value.detach().clone() for key, value in self.semantic_prototypes.items()},
            'tap': {key: value.detach().clone() for key, value in self.tap_prototypes.items()},
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            'schema_version': self.schema_version,
            'tap_layers': list(self.tap_layers),
            'semantic_prototypes': {key: value.detach().cpu() for key, value in self.semantic_prototypes.items()},
            'tap_prototypes': {key: value.detach().cpu() for key, value in self.tap_prototypes.items()},
            'semantic_observation_counts': dict(self.semantic_observation_counts),
            'tap_observation_counts': dict(self.tap_observation_counts),
            'gain_ema': self.gain_ema,
            'semantic_helper_cosine_ema': self.semantic_helper_cosine_ema,
            'semantic_prototype_loss_ema': self.semantic_prototype_loss_ema,
            'semantic_gain_drift_ema': self.semantic_gain_drift_ema,
            'helper_gain_ema': dict(self.helper_gain_ema),
            'helper_latch': asdict(self.helper_latch),
            'curriculum': asdict(self.curriculum),
            'last_update_step': self.last_update_step,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object], *, device=None) -> 'SemanticScaffoldState':
        layers = tuple(int(value) for value in state['tap_layers'])
        result = cls(tap_layers=layers)
        result.schema_version = int(state.get('schema_version', 1))
        for field_name in ('semantic_prototypes', 'tap_prototypes'):
            target = getattr(result, field_name)
            for key, value in dict(state.get(field_name, {})).items():
                tensor = value.detach().clone() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                target[str(key)] = tensor.to(device=device) if device is not None else tensor
        result.semantic_observation_counts.update({str(k): int(v) for k, v in dict(state.get('semantic_observation_counts', {})).items()})
        result.tap_observation_counts.update({str(k): int(v) for k, v in dict(state.get('tap_observation_counts', {})).items()})
        result.gain_ema = state.get('gain_ema')
        result.semantic_helper_cosine_ema = state.get('semantic_helper_cosine_ema')
        result.semantic_prototype_loss_ema = state.get('semantic_prototype_loss_ema')
        result.semantic_gain_drift_ema = state.get('semantic_gain_drift_ema')
        result.helper_gain_ema = {str(k): float(v) for k, v in dict(state.get('helper_gain_ema', {})).items()}
        result.helper_latch = HelperLatchState(**dict(state.get('helper_latch', {})))
        result.curriculum = TapCurriculumState(**dict(state.get('curriculum', {})))
        result.last_update_step = int(state.get('last_update_step', 0))
        return result

    def save(self, json_path: str, tensor_path: str) -> None:
        payload = self.state_dict()
        tensors = {}
        for group in ('semantic_prototypes', 'tap_prototypes'):
            for key, value in payload[group].items():
                tensors[f'{group}.{key}'] = value
        if not tensors:
            tensors['__empty_state__'] = torch.empty(0, dtype=torch.float32)
        save_file(tensors, tensor_path, metadata={'semantic_scaffold_state': '1'})
        import json
        with open(json_path, 'w', encoding='utf-8') as handle:
            json.dump({key: value for key, value in payload.items() if key not in ('semantic_prototypes', 'tap_prototypes')}, handle, ensure_ascii=False, sort_keys=True, indent=2)

    @classmethod
    def load(cls, json_path: str, tensor_path: str, *, device=None) -> 'SemanticScaffoldState':
        import json
        with open(json_path, 'r', encoding='utf-8') as handle:
            metadata = json.load(handle)
        loaded = load_file(tensor_path, device='cpu')
        metadata.update({
            'semantic_prototypes': {
                key.split('.', 1)[1]: value.to(device=device) if device is not None else value
                for key, value in loaded.items()
                if key.startswith('semantic_prototypes.')
            },
            'tap_prototypes': {
                key.split('.', 1)[1]: value.to(device=device) if device is not None else value
                for key, value in loaded.items()
                if key.startswith('tap_prototypes.')
            },
        })
        return cls.from_state_dict(metadata, device=device)
