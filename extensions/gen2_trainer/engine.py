"""One authoritative logical-update loop for the four Gen2 parameter families."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
import inspect
import math
import random
import time
from typing import Mapping

import torch

from .config import ConfigError, ROLES, family_horizons, scheduler_kwargs
from .objectives import per_example_mse

ACTIVE = {"D": ("diffusion",), "A": ("embedding", "text_adapter"), "G": ("gates",)}


@dataclass(frozen=True)
class PhaseSchedule:
    warmup_updates: int = 450
    refinement_updates: int = 2100
    calibration_updates: int = 450
    diffusion_updates_per_cycle: int = 4
    conditioning_updates_per_cycle: int = 1

    def __post_init__(self):
        for key, value in self.as_dict().items():
            minimum = 1 if key.endswith("per_cycle") else 0
            if type(value) is not int or value < minimum:
                raise ConfigError(f"phases.{key} must be an integer >= {minimum}")
        if self.total < 1: raise ConfigError("A training schedule needs at least one update")

    def as_dict(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__}

    @property
    def total(self):
        return self.warmup_updates + self.refinement_updates + self.calibration_updates

    @property
    def horizons(self):
        return family_horizons(self.as_dict())

    @property
    def boundaries(self):
        return frozenset({0, self.warmup_updates, self.warmup_updates + self.refinement_updates, self.total})

    def stage_at(self, update):
        if type(update) is not int or not 0 <= update <= self.total: raise ValueError("logical update out of range")
        if update == self.total: return "complete"
        if update < self.warmup_updates: return "warmup"
        if update < self.warmup_updates + self.refinement_updates: return "refinement"
        return "calibration"

    def kind_at(self, update):
        stage = self.stage_at(update)
        if stage == "complete": raise StopIteration("The Gen2 schedule is complete")
        if stage == "warmup": return "D"
        if stage == "calibration": return "G"
        pos = (update - self.warmup_updates) % (self.diffusion_updates_per_cycle + self.conditioning_updates_per_cycle)
        return "D" if pos < self.diffusion_updates_per_cycle else "A"

    next_kind = kind_at

    def position(self, update):
        stage = self.stage_at(update)
        offset = update if stage == "warmup" else update - self.warmup_updates if stage == "refinement" else update - self.warmup_updates - self.refinement_updates
        cycle = self.diffusion_updates_per_cycle + self.conditioning_updates_per_cycle
        return {"stage": stage, "stage_update": max(0, offset),
                "cycle_index": offset // cycle if stage == "refinement" else None,
                "cycle_position": offset % cycle if stage == "refinement" else None}

    def counts_at(self, update):
        self.stage_at(update)
        warm = min(update, self.warmup_updates)
        refinement = max(0, min(update - self.warmup_updates, self.refinement_updates))
        calibration = max(0, update - self.warmup_updates - self.refinement_updates)
        return family_horizons({**self.as_dict(), "warmup_updates": warm,
                               "refinement_updates": refinement, "calibration_updates": calibration})


def gradient_statistics(parameters):
    """Inspect full-family gradients without replacing or mutating them."""
    parameters = tuple(parameters)
    squared = 0.0
    count = finite_count = nonzero_count = missing = 0
    largest = 0.0
    for p in parameters:
        if p.grad is None:
            missing += 1
            continue
        grad = p.grad.detach()
        if grad.is_sparse: raise ValueError("Gen2 dense adapters must not produce sparse gradients")
        if grad.dtype != torch.float32: raise RuntimeError("Trainable-master gradients must remain float32")
        finite = torch.isfinite(grad)
        count += grad.numel()
        finite_count += int(finite.sum())
        nonzero_count += int((finite & (grad != 0)).sum())
        if bool(finite.all()):
            squared += float(grad.double().square().sum())
            largest = max(largest, float(grad.abs().max()))
    all_finite = finite_count == count
    return {"method": "exact", "l2_norm": math.sqrt(squared) if all_finite and count else None,
            "rms": math.sqrt(squared / count) if all_finite and count else None,
            "max_abs": largest if all_finite and count else None,
            "finite_fraction": finite_count / count if count else None,
            "nonzero_fraction": nonzero_count / count if count else None,
            "missing_gradient_tensors": missing, "gradient_tensor_count": len(parameters) - missing,
            "gradient_elements": count, "parameter_elements": sum(p.numel() for p in parameters),
            "all_finite": all_finite, "undefined_reason": "missing_gradients" if not count else None}


def _native_optimizer(*args, **kwargs):
    from toolkit.optimizer import get_optimizer
    return get_optimizer(*args, **kwargs)


def _native_scheduler(*args, **kwargs):
    from toolkit.scheduler import get_lr_scheduler
    return get_lr_scheduler(*args, **kwargs)


def _resolve_external_scheduler(name, kwargs, horizon):
    """Inspect diffusers' actual constructor for factory fall-through names."""
    native = {"cosine", "cosine_with_restarts", "step", "constant", "linear", "constant_with_warmup"}
    if name in native: return kwargs
    from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
    try: function = TYPE_TO_SCHEDULER_FUNCTION[SchedulerType(name)]
    except (ValueError, KeyError) as error: raise ConfigError(f"Unknown native LR scheduler {name!r}") from error
    signature = inspect.signature(function)
    resolved = deepcopy(kwargs)
    if "num_training_steps" in signature.parameters:
        resolved.setdefault("num_training_steps", horizon)
    return resolved


class PartialUpdateError(RuntimeError):
    """An optimizer/scheduler may have mutated state; reload the last checkpoint."""


class Gen2Engine:
    def __init__(self, config, backend, accelerator=None, recorder=None,
                 optimizer_factory=None, scheduler_factory=None):
        self.config = deepcopy(config)
        self.gen2 = self.config["gen2"]
        self.backend = backend
        self.accelerator = accelerator
        self.recorder = recorder
        self.schedule = PhaseSchedule(**self.gen2["phases"])
        self.families = {key: tuple(value) for key, value in backend.parameter_families().items()}
        if set(self.families) != set(ROLES): raise ValueError(f"Backend must expose exactly {ROLES}")
        ids = [id(p) for role in ROLES for p in self.families[role]]
        if len(ids) != len(set(ids)): raise ValueError("Gen2 parameter families overlap")
        for role, params in self.families.items():
            if not params: raise ValueError(f"Parameter family {role} is empty")
            if any(not isinstance(p, torch.nn.Parameter) or p.dtype != torch.float32 for p in params):
                raise ValueError(f"{role} must own float32 Parameter masters")
        if accelerator is not None and getattr(accelerator, "gradient_accumulation_steps", 1) != 1:
            raise ValueError("Gen2 owns example-weighted accumulation: accelerator.gradient_accumulation_steps must be 1")
        self.scaler = getattr(accelerator, "scaler", None) if accelerator is not None else None
        self.optimizers = {}
        self.schedulers = {}
        self.scheduler_descriptors = {}
        self.factory_manifest = {}
        optimizer_factory = optimizer_factory or _native_optimizer
        for role in ROLES:
            spec = self.gen2["optimizers"][role]
            optimizer_args = {"optimizer_type": spec["optimizer"], "learning_rate": spec["lr"],
                              "optimizer_params": deepcopy(spec["optimizer_params"])}
            try:
                optimizer = optimizer_factory(list(self.families[role]), **optimizer_args)
            except (ImportError, TypeError, ValueError) as error:
                raise ConfigError(f"Native optimizer construction failed for {role} ({spec['optimizer']}): {error}; no fallback selected") from error
            expected = [id(p) for p in self.families[role]]
            actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
            if actual != expected: raise RuntimeError(f"Native {role} optimizer changed parameter ownership/order")
            self.optimizers[role] = optimizer
            horizon = self.schedule.horizons[role]
            kwargs = scheduler_kwargs(spec["lr_scheduler"], spec["lr_scheduler_params"], horizon)
            requested = {"name": spec["lr_scheduler"], "kwargs": deepcopy(spec["lr_scheduler_params"])}
            if horizon:
                if scheduler_factory is None:
                    kwargs = _resolve_external_scheduler(spec["lr_scheduler"], kwargs, horizon)
                try:
                    scheduler = (scheduler_factory or _native_scheduler)(spec["lr_scheduler"], optimizer, **kwargs)
                except (ImportError, TypeError, ValueError) as error:
                    raise ConfigError(f"Native scheduler construction failed for {role}: {error}; kwargs={kwargs!r}") from error
            else:
                scheduler = None
            self.schedulers[role] = scheduler
            self.scheduler_descriptors[role] = {"active": bool(horizon), "horizon": horizon,
                                              "requested": requested, "constructor_kwargs": kwargs}
            self.factory_manifest[role] = {**optimizer_args, "actual_class": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
                "initial_group_lrs": [float(g["lr"]) for g in optimizer.param_groups],
                "native_defaults": deepcopy(optimizer.defaults),
                "native_parameter_group_settings": [{key: deepcopy(value) for key, value in group.items() if key != "params"}
                                                     for group in optimizer.param_groups],
                "scheduler": deepcopy(self.scheduler_descriptors[role])}
        self.committed_update = 0
        self.update_attempt = 0
        self.family_steps = {role: 0 for role in ROLES}
        self.failed = False
        self.partial_commit = False
        self.accumulation_status = "boundary"
        # The runner binds this to Evaluation.retained_bytes after initialization.
        # Training tensors/gradients are not diagnostic buffers.
        self.external_diagnostic_bytes = 0
        rng = random.Random(self.gen2["execution"]["diagnostic_seed"])
        limit = self.gen2["diagnostics"]["parameter_sample_elements_per_family"]
        self.coordinate_indices = {}
        coordinate_bytes = sum(min(sum(p.numel() for p in self.families[role]), limit) * 8 for role in ROLES)
        if coordinate_bytes > self.gen2["diagnostics"]["tensor_memory_budget_mb"] * 1024 ** 2:
            raise ConfigError("Diagnostic tensor budget cannot hold the configured mandatory update-coordinate indices")
        for role in ROLES:
            total = sum(p.numel() for p in self.families[role])
            self.coordinate_indices[role] = torch.tensor(sorted(rng.sample(range(total), min(total, limit))), dtype=torch.int64)
        if self.recorder:
            self.recorder.event("optimizer_construction", factories=self.factory_manifest)
            self.recorder.event("parameter_coordinate_selection", seed=self.gen2["execution"]["diagnostic_seed"],
                indices={role: value.tolist() for role, value in self.coordinate_indices.items()},
                sampling="uniform_without_replacement", purpose="update_magnitudes")

    @property
    def logical_update(self): return self.committed_update

    def retained_diagnostic_bytes(self):
        return sum(value.numel() * value.element_size() for value in self.coordinate_indices.values())

    def _external_diagnostic_bytes(self):
        value = self.external_diagnostic_bytes
        return int(value() if callable(value) else value)

    def _context(self, kind, stage):
        return {"logical_update": self.committed_update, "update_attempt_id": self.update_attempt,
                "stage": stage, "update_kind": kind, "pass_type": "training"}

    def _assert_and_set_ownership(self, kind):
        if hasattr(self.backend, "assert_frozen"): self.backend.assert_frozen()
        active = ACTIVE[kind]
        for role, params in self.families.items():
            for p in params:
                if p.dtype != torch.float32: raise RuntimeError(f"{role} master dtype changed")
                p.grad = None
                p.requires_grad_(role in active)
        if hasattr(self.backend, "set_gradient_ownership"): self.backend.set_gradient_ownership(kind)
        for role, params in self.families.items():
            if any(p.requires_grad != (role in active) for p in params):
                raise RuntimeError(f"Backend violated {role} gradient ownership")
        for role in active:
            self.optimizers[role].zero_grad(set_to_none=True)

    def _backward(self, loss):
        if loss.ndim != 0 or not bool(torch.isfinite(loss)): raise FloatingPointError("Nonfinite or nonscalar training objective")
        if not loss.requires_grad: raise RuntimeError("Active objective is disconnected from trainable parameters")
        if self.accelerator is None:
            loss.backward()
        else:
            self.accelerator.backward(loss)

    def _autocast(self):
        return self.accelerator.autocast() if self.accelerator is not None else nullcontext()

    @staticmethod
    def _numbers(values): return [float(v) for v in values.detach().float().cpu()]

    def _microbatch(self, batch, kind, weight, accumulation_index, context):
        required = ("qs", "z0", "noise", "tau", "zt", "target")
        if any(key not in batch for key in required): raise ValueError(f"Prepared batch must contain {required}")
        qs, tau, zt, target = batch["qs"], batch["tau"], batch["zt"], batch["target"]
        count = len(qs)
        metadata = batch.get("metadata", [{} for _ in qs])
        if len(metadata) != count:
            raise ValueError("Prepared batch metadata must have one entry per example")
        # Activation hooks also execute during the backward recomputation inside
        # this microbatch, so keep its sample identity available until it ends.
        self.backend.active_batch_metadata = metadata
        self.backend.active_accumulation_index = accumulation_index
        if any(value.shape[0] != count for value in (tau, zt, target, batch["z0"], batch["noise"])):
            raise ValueError("Prepared batch has inconsistent example counts")
        if tau.ndim != 1 or not bool(torch.isfinite(tau).all()) or bool(((tau <= 0) | (tau > 1)).any()):
            raise ValueError("Training tau must be one finite (0,1] toolkit noise fraction per example")
        gates = "learned" if kind == "G" else "one"
        coefficients = self.gen2["losses"]
        neutral_needed = kind in ("D", "G") and coefficients["neutral_weight"] > 0
        timings = {}
        started = time.perf_counter()
        neutral = teacher = None
        if neutral_needed:
            with torch.no_grad(), self._autocast():
                neutral = self.backend.encode(qs, styled=False, gradients=False)
                timings["neutral_encoder_seconds"] = time.perf_counter() - started
                teacher_start = time.perf_counter()
                with self.backend.branch(tau, lora_enabled=False, gate_mode="one", strength=1.0, name="base_teacher"):
                    teacher = self.backend.predict(zt, tau, neutral).detach()
            timings["teacher_seconds"] = time.perf_counter() - teacher_start
        else:
            timings["teacher_seconds"] = None
            timings["neutral_encoder_seconds"] = None
        started = time.perf_counter()
        with self.backend.branch(tau, lora_enabled=True, gate_mode=gates, strength=1.0, name="styled_student"):
            with self._autocast():
                styled = self.backend.encode(qs, styled=True, gradients=kind == "A")
                timings["styled_encoder_seconds"] = time.perf_counter() - started
                forward_start = time.perf_counter()
                prediction = self.backend.predict(zt, tau, styled)
                timings["styled_forward_seconds"] = time.perf_counter() - forward_start
            ls = per_example_mse(prediction, target, batch.get("valid_mask"))
            rt = styled.rt_per_example.float() if kind == "A" else None
            if rt is not None and rt.shape != (count,): raise ValueError("Text regularizer must have one value per example")
            objective = ls.mean()
            if rt is not None: objective = objective + coefficients["text_adapter_weight"] * rt.mean()
            backward_start = time.perf_counter()
            self._backward(weight * objective)
            timings["styled_backward_seconds"] = time.perf_counter() - backward_start
            ls_values = self._numbers(ls)
            rt_values = self._numbers(rt) if rt is not None else None
            encoded_metadata = deepcopy(getattr(styled, "metadata", [{} for _ in qs]))
            del prediction, objective, ls, rt, styled
        timings["styled_forward_backward_seconds"] = time.perf_counter() - started
        ln_values = None
        if neutral_needed:
            started = time.perf_counter()
            with self.backend.branch(tau, lora_enabled=True, gate_mode=gates, strength=1.0, name="neutral_student"):
                with self._autocast(): prediction = self.backend.predict(zt, tau, neutral)
                timings["neutral_forward_seconds"] = time.perf_counter() - started
                ln = per_example_mse(prediction, teacher, batch.get("valid_mask"))
                backward_start = time.perf_counter()
                self._backward(weight * coefficients["neutral_weight"] * ln.mean())
                timings["neutral_backward_seconds"] = time.perf_counter() - backward_start
                ln_values = self._numbers(ln)
                del prediction, ln
            timings["neutral_forward_backward_seconds"] = time.perf_counter() - started
        else:
            timings["neutral_forward_backward_seconds"] = None
            timings["neutral_forward_seconds"] = None
            timings["neutral_backward_seconds"] = None
        del neutral, teacher
        if self.recorder:
            for i, q in enumerate(qs):
                metadata = batch.get("metadata", [{} for _ in qs])[i]
                token_metadata = encoded_metadata[i] if isinstance(encoded_metadata, list) else encoded_metadata
                tau_i = float(tau[i])
                self.recorder.record("microbatches", {**metadata, "canonical_caption": q, "conditioning": token_metadata,
                    "accumulation_index": accumulation_index, "microbatch_size": count, "example_weight": weight / count,
                    "tau": tau_i, "native_timestep": tau_i * 1000., "internal_ideogram_time": 1. - tau_i,
                    "latent_shape": list(zt[i].shape), "valid_elements": int(zt[i].numel()) if batch.get("valid_mask") is None else int(torch.broadcast_to(batch["valid_mask"], zt.shape)[i].sum()),
                    "state_rms": {key: float(batch[key][i].detach().float().square().mean().sqrt()) for key in ("z0", "noise", "zt", "target")},
                    "losses": {"L_S": ls_values[i], "L_N": ln_values[i] if ln_values is not None else None,
                               "R_T": rt_values[i] if rt_values is not None else None},
                    "undefined_reasons": {key: "not_in_objective" for key, values in (("L_N", ln_values), ("R_T", rt_values)) if values is None},
                    "branches": {"styled_student": {"lora_enabled": True, "gate_mode": gates},
                                 "neutral_student": {"lora_enabled": True, "gate_mode": gates} if neutral_needed else None,
                                 "base_teacher": {"lora_enabled": False, "detached": True} if neutral_needed else None},
                    "timing": timings, "timing_method": "host_wall_clock_no_cuda_synchronization"}, **context)
        return {"L_S": sum(ls_values) / count,
                "L_N": sum(ln_values) / count if ln_values is not None else None,
                "R_T": sum(rt_values) / count if rt_values is not None else None}, timings

    def _sample_values(self, role):
        indices = self.coordinate_indices[role]
        values = []
        start = 0
        for param in self.families[role]:
            end = start + param.numel()
            chosen = indices[(indices >= start) & (indices < end)] - start
            if chosen.numel(): values.append(param.detach().reshape(-1)[chosen.to(param.device)].float().cpu())
            start = end
        return torch.cat(values)

    def step(self, window):
        """Commit one D/A/G update from an explicitly prepared accumulation window."""
        if self.failed: raise RuntimeError("This engine aborted; restore a complete checkpoint before continuing")
        if self.accumulation_status != "boundary": raise RuntimeError("An accumulation window is already in progress")
        if len(window) != self.config["train"]["gradient_accumulation_steps"]:
            raise ValueError("Each logical update requires exactly gradient_accumulation_steps microbatches")
        if not window or any(not batch.get("qs") for batch in window): raise ValueError("Empty accumulation microbatch")
        kind = self.schedule.kind_at(self.committed_update)
        stage = self.schedule.stage_at(self.committed_update)
        active = ACTIVE[kind]
        self.update_attempt += 1
        context = self._context(kind, stage)
        self.accumulation_status = "accumulating"
        started = time.perf_counter()
        attempted_roles = []
        try:
            self._assert_and_set_ownership(kind)
            total = sum(len(batch["qs"]) for batch in window)
            losses = {key: None for key in ("L_S", "L_N", "R_T", "R_C", "R_H")}
            timings = {}
            for index, batch in enumerate(window):
                weight = len(batch["qs"]) / total
                values, elapsed = self._microbatch(batch, kind, weight, index, context)
                for key, value in values.items():
                    if value is not None: losses[key] = (losses[key] or 0.) + weight * value
                for key, value in elapsed.items():
                    if value is not None: timings[key] = timings.get(key, 0.) + value
            coefficients = self.gen2["losses"]
            if kind == "G":
                gate_start = time.perf_counter()
                rc, rh = self.backend.gates.regularizers()
                self._backward(coefficients["gate_center_weight"] * rc + coefficients["gate_smoothness_weight"] * rh)
                losses["R_C"], losses["R_H"] = float(rc.detach()), float(rh.detach())
                timings["gate_regularizer_forward_backward_seconds"] = time.perf_counter() - gate_start
            if self.config["model"].get("layer_offloading") or self.config["model"].get("low_vram"):
                # Native offload gradients can still be asynchronous CPU copies.
                # Join before unscaling, inspecting, clipping or stepping them.
                from toolkit.memory_management import sync_grad_transfers
                sync_grad_transfers()
            if self.scaler is not None:
                for role in active: self.scaler.unscale_(self.optimizers[role])
            gradients = {role: {"pre_clip": gradient_statistics(self.families[role])} for role in active}
            # Validate every A-family before the first of its two optimizer steps.
            for role, stats in gradients.items():
                if not stats["pre_clip"]["all_finite"]: raise FloatingPointError(f"Nonfinite {role} gradients; no optimizer committed")
                if stats["pre_clip"]["gradient_tensor_count"] == 0: raise RuntimeError(f"Missing all {role} gradients: disconnected production path")
            max_norm = self.config["train"]["max_grad_norm"]
            for role in active:
                clip = self.gen2["optimizers"][role]["optimizer"] != "adafactor"
                if clip: torch.nn.utils.clip_grad_norm_(self.families[role], max_norm, error_if_nonfinite=True)
                gradients[role]["clipping_applied"] = clip
                gradients[role]["clipping_policy"] = "native_adafactor_exemption" if self.gen2["optimizers"][role]["optimizer"] == "adafactor" else "norm" if clip else "disabled"
                gradients[role]["post_clip"] = gradient_statistics(self.families[role])
            budget = self.gen2["diagnostics"]["tensor_memory_budget_mb"] * 1024 ** 2
            retained = self.retained_diagnostic_bytes() + self._external_diagnostic_bytes()
            sample_bytes = {role: self.coordinate_indices[role].numel() * 4 for role in active}
            # Before vectors for all active families + current after/difference.
            sample_working_bytes = sum(sample_bytes.values()) + 2 * max(sample_bytes.values())
            if retained + sample_working_bytes > budget:
                raise RuntimeError("Diagnostic tensor budget cannot hold mandatory sampled update snapshots alongside the fixed probes; increase the budget or reduce configured coordinate counts before a new run")
            snapshots = {role: self._sample_values(role) for role in active}
            full_interval = self.gen2["diagnostics"]["full_update_norm_every"]
            full_due = full_interval > 0 and (self.committed_update + 1) % full_interval == 0
            full_bytes = sum(p.numel() * p.element_size() for role in active for p in self.families[role])
            # Conservative bound for CPU snapshots plus fp64 diff/square reductions.
            full_peak_bytes = full_bytes + 5 * max(p.numel() * p.element_size() for role in active for p in self.families[role])
            full_snapshots = {role: [p.detach().cpu().clone() for p in self.families[role]] for role in active} if full_due and retained + sample_working_bytes + full_peak_bytes <= budget else None
            optimizer_start = time.perf_counter()
            rates = {}
            for role in active:
                optimizer = self.optimizers[role]
                rates[role] = {"requested_lr": self.gen2["optimizers"][role]["lr"],
                    "group_lrs_used": [float(group["lr"]) for group in optimizer.param_groups],
                    "weight_decay": [float(group.get("weight_decay", 0.)) for group in optimizer.param_groups],
                    "optimizer": self.gen2["optimizers"][role]["optimizer"], "family_step_before": self.family_steps[role]}
                attempted_roles.append(role)
                if self.scaler is None: optimizer.step()
                else: self.scaler.step(optimizer)
            if self.scaler is not None: self.scaler.update()
            for role in active:
                if any(not bool(torch.isfinite(p).all()) for p in self.families[role]):
                    raise FloatingPointError(f"Native optimizer produced nonfinite {role} parameters")
            for role in active:
                if self.schedulers[role] is None: raise RuntimeError("A zero-horizon family was scheduled for update")
                self.schedulers[role].step()
                rates[role]["group_lrs_next"] = [float(group["lr"]) for group in self.optimizers[role].param_groups]
                rates[role]["native_adaptive"] = [{key: float(group[key]) for key in ("d", "d_hat", "d_max") if key in group and _finite_scalar(group[key])}
                                                   for group in self.optimizers[role].param_groups]
            timings["optimizer_scheduler_seconds"] = time.perf_counter() - optimizer_start
            updates = {}
            for role in active:
                after = self._sample_values(role)
                diff = after - snapshots[role]
                elements = sum(p.numel() for p in self.families[role])
                sampled = diff.numel() < elements
                updates[role] = {"method": "sampled" if sampled else "exact", "sample_size": diff.numel(),
                    "total_elements": elements, "coverage": diff.numel() / elements,
                    "estimated_squared_l2": float(diff.double().square().sum()) * elements / diff.numel(),
                    "update_rms": float(diff.double().square().mean().sqrt()),
                    "parameter_l2": math.sqrt(sum(float(p.detach().double().square().sum()) for p in self.families[role]))}
                if full_snapshots is not None:
                    updates[role]["full_update_l2"] = math.sqrt(sum(float((p.detach().cpu() - before).double().square().sum()) for p, before in zip(self.families[role], full_snapshots[role])))
                elif full_due:
                    updates[role]["full_update_l2"] = None
                    updates[role]["full_update_reason"] = "tensor_memory_budget_exceeded"
            for role in ROLES:
                if role not in active and any(p.grad is not None for p in self.families[role]):
                    raise RuntimeError(f"Inactive family {role} received gradients")
            if hasattr(self.backend, "assert_frozen"): self.backend.assert_frozen()
            for role in active: self.family_steps[role] += 1
            self.committed_update += 1
            self.accumulation_status = "boundary"
            weighted = {"L_S": losses["L_S"], "L_N": None if losses["L_N"] is None else coefficients["neutral_weight"] * losses["L_N"],
                        "R_T": None if losses["R_T"] is None else coefficients["text_adapter_weight"] * losses["R_T"],
                        "R_C": None if losses["R_C"] is None else coefficients["gate_center_weight"] * losses["R_C"],
                        "R_H": None if losses["R_H"] is None else coefficients["gate_smoothness_weight"] * losses["R_H"]}
            result = {"logical_update": self.committed_update, "update_attempt_id": self.update_attempt,
                "stage": stage, "update_kind": kind, "active_families": list(active), "family_steps": deepcopy(self.family_steps),
                "feature_version": self.committed_update - 1, "gradient_ownership_asserted": True,
                "microbatch_count": len(window), "example_count": total, "effective_batch_size": total,
                "losses": losses, "coefficients": deepcopy(coefficients), "weighted_losses": weighted,
                "active_total": sum(value for value in weighted.values() if value is not None),
                "undefined_reasons": {key: "not_in_objective" for key, value in losses.items() if value is None},
                "gradients": gradients, "parameter_updates": updates, "learning_rates": rates,
                "phase_position_before": self.schedule.position(self.committed_update - 1),
                "timing": {**timings, "total_seconds": time.perf_counter() - started},
                "timing_method": "host_wall_clock_no_cuda_synchronization", "loss_scaling": {"owner": "accelerator" if self.accelerator else "engine", "accumulation_divisor": 1},
                "skipped_updates": 0, "failed_attempts": 0, "accumulation_status": "boundary"}
            if torch.cuda.is_available():
                result["cuda_memory"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            try:
                import psutil
                result["host_rss_bytes"] = psutil.Process().memory_info().rss
            except ImportError:
                result["host_rss_bytes"] = None
                result["host_rss_reason"] = "psutil_not_installed"
            if self.recorder:
                if hasattr(self.recorder, "status"):
                    result["recorder_status"] = self.recorder.status()
                self.recorder.record("updates", result, **{**context, "logical_update": self.committed_update})
            return result
        except Exception as error:
            self.failed = True
            self.partial_commit = bool(attempted_roles)
            self.accumulation_status = "failed"
            if self.recorder:
                self.recorder.event("partial_update_failure" if self.partial_commit else "update_aborted",
                    error_type=type(error).__name__, message=str(error), attempted_optimizer_roles=attempted_roles,
                    resume_required=True, **context)
                self.recorder.flush()
            if self.partial_commit:
                raise PartialUpdateError(f"Update attempt {self.update_attempt} may have partially committed {attempted_roles}; reload last coherent checkpoint: {error}") from error
            raise

    def state_dict(self):
        if self.failed or self.accumulation_status != "boundary":
            raise RuntimeError("Only coherent accumulation boundaries may be checkpointed")
        return {"schema_version": 1, "schedule": self.schedule.as_dict(), "committed_update": self.committed_update,
            "update_attempt": self.update_attempt, "family_steps": deepcopy(self.family_steps),
            "optimizers": {role: optimizer.state_dict() for role, optimizer in self.optimizers.items()},
            "optimizer_config": deepcopy(self.gen2["optimizers"]),
            "schedulers": {role: {**deepcopy(self.scheduler_descriptors[role]), "state": scheduler.state_dict() if scheduler is not None else None}
                           for role, scheduler in self.schedulers.items()},
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "coordinate_indices": {role: value.clone() for role, value in self.coordinate_indices.items()},
            "accumulation_status": "boundary", "failed": False}

    def load_state_dict(self, state):
        required = {"schema_version", "schedule", "committed_update", "update_attempt", "family_steps", "optimizers",
                    "optimizer_config", "schedulers", "scaler", "coordinate_indices", "accumulation_status", "failed"}
        if not isinstance(state, dict) or required - state.keys(): raise ValueError("Missing required Gen2 engine checkpoint state")
        if state["schema_version"] != 1 or state["failed"] or state["accumulation_status"] != "boundary":
            raise ValueError("Checkpoint is not a supported coherent Gen2 engine state")
        if state["schedule"] != self.schedule.as_dict() or state["optimizer_config"] != self.gen2["optimizers"]:
            raise ValueError("Strict resume rejects changed phase or optimizer/scheduler settings")
        update = state["committed_update"]
        if state["family_steps"] != self.schedule.counts_at(update): raise ValueError("Checkpoint family counters contradict its logical update")
        if type(state["update_attempt"]) is not int or state["update_attempt"] < update: raise ValueError("Invalid checkpoint update-attempt counter")
        for key in ("optimizers", "schedulers", "coordinate_indices"):
            if set(state[key]) != set(ROLES): raise ValueError(f"Checkpoint {key} must contain all four families")
        if (state["scaler"] is None) != (self.scaler is None): raise ValueError("Strict resume requires identical loss-scaling configuration")
        for role in ROLES:
            descriptor = state["schedulers"][role]
            if any(descriptor.get(key) != value for key, value in self.scheduler_descriptors[role].items()):
                raise ValueError(f"Checkpoint scheduler descriptor mismatch for {role}")
            if (descriptor.get("state") is None) != (self.schedulers[role] is None): raise ValueError(f"Missing/extra scheduler state for {role}")
            expected = self.coordinate_indices[role]
            if not torch.equal(state["coordinate_indices"][role].cpu(), expected): raise ValueError(f"Checkpoint diagnostic coordinates changed for {role}")
            optimizer_state = state["optimizers"][role]
            if not isinstance(optimizer_state, dict) or not {"state", "param_groups"} <= optimizer_state.keys():
                raise ValueError(f"Missing native optimizer state for {role}")
            if len(optimizer_state["param_groups"]) != len(self.optimizers[role].param_groups): raise ValueError(f"Optimizer groups mismatch for {role}")
            for saved, current in zip(optimizer_state["param_groups"], self.optimizers[role].param_groups):
                if len(saved["params"]) != len(current["params"]): raise ValueError(f"Optimizer parameter counts mismatch for {role}")
        for role in ROLES:
            self.optimizers[role].load_state_dict(state["optimizers"][role])
            if self.schedulers[role] is not None: self.schedulers[role].load_state_dict(state["schedulers"][role]["state"])
        if self.scaler is not None: self.scaler.load_state_dict(state["scaler"])
        self.committed_update = update
        self.update_attempt = state["update_attempt"]
        self.family_steps = deepcopy(state["family_steps"])
        self.failed = self.partial_commit = False
        self.accumulation_status = "boundary"
        for params in self.families.values():
            for p in params: p.grad = None


def _finite_scalar(value):
    try: return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError): return False
