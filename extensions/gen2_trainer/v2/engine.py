"""One optimizer for the standalone v2 activator; the native models stay frozen."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import math
import time

import torch

from ..objectives import per_example_mse


SCHEMA_VERSION = "2.0.0"
NOISE_EDGES = (0., .1, .25, .5, .75, .9, 1.)


class V2UpdateError(RuntimeError):
    """An update failed; the engine must be restored from a coherent checkpoint."""


def _statistics(value):
    flat = value.detach().double().reshape(-1)
    finite = bool(torch.isfinite(flat).all())
    return {"finite": finite, "elements": flat.numel(),
            "l2": float(flat.norm()) if finite else None,
            "rms": float(flat.square().mean().sqrt()) if finite else None,
            "max_abs": float(flat.abs().max()) if finite else None,
            "nonzero_count": int(torch.count_nonzero(flat))}


def _token_statistics(current, initial):
    current, initial = current.detach().double(), initial.detach().double()
    norms, initial_norms = current.norm(dim=1), initial.norm(dim=1)
    pairwise = current @ current.T
    denominator = norms[:, None] * norms[None, :]
    pairwise = pairwise / denominator.clamp_min(torch.finfo(torch.float64).tiny)
    pairs = [[float(pairwise[i, j].clamp(-1, 1)) if denominator[i, j] > 0 else None
              for j in range(len(norms))] for i in range(len(norms))]
    initial_denominator = norms * initial_norms
    similarities = (current * initial).sum(1) / initial_denominator.clamp_min(torch.finfo(torch.float64).tiny)
    return {"norms": norms.tolist(), "initial_norms": initial_norms.tolist(),
            "movement_from_initial": (current-initial).norm(dim=1).tolist(),
            "cosine_to_initial": [float(value.clamp(-1, 1)) if scale > 0 else None
                                  for value, scale in zip(similarities, initial_denominator)],
            "pairwise_cosine": pairs}


def _feature_statistics(value):
    """At most 4096 deterministic coordinates, with no full feature cast/copy."""
    value = value.detach()
    total = value.numel()
    if total == 0:
        raise ValueError("V2 conditioning features cannot be empty during training")
    stride = max(1, math.ceil(total/4096))
    indices = torch.arange(0, total, stride, device=value.device)
    if value.ndim == 0:
        sample = value.reshape(1)
    else:
        # Coordinate indexing also avoids a full contiguous copy when native
        # packed features are a noncontiguous view.
        remaining, coordinates = indices, []
        for dimension in reversed(value.shape):
            coordinates.append(remaining.remainder(dimension))
            remaining = remaining // dimension
        sample = value[tuple(reversed(coordinates))]
    stats = _statistics(sample)
    if not stats["finite"]:
        raise FloatingPointError("Nonfinite sampled conditioning features")
    return {"method": "deterministic_strided_coordinates", "sampled": sample.numel() < total,
            "sample_coordinates": sample.numel(), "total_coordinates": total, "stride": stride,
            "rms": stats["rms"], "max_abs": stats["max_abs"], "finite": True}


def _compact_example_metadata(metadata, compiled, caption):
    source_keys = {"sample_id", "example_id", "dataset_index", "split", "group", "path", "relative_path",
                   "content_hash", "caption_hash", "canonical_caption_hash", "caption_provenance",
                   "time_table_index", "time_bin", "time_bin_bounds", "timestep", "tau", "internal_ideogram_time",
                   "latent_shape", "valid_elements", "z0_rms", "noise_rms", "z_tau_rms", "target_rms",
                   "transforms", "augmentation_replay_seed", "augmentation_replay_reason"}
    compiled_keys = {"mode", "occurrence_count", "occurrences", "insertion_spans", "occurrence_token_spans",
                     "normalized_marker_character_spans", "num_vectors", "original_length", "expanded_length",
                     "resulting_length", "omitted_content_tokens", "truncated", "retained_character_count",
                     "common_ordinary_content_sha256", "shared_comparison_content", "overflow_policy", "max_length", "placement"}
    source = {key: deepcopy(value) for key, value in metadata.items() if key in source_keys}
    source["caption_sha256"] = hashlib.sha256(caption.encode("utf-8")).hexdigest()
    conditioning = {key: deepcopy(value) for key, value in compiled.items() if key in compiled_keys}
    for key in ("normalized_caption", "serialized_text"):
        if key in compiled:
            conditioning[key+"_sha256"] = hashlib.sha256(compiled[key].encode("utf-8")).hexdigest()
    if "input_ids" in compiled:
        conditioning["input_ids_sha256"] = hashlib.sha256(json.dumps(compiled["input_ids"], separators=(",", ":")).encode()).hexdigest()
    return {**source, "conditioning": conditioning}


def _finite_tree(value):
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


class V2Engine:
    """Accumulate example-weighted losses, then commit exactly one token update.

    Prepared windows contain small latent maps, not live computation graphs.
    Each microbatch completes its backward before the next one is encoded.
    RNG and data-stream restoration belong to the runner.
    """

    def __init__(self, config, backend, accelerator=None):
        self.config, self.backend = deepcopy(config), backend
        self.accelerator = accelerator
        self.parameter = backend.tokens.E
        if not isinstance(self.parameter, torch.nn.Parameter) or self.parameter.dtype != torch.float32:
            raise ValueError("V2 requires one FP32 learned token Parameter E")
        if self.parameter.ndim != 2 or not self.parameter.numel():
            raise ValueError("V2 token bank must be a nonempty M by embedding-dimension matrix")
        parameters = list(backend.tokens.parameters())
        if len(parameters) != 1 or parameters[0] is not self.parameter:
            raise ValueError("The primary v2 experiment trains only E")
        if backend.tokens.initial.shape != self.parameter.shape or backend.tokens.initial.requires_grad:
            raise ValueError("Token initialization must be a frozen buffer matching E")
        train = self.config["train"]
        self.total_updates = train["steps"]
        self.accumulation_steps = train["gradient_accumulation_steps"]
        self.max_grad_norm = float(train.get("max_grad_norm", 0.))
        self.noise_edges = tuple(float(value) for value in self.config["gen2"].get("diagnostics", {}).get("time_bin_edges", NOISE_EDGES))
        if (len(self.noise_edges) < 2 or self.noise_edges[0] != 0 or self.noise_edges[-1] != 1 or
                any(not math.isfinite(value) for value in self.noise_edges) or
                any(a >= b for a, b in zip(self.noise_edges, self.noise_edges[1:]))):
            raise ValueError("Noise diagnostic edges must increase strictly from zero to one")
        if (type(self.total_updates) is not int or self.total_updates < 1 or
                type(self.accumulation_steps) is not int or self.accumulation_steps < 1):
            raise ValueError("V2 update and accumulation counts must be positive integers")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm < 0:
            raise ValueError("V2 max_grad_norm must be finite and >=0; zero disables clipping")
        if accelerator is not None and getattr(accelerator, "gradient_accumulation_steps", 1) != 1:
            raise ValueError("V2 owns accumulation scaling; Accelerator accumulation must be one")
        self.scaler = getattr(accelerator, "scaler", None)
        settings = self.config["gen2"]["optimizer"]
        self.optimizer_settings = {"type": settings["type"].lower(), "lr": float(settings["lr"]),
            "eps": float(settings["eps"]), "betas": tuple(float(x) for x in settings["betas"]),
            "weight_decay": float(settings["weight_decay"])}
        if self.optimizer_settings["type"] not in ("adamw", "adamw8bit"):
            raise ValueError("V2 supports native AdamW and AdamW8bit")
        if (self.optimizer_settings["lr"] <= 0 or self.optimizer_settings["eps"] <= 0 or
                self.optimizer_settings["weight_decay"] < 0 or
                len(self.optimizer_settings["betas"]) != 2 or
                any(not 0 <= beta < 1 for beta in self.optimizer_settings["betas"]) or
                not _finite_tree(self.optimizer_settings)):
            raise ValueError("Invalid v2 optimizer settings")
        from toolkit.optimizer import get_optimizer
        self.parameter.requires_grad_(True)
        # The native factory supplies eps=1e-6 itself. A parameter-group override
        # is native optimizer behavior and avoids duplicate constructor keywords.
        self.optimizer = get_optimizer([{"params": [self.parameter], "eps": self.optimizer_settings["eps"]}],
            optimizer_type=self.optimizer_settings["type"], learning_rate=self.optimizer_settings["lr"],
            optimizer_params={"betas": self.optimizer_settings["betas"],
                              "weight_decay": self.optimizer_settings["weight_decay"]})
        self._assert_optimizer_settings()
        self.logical_update = 0
        self.update_attempt = 0
        self.failed = False
        self.partial_commit = False
        self.accumulation_status = "boundary"
        self.factory_manifest = {"factory": "toolkit.optimizer.get_optimizer",
            "optimizer_class": f"{type(self.optimizer).__module__}.{type(self.optimizer).__name__}",
            "settings": deepcopy(self.optimizer_settings), "parameter_family": "embedding",
            "parameter_elements": self.parameter.numel(), "master_dtype": "torch.float32",
            "epsilon_source": "explicit_parameter_group", "schedule": "constant",
            "scheduler": None, "max_grad_norm": self.max_grad_norm}
        self._assert_frozen()

    def _assert_optimizer_settings(self):
        if len(self.optimizer.param_groups) != 1:
            raise ValueError("V2 requires exactly one active optimizer parameter group")
        group = self.optimizer.param_groups[0]
        if len(group["params"]) != 1 or group["params"][0] is not self.parameter:
            raise ValueError("V2 optimizer must own E exclusively")
        for key in ("lr", "eps", "weight_decay"):
            if float(group[key]) != self.optimizer_settings[key]:
                raise ValueError(f"Actual optimizer {key} does not match resolved v2 configuration")
        if tuple(group["betas"]) != self.optimizer_settings["betas"]:
            raise ValueError("Actual optimizer betas do not match resolved v2 configuration")

    def _assert_frozen(self):
        self.backend.assert_frozen()
        if self.parameter.dtype != torch.float32 or not self.parameter.requires_grad:
            raise RuntimeError("V2 learned token ownership or FP32 master dtype changed")
        for parameter in self.backend.frozen_parameters():
            if parameter is self.parameter or parameter.requires_grad or parameter.grad is not None:
                raise RuntimeError("A frozen native parameter received gradient ownership")

    def _autocast(self):
        return self.accelerator.autocast() if self.accelerator is not None else nullcontext()

    def step(self, window):
        if self.failed:
            raise V2UpdateError("This engine aborted; restore a coherent resume checkpoint")
        if self.accumulation_status != "boundary":
            raise V2UpdateError("An accumulation window is already active")
        if self.logical_update >= self.total_updates:
            raise StopIteration("The configured v2 pilot is complete")
        if not isinstance(window, (list, tuple)) or len(window) != self.accumulation_steps:
            raise ValueError("A v2 update needs a complete list of accumulation microbatches")
        if any(not batch.get("qs") for batch in window):
            raise ValueError("Empty v2 accumulation microbatch")
        total = sum(len(batch["qs"]) for batch in window)
        self.update_attempt += 1
        self.accumulation_status = "accumulating"
        started, step_attempted = time.perf_counter(), False
        rows, timings, bins = [], {}, [[] for _ in range(len(self.noise_edges)-1)]
        if self.parameter.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.parameter.device)
        try:
            self._assert_frozen()
            self._assert_optimizer_settings()
            self.optimizer.zero_grad(set_to_none=True)
            for index, batch in enumerate(window):
                qs, zt, tau, target = (batch[key] for key in ("qs", "zt", "tau", "target"))
                count = len(qs)
                metadata = batch.get("metadata", [{} for _ in qs])
                if len(metadata) != count or any(x.shape[0] != count for x in (zt, tau, target)):
                    raise ValueError("V2 prepared microbatch example counts disagree")
                if tau.ndim != 1 or not bool(torch.isfinite(tau).all()) or bool(((tau <= 0) | (tau > 1)).any()):
                    raise ValueError("V2 training requires one finite noise fraction in (0,1] per example")
                self.backend.active_batch_metadata = metadata
                self.backend.active_accumulation_index = index
                encoded_start = time.perf_counter()
                with self._autocast():
                    conditioning = self.backend.encode(qs, mode="learned", gradients=True)
                    predicted_start = time.perf_counter()
                    prediction = self.backend.predict(zt, tau, conditioning)
                backward_start = time.perf_counter()
                losses = per_example_mse(prediction, target, batch.get("valid_mask"))
                objective = losses.sum() / total
                if not bool(torch.isfinite(objective)) or not objective.requires_grad:
                    raise FloatingPointError("Nonfinite loss or disconnected activator objective")
                if self.accelerator is None:
                    objective.backward()
                else:
                    self.accelerator.backward(objective)
                completed = time.perf_counter()
                for key, elapsed in (("encoder_seconds", predicted_start-encoded_start),
                                     ("predict_seconds", backward_start-predicted_start),
                                     ("loss_backward_seconds", completed-backward_start)):
                    timings[key] = timings.get(key, 0.) + elapsed
                compiled = getattr(conditioning, "metadata", [{} for _ in qs])
                for i, (q, loss, noise) in enumerate(zip(qs, losses.detach().cpu().tolist(), tau.detach().cpu().tolist())):
                    bin_index = next(j for j in range(len(self.noise_edges)-1)
                                     if noise < self.noise_edges[j+1] or j == len(self.noise_edges)-2)
                    bins[bin_index].append(loss)
                    feature_stats = _feature_statistics(conditioning.features[i])
                    rows.append({**_compact_example_metadata(metadata[i], compiled[i], q), "loss": loss, "tau": noise,
                                 "noise_bin": bin_index, "accumulation_index": index,
                                 "example_weight": 1./total, "conditioning_features": feature_stats})
                del objective, losses, prediction, conditioning
            if self.config.get("model", {}).get("layer_offloading") or self.config.get("model", {}).get("low_vram"):
                from toolkit.memory_management import sync_grad_transfers
                sync_grad_transfers()
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            self._assert_frozen()
            gradient = self.parameter.grad
            if gradient is None or gradient.dtype != torch.float32:
                raise RuntimeError("Activator E has no FP32 gradient")
            gradients = {"before_clip": _statistics(gradient), "clipping_applied": self.max_grad_norm > 0}
            if not gradients["before_clip"]["finite"] or not gradients["before_clip"]["nonzero_count"]:
                raise FloatingPointError("Activator gradients are nonfinite or entirely zero")
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_([self.parameter], self.max_grad_norm, error_if_nonfinite=True)
            gradients["after_clip"] = _statistics(self.parameter.grad)
            before = self.parameter.detach().clone()
            optimizer_start = time.perf_counter()
            step_attempted = True
            if self.scaler is None:
                self.optimizer.step()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            update = _statistics(self.parameter.detach()-before)
            if not update["finite"] or not update["nonzero_count"] or not bool(torch.isfinite(self.parameter).all()):
                raise FloatingPointError("Activator optimizer produced nonfinite or entirely zero parameter updates")
            if not _finite_tree(self.optimizer.state):
                raise FloatingPointError("Activator optimizer produced nonfinite moment or quantizer state")
            self._assert_frozen()
            self._assert_optimizer_settings()
            token_stats = _token_statistics(self.parameter, self.backend.tokens.initial)
            timings["optimizer_seconds"] = time.perf_counter()-optimizer_start
            # No failure after this point is treated as a completed update until
            # the scalar return record is fully assembled.
            result = {"logical_update": self.logical_update+1, "feature_version": self.logical_update,
                "update_attempt_id": self.update_attempt, "loss": sum(row["loss"] for row in rows)/total,
                "example_count": total, "microbatch_count": len(window), "effective_batch_size": total,
                "gradients": gradients, "parameter_update": update, "tokens": token_stats,
                "optimizer": deepcopy(self.optimizer_settings), "gradient_ownership_asserted": True,
                "examples": rows, "noise_bins": [
                    {"index": i, "bounds": [self.noise_edges[i], self.noise_edges[i+1]], "count": len(values),
                     "mean_loss": sum(values)/len(values) if values else None}
                    for i, values in enumerate(bins)],
                "noise_bin_convention": "left_closed_right_open_except_final_closed",
                "timing": {**timings, "total_seconds": time.perf_counter()-started},
                "timing_method": "host_wall_clock_no_cuda_synchronization",
                "accumulation_status": "boundary", "skipped_updates": 0, "failed_attempts": 0}
            if self.parameter.is_cuda:
                result["cuda_memory"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(self.parameter.device),
                                         "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.parameter.device)}
            self.logical_update += 1
            self.accumulation_status = "boundary"
            return result
        except Exception as error:
            self.failed, self.partial_commit = True, step_attempted
            self.accumulation_status = "failed"
            raise V2UpdateError(f"V2 attempt {self.update_attempt} aborted; optimizer_step_attempted={step_attempted}; "
                                f"reload the last coherent checkpoint: {error}") from error

    def state_dict(self):
        if self.failed or self.accumulation_status != "boundary":
            raise V2UpdateError("Only a coherent successful update boundary may be saved")
        self._assert_optimizer_settings()
        state = {"schema_version": SCHEMA_VERSION, "logical_update": self.logical_update,
                 "update_attempt": self.update_attempt, "total_updates": self.total_updates,
                 "accumulation_steps": self.accumulation_steps, "max_grad_norm": self.max_grad_norm,
                 "optimizer_settings": deepcopy(self.optimizer_settings), "optimizer": self.optimizer.state_dict(),
                 "scheduler": None, "scaler": self.scaler.state_dict() if self.scaler is not None else None,
                 "accumulation_status": "boundary", "failed": False}
        if not _finite_tree(state):
            raise V2UpdateError("Nonfinite optimizer state cannot be checkpointed")
        return state

    def validate_state_dict(self, state):
        required = {"schema_version", "logical_update", "update_attempt", "total_updates", "accumulation_steps",
                    "max_grad_norm", "optimizer_settings", "optimizer", "scheduler", "scaler", "accumulation_status", "failed"}
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError("Incomplete or unsupported v2 engine state")
        if state["schema_version"] != SCHEMA_VERSION or state["failed"] or state["accumulation_status"] != "boundary":
            raise ValueError("V2 resume state is not a coherent supported boundary")
        for key in ("total_updates", "accumulation_steps", "max_grad_norm", "optimizer_settings"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Strict v2 resume rejects changed {key}")
        if (type(state["logical_update"]) is not int or not 0 <= state["logical_update"] <= self.total_updates or
                type(state["update_attempt"]) is not int or state["update_attempt"] != state["logical_update"]):
            raise ValueError("V2 resume counters are inconsistent")
        if state["scheduler"] is not None or (state["scaler"] is None) != (self.scaler is None):
            raise ValueError("V2 resume schedule or loss scaler changed")
        optimizer = state["optimizer"]
        if set(optimizer) != {"state", "param_groups"} or len(optimizer["param_groups"]) != 1:
            raise ValueError("V2 resume must contain one token-only optimizer")
        group = optimizer["param_groups"][0]
        if len(group["params"]) != 1 or set(optimizer["state"]) - set(group["params"]):
            raise ValueError("Unexpected parameter ownership in v2 optimizer state")
        for key in ("lr", "eps", "weight_decay", "betas"):
            value = tuple(group[key]) if key == "betas" else group[key]
            if value != self.optimizer_settings[key]:
                raise ValueError(f"V2 saved optimizer group {key} contradicts resolved settings")
        if state["logical_update"] > 0 and not optimizer["state"]:
            raise ValueError("V2 resume omits the active optimizer moments")
        if state["logical_update"] == 0 and optimizer["state"]:
            raise ValueError("Initial v2 state unexpectedly contains active optimizer moments")
        if optimizer["state"]:
            moments = next(iter(optimizer["state"].values()))
            step = moments.get("step")
            if isinstance(step, torch.Tensor):
                if step.numel() != 1:
                    raise ValueError("Invalid optimizer step tensor")
                step = step.item()
            if step != state["logical_update"]:
                raise ValueError("V2 optimizer step contradicts the committed update counter")
            moment_names = ("exp_avg", "exp_avg_sq") if self.optimizer_settings["type"] == "adamw" else ("state1", "state2")
            for key in moment_names:
                moment = moments.get(key)
                if not isinstance(moment, torch.Tensor) or moment.shape != self.parameter.shape:
                    raise ValueError(f"V2 optimizer {key} has an incompatible token shape")
                allowed_dtypes = (torch.float32,) if self.optimizer_settings["type"] == "adamw" else (torch.float32, torch.uint8)
                if moment.dtype not in allowed_dtypes:
                    raise ValueError(f"V2 optimizer {key} has an incompatible state dtype")
        if not _finite_tree(state):
            raise ValueError("Nonfinite v2 resume state")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        try:
            self.optimizer.load_state_dict(state["optimizer"])
            if self.scaler is not None:
                self.scaler.load_state_dict(state["scaler"])
            self.optimizer.zero_grad(set_to_none=True)
            self._assert_optimizer_settings()
            self._assert_frozen()
            self.logical_update, self.update_attempt = state["logical_update"], state["update_attempt"]
            self.failed = self.partial_commit = False
            self.accumulation_status = "boundary"
        except Exception:
            self.failed = True
            self.accumulation_status = "failed"
            raise
