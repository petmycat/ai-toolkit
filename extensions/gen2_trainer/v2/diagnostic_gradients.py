"""Real-model checkpoint and represented-weight input-gradient comparisons.

This module never optimizes the bank or changes a source package. A successful
comparison is bounded numerical evidence for one packet, not a proof of global
gradient correctness. CPU fixtures test the harness, not real GPU acceptance.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import gc
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..objectives import per_example_mse


LIMITATIONS = [
    "One unchanged real packet and the current learned bank; no global gradient guarantee.",
    "Checkpoint comparisons share the same quantized model and cannot establish full-precision model equivalence.",
    "Linear references use the same represented dequantized weights, a few captured rows, and one deterministic cotangent.",
    "Configured tolerances are diagnostic thresholds; repeat-baseline noise is reported without relaxing thresholds.",
    "An OOM, missing capture, nonfinite or disconnected result is incomplete, never a passing comparison.",
]


def _autocast():
    from toolkit.accelerator import get_accelerator
    return get_accelerator().autocast()


def _rng_state():
    return {"python": random.getstate(), "numpy": deepcopy(np.random.get_state()),
            "torch": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def _preserve_backend(backend):
    parameter = backend.tokens.E
    transformer = backend.model.transformer
    encoder_flag = backend.encoder_checkpointing
    dit_flag = getattr(transformer, "gradient_checkpointing", None)
    if type(encoder_flag) is not bool or type(dit_flag) is not bool:
        raise TypeError("Gradient diagnostic requires bool encoder_checkpointing and native transformer.gradient_checkpointing")
    state = _rng_state()
    values = parameter.detach().cpu().clone()
    previous_grad = parameter.grad
    grad_values = None if previous_grad is None else previous_grad.detach().clone()
    required = parameter.requires_grad
    try:
        parameter.requires_grad_(True)
        yield state, (encoder_flag, dit_flag)
    finally:
        # Each restoration is attempted even if another restoration fails.
        try:
            backend.encoder_checkpointing = encoder_flag
            transformer.gradient_checkpointing = dit_flag
            with torch.no_grad():
                parameter.copy_(values.to(parameter.device))
                if previous_grad is not None:
                    previous_grad.copy_(grad_values)
            parameter.grad = previous_grad
            parameter.requires_grad_(required)
        finally:
            _restore_rng(state)


def _statistics(tensor):
    value = tensor.detach().double().cpu().reshape(-1)
    finite = bool(torch.isfinite(value).all())
    return {"elements": value.numel(), "finite": finite,
            "l2": float(value.norm()) if finite else None,
            "max_abs": float(value.abs().max()) if finite and value.numel() else None,
            "nonzero_count": int(torch.count_nonzero(value)) if finite else None}


def _difference(reference, actual):
    reference = reference.detach().double().cpu().reshape(-1)
    actual = actual.detach().double().cpu().reshape(-1)
    if reference.shape != actual.shape:
        raise ValueError("Comparison tensor shapes differ")
    if not bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()):
        raise FloatingPointError("Nonfinite comparison tensor")
    delta = actual-reference
    nr, na = float(reference.norm()), float(actual.norm())
    cosine = float((reference @ actual)/(nr*na)) if nr > 0 and na > 0 else None
    return {"reference_l2": nr, "actual_l2": na, "difference_l2": float(delta.norm()),
            "relative_l2": float(delta.norm())/max(nr, 1e-30),
            "max_abs": float(delta.abs().max()) if delta.numel() else 0.,
            "cosine": max(-1., min(1., cosine)) if cosine is not None else None}


def _is_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
        and any(word in str(error).lower() for word in ("cuda", "cudnn", "cublas")))


def _attempt(operation):
    """Exception tracebacks die on return, before the caller empties CUDA cache."""
    try:
        return operation(), None
    except Exception as error:
        reason = "cuda_oom" if _is_oom(error) else (
            "nonfinite_or_disconnected" if isinstance(error, FloatingPointError) else "execution_error")
        return None, {"status": "inconclusive", "complete": False, "passed": None,
                      "reason": reason, "error_type": type(error).__name__, "error": str(error)}


def _release_failed_attempt(error):
    if error is not None:
        gc.collect()
        if error["reason"] == "cuda_oom" and torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def _memory(device):
    if device.type != "cuda":
        return None
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def _packet_on_device(packet, device):
    result = dict(packet)
    for key in ("zt", "tau", "target", "valid_mask"):
        if key in result and result[key] is not None:
            result[key] = result[key].to(device)
    count = len(result["qs"])
    if count < 1 or any(result[key].shape[0] != count for key in ("zt", "tau", "target")):
        raise ValueError("Gradient diagnostic packet has inconsistent example counts")
    return result


def _packet_gradient(backend, packet):
    with torch.enable_grad():
        with _autocast():
            conditioning = backend.encode(packet["qs"], mode="learned", gradients=True)
            prediction = backend.predict(packet["zt"], packet["tau"], conditioning)
        losses = per_example_mse(prediction, packet["target"], packet.get("valid_mask"))
        loss = losses.mean()
        if not bool(torch.isfinite(loss)) or not loss.requires_grad:
            raise FloatingPointError("Diagnostic loss is nonfinite or disconnected")
        gradient, = torch.autograd.grad(loss, backend.tokens.E, create_graph=False, retain_graph=False)
    if not bool(torch.isfinite(gradient).all()) or not bool((gradient != 0).any()):
        raise FloatingPointError("Diagnostic E gradient is nonfinite or entirely zero")
    return {"loss": float(loss.detach()), "per_example_losses": losses.detach().cpu().tolist(),
            "prediction": prediction.detach().cpu(), "gradient": gradient.detach().cpu(),
            "conditioning_lengths": [int(value.shape[0]) for value in conditioning.features]}


def _quantized(module):
    return (type(module).__module__.startswith(("optimum.quanto", "torchao"))
            or hasattr(module, "qweight") or bool(getattr(module.weight, "is_quantized", False)))


def _select_linears(model, count, family=None):
    # The native Qwen inner model can retain its unused vision tower. Restrict
    # capture to the decoder actually executed by qwen_features. Ideogram's
    # transformer block projections also all execute in its conditional path.
    scope, prefix = model, ""
    language = getattr(model, "language_model", None)
    if family == "encoder" and getattr(language, "layers", None) is not None:
        scope, prefix = language.layers, "language_model.layers."
    elif family == "transformer" and getattr(model, "layers", None) is not None:
        scope, prefix = model.layers, "layers."
    candidates = [(prefix+name, module) for name, module in scope.named_modules()
                  if isinstance(module, nn.Linear) or (
                      hasattr(module, "qweight") and getattr(getattr(module, "weight", None), "ndim", None) == 2)]
    # Prefer genuinely quantized projections. Plain nn.Linear remains useful
    # for an explicitly unquantized model and is labelled as such in evidence.
    candidates.sort(key=lambda pair: (not _quantized(pair[1]), pair[0]))
    if len(candidates) <= count:
        return candidates
    quantized = [pair for pair in candidates if _quantized(pair[1])]
    pool = quantized if len(quantized) >= count else candidates
    indices = [round(index*(len(pool)-1)/max(1, count-1)) for index in range(count)]
    return [pool[index] for index in indices]


def _plain(value):
    if bool(getattr(value, "is_quantized", False)) or type(value).__module__.startswith(("optimum.quanto", "torchao")):
        return value.dequantize()
    return value


@torch.no_grad()
def _sample_rows(value, max_rows):
    if not isinstance(value, torch.Tensor) or value.ndim < 1:
        raise TypeError("Linear diagnostic expected a Tensor activation")
    value = value.detach()
    shape = value.shape[:-1]
    rows = math.prod(shape)
    if rows < 1:
        raise ValueError("Linear diagnostic captured an empty activation")
    indices = torch.linspace(0, rows-1, min(max_rows, rows), device=value.device).long()
    coordinates, remaining = [], indices
    for size in reversed(shape):
        coordinates.append(remaining.remainder(size))
        remaining = remaining // size
    selected = value[tuple(reversed(coordinates))] if shape else value.unsqueeze(0)
    return _plain(selected.detach()).clone().cpu()


@contextmanager
def _capture_inputs(selected, samples, errors, row_count):
    handles = []
    try:
        for family, name, module in selected:
            key = (family, name)
            def capture(owner, args, kwargs, key=key):
                if key in samples or key in errors:
                    return
                try:
                    value = args[0] if args else kwargs.get("input", kwargs.get("x"))
                    samples[key] = _sample_rows(value, row_count)
                except Exception as error:
                    errors[key] = {"error_type": type(error).__name__, "error": str(error)}
            handles.append(module.register_forward_pre_hook(capture, with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _linear_reference(module, sample, seed):
    weight = module.qweight if hasattr(module, "qweight") else module.weight
    device = weight.device
    actual_input = sample.to(device).detach().requires_grad_(True)
    # Reference dequantization is limited to this one projection, after the
    # end-to-end graph has been released. Never dequantize an entire model.
    represented_weight = _plain(weight.detach()).to(dtype=actual_input.dtype)
    bias = getattr(module, "bias", None)
    represented_bias = None if bias is None else _plain(bias.detach()).to(dtype=actual_input.dtype)
    reference_input = actual_input.detach().clone().requires_grad_(True)
    with torch.enable_grad(), _autocast():
        actual = module(actual_input)
        expected = F.linear(reference_input, represented_weight, represented_bias)
    if actual.shape != expected.shape:
        raise ValueError("Native and reference linear output shapes differ")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cotangent = torch.randn(actual.shape, generator=generator, dtype=torch.float32).to(device, actual.dtype)
    actual_gradient, = torch.autograd.grad(actual, actual_input, grad_outputs=cotangent)
    reference_gradient, = torch.autograd.grad(expected, reference_input, grad_outputs=cotangent)
    if not bool((reference_gradient != 0).any()):
        raise FloatingPointError("Reference input gradient is entirely zero; cotangent probe is inconclusive")
    return {"output": _difference(expected, actual),
            "input_gradient": _difference(reference_gradient, actual_gradient),
            "weight_shape": list(weight.shape), "input_shape": list(sample.shape),
            "compute_dtype": str(actual_input.dtype), "cotangent_seed": seed,
            "reference": "F.linear with the same represented dequantized frozen weight and bias"}


def _settings(settings):
    defaults = {"gradient_cosine_min": .999, "gradient_relative_l2_max": .02,
                "prediction_relative_l2_max": .02, "loss_relative_error_max": .02,
                "linear_modules_per_model": 2, "linear_input_rows": 4, "linear_seed": 1729}
    result = {key: settings.get(key, value) for key, value in defaults.items()}
    for key in ("gradient_cosine_min", "gradient_relative_l2_max", "prediction_relative_l2_max", "loss_relative_error_max"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid gradient diagnostic setting {key}")
    if not -1 <= result["gradient_cosine_min"] <= 1 or any(result[key] < 0 for key in (
            "gradient_relative_l2_max", "prediction_relative_l2_max", "loss_relative_error_max")):
        raise ValueError("Gradient diagnostic comparison tolerances are outside their valid ranges")
    for key, lower, upper in (("linear_modules_per_model", 1, 2), ("linear_input_rows", 1, 32), ("linear_seed", 0, 2**63-1)):
        if type(result[key]) is not int or not lower <= result[key] <= upper:
            raise ValueError(f"Gradient diagnostic {key} must be an integer in [{lower}, {upper}]")
    return result


def _gradient_passed(metrics, settings):
    return (metrics["cosine"] is not None and metrics["cosine"] >= settings["gradient_cosine_min"]
            and metrics["relative_l2"] <= settings["gradient_relative_l2_max"])


def _write(recorder, record):
    if recorder is not None:
        recorder.record("gradient_fidelity", record)


def _progress(message):
    print(f"[gen2 diagnostic gradients] {message}", flush=True)


def run_gradient_diagnostics(backend, packet, settings, recorder):
    """Compare actual native checkpoint paths and selected projection gradients.

    ``packet`` is an already selected, unchanged CPU training packet containing
    qs, zt, tau, target, id and metadata. No captions or latent maps are resized.
    Every comparison starts with identical RNG and E. All original E values,
    its gradient object/contents, requires_grad, flags and RNG are restored.
    """
    settings = _settings(settings)
    rows, linear_rows = [], []
    started = time.perf_counter()
    with _preserve_backend(backend) as (rng, configured):
        backend.assert_frozen()
        device = backend.tokens.E.device
        batch = _packet_on_device(packet, device)
        count = settings["linear_modules_per_model"]
        # At most four capture candidates per family, then at most two actual
        # comparisons. A candidate not reached in the packet can be replaced
        # only by a candidate whose real input was captured in this same pass.
        candidates = [(family, name, module) for family, model in (
            ("encoder", backend.model.text_encoder), ("transformer", backend.model.transformer))
            for name, module in _select_linears(model, count*2, family)]
        samples, capture_errors = {}, {}
        variants = [("configured", *configured), ("configured_repeat", *configured),
                    ("encoder_off", False, configured[1]), ("dit_off", configured[0], False),
                    ("both_off", False, False)]
        baseline = None
        for index, (name, encoder, dit) in enumerate(variants):
            _restore_rng(rng)
            backend.encoder_checkpointing = encoder
            backend.model.transformer.gradient_checkpointing = dit
            _progress(f"checkpoint {index+1}/{len(variants)} {name}: starting "
                      f"(encoder={encoder}, DiT={dit})")
            if recorder is not None:
                recorder.event("gradient_variant_started", variant=name, packet_id=packet.get("id"))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            attempt_started = time.perf_counter()
            def execute():
                if index == 0:
                    with _capture_inputs(candidates, samples, capture_errors, settings["linear_input_rows"]):
                        return _packet_gradient(backend, batch)
                return _packet_gradient(backend, batch)
            output, error = _attempt(execute)
            _release_failed_attempt(error)
            row = {"kind": "checkpoint", "variant": name, "packet_id": packet.get("id"),
                   "encoder_checkpointing": encoder, "dit_checkpointing": dit,
                   "seconds": time.perf_counter()-attempt_started, "memory": _memory(device)}
            if error is not None:
                row.update(error)
            else:
                row.update({"loss": output["loss"], "per_example_losses": output["per_example_losses"],
                            "gradient": _statistics(output["gradient"]),
                            "gradient_dtype": str(output["gradient"].dtype),
                            "conditioning_lengths": output["conditioning_lengths"], "complete": True})
                if index == 0:
                    baseline = output
                    row.update(status="reference", passed=None)
                elif baseline is None:
                    row.update(status="inconclusive", complete=False, passed=None, reason="configured_reference_unavailable")
                else:
                    gradient = _difference(baseline["gradient"], output["gradient"])
                    prediction = _difference(baseline["prediction"], output["prediction"])
                    loss_delta = abs(output["loss"]-baseline["loss"])
                    loss_relative = loss_delta/max(abs(baseline["loss"]), 1e-30)
                    passed = (_gradient_passed(gradient, settings)
                              and prediction["relative_l2"] <= settings["prediction_relative_l2_max"]
                              and loss_relative <= settings["loss_relative_error_max"])
                    row.update(status="passed" if passed else "failed", passed=passed,
                               reference="configured", gradient_difference=gradient,
                               prediction_difference=prediction, loss_absolute_difference=loss_delta,
                               loss_relative_difference=loss_relative)
                del output
            rows.append(row)
            _progress(f"checkpoint {index+1}/{len(variants)} {name}: {row['status']} "
                      f"in {row['seconds']:.1f}s" + (f" ({row['reason']})" if "reason" in row else ""))
            _write(recorder, row)
        # No end-to-end autograd graph remains; only CPU reference tensors and
        # a tiny detached activation sample per selected projection are held.
        selected = []
        for family in ("encoder", "transformer"):
            family_candidates = [item for item in candidates if item[0] == family]
            captured = [item for item in family_candidates if item[:2] in samples]
            missing = [item for item in family_candidates if item[:2] not in samples]
            # Prefer reached projections, retaining missing placeholders only
            # when not enough real captures exist. Untested modules never pass.
            selected.extend((captured+missing)[:count])
            if not family_candidates:
                row = {"kind": "linear", "family": family, "status": "inconclusive", "complete": False,
                       "passed": None, "reason": "no_linear_modules_found"}
                linear_rows.append(row)
                _progress(f"linear {family}: inconclusive (no_linear_modules_found)")
                _write(recorder, row)
        for index, (family, name, module) in enumerate(selected):
            _progress(f"linear {index+1}/{len(selected)} {family}.{name}: starting")
            linear_started = time.perf_counter()
            key = (family, name)
            row = {"kind": "linear", "family": family, "module": name, "packet_id": packet.get("id"),
                   "module_class": f"{type(module).__module__}.{type(module).__name__}",
                   "quantized": _quantized(module)}
            if key not in samples:
                row.update(status="inconclusive", complete=False, passed=None, reason="real_input_capture_unavailable",
                           capture_error=capture_errors.get(key))
            else:
                _restore_rng(rng)
                output, error = _attempt(lambda: _linear_reference(module, samples[key], settings["linear_seed"]))
                _release_failed_attempt(error)
                if error is not None:
                    row.update(error)
                else:
                    passed = (_gradient_passed(output["input_gradient"], settings)
                              and output["output"]["relative_l2"] <= settings["prediction_relative_l2_max"])
                    row.update(output, status="passed" if passed else "failed", complete=True, passed=passed)
            row["seconds"] = time.perf_counter()-linear_started
            linear_rows.append(row)
            _progress(f"linear {index+1}/{len(selected)} {family}.{name}: {row['status']} "
                      f"in {row['seconds']:.1f}s" + (f" ({row['reason']})" if "reason" in row else ""))
            _write(recorder, row)
        backend.assert_frozen()
    all_rows = rows+linear_rows
    complete = all(row["complete"] for row in all_rows)
    failed = any(row["status"] == "failed" for row in all_rows)
    status = "failed" if failed else "passed" if complete else "inconclusive"
    summary = {"status": status, "complete": complete, "passed": False if failed else True if complete else None,
               "packet_id": packet.get("id"), "settings": settings, "checkpoint_variants": rows,
               "linear_comparisons": linear_rows, "seconds": time.perf_counter()-started,
               "linear_capture_candidates": [{"family": family, "module": name,
                    "captured": (family, name) in samples,
                    "selected": any(item[:2] == (family, name) for item in selected)}
                    for family, name, _ in candidates],
               "limitations": LIMITATIONS, "optimizer_steps": 0, "state_restored": True}
    if recorder is not None:
        recorder.event("gradient_diagnostics_complete", status=status, complete=complete, packet_id=packet.get("id"))
    return summary
