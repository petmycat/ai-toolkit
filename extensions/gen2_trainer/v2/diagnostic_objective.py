"""Matched-packet conditioning comparisons and reversible token-only descent.

This measures a fixed denoising objective. It makes no claim about visual style
quality and never advances a training engine or writes a model checkpoint.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import math

import torch

from ..diagnostics import isolated_rng
from ..objectives import per_example_mse


def _stats(value):
    value = value.detach().double()
    finite = bool(torch.isfinite(value).all())
    if not finite:
        raise FloatingPointError("Nonfinite objective diagnostic tensor")
    return {"finite": True, "elements": value.numel(), "l2": float(value.norm()),
            "rms": float(value.square().mean().sqrt()), "max_abs": float(value.abs().max()),
            "nonzero_count": int(torch.count_nonzero(value))}


def _frozen(backend):
    backend.assert_frozen()
    if any(p.requires_grad or p.grad is not None for p in backend.frozen_parameters()):
        raise RuntimeError("Objective diagnostics require frozen native parameters with no gradients")


def _prepare(backend, packets, modes, phrase):
    if not isinstance(packets, (list, tuple)) or not packets:
        raise ValueError("Objective diagnostics need a nonempty fixed packet list")
    prepared, identifiers, comparisons = [], set(), {}
    for index, packet in enumerate(packets):
        qs = packet.get("qs")
        if not isinstance(qs, (list, tuple)) or not qs or any(not isinstance(q, str) for q in qs):
            raise ValueError("Each diagnostic packet needs nonempty caption strings")
        count = len(qs)
        zt, tau, target = (packet[key] for key in ("zt", "tau", "target"))
        if any(not torch.is_tensor(v) for v in (zt, tau, target)):
            raise ValueError("Diagnostic zt, tau and target must be tensors")
        if zt.shape != target.shape or zt.ndim < 2 or zt.shape[0] != count or tau.shape != (count,):
            raise ValueError("Diagnostic packet tensor/caption shapes disagree")
        if any(not bool(torch.isfinite(v).all()) for v in (zt, tau, target)):
            raise FloatingPointError("Nonfinite fixed diagnostic packet")
        if bool(((tau <= 0) | (tau > 1)).any()):
            raise ValueError("Diagnostic tau must be in (0,1]")
        metadata = packet.get("metadata", [{} for _ in qs])
        if len(metadata) != count:
            raise ValueError("Diagnostic metadata count differs from caption count")
        identifier = str(packet.get("id", packet.get("packet_id", index)))
        if identifier in identifiers:
            raise ValueError("Diagnostic packet identifiers must be unique")
        identifiers.add(identifier)
        compiled = {mode: [] for mode in modes}
        for q in qs:
            if q not in comparisons:
                comparisons[q] = backend.compiler.comparison(q, named_phrase=phrase)
            comparison = comparisons[q]
            hashes = []
            for mode in modes:
                item = comparison[mode]
                info = item.metadata
                digest = info.get("common_ordinary_content_sha256")
                if (not isinstance(digest, str) or not digest or not info.get("shared_comparison_content")
                        or info.get("mode") != mode or info.get("original_caption") != q):
                    raise ValueError("Invalid or unaligned compiled diagnostic comparison")
                hashes.append(digest)
                compiled[mode].append(item)
            if len(set(hashes)) != 1:
                raise ValueError("Diagnostic modes retain different ordinary caption content")
        prepared.append({"source": packet, "qs": list(qs), "metadata": metadata,
                         "id": identifier, "compiled": compiled})
    return prepared


def _tensors(packet, device):
    source = packet["source"]
    return {key: source[key].detach().to(device) for key in ("zt", "tau", "target")}


def _autocast(backend):
    if backend.tokens.E.device.type == "cpu":
        return nullcontext()  # Bounded numerical fixtures have no native Accelerator.
    from toolkit.accelerator import get_accelerator
    return get_accelerator().autocast()


def _predict(backend, packet, values, mode, gradients):
    # Match V2Engine's native forward precision context; reductions stay FP32
    # outside autocast, and descent's FP32 token gradients need no AMP scaler.
    with _autocast(backend):
        condition = backend.encode(packet["qs"], mode=mode, gradients=gradients,
                                   compiled=packet["compiled"][mode])
        prediction = backend.predict(values["zt"], values["tau"], condition)
    losses = per_example_mse(prediction, values["target"], _mask(packet, prediction.device))
    if not bool(torch.isfinite(losses).all()):
        raise FloatingPointError("Nonfinite paired diagnostic loss")
    return prediction, losses


def _mask(packet, device):
    mask = packet["source"].get("valid_mask")
    return mask.detach().to(device) if mask is not None else None


def _loss_summary(rows, modes):
    count = len(rows)
    mean = {mode: sum(row["loss_by_mode"][mode] for row in rows)/count for mode in modes}
    comparisons = {f"learned_minus_{mode}": mean["learned"]-mean[mode]
                   for mode in modes if mode != "learned"}
    return {"example_count": count, "mean_loss_by_mode": mean,
            "paired_mean_loss_delta": comparisons,
            "learned_lower_loss_example_counts": {
                mode: sum(row["loss_by_mode"]["learned"] < row["loss_by_mode"][mode] for row in rows)
                for mode in modes if mode != "learned"}}


def _paired(backend, packets, modes, recorder):
    rows = []
    for index, packet in enumerate(packets, 1):
        print(f"[Gen2 v2 diagnostic] paired packet {index}/{len(packets)} | {packet['id']}", flush=True)
        values = _tensors(packet, backend.tokens.E.device)
        losses, differences = {}, {}
        with torch.no_grad():
            anchor, learned = _predict(backend, packet, values, "learned", False)
            losses["learned"] = learned.cpu().tolist()
            for mode in modes:
                if mode == "learned":
                    continue
                prediction, loss = _predict(backend, packet, values, mode, False)
                losses[mode] = loss.cpu().tolist()
                mse = per_example_mse(prediction, anchor, _mask(packet, prediction.device))
                if not bool(torch.isfinite(mse).all()):
                    raise FloatingPointError("Nonfinite diagnostic prediction difference")
                differences[mode] = mse.cpu().tolist()
                del prediction, loss, mse
            del anchor, learned
        _frozen(backend)
        example_rows = []
        for i, metadata in enumerate(packet["metadata"]):
            compiled = packet["compiled"]["learned"][i].metadata
            resolution = metadata.get("resolution", metadata.get("dataset_resolution",
                packet["source"].get("resolution")))
            if resolution is None:
                resolution = "latent:" + "x".join(str(x) for x in values["zt"].shape[2:])
            row = {"packet_id": packet["id"], "example_index": i,
                "sample_id": metadata.get("sample_id", metadata.get("example_id")),
                "tau": float(values["tau"][i]), "resolution": resolution,
                "latent_shape": list(values["zt"].shape[1:]),
                "common_ordinary_content_sha256": compiled["common_ordinary_content_sha256"],
                "resulting_lengths": {mode: packet["compiled"][mode][i].metadata.get("resulting_length") for mode in modes},
                "loss_by_mode": {mode: losses[mode][i] for mode in modes},
                "prediction_difference_to_learned": {
                    mode: {"mse": values[i], "rms": math.sqrt(values[i])}
                    for mode, values in differences.items()}}
            row["paired_loss_delta"] = {f"learned_minus_{mode}": losses["learned"][i]-losses[mode][i]
                                        for mode in modes if mode != "learned"}
            example_rows.append(row)
        payload = {"packet_id": packet["id"], **_loss_summary(example_rows, modes), "examples": example_rows}
        recorder.record("paired_losses", payload)
        rows.extend(example_rows)
        del values
    groups = {}
    for label in ("tau", "resolution", "tau_resolution"):
        grouped = defaultdict(list)
        for row in rows:
            tau = format(row["tau"], ".8g")
            key = tau if label == "tau" else str(row["resolution"]) if label == "resolution" else f"{tau}|{row['resolution']}"
            grouped[key].append(row)
        groups[f"by_{label}"] = {key: _loss_summary(items, modes) for key, items in grouped.items()}
    return {**_loss_summary(rows, modes), **groups}


def _fixed_objective(backend, packets, *, backward):
    total = sum(len(packet["qs"]) for packet in packets)
    loss_sum = 0.
    for packet in packets:
        values = _tensors(packet, backend.tokens.E.device)
        with torch.set_grad_enabled(backward):
            prediction, losses = _predict(backend, packet, values, "learned", backward)
            objective = losses.sum()/total
            if backward:
                if not objective.requires_grad:
                    raise RuntimeError("Controlled diagnostic objective is disconnected from learned tokens")
                objective.backward()
        loss_sum += sum(losses.detach().cpu().tolist())
        del prediction, losses, objective, values
    _frozen(backend)
    return loss_sum/total


def _optimizer_options(settings):
    source = settings["optimizer"]
    if source.get("type") != "adamw":
        raise ValueError("Controlled objective diagnostics require explicit source optimizer.type: adamw")
    options = {key: float(source[key]) for key in ("lr", "eps", "weight_decay")}
    options["betas"] = tuple(float(value) for value in source["betas"])
    if (any(not math.isfinite(options[key]) for key in ("lr", "eps", "weight_decay"))
            or options["lr"] <= 0 or options["eps"] <= 0 or options["weight_decay"] < 0
            or len(options["betas"]) != 2 or any(not math.isfinite(v) or not 0 <= v < 1 for v in options["betas"])):
        raise ValueError("Invalid controlled-descent AdamW settings")
    return options


def _select_descent_packets(prepared, settings):
    requested = settings.get("descent_packet_ids")
    if requested is not None:
        if (not isinstance(requested, (list, tuple)) or not requested
                or len(set(map(str, requested))) != len(requested)):
            raise ValueError("descent_packet_ids must be a nonempty list of unique identifiers")
        by_id = {packet["id"]: packet for packet in prepared}
        if any(str(identifier) not in by_id for identifier in requested):
            raise ValueError("A requested descent packet is absent from the fixed packet list")
        return [by_id[str(identifier)] for identifier in requested]
    count = min(settings["descent_packet_count"], len(prepared))
    indices = ([len(prepared)//2] if count == 1 else
               [round(index*(len(prepared)-1)/(count-1)) for index in range(count)])
    return [prepared[index] for index in indices]


def _descent(backend, packets, settings, starts, recorder):
    options = _optimizer_options(settings)
    results = {}
    parameter = backend.tokens.E
    for label, start in starts.items():
        with torch.no_grad():
            parameter.copy_(start)
        parameter.requires_grad_(True)
        parameter.grad = None
        optimizer = torch.optim.AdamW([parameter], **options)
        before = initial_loss = _fixed_objective(backend, packets, backward=False)
        recorder.record("descent", {"start": label, "step": 0, "loss": before,
            "packet_ids": [p["id"] for p in packets], "example_count": sum(len(p["qs"]) for p in packets),
            "optimizer_class": "torch.optim.AdamW", "optimizer_state": "fresh", "optimizer_settings": options})
        print(f"[Gen2 v2 diagnostic] descent {label} | step 0/{settings['descent_steps']} | loss {before:.9g}", flush=True)
        for step in range(1, settings["descent_steps"]+1):
            optimizer.zero_grad(set_to_none=True)
            backward_loss = _fixed_objective(backend, packets, backward=True)
            if parameter.grad is None:
                raise RuntimeError("Controlled diagnostic objective produced no token gradient")
            gradient = _stats(parameter.grad)
            previous = parameter.detach().clone()
            optimizer.step()
            update = _stats(parameter.detach()-previous)
            parameters = _stats(parameter)
            for state in optimizer.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        _stats(value)
            after = _fixed_objective(backend, packets, backward=False)
            recorder.record("descent", {"start": label, "step": step,
                "loss_before": before, "backward_objective": backward_loss, "loss": after,
                "loss_delta_from_previous": after-before, "loss_delta_from_start": after-initial_loss,
                "gradients": gradient, "parameter_update": update,
                "parameters": parameters,
                "movement_from_start": _stats(parameter.detach()-start)})
            print(f"[Gen2 v2 diagnostic] descent {label} | step {step}/{settings['descent_steps']} | loss {after:.9g}", flush=True)
            before = after
        results[label] = {"initial_loss": initial_loss, "final_loss": before,
                          "loss_delta": before-initial_loss, "steps": settings["descent_steps"],
                          "movement_from_start": _stats(parameter.detach()-start)}
        del optimizer
    return results


def run_objective_diagnostics(backend, packets, settings, recorder):
    """Measure matched conditions, then restore tokens/gradients/RNG exactly.

    ``packets`` own fixed qs/zt/tau/target/metadata; CPU tensors are transferred
    one packet at a time. Descent averages examples, including uneven batches.
    Settings include named_phrase, optimizer settings, descent_steps and
    descent_packet_count (defaults 12 and 4), and an optional isolated RNG seed.
    The temporary optimizer is always fresh torch.AdamW, not a resumed optimizer.
    """
    settings = dict(settings)
    settings.setdefault("descent_steps", 12)
    settings.setdefault("descent_packet_count", 4)
    if (type(settings["descent_steps"]) is not int or settings["descent_steps"] < 0
            or type(settings["descent_packet_count"]) is not int or settings["descent_packet_count"] < 1):
        raise ValueError("Diagnostic descent steps must be nonnegative; packet count must be positive")
    parameter, initial = backend.tokens.E, backend.tokens.initial
    if parameter.dtype != torch.float32 or initial.shape != parameter.shape:
        raise ValueError("Objective diagnostics require FP32 token masters and matching initial bank")
    saved, saved_initial = parameter.detach().clone(), initial.detach().clone()
    saved_grad = parameter.grad
    grad_values = None if saved_grad is None else saved_grad.detach().clone()
    required, initial_required = parameter.requires_grad, initial.requires_grad
    phrase = settings.get("named_phrase")
    modes = ["base", "init", "learned"] + (["named"] if phrase else [])
    try:
        with isolated_rng(settings.get("seed")):
            _frozen(backend)
            prepared = _prepare(backend, packets, modes, phrase)
            selected = _select_descent_packets(prepared, settings)
            _optimizer_options(settings)
            paired = _paired(backend, prepared, modes, recorder)
            descent = _descent(backend, selected, settings, {"initial": saved_initial, "learned": saved}, recorder)
            _frozen(backend)
            result = {"diagnostic": "fixed_packet_objective", "completed": True,
                "style_acceptance": "not_measured", "packet_count": len(prepared), "modes": modes, "paired": paired,
                "descent": descent, "descent_packet_ids": [p["id"] for p in selected],
                "optimizer": {"class": "torch.optim.AdamW", "state": "fresh", "settings": _optimizer_options(settings)},
                "observations": {"paired_differences_use_identical_latents_noise_targets": True,
                    "ordinary_caption_content_aligned": True, "descent_uses_fixed_examples": True,
                    "source_training_optimizer_used": False, "visual_acceptance": "not_assessed"}}
    finally:
        with torch.no_grad():
            parameter.copy_(saved)
            initial.copy_(saved_initial)
            if saved_grad is not None:
                saved_grad.copy_(grad_values)
        parameter.grad = saved_grad
        parameter.requires_grad_(required)
        initial.requires_grad_(initial_required)
    result["source_token_state_restored"] = True
    return result
