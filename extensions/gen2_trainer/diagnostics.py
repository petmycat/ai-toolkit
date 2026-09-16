"""Detached diagnostics and repeatable exports; none of these are training losses."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.metadata
import io
import itertools
import json
import math
import os
import platform
import random
import statistics
import subprocess
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from .recording import (STREAMS, assert_writable_path, iter_records, json_safe,
                        sha256, write_json)


def capture_rng_state(generators: Mapping[str, torch.Generator] | None = None) -> dict:
    """All global streams plus explicitly registered diagnostic/preview streams.

    NumPy's uint32 array is represented as a tensor so restricted torch loading
    never requires arbitrary NumPy pickle globals.
    """
    import numpy as np
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"algorithm": state[0], "keys": torch.tensor(state[1].astype("int64")),
                      "position": int(state[2]), "has_gauss": int(state[3]), "cached_gaussian": float(state[4])},
            "torch_cpu": torch.random.get_rng_state().clone(),
            "torch_cuda": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
            "generators": {name: {"device": str(generator.device), "state": generator.get_state().cpu().clone()}
                           for name, generator in (generators or {}).items()}}


def restore_rng_state(state: Mapping, generators: Mapping[str, torch.Generator] | None = None):
    import numpy as np
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy["algorithm"], numpy["keys"].cpu().numpy().astype("uint32"),
                         numpy["position"], numpy["has_gauss"], numpy["cached_gaussian"]))
    torch.random.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG state requires the original CUDA device configuration")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    saved = state.get("generators", {})
    if generators is not None:
        if set(generators) != set(saved):
            raise ValueError("Named RNG streams differ from saved state")
        for name, generator in generators.items():
            if str(generator.device) != saved[name]["device"]:
                raise ValueError(f"RNG device mismatch: {name}")
            generator.set_state(saved[name]["state"].cpu())


@contextlib.contextmanager
def isolated_rng(seed: int | None = None, generators=None):
    """Diagnostic/preview execution never changes a subsequent training draw."""
    saved = capture_rng_state(generators)
    try:
        if seed is not None:
            import numpy as np
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
        yield
    finally:
        restore_rng_state(saved, generators)


@torch.no_grad()
def tensor_statistics(tensor: torch.Tensor) -> dict:
    value = tensor.detach().float()
    total = value.numel()
    if total == 0:
        return {"count": 0, "rms": None, "l2": None, "max_abs": None,
                "finite_fraction": None, "nonzero_fraction": None, "reason": "empty_tensor"}
    finite = torch.isfinite(value)
    finite_count = int(finite.sum().item())
    valid = finite_count == total
    return {"count": total, "rms": float(value.square().mean().sqrt().item()) if valid else None,
            "l2": float(torch.linalg.vector_norm(value).item()) if valid else None,
            "max_abs": float(value.abs().max().item()) if valid else None,
            "finite_fraction": finite_count / total,
            "nonzero_fraction": int(((value != 0) & finite).sum().item()) / total,
            "reason": None if valid else "nonfinite_tensor"}


@torch.no_grad()
def gradient_statistics(parameters: Sequence[torch.Tensor]) -> dict:
    parameters = list(parameters)
    total = sum(parameter.numel() for parameter in parameters)
    count = finite_count = nonzero = missing = 0
    square_sum = 0.0
    maximum = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            missing += 1
            continue
        grad = parameter.grad.detach()
        if grad.is_sparse:
            grad = grad.to_dense()
        value = grad.float()
        finite = torch.isfinite(value)
        count += value.numel()
        finite_count += int(finite.sum().item())
        nonzero += int(((value != 0) & finite).sum().item())
        if bool(finite.all()):
            # Float64 scalar accumulation avoids overflow for large families.
            square_sum += float(value.double().square().sum().item())
            maximum = max(maximum, float(value.abs().max().item())) if value.numel() else maximum
    valid = count > 0 and finite_count == count
    norm = math.sqrt(square_sum) if valid else None
    return {"method": "exact", "parameter_count": total, "gradient_elements": count,
            "gradient_tensors": len(parameters) - missing, "missing_gradient_tensors": missing,
            "l2": norm, "norm": norm, "rms": math.sqrt(square_sum / total) if valid and total else None,
            "max_abs": maximum if valid else None,
            "finite_fraction": finite_count / count if count else None,
            "nonzero_fraction": nonzero / total if total and count else None,
            "reason": None if valid else ("no_gradients" if not count else "nonfinite_gradients")}


def coordinate_indices(total: int, count: int, seed: int) -> torch.Tensor:
    """Uniform sampling without replacement in O(sample_size) host storage."""
    if total < 0 or count < 0:
        raise ValueError("Coordinate counts cannot be negative")
    count = total if count == 0 else min(total, count)
    if count == total:
        return torch.arange(total, dtype=torch.int64)
    return torch.tensor(sorted(random.Random(seed).sample(range(total), count)), dtype=torch.int64)


@torch.no_grad()
def sample_coordinates(parameters: Sequence[torch.Tensor], indices: torch.Tensor,
                       gradients: bool = False) -> torch.Tensor:
    parameters = list(parameters)
    indices = indices.detach().cpu().long()
    total = sum(parameter.numel() for parameter in parameters)
    if indices.ndim != 1 or (indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= total)):
        raise ValueError("Coordinates are outside the flattened parameter family")
    output = torch.zeros(indices.numel(), dtype=torch.float32)
    offset = 0
    for parameter in parameters:
        selected = (indices >= offset) & (indices < offset + parameter.numel())
        if bool(selected.any()):
            tensor = parameter.grad if gradients else parameter
            if tensor is not None:
                if tensor.is_sparse:
                    tensor = tensor.to_dense()
                positions = (indices[selected] - offset).to(tensor.device)
                output[selected] = tensor.detach().reshape(-1)[positions].float().cpu()
        offset += parameter.numel()
    return output


def sampled_update_statistics(before: torch.Tensor, after: torch.Tensor, total: int) -> dict:
    if before.shape != after.shape or before.numel() > total:
        raise ValueError("Mismatched parameter coordinate snapshots")
    difference = after.detach().double() - before.detach().double()
    sampled = difference.numel()
    if not sampled:
        return {"method": "sampled", "sample_size": 0, "total_coordinates": total,
                "l2": None, "reason": "no_coordinates"}
    squared = float(difference.square().sum()) * total / sampled
    reference = float(before.detach().double().square().sum()) * total / sampled
    return {"method": "exact" if sampled == total else "sampled", "sample_size": sampled,
            "total_coordinates": total, "coverage": sampled / total,
            "l2": math.sqrt(squared), "rms": math.sqrt(squared / total),
            "reference_l2": math.sqrt(reference),
            "relative_l2": math.sqrt(squared / reference) if reference > 0 else None,
            "relative_reason": None if reference > 0 else "zero_reference_norm"}


@torch.no_grad()
def tensor_hash(tensor: torch.Tensor) -> str:
    """Full values, shape and dtype; never a sampled hash."""
    digest = hashlib.sha256()
    value = tensor.detach()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    if value.is_quantized:
        digest.update(str(value.qscheme()).encode())
        if value.qscheme() in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            digest.update(repr((value.q_scale(), value.q_zero_point())).encode())
        else:
            digest.update(tensor_hash(value.q_per_channel_scales()).encode())
            digest.update(tensor_hash(value.q_per_channel_zero_points()).encode())
            digest.update(str(value.q_per_channel_axis()).encode())
        value = value.int_repr()
    flat = value.reshape(-1)
    for offset in range(0, flat.numel(), 1024 * 1024):
        chunk = flat[offset:offset + 1024 * 1024].cpu().contiguous()
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def module_hash(module) -> str:
    digest = hashlib.sha256()
    state = module.state_dict() if hasattr(module, "state_dict") else module
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_hash(state[name]).encode("ascii"))
    return digest.hexdigest()


@torch.no_grad()
def _cosine(first, second):
    a, b = first.detach().float().reshape(-1), second.detach().float().reshape(-1)
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if not bool(torch.isfinite(denominator)) or float(denominator) == 0:
        return None
    return float((torch.dot(a, b) / denominator).clamp(-1, 1).item())


@torch.no_grad()
def region_summaries(inputs: torch.Tensor, base: torch.Tensor, residual: torch.Tensor,
                     applied: torch.Tensor, regions: Mapping[str, torch.Tensor]) -> list[dict]:
    """Actual packing masks define regions; omit regions with no actual tokens."""
    if base.shape != residual.shape or base.shape != applied.shape:
        raise ValueError("Residual and frozen-output shapes differ")
    suffix_mask = regions.get("suffix", regions.get("learned_suffix"))
    if suffix_mask is not None:
        outside = ~suffix_mask.detach().to(device=applied.device, dtype=torch.bool)
        outside_values = applied.detach()[outside]
        leakage = tensor_statistics(outside_values)
        leakage_max, leakage_rms = leakage["max_abs"], leakage["rms"]
        leakage_reason = leakage["reason"]
    else:
        leakage_max = leakage_rms = None
        leakage_reason = "suffix_mask_not_supplied"
    summaries = []
    for name, mask in regions.items():
        mask = mask.detach().to(device=base.device, dtype=torch.bool)
        if tuple(mask.shape) != tuple(base.shape[:-1]):
            raise ValueError(f"Region mask shape differs for {name}")
        count = int(mask.sum().item())
        if not count:
            continue
        x, b, a, g = inputs.detach()[mask], base.detach()[mask], residual.detach()[mask], applied.detach()[mask]
        bs, rs, gs = tensor_statistics(b), tensor_statistics(a), tensor_statistics(g)
        denominator = bs["rms"]
        cosine = _cosine(a, b)
        numerator = a.float().square().mean(-1)
        base_power = b.float().square().mean(-1)
        def power_summary(value):
            return {"count": value.numel(), "mean": float(value.mean()),
                    "minimum": float(value.min()), "maximum": float(value.max()),
                    "standard_deviation": float(value.std(unbiased=False))}
        summaries.append({"token_region": name, "token_count": count,
                          "input_rms": tensor_statistics(x)["rms"], "base_rms": denominator,
                          "residual_rms": rs["rms"], "applied_rms": gs["rms"],
                          "residual_base_ratio": rs["rms"] / denominator if denominator else None,
                          "residual_base_ratio_reason": None if denominator else "zero_base_norm",
                          "applied_base_ratio": gs["rms"] / denominator if denominator else None,
                          "residual_base_cosine": cosine,
                          "cosine_reason": None if cosine is not None else "zero_or_nonfinite_norm",
                          "residual_power": power_summary(numerator),
                          "base_power": power_summary(base_power),
                          "normalized_residual_power": power_summary(numerator / (base_power + 1e-6))})
        summaries[-1].update({"outside_suffix_applied_max_abs": leakage_max,
                              "outside_suffix_applied_rms": leakage_rms,
                              "outside_suffix_reason": leakage_reason})
    return summaries


@torch.no_grad()
def low_rank_spectrum(A: torch.Tensor, B: torch.Tensor, scale: float = 1.0) -> dict:
    """Thin QR then small SVD: no dense B@A allocation."""
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]:
        raise ValueError("Expected A(rank,in), B(out,rank)")
    _, rb = torch.linalg.qr(B.detach().float(), mode="reduced")
    _, ra = torch.linalg.qr(A.detach().float().T, mode="reduced")
    singular = torch.linalg.svdvals(rb @ ra.T) * abs(float(scale))
    energy = singular.double().square()
    total = float(energy.sum().item())
    if total:
        probabilities = energy / total
        positive = probabilities[probabilities > 0]
        rank95 = int(torch.searchsorted(probabilities.cumsum(0),
                                       probabilities.new_tensor(0.95)).item()) + 1
        entropy_rank = float((-(positive * positive.log()).sum()).exp().item())
    else:
        rank95, entropy_rank = None, None
    return {"singular_values": singular.cpu().tolist(), "frobenius_norm": math.sqrt(total),
            "largest_singular_value": float(singular[0]) if singular.numel() else None,
            "rank_95_squared_energy": rank95, "entropy_effective_rank": entropy_rank,
            "energy_convention": "squared_singular_values", "alpha_rank_scale": float(scale),
            "reason": None if total else "all_zero_update"}


@torch.no_grad()
def token_statistics(U: torch.Tensor, e: torch.Tensor, e_init: torch.Tensor) -> dict:
    raw, effective, initial = (item.detach().float() for item in (U, e, e_init))
    if raw.shape != effective.shape or raw.shape != initial.shape:
        raise ValueError("Token initialization/current shapes differ")
    raw_norms = torch.linalg.vector_norm(raw, dim=-1)
    norms = torch.linalg.vector_norm(effective, dim=-1)
    pairwise = []
    for first in effective:
        pairwise.append([_cosine(first, second) for second in effective])
    return {"raw_norms": raw_norms.cpu().tolist(), "effective_norms": norms.cpu().tolist(),
            "cosine_to_initial": [_cosine(current, start) for current, start in zip(effective, initial)],
            "pairwise_cosines": pairwise,
            "effective_singular_values": torch.linalg.svdvals(effective).cpu().tolist()}


@torch.no_grad()
def prefix_comparison(native: torch.Tensor, styled: torch.Tensor, atol: float = 0.001,
                      rtol: float = 0.01, baseline: torch.Tensor | None = None) -> dict:
    if native.shape != styled.shape:
        raise ValueError("Prefix comparisons require equal original-position shapes")
    reference = native.detach().float()
    difference = styled.detach().float() - reference
    result = {"rms_difference": tensor_statistics(difference)["rms"],
              "max_abs_difference": tensor_statistics(difference)["max_abs"],
              "reference_rms": tensor_statistics(reference)["rms"],
              "atol": atol, "rtol": rtol,
              "within_tolerance": bool(torch.allclose(reference, styled.detach().float(), atol=atol, rtol=rtol)),
              "baseline_noise": None, "baseline_reason": "baseline_not_supplied"}
    if baseline is not None:
        if baseline.shape != native.shape:
            raise ValueError("Native numerical-noise baseline shape differs")
        result["baseline_noise"] = tensor_statistics(baseline.detach().float() - reference)
        result["baseline_reason"] = None
    return result


@torch.no_grad()
def interaction_metrics(v11: torch.Tensor, v10: torch.Tensor, v01: torch.Tensor,
                        v00: torch.Tensor, target: torch.Tensor | None = None,
                        reference_v00: torch.Tensor | None = None) -> list[dict]:
    values = {"v11": v11, "v10": v10, "v01": v01, "v00": v00}
    if len({tuple(value.shape) for value in values.values()}) != 1 or v00.ndim < 2:
        raise ValueError("All probe predictions must share a batched shape")
    differences = {"v11-v10": v11.float() - v10.float(), "v01-v00": v01.float() - v00.float(),
                   "v10-v00": v10.float() - v00.float(), "v11-v01": v11.float() - v01.float()}
    differences["interaction"] = differences["v11-v10"] - differences["v01-v00"]
    rows = []
    for i in range(v00.shape[0]):
        row = {"example_index": i,
               "prediction_rms": {name: tensor_statistics(value[i])["rms"] for name, value in values.items()},
               "difference_rms": {name: tensor_statistics(value[i])["rms"] for name, value in differences.items()},
               "pairwise_cosines": {f"{first}:{second}": _cosine(values[first][i], values[second][i])
                                    for first, second in itertools.combinations(values, 2)},
               "intervention_cosines": {f"{first}:{second}": _cosine(differences[first][i], differences[second][i])
                                        for first, second in itertools.combinations(differences, 2)},
               "cosine_null_reason": "zero_or_nonfinite_norm",
               "reconstruction_losses": None, "reconstruction_reason": "no_target_image",
               "base_drift": None, "base_drift_reason": "initial_reference_not_supplied"}
        if target is not None:
            row["reconstruction_losses"] = {name: float((value[i].float() - target[i].float()).square().mean())
                                             for name, value in values.items()}
            row["reconstruction_reason"] = None
        if reference_v00 is not None:
            row["base_drift"] = tensor_statistics(v00[i].float() - reference_v00[i].float())
            row["base_drift_reason"] = None
        rows.append(row)
    return rows


@torch.no_grad()
def gate_statistics(beta: torch.Tensor, values: torch.Tensor, derivative: torch.Tensor,
                    rho: float) -> dict:
    if values.shape != derivative.shape or values.ndim != 2 or values.shape[-1] != beta.shape[0]:
        raise ValueError("Gate diagnostics expect native (grid,blocks) values/derivatives")
    values, derivative = values.T, derivative.T
    rows = []
    for block in range(values.shape[0]):
        g, d = values[block].detach().float(), derivative[block].detach().float()
        rows.append({"block_id": block, "beta": beta[block].detach().float().cpu().tolist(),
                     "values": g.cpu().tolist(), "mean": float(g.mean()), "minimum": float(g.min()),
                     "maximum": float(g.max()), "derivative_rms": float(d.square().mean().sqrt()),
                     "center_penalty": float((g - 1).square().mean()),
                     "smoothness_penalty": float(d.square().mean()),
                     "saturation_fraction": float((((g - 1) / rho).abs() >= 0.95).float().mean())})
    return {"blocks": rows, "R_C": float((values.detach().float() - 1).square().mean()),
            "R_H": float(derivative.detach().float().square().mean()),
            "grid_points": values.shape[1], "grid_convention": "linspace_0_1_endpoints_included"}


def _sample_gradient_tuple(gradients, parameters, indices):
    # No writes to training .grad, even temporarily.
    result = torch.zeros(indices.numel(), dtype=torch.float32)
    offset = 0
    for gradient, parameter in zip(gradients, parameters):
        selection = (indices >= offset) & (indices < offset + parameter.numel())
        if gradient is not None and bool(selection.any()):
            positions = (indices[selection] - offset).to(gradient.device)
            result[selection] = gradient.detach().reshape(-1)[positions].float().cpu()
        offset += parameter.numel()
    return result


def gradient_pair_metrics(first: torch.Tensor, second: torch.Tensor, total: int) -> dict:
    if first.shape != second.shape:
        raise ValueError("Gradient coordinate vectors differ")
    count = first.numel()
    scale = total / count if count else 0
    a = math.sqrt(float(first.double().square().sum()) * scale)
    b = math.sqrt(float(second.double().square().sum()) * scale)
    return {"method": "exact" if total == count else "sampled", "sample_size": count,
            "total_coordinates": total, "coverage": count / total if total else None,
            "first_l2": a, "second_l2": b, "cosine": _cosine(first, second),
            "cosine_reason": None if a and b else "zero_norm",
            "norm_ratio_second_first": b / a if a else None,
            "ratio_reason": None if a else "zero_first_norm"}


def isolated_gradient_probe(loss_contexts: Mapping[str, Callable], families: Mapping[str, Sequence],
                            max_coordinates: int = 65536, seed: int = 314159,
                            memory_budget_mb: float = 512) -> dict:
    """Compare isolated losses sequentially, finishing autograd inside each context.

    A loss callable should return a context manager yielding its scalar while all
    branch flags remain bound. A tensor return is accepted for immutable toy graphs.
    The probe never sets .grad or steps an optimizer. Inactive families temporarily
    enable parameter gradients, then restore every original requires_grad flag.
    Coordinate storage and returned working gradients share the supplied budget.
    Small budgets split one loss graph into sequential parameter-gradient chunks;
    no two complete forward graphs or complete family gradients are retained.
    """
    families = {name: list(parameters) for name, parameters in families.items()}
    all_parameters = [parameter for parameters in families.values() for parameter in parameters]
    if len({id(parameter) for parameter in all_parameters}) != len(all_parameters):
        raise ValueError("Probe parameter families must be disjoint")
    saved_flags = [parameter.requires_grad for parameter in all_parameters]
    saved_grads = [parameter.grad for parameter in all_parameters]
    budget = int(memory_budget_mb * 1024 * 1024)
    if budget <= 0 or max_coordinates < 0:
        raise ValueError("Invalid probe diagnostic memory budget")
    names = list(loss_contexts)
    coordinates, totals, offsets = {}, {}, {}
    largest_gradient = max((parameter.numel() * parameter.element_size() for parameter in all_parameters), default=0)
    if all_parameters and budget <= largest_gradient:
        raise ValueError(f"Diagnostic memory budget {budget} bytes cannot hold one native parameter gradient "
                         f"({largest_gradient} bytes) plus sampled coordinates")
    remaining = budget - largest_gradient
    for index, (name, parameters) in enumerate(families.items()):
        total = sum(parameter.numel() for parameter in parameters)
        desired = total if max_coordinates == 0 else min(total, max_coordinates)
        # Stored int64 indices + fp32 vectors, with conservative index/gather
        # workspace for CPU/GPU transfers. Shared budget across families.
        available = remaining // max(1, len(families) - index) // (32 + 4 * max(1, len(names)))
        count = min(desired, available)
        if total and count == 0:
            raise ValueError("Diagnostic memory budget cannot retain even one coordinate per family")
        coordinates[name] = coordinate_indices(total, count, seed + index) if count else torch.empty(0, dtype=torch.long)
        totals[name] = total
        offsets[name] = sum(len(previous) for previous in list(families.values())[:index])
        remaining -= count * (32 + 4 * max(1, len(names)))
    retained_coordinate_bytes = sum(index.numel() * (32 + 4 * max(1, len(names))) for index in coordinates.values())
    working_budget = budget - retained_coordinate_bytes
    # Every plan entry maps one full parameter gradient to its selected family
    # coordinates. Unselected parameters need no requested autograd output.
    plan = []
    for family, parameters in families.items():
        parameter_offset = 0
        for parameter in parameters:
            indices = coordinates[family]
            start = int(torch.searchsorted(indices, parameter_offset))
            end = int(torch.searchsorted(indices, parameter_offset + parameter.numel()))
            if start < end:
                plan.append((family, parameter, start, end, parameter_offset))
            parameter_offset += parameter.numel()
    chunks, chunk, chunk_bytes = [], [], 0
    for entry in plan:
        needed = entry[1].numel() * entry[1].element_size()
        if needed > working_budget:
            raise ValueError("Diagnostic coordinate allocation leaves insufficient working gradient memory")
        if chunk and chunk_bytes + needed > working_budget:
            chunks.append(chunk)
            chunk, chunk_bytes = [], 0
        chunk.append(entry)
        chunk_bytes += needed
    if chunk:
        chunks.append(chunk)
    samples, losses = {}, {}
    try:
        for parameter in all_parameters:
            parameter.requires_grad_(True)
        with isolated_rng():
            for loss_name, make_context in loss_contexts.items():
                supplied = make_context()
                context = contextlib.nullcontext(supplied) if isinstance(supplied, torch.Tensor) else supplied
                with context as loss:
                    if loss.numel() != 1 or not bool(torch.isfinite(loss.detach())):
                        raise ValueError(f"Probe {loss_name} did not return a finite scalar")
                    losses[loss_name] = float(loss.detach())
                    samples[loss_name] = {family: torch.zeros(index.numel(), dtype=torch.float32)
                                          for family, index in coordinates.items()}
                    if loss.requires_grad:
                        for chunk_index, entries in enumerate(chunks):
                            gradients = torch.autograd.grad(loss, [entry[1] for entry in entries],
                                allow_unused=True, retain_graph=chunk_index < len(chunks) - 1)
                            for (family, _, start, end, parameter_offset), gradient in zip(entries, gradients):
                                if gradient is not None:
                                    positions = coordinates[family][start:end] - parameter_offset
                                    samples[loss_name][family][start:end] = gradient.detach().reshape(-1)[
                                        positions.to(gradient.device)].float().cpu()
                            del gradients
    finally:
        for parameter, flag, gradient in zip(all_parameters, saved_flags, saved_grads):
            parameter.requires_grad_(flag)
            if parameter.grad is not gradient:
                # No legitimate code in this helper writes .grad. Guard callbacks.
                parameter.grad = gradient
    return {"losses": losses, "working_gradient_chunks": len(chunks),
            "coordinate_buffer_bytes": retained_coordinate_bytes, "working_gradient_budget_bytes": working_budget,
            "families": {family: {"coordinate_indices": coordinates[family].tolist(),
                                   "total_coordinates": totals[family],
                                   "method": "exact" if coordinates[family].numel() == totals[family] else "sampled",
                                   "budget_reason": "coordinate_storage_budget" if coordinates[family].numel() <
                                       (totals[family] if max_coordinates == 0 else min(totals[family], max_coordinates)) else None,
                                   "pairs": {f"{first}:{second}": gradient_pair_metrics(samples[first][family],
                                              samples[second][family], totals[family])
                                             for first, second in itertools.combinations(names, 2)}}
                         for family in families}}


def save_fixed_probe_packet(path: str | Path, examples: Sequence[Mapping], taus: Sequence[float],
                            seed: int, source: str = "training") -> Path:
    from safetensors.torch import save_file
    path = assert_writable_path(path)
    if (not examples or source not in {"training", "validation"} or seed < 0 or not taus
            or len(set(taus)) != len(taus) or any(not math.isfinite(tau) or not 0 < tau <= 1 for tau in taus)):
        raise ValueError("Fixed probes require real examples, a valid source/seed, and unique taus in (0,1]")
    if path.exists():
        raise FileExistsError(f"Fixed probe packet is immutable and already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.incomplete")
    temporary.mkdir()
    tensors, metadata = {}, []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for index, example in enumerate(examples):
        z0 = example["z0"].detach().cpu().float().contiguous()
        if not z0.numel() or not bool(torch.isfinite(z0).all()):
            raise ValueError("Fixed probe z0 must contain finite latent values")
        noise = torch.randn(z0.shape, dtype=torch.float32, generator=generator)
        tensors[f"z0_{index:04d}"] = z0
        tensors[f"epsilon_{index:04d}"] = noise
        metadata.append({key: json_safe(value) for key, value in example.items() if key != "z0"})
        metadata[-1].update({"packet_index": index, "z0_hash": tensor_hash(z0), "epsilon_hash": tensor_hash(noise)})
    save_file(tensors, str(temporary / "latents.safetensors"))
    with (temporary / "latents.safetensors").open("r+b") as handle:
        os.fsync(handle.fileno())
    write_json(temporary / "manifest.json", {"schema_version": "1.0.0", "seed": seed, "source": source,
                                        "taus": list(taus), "examples": metadata,
                                        "packet_sha256": sha256(temporary / "latents.safetensors")})
    write_json(temporary / "COMPLETE.json", {"manifest_sha256": sha256(temporary / "manifest.json")})
    os.replace(temporary, path)
    return path


def load_fixed_probe_packet(path: str | Path) -> dict:
    from safetensors.torch import load_file
    path = Path(path)
    if not (path / "COMPLETE.json").is_file() or not (path / "manifest.json").is_file():
        raise ValueError("Incomplete fixed probe packet")
    complete = json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
    if sha256(path / "manifest.json") != complete.get("manifest_sha256"):
        raise ValueError("Fixed probe metadata checksum mismatch")
    manifest = json_safe(json.loads((path / "manifest.json").read_text(encoding="utf-8")))
    if manifest.get("schema_version") != "1.0.0" or not manifest.get("examples"):
        raise ValueError("Invalid fixed probe manifest schema/examples")
    if sha256(path / "latents.safetensors") != manifest["packet_sha256"]:
        raise ValueError("Fixed probe packet checksum mismatch")
    tensors = load_file(str(path / "latents.safetensors"), device="cpu")
    expected = {f"{name}_{index:04d}" for index in range(len(manifest["examples"])) for name in ("z0", "epsilon")}
    if set(tensors) != expected:
        raise ValueError("Fixed probe packet tensor mapping differs from manifest")
    for index, example in enumerate(manifest["examples"]):
        z0, epsilon = tensors[f"z0_{index:04d}"], tensors[f"epsilon_{index:04d}"]
        if example.get("packet_index") != index or z0.shape != epsilon.shape:
            raise ValueError("Fixed probe example index/shape mismatch")
        if tensor_hash(z0) != example["z0_hash"] or tensor_hash(epsilon) != example["epsilon_hash"]:
            raise ValueError("Fixed probe tensor identity differs from its metadata")
    return {"manifest": manifest, "tensors": tensors}


class TensorDumpBudget:
    """Optional selected tensors only; limits never affect mandatory probes."""
    ALLOWED = frozenset({"suffix_features", "probe_velocities"})

    def __init__(self, root, max_packets=4, max_total_mb=256, enabled=False):
        self.root = assert_writable_path(root)
        self.max_packets = int(max_packets)
        self.max_bytes = int(max_total_mb * 1024 * 1024)
        self.enabled = enabled

    def save(self, packet_id: str, tensors: Mapping[str, torch.Tensor]) -> dict:
        from safetensors.torch import save_file
        if not self.enabled:
            return {"saved": False, "reason": "disabled"}
        if not tensors or any(name.split("/")[0] not in self.ALLOWED for name in tensors):
            raise ValueError("Only suffix_features and probe_velocities are allowed tensor dumps")
        if not packet_id or Path(packet_id).name != packet_id or any(char in packet_id for char in "\\/:."):
            raise ValueError("Invalid tensor packet ID")
        existing = list(self.root.glob("*.safetensors")) if self.root.exists() else []
        if len(existing) >= self.max_packets:
            return {"saved": False, "reason": "packet_count_budget"}
        estimate = sum(value.numel() * value.element_size() for value in tensors.values())
        used = sum(path.stat().st_size for path in existing)
        if used + estimate > self.max_bytes:
            return {"saved": False, "reason": "tensor_byte_budget"}
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{packet_id}.safetensors"
        if path.exists():
            raise ValueError("Optional tensor packet ID already exists")
        save_file({name: value.detach().cpu().contiguous().clone() for name, value in tensors.items()}, str(path))
        if used + path.stat().st_size > self.max_bytes:
            path.unlink()
            return {"saved": False, "reason": "tensor_byte_budget_including_header"}
        return {"saved": True, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def provenance(repo_root: str | Path) -> dict:
    """Explicit allowlist of environment facts; never dump environment variables."""
    repo_root = Path(repo_root)
    def git(*args):
        result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, check=False)
        return result.stdout if result.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    diff = git("diff", "HEAD", "--binary")
    packages = {}
    for name in ("torch", "transformers", "accelerate", "diffusers", "safetensors", "optimum-quanto"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": packages, "cuda_runtime": torch.version.cuda,
              "git_revision": commit.decode().strip() if commit else None,
              "dirty_diff_sha256": hashlib.sha256(diff).hexdigest() if diff is not None else None,
              "gpu": [], "gpu_reason": None if torch.cuda.is_available() else "cuda_unavailable"}
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            result["gpu"].append({"index": index, "name": properties.name,
                                  "total_memory": properties.total_memory})
    return result


class _Moments:
    def __init__(self):
        self.count = self.missing = 0
        self.mean = self.m2 = 0.0
        self.minimum = self.maximum = None
        self.magnitude_histogram = defaultdict(int)

    def add(self, value):
        if value is None:
            self.missing += 1
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        bucket = "zero" if value == 0 else f"{'positive' if value > 0 else 'negative'}_decade_{math.floor(math.log10(abs(value)))}"
        self.magnitude_histogram[bucket] += 1

    def result(self):
        return {"count": self.count, "missing_count": self.missing,
                "mean": self.mean if self.count else None,
                "standard_deviation": math.sqrt(max(0, self.m2) / self.count) if self.count else None,
                "minimum": self.minimum, "maximum": self.maximum,
                "magnitude_histogram": dict(sorted(self.magnitude_histogram.items())),
                "histogram_convention": "sign_and_floor_log10_absolute_value; zero has its own bin",
                "reason": None if self.count else "no_numeric_observations"}


def _numeric_leaves(value, prefix=""):
    for key, item in value.items():
        if key in {"record_sequence", "logical_update", "update_attempt_id", "example_index", "block_id",
                   "original_ids", "input_ids", "initializer_token_ids", "positions", "suffix_mask",
                   "coordinate_indices", "example_ids"}:
            continue
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            yield from _numeric_leaves(item, path)
        elif isinstance(item, list):
            for index, element in enumerate(item):
                if isinstance(element, Mapping):
                    label = element.get("token_region", f"block_{element['block_id']}" if "block_id" in element else "examples")
                    yield from _numeric_leaves(element, f"{path}.{label}")
                elif element is None or isinstance(element, (int, float)) and not isinstance(element, bool):
                    yield f"{path}[{index}]", element
        elif item is None or isinstance(item, (int, float)) and not isinstance(item, bool):
            yield path, item


def summarize_run(run_dir: str | Path, *, write=True) -> dict:
    """Streaming aggregates keep D/A/G, split and known content groups separate."""
    root = Path(run_dir).resolve()
    groups, counts, boundaries = defaultdict(lambda: defaultdict(_Moments)), {}, []
    dimensions = ("stage", "update_kind", "checkpoint_hash", "time_bin", "content_group", "split")
    for stream in sorted(STREAMS):
        count = 0
        for row in iter_records(root, stream):
            count += 1
            row = dict(row)
            row.setdefault("content_group", row.get("group", row.get("prompt_group", "unassigned")))
            row.setdefault("time_bin", row.get("tau_bin", "unassigned"))
            if stream == "events" and row.get("event") in {"phase_transition", "stage_boundary"}:
                boundaries.append(row)
            key = (stream, *(str(row.get(name, "unassigned")) for name in dimensions))
            for field, value in _numeric_leaves(row):
                if field not in {"record_sequence", "logical_update", "update_attempt_id"}:
                    groups[key][field].add(value)
        counts[stream] = count
    aggregate = [{"stream": key[0], **dict(zip(dimensions, key[1:])),
                  "metrics": {field: moments.result() for field, moments in sorted(fields.items())}}
                 for key, fields in sorted(groups.items())]
    # Rating cells are interpreted only when a human has actually supplied numbers.
    rating_groups = defaultdict(lambda: defaultdict(_Moments))
    sample_metadata = {row["image_id"]: row for row in iter_records(root, "samples/manifest") if "image_id" in row}
    rated_samples = []
    rating_path = root / "human_ratings.csv"
    if rating_path.exists():
        with rating_path.open(newline="", encoding="utf-8") as handle:
            for rating in csv.DictReader(handle):
                image = sample_metadata.get(rating.get("image_id"), {})
                group = str(image.get("prompt_group", image.get("content_group", image.get("group", "unassigned"))))
                record = {**image, "rater": rating.get("rater", ""), "scores": {},
                          "mode": image.get("ablation_mode", image.get("mode"))}
                record.setdefault("sample_steps", record.get("steps", ""))
                record.setdefault("guidance_scale", record.get("guidance", ""))
                for name in ("target_style_fidelity", "content_fulfillment", "visible_artifacts", "copying_suspicious_similarity"):
                    raw = rating.get(name, "").strip()
                    try:
                        value = float(raw) if raw else None
                    except ValueError:
                        value = None
                    if value is not None and not math.isfinite(value):
                        value = None
                    rating_groups[group][name].add(value)
                    record["scores"][name] = value
                rated_samples.append(record)
    ratings = {group: {name: moments.result() for name, moments in fields.items()}
               for group, fields in rating_groups.items()}
    pairs = []
    # Full sampling identity prevents pairing incomparable prompts/noise/settings.
    pair_keys = ("checkpoint_hash", "prompt_id", "prompt_hash", "seed", "width", "height",
                 "sample_steps", "guidance_scale", "sampler", "sigma_schedule", "lora_strength",
                 "unconditional_adapter", "rng_backend", "initial_noise_source", "rater")
    matched = defaultdict(list)
    for row in rated_samples:
        matched[tuple(str(row.get(key, "")) for key in pair_keys)].append(row)
    for key, rows in matched.items():
        for first, second in itertools.combinations(rows, 2):
            if first.get("mode") == second.get("mode"):
                continue
            difference = {name: second["scores"][name] - first["scores"][name]
                          for name in first["scores"] if first["scores"][name] is not None and second["scores"][name] is not None}
            if difference:
                pairs.append({**dict(zip(pair_keys, key)), "first_mode": first.get("mode"),
                              "second_mode": second.get("mode"), "second_minus_first": difference})
    ranked = sorted(((group, fields["target_style_fidelity"]["mean"]) for group, fields in ratings.items()
                     if fields.get("target_style_fidelity", {}).get("count", 0)), key=lambda item: item[1])
    summary = {"schema_version": "1.0.0", "record_counts": counts, "stage_boundaries": boundaries,
               "aggregates": aggregate, "human_rating_groups": ratings,
               "lower_scoring_style_groups": [{"group": group, "mean": mean} for group, mean in ranked],
               "paired_human_differences": pairs,
               "interpretation": "Numerical responses are not visual style scores. D/A/G objectives remain separate.",
               "human_score_status": "actual_numeric_ratings" if any(
                   metric["count"] for fields in ratings.values() for metric in fields.values()) else "unscored"}
    if write:
        write_json(root / "summary.json", summary)
        path = assert_writable_path(root / "summary.md")
        lines = ["# Gen2 recorded evidence", "", summary["interpretation"], "", "## Record counts", ""]
        lines.extend(f"- {name}: {count}" for name, count in counts.items())
        lines.extend(["", "## Human evaluation", "", f"Status: {summary['human_score_status']}.",
                      "See summary.json for per-stage, update-kind, checkpoint, time-bin, group and split distributions.",
                      "Undefined metrics retain their missing counts; blank human scores are never replaced by zero.", ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        csv_path = assert_writable_path(root / "summary.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fields = ["stream", *dimensions, "metric", "count", "missing_count", "mean", "standard_deviation", "minimum", "maximum", "reason"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for group in aggregate:
                for name, values in group["metrics"].items():
                    writer.writerow({key: value for key, value in {**group, "metric": name, **values}.items() if key in fields})
    return summary


def export_diagnostics(run_dir: str | Path, destination: str | Path,
                       include_images: bool = False) -> Path:
    """Export an allowlisted, portable evidence bundle, excluding model/dataset files."""
    root = Path(run_dir).resolve()
    output = assert_writable_path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Evidence can live in the user's protected local gen2 folder. Export reads
    # it only; regenerated summaries are written straight into the output zip.
    summary = summarize_run(root, write=False)
    approved = []
    root_metadata = {"run_manifest.json", "module_manifest.json", "probe_manifest.json", "caption_token_report.json",
                     "config.requested.yaml", "config.resolved.yaml", "human_ratings.csv",
                     }
    for path in root.rglob("*"):
        if not path.is_file() or path.resolve() == output:
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        if root not in path.resolve().parents:
            # A symlink must not turn an evidence export into a dataset/weight copy.
            continue
        if any(part in {"weights", "dataset", "datasets", "tensor_dumps", "cache", ".cache"} for part in parts):
            continue
        suffix = path.suffix.lower()
        stream_file = (len(parts) == 1 and any(path.name == f"{stream}.jsonl" or
                      path.name.startswith(f"{stream}.") and path.name.endswith((".jsonl", ".jsonl.gz"))
                      for stream in STREAMS if "/" not in stream))
        metadata = (relative.as_posix() in root_metadata or stream_file
                    or parts[0] == "samples" and suffix in {".json", ".jsonl", ".gz"}
                    or parts[0] == "gradient_coordinates" and suffix == ".json"
                    or parts[0] == "checkpoints" and path.name in {"manifest.json", "COMPLETE.json"})
        fixed = (any(part in {"fixed_probe", "fixed_probes", "probe_packet", "fixed_probe_packet"} for part in parts)
                 and suffix == ".safetensors") or relative.as_posix() == "fixed_probe_packet.pt"
        if parts[0] in {"fixed_probe", "fixed_probes", "probe_packet", "fixed_probe_packet"} and path.name in {"manifest.json", "COMPLETE.json"}:
            metadata = True
        picture = include_images and parts[0] == "samples" and suffix in {".png", ".jpg", ".jpeg", ".webp"}
        if metadata or fixed or picture:
            approved.append(path)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(approved):
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("summary.json", json.dumps(json_safe(summary, redact=True), ensure_ascii=False,
                                                    allow_nan=False, sort_keys=True, indent=2) + "\n")
        markdown = ["# Gen2 recorded evidence", "", summary["interpretation"], "", "## Record counts", ""]
        markdown.extend(f"- {stream}: {count}" for stream, count in summary["record_counts"].items())
        markdown.extend(["", "## Human evaluation", "", f"Status: {summary['human_score_status']}.",
                         "The JSON/CSV summaries retain separate stage, update-kind, checkpoint, time-bin, content-group and split aggregates.", ""])
        archive.writestr("summary.md", "\n".join(markdown))
        csv_buffer = io.StringIO(newline="")
        fields = ["stream", "stage", "update_kind", "checkpoint_hash", "time_bin", "content_group", "split",
                  "metric", "count", "missing_count", "mean", "standard_deviation", "minimum", "maximum", "reason"]
        writer = csv.DictWriter(csv_buffer, fieldnames=fields)
        writer.writeheader()
        for group in summary["aggregates"]:
            for name, values in group["metrics"].items():
                writer.writerow({key: value for key, value in {**group, "metric": name, **values}.items() if key in fields})
        archive.writestr("summary.csv", csv_buffer.getvalue())
    return output
