from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


GROUP_KINDS = ("attention", "mlp", "adaln", "other")
_BLOCK_RE = re.compile(r"(?:^|[.$_])layers(?:[.$_])(\d+)(?:[.$_]|$)")


def sha256_file(path: os.PathLike | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: os.PathLike | str, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_jsonl(path: os.PathLike | str, records: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_json(path: os.PathLike | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def load_jsonl(path: os.PathLike | str) -> List[Dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    output = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object: {source}")
            output.append(value)
    return output


def require_file_hash(path: os.PathLike | str, expected_sha256: Optional[str], label: str) -> Path:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not expected_sha256 or len(str(expected_sha256)) != 64:
        raise RuntimeError(f"{label} is missing a valid SHA-256 contract record")
    actual = sha256_file(source)
    if actual != str(expected_sha256):
        raise RuntimeError(f"{label} SHA-256 mismatch: {actual} != {expected_sha256}")
    return source


def parse_activator_a2_contract(contract: Mapping[str, Any]) -> Dict[str, Any]:
    if contract.get("schema") != "ai-toolkit.ideogram4-v3-activator-stage-contract":
        raise RuntimeError("A2 contract schema is not the Ideogram4 V3 activator stage contract")
    if int(contract.get("schema_version", 0)) != 1:
        raise RuntimeError("unsupported Ideogram4 V3 activator contract schema version")
    if contract.get("pipeline") != "ideogram4_v3_activator" or contract.get("stage") != "te_calibration":
        raise RuntimeError("residual ablation requires the te_calibration stage contract")
    if contract.get("status") != "completed" or contract.get("return_code") != 0:
        raise RuntimeError("Ideogram4 V3 activator A2 contract is not completed")
    if contract.get("training", {}).get("network") != "V3 active/frozen":
        raise RuntimeError("A2 contract must record V3 active/frozen training")
    sources = contract.get("sources")
    artifacts = contract.get("artifacts")
    if not isinstance(sources, Mapping) or not isinstance(artifacts, Mapping):
        raise RuntimeError("A2 contract lacks sources or artifacts")
    required = {
        "snapshot": contract.get("config_snapshot"),
        "snapshot_sha256": contract.get("config_snapshot_sha256"),
        "v3_weights": (sources.get("v3_weights") or {}).get("path"),
        "v3_weights_sha256": (sources.get("v3_weights") or {}).get("sha256"),
        "best_embedding": artifacts.get("best_embedding"),
        "best_te_adapter": artifacts.get("best_te_adapter"),
        "best_manifest": artifacts.get("best_manifest"),
    }
    missing = sorted(key for key, value in required.items() if not value)
    if missing:
        raise RuntimeError(f"A2 activator contract lacks required fields: {missing}")
    return required


def module_display_name(module: Any) -> str:
    name = str(getattr(module, "lora_name", ""))
    return name.replace("$$", ".").replace("diffusion_model.", "transformer.")


def classify_lora_module(name: str) -> Tuple[Optional[int], str]:
    normalized = name.replace("$$", ".").lower()
    match = _BLOCK_RE.search(normalized)
    block_index = int(match.group(1)) if match else None
    if ".attention." in normalized or ".attn." in normalized:
        kind = "attention"
    elif ".feed_forward." in normalized or ".mlp." in normalized or any(
        token in normalized for token in (".w1", ".w2", ".w3")
    ):
        kind = "mlp"
    elif "adaln" in normalized or "ada_ln" in normalized:
        kind = "adaln"
    else:
        kind = "other"
    return block_index, kind


def build_module_registry(modules: Sequence[Any]) -> List[Dict[str, Any]]:
    registry = []
    seen = set()
    for index, module in enumerate(modules):
        name = module_display_name(module)
        if not name or name in seen:
            raise ValueError(f"LoRA registry requires unique non-empty names: {name!r}")
        seen.add(name)
        block_index, kind = classify_lora_module(name)
        group_id = f"block:{block_index}:{kind}" if block_index is not None else f"global:{kind}"
        down = getattr(getattr(module, "lora_down", None), "weight", None)
        up = getattr(getattr(module, "lora_up", None), "weight", None)
        registry.append({
            "module_index": index,
            "module_name": name,
            "block_index": block_index,
            "kind": kind,
            "group_id": group_id,
            "rank": int(getattr(module, "lora_dim", down.shape[0] if down is not None else 0)),
            "down_shape": list(down.shape) if down is not None else None,
            "up_shape": list(up.shape) if up is not None else None,
        })
    if not registry:
        raise ValueError("LoRA registry is empty")
    return registry


def validate_complete_partition(registry: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    names = [str(row["module_name"]) for row in registry]
    if len(names) != len(set(names)):
        raise ValueError("LoRA registry contains duplicate module names")
    invalid = [row for row in registry if row.get("kind") not in GROUP_KINDS or not row.get("group_id")]
    if invalid:
        raise ValueError("LoRA registry partition is incomplete")
    counts = {kind: sum(row["kind"] == kind for row in registry) for kind in GROUP_KINDS}
    groups = sorted({str(row["group_id"]) for row in registry})
    return {"module_count": len(registry), "group_count": len(groups), "kind_counts": counts, "groups": groups}


def registry_maps(registry: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    module_to_group = {str(row["module_name"]): str(row["group_id"]) for row in registry}
    group_to_modules: Dict[str, List[str]] = defaultdict(list)
    for module_name, group_id in module_to_group.items():
        group_to_modules[group_id].append(module_name)
    return module_to_group, {key: sorted(value) for key, value in group_to_modules.items()}


@dataclass
class _RunningTensorStats:
    count: int = 0
    numel: int = 0
    sum_sq: float = 0.0
    max_abs: float = 0.0
    nonfinite: int = 0

    def update(self, tensor: Any) -> None:
        value = tensor.detach().float()
        self.count += 1
        self.numel += value.numel()
        finite = value.isfinite()
        self.nonfinite += int((~finite).sum().item())
        if finite.any():
            valid = value[finite]
            self.sum_sq += float(valid.square().sum().item())
            self.max_abs = max(self.max_abs, float(valid.abs().max().item()))

    def as_dict(self) -> Dict[str, Any]:
        finite_numel = self.numel - self.nonfinite
        return {
            "call_count": self.count,
            "numel": self.numel,
            "nonfinite_count": self.nonfinite,
            "rms": math.sqrt(self.sum_sq / finite_numel) if finite_numel > 0 else None,
            "max_abs": self.max_abs if finite_numel > 0 else None,
        }


class ResidualAblationRuntime:
    def __init__(
        self,
        module_to_group: Mapping[str, str],
        *,
        capture: bool = True,
        capture_vectors: bool = False,
        sketch_size: int = 256,
    ):
        self.module_to_group = dict(module_to_group)
        self.capture = bool(capture)
        self.capture_vectors = bool(capture_vectors)
        self.sketch_size = int(sketch_size)
        if self.sketch_size <= 0:
            raise ValueError("sketch_size must be positive")
        self.gates: Dict[str, Any] = {}
        self._module_stats: Dict[str, _RunningTensorStats] = defaultdict(_RunningTensorStats)
        self._group_stats: Dict[str, _RunningTensorStats] = defaultdict(_RunningTensorStats)
        self._group_sketches: Dict[str, List[Any]] = defaultdict(list)
        self._projection_basis: Dict[str, Any] = {}
        self._projection_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "energy": 0.0, "projected_energy": 0.0})

    def set_gates(self, gates: Mapping[str, Any]) -> None:
        unknown = set(gates).difference(self.module_to_group.values())
        if unknown:
            raise KeyError(f"unknown LoRA group gates: {sorted(unknown)}")
        self.gates = dict(gates)

    def set_projection_basis(self, basis_by_group: Mapping[str, Any]) -> None:
        self._projection_basis = dict(basis_by_group)

    def reset(self) -> None:
        self._module_stats.clear()
        self._group_stats.clear()
        self._group_sketches.clear()
        self._projection_stats.clear()

    def captured_group_vectors(self) -> Dict[str, Any]:
        import torch

        output = {}
        for group, vectors in self._group_sketches.items():
            if vectors:
                output[group] = torch.stack(vectors).mean(dim=0)
        return output

    def apply(self, module: Any, residual: Any):
        name = module_display_name(module)
        group = self.module_to_group.get(name)
        if group is None:
            raise KeyError(f"LoRA observer saw unregistered module: {name}")
        if self.capture:
            self._module_stats[name].update(residual)
            self._group_stats[group].update(residual)
            flat = residual.detach().float().reshape(-1)
            sketch = pooled_residual_sketch(flat, self.sketch_size)
            if self.capture_vectors:
                self._group_sketches[group].append(sketch.cpu())
            basis = self._projection_basis.get(group)
            if basis is not None:
                projected = project_onto_subspace(sketch, basis)
                state = self._projection_stats[group]
                state["count"] += 1
                state["energy"] += float(sketch.square().sum().item())
                state["projected_energy"] += float(projected.square().sum().item())
        gate = self.gates.get(group, 1.0)
        return residual * gate

    def summary(self) -> Dict[str, Any]:
        projections = {}
        for group, state in self._projection_stats.items():
            energy = state["energy"]
            projections[group] = {
                "call_count": int(state["count"]),
                "energy_fraction": state["projected_energy"] / energy if energy > 0 else None,
            }
        return {
            "modules": {key: value.as_dict() for key, value in sorted(self._module_stats.items())},
            "groups": {key: value.as_dict() for key, value in sorted(self._group_stats.items())},
            "helper_subspace_projection": projections,
        }


@contextlib.contextmanager
def residual_runtime_context(network: Any, runtime: ResidualAblationRuntime) -> Iterator[ResidualAblationRuntime]:
    previous = getattr(network, "_residual_ablation_runtime", None)
    network._residual_ablation_runtime = runtime
    try:
        yield runtime
    finally:
        network._residual_ablation_runtime = previous


def pooled_residual_sketch(tensor: Any, width: int = 256):
    import torch

    flat = tensor.detach().float().reshape(-1)
    if flat.numel() == width:
        return flat
    if flat.numel() < width:
        return torch.nn.functional.pad(flat, (0, width - flat.numel()))
    boundaries = torch.linspace(0, flat.numel(), width + 1, device=flat.device).long()
    return torch.stack([flat[boundaries[index]:boundaries[index + 1]].mean() for index in range(width)])


def helper_response_spectrum(
    vectors: Sequence[Any],
    *,
    epsilon: float = 1.0e-12,
) -> Dict[str, Any]:
    import torch

    flattened = [vector.detach().float().reshape(-1) for vector in vectors if vector is not None and vector.numel()]
    if not flattened:
        return {
            "helper_count": 0,
            "numerical_rank": 0,
            "effective_rank": 0.0,
            "top_energy_fraction": 0.0,
            "singular_values": [],
            "energy_fractions": [],
        }
    width = flattened[0].numel()
    if any(vector.numel() != width for vector in flattened):
        raise ValueError("helper response vectors must have matching dimensions")
    matrix = torch.stack(flattened, dim=1)
    singular_values = torch.linalg.svdvals(matrix)
    energies = singular_values.square()
    total_energy = float(energies.sum().item())
    if total_energy <= epsilon:
        fractions = torch.zeros_like(energies)
        effective_rank = 0.0
        top_fraction = 0.0
        numerical_rank = 0
    else:
        fractions = energies / total_energy
        positive = fractions > epsilon
        entropy = -torch.sum(fractions[positive] * torch.log(fractions[positive]))
        effective_rank = float(torch.exp(entropy).item())
        top_fraction = float(fractions[0].item())
        tolerance = max(matrix.shape) * torch.finfo(singular_values.dtype).eps * float(singular_values[0].item())
        numerical_rank = int((singular_values > tolerance).sum().item())
    return {
        "helper_count": len(flattened),
        "numerical_rank": numerical_rank,
        "effective_rank": effective_rank,
        "top_energy_fraction": top_fraction,
        "singular_values": [float(value) for value in singular_values.tolist()],
        "energy_fractions": [float(value) for value in fractions.tolist()],
    }


def build_helper_response_bases(
    captured_by_helper: Mapping[str, Mapping[str, Any]],
    *,
    epsilon: float = 1.0e-8,
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Dict[str, Any]]]:
    groups = sorted({group for captured in captured_by_helper.values() for group in captured})
    bases = {}
    mean_norms = {}
    spectra = {}
    for group in groups:
        vectors = [captured[group] for captured in captured_by_helper.values() if group in captured]
        basis = orthonormal_basis(vectors, epsilon=epsilon)
        if basis is not None:
            bases[group] = basis
        norms = [float(vector.detach().float().norm().item()) for vector in vectors]
        mean_norms[group] = statistics.fmean(norms) if norms else 0.0
        spectra[group] = helper_response_spectrum(vectors)
    return bases, mean_norms, spectra


def orthonormal_basis(vectors: Sequence[Any], *, epsilon: float = 1.0e-8):
    import torch

    flattened = [vector.detach().float().reshape(-1) for vector in vectors if vector is not None and vector.numel()]
    if not flattened:
        return None
    width = flattened[0].numel()
    if any(vector.numel() != width for vector in flattened):
        raise ValueError("helper subspace vectors must have matching dimensions")
    matrix = torch.stack(flattened, dim=1)
    q, r = torch.linalg.qr(matrix, mode="reduced")
    diagonal = torch.abs(torch.diagonal(r))
    rank = int((diagonal > epsilon).sum().item())
    return q[:, :rank].contiguous() if rank else None


def project_onto_subspace(tensor: Any, basis: Any):
    flat = tensor.detach().float().reshape(-1)
    if basis is None:
        return flat.new_zeros(flat.shape)
    if basis.shape[0] != flat.numel():
        raise ValueError(f"subspace dimension {basis.shape[0]} does not match residual dimension {flat.numel()}")
    basis = basis.to(flat)
    return basis @ (basis.transpose(0, 1) @ flat)


def projection_metrics(tensor: Any, basis: Any, *, epsilon: float = 1.0e-12) -> Dict[str, Any]:
    import torch

    flat = tensor.detach().float().reshape(-1)
    projected = project_onto_subspace(flat, basis)
    energy = float(flat.square().sum().item())
    projected_energy = float(projected.square().sum().item())
    cosine = None
    if energy > epsilon and projected_energy > epsilon:
        cosine = float(torch.nn.functional.cosine_similarity(flat[None], projected[None], dim=1).item())
    return {
        "energy": energy,
        "projected_energy": projected_energy,
        "energy_fraction": projected_energy / (energy + epsilon),
        "cosine": cosine,
    }


def gate_gradients(loss: Any, gates: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    import torch

    ordered = list(gates)
    gradients = torch.autograd.grad(loss, [gates[key] for key in ordered], allow_unused=True, retain_graph=False)
    return {key: None if gradient is None else float(gradient.detach().float().item()) for key, gradient in zip(ordered, gradients)}


def finite_difference_metrics(losses: Mapping[float, float], *, center: float = 1.0) -> Dict[str, float]:
    ordered = sorted((float(scale), float(loss)) for scale, loss in losses.items())
    if center not in dict(ordered):
        raise ValueError("finite differences require a center loss")
    lower = max((item for item in ordered if item[0] < center), default=None)
    upper = min((item for item in ordered if item[0] > center), default=None)
    if lower is None or upper is None:
        raise ValueError("finite differences require scales on both sides of center")
    center_loss = dict(ordered)[center]
    left_slope = (center_loss - lower[1]) / (center - lower[0])
    right_slope = (upper[1] - center_loss) / (upper[0] - center)
    secant = (upper[1] - lower[1]) / (upper[0] - lower[0])
    curvature = 2.0 * (right_slope - left_slope) / (upper[0] - lower[0])
    best_scale, best_loss = min(ordered, key=lambda item: (item[1], abs(item[0] - center)))
    return {
        "left_slope": left_slope,
        "right_slope": right_slope,
        "central_secant": secant,
        "curvature": curvature,
        "best_scale": best_scale,
        "best_loss": best_loss,
        "center_loss": center_loss,
    }


def resume_key(record: Mapping[str, Any]) -> Tuple[str, str, str]:
    return str(record["run_id"]), str(record["probe_case_id"]), str(record["group_id"])


def annotate_viability_metrics(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    mutable = [{**record, "metrics": dict(record.get("metrics", {}))} for record in records]
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in mutable:
        by_group[str(record["group_id"])].append(record)
    for values in by_group.values():
        gradients = [float(row["metrics"]["dL_dg"]) for row in values if row["metrics"].get("dL_dg") is not None]
        mean_grad = statistics.fmean(gradients) if gradients else None
        if gradients:
            positive = sum(value > 0 for value in gradients)
            negative = sum(value < 0 for value in gradients)
            sign_consistency = max(positive, negative) / len(gradients)
        else:
            sign_consistency = None
        train = [float(row["metrics"]["dL_dg"]) for row in values if row.get("split") == "train" and row["metrics"].get("dL_dg") is not None]
        heldout = [float(row["metrics"]["dL_dg"]) for row in values if row.get("split") == "heldout" and row["metrics"].get("dL_dg") is not None]
        heldout_agreement = None
        if train and heldout:
            train_mean = statistics.fmean(train)
            heldout_mean = statistics.fmean(heldout)
            heldout_agreement = 1.0 if train_mean == 0.0 == heldout_mean else float(train_mean * heldout_mean > 0)
        projection_values = [float(row["metrics"]["projection_p_j"]) for row in values if row["metrics"].get("projection_p_j") is not None]
        magnitude_values = [float(row["metrics"]["magnitude_ratio_m_j"]) for row in values if row["metrics"].get("magnitude_ratio_m_j") is not None]
        projection = statistics.fmean(projection_values) if projection_values else 0.0
        magnitude = statistics.fmean(magnitude_values) if magnitude_values else 0.0
        agreement = heldout_agreement if heldout_agreement is not None else 0.0
        consistency = sign_consistency if sign_consistency is not None else 0.0
        sensitivity = abs(mean_grad) if mean_grad is not None else 0.0
        candidate_score = projection * min(magnitude, 2.0) * (0.5 + 0.5 * consistency) * (0.5 + 0.5 * agreement) / (1.0 + sensitivity)
        for row in values:
            metrics = row["metrics"]
            metrics["mean_grad"] = mean_grad
            metrics["gradient_sign_consistency"] = sign_consistency
            metrics["heldout_agreement"] = heldout_agreement
            metrics["candidate_score"] = candidate_score
    return mutable


def aggregate_records(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = list(records)
    metric_names = sorted({
        key for row in rows for key, value in row.get("metrics", {}).items()
        if value is None or isinstance(value, (int, float))
    })
    dimensions = (
        ("split_group", ("split", "group_id")),
        ("split_kind", ("split", "kind")),
        ("split_block_kind", ("split", "block_index", "kind")),
    )
    output = []
    for group_name, keys in dimensions:
        grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[tuple(row.get(key) for key in keys)].append(row)
        for identity, values in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
            entry = {"group": group_name, **dict(zip(keys, identity)), "record_count": len(values), "metrics": {}}
            for metric in metric_names:
                samples = [float(row["metrics"][metric]) for row in values if row.get("metrics", {}).get(metric) is not None]
                entry["metrics"][metric] = {
                    "count": len(samples),
                    "mean": statistics.fmean(samples) if samples else None,
                    "median": statistics.median(samples) if samples else None,
                    "std": statistics.pstdev(samples) if samples else None,
                    "positive_fraction": sum(value > 0 for value in samples) / len(samples) if samples else None,
                }
            output.append(entry)
    return output


REPORT_METRIC_FIELDS = (
    "residual_rms",
    "normalized_rms",
    "projection_p_j",
    "magnitude_ratio_m_j",
    "helper_top_energy_fraction",
    "helper_effective_rank",
    "helper_numerical_rank",
    "mean_grad",
    "gradient_sign_consistency",
    "heldout_agreement",
    "fd_result",
    "candidate_score",
)


def build_report(records: Sequence[Mapping[str, Any]], registry_summary: Mapping[str, Any]) -> str:
    lines = [
        "# Ideogram4 V3 LoRA Viability Ablation",
        "",
        "该报告消费 calibrated A2 activator，仅汇总 inference/diagnostics 结果，不会启动 Phase C。",
        "",
        f"- LoRA modules: {registry_summary['module_count']}",
        f"- Dynamic groups: {registry_summary['group_count']}",
        f"- Probe/group records: {len(records)}",
        "",
        "## Required viability fields",
        "",
        "- Block / family",
        "- RMS / normalized RMS",
        "- Projection `p_j` / magnitude ratio `m_j`",
        "- Helper singular spectrum / top-mode energy / effective rank",
        "- Mean gate gradient / sign consistency / heldout agreement",
        "- Finite-difference result / candidate score",
        "",
        "## Partition",
        "",
    ]
    for kind in GROUP_KINDS:
        lines.append(f"- `{kind}`: {registry_summary['kind_counts'].get(kind, 0)} modules")
    lines.extend(["", "## Candidate scale wins", ""])
    wins: Dict[float, int] = defaultdict(int)
    references: Dict[str, int] = defaultdict(int)
    for record in records:
        references[str(record.get("reference_type", "calibrated_activator"))] += 1
        best = record.get("metrics", {}).get("best_scale")
        if best is not None:
            wins[float(best)] += 1
    for scale, count in sorted(wins.items()):
        lines.append(f"- `{scale:g}`: {count}")
    lines.extend(["", "## Prompt references", ""])
    for name, count in sorted(references.items()):
        lines.append(f"- `{name}`: {count}")

    lines.extend(["", "## Helper response spectrum", ""])
    spectrum_rows: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        spectrum = record.get("helper_response_spectrum")
        if isinstance(spectrum, Mapping):
            spectrum_rows[str(record.get("group_id"))].append(spectrum)
    if not spectrum_rows:
        lines.append("- No helper spectrum records were emitted.")
    else:
        for group, values in sorted(spectrum_rows.items()):
            top = [float(value["top_energy_fraction"]) for value in values if value.get("top_energy_fraction") is not None]
            ranks = [float(value["effective_rank"]) for value in values if value.get("effective_rank") is not None]
            numerical = [float(value["numerical_rank"]) for value in values if value.get("numerical_rank") is not None]
            lines.append(
                f"- `{group}`: top-mode energy={statistics.fmean(top):.6f}, "
                f"effective rank={statistics.fmean(ranks):.4f}, "
                f"numerical rank={statistics.fmean(numerical):.2f}"
            )
    lines.extend(["", "## Phase C", "", "未自动启动；后续训练决策必须由人工消费本报告和 manifest。", ""])
    return "\n".join(lines)
