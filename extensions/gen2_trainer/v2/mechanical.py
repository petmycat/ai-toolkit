"""Bounded real-data input-gradient/memory probes without advancing training.

Select existing native bucket batches directly. No sampler iteration, epoch
setup, optimizer call, synthetic resizing, or caption shortening is introduced.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import torch

from ..diagnostics import isolated_rng
from ..objectives import per_example_mse


LIMITATION = (
    "At most two existing native batches per expanded dataset/resolution: the "
    "largest combined image/text activation proxy and the longest expanded "
    "caption batch. This samples observed shapes; it does not guarantee that "
    "every batch, later bucket shuffle, allocator state, or sampling run fits."
)


def _native_tools():
    from toolkit.data_loader import get_dataloader_datasets, dto_collation
    from toolkit.accelerator import get_accelerator
    return get_dataloader_datasets, dto_collation, get_accelerator


def _prepare_batch(dto, backend, config, manifest_by_path):
    # The runner imports this module at its use site, so the inverse reference
    # also belongs here rather than at module import time.
    from .process import prepare_batch
    return prepare_batch(dto, backend.model, config, manifest_by_path)


def _candidate_batches(backend, datasets, manifest_by_path):
    """Rank actual native batches, using expanded token counts, not characters."""
    divisor = int(backend.model.vae_scale_factor * backend.model.patch_size)
    if divisor < 1:
        raise ValueError("Invalid native image-token divisor")
    lengths, selected = {}, []
    for dataset_index, dataset in enumerate(datasets):
        if not getattr(dataset.dataset_config, "buckets", False):
            raise ValueError("V2 mechanical probes require native bucketed datasets")
        indices = dataset.batch_indices
        if not isinstance(indices, list) or not indices:
            raise ValueError("V2 mechanical probe found no native bucket batches")
        candidates = []
        for batch_index, file_indices in enumerate(indices):
            if not file_indices:
                raise ValueError("Native bucket batch is empty")
            paths, text_lengths, image_tokens, crop_shapes = [], [], [], []
            for file_index in file_indices:
                item = dataset.file_list[file_index]
                path = str(Path(item.path).resolve())
                source = manifest_by_path[path]
                caption = source["original_caption"]
                if caption not in lengths:
                    lengths[caption] = len(backend.compiler.compile(caption, require_trigger=True).ids)
                width, height = item.crop_width, item.crop_height
                if (type(width) is not int or type(height) is not int or
                        width < divisor or height < divisor or width % divisor or height % divisor):
                    raise ValueError(f"Invalid native crop geometry for memory probe: {path}: {width}x{height}")
                paths.append(path)
                text_lengths.append(lengths[caption])
                image_tokens.append((width // divisor) * (height // divisor))
                crop_shapes.append([height, width])
            if len(set(tuple(shape) for shape in crop_shapes)) != 1:
                raise ValueError("Native bucket batch contains inconsistent image shapes")
            # Qwen encodes captions separately; DiT uses batch-padded text. This
            # rank proxy captures both costs but is deliberately not a byte model.
            max_text, max_image = max(text_lengths), max(image_tokens)
            score = sum(length * length for length in text_lengths) + len(paths) * (max_text + max_image) ** 2
            candidates.append({"expanded_dataset_index": dataset_index,
                "resolution": int(dataset.dataset_config.resolution), "native_batch_index": batch_index,
                "native_file_indices": list(file_indices), "paths": paths,
                "compiled_token_lengths": text_lengths, "crop_shapes_hw": crop_shapes,
                "estimated_image_tokens": image_tokens, "activation_rank_proxy": score,
                "candidate_batch_count": len(indices), "maximum_caption_tokens": max_text,
                "maximum_image_tokens": max_image})
        largest = max(candidates, key=lambda row: (row["activation_rank_proxy"], row["maximum_image_tokens"],
                                                  row["maximum_caption_tokens"], -row["native_batch_index"]))
        longest = max(candidates, key=lambda row: (row["maximum_caption_tokens"], row["activation_rank_proxy"],
                                                  -row["native_batch_index"]))
        largest = {**largest, "selection_reasons": ["largest_combined_activation_proxy"]}
        if longest["native_batch_index"] == largest["native_batch_index"]:
            largest["selection_reasons"].append("longest_expanded_caption")
            selected.append(largest)
        else:
            selected.extend((largest, {**longest, "selection_reasons": ["longest_expanded_caption"]}))
    return selected


def _generators(loader):
    result, seen = {}, set()
    for name, owner in (("loader", loader), ("sampler", getattr(loader, "sampler", None)),
                        ("batch_sampler", getattr(loader, "batch_sampler", None))):
        generator = getattr(owner, "generator", None)
        if isinstance(generator, torch.Generator) and id(generator) not in seen:
            result[name] = generator
            seen.add(id(generator))
    return result


def _cuda_memory(device):
    if torch.device(device).type != "cuda":
        return {"device": str(device), "cuda_available": False}
    torch.cuda.synchronize(device)
    return {"device": str(device), "cuda_available": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device)}


def _assert_frozen(backend):
    backend.assert_frozen()
    if any(parameter.requires_grad or parameter.grad is not None for parameter in backend.frozen_parameters()):
        raise RuntimeError("Mechanical probe found gradients or trainable state on frozen model parameters")


def _probe_one(backend, dataset, choice, config, manifest_by_path, collate, accelerator):
    device = backend.tokens.E.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    dto = None
    prepared = False
    try:
        # Native __getitem__ deep-copies these FileItemDTOs. It neither advances
        # the sampler nor calls setup_epoch; native collation is identical.
        dto = collate(dataset[choice["native_batch_index"]])
        batch = _prepare_batch(dto, backend, config, manifest_by_path)
        prepared = True  # prepare_batch already called the native DTO cleanup.
        if len(batch["qs"]) != len(choice["paths"]):
            raise RuntimeError("Mechanical probe's actual native batch size changed")
        with torch.enable_grad(), accelerator.autocast():
            conditioning = backend.encode(batch["qs"], mode="learned", gradients=True)
            sequence_lengths = [int(value.shape[0]) for value in conditioning.features]
            if sequence_lengths != choice["compiled_token_lengths"]:
                raise RuntimeError("Mechanical probe's encoded lengths differ from selection preflight")
            prediction = backend.predict(batch["zt"], batch["tau"], conditioning)
            loss = per_example_mse(prediction, batch["target"], batch.get("valid_mask")).mean()
            if not loss.requires_grad or not bool(torch.isfinite(loss)):
                raise FloatingPointError("Mechanical probe has a disconnected or nonfinite denoising objective")
            gradient, = torch.autograd.grad(loss, (backend.tokens.E,), create_graph=False, retain_graph=False)
        finite = bool(torch.isfinite(gradient).all())
        nonzero = bool((gradient != 0).any())
        if not finite or not nonzero or gradient.dtype != torch.float32:
            raise FloatingPointError("Mechanical probe must produce finite, nonzero FP32 gradients through Qwen and DiT")
        _assert_frozen(backend)
        evidence = {**choice, "probe": "native_batch_denoising_input_gradient",
            "status": "passed", "scope_limitation": LIMITATION, "optimizer_steps": 0,
            "training_update_counter_changed": False, "loss": float(loss.detach()),
            "actual_batch_size": len(batch["qs"]), "actual_latent_shape": list(batch["zt"].shape),
            "actual_text_sequence_lengths": sequence_lengths,
            "noise_fractions": batch["tau"].detach().float().cpu().tolist(),
            "gradient_finite": finite, "gradient_nonzero": nonzero,
            "gradient_norm": float(gradient.float().norm()),
            "gradient_dtype": str(gradient.dtype), "frozen_gradient_ownership_passed": True,
            "memory_measurement_scope": "native DTO preparation plus Qwen/DiT forward and autograd.grad",
            "memory": _cuda_memory(device), "seconds": time.perf_counter() - started}
        return evidence
    finally:
        if dto is not None and not prepared:
            # Preserve the original load/forward error if cleanup itself fails
            # because a native DTO was only partly constructed or cleaned.
            try:
                dto.cleanup()
            except Exception:
                pass


def run_memory_probes(backend, loader, config, manifest_by_path, recorder):
    """Run initial real-batch probes without touching tokens, grads or optimizer.

    The caller invokes this only at committed update zero. No engine/optimizer is
    accepted by this function, and it never obtains an iterator from the loader.
    """
    get_datasets, collate, get_accelerator = _native_tools()
    datasets = list(get_datasets(loader))
    if not datasets:
        raise ValueError("No expanded native datasets for mechanical memory probes")
    parameter = backend.tokens.E
    values_before = parameter.detach().clone()
    grad_before = parameter.grad
    grad_values_before = None if grad_before is None else grad_before.detach().clone()
    requires_grad_before = parameter.requires_grad
    data_before = [(dataset, getattr(dataset, "epoch_num", None), dataset.batch_indices,
                    deepcopy(dataset.batch_indices)) for dataset in datasets]
    evidence = []
    try:
        _assert_frozen(backend)
        with isolated_rng(config["gen2"]["execution"]["training_seed"], generators=_generators(loader)):
            choices = _candidate_batches(backend, datasets, manifest_by_path)
            recorder.event("memory_gradient_probes_started", selected_batches=len(choices),
                           expanded_datasets=len(datasets), scope_limitation=LIMITATION)
            accelerator = get_accelerator()
            parameter.requires_grad_(True)
            for index, choice in enumerate(choices, 1):
                print(f"[Gen2 v2 mechanical] memory/input-gradient batch {index}/{len(choices)} "
                      f"| resolution {choice['resolution']} | text tokens {choice['compiled_token_lengths']} "
                      f"| native crop {choice['crop_shapes_hw']}", flush=True)
                try:
                    result = _probe_one(backend, datasets[choice["expanded_dataset_index"]], choice,
                                        config, manifest_by_path, collate, accelerator)
                except Exception as error:
                    recorder.event("memory_gradient_probe_failed", **choice,
                                   error_type=type(error).__name__, error=str(error))
                    raise
                evidence.append(result)
                recorder.record("probes", result)
                recorder.event("memory_gradient_probe_passed", probe_index=index,
                               resolution=choice["resolution"], memory=result["memory"])
    finally:
        # autograd.grad leaves .grad unchanged. Check the claim and restore the
        # tiny mutable bank if a future backend violates this pure-probe contract.
        changed = []
        if not torch.equal(parameter.detach(), values_before):
            changed.append("token values")
            with torch.no_grad():
                parameter.copy_(values_before)
        if parameter.grad is not grad_before or (grad_before is not None and
                not torch.equal(grad_before, grad_values_before)):
            changed.append("token .grad")
            parameter.grad = grad_before
            if grad_before is not None:
                grad_before.copy_(grad_values_before)
        parameter.requires_grad_(requires_grad_before)
        for dataset, epoch, indices_reference, indices_values in data_before:
            if getattr(dataset, "epoch_num", None) != epoch:
                changed.append("dataset epoch")
                dataset.epoch_num = epoch
            if dataset.batch_indices is not indices_reference or dataset.batch_indices != indices_values:
                changed.append("native batch indices")
                indices_reference[:] = indices_values
                dataset.batch_indices = indices_reference
        if changed:
            raise RuntimeError("Mechanical probes unexpectedly mutated " + ", ".join(changed))
    recorder.event("memory_gradient_probes_complete", passed_batches=len(evidence),
                   optimizer_steps=0, sampler_advanced=False, state_preserved=True,
                   scope_limitation=LIMITATION)
    return evidence
