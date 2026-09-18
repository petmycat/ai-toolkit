"""Fixed native crops and noise packets, held on CPU between measurements."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from ..diagnostics import isolated_rng


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), value.dtype)).encode())
    # Dtype reinterpretation requires a dimension when element sizes differ.
    # TokenBank also stores scalar norm/seed buffers. Preserve their original
    # shape in the header above, then flatten only for the raw-byte view.
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def spaced_indices(length, count):
    if count >= length:
        return list(range(length))
    if count == 1:
        return [0]
    return [round(i * (length - 1) / (count - 1)) for i in range(count)]


def select_native_batches(datasets, compiler, manifest_by_path, count):
    choices = []
    for dataset_index, dataset in enumerate(datasets):
        candidates, seen = [], set()
        for batch_index, indices in enumerate(dataset.batch_indices):
            if len(indices) != 1:
                raise ValueError("Diagnostics require physical native batch size one")
            item = dataset.file_list[indices[0]]
            path = str(Path(item.path).resolve())
            if path in seen:
                continue
            seen.add(path)
            source = manifest_by_path[path]
            length = len(compiler.compile(source["original_caption"], require_trigger=True).ids)
            candidates.append({"expanded_dataset_index": dataset_index,
                "native_batch_index": batch_index, "path": path, "sample_id": source["sample_id"],
                "resolution": int(dataset.dataset_config.resolution), "compiled_token_length": length,
                "selection": "evenly_spaced_caption_lengths_including_shortest_and_longest"})
        if not candidates:
            raise ValueError("Diagnostic selection found an empty expanded dataset")
        candidates.sort(key=lambda row: (row["compiled_token_length"], row["path"]))
        choices.extend(candidates[index] for index in spaced_indices(len(candidates), count))
    return choices


def packets_from_latent(model, z0, qs, metadata, choice, settings):
    """Reuse each exact native Gaussian draw across every requested timestep."""
    z0 = z0.detach().to(model.device_torch, model.torch_dtype)
    result = []
    for noise_seed in settings["noise_seeds"]:
        with isolated_rng(noise_seed), torch.no_grad():
            noise = model.get_latent_noise_from_latents(z0, noise_offset=0.0)
        target = (noise.float() - z0.float()).detach().cpu()
        for noise_fraction in settings["noise_fractions"]:
            timestep = torch.full((len(qs),), noise_fraction * 1000., device=z0.device, dtype=torch.float32)
            tau = timestep / 1000.
            with torch.no_grad():
                zt = model.add_noise(z0, noise, timestep).detach().cpu()
            packet_id = (f"d{choice['expanded_dataset_index']}-{choice['sample_id']}"
                         f"-s{noise_seed}-t{noise_fraction:g}")
            row = {**choice, "packet_id": packet_id, "noise_seed": noise_seed,
                "tau": float(tau[0]), "latent_shape": list(z0.shape),
                "z0_sha256": tensor_digest(z0), "noise_sha256": tensor_digest(noise),
                "zt_sha256": tensor_digest(zt), "target_sha256": tensor_digest(target),
                "native_transforms": metadata[0].get("transforms", {})}
            result.append({"id": packet_id, "qs": list(qs), "zt": zt, "tau": tau.cpu(),
                "target": target, "metadata": [{**metadata[0], **row}], "diagnostic": row})
    return result


def build_fixed_packets(runner, settings, recorder):
    from toolkit.data_loader import get_dataloader_datasets, dto_collation
    from .process import prepare_batch
    datasets = list(get_dataloader_datasets(runner.loader))
    choices = select_native_batches(datasets, runner.backend.compiler, runner.manifest_by_path,
                                    settings["examples_per_resolution"])
    packet_count = len(choices) * len(settings["noise_seeds"]) * len(settings["noise_fractions"])
    print(f"[Gen2 v2 diagnostic] {len(choices)} native examples; {packet_count} fixed packets; "
          f"{packet_count * 4} paired diffusion forwards; "
          f"{settings['descent_steps']} temporary updates from each of two starting banks", flush=True)
    packets, retained = [], 0
    with isolated_rng(settings["seed"]):
        for index, choice in enumerate(choices, 1):
            print(f"[Gen2 v2 diagnostic] fixing example {index}/{len(choices)} | "
                  f"resolution {choice['resolution']} | text tokens {choice['compiled_token_length']}", flush=True)
            dto = dto_collation(datasets[choice["expanded_dataset_index"]][choice["native_batch_index"]])
            try:
                batch = prepare_batch(dto, runner.backend.model, runner.config, runner.manifest_by_path)
            except BaseException:
                try:
                    dto.cleanup()
                except Exception:
                    pass
                raise
            if batch["metadata"][0]["sample_id"] != choice["sample_id"]:
                raise RuntimeError("Selected native image differs from the loaded image")
            added = packets_from_latent(runner.backend.model, batch["z0"], batch["qs"],
                                        batch["metadata"], choice, settings)
            # Conservative accounting counts shared targets more than once.
            retained += sum(p[key].numel() * p[key].element_size()
                            for p in added for key in ("zt", "tau", "target"))
            if retained > settings["max_packet_memory_mb"] * 1024**2:
                raise MemoryError("Fixed packet storage exceeds diagnostic.max_packet_memory_mb")
            packets.extend(added)
            del batch, added, dto
    if not packets:
        raise ValueError("No diagnostic packets were built")
    recorder.event("fixed_packets_ready", packet_count=len(packets), native_examples=len(choices),
                   conservative_cpu_tensor_bytes=retained)
    fidelity = min(packets, key=lambda p: (p["diagnostic"]["resolution"],
        p["diagnostic"]["compiled_token_length"], abs(p["diagnostic"]["tau"] - .5), p["id"]))
    return packets, fidelity, {"selection": choices, "packets": [p["diagnostic"] for p in packets],
        "fidelity_packet_id": fidelity["id"], "conservative_cpu_tensor_bytes": retained,
        "scope": "Subset of training images; no held-out generalization or visual style acceptance is measured."}
