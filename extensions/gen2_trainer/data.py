"""Native data integration, exact state sampling, and deterministic loader replay.

There is no alternate image pipeline or latent cache here. The small native
dataset subclass changes failure policy only: an unreadable item aborts a run.
"""
from __future__ import annotations

import copy
import bisect
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_ROOT = REPO_ROOT / "gen2"


def assert_writable_path(path: str | Path) -> Path:
    """Resolve symlinks before allowing even incidental cache/output writes."""
    resolved = Path(path).expanduser().resolve()
    if resolved == PROTECTED_ROOT.resolve() or PROTECTED_ROOT.resolve() in resolved.parents:
        raise ValueError(f"Protected read-only input directory: {resolved}. Use a working copy outside {PROTECTED_ROOT}.")
    return resolved


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_caption(caption: str, trigger: str) -> dict:
    from .conditioning import compile_trigger
    return compile_trigger(caption, trigger)


def preflight_datasets(config: dict) -> list[dict]:
    """Verify files before the native loader can skip a broken source item."""
    from PIL import Image
    from toolkit.data_loader import image_extensions
    rows: list[dict] = []
    group_labels = {}
    groups_file = config["gen2"]["data"]["content_groups_file"]
    if groups_file:
        with Path(groups_file).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"relative_path", "group"}.issubset(reader.fieldnames or []):
                raise ValueError("Content groups CSV requires relative_path and group columns")
            if len(config["datasets"]) > 1 and "dataset_index" not in (reader.fieldnames or []):
                raise ValueError("Multiple data directories require dataset_index in content groups CSV")
            for item in reader:
                key = (int(item.get("dataset_index") or 0), Path(item["relative_path"]).as_posix())
                if key in group_labels:
                    raise ValueError(f"Duplicate content-group key: {key}")
                group_labels[key] = item["group"]
    for dataset_index, dataset in enumerate(config["datasets"]):
        source = Path(dataset.get("dataset_path") or dataset["folder_path"]).expanduser().resolve()
        assert_writable_path(source if source.is_dir() else source.parent)
        if not source.exists():
            raise FileNotFoundError(f"Dataset {dataset_index}: {source}")
        captions = None
        root = source if source.is_dir() else source.parent
        if source.is_file():
            captions = json.loads(source.read_text(encoding="utf-8"))
            # The native JSON loader resolves path keys against cwd, not the
            # JSON directory. Keep that behavior, but retain the original keys.
            captions = {str(Path(p).resolve()): entry for p, entry in captions.items()}
            paths = [Path(p) for p in captions]
        else:
            paths = sorted(p for p in source.rglob("*") if p.is_file()
                           and p.suffix.lower() in image_extensions
                           and not any(part.startswith(".") for part in p.relative_to(source).parts)
                           and p.parent.name != "_controls")
        if not paths:
            raise ValueError(f"Dataset {dataset_index} contains no supported images: {source}")
        for path in paths:
            assert_writable_path(path.parent)
            with Image.open(path) as image:
                dimensions = [image.width, image.height]
                image.verify()
            ext = dataset.get("caption_ext", "txt").lstrip(".")
            caption_path = path.with_suffix("." + ext)
            if captions is not None:
                entry = captions[str(path)]
                caption = entry.get("caption_short", entry.get("caption")) if dataset.get("use_short_captions") else entry.get("caption")
                provenance = str(source)
            elif caption_path.exists():
                caption = caption_path.read_text(encoding="utf-8")
                if not caption.strip() and dataset.get("default_caption") is not None:
                    caption = dataset["default_caption"]
                provenance = str(caption_path)
            else:
                caption = dataset.get("default_caption")
                provenance = "native_default_caption"
            if not isinstance(caption, str):
                raise ValueError(f"Missing content caption for {path}")
            compiled = canonical_caption(caption, config["trigger_word"])
            content_hash = file_hash(path)
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                relative = str(path)
            sample_id = hashlib.sha256(f"{dataset_index}:{relative}:{content_hash}".encode()).hexdigest()[:24]
            rows.append({"sample_id": sample_id, "dataset_index": dataset_index,
                         "path": str(path), "relative_path": relative, "content_hash": content_hash,
                         "source_dimensions": dimensions, "split": "training", "group": group_labels.get((dataset_index, relative), "unassigned"),
                         "caption_provenance": provenance, **compiled,
                         "caption_hash": hashlib.sha256(caption.encode()).hexdigest(),
                         "canonical_caption_hash": hashlib.sha256(compiled["q"].encode()).hexdigest()})
    hashes = {row["content_hash"] for row in rows}
    paths = [row["path"] for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("Dataset sources overlap the same image path; use native num_repeats or resolution lists within one dataset so caption/group provenance stays unambiguous")
    validation = config["train"].get("validation_config") or {}
    for item in validation.get("validation_items", []):
        path = Path(item["image_path"])
        with Image.open(path) as image:
            image.verify()
        if config["gen2"]["data"]["reject_train_validation_duplicates"] and file_hash(path) in hashes:
            raise ValueError(f"Validation image duplicates training image bytes: {path}")
    return rows


def make_native_loader(config: dict, model: Any, manifest: list[dict]):
    from toolkit.config_modules import DatasetConfig, preprocess_dataset_raw_config
    from toolkit.data_loader import get_dataloader_from_datasets, get_dataloader_datasets
    from .native_data import Gen2AbortDataset

    native, source_indices = [], []
    for source_index, source in enumerate(config["datasets"]):
        for raw in preprocess_dataset_raw_config([copy.deepcopy(source)]):
            raw["trigger_word"] = None
            item = DatasetConfig(**raw)
            item.gen2_initialization_seed = config["gen2"]["execution"]["training_seed"] + len(native)
            native.append(item)
            source_indices.append(source_index)
    loader = get_dataloader_from_datasets(native, batch_size=config["train"]["batch_size"], sd=model, dataset_class=Gen2AbortDataset)
    datasets = get_dataloader_datasets(loader)
    if len(datasets) != len(native):
        raise RuntimeError("Native loader changed the number of datasets after resolution expansion")
    # Check every expanded dataset: a surviving copy at another resolution
    # must not conceal a native constructor skipping an invalid source item.
    for dataset, native_config, source_index in zip(datasets, native, source_indices):
        observed = {str(Path(item.path).resolve()) for item in dataset.file_list}
        expected = {row["path"] for row in manifest if row["dataset_index"] == source_index}
        if observed != expected:
            raise RuntimeError(
                f"Native loader changed dataset membership for source {source_index} "
                f"at resolution {native_config.resolution}: "
                f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
            )
    return loader


class NativeDataStream:
    """Own only native sampler RNG/cursor, preserving native bucket/transforms."""
    def __init__(self, loader, seed: int):
        self.loader = loader
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.loader.generator = self.generator
        self.loader.sampler.generator = self.generator
        self.epoch = 0
        self.cursor = 0
        self.iterator = None
        self.epoch_rng = None
        self.epoch_generator = None

    def _begin(self):
        from .diagnostics import capture_rng_state
        self.cursor = 0
        self.epoch_rng = capture_rng_state()
        self.epoch_generator = self.generator.get_state().clone()
        self.iterator = iter(self.loader)

    def next(self):
        if self.iterator is None:
            self._begin()
        try:
            batch = next(self.iterator)
        except StopIteration:
            from toolkit.data_loader import trigger_dataloader_setup_epoch
            self.epoch += 1
            trigger_dataloader_setup_epoch(self.loader)
            self._begin()
            batch = next(self.iterator)
        self.cursor += 1
        return batch

    def state_dict(self):
        from toolkit.data_loader import get_dataloader_datasets
        return {"epoch": self.epoch, "cursor": self.cursor, "epoch_rng": self.epoch_rng,
                "epoch_generator": self.epoch_generator, "generator": self.generator.get_state(),
                "datasets": [{"batch_indices": copy.deepcopy(getattr(ds, "batch_indices", None)),
                              "epoch_num": ds.epoch_num,
                              "paths": [str(Path(item.path).resolve()) for item in ds.file_list]}
                             for ds in get_dataloader_datasets(self.loader)],
                "num_workers": self.loader.num_workers,
                "replay_limit": None if self.loader.num_workers == 0 else "Prefetched worker augmentation state is not restorable; zero workers is reference mode"}

    def load_state_dict(self, state):
        from .diagnostics import capture_rng_state, restore_rng_state
        from toolkit.data_loader import get_dataloader_datasets
        if state["num_workers"] != self.loader.num_workers:
            raise ValueError("Strict resume requires unchanged loader worker count")
        datasets = get_dataloader_datasets(self.loader)
        if len(datasets) != len(state["datasets"]):
            raise ValueError("Dataset count changed on resume")
        for ds, saved in zip(datasets, state["datasets"]):
            if saved["paths"] != [str(Path(item.path).resolve()) for item in ds.file_list]:
                raise ValueError("Native dataset ordering changed on resume")
            if saved["batch_indices"] is not None:
                ds.batch_indices = copy.deepcopy(saved["batch_indices"])
            ds.epoch_num = saved["epoch_num"]
        self.epoch, self.cursor = state["epoch"], state["cursor"]
        self.epoch_rng, self.epoch_generator = state["epoch_rng"], state["epoch_generator"]
        if self.epoch_rng is None:
            self.generator.set_state(state["generator"])
            return
        current = capture_rng_state()
        try:
            restore_rng_state(self.epoch_rng)
            self.generator.set_state(self.epoch_generator)
            self.iterator = iter(self.loader)
            for _ in range(self.cursor):
                batch = next(self.iterator)
                batch.cleanup()
        finally:
            restore_rng_state(current)
            self.generator.set_state(state["generator"])


def prepare_batch(dto, model, config: dict, manifest_by_path: dict) -> dict:
    device, dtype = model.device_torch, model.torch_dtype
    with torch.no_grad():
        z0 = dto.latents
        if z0 is None:
            z0 = model.encode_images(dto.tensor, device=device, dtype=dtype)
        # Gen2 is image-only; persist plain tensors, not a native DTO subclass
        # requiring an unsafe pickle allowlist when loading a probe packet.
        z0 = getattr(z0, "tensor", z0)
        z0 = z0.to(device=device, dtype=dtype).detach()
        noise = model.get_latent_noise_from_latents(z0, noise_offset=0.0)
        table = model.noise_scheduler.timesteps.to(device=device, dtype=torch.float32)
        indices = torch.randint(0, table.numel(), (z0.shape[0],), device=device)
        timestep = table[indices]
        tau = timestep / 1000.0
        zt = model.add_noise(z0, noise, timestep).detach()
        target = (noise.float() - z0.float()).detach()
    metadata, qs = [], []
    for index, item in enumerate(dto.file_items):
        path = str(Path(item.path).resolve())
        source = manifest_by_path[path]
        # Native load_caption() cleans/reformats sidecars. The immutable
        # preflight text preserves JSON, commas, and internal whitespace.
        raw = source["original_caption"]
        compiled = canonical_caption(raw, config["trigger_word"])
        qs.append(compiled["q"])
        transforms = {}
        for key in ("crop_x", "crop_y", "crop_width", "crop_height", "flip_x", "flip_y", "scale_to_width", "scale_to_height", "bucket_width", "bucket_height"):
            value = getattr(item, key, None)
            if isinstance(value, (int, float, bool, str)):
                transforms[key] = value
        def rms(t):
            return float(t[index].float().square().mean().sqrt().item())
        edges = config["gen2"]["diagnostics"]["time_bin_edges"]
        time_bin = min(len(edges)-2, max(0, bisect.bisect_right(edges, float(tau[index]))-1))
        metadata.append({"example_id": source["sample_id"], "sample_id": source["sample_id"],
                         "dataset_index": source["dataset_index"], "split": "training", "group": source["group"],
                         **compiled, "time_table_index": int(indices[index].item()),
                         "time_bin": time_bin, "time_bin_bounds": edges[time_bin:time_bin+2],
                         "timestep": float(timestep[index].item()), "tau": float(tau[index].item()),
                         "internal_ideogram_time": float(1 - tau[index].item()),
                         "latent_shape": list(z0[index].shape), "valid_elements": z0[index].numel(),
                         "z0_rms": rms(z0), "noise_rms": rms(noise), "z_tau_rms": rms(zt), "target_rms": rms(target),
                         "transforms": transforms, "augmentation_replay_seed": None,
                         "augmentation_replay_reason": "Native transform does not expose a per-example replay seed; run RNG and loader cursor saved"})
    dto.cleanup()
    return {"qs": qs, "z0": z0, "noise": noise, "tau": tau, "zt": zt, "target": target, "metadata": metadata}


def validation_examples(model, validation_config, trigger: str, *, memory_budget_bytes=None) -> list[dict]:
    """Same native validation item parsing, bucket resize, and VAE encoding."""
    from PIL import Image, ImageOps
    from torchvision import transforms
    from toolkit.buckets import get_bucket_for_image_size
    from toolkit.config_modules import ValidationConfig
    native = ValidationConfig(**validation_config)
    results = []
    retained = 0
    with torch.no_grad():
        for index, item in enumerate(native.validation_items):
            with Image.open(item.image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            bucket = get_bucket_for_image_size(image.width, image.height, resolution=native.resolution,
                                               divisibility=model.get_bucket_divisibility())
            image = image.resize((bucket["width"], bucket["height"]), Image.BICUBIC)
            tensor = transforms.ToTensor()(image) * 2.0 - 1.0
            z0 = model.encode_images([tensor], device=model.device_torch, dtype=model.torch_dtype)
            z0 = getattr(z0, "tensor", z0).detach().cpu()
            retained += z0.numel() * (z0.element_size() + 4)  # reserved fixed fp32 noise
            if memory_budget_bytes is not None and retained > memory_budget_bytes:
                raise MemoryError("Held-out latent/noise packet exceeds the configured diagnostic tensor budget")
            results.append({"sample_id": f"validation-{index:04d}-{file_hash(item.image_path)[:16]}",
                            "path": item.image_path, "content_hash": file_hash(item.image_path),
                            "q": canonical_caption(item.prompt, trigger)["q"], "split": "validation",
                            "group": "unassigned", "z0": z0})
    return results
