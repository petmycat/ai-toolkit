"""Scheduled numerical interventions and matched visual comparisons.

All probes run on fixed native-preprocessed real latents, in isolated RNG/state
contexts. This module deliberately does not assign an automated style score.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import time
from pathlib import Path

import torch

from .data import prepare_batch, validation_examples
from .objectives import per_example_mse
from .recording import append_rating_template, write_json


def bundle_identity(backend) -> tuple[str, dict]:
    from .diagnostics import module_hash
    hashes = {name: module_hash(module) for name, module in backend.components().items()}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


class Evaluation:
    def __init__(self, backend, config, recorder, root):
        self.backend, self.config, self.recorder, self.root = backend, config, recorder, Path(root)
        self.options = config["gen2"]["diagnostics"]
        self.seed = config["gen2"]["execution"]["diagnostic_seed"]
        self.examples = []
        self.validation = []
        self.references = {}
        self.sampled_requests = set()
        self.tensor_packets = 0
        self.tensor_bytes = 0
        self.external_diagnostic_bytes = lambda: 0
        self.model_identities = None

    def initialize(self, loader, manifest):
        from .diagnostics import isolated_rng
        from toolkit.data_loader import get_dataloader_datasets
        from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
        requested = self.options["probes"]["num_examples"]
        with isolated_rng(self.seed):
            validation = self.config["train"].get("validation_config")
            if validation:
                self.validation = validation_examples(self.backend.model, validation, self.config["trigger_word"],
                    memory_budget_bytes=self.options["tensor_memory_budget_mb"] * 1024 * 1024 - self.external_diagnostic_bytes())
            if self.options["probes"]["source"] == "validation":
                candidates = self.validation
                if len(candidates) < requested:
                    raise ValueError(f"Fixed probes need {requested} real validation examples; found {len(candidates)}")
                indices = torch.randperm(len(candidates))[:requested].tolist()
                self.examples = [{**candidates[i], "z0": candidates[i]["z0"].clone()} for i in indices]
            else:
                if len(manifest) < requested:
                    raise ValueError(f"Fixed probes need {requested} real training examples; found {len(manifest)}")
                selected = torch.randperm(len(manifest))[:requested].tolist()
                lookup = {}
                for dataset in get_dataloader_datasets(loader):
                    for index, item in enumerate(dataset.file_list):
                        lookup.setdefault(str(Path(item.path).resolve()), (dataset, index))
                by_path = {row["path"]: row for row in manifest}
                for index in selected:
                    row = manifest[index]
                    dataset, item_index = lookup[row["path"]]
                    dto = DataLoaderBatchDTO(file_items=[dataset._get_single_item(item_index)])
                    batch = prepare_batch(dto, self.backend.model, self.config, by_path)
                    self.examples.append({"sample_id": row["sample_id"], "q": batch["qs"][0],
                        "split": "training", "group": row["group"], "content_hash": row["content_hash"],
                        "z0": batch["z0"].detach().cpu(), "metadata": batch["metadata"][0]})
                    self._check_memory()
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            for example in [*self.examples, *self.validation]:
                self._check_memory(example["z0"].numel() * 4)
                example["noise"] = torch.randn(example["z0"].shape, generator=generator, dtype=torch.float32)
        self._check_memory()
        self._write_probe_packet()

    def _write_probe_packet(self):
        from .diagnostics import tensor_hash
        from .recording import sha256
        from .config import SPEC_SHA256
        target = self.root / "fixed_probe_packet.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        torch.save({"examples": self.examples, "validation": self.validation, "diagnostic_seed": self.seed}, temporary)
        temporary.replace(target)
        write_json(self.root / "probe_manifest.json", {
            "schema_version": "1.0.0", "spec_sha256": SPEC_SHA256,
            "packet_sha256": sha256(target), "packet_bytes": target.stat().st_size,
            "diagnostic_seed": self.seed, "taus": self.options["probes"]["taus"],
            "packet": target.name, "examples": [{k: v for k, v in example.items() if k not in ("z0", "noise")}
                | {"z0_sha256": tensor_hash(example["z0"]), "noise_sha256": tensor_hash(example["noise"])}
                for example in self.examples]})

    def state_dict(self):
        return {"examples": self.examples, "validation": self.validation, "references": self.references,
                "sampled_requests": sorted(self.sampled_requests), "tensor_packets": self.tensor_packets,
                "tensor_bytes": self.tensor_bytes}

    def retained_bytes(self):
        """Account for every retained latent/noise/reference tensor once."""
        seen = set()
        def count(value):
            if isinstance(value, torch.Tensor):
                key = (str(value.device), value.untyped_storage().data_ptr())
                if key in seen:
                    return 0
                seen.add(key)
                return value.untyped_storage().nbytes()
            if isinstance(value, dict):
                return sum(count(v) for v in value.values())
            if isinstance(value, (tuple, list)):
                return sum(count(v) for v in value)
            return 0
        return count((self.examples, self.validation, self.references))

    def _check_memory(self, extra_bytes=0):
        used = self.retained_bytes() + self.external_diagnostic_bytes() + extra_bytes
        budget = self.options["tensor_memory_budget_mb"] * 1024 * 1024
        if used > budget:
            raise MemoryError(f"Mandatory diagnostic tensors need {used} bytes, exceeding configured {budget:g} byte budget")

    def load_state_dict(self, state):
        self.examples, self.validation = state["examples"], state["validation"]
        self.references = state["references"]
        self.sampled_requests = set(state["sampled_requests"])
        self.tensor_packets, self.tensor_bytes = state["tensor_packets"], state["tensor_bytes"]
        self._check_memory()
        self._write_probe_packet()

    def gate_mode(self, update):
        phases = self.config["gen2"]["phases"]
        return "learned" if update > phases["warmup_updates"] + phases["refinement_updates"] else "one"

    @contextmanager
    def activation_context(self, enabled):
        if not enabled:
            yield
            return
        from .diagnostics import region_summaries
        def observe(adapter, inputs, base, residual, applied, regions):
            branch = self.backend.current_branch
            path = adapter.gen2_original_path
            role = "text_adapter" if "down_proj" in path else "diffusion"
            index = getattr(adapter, "gen2_example_index", None)
            taus = branch.tau.detach().float().cpu().tolist() if branch is not None else []
            if index is not None:
                taus = [taus[index]]
            if len(taus) != inputs.shape[0]:
                raise RuntimeError("Activation diagnostics need an exact noise time for each example")
            edges = self.options["time_bin_edges"]
            for i, tau in enumerate(taus):
                summaries = region_summaries(inputs[i:i+1], base[i:i+1], residual[i:i+1], applied[i:i+1],
                    {name: mask[i:i+1] for name, mask in regions.items()})
                time_bin = min(len(edges)-2, max(0, __import__("bisect").bisect_right(edges, tau)-1))
                self.recorder.record("activations", {"original_path": path, "family": role,
                    "example_index": index if index is not None else i,
                    "block_id": adapter.gen2_block_id, "branch": branch.name,
                    "tau": tau, "time_bin": time_bin, "time_bin_bounds": edges[time_bin:time_bin+2],
                    "time_bin_convention": "left_closed_right_open_except_final_closed",
                    "regions": summaries, "pass_type": "training", "measurement": "exact_detached_reduction",
                    "accumulation_index": getattr(self.backend, "active_accumulation_index", None),
                    **self._activation_provenance(index if index is not None else i)})
        with self.backend.diagnostics(observe):
            yield

    def _activation_provenance(self, index):
        examples = getattr(self.backend, "active_batch_metadata", [])
        if index >= len(examples):
            return {"sample_id": None, "split": None, "group": "unassigned", "provenance_reason": "fixture_without_data_provenance"}
        return {key: examples[index].get(key) for key in ("sample_id", "split", "group")}

    def _state(self, example, tau):
        model = self.backend.model
        z0 = example["z0"].to(model.device_torch, model.torch_dtype)
        noise = example["noise"].to(model.device_torch, model.torch_dtype)
        times = torch.tensor([tau], device=model.device_torch, dtype=torch.float32)
        zt = model.add_noise(z0, noise, times * 1000)
        return zt, times, noise.float() - z0.float()

    def numerical_probes(self, update, reasons):
        from .diagnostics import interaction_metrics, isolated_rng
        mode = self.gate_mode(update)
        with isolated_rng(self.seed), torch.no_grad():
            for example in self.examples:
                neutral = self.backend.encode([example["q"]], styled=False)
                styled = self.backend.encode([example["q"]], styled=True)
                from .backend_ideogram4 import EXPECTED_ACTIVATION_LAYERS
                suffix = styled.features[0][-self.config["gen2"]["conditioning"]["num_tokens"]:].float()
                packed = suffix.reshape(suffix.shape[0], -1, len(EXPECTED_ACTIVATION_LAYERS))
                self.recorder.record("probes", {"probe_type": "suffix_representation", "sample_id": example["sample_id"],
                    "split": example["split"], "group": example["group"], "tap_ids": list(EXPECTED_ACTIVATION_LAYERS),
                    "rms_by_tap": packed.square().mean((0, 1)).sqrt().cpu().tolist(),
                    "R_T": float(styled.rt_per_example.mean()), "measurement": "exact_current_conditioning"})
                for tau in self.options["probes"]["taus"]:
                    zt, times, target = self._state(example, tau)
                    velocities = {}
                    for name, enabled, condition in (("v11", True, styled), ("v10", True, neutral),
                                                     ("v01", False, styled), ("v00", False, neutral)):
                        with self.backend.branch(times, lora_enabled=enabled, gate_mode=mode, name=f"probe_{name}"):
                            velocities[name] = self.backend.predict(zt, times, condition).float()
                    key = f"{example['sample_id']}:{tau}"
                    if key not in self.references:
                        if update != 0:
                            raise RuntimeError("Missing initialized frozen-base reference on resumed probe")
                        self._check_memory(velocities["v00"].numel() * velocities["v00"].element_size())
                        self.references[key] = velocities["v00"].cpu()
                    reference = self.references[key].to(zt.device)
                    drift_ok = torch.allclose(velocities["v00"], reference,
                        atol=self.options["prefix_atol"], rtol=self.options["prefix_rtol"])
                    result = interaction_metrics(**velocities, target=target, reference_v00=reference)
                    self.recorder.record("probes", {"probe_type": "velocity_interaction", "sample_id": example["sample_id"],
                        "split": example["split"], "group": example["group"], "tau": tau,
                        "metrics": result, "frozen_prediction_stable": drift_ok, "reasons": reasons,
                        "prefix_difference_max": float((styled.features[0][:len(neutral.features[0])].float()-neutral.features[0].float()).abs().max())})
                    if not drift_ok:
                        raise RuntimeError("Frozen-base probe prediction drift exceeds configured tolerance")
                    self._optional_dump(update, example["sample_id"], tau, styled, velocities)
            try:
                prefix = self.backend.verify_prefix([e["q"] for e in self.examples],
                    atol=self.options["prefix_atol"], rtol=self.options["prefix_rtol"])
            except RuntimeError as error:
                for row in getattr(error, "records", []):
                    self.recorder.record("probes", {"probe_type": "prefix_acceptance", **row})
                raise
            for row in prefix:
                self.recorder.record("probes", {"probe_type": "prefix_acceptance", **row})

    def _optional_dump(self, update, sample_id, tau, styled, velocities):
        options = self.options["tensor_dumps"]
        if not options["enabled"]:
            return
        packet = {}
        if "suffix_features" in options["names"]:
            count = self.config["gen2"]["conditioning"]["num_tokens"]
            packet["suffix_features"] = styled.features[0][-count:].detach()
        if "probe_velocities" in options["names"]:
            packet.update(velocities)
        needed = sum(t.numel() * t.element_size() for t in packet.values())
        budget = options["max_total_mb"] * 1024 * 1024
        if self.tensor_packets >= options["max_packets"] or self.tensor_bytes + needed > budget:
            self.recorder.event("optional_tensor_dump_skipped", reason="packet_or_byte_budget", needed_bytes=needed)
            return
        folder = self.root / "tensor_dumps"
        folder.mkdir(exist_ok=True)
        filename = folder / f"{update:08d}_{sample_id}_{tau}.pt"
        temporary = filename.with_suffix(".tmp")
        torch.save({name: tensor.cpu() for name, tensor in packet.items()}, temporary)
        actual_bytes = temporary.stat().st_size
        if self.tensor_bytes + actual_bytes > budget:
            temporary.unlink()
            self.recorder.event("optional_tensor_dump_skipped", reason="serialized_byte_budget", needed_bytes=actual_bytes)
            return
        temporary.replace(filename)
        self.tensor_packets += 1
        self.tensor_bytes += filename.stat().st_size

    def gradient_probes(self, update):
        from .diagnostics import isolated_gradient_probe, isolated_rng, tensor_hash
        families = self.backend.parameter_families()
        saved_flags = {p: p.requires_grad for ps in families.values() for p in ps}
        losses = self.config["gen2"]["losses"]
        try:
            with isolated_rng(self.seed):
                for example in self.examples[:self.options["gradient_probe_examples"]]:
                    for tau in self.options["gradient_probe_taus"]:
                        zt, times, target = self._state(example, tau)
                        c0 = self.backend.encode([example["q"]], styled=False)
                        with torch.no_grad(), self.backend.branch(times, lora_enabled=False, name="gradient_probe_teacher"):
                            teacher = self.backend.predict(zt, times, c0).detach()
                        for probe_roles, names in ((["diffusion"], ("styled", "neutral_weighted")),
                                                   (["embedding", "text_adapter"], ("styled", "text_regularizer_weighted"))):
                            for role, parameters in families.items():
                                for p in parameters:
                                    p.requires_grad_(role in probe_roles)
                            def context_for(name):
                                @contextmanager
                                def objective():
                                    with torch.enable_grad(), self.backend.branch(times, gate_mode=self.gate_mode(update), name="gradient_probe_"+name):
                                        if name == "neutral_weighted":
                                            prediction = self.backend.predict(zt, times, c0)
                                            value = per_example_mse(prediction, teacher).mean() * losses["neutral_weight"]
                                        else:
                                            conditioning = self.backend.encode([example["q"]], styled=True, gradients="embedding" in probe_roles)
                                            if name == "text_regularizer_weighted":
                                                value = conditioning.rt_per_example.mean() * losses["text_adapter_weight"]
                                            else:
                                                value = per_example_mse(self.backend.predict(zt, times, conditioning), target).mean()
                                        yield value
                                return objective
                            result = isolated_gradient_probe({name: context_for(name) for name in names},
                                {role: families[role] for role in probe_roles},
                                max_coordinates=self.options["gradient_probe_max_coordinates"], seed=self.seed,
                                memory_budget_mb=max(self.options["tensor_memory_budget_mb"] * 1024 * 1024
                                    - self.retained_bytes() - self.external_diagnostic_bytes(), 0) / (1024 * 1024))
                            # Uniform coordinate selections are immutable evidence,
                            # saved once per family. Probe rows reference their full
                            # hash instead of duplicating tens of thousands of IDs.
                            for role, family_result in result["families"].items():
                                indices = family_result.pop("coordinate_indices")
                                coordinate_hash = tensor_hash(torch.tensor(indices, dtype=torch.int64))
                                relative_path = f"gradient_coordinates/{role}.json"
                                destination = self.root / relative_path
                                provenance = {"schema_version": "1.0.0", "family": role,
                                    "total_coordinates": family_result["total_coordinates"],
                                    "sample_size": len(indices), "coordinate_sha256": coordinate_hash,
                                    "sampling": "uniform_without_replacement", "diagnostic_seed": self.seed,
                                    "indices": indices}
                                if destination.exists():
                                    existing = json.loads(destination.read_text(encoding="utf-8"))
                                    if existing != provenance:
                                        raise RuntimeError(f"Fixed gradient coordinate provenance changed for {role}")
                                else:
                                    write_json(destination, provenance)
                                    self.recorder.event("gradient_coordinates_initialized", family=role,
                                        path=relative_path, coordinate_sha256=coordinate_hash, sample_size=len(indices))
                                family_result["coordinate_reference"] = relative_path
                                family_result["coordinate_sha256"] = coordinate_hash
                            self.recorder.record("probes", {"probe_type": "gradient_decomposition", "sample_id": example["sample_id"],
                                "split": example["split"], "group": example["group"], "tau": tau, "results": result})
        finally:
            for p, flag in saved_flags.items():
                p.requires_grad_(flag)

    def representations(self, spectra=False):
        from .diagnostics import gate_statistics, low_rank_spectrum, token_statistics
        with torch.no_grad():
            gates = self.backend.gates
            self.recorder.record("gates", {"grid": gates.grid.detach().cpu().tolist(),
                **gate_statistics(gates.beta, gates(gates.grid), gates.derivative(gates.grid), float(gates.rho))})
            self.recorder.record("probes", {"probe_type": "token_representation", **token_statistics(
                self.backend.tokens.U, self.backend.tokens(), self.backend.tokens.e_init)})
            if spectra:
                for role, network in (("diffusion", self.backend.diffusion_network), ("text_adapter", self.backend.text_network)):
                    for adapter in network.unet_loras:
                        scale = float(adapter.scale)
                        row = low_rank_spectrum(adapter.lora_down.weight, adapter.lora_up.weight, scale=scale)
                        self.recorder.record("spectra", {"family": role, "original_path": adapter.gen2_original_path, **row})

    def validate(self, update):
        from .diagnostics import isolated_rng
        native = self.config["train"].get("validation_config")
        if not native:
            return
        sigmas = native.get("validation_sigmas", [1.0, .75, .5, .25])
        with isolated_rng(self.seed), torch.no_grad():
            for example in self.validation:
                c0 = self.backend.encode([example["q"]], styled=False)
                cp = self.backend.encode([example["q"]], styled=True)
                for sigma in sigmas:
                    zt, times, target = self._state(example, sigma)
                    values = {}
                    for name, condition, enabled in (("base", c0, False), ("styled", cp, True), ("neutral", c0, True)):
                        with self.backend.branch(times, lora_enabled=enabled, gate_mode=self.gate_mode(update), name="validation_"+name):
                            values[name] = self.backend.predict(zt, times, condition).float()
                    self.recorder.record("probes", {"probe_type": "held_out_validation", "sample_id": example["sample_id"],
                        "split": "validation", "group": example["group"], "tau": sigma,
                        "styled_mse": float(per_example_mse(values["styled"], target).mean()),
                        "neutral_mse": float(per_example_mse(values["neutral"], values["base"]).mean())})

    def sample(self, update, modes, seeds, reasons, package_hash, component_hashes):
        from .diagnostics import isolated_rng
        from .inference import generate
        from .provenance import code_identity
        from toolkit.config_modules import GenerateImageConfig
        from PIL import Image, ImageDraw
        options = self.config["sample"]
        evaluation = self.config["gen2"]["evaluation"]
        records, images = [], []
        with isolated_rng(self.seed):
            for prompt_index, prompt in enumerate(options["prompts"]):
                if not isinstance(prompt, str):
                    prompt = prompt["prompt"]
                prompt_id = f"p{prompt_index:03d}"
                for seed in seeds:
                    for mode in modes:
                        settings = {"package_hash": package_hash, "prompt": prompt, "seed": seed, "mode": mode,
                            "width": options["width"], "height": options["height"], "steps": options["sample_steps"],
                            "guidance": options["guidance_scale"], "strength": self.config["gen2"]["inference"]["lora_strength"],
                            "gate_mode": self.gate_mode(update)}
                        request = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
                        if request in self.sampled_requests:
                            continue
                        image, metadata = generate(self.backend, **{key: value for key, value in settings.items() if key != "package_hash"})
                        filename = f"u{update:08d}_{prompt_id}_s{seed}_{mode}_{request[:10]}.png"
                        destination = self.root / "samples" / filename
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        native = GenerateImageConfig(prompt=prompt, output_path=str(destination), output_ext="png", seed=seed,
                            width=options["width"], height=options["height"], num_inference_steps=options["sample_steps"],
                            guidance_scale=options["guidance_scale"], add_prompt_file=False)
                        native.save_image_atomic(image)
                        row = {"image_id": request, "prompt_id": prompt_id, "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                            "group": evaluation["prompt_groups"].get(prompt_id, "unassigned"), "checkpoint_hash": package_hash,
                            "component_hashes": component_hashes, "model_identities": self.model_identities,
                            "path": f"samples/{filename}", "ablation_mode": mode,
                            "reasons": reasons, "inference_code": code_identity(), **metadata}
                        write_json(destination.with_suffix(".json"), row)
                        self.recorder.record("samples/manifest", row)
                        self.sampled_requests.add(request)
                        records.append(row)
                        images.append((image, f"{prompt_id} / {mode} / seed {seed}"))
        append_rating_template(self.root / "human_ratings.csv", records)
        if images and evaluation["make_contact_sheets"]:
            cell = 256
            columns = min(4, len(images))
            rows = (len(images) + columns - 1) // columns
            sheet = Image.new("RGB", (columns * cell, rows * (cell + 34)), "white")
            draw = ImageDraw.Draw(sheet)
            for i, (image, label) in enumerate(images):
                thumbnail = image.copy()
                thumbnail.thumbnail((cell, cell))
                x, y = (i % columns) * cell, (i // columns) * (cell + 34)
                sheet.paste(thumbnail, (x, y))
                draw.text((x + 4, y + cell + 4), label, fill="black")
            sheet.save(self.root / "samples" / f"u{update:08d}_contact_sheet.png")
