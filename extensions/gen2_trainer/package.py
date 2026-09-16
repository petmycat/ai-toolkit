"""Strict complete-package loading for inference, without optimizer restoration."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path


@dataclass
class LoadedPackage:
    backend: object
    manifest: dict
    config: dict
    checkpoint_path: Path

    @property
    def gate_mode(self):
        phases = self.config["gen2"]["phases"]
        return "learned" if self.manifest["logical_update"] > phases["warmup_updates"]+phases["refinement_updates"] else "one"

    def generate(self, prompt, **kwargs):
        import torch
        from .inference import generate
        kwargs.setdefault("gate_mode", self.gate_mode)
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(self.config["gen2"]["execution"]["deterministic_algorithms"])
            image, metadata = generate(self.backend, prompt, **kwargs)
        finally:
            torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        metadata.update(package_hash=self.manifest["package_hash"],
                        checkpoint=str(self.checkpoint_path), logical_update=self.manifest["logical_update"],
                        spec_sha256=self.manifest["spec_sha256"],
                        model_identities=self.manifest["metadata"]["model_identities"],
                        component_files=self.manifest["components"])
        return image, metadata

    def release(self):
        import gc
        import torch
        self.backend = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def tokenizer_identity(model):
    return {"name_or_path": model.tokenizer.name_or_path,
            "vocabulary_sha256": hashlib.sha256(json.dumps(model.tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
            "chat_template": model.tokenizer.chat_template,
            "encoder_commit": getattr(model.text_encoder.config, "_commit_hash", None)}


def load_package(checkpoint_path, *, device=None) -> LoadedPackage:
    """Load native frozen weights and all four masters, verifying exact identities.

    Local base weights, tokenizer and module layout must match the checkpoint.
    The package does not contain or redistribute those original model weights.
    """
    import torch
    from .backend_ideogram4 import Ideogram4Backend
    from .checkpointing import CheckpointManager, CheckpointError, load_manifest
    from .config import resolve_process_config, SPEC_SHA256
    from .diagnostics import isolated_rng
    from .process import native_configuration
    from .provenance import frozen_model_hashes
    from .original_unconditional import load_original_unconditional
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    manifest = load_manifest(checkpoint_path, expected_spec_sha256=SPEC_SHA256)
    metadata = manifest["metadata"]
    required = {"resolved_config", "model_identities", "tokenizer_identity", "module_mapping"}
    if not required <= metadata.keys():
        raise CheckpointError(f"Package metadata missing {sorted(required-metadata.keys())}")
    config = resolve_process_config(copy.deepcopy(metadata["resolved_config"]))
    if device is not None:
        config["device"] = device
    if not str(config["device"]).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Gen2 complete-package inference requires the real CUDA Ideogram backend")
    from extensions_built_in.diffusion_models.ideogram4.ideogram4 import Ideogram4Model
    native = native_configuration(config)
    # Reproduce native model quantization/init RNG, then restore the caller RNG.
    with isolated_rng(config["gen2"]["execution"]["training_seed"]):
        model = Ideogram4Model(config["device"], native["model"], dtype=config["train"]["dtype"])
        model.load_model()
        load_original_unconditional(model, config["gen2"])
        identities = frozen_model_hashes(model)
        backend = Ideogram4Backend.from_native(model, config["gen2"], config["network"],
            config["trigger_word"], gradient_checkpointing=config["train"]["gradient_checkpointing"])
        expected = {"model_identities": identities, "tokenizer_identity": tokenizer_identity(model),
            "module_mapping": [{key: value for key, value in row.items() if key != "device"}
                               for row in backend.module_manifest()]}
        # This manager only reads. The embedded immutable copy avoids depending
        # on the original machine's configured spec path.
        manager = CheckpointManager(checkpoint_path.parent, checkpoint_path/"specification.md", SPEC_SHA256, read_only=True)
        manager.load(checkpoint_path, expected_metadata=expected, components=backend.components(), load_training_state=False)
        for parameters in backend.parameter_families().values():
            for parameter in parameters:
                parameter.requires_grad_(False)
                parameter.grad = None
        backend.assert_frozen()
    return LoadedPackage(backend, manifest, config, checkpoint_path)
