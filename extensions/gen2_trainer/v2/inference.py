"""Token-package inference using native Ideogram velocity and Euler schedule."""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def generate(backend, prompt, mode=None, *, width=1024, height=1024, seed=42,
             steps=30, guidance=7., compiled=None, progress_callback=None):
    from diffusers.utils.torch_utils import randn_tensor
    from PIL import Image
    from extensions_built_in.diffusion_models.ideogram4.src.pipeline import get_ideogram4_sigmas
    model = backend.model
    divisor = model.vae_scale_factor * model.patch_size
    if width < divisor or height < divisor or width % divisor or height % divisor:
        raise ValueError(f"Ideogram image dimensions must be positive multiples of {divisor}")
    if steps < 1 or not math.isfinite(guidance) or guidance < 0 or seed < 0:
        raise ValueError("Sampling needs positive steps, nonnegative guidance and seed")
    if mode is None:
        mode = "learned" if "[trigger]" in prompt or backend.trigger_word in prompt else "base"
    if mode not in ("base", "named", "init", "learned"):
        raise ValueError(f"Unknown V2 sampling mode: {mode}")
    condition = backend.encode([prompt], mode=mode, gradients=False,
                               compiled=[compiled] if compiled is not None else None)
    kwargs = model.model_config.model_kwargs
    mu, std = float(kwargs.get("ideogram_schedule_mu", 0.)), float(kwargs.get("ideogram_schedule_std", 1.75))
    sigmas = get_ideogram4_sigmas(steps, width, height, mu=mu, std=std, device=model.device_torch)
    shape = (1, model.transformer.config.in_channels, height // divisor, width // divisor)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = randn_tensor(shape, generator=generator, device=model.device_torch, dtype=torch.float32) * sigmas[0]
    for index, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:]), 1):
        tau = sigma.expand(1)
        conditional = backend.predict(latents, tau, condition)
        if guidance > 1:
            unconditional = backend.predict_unconditional(latents, tau)
            velocity = unconditional + guidance * (conditional - unconditional)
        else:
            # Match native Ideogram's guidance switch exactly.
            velocity = conditional
        latents = latents + velocity.float() * (sigma_next - sigma)
        if not bool(torch.isfinite(latents).all()):
            raise FloatingPointError("Nonfinite V2 sampling latent")
        if progress_callback is not None:
            progress_callback(index, len(sigmas) - 1)
    pixels = model.decode_latents(latents, device=model.device_torch, dtype=model.torch_dtype)
    pixels = ((pixels.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    image = Image.fromarray(pixels.permute(0, 2, 3, 1)[0].cpu().numpy())
    metadata = {"trainer_version": "2.0.0", "mode": mode, "prompt": prompt,
        "conditioning": condition.metadata[0], "seed": seed, "width": width, "height": height,
        "steps": steps, "guidance_scale": guidance, "sampler": "native_ideogram_euler",
        "sigma_schedule": sigmas.cpu().tolist(), "schedule_mu": mu, "schedule_std": std,
        "rng_backend": "torch.Generator(cpu)", "diffusion_lora_enabled": False,
        "text_adapter_enabled": False, "unconditional_backend": "original_transformer",
        "unconditional_model_source": backend.config["gen2"]["inference"]["unconditional_model_path"],
        "unconditional_branch_executed": guidance > 1,
        "unconditional_activator_enabled": False}
    return image, metadata


class LoadedV2Package:
    def __init__(self, backend, manifest, config, path):
        self.backend, self.manifest, self.config, self.path = backend, manifest, config, path

    def generate(self, prompt, **kwargs):
        from ..diagnostics import isolated_rng
        old = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(self.config["gen2"]["execution"]["deterministic_algorithms"])
            with isolated_rng():
                image, metadata = generate(self.backend, prompt, **kwargs)
        finally:
            torch.use_deterministic_algorithms(old, warn_only=warn)
        metadata.update(spec_sha256=self.manifest["spec_sha256"], checkpoint=str(self.path),
                        logical_update=self.manifest["logical_update"],
                        model_identities=self.manifest["metadata"]["model_identities"])
        return image, metadata

    def release(self):
        import gc
        self.backend = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_package(path, *, device=None):
    import copy
    from pathlib import Path
    from extensions_built_in.diffusion_models.ideogram4.ideogram4 import Ideogram4Model
    from ..diagnostics import isolated_rng
    from ..original_unconditional import load_original_unconditional
    from ..package import tokenizer_identity
    from ..provenance import frozen_model_hashes
    from .config import resolve_process_config, SPEC_SHA256
    from .checkpoint import V2CheckpointManager, load_manifest
    from .backend import V2Backend
    from .process import native_configuration
    path = Path(path).expanduser().resolve()
    manifest = load_manifest(path, expected_sha=SPEC_SHA256)
    config = resolve_process_config(copy.deepcopy(manifest["metadata"]["resolved_config"]))
    if device is not None:
        config["device"] = device
    if not str(config["device"]).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V2 package inference requires the real CUDA Ideogram backend")
    with isolated_rng(config["gen2"]["execution"]["training_seed"]):
        native = native_configuration(config)
        model = Ideogram4Model(config["device"], native["model"], dtype=config["train"]["dtype"])
        model.load_model()
        load_original_unconditional(model, config["gen2"])
        expected = {"model_identities": frozen_model_hashes(model), "tokenizer_identity": tokenizer_identity(model)}
        backend = V2Backend(model, config)
        manager = V2CheckpointManager(path.parent, path / "specification.md", SPEC_SHA256)
        manager.load_inference(path, backend, expected_metadata=expected)
        backend.tokens.requires_grad_(False)
        backend.assert_frozen()
    return LoadedV2Package(backend, manifest, config, path)
