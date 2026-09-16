"""Complete-package conditional routing and native Ideogram Euler sampling."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from .conditioning import compile_trigger


@dataclass(frozen=True)
class InferenceRoute:
    mode: str
    styled: bool
    lora_enabled: bool
    token_mode: str = "learned"
    adapter_enabled: bool = True
    gate_mode: str = "learned"
    unconditional_lora_strength: float = 0.


def resolve_route(mode: str | None, trigger_present: bool,
                  missing_trigger_policy="learned_neutral", gate_mode="learned"):
    if missing_trigger_policy not in ("learned_neutral", "base_bypass"):
        raise ValueError(f"Unknown missing-trigger policy {missing_trigger_policy}")
    if mode is None:
        mode = "full" if trigger_present else ("neutral_lora_on" if missing_trigger_policy == "learned_neutral" else "base")
    routes = {
        "full": InferenceRoute("full", True, True, gate_mode=gate_mode),
        "neutral_lora_on": InferenceRoute("neutral_lora_on", False, True, adapter_enabled=False, gate_mode=gate_mode),
        "base": InferenceRoute("base", False, False, adapter_enabled=False, gate_mode="bypassed"),
        "base_with_tokens": InferenceRoute("base_with_tokens", True, False, adapter_enabled=False, gate_mode="bypassed"),
        "base_with_conditioning": InferenceRoute("base_with_conditioning", True, False, gate_mode="bypassed"),
        "conditioning_init": InferenceRoute("conditioning_init", True, True, "init", False, gate_mode),
        "encoder_adapter_off": InferenceRoute("encoder_adapter_off", True, True, "learned", False, gate_mode),
        "tokens_init": InferenceRoute("tokens_init", True, True, "init", True, gate_mode),
        "gates_one": InferenceRoute("gates_one", True, True, gate_mode="one"),
        "gates_time_mean": InferenceRoute("gates_time_mean", True, True, gate_mode="time_mean"),
        "full_uncond_half": InferenceRoute("full_uncond_half", True, True, gate_mode=gate_mode,
                                           unconditional_lora_strength=.5),
        "full_uncond_full": InferenceRoute("full_uncond_full", True, True, gate_mode=gate_mode,
                                           unconditional_lora_strength=1.),
    }
    if mode not in routes:
        raise ValueError(f"Unknown Gen2 inference mode: {mode}")
    return routes[mode]


@torch.no_grad()
def generate(backend, prompt: str, mode: str | None = None, *, width=1024, height=1024,
             seed=42, steps=30, guidance=7., strength=None, gate_mode="learned",
             initial_noise=None, progress_callback=None):
    """Return one PIL image and reproducibility metadata, without mutating components.

    Explicit diagnostic modes force their route regardless of literal trigger.
    An omitted mode uses the package's production missing-trigger policy.
    The two full_uncond_* diagnostics apply the trained diffusion adapter to
    the empty image-only pass at absolute strengths .5 and 1.; these strengths
    are independent of the conditional strength. Learned tokens and the text
    adapter remain confined to the conditional pass.

    When supplied, ``progress_callback(completed_steps, total_steps)`` runs
    after each successful denoising step. It is observational: no extra device
    synchronization or sampling work is performed for progress reporting.
    """
    from diffusers.utils.torch_utils import randn_tensor
    from PIL import Image
    from extensions_built_in.diffusion_models.ideogram4.src.pipeline import get_ideogram4_sigmas
    model = backend.model
    divisor = model.vae_scale_factor*model.patch_size
    if width < divisor or height < divisor or width % divisor or height % divisor:
        raise ValueError(f"Ideogram image dimensions must be positive multiples of {divisor}")
    if steps < 1 or not math.isfinite(guidance) or guidance < 0 or seed < 0:
        raise ValueError("Sampling needs steps >=1, guidance >=0, seed >=0")
    setting = backend.config["inference"]
    strength = setting["lora_strength"] if strength is None else strength
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("Inference LoRA strength must be finite and nonnegative")
    compilation = compile_trigger(prompt, backend.trigger_word)
    route = resolve_route(mode, compilation["trigger_present"], setting["missing_trigger_policy"], gate_mode)
    if route.unconditional_lora_strength > 0 and guidance <= 1:
        raise ValueError("Unconditional LoRA comparisons require guidance >1 so native CFG runs both branches")
    condition = backend.encode([prompt], styled=route.styled, gradients=False,
                               token_mode=route.token_mode, adapter_enabled=route.adapter_enabled)
    kwargs = model.model_config.model_kwargs
    mu, std = float(kwargs.get("ideogram_schedule_mu", 0.)), float(kwargs.get("ideogram_schedule_std", 1.75))
    sigmas = get_ideogram4_sigmas(steps, width, height, mu=mu, std=std, device=model.device_torch)
    shape = (1, model.transformer.config.in_channels, height//divisor, width//divisor)
    # A private CPU generator never advances training or global CUDA RNG state.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if initial_noise is None:
        noise = randn_tensor(shape, generator=generator, device=model.device_torch, dtype=torch.float32)
    else:
        if tuple(initial_noise.shape) != shape:
            raise ValueError(f"Initial latent noise shape must be {shape}")
        noise = initial_noise.to(model.device_torch, torch.float32).clone()
    latents = noise*sigmas[0]
    empty = backend.empty_conditioning(1, condition.features[0].shape[-1])
    total_steps = len(sigmas) - 1
    for completed_steps, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:]), start=1):
        tau = sigma.expand(1)
        with backend.branch(tau, lora_enabled=route.lora_enabled, gate_mode=route.gate_mode,
                            strength=strength, name="cfg_conditional"):
            conditional = backend.predict(latents, tau, condition)
        # This comparison intentionally follows the native pipeline's CFG switch.
        # For guidance <=1 native sampling emits the conditional pass directly.
        if guidance > 1:
            uncond_enabled = route.unconditional_lora_strength > 0
            with backend.branch(tau, lora_enabled=uncond_enabled,
                                gate_mode=route.gate_mode if uncond_enabled else "bypassed",
                                strength=route.unconditional_lora_strength,
                                unconditional=True, allow_unconditional_lora=uncond_enabled,
                                name="cfg_unconditional"):
                unconditional = backend.predict(latents, tau, empty)
            velocity = unconditional+guidance*(conditional-unconditional)
        else:
            velocity = conditional
        latents = latents+velocity.float()*(sigma_next-sigma)
        if not bool(torch.isfinite(latents).all()):
            raise FloatingPointError("Nonfinite Gen2 sampling latent")
        if progress_callback is not None:
            progress_callback(completed_steps, total_steps)
    pixels = model.decode_latents(latents, device=model.device_torch, dtype=model.torch_dtype)
    pixels = ((pixels.float().clamp(-1., 1.)+1.)*127.5).round().to(torch.uint8)
    image = Image.fromarray(pixels.permute(0, 2, 3, 1)[0].cpu().numpy())
    metadata = {"prompt": prompt, "compiler": compilation, "conditioning": condition.metadata[0],
        "route": asdict(route), "seed": seed, "rng_backend": "torch.Generator(cpu)",
        "initial_noise_source": "explicit_tensor" if initial_noise is not None else "native_randn_tensor",
        "sampler": "native_ideogram_euler", "sigma_schedule": sigmas.cpu().tolist(),
        "schedule_mu": mu, "schedule_std": std, "steps": steps, "width": width, "height": height,
        "guidance_scale": guidance, "lora_strength": strength,
        "conditional_lora_strength": strength if route.lora_enabled else 0.,
        "conditional_gate_mode": route.gate_mode, "conditional_embedding_enabled": route.styled,
        "conditional_text_adapter_enabled": bool(route.styled and route.adapter_enabled),
        "unconditional_lora_strength": route.unconditional_lora_strength,
        "unconditional_lora_strength_kind": "absolute",
        "unconditional_gate_mode": route.gate_mode if route.unconditional_lora_strength > 0 else "bypassed",
        "unconditional_branch_executed": guidance > 1,
        "unconditional_image_only": True, "unconditional_text_tokens": 0,
        "unconditional_embedding_enabled": False, "unconditional_text_adapter_enabled": False,
        "unconditional_personalization_lora": route.unconditional_lora_strength > 0,
        "unconditional_adapter": model.model_config.unconditional_lora_path}
    return image, metadata
