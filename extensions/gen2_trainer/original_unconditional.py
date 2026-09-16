"""Load the original image-only CFG model through the native Ideogram loader."""
from __future__ import annotations

from pathlib import Path

from .diagnostics import isolated_rng


def load_original_unconditional(model, gen2_config):
    """Attach a separate frozen original transformer, preserving training RNG.

    ``unconditional_model_path`` is a Hub repository or a local repository root
    containing the ``unconditional_transformer`` component. Native strict weight
    loading, FP8 scale reconstruction, quantization, and offloading all apply.
    An absent path leaves the native model and all RNG streams untouched.
    """
    source = gen2_config["inference"].get("unconditional_model_path")
    if source is None:
        return None
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("unconditional_model_path must identify a model repository or local root")
    native_config = getattr(model, "model_config", None)
    if (getattr(model, "unconditional_lora", None) is not None or
            getattr(native_config, "unconditional_lora_path", None) is not None):
        raise ValueError("Original unconditional model conflicts with the unconditional correction LoRA")

    conditional = model.transformer
    conditional_parameters = {id(parameter) for parameter in conditional.parameters()}
    print(f"[Gen2] Loading original unconditional transformer from {source}", flush=True)
    with isolated_rng(gen2_config["execution"]["training_seed"]):
        transformer = model._load_transformer(str(source), component="unconditional_transformer")
        if transformer is conditional or any(
                id(parameter) in conditional_parameters for parameter in transformer.parameters()):
            raise ValueError("Original unconditional transformer must be separate from the conditional model")
        transformer.aitk_post_load(**model.component_load_kwargs("transformer"))
        transformer.eval()
        transformer.requires_grad_(False)
        transformer.disable_gradient_checkpointing()
        transformer.set_attention_backend(gen2_config["execution"]["dit_attention_backend"])
        meta_names = [name for name, tensor in
                      list(transformer.named_parameters()) + list(transformer.named_buffers())
                      if tensor.is_meta]
        if meta_names:
            raise ValueError(f"Original unconditional model has unloaded meta tensors: {meta_names[:8]}")

    model.unconditional_transformer = transformer
    print("[Gen2] Original unconditional transformer loaded and frozen (inference only)", flush=True)
    return transformer
