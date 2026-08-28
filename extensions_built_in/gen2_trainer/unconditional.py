from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from extensions_built_in.diffusion_models.ideogram4.src.transformer import Ideogram4Transformer2DModel


@dataclass
class OfficialUnconditionalTransformer:
    transformer: "Ideogram4Transformer2DModel"
    source: str
    component: str = "unconditional_transformer"
    backend: str = "official image-only unconditional transformer"


def _convert_split_attention_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = dict(state_dict)
    prefixes = set()
    for key in state_dict:
        for suffix in (".to_q.weight", ".to_k.weight", ".to_v.weight"):
            if key.endswith(suffix):
                prefixes.add(key[: -len(suffix)])
    for prefix in prefixes:
        q_key = f"{prefix}.to_q.weight"
        k_key = f"{prefix}.to_k.weight"
        v_key = f"{prefix}.to_v.weight"
        if not all(key in state_dict for key in (q_key, k_key, v_key)):
            raise RuntimeError(f"incomplete split QKV weights under {prefix}")
        converted[f"{prefix}.qkv.weight"] = torch.cat(
            [state_dict[q_key], state_dict[k_key], state_dict[v_key]], dim=0
        )
        for key in (q_key, k_key, v_key):
            converted.pop(key, None)
        q_scale = f"{q_key}_scale"
        k_scale = f"{k_key}_scale"
        v_scale = f"{v_key}_scale"
        if any(key in state_dict for key in (q_scale, k_scale, v_scale)):
            if not all(key in state_dict for key in (q_scale, k_scale, v_scale)):
                raise RuntimeError(f"incomplete split QKV scales under {prefix}")
            converted[f"{prefix}.qkv.weight_scale"] = torch.cat(
                [state_dict[q_scale], state_dict[k_scale], state_dict[v_scale]], dim=0
            )
            for key in (q_scale, k_scale, v_scale):
                converted.pop(key, None)

    for key in list(state_dict):
        if key.endswith(".to_out.0.weight"):
            converted[key[: -len(".to_out.0.weight")] + ".o.weight"] = state_dict[key]
            converted.pop(key, None)
            scale_key = f"{key}_scale"
            if scale_key in state_dict:
                converted[key[: -len(".to_out.0.weight")] + ".o.weight_scale"] = state_dict[scale_key]
                converted.pop(scale_key, None)
    return converted


class OfficialIdeogramUnconditionalLoader:
    def __init__(self, conditional_source: str, dtype: torch.dtype, device: torch.device, quantize: bool = True, qtype: str = "qfloat8") -> None:
        self.conditional_source = conditional_source
        self.dtype = dtype
        self.device = device
        self.quantize = quantize
        self.qtype = qtype
        self.loaded: OfficialUnconditionalTransformer | None = None

    def _component_source(self) -> tuple[str, str]:
        root = Path(self.conditional_source)
        if root.exists():
            component = root / "unconditional_transformer"
            if not component.is_dir():
                raise FileNotFoundError(f"official component not found in local model directory: {component}")
            return str(root), component.name

        from huggingface_hub import hf_hub_download

        try:
            hf_hub_download(
                repo_id=self.conditional_source,
                filename="unconditional_transformer/config.json",
                token=os.getenv("HF_TOKEN"),
            )
        except Exception as error:
            raise FileNotFoundError(
                f"official unconditional component is unavailable from {self.conditional_source}: {error}"
            ) from error
        return self.conditional_source, "unconditional_transformer"

    def load(self) -> OfficialUnconditionalTransformer:
        component_source, component_name = self._component_source()
        config_path = Path(component_source) / component_name / "config.json" if Path(component_source).exists() else None
        if config_path is not None and not config_path.is_file():
            raise FileNotFoundError(f"official unconditional config missing: {config_path}")
        from extensions_built_in.diffusion_models.ideogram4.ideogram4 import (
            _dequantize_fp8_state_dict,
            _load_component_state_dict,
        )
        from extensions_built_in.diffusion_models.ideogram4.src.transformer import (
            Ideogram4Config,
            Ideogram4Transformer2DModel,
        )

        config = Ideogram4Config()
        with torch.device("meta"):
            transformer = Ideogram4Transformer2DModel(config)
        state_dict = _load_component_state_dict(
            component_source, component_name, "diffusion_pytorch_model"
        )
        state_dict = _convert_split_attention_state_dict(state_dict)
        state_dict = _dequantize_fp8_state_dict(
            state_dict, self.dtype, self.device, low_vram=True
        )
        transformer.load_state_dict(state_dict, assign=True)
        head_dim = config.emb_dim // config.num_heads
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        transformer.rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
        target_device = torch.device("cpu") if self.quantize else self.device
        transformer.to(target_device)
        transformer.eval()
        transformer.requires_grad_(False)
        source_label = str(Path(component_source) / component_name) if Path(component_source).exists() else f"{component_source}/{component_name}"
        self.loaded = OfficialUnconditionalTransformer(transformer=transformer, source=source_label)
        return self.loaded

    def assert_revision_matches(self, revision: str | None) -> None:
        if revision is None:
            return
        revision_file = Path(self.conditional_source) / "_gen2_revision.txt"
        if not revision_file.is_file():
            raise RuntimeError("cannot prove official unconditional revision matches conditional revision")
        actual = revision_file.read_text(encoding="utf-8").strip()
        if actual != revision:
            raise RuntimeError(f"official unconditional revision mismatch: expected {revision}, got {actual}")

    def offload(self) -> None:
        if self.loaded is not None:
            self.loaded.transformer.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def assert_official_backend(model: OfficialUnconditionalTransformer) -> None:
    if model.component != "unconditional_transformer" or model.backend != "official image-only unconditional transformer":
        raise RuntimeError("Gen2 requires the official image-only unconditional transformer backend")


def official_asymmetric_cfg(v_c: torch.Tensor, v_u: torch.Tensor, guidance_scale: float, eta_c: float = 1.0, eta_u: float = 0.0, delta_c: torch.Tensor | None = None, delta_u: torch.Tensor | None = None) -> torch.Tensor:
    if eta_c < 0 or eta_u < 0:
        raise ValueError("eta_c and eta_u must be non-negative")
    c = v_c if delta_c is None else v_c + eta_c * delta_c
    u = v_u if delta_u is None else v_u + eta_u * delta_u
    return u + guidance_scale * (c - u)
