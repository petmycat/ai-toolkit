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


class OfficialIdeogramUnconditionalLoader:
    def __init__(self, conditional_source: str, dtype: torch.dtype, device: torch.device, quantize: bool = True, qtype: str = "qfloat8") -> None:
        self.conditional_source = conditional_source
        self.dtype = dtype
        self.device = device
        self.quantize = quantize
        self.qtype = qtype
        self.loaded: OfficialUnconditionalTransformer | None = None

    def _component_path(self) -> Path:
        root = Path(self.conditional_source)
        if not root.exists():
            cache_roots = [
                Path(os.environ[key])
                for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE")
                if os.environ.get(key)
            ]
            cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")
            repo_key = "models--" + self.conditional_source.replace("/", "--")
            snapshots = []
            for cache_root in cache_roots:
                repo_root = cache_root / repo_key
                snapshots.extend((repo_root / "snapshots").glob("*") if (repo_root / "snapshots").is_dir() else [])
            if not snapshots:
                raise FileNotFoundError(f"official unconditional model is not present in local HF cache: {self.conditional_source}")
            root = max(snapshots, key=lambda item: item.stat().st_mtime)
        component = root / "unconditional_transformer"
        if not component.is_dir():
            raise FileNotFoundError(f"official component not found: {component}")
        return component

    def load(self) -> OfficialUnconditionalTransformer:
        component = self._component_path()
        config_path = component / "config.json"
        if not config_path.is_file():
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
            str(component.parent), component.name, "diffusion_pytorch_model"
        )
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
        self.loaded = OfficialUnconditionalTransformer(transformer=transformer, source=str(component))
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
