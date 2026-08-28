from __future__ import annotations

import torch


def temporal_statistics(fields) -> dict[str, float]:
    values = torch.cat([field(torch.linspace(0.0, 1.0, 16, device=field.values.device)).flatten() for field in fields])
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def branch_alignment(delta_c: torch.Tensor, delta_u: torch.Tensor, eps: float = 1e-8) -> dict[str, float]:
    c = delta_c.float().flatten()
    u = delta_u.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(c[None], u[None], dim=-1, eps=eps)[0]
    rho = u.norm() / c.norm().clamp_min(eps)
    return {"cosine": float(cosine.item()), "rho": float(rho.item())}
