"""The v1 cubic Bernstein gates, in toolkit noise-fraction coordinates."""
from __future__ import annotations

import torch
from torch import nn


def bernstein_basis(tau: torch.Tensor) -> torch.Tensor:
    t = tau.float()
    return torch.stack(((1-t)**3, 3*t*(1-t)**2, 3*t*t*(1-t), t**3), dim=-1)


class CubicTimeGates(nn.Module):
    """One four-coefficient function per actual DiT block; fp32 masters."""
    def __init__(self, num_blocks: int, amplitude: float = .5, grid_points: int = 65):
        super().__init__()
        if num_blocks < 1 or not 0 < amplitude < 1 or grid_points < 5:
            raise ValueError("Gates require blocks >=1, 0 < amplitude < 1, grid_points >=5")
        self.beta = nn.Parameter(torch.zeros(num_blocks, 4, dtype=torch.float32))
        self.register_buffer("rho", torch.tensor(amplitude, dtype=torch.float32))
        self.register_buffer("grid", torch.linspace(0, 1, grid_points, dtype=torch.float32))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """Return (..., blocks); never average different examples' times."""
        p = bernstein_basis(tau.to(self.beta.device)) @ self.beta.float().T
        return 1 + self.rho.float() * p.tanh()

    def derivative(self, tau: torch.Tensor) -> torch.Tensor:
        t = tau.to(self.beta.device, torch.float32).unsqueeze(-1)
        b = self.beta.float()
        dp = 3*((b[:, 1]-b[:, 0])*(1-t)**2
                + 2*(b[:, 2]-b[:, 1])*t*(1-t)
                + (b[:, 3]-b[:, 2])*t*t)
        p = bernstein_basis(tau.to(self.beta.device)) @ b.T
        return self.rho.float() * (1-p.tanh().square()) * dp

    def regularizers(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self(self.grid)-1).square().mean(), self.derivative(self.grid).square().mean()

    def time_mean(self) -> torch.Tensor:
        return self(self.grid).mean(dim=0)

    def values(self, tau: torch.Tensor, mode: str = "learned") -> torch.Tensor:
        if mode == "learned":
            return self(tau)
        if mode in ("one", "bypassed"):
            return torch.ones((*tau.shape, self.beta.shape[0]), device=tau.device, dtype=torch.float32)
        if mode == "time_mean":
            return self.time_mean().expand(*tau.shape, -1)
        raise ValueError(f"Unknown gate mode: {mode}")
