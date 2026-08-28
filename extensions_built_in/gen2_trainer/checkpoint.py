from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_phase_checkpoint(path: str | Path, tensors: dict[str, torch.Tensor], metadata: dict[str, Any], optimizer: torch.optim.Optimizer | None = None, scheduler: Any = None) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, root / "tensors.pt")
    state = {"metadata": metadata, "rng": capture_rng_state()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, root / "training_state.pt")
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def load_phase_checkpoint(path: str | Path, optimizer: torch.optim.Optimizer | None = None, scheduler: Any = None) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    root = Path(path)
    tensors = torch.load(root / "tensors.pt", map_location="cpu", weights_only=True)
    state = torch.load(root / "training_state.pt", map_location="cpu", weights_only=False)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(state["rng"])
    return tensors, state["metadata"]
