"""Checkpoint helpers for saving/loading training state."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: Path | str,
    *,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    step: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist a training checkpoint to ``path``.

    Parameters
    ----------
    path:
        Destination file. Parent directories are created automatically.
    model_state:
        ``state_dict`` for the model.
    optimizer_state:
        Optional optimiser ``state_dict``.
    step:
        Training step to record as metadata.
    extra:
        Additional metadata (e.g., epoch counters).
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {"model": model_state}
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state
    if step is not None:
        payload["step"] = step
    if extra:
        payload.update(extra)

    torch.save(payload, target)
    return target


def load_checkpoint(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Load a checkpoint and optionally restore model/optimizer state."""

    checkpoint = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


__all__ = ["save_checkpoint", "load_checkpoint"]
