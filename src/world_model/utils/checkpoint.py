"""Checkpoint save / load helpers.

A checkpoint is a single ``torch.save`` archive containing:

* ``state_dict``           -- model parameters
* ``optimizer_state_dict`` -- optimizer parameters (optional)
* ``scheduler_state_dict`` -- LR scheduler parameters (optional)
* ``config``               -- a flat dict capturing the constructor args
* ``metadata``             -- arbitrary user-supplied info (epoch, step, ...)

Keeping this in a single file avoids the directory-per-checkpoint
proliferation common in ML projects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    *,
    state_dict: dict[str, Any],
    config: dict[str, Any] | None = None,
    optimizer_state_dict: dict[str, Any] | None = None,
    scheduler_state_dict: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist a model + optimizer + metadata bundle to ``path``."""
    import torch  # noqa: PLC0415

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "state_dict": state_dict,
        "config": config or {},
        "metadata": metadata or {},
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    if scheduler_state_dict is not None:
        payload["scheduler_state_dict"] = scheduler_state_dict
    torch.save(payload, path)
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(
    path: str | Path,
    map_location: str | Any = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint dict written by :func:`save_checkpoint`."""
    import torch  # noqa: PLC0415

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    logger.info("Loaded checkpoint from %s", path)
    return payload


__all__ = ["load_checkpoint", "save_checkpoint"]
