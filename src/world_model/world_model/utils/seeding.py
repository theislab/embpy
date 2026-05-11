"""Single seed entry-point for python, numpy and torch RNGs."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and (if available) torch RNGs.

    The torch import is lazy so this module remains importable in
    minimal environments (CI without GPU, doc builds, ...).

    Parameters
    ----------
    seed
        Non-negative integer seed.
    deterministic
        If True, also configure cuDNN for deterministic kernels. This
        slows training and is only worth it for debugging numerical
        regressions.
    """
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # These flags are intentionally noisy: matches PyTorch upstream guidance.
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def torch_generator(seed: int) -> Any:
    """Return a fresh ``torch.Generator`` seeded with ``seed`` (lazy import)."""
    import torch  # noqa: PLC0415

    g = torch.Generator()
    g.manual_seed(seed)
    return g


__all__ = ["seed_everything", "torch_generator"]
