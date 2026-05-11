"""Learning-rate schedulers used by the world model trainer.

Implemented via :class:`torch.optim.lr_scheduler.LambdaLR` so the same
"step the scheduler every optimizer step" rule applies regardless of
the underlying schedule shape.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


def _cosine_with_warmup(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cos


def _linear_warmup(step: int, warmup_steps: int) -> float:
    if step >= warmup_steps:
        return 1.0
    return float(step) / max(1, warmup_steps)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    kind: str,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build a step-level LR scheduler.

    Parameters
    ----------
    optimizer
        PyTorch optimizer to schedule.
    kind
        ``"cosine"``, ``"linear_warmup"`` or ``"constant"``.
    total_steps
        Number of optimizer steps in the run; used by the cosine schedule.
    warmup_steps
        Number of warmup steps. Ignored for ``"constant"``.
    min_lr_ratio
        Floor of the cosine schedule, expressed as a fraction of the base LR.
    """
    if kind == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)

    if kind == "linear_warmup":
        fn: Callable[[int], float] = lambda step: _linear_warmup(step, warmup_steps)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)

    if kind == "cosine":
        if total_steps <= 0:
            raise ValueError("total_steps must be positive for cosine schedule")
        fn = lambda step: _cosine_with_warmup(step, total_steps, warmup_steps, min_lr_ratio)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)

    raise ValueError(f"Unknown scheduler kind {kind!r}. Use 'cosine', 'linear_warmup' or 'constant'.")


__all__ = ["build_scheduler"]
