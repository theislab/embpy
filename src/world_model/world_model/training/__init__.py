"""Training-loop primitives."""

from __future__ import annotations

from .hooks import (
    CSVLossLogger,
    CheckpointHook,
    ConsoleLogger,
    Hook,
    HookState,
    LossPlotter,
    TensorBoardLogger,
)
from .losses import delta_mse, gaussian_nll, info_nce, latent_mse
from .schedulers import build_scheduler
from .trainer import WorldModelTrainer, iter_trainable_params
from .transfer import apply_encoder_swap

__all__ = [
    "CSVLossLogger",
    "CheckpointHook",
    "ConsoleLogger",
    "Hook",
    "HookState",
    "LossPlotter",
    "TensorBoardLogger",
    "WorldModelTrainer",
    "apply_encoder_swap",
    "build_scheduler",
    "delta_mse",
    "gaussian_nll",
    "info_nce",
    "iter_trainable_params",
    "latent_mse",
]
