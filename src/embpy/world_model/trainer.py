"""Minimal training loop for ``embpy.world_model`` models.

Deliberately framework-agnostic on top of plain PyTorch -- no Lightning
or Accelerate dependency. The trainer is meant as a sensible default for
small/medium experiments; for distributed or large-scale training, drop
into PyTorch Lightning or your existing infrastructure and reuse the
:class:`~embpy.world_model.base.BaseWorldModel` / loss building blocks.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .base import BaseWorldModel
from .losses import latent_mse

logger = logging.getLogger(__name__)


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class TrainConfig:
    """Hyper-parameters for :class:`WorldModelTrainer`."""

    n_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0
    log_every_n_steps: int = 50
    device: str = "auto"
    seed: int | None = 0
    extra: dict[str, Any] = field(default_factory=dict)


class WorldModelTrainer:
    """Training loop for a :class:`BaseWorldModel`.

    The trainer expects each batch to be a dict containing at least
    ``z_basal``, ``z_perturbed`` and (optionally) ``cond`` -- the format
    produced by :class:`embpy.world_model.dataset.PerturbationLatentDataset`.

    Parameters
    ----------
    model
        The world model to train.
    loss_fn
        Callable ``loss_fn(pred, target) -> scalar tensor``. Defaults to
        :func:`embpy.world_model.losses.latent_mse`.
    config
        Hyper-parameters; see :class:`TrainConfig`.
    optimizer_factory
        Callable returning an :class:`torch.optim.Optimizer` from a
        parameter iterable. Defaults to AdamW with the config's
        ``lr`` / ``weight_decay``.
    """

    def __init__(
        self,
        model: BaseWorldModel,
        loss_fn: LossFn = latent_mse,
        config: TrainConfig | None = None,
        optimizer_factory: Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer] | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.config = config or TrainConfig()
        self.device = self._resolve_device(self.config.device)

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        self.model.to(self.device)

        if optimizer_factory is None:
            self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = optimizer_factory(self.model.parameters())

        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, list[float]]:
        """Run the full training loop, returning the loss history."""
        for epoch in range(1, self.config.n_epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(train_loader, epoch)
            self.history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.history["val_loss"].append(val_loss)
                logger.info(
                    "[epoch %d/%d] train=%.4f val=%.4f (%.1fs)",
                    epoch, self.config.n_epochs, train_loss, val_loss, time.time() - t0,
                )
            else:
                logger.info(
                    "[epoch %d/%d] train=%.4f (%.1fs)",
                    epoch, self.config.n_epochs, train_loss, time.time() - t0,
                )
        return self.history

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        """Compute the average loss over a data loader."""
        self.model.eval()
        total = 0.0
        n = 0
        for batch in loader:
            pred, target = self._step(batch)
            loss = self.loss_fn(pred, target)
            bsz = target.size(0)
            total += float(loss.item()) * bsz
            n += bsz
        return total / max(n, 1)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        running = 0.0
        n = 0
        for step, batch in enumerate(loader, start=1):
            pred, target = self._step(batch)
            loss = self.loss_fn(pred, target)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            bsz = target.size(0)
            running += float(loss.item()) * bsz
            n += bsz

            if step % self.config.log_every_n_steps == 0:
                logger.debug(
                    "epoch %d step %d loss=%.4f", epoch, step, running / max(n, 1)
                )
        return running / max(n, 1)

    def _step(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        z_basal = batch["z_basal"].to(self.device, non_blocking=True)
        z_perturbed = batch["z_perturbed"].to(self.device, non_blocking=True)
        cond = batch.get("cond")
        if cond is not None and isinstance(cond, torch.Tensor):
            cond = cond.to(self.device, non_blocking=True)
        else:
            cond = None
        pred = self.model(z_basal, cond)
        return pred, z_perturbed

    @staticmethod
    def _resolve_device(spec: str) -> torch.device:
        if spec == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(spec)


__all__ = ["TrainConfig", "WorldModelTrainer"]
