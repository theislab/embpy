"""Minimal config-driven trainer for the world model.

Pure PyTorch; no Lightning or Accelerate dependency. Designed for two
audiences:

1. Single-GPU (or CPU) experiments started directly from the
   :mod:`world_model.scripts.train` entry-point.
2. Programmatic use from notebooks where the user wants to call
   ``trainer.fit(loader)`` and inspect ``trainer.history``.

Distributed training is out of scope -- if you need it, wrap the
underlying :class:`torch.nn.Module` in DDP yourself; the loss helpers
and the model class are DDP-friendly.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

from ..configs import LossConfig, OptimConfig, TrainConfig
from ..utils.checkpoint import save_checkpoint
from .schedulers import build_scheduler

if TYPE_CHECKING:
    # WorldModel is only used as a type annotation. Importing it at
    # runtime would create a cycle: world_model -> training.losses
    # -> training -> trainer -> world_model.
    from ..models.world_model import WorldModel

logger = logging.getLogger(__name__)


class WorldModelTrainer:
    """Wraps a :class:`WorldModel` plus optimizer plus scheduler.

    Parameters
    ----------
    model
        The composed world model.
    optim_cfg
        Optimizer configuration.
    loss_cfg
        Multi-term loss configuration.
    train_cfg
        Training-loop knobs (epochs, AMP, device, ...).
    output_dir
        Where to write checkpoints. Created if it does not exist.
    run_name
        Used as the checkpoint filename prefix.
    """

    def __init__(
        self,
        model: "WorldModel",
        optim_cfg: OptimConfig,
        loss_cfg: LossConfig,
        train_cfg: TrainConfig,
        output_dir: str | Path = "outputs/world_model",
        run_name: str = "wm_run",
    ) -> None:
        self.model = model
        self.optim_cfg = optim_cfg
        self.loss_cfg = loss_cfg
        self.train_cfg = train_cfg
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device(train_cfg.device)
        self.model.to(self.device)

        decay, no_decay = self._split_params_for_weight_decay(self.model)
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": optim_cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=optim_cfg.lr,
            betas=optim_cfg.betas,
        )
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.use_amp = train_cfg.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.global_step = 0

    # ------------------------------------------------------------------
    # Public training API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, list[float]]:
        """Run the full training loop, returning the loss history."""
        total_steps = max(1, len(train_loader)) * self.train_cfg.n_epochs
        self.scheduler = build_scheduler(
            self.optimizer,
            kind=self.optim_cfg.scheduler,
            total_steps=total_steps,
            warmup_steps=self.optim_cfg.warmup_steps,
        )

        for epoch in range(1, self.train_cfg.n_epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(train_loader, epoch)
            self.history["train_loss"].append(train_loss)

            if val_loader is not None and (epoch % self.train_cfg.eval_every_n_epochs == 0):
                val_loss = self.evaluate(val_loader)
                self.history["val_loss"].append(val_loss)
                logger.info(
                    "[epoch %d/%d] train=%.4f val=%.4f (%.1fs)",
                    epoch, self.train_cfg.n_epochs, train_loss, val_loss, time.time() - t0,
                )
            else:
                logger.info(
                    "[epoch %d/%d] train=%.4f (%.1fs)",
                    epoch, self.train_cfg.n_epochs, train_loss, time.time() - t0,
                )

            if epoch % self.train_cfg.save_every_n_epochs == 0:
                self._save("ckpt_epoch%03d.pt" % epoch, epoch=epoch)

        self._save(f"{self.run_name}_final.pt", epoch=self.train_cfg.n_epochs)
        return self.history

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        n = 0
        for batch in loader:
            batch = self._move_batch(batch)
            loss, _ = self.model.loss(
                batch,
                latent_mse_weight=self.loss_cfg.latent_mse,
                decoder_mse_weight=self.loss_cfg.decoder_mse,
                info_nce_weight=self.loss_cfg.info_nce,
                info_nce_temperature=self.loss_cfg.info_nce_temperature,
            )
            bsz = batch["obs_stack"].size(0)
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
            batch = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss, components = self.model.loss(
                    batch,
                    latent_mse_weight=self.loss_cfg.latent_mse,
                    decoder_mse_weight=self.loss_cfg.decoder_mse,
                    info_nce_weight=self.loss_cfg.info_nce,
                    info_nce_temperature=self.loss_cfg.info_nce_temperature,
                )

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.optim_cfg.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.optim_cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.optim_cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.optim_cfg.grad_clip)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1
            bsz = batch["obs_stack"].size(0)
            running += float(loss.item()) * bsz
            n += bsz

            if step % self.train_cfg.log_every_n_steps == 0:
                comp_str = " ".join(f"{k}={float(v):.4f}" for k, v in components.items())
                logger.info("epoch %d step %d %s", epoch, step, comp_str)
        return running / max(n, 1)

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        moved: dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device, non_blocking=True)
            else:
                moved[k] = v
        return moved

    def _save(self, filename: str, epoch: int) -> None:
        path = self.output_dir / filename
        save_checkpoint(
            path,
            state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict() if self.scheduler is not None else None,
            metadata={"epoch": epoch, "global_step": self.global_step, "run_name": self.run_name},
        )

    @staticmethod
    def _resolve_device(spec: str) -> torch.device:
        if spec == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(spec)

    @staticmethod
    def _split_params_for_weight_decay(
        model: torch.nn.Module,
    ) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        # Standard AdamW recipe: do not decay biases or LayerNorm weights.
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith(".bias") or "norm" in name.lower() or "embed" in name.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        return decay, no_decay


__all__ = ["WorldModelTrainer"]
