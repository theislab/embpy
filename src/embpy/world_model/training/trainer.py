"""Hook-based training loop for the world model.

Pure PyTorch; no Lightning or Accelerate dependency. The trainer
delegates everything except the forward/backward/step kernel to a
small set of pluggable :class:`world_model.training.hooks.Hook`
instances. Default hooks: console + CSV + TensorBoard + checkpointing
+ end-of-training loss plot. Drop them by passing ``hooks=[...]`` to
override.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

from ..configs import LossConfig, OptimConfig, TrainConfig
from ..utils.checkpoint import save_checkpoint
from .hooks import (
    CSVLossLogger,
    CheckpointHook,
    ConsoleLogger,
    Hook,
    HookState,
    LossPlotter,
    TensorBoardLogger,
)
from .schedulers import build_scheduler

if TYPE_CHECKING:
    from ..models.world_model import WorldModel

logger = logging.getLogger(__name__)


class WorldModelTrainer:
    """Wraps a :class:`WorldModel` plus optimizer plus scheduler.

    Parameters
    ----------
    model
        The composed world model.
    optim_cfg, loss_cfg, train_cfg
        Hyperparameter configs.
    output_dir, run_name
        Where to write checkpoints / logs.
    hooks
        Optional override for the default hook list. ``None`` builds the
        package default (console + CSV + TB + checkpoint + plotter). Pass
        ``[]`` to disable hooks entirely (silent training).
    """

    def __init__(
        self,
        model: "WorldModel",
        optim_cfg: OptimConfig,
        loss_cfg: LossConfig,
        train_cfg: TrainConfig,
        output_dir: str | Path = "outputs/world_model",
        run_name: str = "wm_run",
        hooks: list[Hook] | None = None,
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

        self.hooks: list[Hook] = hooks if hooks is not None else self._default_hooks()
        self.state = HookState(output_dir=self.output_dir, run_name=run_name)

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
        # Wire any checkpoint hooks now that optimizer / scheduler exist.
        for h in self.hooks:
            if isinstance(h, CheckpointHook):
                h.model = self.model
                h.optimizer = self.optimizer
                h.scheduler = self.scheduler

        self._fire("on_train_start")

        for epoch in range(1, self.train_cfg.n_epochs + 1):
            self.state.epoch = epoch
            self._fire("on_epoch_start")
            t0 = time.time()
            train_loss = self._train_one_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.state.train_loss = train_loss

            if val_loader is not None and (epoch % self.train_cfg.eval_every_n_epochs == 0):
                val_loss = self.evaluate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.state.val_loss = val_loss
            else:
                self.state.val_loss = None

            self.state.extra["epoch_time_s"] = time.time() - t0
            self._fire("on_epoch_end")

        self._save(f"{self.run_name}_final.pt", epoch=self.train_cfg.n_epochs)
        self._fire("on_train_end")
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

    def load_state(self, ckpt_path: str | Path, *, strict: bool = False) -> None:
        """Load model weights (and optionally optimizer/scheduler) from a checkpoint.

        Used by the transfer setup to seed fine-tuning with the
        pretrained weights.
        """
        from ..utils.checkpoint import load_checkpoint  # noqa: PLC0415

        payload = load_checkpoint(ckpt_path, map_location=self.device)
        missing = self.model.load_state_dict(payload["state_dict"], strict=strict)
        logger.info("Loaded weights from %s (missing/unexpected: %s)", ckpt_path, missing)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        running = 0.0
        n = 0
        for batch in loader:
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
                grad_norm = None
                if self.optim_cfg.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.optim_cfg.grad_clip,
                    ))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = None
                if self.optim_cfg.grad_clip is not None:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.optim_cfg.grad_clip,
                    ))
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1
            self.state.global_step = self.global_step
            self.state.lr = float(self.optimizer.param_groups[0]["lr"])
            self.state.grad_norm = grad_norm
            self.state.components = {k: float(v) for k, v in components.items()}

            self._fire("on_step_end")

            bsz = batch["obs_stack"].size(0)
            running += float(loss.item()) * bsz
            n += bsz
        return running / max(n, 1)

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        moved: dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device, non_blocking=True)
            else:
                moved[k] = v
        return moved

    def _fire(self, event: str) -> None:
        for h in self.hooks:
            getattr(h, event)(self.state)

    def _save(self, filename: str, epoch: int) -> None:
        path = self.output_dir / filename
        save_checkpoint(
            path,
            state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict() if self.scheduler is not None else None,
            metadata={"epoch": epoch, "global_step": self.global_step, "run_name": self.run_name},
        )

    def _default_hooks(self) -> list[Hook]:
        out: list[Hook] = [ConsoleLogger(log_every_n_steps=self.train_cfg.log_every_n_steps)]
        if self.train_cfg.enable_csv_log:
            out.append(CSVLossLogger())
        if self.train_cfg.enable_tensorboard:
            out.append(TensorBoardLogger())
        out.append(CheckpointHook(every_n_epochs=self.train_cfg.save_every_n_epochs))
        out.append(LossPlotter())
        return out

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
