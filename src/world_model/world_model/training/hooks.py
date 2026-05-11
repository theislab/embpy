"""Trainer hook system.

Each hook is a small object that implements one or more of the
following methods (all optional):

* ``on_train_start(state)``       once at fit-start
* ``on_epoch_start(state)``       once at the start of every epoch
* ``on_step_end(state)``          after every optimizer step
* ``on_epoch_end(state)``         once at the end of every epoch (after eval)
* ``on_train_end(state)``         once at fit-end

``state`` is a small mutable namespace passed by the trainer; see
:class:`HookState` for the standard fields.

Default hooks shipped with the package:

* :class:`CSVLossLogger`     -- writes per-epoch metrics to a CSV file.
* :class:`TensorBoardLogger` -- mirrors metrics to TensorBoard.
* :class:`LossPlotter`       -- writes a PNG of the loss curves on
  ``on_train_end``.
* :class:`ConsoleLogger`     -- prints per-step diagnostics every
  ``log_every_n_steps`` (lr, grad norm, loss components).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter

logger = logging.getLogger(__name__)


@dataclass
class HookState:
    """Mutable training-state namespace shared with hooks."""

    epoch: int = 0
    global_step: int = 0
    lr: float = 0.0
    grad_norm: float | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    components: dict[str, float] = field(default_factory=dict)
    output_dir: Path = Path(".")
    run_name: str = "wm_run"
    extra: dict[str, Any] = field(default_factory=dict)


class Hook:
    """Base hook class -- override only the methods you care about."""

    def on_train_start(self, state: HookState) -> None:  # pragma: no cover - default no-op
        pass

    def on_epoch_start(self, state: HookState) -> None:  # pragma: no cover
        pass

    def on_step_end(self, state: HookState) -> None:  # pragma: no cover
        pass

    def on_epoch_end(self, state: HookState) -> None:  # pragma: no cover
        pass

    def on_train_end(self, state: HookState) -> None:  # pragma: no cover
        pass


# ----------------------------------------------------------------------
# Concrete hooks
# ----------------------------------------------------------------------


class ConsoleLogger(Hook):
    """Print per-step diagnostics every ``log_every_n_steps`` steps."""

    def __init__(self, log_every_n_steps: int = 50) -> None:
        self.log_every_n_steps = max(1, int(log_every_n_steps))

    def on_step_end(self, state: HookState) -> None:
        if state.global_step % self.log_every_n_steps != 0:
            return
        comp = " ".join(f"{k}={v:.4f}" for k, v in state.components.items())
        gn = "n/a" if state.grad_norm is None else f"{state.grad_norm:.3f}"
        logger.info(
            "epoch %d step %d lr=%.2e grad_norm=%s %s",
            state.epoch, state.global_step, state.lr, gn, comp,
        )

    def on_epoch_end(self, state: HookState) -> None:
        if state.val_loss is None:
            logger.info(
                "[epoch %d] train=%.4f", state.epoch, state.train_loss or float("nan"),
            )
        else:
            logger.info(
                "[epoch %d] train=%.4f val=%.4f",
                state.epoch, state.train_loss or float("nan"), state.val_loss,
            )


class CSVLossLogger(Hook):
    """Append a row per epoch to ``{output_dir}/train_log.csv``."""

    def __init__(self, filename: str = "train_log.csv") -> None:
        self.filename = filename
        self._fp = None
        self._writer = None
        self._header_written = False

    def on_train_start(self, state: HookState) -> None:
        path = state.output_dir / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(path, "w", newline="")
        self._writer = csv.writer(self._fp)

    def on_epoch_end(self, state: HookState) -> None:
        if self._writer is None:
            return
        row: dict[str, Any] = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "train_loss": state.train_loss,
            "val_loss": state.val_loss,
            "lr": state.lr,
            "grad_norm": state.grad_norm,
        }
        row.update({f"component_{k}": v for k, v in state.components.items()})
        if not self._header_written:
            self._writer.writerow(list(row.keys()))
            self._header_written = True
        self._writer.writerow(list(row.values()))
        self._fp.flush()  # type: ignore[union-attr]

    def on_train_end(self, state: HookState) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None


class TensorBoardLogger(Hook):
    """Mirror per-step and per-epoch metrics to TensorBoard.

    SummaryWriter is imported lazily so the module remains importable
    without TensorBoard installed.
    """

    def __init__(self, subdir: str = "tb") -> None:
        self.subdir = subdir
        self.writer: SummaryWriter | None = None

    def on_train_start(self, state: HookState) -> None:
        try:
            from torch.utils.tensorboard.writer import SummaryWriter  # noqa: PLC0415
        except ImportError:
            logger.warning("TensorBoard not available; skipping TensorBoardLogger.")
            return
        path = state.output_dir / self.subdir
        path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(path))

    def on_step_end(self, state: HookState) -> None:
        if self.writer is None:
            return
        for k, v in state.components.items():
            self.writer.add_scalar(f"step/{k}", v, state.global_step)
        self.writer.add_scalar("step/lr", state.lr, state.global_step)
        if state.grad_norm is not None:
            self.writer.add_scalar("step/grad_norm", state.grad_norm, state.global_step)

    def on_epoch_end(self, state: HookState) -> None:
        if self.writer is None:
            return
        if state.train_loss is not None:
            self.writer.add_scalar("epoch/train_loss", state.train_loss, state.epoch)
        if state.val_loss is not None:
            self.writer.add_scalar("epoch/val_loss", state.val_loss, state.epoch)

    def on_train_end(self, state: HookState) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class LossPlotter(Hook):
    """Plot loss curves at end of training.

    Reads back the CSV produced by :class:`CSVLossLogger` so the plot
    contains exactly the persisted values (no in-memory drift).
    """

    def __init__(self, filename: str = "loss_curves.png", csv_filename: str = "train_log.csv") -> None:
        self.filename = filename
        self.csv_filename = csv_filename

    def on_train_end(self, state: HookState) -> None:
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
            import pandas as pd  # noqa: PLC0415
        except ImportError:
            logger.warning("matplotlib/pandas missing; skipping LossPlotter.")
            return
        csv_path = state.output_dir / self.csv_filename
        if not csv_path.exists():
            logger.warning("No CSV log at %s; skipping LossPlotter.", csv_path)
            return
        df = pd.read_csv(csv_path)
        if df.empty:
            return
        out_dir = state.output_dir / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        if "train_loss" in df:
            ax.plot(df["epoch"], df["train_loss"], label="train", marker="o")
        if "val_loss" in df:
            ax.plot(df["epoch"], df["val_loss"], label="val", marker="s")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(f"{state.run_name} -- loss curves")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        png_path = out_dir / self.filename
        svg_path = png_path.with_suffix(".svg")
        fig.savefig(png_path, dpi=150)
        fig.savefig(svg_path)
        plt.close(fig)
        logger.info("Saved loss curves to %s and %s", png_path, svg_path)


class CheckpointHook(Hook):
    """Save a checkpoint every ``every_n_epochs`` epochs.

    The actual save mechanics live in
    :func:`world_model.utils.checkpoint.save_checkpoint`; this hook
    just orchestrates when to call it.
    """

    def __init__(self, every_n_epochs: int = 5, model: torch.nn.Module | None = None,
                 optimizer: torch.optim.Optimizer | None = None,
                 scheduler: Any | None = None) -> None:
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def on_epoch_end(self, state: HookState) -> None:
        if state.epoch % self.every_n_epochs != 0:
            return
        from ..utils.checkpoint import save_checkpoint  # noqa: PLC0415

        path = state.output_dir / f"ckpt_epoch{state.epoch:03d}.pt"
        save_checkpoint(
            path,
            state_dict=self.model.state_dict() if self.model is not None else {},
            optimizer_state_dict=(self.optimizer.state_dict() if self.optimizer is not None else None),
            scheduler_state_dict=(self.scheduler.state_dict() if self.scheduler is not None else None),
            metadata={
                "epoch": state.epoch,
                "global_step": state.global_step,
                "run_name": state.run_name,
            },
        )


__all__ = [
    "CSVLossLogger",
    "CheckpointHook",
    "ConsoleLogger",
    "Hook",
    "HookState",
    "LossPlotter",
    "TensorBoardLogger",
]
