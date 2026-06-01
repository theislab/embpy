"""Tests for the trainer hook system."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from world_model.training.hooks import (
    CSVLossLogger,
    ConsoleLogger,
    Hook,
    HookState,
    LossPlotter,
)


class _Counter(Hook):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_train_start(self, state: HookState) -> None:
        self.events.append("train_start")

    def on_epoch_start(self, state: HookState) -> None:
        self.events.append(f"epoch_start:{state.epoch}")

    def on_step_end(self, state: HookState) -> None:
        self.events.append(f"step:{state.global_step}")

    def on_epoch_end(self, state: HookState) -> None:
        self.events.append(f"epoch_end:{state.epoch}")

    def on_train_end(self, state: HookState) -> None:
        self.events.append("train_end")


def test_hook_lifecycle_fires_in_order(tmp_path: Path):
    counter = _Counter()
    state = HookState(output_dir=tmp_path, run_name="t")
    counter.on_train_start(state)
    for epoch in (1, 2):
        state.epoch = epoch
        counter.on_epoch_start(state)
        for step in range(2):
            state.global_step += 1
            counter.on_step_end(state)
        counter.on_epoch_end(state)
    counter.on_train_end(state)

    assert counter.events[0] == "train_start"
    assert counter.events[-1] == "train_end"
    assert "epoch_start:1" in counter.events
    assert "epoch_end:2" in counter.events


def test_csv_logger_writes_header_and_rows(tmp_path: Path):
    state = HookState(output_dir=tmp_path, run_name="t")
    state.train_loss = 1.5
    state.val_loss = 1.2
    state.lr = 1e-3
    state.grad_norm = 0.5
    state.components = {"latent_mse": 0.9, "decoder_mse": 0.6}

    h = CSVLossLogger()
    h.on_train_start(state)
    state.epoch = 1
    state.global_step = 10
    h.on_epoch_end(state)
    state.epoch = 2
    state.global_step = 20
    state.train_loss = 1.0
    state.val_loss = 0.9
    h.on_epoch_end(state)
    h.on_train_end(state)

    csv_path = tmp_path / "train_log.csv"
    assert csv_path.exists()
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == 3  # header + 2 rows
    assert "train_loss" in lines[0]
    assert "component_latent_mse" in lines[0]


def test_console_logger_runs_without_crashing(tmp_path: Path, caplog):
    state = HookState(output_dir=tmp_path, run_name="t")
    state.epoch = 1
    state.global_step = 2
    state.lr = 1e-3
    state.components = {"a": 1.0}
    state.extra.update(
        {
            "epoch_step": 2,
            "steps_per_epoch": 10,
            "total_epochs": 3,
            "total_steps": 30,
            "epoch_start_time_s": 0.0,
            "fit_start_time_s": 0.0,
        }
    )
    h = ConsoleLogger(log_every_n_steps=1)
    with caplog.at_level("INFO"):
        h.on_step_end(state)
    h.on_epoch_end(state)
    text = caplog.text
    assert "epoch 1/3" in text
    assert "epoch_step=2/10" in text
    assert "eta_epoch=" in text
