"""Evaluate a trained world model and produce the full report.

Loads a checkpoint produced by :mod:`world_model.scripts.train` and
runs the perturbation evaluation pipeline (cell-eval-style metrics +
baselines + plots + report.md). Used by the ``eval_only.sbatch``
launcher when you want to re-evaluate without re-training.

Usage:

    python -m world_model.scripts.eval \\
        --config src/world_model/world_model/configs/experiments/single_replogle.yaml \\
        --checkpoint runs/world_model/single_replogle/single_replogle_final.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from world_model.configs import WorldModelConfig, apply_cli_overrides, load_yaml_config
from world_model.data import build_dataloaders
from world_model.models.world_model import build_world_model
from world_model.scripts.train import _run_eval_and_report
from world_model.utils import load_checkpoint, seed_everything, setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained world model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg: WorldModelConfig = load_yaml_config(args.config)
    cfg = apply_cli_overrides(cfg, args.overrides)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=logging.INFO, log_file=output_dir / "eval.log")
    seed_everything(cfg.seed)

    # Pass action_cfg + state_backbone_cfg so the action-embedding source
    # (bio_embedder vs precomputed) matches the train run. See the same
    # fix in run_baselines.py: without these, build_dataloaders falls back
    # to data.gene_embedding_path (genept 3072d), and load_state_dict
    # crashes with a shape mismatch when the checkpoint used a non-genept
    # embedding (e.g. borzoi_v0 -> 1536d).
    artifacts = build_dataloaders(
        cfg.data,
        split_cfg=cfg.split,
        action_cfg=cfg.action_embedding,
        state_backbone_cfg=cfg.state_backbone,
        seed=cfg.seed,
        output_dir=output_dir,
    )
    n_genes = len(artifacts.gene_symbols)

    model = build_world_model(
        n_genes=n_genes,
        gene_embedding_table=torch.from_numpy(artifacts.gene_table),
        encoder_kind=cfg.encoder.kind,
        d_model=cfg.encoder.d_model,
        stack_size=cfg.data.stack_size,
        encoder_layers=cfg.encoder.n_layers,
        encoder_heads=cfg.encoder.n_heads,
        dynamics_layers=cfg.dynamics.n_layers,
        dynamics_heads=cfg.dynamics.n_heads,
        dropout=cfg.dynamics.dropout,
        max_sequence_length=cfg.dynamics.max_sequence_length,
        use_action_token=cfg.dynamics.use_action_token,
    )
    payload = load_checkpoint(args.checkpoint, map_location="cpu")
    model.load_state_dict(payload["state_dict"], strict=False)

    _run_eval_and_report(cfg, output_dir, model, artifacts)


if __name__ == "__main__":  # pragma: no cover
    main()
