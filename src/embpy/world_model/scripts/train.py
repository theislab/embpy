"""End-to-end training script.

Usage:

    python -m embpy.world_model.scripts.train --config src/embpy/world_model/configs/replogle.yaml

The script is intentionally thin: it loads a YAML config, builds the
data loaders, builds the world model, fits the trainer and persists
the final checkpoint. All heavy lifting lives in the underlying
modules, so the same logic can also be invoked from a notebook.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from embpy.world_model.configs import WorldModelConfig, load_yaml_config
from embpy.world_model.data import build_dataloaders
from embpy.world_model.models.world_model import build_world_model
from embpy.world_model.training import WorldModelTrainer
from embpy.world_model.utils import seed_everything, setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the perturbation world model.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg: WorldModelConfig = load_yaml_config(args.config)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=getattr(logging, args.log_level), log_file=output_dir / "train.log")
    seed_everything(cfg.seed)

    logger.info("Run %s -- output_dir=%s", cfg.run_name, output_dir)
    logger.info("Config: %s", cfg.to_dict())

    train_loader, val_loader, gene_table, indexer, gene_symbols = build_dataloaders(
        cfg.data, seed=cfg.seed,
    )
    n_genes = len(gene_symbols)
    logger.info("Dataset built: n_genes=%d, n_action_rows=%d", n_genes, len(indexer))

    import torch  # noqa: PLC0415

    model = build_world_model(
        n_genes=n_genes,
        gene_embedding_table=torch.from_numpy(gene_table),
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
    logger.info("Model built: %d trainable params", model.num_parameters())

    trainer = WorldModelTrainer(
        model=model,
        optim_cfg=cfg.optim,
        loss_cfg=cfg.loss,
        train_cfg=cfg.train,
        output_dir=output_dir,
        run_name=cfg.run_name,
    )
    trainer.fit(train_loader=train_loader, val_loader=val_loader)


if __name__ == "__main__":  # pragma: no cover
    main()
