"""End-to-end evaluation script.

Loads a checkpoint produced by :mod:`world_model.scripts.train` and
runs autoregressive rollouts on the validation split, printing the
aggregated metrics.

Usage:

    python -m embpy.world_model.scripts.eval \\
        --config src/embpy/world_model/configs/replogle.yaml \\
        --checkpoint outputs/world_model/replogle_k562_essential/replogle_k562_essential_final.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from embpy.world_model.configs import WorldModelConfig, load_yaml_config
from embpy.world_model.data import build_dataloaders
from embpy.world_model.evaluation import imagined_rollout
from embpy.world_model.models.world_model import build_world_model
from embpy.world_model.utils import load_checkpoint, seed_everything, setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained world model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args(argv)


def _resolve_device(spec: str):
    import torch  # noqa: PLC0415

    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(spec)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg: WorldModelConfig = load_yaml_config(args.config)
    output_dir = Path(cfg.output_dir)
    setup_logging(level=logging.INFO, log_file=output_dir / "eval.log")
    seed_everything(cfg.seed)

    _, val_loader, gene_table, indexer, gene_symbols = build_dataloaders(
        cfg.data, seed=cfg.seed,
    )
    n_genes = len(gene_symbols)

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
    payload = load_checkpoint(args.checkpoint, map_location="cpu")
    model.load_state_dict(payload["state_dict"], strict=False)
    device = _resolve_device(args.device)
    metrics = imagined_rollout(model, val_loader, device=device)
    logger.info("Final metrics: %s", metrics)


if __name__ == "__main__":  # pragma: no cover
    main()
