"""Run every baseline against the same train/test split as a world-model run.

Loads the same :class:`SplitConfig` as the trainer, fits all baselines
on the train side, evaluates them on the test side, and writes:

* ``runs/<run_id>/baselines.csv``   -- one row per (baseline, metric)
* ``runs/<run_id>/comparison.csv``  -- baselines + (optional) world model

Usage:

    python -m world_model.scripts.run_baselines \\
        --config src/world_model/world_model/configs/experiments/single_replogle.yaml \\
        --checkpoint runs/world_model/single_replogle/single_replogle_final.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from world_model.configs import (
    WorldModelConfig,
    apply_cli_overrides,
    load_yaml_config,
)
from world_model.data import build_dataloaders
from world_model.evaluation import run_evaluation
from world_model.evaluation.plots import plot_baseline_comparison
from world_model.models.world_model import build_world_model
from world_model.utils import load_checkpoint, seed_everything, setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baselines for a world-model run.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional world-model checkpoint to evaluate alongside baselines.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dotted overrides like 'eval.use_cell_eval=false'.",
    )
    return parser.parse_args(argv)


def _build_optional_model(cfg: WorldModelConfig, n_genes: int, gene_table, ckpt_path: str | None):
    if ckpt_path is None:
        return None
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
    payload = load_checkpoint(ckpt_path, map_location="cpu")
    model.load_state_dict(payload["state_dict"], strict=False)
    return model


def main(argv: list[str] | None = None) -> None:
    import pandas as pd  # noqa: PLC0415

    args = parse_args(argv)
    cfg = load_yaml_config(args.config)
    cfg = apply_cli_overrides(cfg, args.overrides)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=logging.INFO, log_file=output_dir / "baselines.log")
    seed_everything(cfg.seed)

    artifacts = build_dataloaders(
        cfg.data,
        split_cfg=cfg.split,
        seed=cfg.seed,
        output_dir=output_dir,
    )

    model = _build_optional_model(
        cfg, len(artifacts.gene_symbols), artifacts.gene_table, args.checkpoint,
    )
    device = "cuda" if (cfg.train.device == "auto" and torch.cuda.is_available()) else (
        cfg.train.device if cfg.train.device != "auto" else "cpu"
    )

    results = run_evaluation(
        artifacts=artifacts,
        output_dir=output_dir,
        eval_cfg=cfg.eval,
        perturbation_key=cfg.data.perturbation_key,
        control_label=cfg.data.control_label,
        model=model,
        device=device,
    )

    rows = []
    long_rows = []
    for name, res in results.items():
        if res.aggregate is None or res.aggregate.empty:
            continue
        agg = res.aggregate.copy()
        agg["name"] = name
        rows.append(agg)
        for col in res.aggregate.columns:
            long_rows.append({"baseline": name, "metric": col, "value": float(res.aggregate[col].iloc[0])})
    aggregate_table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not aggregate_table.empty:
        cols = ["name"] + [c for c in aggregate_table.columns if c != "name"]
        aggregate_table = aggregate_table[cols]
        aggregate_table.to_csv(output_dir / "comparison.csv", index=False)
        plot_baseline_comparison(aggregate_table, output_dir / "plots" / "comparison.png")
    if long_rows:
        pd.DataFrame(long_rows).to_csv(output_dir / "baselines.csv", index=False)
        logger.info("Wrote baselines.csv (long format) and comparison.csv (wide format).")


if __name__ == "__main__":  # pragma: no cover
    main()
