r"""Run every baseline against the same train/test split as a world-model run.

Loads the same :class:`SplitConfig` as the trainer, fits all baselines
on the train side, evaluates them on the test side, and writes:

* ``runs/<run_id>/baselines.csv``   -- one row per (baseline, metric)
* ``runs/<run_id>/comparison.csv``  -- baselines + (optional) world model

Usage:

        python -m world_model.scripts.run_baselines \\
            --config src/world_model/world_model/configs/datasets/replogle.yaml \\
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
from world_model.evaluation.plots import (
    per_perturbation_tables_to_long,
    plot_baseline_comparison,
)
from world_model.models.world_model import build_world_model
from world_model.utils import load_checkpoint, seed_everything, setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
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


def _build_optional_model(
    cfg: WorldModelConfig,
    n_genes: int,
    gene_table,
    ckpt_path: str | None,
    *,
    query_gene_table=None,
    state_backbone_provider=None,
    state_backbone_embedding_dim: int | None = None,
):
    if ckpt_path is None:
        return None
    model = build_world_model(
        n_genes=n_genes,
        gene_embedding_table=torch.from_numpy(gene_table),
        query_gene_embedding_table=(torch.from_numpy(query_gene_table) if query_gene_table is not None else None),
        encoder_kind=cfg.encoder.kind,
        d_model=cfg.encoder.d_model,
        stack_size=cfg.data.stack_size,
        encoder_layers=cfg.encoder.n_layers,
        encoder_heads=cfg.encoder.n_heads,
        dynamics_layers=cfg.dynamics.n_layers,
        dynamics_heads=cfg.dynamics.n_heads,
        dynamics_kind=cfg.dynamics.kind,
        incontext_support_size=getattr(cfg.data, "incontext_support_size", 16),
        dropout=cfg.dynamics.dropout,
        max_sequence_length=cfg.dynamics.max_sequence_length,
        use_action_token=cfg.dynamics.use_action_token,
        action_adapter_cfg=cfg.action_adapter,
        state_backbone_cfg=cfg.state_backbone,
        state_backbone_provider=state_backbone_provider,
        state_backbone_embedding_dim=state_backbone_embedding_dim,
    )
    payload = load_checkpoint(ckpt_path, map_location="cpu")
    model.load_state_dict(payload["state_dict"], strict=False)
    return model


def main(argv: list[str] | None = None) -> None:
    """Run baseline evaluation from CLI arguments."""
    import pandas as pd

    args = parse_args(argv)
    cfg = load_yaml_config(args.config)
    cfg = apply_cli_overrides(cfg, args.overrides)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=logging.INFO, log_file=output_dir / "baselines.log")
    seed_everything(cfg.seed)

    # Pass action_cfg + state_backbone_cfg so the AnnData obsm keys and
    # model dimensions match the train run.
    artifacts = build_dataloaders(
        cfg.data,
        split_cfg=cfg.split,
        action_cfg=cfg.action_embedding,
        query_action_cfg=cfg.query_action_embedding,
        state_backbone_cfg=cfg.state_backbone,
        seed=cfg.seed,
        output_dir=output_dir,
    )

    model = _build_optional_model(
        cfg,
        artifacts.state_backbone_embedding_dim
        if artifacts.state_backbone_embedding_dim is not None
        else len(artifacts.gene_symbols),
        artifacts.gene_table,
        args.checkpoint,
        query_gene_table=artifacts.query_gene_table,
        state_backbone_provider=artifacts.state_backbone,
        state_backbone_embedding_dim=artifacts.state_backbone_embedding_dim,
    )
    device = (
        "cuda"
        if (cfg.train.device == "auto" and torch.cuda.is_available())
        else (cfg.train.device if cfg.train.device != "auto" else "cpu")
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
    per_pert_tables = {}
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for name, res in results.items():
        if res.aggregate is None or res.aggregate.empty:
            continue
        agg = res.aggregate.copy()
        agg["name"] = name
        rows.append(agg)
        per_pert_tables[name] = res.per_perturbation
        if res.per_perturbation is not None and not res.per_perturbation.empty:
            res.per_perturbation.to_csv(eval_dir / f"per_pert_{name}.csv", index=False)
        for col in res.aggregate.columns:
            long_rows.append({"baseline": name, "metric": col, "value": float(res.aggregate[col].iloc[0])})
    aggregate_table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    per_pert_long = per_perturbation_tables_to_long(
        per_pert_tables,
        dataset=cfg.data.dataset,
        seed=cfg.seed,
    )
    if not per_pert_long.empty:
        per_pert_long.to_csv(eval_dir / "per_perturbation_long.csv", index=False)
    if not aggregate_table.empty:
        cols = ["name"] + [c for c in aggregate_table.columns if c != "name"]
        aggregate_table = aggregate_table[cols]
        aggregate_table.to_csv(output_dir / "comparison.csv", index=False)
        plot_baseline_comparison(
            aggregate_table,
            output_dir / "plots" / "comparison.png",
            per_perturbation_long=per_pert_long,
            dataset=cfg.data.dataset,
            seed=cfg.seed,
        )
    if long_rows:
        pd.DataFrame(long_rows).to_csv(output_dir / "baselines.csv", index=False)
        logger.info("Wrote baselines.csv (long format) and comparison.csv (wide format).")


if __name__ == "__main__":  # pragma: no cover
    main()
