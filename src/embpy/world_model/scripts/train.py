"""End-to-end training entry-point.

Handles both ``mode: single`` (one dataset, train + test split, single
fit) and ``mode: transfer`` (pretrain on Nadig, fine-tune on a fraction
of Replogle, evaluate on the rest of Replogle).

Usage:

    python -m embpy.world_model.scripts.train \\
        --config src/embpy/world_model/configs/experiments/single_replogle.yaml \\
        encoder.d_model=256 dynamics.n_layers=8

Positional arguments after ``--config`` are treated as dotted CLI
overrides applied on top of the YAML.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from embpy.world_model.configs import (
    DataConfig,
    TrainConfig,
    WorldModelConfig,
    apply_cli_overrides,
    load_yaml_config,
)
from embpy.world_model.data import build_dataloaders
from embpy.world_model.data.splits import subsample_train_perturbations
from embpy.world_model.evaluation import run_evaluation
from embpy.world_model.evaluation.plots import (
    plot_baseline_comparison,
    plot_deg_overlap_bar,
    plot_per_perturbation_metric,
    plot_pred_vs_real_scatter,
)
from embpy.world_model.evaluation.report import write_report
from embpy.world_model.models.world_model import build_world_model
from embpy.world_model.training import WorldModelTrainer
from embpy.world_model.utils import seed_everything, setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the perturbation world model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dotted overrides like 'encoder.d_model=256'.",
    )
    return parser.parse_args(argv)


def _build_model(cfg: WorldModelConfig, n_genes: int, gene_table: np.ndarray):
    return build_world_model(
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


def _freeze_components(model, *, encoder: bool, dynamics: bool) -> None:
    if encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        logger.info("Encoder frozen for fine-tuning.")
    if dynamics:
        for p in model.dynamics.parameters():
            p.requires_grad_(False)
        logger.info("Dynamics frozen for fine-tuning.")


def _run_single(cfg: WorldModelConfig, output_dir: Path) -> tuple[object, object]:
    """Single-dataset train. Returns (model, artifacts)."""
    artifacts = build_dataloaders(
        cfg.data,
        split_cfg=cfg.split,
        seed=cfg.seed,
        output_dir=output_dir,
    )
    n_genes = len(artifacts.gene_symbols)
    model = _build_model(cfg, n_genes, artifacts.gene_table)
    logger.info("Model: %d trainable params", model.num_parameters())

    trainer = WorldModelTrainer(
        model=model,
        optim_cfg=cfg.optim,
        loss_cfg=cfg.loss,
        train_cfg=cfg.train,
        output_dir=output_dir,
        run_name=cfg.run_name,
    )
    trainer.fit(train_loader=artifacts.train_loader, val_loader=artifacts.val_loader)
    return model, artifacts


def _run_transfer(cfg: WorldModelConfig, output_dir: Path) -> tuple[object, object]:
    """Pretrain on Nadig, fine-tune on a fraction of Replogle."""
    pretrain_data = DataConfig(
        dataset=cfg.transfer.pretrain_dataset,
        h5ad_path=cfg.transfer.pretrain_h5ad_path,
        gene_embedding_path=cfg.transfer.pretrain_gene_embedding_path,
        perturbation_key=cfg.data.perturbation_key,
        control_label=cfg.data.control_label,
        cell_type_key=None,
        n_top_genes=cfg.data.n_top_genes,
        log_normalize=cfg.data.log_normalize,
        stack_size=cfg.data.stack_size,
        sequence_length=cfg.data.sequence_length,
        n_pert=cfg.data.n_pert,
        batch_size=cfg.data.batch_size,
        val_fraction=cfg.data.val_fraction,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        n_sequences_per_epoch=cfg.data.n_sequences_per_epoch,
    )

    pretrain_dir = output_dir / "pretrain"
    pretrain_artifacts = build_dataloaders(
        pretrain_data,
        split_cfg=cfg.split,
        seed=cfg.seed,
        output_dir=pretrain_dir,
    )
    n_genes = len(pretrain_artifacts.gene_symbols)
    model = _build_model(cfg, n_genes, pretrain_artifacts.gene_table)

    pretrain_train_cfg = dataclasses.replace(cfg.train, n_epochs=cfg.transfer.pretrain_epochs)
    pre_trainer = WorldModelTrainer(
        model=model,
        optim_cfg=cfg.optim,
        loss_cfg=cfg.loss,
        train_cfg=pretrain_train_cfg,
        output_dir=pretrain_dir,
        run_name=f"{cfg.run_name}_pretrain",
    )

    if cfg.transfer.pretrain_checkpoint:
        pre_trainer.load_state(cfg.transfer.pretrain_checkpoint, strict=False)
        logger.info("Skipping pretrain phase; loaded %s", cfg.transfer.pretrain_checkpoint)
    else:
        pre_trainer.fit(
            train_loader=pretrain_artifacts.train_loader,
            val_loader=pretrain_artifacts.val_loader,
        )
        ckpt_path = pretrain_dir / f"{cfg.run_name}_pretrain_final.pt"
        logger.info("Pretrain done. Checkpoint at %s", ckpt_path)

    # Fine-tune phase: Replogle, p% of train perturbations.
    finetune_dir = output_dir / "finetune"
    artifacts = build_dataloaders(
        cfg.data,
        split_cfg=cfg.split,
        seed=cfg.seed,
        output_dir=finetune_dir,
    )
    sub_split = subsample_train_perturbations(
        artifacts.split,
        artifacts.full_dataset.perturbation_labels,
        fraction=cfg.transfer.finetune_fraction,
        seed=cfg.seed,
    )
    artifacts = build_dataloaders(
        cfg.data,
        split_cfg=cfg.split,
        seed=cfg.seed,
        output_dir=finetune_dir,
        finetune_perturbations=sub_split.train_perturbations,
    )

    # Sanity: the gene set may differ between pretrain (Nadig) and
    # fine-tune (Replogle). The encoder operates on a flat (G,) input so
    # we must rebuild it for the fine-tune gene count if it differs.
    if len(artifacts.gene_symbols) != n_genes:
        logger.warning(
            "Pretrain n_genes (%d) != fine-tune n_genes (%d). Rebuilding encoder/decoder.",
            n_genes, len(artifacts.gene_symbols),
        )
        new_model = _build_model(cfg, len(artifacts.gene_symbols), artifacts.gene_table)
        # Copy over only the dynamics + action encoder, which are gene-agnostic.
        new_model.dynamics.load_state_dict(model.dynamics.state_dict())
        new_model.action_encoder.load_state_dict(model.action_encoder.state_dict())
        model = new_model

    _freeze_components(
        model,
        encoder=cfg.transfer.freeze_encoder_during_finetune,
        dynamics=cfg.transfer.freeze_dynamics_during_finetune,
    )

    finetune_train_cfg = dataclasses.replace(cfg.train, n_epochs=cfg.transfer.finetune_epochs)
    ft_trainer = WorldModelTrainer(
        model=model,
        optim_cfg=cfg.optim,
        loss_cfg=cfg.loss,
        train_cfg=finetune_train_cfg,
        output_dir=finetune_dir,
        run_name=f"{cfg.run_name}_finetune",
    )
    ft_trainer.fit(
        train_loader=artifacts.train_loader,
        val_loader=artifacts.val_loader,
    )
    return model, artifacts


def _dump_config_yaml(cfg: WorldModelConfig, path: Path) -> None:
    """Persist the resolved config so compare.py / make_report.py can re-read it."""
    try:
        import yaml  # noqa: PLC0415

        with open(path, "w") as fp:
            yaml.safe_dump(cfg.to_dict(), fp, sort_keys=False)
    except ImportError:
        import json  # noqa: PLC0415

        with open(path, "w") as fp:
            json.dump(cfg.to_dict(), fp, indent=2, default=str)


def _run_eval_and_report(cfg: WorldModelConfig, output_dir: Path, model, artifacts) -> None:
    import pandas as pd  # noqa: PLC0415

    _dump_config_yaml(cfg, output_dir / "config.yaml")

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
    long_rows: list[dict[str, object]] = []
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
            long_rows.append({
                "baseline": name,
                "metric": col,
                "value": float(res.aggregate[col].iloc[0]),
            })

    aggregate_table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not aggregate_table.empty:
        cols = ["name"] + [c for c in aggregate_table.columns if c != "name"]
        aggregate_table = aggregate_table[cols]
        aggregate_table.to_csv(output_dir / "comparison.csv", index=False)

        wm_only = aggregate_table[aggregate_table["name"] == "world_model"]
        if not wm_only.empty:
            wm_only.to_csv(output_dir / "world_model_metrics.csv", index=False)

        baseline_only = aggregate_table[aggregate_table["name"] != "world_model"]
        if not baseline_only.empty and long_rows:
            pd.DataFrame(
                [r for r in long_rows if r["baseline"] != "world_model"]
            ).to_csv(output_dir / "baselines.csv", index=False)
        logger.info("Wrote comparison.csv with shape %s", aggregate_table.shape)

    plot_dir = output_dir / "plots"
    plot_paths: dict[str, str] = {}
    if "world_model" in results:
        wm = results["world_model"]
        plot_pred_vs_real_scatter(wm.real_adata, wm.pred_adata, plot_dir / "scatter_world_model.png")
        plot_per_perturbation_metric(wm.per_perturbation, plot_dir / "perpert_r2.png", metric="r2")
        plot_deg_overlap_bar(wm.per_perturbation, plot_dir / "deg_overlap.png")
        plot_paths.update({
            "Predicted vs real (world model)": "plots/scatter_world_model.png",
            "Per-perturbation R^2 (world model)": "plots/perpert_r2.png",
            "DEG overlap": "plots/deg_overlap.png",
        })
    if not aggregate_table.empty:
        plot_baseline_comparison(aggregate_table, plot_dir / "comparison.png")
        plot_paths["Baselines vs world model"] = "plots/comparison.png"

    if (output_dir / "plots" / "loss_curves.png").exists():
        plot_paths["Loss curves"] = "plots/loss_curves.png"

    write_report(
        output_dir=output_dir,
        run_name=cfg.run_name,
        cfg_dict=cfg.to_dict(),
        aggregate_table=aggregate_table,
        per_pert_tables=per_pert_tables,
        plot_paths=plot_paths,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg: WorldModelConfig = load_yaml_config(args.config)
    cfg = apply_cli_overrides(cfg, args.overrides)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=getattr(logging, args.log_level), log_file=output_dir / "train.log")
    seed_everything(cfg.seed)

    logger.info("Run %s -- output_dir=%s -- mode=%s", cfg.run_name, output_dir, cfg.mode)
    logger.info("Config: %s", cfg.to_dict())

    if cfg.mode == "transfer":
        if not cfg.transfer.enabled:
            raise ValueError("mode='transfer' but transfer.enabled=False")
        model, artifacts = _run_transfer(cfg, output_dir)
    elif cfg.mode == "single":
        model, artifacts = _run_single(cfg, output_dir)
    else:
        raise ValueError(f"Unknown mode {cfg.mode!r}; use 'single' or 'transfer'.")

    _run_eval_and_report(cfg, output_dir, model, artifacts)


if __name__ == "__main__":  # pragma: no cover
    main()
