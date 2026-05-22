"""End-to-end training entry-point.

Handles both ``mode: single`` (one dataset, train + test split, single
fit) and ``mode: transfer`` (pretrain on Nadig, fine-tune on a fraction
of Replogle, evaluate on the rest of Replogle).

Usage:

    python -m world_model.scripts.train \\
        --config src/world_model/world_model/configs/experiments/single_replogle.yaml \\
        encoder.d_model=256 dynamics.n_layers=8

Positional arguments after ``--config`` are treated as dotted CLI
overrides applied on top of the YAML.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model.configs import (
    ActionEmbeddingConfig,
    DataConfig,
    WorldModelConfig,
    apply_cli_overrides,
    load_yaml_config,
)
from world_model.data import build_dataloaders
from world_model.data.inspect import dump_sample_contexts
from world_model.data.splits import subsample_train_perturbations
from world_model.evaluation import run_evaluation
from world_model.evaluation.plots import (
    plot_baseline_comparison,
    plot_deg_overlap_bar,
    plot_per_perturbation_metric,
    plot_pred_vs_real_scatter,
)
from world_model.evaluation.report import write_report
from world_model.models.world_model import build_world_model
from world_model.training import WorldModelTrainer, apply_encoder_swap
from world_model.utils import seed_everything, setup_logging

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


def _build_model(
    cfg: WorldModelConfig,
    n_genes: int,
    gene_table: np.ndarray,
    *,
    state_backbone_provider: Any = None,
    state_backbone_embedding_dim: int | None = None,
):
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


def _freeze_components(model, *, encoder: bool, dynamics: bool) -> None:
    if encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        logger.info("Encoder frozen for fine-tuning.")
    if dynamics:
        for p in model.dynamics.parameters():
            p.requires_grad_(False)
        logger.info("Dynamics frozen for fine-tuning.")


def _pretrain_action_cfg(cfg: WorldModelConfig) -> ActionEmbeddingConfig:
    """Action-embedding config for the pretrain phase.

    Resolution rule:
    1. If ``transfer.pretrain_action_encoder`` is set, use it verbatim.
    2. Else fall back to ``cfg.action_embedding`` and, for the legacy
       ``precomputed`` route, honour ``transfer.pretrain_gene_embedding_path``
       so existing YAMLs keep working.
    """
    if cfg.transfer.pretrain_action_encoder is not None:
        return dataclasses.replace(cfg.transfer.pretrain_action_encoder)
    pre_cfg = dataclasses.replace(cfg.action_embedding)
    if pre_cfg.source == "precomputed" and cfg.transfer.pretrain_gene_embedding_path:
        pre_cfg.path = cfg.transfer.pretrain_gene_embedding_path
    return pre_cfg


def _finetune_action_cfg(cfg: WorldModelConfig) -> ActionEmbeddingConfig:
    """Action-embedding config for the fine-tune phase.

    Defaults to the run's top-level ``action_embedding`` (Phase-1/2
    behaviour). When ``transfer.finetune_action_encoder`` is set, that
    block wins -- this is how the leave-one-encoder-out runner asks for
    Y on the fine-tune side while the run's nominal encoder is X.
    """
    if cfg.transfer.finetune_action_encoder is not None:
        return dataclasses.replace(cfg.transfer.finetune_action_encoder)
    return dataclasses.replace(cfg.action_embedding)


def _run_single(cfg: WorldModelConfig, output_dir: Path) -> tuple[object, object]:
    """Single-dataset train. Returns (model, artifacts)."""
    artifacts = build_dataloaders(
        cfg.data,
        split_cfg=cfg.split,
        action_cfg=cfg.action_embedding,
        state_backbone_cfg=cfg.state_backbone,
        seed=cfg.seed,
        output_dir=output_dir,
    )
    try:
        dump_sample_contexts(
            artifacts.train_dataset,
            output_dir / "sample_contexts.txt",
            n_sequences=5,
        )
    except Exception as exc:  # noqa: BLE001
        # Inspector is diagnostic-only; never fail the run on it.
        logger.warning("Skipping sample-context inspector: %s", exc)
    # When a foreign backbone replaces the in-memory expression with
    # cell embeddings, the decoder reconstructs back into embedding
    # space; its output dim must equal the backbone's embedding_dim,
    # not the original HVG count.
    n_genes = (
        artifacts.state_backbone_embedding_dim
        if artifacts.state_backbone_embedding_dim is not None
        else len(artifacts.gene_symbols)
    )
    model = _build_model(
        cfg, n_genes, artifacts.gene_table,
        state_backbone_provider=artifacts.state_backbone,
        state_backbone_embedding_dim=artifacts.state_backbone_embedding_dim,
    )
    logger.info(
        "Model: %d trainable params (action_dim=%d, decoder_dim=%d, hvgs=%d)",
        model.num_parameters(), artifacts.gene_table.shape[1],
        n_genes, len(artifacts.gene_symbols),
    )

    trainer = WorldModelTrainer(
        model=model,
        optim_cfg=cfg.optim,
        loss_cfg=cfg.loss,
        train_cfg=cfg.train,
        output_dir=output_dir,
        run_name=cfg.run_name,
        state_backbone_cfg=cfg.state_backbone,
        backbone_provider=artifacts.state_backbone,
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
    pretrain_action_cfg = _pretrain_action_cfg(cfg)

    pretrain_dir = output_dir / "pretrain"
    pretrain_artifacts = build_dataloaders(
        pretrain_data,
        split_cfg=cfg.split,
        action_cfg=pretrain_action_cfg,
        state_backbone_cfg=cfg.state_backbone,
        seed=cfg.seed,
        output_dir=pretrain_dir,
    )
    # Decoder output dim == observation dim. See _run_single for context.
    n_genes = (
        pretrain_artifacts.state_backbone_embedding_dim
        if pretrain_artifacts.state_backbone_embedding_dim is not None
        else len(pretrain_artifacts.gene_symbols)
    )
    pretrain_action_dim = pretrain_artifacts.gene_table.shape[1]
    model = _build_model(
        cfg, n_genes, pretrain_artifacts.gene_table,
        state_backbone_provider=pretrain_artifacts.state_backbone,
        state_backbone_embedding_dim=pretrain_artifacts.state_backbone_embedding_dim,
    )

    pretrain_train_cfg = dataclasses.replace(cfg.train, n_epochs=cfg.transfer.pretrain_epochs)
    pre_trainer = WorldModelTrainer(
        model=model,
        optim_cfg=cfg.optim,
        loss_cfg=cfg.loss,
        train_cfg=pretrain_train_cfg,
        output_dir=pretrain_dir,
        run_name=f"{cfg.run_name}_pretrain",
        state_backbone_cfg=cfg.state_backbone,
        backbone_provider=pretrain_artifacts.state_backbone,
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

    finetune_action_cfg = _finetune_action_cfg(cfg)
    finetune_dir = output_dir / "finetune"
    artifacts = build_dataloaders(
        cfg.data,
        split_cfg=cfg.split,
        action_cfg=finetune_action_cfg,
        state_backbone_cfg=cfg.state_backbone,
        seed=cfg.seed,
        output_dir=finetune_dir,
    )
    finetune_action_dim = artifacts.gene_table.shape[1]

    sub_split = subsample_train_perturbations(
        artifacts.split,
        artifacts.full_dataset.perturbation_labels,
        fraction=cfg.transfer.finetune_fraction,
        seed=cfg.seed,
    )
    artifacts = build_dataloaders(
        cfg.data,
        split_cfg=cfg.split,
        action_cfg=finetune_action_cfg,
        state_backbone_cfg=cfg.state_backbone,
        seed=cfg.seed,
        output_dir=finetune_dir,
        finetune_perturbations=sub_split.train_perturbations,
        state_backbone_override=pretrain_artifacts.state_backbone,
    )

    finetune_n_genes = (
        artifacts.state_backbone_embedding_dim
        if artifacts.state_backbone_embedding_dim is not None
        else len(artifacts.gene_symbols)
    )
    if finetune_n_genes != n_genes:
        logger.warning(
            "Pretrain decoder_dim (%d) != fine-tune decoder_dim (%d). Rebuilding encoder/decoder.",
            n_genes, finetune_n_genes,
        )
        new_model = _build_model(
            cfg, finetune_n_genes, artifacts.gene_table,
            state_backbone_provider=artifacts.state_backbone,
            state_backbone_embedding_dim=artifacts.state_backbone_embedding_dim,
        )
        new_model.dynamics.load_state_dict(model.dynamics.state_dict())
        if pretrain_action_dim == finetune_action_dim:
            new_model.action_encoder.load_state_dict(model.action_encoder.state_dict())
        else:
            logger.warning(
                "Action embedding dim changed during rebuild "
                "(%d -> %d); skipping action_encoder weight transfer.",
                pretrain_action_dim, finetune_action_dim,
            )
        model = new_model

    apply_encoder_swap(
        model,
        pretrain_provider=pretrain_artifacts.provider,
        finetune_provider=artifacts.provider,
        pretrain_table=pretrain_artifacts.gene_table,
        finetune_table=artifacts.gene_table,
        pretrain_indexer=pretrain_artifacts.indexer,
        finetune_indexer=artifacts.indexer,
        strategy=cfg.transfer.swap_strategy,
        adapter_cfg=cfg.action_adapter,
        d_model=cfg.encoder.d_model,
        alignment_epochs=cfg.transfer.alignment_epochs,
        alignment_lr=cfg.transfer.alignment_lr,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )

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
        state_backbone_cfg=cfg.state_backbone,
        backbone_provider=artifacts.state_backbone,
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


def _git_provenance(repo_dir: Path | str = ".") -> dict[str, Any]:
    """Capture the git SHA, branch, and dirty flag of ``repo_dir``.

    All keys default to ``None`` when git is unavailable, the directory
    is not a repo, or any individual call fails. Best-effort -- never
    raises.
    """
    import subprocess  # noqa: PLC0415

    out: dict[str, Any] = {"sha": None, "short_sha": None, "branch": None, "dirty": None}
    try:
        out["sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir), stderr=subprocess.DEVNULL, text=True,
        ).strip() or None
        out["short_sha"] = (out["sha"] or "")[:12] or None
    except Exception:  # noqa: BLE001
        return out
    try:
        out["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_dir), stderr=subprocess.DEVNULL, text=True,
        ).strip() or None
    except Exception:  # noqa: BLE001
        pass
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir), stderr=subprocess.DEVNULL, text=True,
        )
        out["dirty"] = bool(status.strip())
    except Exception:  # noqa: BLE001
        pass
    return out


def _slurm_provenance() -> dict[str, str | None]:
    """Snapshot SLURM env vars. Returns all-None outside of a SLURM job."""
    import os  # noqa: PLC0415

    keys = {
        "job_id": "SLURM_JOB_ID",
        "job_name": "SLURM_JOB_NAME",
        "partition": "SLURM_JOB_PARTITION",
        "node": "SLURMD_NODENAME",
        "nodelist": "SLURM_JOB_NODELIST",
        "ntasks": "SLURM_NTASKS",
        "cpus_per_task": "SLURM_CPUS_PER_TASK",
        "submit_time": "SLURM_JOB_START_TIME",
        "array_job_id": "SLURM_ARRAY_JOB_ID",
        "array_task_id": "SLURM_ARRAY_TASK_ID",
    }
    return {k: os.environ.get(v) for k, v in keys.items()}


def _apply_auto_suffix(cfg: WorldModelConfig) -> str:
    """Append a unique ``__job{id}__{timestamp}`` suffix to ``cfg.output_dir``.

    Two submissions of the same logical ``run_name`` would otherwise
    write into the same directory and clobber each other (train.log
    race condition, overwritten checkpoints, etc.). With the suffix,
    every run has its own directory keyed by SLURM job id (or by
    timestamp alone outside SLURM).

    Opt out by setting ``EMBPY_NO_AUTO_SUFFIX=1`` in the environment,
    or by passing a ``run_name`` / ``output_dir`` that already
    contains ``__job`` (idempotent).

    Returns the resulting suffix (or an empty string when no-op).
    """
    import datetime as _dt  # noqa: PLC0415
    import os  # noqa: PLC0415

    if os.environ.get("EMBPY_NO_AUTO_SUFFIX", "").strip() == "1":
        return ""
    if "__job" in str(cfg.output_dir) or "__local" in str(cfg.output_dir):
        return ""

    jobid = os.environ.get("SLURM_JOB_ID")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"__job{jobid}__{ts}" if jobid else f"__local__{ts}"
    cfg.output_dir = f"{cfg.output_dir}{suffix}"
    return suffix


def _key_hyperparams(cfg: WorldModelConfig) -> dict[str, Any]:
    """Pick out a small, comparable subset of hyperparameters.

    Used by ``run_info.json`` and the ``list_runs`` CLI so a single
    table can show "what changed between runs" without dumping the
    whole 200-key resolved config.
    """
    return {
        "data.dataset": cfg.data.dataset,
        "data.sequence_length": cfg.data.sequence_length,
        "data.stack_size": cfg.data.stack_size,
        "data.batch_size": cfg.data.batch_size,
        "data.n_top_genes": cfg.data.n_top_genes,
        "data.sequence_bucket_key": getattr(cfg.data, "sequence_bucket_key", None),
        "encoder.kind": cfg.encoder.kind,
        "encoder.d_model": cfg.encoder.d_model,
        "dynamics.kind": cfg.dynamics.kind,
        "dynamics.d_model": cfg.dynamics.d_model,
        "dynamics.n_layers": cfg.dynamics.n_layers,
        "dynamics.max_sequence_length": cfg.dynamics.max_sequence_length,
        "optim.lr": cfg.optim.lr,
        "optim.warmup_steps": cfg.optim.warmup_steps,
        "loss.latent_mse": cfg.loss.latent_mse,
        "loss.decoder_mse": cfg.loss.decoder_mse,
        "loss.info_nce": cfg.loss.info_nce,
        "loss.info_nce_mask_same_pert": cfg.loss.info_nce_mask_same_pert,
        "loss.action_counterfactual": cfg.loss.action_counterfactual,
        "state_backbone.kind": cfg.state_backbone.kind,
        "action_embedding.source": cfg.action_embedding.source,
        "action_embedding.model_name": cfg.action_embedding.model_name,
        "train.n_epochs": cfg.train.n_epochs,
    }


def _extract_final_metrics(output_dir: Path) -> dict[str, float | None]:
    """Best-effort pull of headline metrics from disk artifacts.

    Reads ``train_log.csv`` (last row) and ``world_model_metrics.csv``
    (the single eval-time row). Missing files / parse errors return
    a dict with ``None`` values.
    """
    out: dict[str, float | None] = {
        "final_epoch": None,
        "final_train_loss": None,
        "final_val_loss": None,
    }
    train_csv = output_dir / "train_log.csv"
    if train_csv.exists():
        try:
            import pandas as pd  # noqa: PLC0415

            df = pd.read_csv(train_csv)
            if not df.empty:
                last = df.iloc[-1]
                if "epoch" in df.columns:
                    out["final_epoch"] = int(last["epoch"])
                if "train_loss" in df.columns:
                    out["final_train_loss"] = float(last["train_loss"])
                if "val_loss" in df.columns and pd.notna(last.get("val_loss")):
                    out["final_val_loss"] = float(last["val_loss"])
        except Exception:  # noqa: BLE001
            pass
    wm_csv = output_dir / "world_model_metrics.csv"
    if wm_csv.exists():
        try:
            import pandas as pd  # noqa: PLC0415

            df = pd.read_csv(wm_csv)
            if not df.empty:
                row = df.iloc[0].to_dict()
                for k, v in row.items():
                    if k == "name":
                        continue
                    try:
                        out[f"eval.{k}"] = float(v)
                    except (TypeError, ValueError):
                        pass
        except Exception:  # noqa: BLE001
            pass
    return out


def _write_run_info(
    cfg: WorldModelConfig,
    output_dir: Path,
    source_yaml: str | Path | None,
    overrides: list[str] | None,
    *,
    status: str,
    started_at: str,
    finished_at: str | None = None,
    extras: dict[str, Any] | None = None,
) -> None:
    """Write / overwrite ``run_info.json`` for this run.

    Called at start (``status="running"``) and at end
    (``status="completed"`` or ``"failed"``). The end-of-run call
    also includes ``_extract_final_metrics`` so downstream tools can
    rank / filter runs by performance without re-loading checkpoints.
    """
    import datetime as _dt  # noqa: PLC0415
    import json  # noqa: PLC0415
    import socket  # noqa: PLC0415

    info: dict[str, Any] = {
        "status": status,
        "run_name": cfg.run_name,
        "output_dir": str(output_dir),
        "config_source": str(source_yaml) if source_yaml else None,
        "cli_overrides": list(overrides or []),
        "started_at": started_at,
        "finished_at": finished_at,
        "now": _dt.datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "slurm": _slurm_provenance(),
        "git": _git_provenance(),
        "key_hyperparams": _key_hyperparams(cfg),
        "final_metrics": _extract_final_metrics(output_dir) if status != "running" else None,
    }
    if extras:
        info.update(extras)
    try:
        (output_dir / "run_info.json").write_text(
            json.dumps(info, indent=2, default=str)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write run_info.json: %s", exc)


def _snapshot_run_provenance(
    cfg: WorldModelConfig,
    output_dir: Path,
    source_yaml: str | Path,
    overrides: list[str] | None,
) -> None:
    """Write the launch-time provenance into ``output_dir``.

    Persists three artifacts so a finished, mid-flight, or crashed run
    can always be reproduced from disk:

    * ``config.yaml``         -- resolved config (defaults filled in)
    * ``config_source.yaml``  -- verbatim copy of the YAML the user
                                  passed to ``--config``
    * ``cli_overrides.txt``   -- the dotted CLI overrides applied on
                                  top, one per line (empty file if none)

    Failures here never abort training; provenance is best-effort.
    """
    import shutil  # noqa: PLC0415

    try:
        _dump_config_yaml(cfg, output_dir / "config.yaml")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to dump resolved config.yaml: %s", exc)

    source_path = Path(source_yaml)
    target = output_dir / "config_source.yaml"
    try:
        if source_path.exists():
            shutil.copyfile(source_path, target)
        else:
            logger.warning(
                "Source config %s does not exist; skipping verbatim copy.",
                source_path,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to copy source YAML to %s: %s", target, exc)

    try:
        (output_dir / "cli_overrides.txt").write_text(
            "\n".join(overrides or []) + ("\n" if overrides else "")
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write cli_overrides.txt: %s", exc)


def _run_eval_and_report(cfg: WorldModelConfig, output_dir: Path, model, artifacts) -> None:
    import pandas as pd  # noqa: PLC0415

    # Refresh the resolved config -- captures any state mutated during
    # training (e.g. derived dims). The verbatim source YAML and the
    # CLI overrides are already on disk from _snapshot_run_provenance.
    _dump_config_yaml(cfg, output_dir / "config.yaml")

    # run_evaluation routes the in-context set model to its own
    # support-set + query predictor (_predict_incontext); the causal
    # model uses the rollout path. Both return the same
    # (n_test_cells, n_genes) contract, so baselines + compare/report
    # work identically for either.
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
    import datetime as _dt  # noqa: PLC0415

    args = parse_args(argv)
    cfg: WorldModelConfig = load_yaml_config(args.config)
    cfg = apply_cli_overrides(cfg, args.overrides)
    cfg.validate()

    # Append a unique __job{jobid}__{timestamp} suffix to output_dir
    # so concurrent submissions of the same run_name never collide.
    # Opt out with EMBPY_NO_AUTO_SUFFIX=1.
    suffix = _apply_auto_suffix(cfg)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=getattr(logging, args.log_level), log_file=output_dir / "train.log")
    seed_everything(cfg.seed)

    started_at = _dt.datetime.now().isoformat(timespec="seconds")
    logger.info("Run %s -- output_dir=%s -- mode=%s", cfg.run_name, output_dir, cfg.mode)
    if suffix:
        logger.info("Auto-suffix applied to output_dir: '%s'", suffix)
    logger.info("Config: %s", cfg.to_dict())

    # Snapshot launch provenance BEFORE any training so crashed,
    # cancelled, or mid-flight runs still have the exact config on
    # disk. Writes config.yaml (resolved), config_source.yaml
    # (verbatim source), and cli_overrides.txt.
    _snapshot_run_provenance(
        cfg=cfg,
        output_dir=output_dir,
        source_yaml=args.config,
        overrides=args.overrides,
    )
    _write_run_info(
        cfg, output_dir, args.config, args.overrides,
        status="running", started_at=started_at,
    )

    try:
        if cfg.mode == "transfer":
            if not cfg.transfer.enabled:
                raise ValueError("mode='transfer' but transfer.enabled=False")
            model, artifacts = _run_transfer(cfg, output_dir)
        elif cfg.mode == "single":
            model, artifacts = _run_single(cfg, output_dir)
        else:
            raise ValueError(f"Unknown mode {cfg.mode!r}; use 'single' or 'transfer'.")

        _run_eval_and_report(cfg, output_dir, model, artifacts)
    except BaseException as exc:  # noqa: BLE001
        # Catch BaseException so SLURM-side SIGTERMs (KeyboardInterrupt,
        # SystemExit) also leave a "failed" marker on disk.
        _write_run_info(
            cfg, output_dir, args.config, args.overrides,
            status="failed", started_at=started_at,
            finished_at=_dt.datetime.now().isoformat(timespec="seconds"),
            extras={"error_type": type(exc).__name__, "error_message": str(exc)[:512]},
        )
        raise

    _write_run_info(
        cfg, output_dir, args.config, args.overrides,
        status="completed", started_at=started_at,
        finished_at=_dt.datetime.now().isoformat(timespec="seconds"),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
