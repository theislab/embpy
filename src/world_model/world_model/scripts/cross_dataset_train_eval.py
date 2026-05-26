"""Train on one perturb-seq dataset and evaluate on another.

This entry point is intentionally narrower than ``scripts.train``:

* train on 100% of the source dataset,
* optionally fine-tune on the target train split,
* evaluate either all target perturbations (zero-shot) or the held-out
  target split (fine-tune),
* use one shared action-embedding table over source union target
  perturbation labels so target action indices are valid at inference.

It is used by the local MPS launcher for the "Nadig -> Replogle" and
"Replogle -> Nadig" transfer checks.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from world_model.configs import (
    WorldModelConfig,
    apply_cli_overrides,
    load_yaml_config,
    validate_world_model_data_sources,
)
from world_model.data import DataArtifacts, build_dataloaders
from world_model.data.preprocessing import sequence_collate_fn
from world_model.data.splits import SplitArtifact
from world_model.scripts.train import (
    _build_model,
    _run_eval_and_report,
    _snapshot_run_provenance,
    _write_run_info,
)
from world_model.training import WorldModelTrainer
from world_model.utils import seed_everything, setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for cross-dataset training."""
    parser = argparse.ArgumentParser(
        description="Train on one dataset and evaluate on another.",
    )
    parser.add_argument(
        "--mode",
        choices=("zero-shot", "fine-tune"),
        default="zero-shot",
        help="zero-shot evaluates the target directly; fine-tune trains on the target train split first.",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=None,
        help="Number of target fine-tuning epochs. Defaults to train.n_epochs.",
    )
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--source-h5ad", required=True)
    parser.add_argument("--target-h5ad", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args(argv)


def _all_noncontrol_labels(artifacts: DataArtifacts) -> list[str]:
    labels = np.asarray(artifacts.full_dataset.perturbation_labels).astype(str)
    control = str(artifacts.full_dataset.control_label)
    return sorted(set(labels.tolist()) - {control})


def _all_target_split(artifacts: DataArtifacts, *, seed: int) -> SplitArtifact:
    labels = np.asarray(artifacts.full_dataset.perturbation_labels).astype(str)
    control = str(artifacts.full_dataset.control_label)
    train_indices = np.flatnonzero(labels == control)
    test_indices = np.flatnonzero(labels != control)
    return SplitArtifact(
        split_by="perturbation",
        train_indices=train_indices,
        test_indices=test_indices,
        train_perturbations=[],
        test_perturbations=sorted(set(labels.tolist()) - {control}),
        seed=int(seed),
    )


def _set_indexer(artifacts: DataArtifacts, indexer: Any) -> DataArtifacts:
    artifacts.full_dataset.indexer = indexer
    artifacts.train_dataset.indexer = indexer
    artifacts.val_dataset.indexer = indexer
    artifacts.indexer = indexer
    return artifacts


def _full_source_train_loader(cfg: WorldModelConfig, artifacts: DataArtifacts):
    from torch.utils.data import DataLoader

    all_indices = np.arange(artifacts.full_dataset.expression.shape[0], dtype=np.int64)
    train_dataset = artifacts.full_dataset.subset(
        all_indices,
        n_sequences_per_epoch=cfg.data.n_sequences_per_epoch,
        rng=np.random.default_rng(cfg.seed),
    )
    return DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=sequence_collate_fn,
        drop_last=True,
        persistent_workers=False,
        prefetch_factor=4 if cfg.data.num_workers > 0 else None,
    )


def _write_cross_metadata(
    path: Path,
    *,
    mode: str,
    source_cfg: WorldModelConfig,
    target_cfg: WorldModelConfig,
    source_labels: list[str],
    target_labels: list[str],
    union_labels: list[str],
) -> None:
    payload = {
        "source_dataset": source_cfg.data.dataset,
        "target_dataset": target_cfg.data.dataset,
        "source_h5ad_path": source_cfg.data.h5ad_path,
        "target_h5ad_path": target_cfg.data.h5ad_path,
        "n_source_perturbations": len(source_labels),
        "n_target_perturbations": len(target_labels),
        "n_union_action_labels": len(union_labels),
        "source_expression_dim": source_cfg.data.n_top_genes,
        "target_expression_dim": target_cfg.data.n_top_genes,
        "action_embedding": dataclasses.asdict(source_cfg.action_embedding),
        "mode": mode,
        "note": (
            "Cross-dataset transfer trains on all source cells. In zero-shot "
            "mode it evaluates all target non-control perturbations directly. "
            "In fine-tune mode it first trains on the target train split and "
            "evaluates the held-out target split. Local-backbone runs require "
            "matching expression dimensions; use the same n_top_genes unless "
            "a shared gene-space preprocessing layer is added."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    """Run source training, optional target fine-tuning, and target evaluation."""
    import datetime as _dt

    args = parse_args(argv)
    source_cfg: WorldModelConfig = load_yaml_config(args.source_config)
    target_cfg: WorldModelConfig = load_yaml_config(args.target_config)
    source_cfg = apply_cli_overrides(source_cfg, args.overrides)
    target_cfg = apply_cli_overrides(target_cfg, args.overrides)

    source_cfg.data.dataset = args.source_name
    target_cfg.data.dataset = args.target_name
    source_cfg.data.h5ad_path = args.source_h5ad
    target_cfg.data.h5ad_path = args.target_h5ad
    target_cfg.action_embedding = deepcopy(source_cfg.action_embedding)
    target_cfg.state_backbone = deepcopy(source_cfg.state_backbone)
    target_cfg.eval = deepcopy(source_cfg.eval)
    target_cfg.train = deepcopy(source_cfg.train)
    source_cfg.validate()
    target_cfg.validate()
    validate_world_model_data_sources(source_cfg)
    validate_world_model_data_sources(target_cfg)

    source_cfg.output_dir = args.output_dir
    source_cfg.run_name = args.run_name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(level=getattr(logging, args.log_level), log_file=output_dir / "train.log")
    seed_everything(source_cfg.seed)
    started_at = _dt.datetime.now().isoformat(timespec="seconds")
    _snapshot_run_provenance(
        cfg=source_cfg,
        output_dir=output_dir,
        source_yaml=args.source_config,
        overrides=args.overrides,
    )
    _write_run_info(
        source_cfg,
        output_dir,
        args.source_config,
        args.overrides,
        status="running",
        started_at=started_at,
        extras={
            "cross_dataset": {
                "source_config": args.source_config,
                "target_config": args.target_config,
                "source_name": args.source_name,
                "target_name": args.target_name,
                "mode": args.mode,
                "finetune_epochs": args.finetune_epochs,
            }
        },
    )

    try:
        source_artifacts = build_dataloaders(
            source_cfg.data,
            split_cfg=source_cfg.split,
            action_cfg=source_cfg.action_embedding,
            state_backbone_cfg=source_cfg.state_backbone,
            seed=source_cfg.seed,
            output_dir=output_dir / "_source_materialization",
        )
        target_artifacts = build_dataloaders(
            target_cfg.data,
            split_cfg=target_cfg.split,
            action_cfg=target_cfg.action_embedding,
            state_backbone_cfg=target_cfg.state_backbone,
            seed=target_cfg.seed,
            output_dir=output_dir / "_target_materialization",
        )

        source_labels = _all_noncontrol_labels(source_artifacts)
        target_labels = _all_noncontrol_labels(target_artifacts)
        union_labels = sorted(set(source_labels) | set(target_labels))
        union_table, union_indexer = source_artifacts.provider.build_table(union_labels)
        source_artifacts = _set_indexer(source_artifacts, union_indexer)
        target_artifacts = _set_indexer(target_artifacts, union_indexer)
        source_artifacts.gene_table = union_table
        target_artifacts.gene_table = union_table
        if args.mode == "zero-shot":
            target_artifacts.split = _all_target_split(target_artifacts, seed=target_cfg.seed)

        if source_artifacts.full_dataset.n_genes != target_artifacts.full_dataset.n_genes:
            raise ValueError(
                "Cross-dataset local-backbone transfer requires matching expression dimensions; "
                f"source has {source_artifacts.full_dataset.n_genes}, target has "
                f"{target_artifacts.full_dataset.n_genes}. Use the same data.n_top_genes "
                "or add shared gene-space preprocessing before cross-dataset evaluation."
            )

        _write_cross_metadata(
            output_dir / "cross_dataset_meta.json",
            mode=args.mode,
            source_cfg=source_cfg,
            target_cfg=target_cfg,
            source_labels=source_labels,
            target_labels=target_labels,
            union_labels=union_labels,
        )

        model = _build_model(
            source_cfg,
            source_artifacts.full_dataset.n_genes,
            union_table,
            state_backbone_provider=source_artifacts.state_backbone,
            state_backbone_embedding_dim=source_artifacts.state_backbone_embedding_dim,
        )
        trainer = WorldModelTrainer(
            model=model,
            optim_cfg=source_cfg.optim,
            loss_cfg=source_cfg.loss,
            train_cfg=source_cfg.train,
            output_dir=output_dir,
            run_name=source_cfg.run_name,
            state_backbone_cfg=source_cfg.state_backbone,
            backbone_provider=source_artifacts.state_backbone,
        )
        trainer.fit(train_loader=_full_source_train_loader(source_cfg, source_artifacts), val_loader=None)

        if args.mode == "fine-tune":
            finetune_epochs = int(
                args.finetune_epochs
                if args.finetune_epochs is not None
                else source_cfg.train.n_epochs
            )
            finetune_cfg = deepcopy(source_cfg)
            finetune_cfg.data = deepcopy(target_cfg.data)
            finetune_cfg.train = dataclasses.replace(
                source_cfg.train,
                n_epochs=finetune_epochs,
            )
            finetune_trainer = WorldModelTrainer(
                model=model,
                optim_cfg=source_cfg.optim,
                loss_cfg=source_cfg.loss,
                train_cfg=finetune_cfg.train,
                output_dir=output_dir / "finetune",
                run_name=f"{source_cfg.run_name}_finetune_{args.target_name}",
                state_backbone_cfg=source_cfg.state_backbone,
                backbone_provider=target_artifacts.state_backbone,
            )
            finetune_trainer.fit(
                train_loader=target_artifacts.train_loader,
                val_loader=target_artifacts.val_loader,
            )

        eval_cfg = deepcopy(source_cfg)
        eval_cfg.data = deepcopy(target_cfg.data)
        eval_cfg.run_name = f"{source_cfg.run_name}_eval_{args.target_name}"
        _run_eval_and_report(
            eval_cfg,
            output_dir,
            model,
            target_artifacts,
        )
    except BaseException as exc:
        _write_run_info(
            source_cfg,
            output_dir,
            args.source_config,
            args.overrides,
            status="failed",
            started_at=started_at,
            finished_at=_dt.datetime.now().isoformat(timespec="seconds"),
            extras={"error_type": type(exc).__name__, "error_message": str(exc)[:512]},
        )
        raise

    _write_run_info(
        source_cfg,
        output_dir,
        args.source_config,
        args.overrides,
        status="completed",
        started_at=started_at,
        finished_at=_dt.datetime.now().isoformat(timespec="seconds"),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
