"""Dataloader builders.

Wrap dataset construction + train/test splitting + ``DataLoader`` setup
behind a single function keyed off the package's :class:`DataConfig`
and :class:`SplitConfig`. Splits are persisted on disk so the world
model and every baseline see byte-identical train/test sets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..configs import DataConfig, SplitConfig
from .datasets.base import GeneIndexer, PerturbationSequenceDataset
from .datasets.nadig import NadigSequenceDataset
from .datasets.replogle import ReplogleSequenceDataset
from .preprocessing import sequence_collate_fn
from .splits import SplitArtifact, split_or_load

logger = logging.getLogger(__name__)


_DATASET_REGISTRY: dict[str, Any] = {
    "replogle": ReplogleSequenceDataset,
    "nadig": NadigSequenceDataset,
}


@dataclass
class DataArtifacts:
    """Bundle returned by :func:`build_dataloaders`.

    Holds everything downstream code (trainer, baselines, evaluation,
    plotting) needs to address the same train/test indices that the
    world model is being trained on.
    """

    train_loader: Any
    val_loader: Any
    train_dataset: PerturbationSequenceDataset
    val_dataset: PerturbationSequenceDataset
    full_dataset: PerturbationSequenceDataset
    gene_table: np.ndarray
    indexer: GeneIndexer
    gene_symbols: list[str]
    split: SplitArtifact


def build_dataloaders(
    cfg: DataConfig,
    *,
    split_cfg: SplitConfig | None = None,
    seed: int = 0,
    output_dir: str | Path | None = None,
    finetune_perturbations: list[str] | None = None,
) -> DataArtifacts:
    """Build a :class:`DataArtifacts` bundle from a :class:`DataConfig`.

    Parameters
    ----------
    cfg
        Data config (paths + preprocessing).
    split_cfg
        Split policy. Defaults to a fresh :class:`SplitConfig` (perturbation
        split, 80/20).
    seed
        Used for the random sampler RNG; the split's own seed is
        independent so split results are stable across model seeds.
    output_dir
        If set, the split is persisted under ``{output_dir}/splits/{cfg.dataset}.npz``.
    finetune_perturbations
        Optional list of perturbation labels to *intersect* with the
        train side. Used by the transfer setup to fine-tune on a
        fraction of the train perturbations.
    """
    from torch.utils.data import DataLoader  # noqa: PLC0415

    if cfg.dataset not in _DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{cfg.dataset}'. Available: {sorted(_DATASET_REGISTRY)}")
    if split_cfg is None:
        split_cfg = SplitConfig()

    rng = np.random.default_rng(seed)
    cls = _DATASET_REGISTRY[cfg.dataset]
    full_dataset, gene_table, indexer, gene_symbols = cls.from_h5ad(
        h5ad_path=cfg.h5ad_path,
        gene_embedding_path=cfg.gene_embedding_path,
        perturbation_key=cfg.perturbation_key,
        control_label=cfg.control_label,
        n_top_genes=cfg.n_top_genes,
        log_normalize=cfg.log_normalize,
        sequence_length=cfg.sequence_length,
        stack_size=cfg.stack_size,
        n_pert=cfg.n_pert,
        rng=rng,
    )

    cache_path: Path | None = None
    if split_cfg.cache_path is not None:
        cache_path = Path(split_cfg.cache_path)
    elif output_dir is not None:
        cache_path = Path(output_dir) / "splits" / f"{cfg.dataset}.npz"

    spec = split_or_load(
        full_dataset.perturbation_labels,
        cache_path=cache_path,
        control_label=cfg.control_label,
        split_by=split_cfg.split_by,
        train_fraction=split_cfg.train_fraction,
        seed=split_cfg.seed,
        keep_control_in_test=split_cfg.keep_control_in_test,
    )

    train_indices = spec.train_indices
    if finetune_perturbations is not None:
        keep = set(finetune_perturbations)
        labels = full_dataset.perturbation_labels
        is_control = labels == cfg.control_label
        keep_mask = np.array([(lbl in keep) for lbl in labels])
        train_indices = np.unique(np.concatenate([
            np.flatnonzero(is_control),
            np.flatnonzero(keep_mask),
        ]))
        logger.info(
            "Sub-selecting fine-tune train: %d perts, %d cells", len(keep), train_indices.size,
        )

    train_dataset = full_dataset.subset(
        cell_indices=train_indices,
        n_sequences_per_epoch=cfg.n_sequences_per_epoch,
        rng=np.random.default_rng(seed),
    )
    val_dataset = full_dataset.subset(
        cell_indices=spec.test_indices,
        n_sequences_per_epoch=cfg.n_sequences_per_epoch,
        rng=np.random.default_rng(seed + 1),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=sequence_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=sequence_collate_fn,
        drop_last=False,
    )
    return DataArtifacts(
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        full_dataset=full_dataset,
        gene_table=gene_table,
        indexer=indexer,
        gene_symbols=gene_symbols,
        split=spec,
    )


__all__ = ["DataArtifacts", "build_dataloaders"]
