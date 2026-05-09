"""Dataloader builders.

Wrap dataset construction + train/val splitting + ``DataLoader`` setup
behind a single function keyed off the package's :class:`DataConfig`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..configs import DataConfig
from .datasets.base import GeneIndexer, PerturbationSequenceDataset
from .datasets.nadig import NadigSequenceDataset
from .datasets.replogle import ReplogleSequenceDataset
from .preprocessing import sequence_collate_fn

logger = logging.getLogger(__name__)


_DATASET_REGISTRY: dict[str, Any] = {
    "replogle": ReplogleSequenceDataset,
    "nadig": NadigSequenceDataset,
}


class _SubsetDataset:
    """Cheap, picklable dataset slice (avoids importing torch at module load)."""

    def __init__(self, base: PerturbationSequenceDataset, indices: np.ndarray) -> None:
        self.base = base
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.base[int(self.indices[idx])]


def build_dataloaders(
    cfg: DataConfig,
    *,
    seed: int = 0,
) -> tuple[Any, Any, np.ndarray, GeneIndexer, list[str]]:
    """Build ``(train_loader, val_loader, gene_table, indexer, gene_symbols)``.

    The returned ``gene_table`` and ``indexer`` are needed to
    instantiate :class:`world_model.models.action.GeneEmbeddingAction`.
    """
    from torch.utils.data import DataLoader  # noqa: PLC0415

    if cfg.dataset not in _DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{cfg.dataset}'. Available: {sorted(_DATASET_REGISTRY)}"
        )
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
        rng=rng,
    )

    n = len(full_dataset)
    if n == 0:
        raise RuntimeError("Empty dataset after preprocessing -- check filters.")

    n_val = max(1, int(round(cfg.val_fraction * n)))
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_subset = _SubsetDataset(full_dataset, train_idx)
    val_subset = _SubsetDataset(full_dataset, val_idx)
    logger.info("Train/val split: %d / %d sequences", len(train_subset), len(val_subset))

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=sequence_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=sequence_collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader, gene_table, indexer, gene_symbols


__all__ = ["build_dataloaders"]
