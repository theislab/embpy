"""Dataloader builders.

Wrap dataset construction + train/test splitting + ``DataLoader`` setup
behind a single function keyed off the package's :class:`DataConfig`,
:class:`SplitConfig`, and :class:`ActionEmbeddingConfig`. Splits are
persisted on disk so the world model and every baseline see
byte-identical train/test sets.

The action-embedding lookup is materialised eagerly inside
:meth:`from_h5ad`: ``__getitem__`` only ever hits a numpy array.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..configs import ActionEmbeddingConfig, DataConfig, SplitConfig
from .datasets.base import GeneIndexer, PerturbationSequenceDataset
from .datasets.nadig import NadigSequenceDataset
from .datasets.replogle import ReplogleSequenceDataset
from .embeddings import ActionEmbeddingProvider, build_provider
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
    provider: ActionEmbeddingProvider


def build_dataloaders(
    cfg: DataConfig,
    *,
    split_cfg: SplitConfig | None = None,
    action_cfg: ActionEmbeddingConfig | None = None,
    seed: int = 0,
    output_dir: str | Path | None = None,
    finetune_perturbations: list[str] | None = None,
) -> DataArtifacts:
    """Build a :class:`DataArtifacts` bundle from the data + split + action configs.

    Parameters
    ----------
    cfg
        Data config (paths + preprocessing).
    split_cfg
        Split policy. Defaults to a fresh :class:`SplitConfig` (perturbation
        split, 80/20).
    action_cfg
        Action-embedding config. Defaults to ``"precomputed"`` with
        empty path -- the registry then falls back to
        ``cfg.gene_embedding_path``.
    seed
        Used for the random sampler RNG; the split's own seed is
        independent so split results are stable across model seeds.
    output_dir
        If set, the split and ``action_embedding_meta.json`` are
        persisted under that directory.
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
    if action_cfg is None:
        action_cfg = ActionEmbeddingConfig()

    rng = np.random.default_rng(seed)
    cls = _DATASET_REGISTRY[cfg.dataset]
    provider = build_provider(action_cfg, data_cfg=cfg)
    full_dataset, gene_table, indexer, gene_symbols = cls.from_h5ad(
        h5ad_path=cfg.h5ad_path,
        provider=provider,
        perturbation_key=cfg.perturbation_key,
        control_label=cfg.control_label,
        n_top_genes=cfg.n_top_genes,
        log_normalize=cfg.log_normalize,
        sequence_length=cfg.sequence_length,
        stack_size=cfg.stack_size,
        n_pert=cfg.n_pert,
        rng=rng,
    )

    if output_dir is not None:
        # n_unresolved is the count of zero rows excluding the row 0
        # control / padding token we deliberately initialise to zero.
        n_unresolved = int(np.sum(np.all(gene_table[1:] == 0, axis=1))) if gene_table.shape[0] > 1 else 0
        meta = provider.metadata(
            n_symbols=max(gene_table.shape[0] - 1, 0),
            n_unresolved=n_unresolved,
        )
        meta_path = Path(output_dir) / "action_embedding_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(asdict(meta), indent=2, default=str))
        logger.info(
            "Action embedding meta -> %s (source=%s, dim=%d, unresolved=%d/%d)",
            meta_path, meta.source, meta.embedding_dim, meta.n_unresolved, meta.n_symbols,
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
        provider=provider,
    )


__all__ = ["DataArtifacts", "build_dataloaders"]
