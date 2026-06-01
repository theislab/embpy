"""Dataloader builders.

Wrap dataset construction + train/test splitting + ``DataLoader`` setup
behind a single function keyed off the package's :class:`DataConfig`,
:class:`SplitConfig`, and :class:`ActionEmbeddingConfig`. Splits are
persisted on disk so the world model and every baseline see
byte-identical train/test sets.

The model input contract is AnnData-only: state embeddings come from
``adata.obsm[data.state_obsm_key]`` and action embeddings come from
``adata.obsm[action_embedding.obsm_key]``. Both are materialised eagerly
inside ``from_h5ad`` so ``__getitem__`` only ever hits numpy arrays.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from world_model.configs import ActionEmbeddingConfig, DataConfig, SplitConfig, StateBackboneConfig
from world_model.data.datasets.anndata import AnnDataSequenceDataset
from world_model.data.datasets.base import GeneIndexer, PerturbationSequenceDataset
from world_model.data.embeddings import ActionEmbeddingProvider, build_provider
from world_model.data.preprocessing import sequence_collate_fn
from world_model.data.splits import SplitArtifact, split_or_load
from world_model.models.encoders.backbones import StateBackboneProvider

logger = logging.getLogger(__name__)


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
    query_gene_table: np.ndarray | None = None
    query_indexer: GeneIndexer | None = None
    query_provider: ActionEmbeddingProvider | None = None
    state_backbone: StateBackboneProvider | None = None
    state_backbone_embedding_dim: int | None = None


def build_dataloaders(
    cfg: DataConfig,
    *,
    split_cfg: SplitConfig | None = None,
    action_cfg: ActionEmbeddingConfig | None = None,
    query_action_cfg: ActionEmbeddingConfig | None = None,
    state_backbone_cfg: StateBackboneConfig | None = None,
    seed: int = 0,
    output_dir: str | Path | None = None,
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
        Action-embedding config. Must use ``source="anndata_obsm"``;
        precomputed tables and BioEmbedder models should be attached to
        the AnnData before training.
    seed
        Used for the random sampler RNG; the split's own seed is
        independent so split results are stable across model seeds.
    output_dir
        If set, the split and ``action_embedding_meta.json`` are
        persisted under that directory.
    """
    from torch.utils.data import DataLoader

    if split_cfg is None:
        split_cfg = SplitConfig()
    if action_cfg is None:
        action_cfg = ActionEmbeddingConfig()

    rng = np.random.default_rng(seed)
    provider = build_provider(action_cfg, data_cfg=cfg)
    query_provider: ActionEmbeddingProvider | None = None
    if query_action_cfg is not None and getattr(query_action_cfg, "obsm_key", ""):
        if getattr(cfg, "context_mode", "trajectory") != "incontext_set":
            raise ValueError(
                "query_action_embedding was configured, but data.context_mode is "
                f"{getattr(cfg, 'context_mode', None)!r}. Query action embeddings "
                "are only meaningful with data.context_mode='incontext_set'."
            )
        query_provider = build_provider(query_action_cfg, data_cfg=cfg)
    full_dataset, gene_table, indexer, gene_symbols, query_gene_table, query_indexer = AnnDataSequenceDataset.from_h5ad(
        h5ad_path=cfg.h5ad_path,
        provider=provider,
        query_provider=query_provider,
        perturbation_key=cfg.perturbation_key,
        control_label=cfg.control_label,
        state_obsm_key=cfg.state_obsm_key,
        sequence_length=cfg.sequence_length,
        stack_size=cfg.stack_size,
        n_pert=cfg.n_pert,
        rng=rng,
        bucket_key=getattr(cfg, "sequence_bucket_key", None),
        context_mode=getattr(cfg, "context_mode", "trajectory"),
        incontext_support_size=getattr(cfg, "incontext_support_size", 16),
        incontext_support_strategy=getattr(cfg, "incontext_support_strategy", "random"),
    )

    # Capture per-row statuses for downstream metadata. ``provider`` is
    # used by ``AnnDataSequenceDataset.from_h5ad`` to call
    # :meth:`build_table` which now delegates to
    # :meth:`embed_with_status`; the resolved bookkeeping is stored on
    # the provider as ``_last_*`` attributes.
    n_unresolved_attr = getattr(provider, "_last_unresolved", []) or []
    n_control_attr = getattr(provider, "_last_controls", []) or []

    state_backbone, state_embedding_dim = _materialize_state_obsm_metadata(
        full_dataset=full_dataset,
        state_backbone_cfg=state_backbone_cfg,
        state_obsm_key=cfg.state_obsm_key,
        h5ad_path=cfg.h5ad_path,
        output_dir=output_dir,
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
        status_path = Path(output_dir) / "action_embedding_status.json"
        status_path.write_text(
            json.dumps(
                {
                    "n_resolved": int(meta.n_resolved),
                    "n_control": int(meta.n_control),
                    "n_unresolved": int(meta.n_unresolved),
                    "n_mixed": int(meta.n_mixed),
                    "unresolved_symbols": list(meta.unresolved_symbols),
                    "control_symbols": list(meta.control_symbols),
                    "mixed_symbols": list(meta.mixed_symbols),
                    "control_sentinel_seed": meta.control_sentinel_seed,
                },
                indent=2,
                default=str,
            )
        )
        logger.info(
            "Action embedding meta -> %s and %s (source=%s, dim=%d, resolved=%d, control=%d, unresolved=%d/%d)",
            meta_path,
            status_path,
            meta.source,
            meta.embedding_dim,
            meta.n_resolved,
            meta.n_control,
            meta.n_unresolved,
            meta.n_symbols,
        )
        if query_provider is not None and query_gene_table is not None:
            q_n_unresolved = (
                int(np.sum(np.all(query_gene_table[1:] == 0, axis=1))) if query_gene_table.shape[0] > 1 else 0
            )
            q_meta = query_provider.metadata(
                n_symbols=max(query_gene_table.shape[0] - 1, 0),
                n_unresolved=q_n_unresolved,
            )
            q_meta_path = Path(output_dir) / "query_action_embedding_meta.json"
            q_meta_path.write_text(json.dumps(asdict(q_meta), indent=2, default=str))
            q_status_path = Path(output_dir) / "query_action_embedding_status.json"
            q_status_path.write_text(
                json.dumps(
                    {
                        "n_resolved": int(q_meta.n_resolved),
                        "n_control": int(q_meta.n_control),
                        "n_unresolved": int(q_meta.n_unresolved),
                        "n_mixed": int(q_meta.n_mixed),
                        "unresolved_symbols": list(q_meta.unresolved_symbols),
                        "control_symbols": list(q_meta.control_symbols),
                        "mixed_symbols": list(q_meta.mixed_symbols),
                        "control_sentinel_seed": q_meta.control_sentinel_seed,
                    },
                    indent=2,
                    default=str,
                )
            )
            logger.info(
                "Query action embedding meta -> %s and %s (source=%s, dim=%d, resolved=%d, control=%d, unresolved=%d/%d)",
                q_meta_path,
                q_status_path,
                q_meta.source,
                q_meta.embedding_dim,
                q_meta.n_resolved,
                q_meta.n_control,
                q_meta.n_unresolved,
                q_meta.n_symbols,
            )

    if getattr(action_cfg, "fail_on_unresolved", False) and len(n_unresolved_attr) > 0:
        preview = list(n_unresolved_attr)[:10]
        raise RuntimeError(
            f"action_embedding.fail_on_unresolved=True and the provider "
            f"could not embed {len(n_unresolved_attr)} of the requested "
            f"symbols (first 10: {preview}). Either fix the alias drift "
            f"upstream of BioEmbedder (e.g. via "
            f"GeneResolver.resolve_symbol) or set "
            f"action_embedding.fail_on_unresolved=False to permit zero "
            f"rows for the unresolved genes."
        )
    if (
        query_action_cfg is not None
        and query_provider is not None
        and getattr(query_action_cfg, "fail_on_unresolved", False)
        and len(getattr(query_provider, "_last_unresolved", []) or []) > 0
    ):
        q_unresolved = list(getattr(query_provider, "_last_unresolved", []) or [])
        raise RuntimeError(
            "query_action_embedding.fail_on_unresolved=True and the provider "
            f"could not embed {len(q_unresolved)} symbols (first 10: {q_unresolved[:10]})."
        )
    if n_control_attr:
        logger.info(
            "Action embedding: %d CONTROL rows mapped to deterministic sentinel vector (seed=%s).",
            len(n_control_attr),
            getattr(action_cfg, "control_sentinel_seed", 0),
        )

    cache_path: Path | None = None
    if split_cfg.cache_path is not None:
        cache_path = Path(split_cfg.cache_path)
    elif output_dir is not None:
        cache_path = Path(output_dir) / "splits" / f"{_dataset_cache_stem(cfg)}.npz"

    spec = split_or_load(
        full_dataset.perturbation_labels,
        cache_path=cache_path,
        control_label=cfg.control_label,
        split_by=split_cfg.split_by,
        train_fraction=split_cfg.train_fraction,
        seed=split_cfg.seed,
        keep_control_in_test=split_cfg.keep_control_in_test,
    )

    train_dataset = full_dataset.subset(
        cell_indices=spec.train_indices,
        n_sequences_per_epoch=cfg.n_sequences_per_epoch,
        rng=np.random.default_rng(seed),
    )
    val_dataset = full_dataset.subset(
        cell_indices=spec.test_indices,
        n_sequences_per_epoch=cfg.n_sequences_per_epoch,
        rng=np.random.default_rng(seed + 1),
    )

    # prefetch_factor=4: queue 4 batches per worker (default is 2).
    # With num_workers=8 that's 32 ready batches -- enough headroom to
    # keep the H100 fed even when an individual sample is slow.
    #
    # persistent_workers stays False here on purpose. With True, the
    # worker pool is reused across epochs (faster epoch starts), but
    # the well-known DataLoader heap-growth pattern (re-pickling numpy
    # samples through the queue) accumulates without bound and OOMs
    # the host process around ~64 GB after a few hundred steps on this
    # dataset. With persistent_workers=False, workers are torn down at
    # epoch end and the heap is freed; epoch starts cost ~20 s but the
    # run stays alive.
    # Both are no-ops when num_workers=0 (PyTorch ignores them).
    _persistent = False
    _prefetch = 4 if cfg.num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=sequence_collate_fn,
        drop_last=True,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=sequence_collate_fn,
        drop_last=False,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
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
        query_gene_table=query_gene_table,
        query_indexer=query_indexer,
        query_provider=query_provider,
        state_backbone=state_backbone,
        state_backbone_embedding_dim=state_embedding_dim,
    )


def _dataset_cache_stem(cfg: DataConfig) -> str:
    """Return a split-cache stem; ``data.dataset`` is just a human label."""
    if cfg.dataset:
        return str(cfg.dataset)
    if cfg.h5ad_path:
        return Path(cfg.h5ad_path).stem
    return "dataset"


def _materialize_state_obsm_metadata(
    *,
    full_dataset: PerturbationSequenceDataset,
    state_backbone_cfg: StateBackboneConfig | None,
    state_obsm_key: str,
    h5ad_path: str | Path,
    output_dir: str | Path | None,
) -> tuple[StateBackboneProvider | None, int | None]:
    """Persist metadata for the already-attached state embedding matrix."""
    kind = getattr(state_backbone_cfg, "kind", "local") if state_backbone_cfg is not None else "local"
    dim = int(full_dataset.expression.shape[1])
    if output_dir is not None:
        meta = {
            "kind": "anndata_obsm",
            "state_head_kind": kind,
            "h5ad_path": str(h5ad_path),
            "obsm_key": str(state_obsm_key),
            "embedding_dim": dim,
            "n_cells_encoded": int(full_dataset.expression.shape[0]),
        }
        out_path = Path(output_dir) / "state_embedding_meta.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(meta, indent=2, default=str))
        logger.info(
            "State embedding meta -> %s (obsm=%s, head=%s, dim=%d, n_cells=%d)",
            out_path,
            state_obsm_key,
            kind,
            dim,
            full_dataset.expression.shape[0],
        )
    # Non-local heads need this dim so build_world_model can construct a
    # ForeignBackboneHead without instantiating a heavy STATE/STACK provider.
    if kind in {"state", "stack"}:
        return None, dim
    if output_dir is not None:
        logger.debug("Local state head will consume adata.obsm[%r] directly.", state_obsm_key)
    return None, None


__all__ = ["DataArtifacts", "build_dataloaders"]
