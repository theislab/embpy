"""Replogle perturb-seq adapter.

Wraps the per-cell-type AnnData files from the Replogle 2022
``perturb-seq`` releases (``K562_essential``, ``RPE1_genome_wide``,
etc.). The adapter is dataset-aware about ``cell_type`` so a single
file containing multiple lines can be filtered down at load time.

Action embeddings are obtained via an injected
:class:`ActionEmbeddingProvider`. The legacy ``gene_embedding_path``
argument is still accepted for backward compatibility and routed
through a transient :class:`PrecomputedProvider`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..preprocessing import log_normalize_counts, select_highly_variable_genes
from .base import (
    GeneIndexer,
    PerturbationSequenceDataset,
)

if TYPE_CHECKING:
    from ..embeddings.provider import ActionEmbeddingProvider

logger = logging.getLogger(__name__)


def _load_adata(path: str | Path):  # type: ignore[no-untyped-def]
    try:
        import anndata as ad  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("anndata is required to load Replogle datasets.") from exc
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Replogle h5ad not found: {path}")
    return ad.read_h5ad(path)


class ReplogleSequenceDataset(PerturbationSequenceDataset):
    """Replogle 2022 perturb-seq sequence dataset."""

    @classmethod
    def from_h5ad(
        cls,
        h5ad_path: str | Path,
        *,
        provider: "ActionEmbeddingProvider | None" = None,
        gene_embedding_path: str | Path | None = None,
        perturbation_key: str = "perturbation",
        control_label: str = "non-targeting",
        cell_type_key: str | None = "cell_type",
        cell_type_filter: str | None = None,
        n_top_genes: int = 5000,
        log_normalize: bool = True,
        sequence_length: int = 8,
        stack_size: int = 4,
        n_pert: int = 2,
        rng: np.random.Generator | None = None,
        bucket_key: str | None = None,
    ) -> tuple[ReplogleSequenceDataset, np.ndarray, GeneIndexer, list[str]]:
        """Build a dataset from a Replogle ``.h5ad`` file.

        Exactly one of ``provider`` or ``gene_embedding_path`` must be
        set. The path form is accepted for backward compatibility and
        is wrapped in a :class:`PrecomputedProvider` internally.

        ``bucket_key`` (optional) names an ``adata.obs`` column whose
        values define the per-cell context bucket (e.g. ``"batch"``).
        When set, every emitted sequence is anchored to a single bucket
        so the transformer learns invariances of that biological /
        technical substrate under different perturbations.
        """
        adata = _load_adata(h5ad_path)
        if perturbation_key not in adata.obs.columns:
            raise KeyError(
                f"'{perturbation_key}' not in adata.obs (got {list(adata.obs.columns)})"
            )

        if cell_type_filter is not None:
            if cell_type_key is None:
                raise ValueError("cell_type_filter requires cell_type_key")
            mask = adata.obs[cell_type_key].astype(str).values == cell_type_filter
            n_before = adata.n_obs
            adata = adata[mask].copy()
            logger.info(
                "Filtered to cell_type=%s: %d -> %d cells", cell_type_filter, n_before, adata.n_obs,
            )

        x = adata.X
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = np.asarray(x, dtype=np.float32)

        keep = select_highly_variable_genes(x, n_top=n_top_genes)
        x = x[:, keep]
        gene_symbols = [str(s) for s in adata.var_names[keep]]
        if log_normalize:
            x = log_normalize_counts(x)

        labels = adata.obs[perturbation_key].astype(str).values
        unique_perturbed = [
            str(label) for label in np.unique(labels) if str(label) != control_label
        ]

        cell_buckets, bucket_value_map = _extract_bucket_codes(adata, bucket_key)

        provider = _resolve_provider(provider, gene_embedding_path)
        gene_table, indexer = provider.build_table(unique_perturbed)

        dataset = cls(
            expression=x,
            perturbation_labels=labels,
            indexer=indexer,
            sequence_length=sequence_length,
            stack_size=stack_size,
            n_pert=n_pert,
            control_label=control_label,
            rng=rng,
            cell_buckets=cell_buckets,
            bucket_value_map=bucket_value_map,
        )
        return dataset, gene_table, indexer, gene_symbols


def _extract_bucket_codes(
    adata, bucket_key: str | None,
) -> tuple[np.ndarray | None, dict[int, str] | None]:
    """Convert ``adata.obs[bucket_key]`` into an int code array.

    Returns ``(codes, id_to_value)`` or ``(None, None)`` when
    ``bucket_key`` is None. Uses pandas categorical codes so unknown
    / NaN values map to ``-1`` (which is not in our sampleable set
    and is therefore skipped at sample time).
    """
    if bucket_key is None:
        return None, None
    if bucket_key not in adata.obs.columns:
        raise KeyError(
            f"bucket_key='{bucket_key}' not in adata.obs "
            f"(got {list(adata.obs.columns)})"
        )
    import pandas as pd  # noqa: PLC0415

    col = adata.obs[bucket_key]
    cat = col.astype("category")
    codes = np.asarray(cat.cat.codes, dtype=np.int64)
    id_to_value = {int(i): str(v) for i, v in enumerate(cat.cat.categories)}
    n_unique = int((pd.Series(codes) >= 0).sum() and len(id_to_value))
    logger.info(
        "ReplogleSequenceDataset: bucket_key='%s' yields %d unique buckets "
        "(NaN/unknown -> -1, will be skipped).",
        bucket_key, n_unique,
    )
    return codes, id_to_value


def _resolve_provider(
    provider: "ActionEmbeddingProvider | None",
    gene_embedding_path: str | Path | None,
) -> "ActionEmbeddingProvider":
    if provider is not None and gene_embedding_path is not None:
        raise ValueError(
            "Pass either provider= or gene_embedding_path=, not both."
        )
    if provider is not None:
        return provider
    if gene_embedding_path is None:
        raise ValueError(
            "from_h5ad requires either an ActionEmbeddingProvider or a "
            "legacy gene_embedding_path."
        )
    # Lazy import: keeps the dataset module free of the embeddings
    # subpackage at import time, so circular imports stay impossible.
    from ..embeddings.precomputed import PrecomputedProvider  # noqa: PLC0415

    logger.warning(
        "ReplogleSequenceDataset.from_h5ad: gene_embedding_path is "
        "deprecated -- pass an ActionEmbeddingProvider instead.",
    )
    return PrecomputedProvider(gene_embedding_path)


__all__ = ["ReplogleSequenceDataset"]
