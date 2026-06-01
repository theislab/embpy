"""Generic AnnData perturbation-sequence adapter.

The world model does not special-case perturbation datasets. It expects
one AnnData file whose columns and matrices are named by config:

* ``adata.obs[perturbation_key]`` contains action / perturbation labels.
* ``adata.obsm[state_obsm_key]`` contains per-cell state embeddings.
* action embeddings are loaded by the injected provider, usually from
  ``adata.obsm[action_embedding.obsm_key]`` in the same file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from world_model.data.datasets.base import (
    GeneIndexer,
    PerturbationSequenceDataset,
    load_obsm_state_matrix,
)

if TYPE_CHECKING:
    from world_model.data.embeddings.provider import ActionEmbeddingProvider

logger = logging.getLogger(__name__)

_AUTO_BUCKET_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "cell_type",
        "celltype",
        "cell type",
        "cell_type_major",
        "celltype_major",
        "major_cell_type",
        "cell_type_coarse",
        "celltype_coarse",
        "cell_line",
        "lineage",
        "subtype",
        "annotation",
        "cell_annotation",
    ),
    (
        "batch",
        "batch_id",
        "sample",
        "sample_id",
        "donor",
        "donor_id",
        "gem_group",
        "library",
        "library_id",
        "plate",
        "experiment",
        "replicate",
        "orig.ident",
    ),
)


def _load_adata(path: str | Path):  # type: ignore[no-untyped-def]
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("anndata is required to load world-model AnnData inputs.") from exc
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"AnnData input not found: {path}")
    return ad.read_h5ad(path)


class AnnDataSequenceDataset(PerturbationSequenceDataset):
    """Perturbation sequence dataset built from generic AnnData keys."""

    @classmethod
    def from_h5ad(
        cls,
        h5ad_path: str | Path,
        *,
        provider: ActionEmbeddingProvider | None = None,
        query_provider: ActionEmbeddingProvider | None = None,
        perturbation_key: str = "perturbation",
        control_label: str = "non-targeting",
        state_obsm_key: str = "X_state",
        sequence_length: int = 8,
        stack_size: int = 4,
        n_pert: int = 2,
        rng: np.random.Generator | None = None,
        bucket_key: str | None = None,
        context_mode: str = "trajectory",
        incontext_support_size: int = 16,
    ) -> tuple[
        AnnDataSequenceDataset,
        np.ndarray,
        GeneIndexer,
        list[str],
        np.ndarray | None,
        GeneIndexer | None,
    ]:
        """Build a dataset from an AnnData ``.h5ad`` file.

        ``provider`` must be set and should read action embeddings from
        AnnData ``.obsm``. Cell/state embeddings are read directly from
        ``adata.obsm[state_obsm_key]``; ``adata.X`` is not transformed
        on the training path.
        """
        adata = _load_adata(h5ad_path)
        if perturbation_key not in adata.obs.columns:
            raise KeyError(f"{perturbation_key!r} not in adata.obs (got {list(adata.obs.columns)})")

        x, gene_symbols = load_obsm_state_matrix(adata, state_obsm_key)

        labels = adata.obs[perturbation_key].astype(str).to_numpy()
        unique_perturbed = [str(label) for label in np.unique(labels) if str(label) != control_label]

        cell_buckets, bucket_value_map = _extract_bucket_codes(adata, bucket_key)

        provider = _resolve_provider(provider)
        gene_table, indexer = provider.build_table(unique_perturbed)
        query_gene_table: np.ndarray | None = None
        query_indexer: GeneIndexer | None = None
        if query_provider is not None:
            query_gene_table, query_indexer = query_provider.build_table(unique_perturbed)
            if set(query_indexer.symbol_to_index) != set(indexer.symbol_to_index):
                missing_support = sorted(set(query_indexer.symbol_to_index) - set(indexer.symbol_to_index))
                missing_query = sorted(set(indexer.symbol_to_index) - set(query_indexer.symbol_to_index))
                raise ValueError(
                    "Query action table perturbation labels are not aligned with "
                    "the support action table. "
                    f"Only in query table: {missing_support[:10]}; "
                    f"only in support table: {missing_query[:10]}."
                )

        dataset = cls(
            expression=x,
            perturbation_labels=labels,
            indexer=indexer,
            query_indexer=query_indexer,
            sequence_length=sequence_length,
            stack_size=stack_size,
            n_pert=n_pert,
            control_label=control_label,
            rng=rng,
            cell_buckets=cell_buckets,
            bucket_value_map=bucket_value_map,
            context_mode=context_mode,
            incontext_support_size=incontext_support_size,
        )
        return dataset, gene_table, indexer, gene_symbols, query_gene_table, query_indexer


def _extract_bucket_codes(
    adata,  # type: ignore[no-untyped-def]
    bucket_key: str | None,
) -> tuple[np.ndarray | None, dict[int, str] | None]:
    """Convert one or more ``adata.obs`` columns into integer bucket ids.

    ``bucket_key`` supports three modes:

    * ``None`` / ``""`` / ``"none"`` / ``"off"``: disable bucketing.
    * ``"auto"``: choose common biological/technical columns from
      ``.obs`` (cell type and batch/sample/donor when available).
    * ``"col_a,col_b"`` or ``"col_a+col_b"``: build a composite bucket
      from the exact intersection of the requested columns.
    """
    keys = _resolve_bucket_keys(adata, bucket_key)
    if not keys:
        return None, None

    import pandas as pd

    n = int(adata.n_obs)
    valid = np.ones(n, dtype=bool)
    parts: list[np.ndarray] = []
    for key in keys:
        series = adata.obs[key]
        valid &= series.notna().to_numpy()
        values = series.astype("string").fillna("<NA>").astype(str).to_numpy()
        parts.append(np.char.add(f"{key}=", values))

    labels = np.empty(n, dtype=object)
    labels[:] = None
    if parts:
        composite = parts[0]
        for part in parts[1:]:
            composite = np.char.add(np.char.add(composite, "|"), part)
        labels[valid] = composite[valid]

    codes = np.full(n, -1, dtype=np.int64)
    cat = pd.Categorical(labels[valid])
    codes[valid] = np.asarray(cat.codes, dtype=np.int64)
    id_to_value = {int(i): str(v) for i, v in enumerate(cat.categories)}
    n_assigned = int(valid.sum())
    logger.info(
        "AnnDataSequenceDataset: bucket_key=%r resolved to obs columns %s; "
        "built %d composite context buckets across %d assigned cells "
        "(NaN/unknown -> -1, skipped at sample time).",
        bucket_key,
        keys,
        len(id_to_value),
        n_assigned,
    )
    return codes, id_to_value


def _resolve_bucket_keys(
    adata,  # type: ignore[no-untyped-def]
    bucket_key: str | None,
) -> list[str]:
    if bucket_key is None:
        return []
    raw = str(bucket_key).strip()
    if raw == "" or raw.lower() in {"none", "off", "false", "0"}:
        return []
    obs_columns = list(adata.obs.columns)
    if raw.lower() == "auto":
        found: list[str] = []
        lower_to_real = {str(c).lower(): str(c) for c in obs_columns}
        for group in _AUTO_BUCKET_GROUPS:
            for candidate in group:
                real = lower_to_real.get(candidate.lower())
                if real is not None:
                    found.append(real)
                    break
        if not found:
            logger.warning(
                "AnnDataSequenceDataset: bucket_key='auto' found none of the standard context columns "
                "in adata.obs. Bucketing is disabled. Available obs columns: %s",
                obs_columns,
            )
        return found

    keys = [part.strip() for chunk in raw.split(",") for part in chunk.split("+") if part.strip()]
    missing = [key for key in keys if key not in adata.obs.columns]
    if missing:
        raise KeyError(
            f"bucket_key={bucket_key!r} requested missing adata.obs column(s) {missing}; "
            f"available columns: {obs_columns}"
        )
    return keys


def _resolve_provider(provider: ActionEmbeddingProvider | None) -> ActionEmbeddingProvider:
    if provider is not None:
        return provider
    raise ValueError(
        "AnnDataSequenceDataset.from_h5ad requires an ActionEmbeddingProvider. "
        "Attach perturbation/action embeddings to AnnData and use "
        "action_embedding.source='anndata_obsm'."
    )


__all__ = ["AnnDataSequenceDataset"]
