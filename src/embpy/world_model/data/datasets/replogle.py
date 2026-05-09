"""Replogle 2022 perturb-seq adapter.

The on-disk format is a standard ``.h5ad`` with:

* ``adata.X``                 -- raw counts (CSR sparse).
* ``adata.obs[perturbation_key]`` -- gene symbol of the targeted gene
  ("non-targeting" for controls).
* ``adata.var_names``         -- gene symbols matching the embedding table.
* ``adata.obs[cell_type_key]`` -- ``"K562"`` or ``"RPE1"``.

This adapter loads the AnnData, runs the package preprocessing (HVG
filter + log1p) and hands a :class:`PerturbationSequenceDataset` to the
caller.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..preprocessing import log_normalize_counts, select_highly_variable_genes
from .base import (
    GeneIndexer,
    PerturbationSequenceDataset,
    load_gene_embedding_table,
)

logger = logging.getLogger(__name__)


def _load_adata(path: str | Path) -> Any:
    try:
        import anndata as ad  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("anndata is required to load Replogle datasets.") from exc
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Replogle h5ad not found: {path}")
    return ad.read_h5ad(path)


class ReplogleSequenceDataset(PerturbationSequenceDataset):
    """Replogle K562 / RPE1 perturb-seq sequence dataset."""

    @classmethod
    def from_h5ad(
        cls,
        h5ad_path: str | Path,
        gene_embedding_path: str | Path,
        *,
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
    ) -> tuple[ReplogleSequenceDataset, np.ndarray, GeneIndexer, list[str]]:
        """Build a dataset from a Replogle ``.h5ad`` file.

        Returns
        -------
        dataset
            The :class:`ReplogleSequenceDataset` instance.
        gene_embedding_table
            ``(n_pert_rows + 1, embedding_dim)`` table aligned to the
            indexer.
        indexer
            :class:`GeneIndexer` mapping perturbation labels to action indices.
        gene_symbols
            Final list of gene symbols after HVG filtering, in column
            order of ``dataset.expression``.
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

        gene_table, indexer = load_gene_embedding_table(
            gene_embedding_path,
            symbols=unique_perturbed,
        )

        dataset = cls(
            expression=x,
            perturbation_labels=labels,
            indexer=indexer,
            sequence_length=sequence_length,
            stack_size=stack_size,
            n_pert=n_pert,
            control_label=control_label,
            rng=rng,
        )
        return dataset, gene_table, indexer, gene_symbols


__all__ = ["ReplogleSequenceDataset"]
