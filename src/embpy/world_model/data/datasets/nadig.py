"""Nadig & O'Connor 2024 perturb-seq adapter.

Like the Replogle adapter, but the source files split by cell line
(``Jurkat``, ``HepG2``) so the adapter does not filter further on
``cell_type``. The other ``adata.obs`` conventions are identical.
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


def _load_adata(path):  # type: ignore[no-untyped-def]
    try:
        import anndata as ad  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("anndata is required to load Nadig datasets.") from exc
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Nadig h5ad not found: {path}")
    return ad.read_h5ad(path)


class NadigSequenceDataset(PerturbationSequenceDataset):
    """Nadig & O'Connor 2024 perturb-seq sequence dataset."""

    @classmethod
    def from_h5ad(
        cls,
        h5ad_path: str | Path,
        *,
        provider: "ActionEmbeddingProvider | None" = None,
        gene_embedding_path: str | Path | None = None,
        perturbation_key: str = "perturbation",
        control_label: str = "non-targeting",
        n_top_genes: int = 5000,
        log_normalize: bool = True,
        sequence_length: int = 8,
        stack_size: int = 4,
        n_pert: int = 2,
        rng: np.random.Generator | None = None,
    ) -> tuple["NadigSequenceDataset", np.ndarray, GeneIndexer, list[str]]:
        """Build a dataset from a Nadig ``.h5ad`` file.

        Exactly one of ``provider`` or ``gene_embedding_path`` must be
        set. The path form is accepted for backward compatibility and
        is wrapped in a :class:`PrecomputedProvider` internally.
        """
        adata = _load_adata(h5ad_path)
        if perturbation_key not in adata.obs.columns:
            raise KeyError(
                f"'{perturbation_key}' not in adata.obs (got {list(adata.obs.columns)})"
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
        )
        return dataset, gene_table, indexer, gene_symbols


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
    from ..embeddings.precomputed import PrecomputedProvider  # noqa: PLC0415

    logger.warning(
        "NadigSequenceDataset.from_h5ad: gene_embedding_path is "
        "deprecated -- pass an ActionEmbeddingProvider instead.",
    )
    return PrecomputedProvider(gene_embedding_path)


__all__ = ["NadigSequenceDataset"]
