"""LaminDB dataset handler for perturbation data.

Provides thin wrappers around ``lamindb`` (:mod:`ln`) to load benchmark
datasets by friendly name instead of raw artifact UIDs.

Quick start::

    import embpy

    adata = embpy.pp.load_lamin("mcfarland")
    embpy.pp.list_lamin_datasets()

``lamindb`` is an **optional dependency** — it is imported lazily at call
time and a clear error is raised if it is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from anndata import AnnData

logger = logging.getLogger(__name__)

_DEFAULT_LAMIN_INSTANCE = "theislab/pertmodeling"


def _require_lamindb():  # type: ignore[no-untyped-def]
    """Lazy-import lamindb, raising a helpful error if absent."""
    try:
        import lamindb as ln
    except ModuleNotFoundError as exc:
        raise ImportError("lamindb is required for this function. Install it with:  pip install lamindb") from exc
    return ln


# ------------------------------------------------------------------
# Dataset card
# ------------------------------------------------------------------


@dataclass(frozen=True)
class LaminDatasetCard:
    """Metadata for a LaminDB-backed benchmark dataset."""

    name: str
    uid: str
    description: str
    perturbation_column: str = "perturbation"
    perturbation_type: str = "genetic"
    use_case: Literal["protein", "gene", "small_molecule", "text"] = "protein"
    organism: str = "human"
    reference: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Registry – add new datasets here
# ------------------------------------------------------------------

_LAMIN_REGISTRY: dict[str, LaminDatasetCard] = {
    "mcfarland": LaminDatasetCard(
        name="mcfarland",
        uid="rCNr6NMFJRo0cyzl0000",
        description=(
            "Multiplexed single-cell transcriptional response profiling "
            "to define cancer vulnerabilities and therapeutic mechanism "
            "of action (McFarland et al., 2020)."
        ),
        perturbation_column="gene",
        perturbation_type="genetic",
        use_case="protein",
        organism="human",
        reference="https://doi.org/10.1038/s41467-020-17440-w",
    ),
    "replogle": LaminDatasetCard(
        name="replogle",
        uid="cdTgT79SxGj3UJGy0000",
        description=(
            "Genome-scale CRISPRi Perturb-seq in K562 cells "
            "(Replogle et al., 2022)."
        ),
        perturbation_column="pert_target",
        perturbation_type="genetic",
        use_case="protein",
        organism="human",
        reference="https://doi.org/10.1016/j.cell.2022.05.013",
    ),
}


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------


def list_lamin_datasets() -> list[str]:
    """Return the names of all registered LaminDB datasets."""
    return list(_LAMIN_REGISTRY.keys())


def lamin_info(dataset: str) -> LaminDatasetCard:
    """Return the :class:`LaminDatasetCard` for a registered dataset.

    Parameters
    ----------
    dataset
        Name of the dataset (e.g. ``"mcfarland"``).
    """
    if dataset not in _LAMIN_REGISTRY:
        raise ValueError(f"Unknown LaminDB dataset {dataset!r}. Available: {list_lamin_datasets()}")
    return _LAMIN_REGISTRY[dataset]


# ------------------------------------------------------------------
# Loader
# ------------------------------------------------------------------


def load_lamin(
    dataset: str,
    *,
    instance: str | None = None,
    cache: bool = False,
) -> AnnData:
    """Load a benchmark dataset from LaminDB by name.

    Under the hood this calls ``ln.Artifact.get(uid).load()`` using the
    UID stored in the registry.

    Parameters
    ----------
    dataset
        Friendly name of the dataset (see :func:`list_lamin_datasets`).
    instance
        LaminDB instance slug (e.g. ``"theislab/pertmodeling"``).
        Defaults to :data:`_DEFAULT_LAMIN_INSTANCE`.  Calls
        ``ln.connect(instance)`` automatically if not already connected.
    cache
        If ``True``, call ``artifact.cache()`` instead of
        ``artifact.load()`` and read from the cached local path.
        Useful when repeated access is needed without re-downloading.

    Returns
    -------
    :class:`~anndata.AnnData` with ``adata.uns["lamin_card"]``
    containing the dataset metadata.

    Examples
    --------
    >>> adata = embpy.pp.load_lamin("mcfarland")
    >>> adata.uns["lamin_card"]["uid"]
    'rCNr6NMFJRo0cyzl0000'
    """
    ln = _require_lamindb()
    card = lamin_info(dataset)

    instance = instance or _DEFAULT_LAMIN_INSTANCE
    ln.connect(instance)

    logger.info(
        "Loading dataset '%s' (uid=%s) from LaminDB …",
        card.name,
        card.uid,
    )
    artifact = ln.Artifact.get(card.uid)

    if cache:
        import anndata as ad

        local_path = artifact.cache()
        adata = ad.read_h5ad(local_path)
    else:
        adata = artifact.load()

    if not isinstance(adata, AnnData):
        raise TypeError(f"Expected AnnData but got {type(adata).__name__} for dataset {card.name!r} (uid={card.uid}).")

    adata.uns["lamin_card"] = {
        "name": card.name,
        "uid": card.uid,
        "description": card.description,
        "perturbation_column": card.perturbation_column,
        "perturbation_type": card.perturbation_type,
        "use_case": card.use_case,
        "organism": card.organism,
        "reference": card.reference,
    }
    return adata
