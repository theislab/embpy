"""Benchmark datasets for perturbation prediction.

Each public function downloads a ready-to-use :class:`~anndata.AnnData` from
a Hugging Face dataset repository via :class:`~embpy.pp.HFHandler`.

Quick start::

    import embpy

    adata = embpy.dt.replogle()          # download a specific dataset
    embpy.dt.list_datasets()             # see everything available
    embpy.dt.info("replogle")            # metadata for one dataset

Adding a new dataset is as simple as inserting an entry in
``_DATASET_REGISTRY`` — the loader function is generated automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from anndata import AnnData

logger = logging.getLogger(__name__)

_DEFAULT_REPO = "embpy/perturbation-benchmarks"


@dataclass(frozen=True)
class DatasetCard:
    """Metadata for a registered benchmark dataset."""

    name: str
    description: str
    repo: str = _DEFAULT_REPO
    perturbation_column: str = "perturbation"
    perturbation_type: str = "genetic"
    use_case: str = "protein"
    organism: str = "human"
    reference: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Registry – add new datasets here
# ------------------------------------------------------------------

_DATASET_REGISTRY: dict[str, DatasetCard] = {
    "replogle": DatasetCard(
        name="replogle",
        description=(
            "Genome-scale CRISPRi Perturb-seq in K562 cells "
            "(Replogle et al., 2022)."
        ),
        perturbation_column="gene",
        perturbation_type="genetic",
        use_case="protein",
        organism="human",
        reference="https://doi.org/10.1016/j.cell.2022.05.013",
    ),
    "adamson": DatasetCard(
        name="adamson",
        description=(
            "CRISPRi Perturb-seq in K562 cells targeting UPR and "
            "erythroid differentiation genes (Adamson et al., 2016)."
        ),
        perturbation_column="gene",
        perturbation_type="genetic",
        use_case="protein",
        organism="human",
        reference="https://doi.org/10.1016/j.cell.2016.11.048",
    ),
    "norman": DatasetCard(
        name="norman",
        description=(
            "CRISPRa combinatorial Perturb-seq in K562 cells "
            "(Norman et al., 2019)."
        ),
        perturbation_column="gene",
        perturbation_type="genetic",
        use_case="protein",
        organism="human",
        reference="https://doi.org/10.1126/science.aax4438",
    ),
    "sciplex3": DatasetCard(
        name="sciplex3",
        description=(
            "sci-Plex 3 chemical Perturb-seq in A549, K562, and "
            "MCF7 cells (Srivatsan et al., 2020)."
        ),
        perturbation_column="SMILES",
        perturbation_type="chemical",
        use_case="small_molecule",
        organism="human",
        reference="https://doi.org/10.1126/science.aax6234",
    ),
}


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------


def list_datasets() -> list[str]:
    """Return the names of all registered benchmark datasets."""
    return list(_DATASET_REGISTRY.keys())


def info(dataset: str) -> DatasetCard:
    """Return the :class:`DatasetCard` for a registered dataset.

    Parameters
    ----------
    dataset
        Name of the dataset (e.g. ``"replogle"``).
    """
    if dataset not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset {dataset!r}. "
            f"Available: {list_datasets()}"
        )
    return _DATASET_REGISTRY[dataset]


# ------------------------------------------------------------------
# Generic loader
# ------------------------------------------------------------------


def load(
    dataset: str,
    *,
    cache_dir: str | None = None,
    token: str | None = None,
) -> AnnData:
    """Download a benchmark dataset and return it as an AnnData.

    Parameters
    ----------
    dataset
        Name of the dataset (see :func:`list_datasets`).
    cache_dir
        Local cache directory.  Defaults to the Hugging Face Hub cache.
    token
        Optional Hugging Face API token.

    Returns
    -------
    :class:`~anndata.AnnData` ready for use with :func:`embpy.tl.run_pipeline`.
    """
    card = info(dataset)
    from ..pp.hf_handler import HFHandler

    hf = HFHandler(card.repo, token=token)
    logger.info("Downloading dataset '%s' from %s …", card.name, card.repo)
    adata = hf.download_dataset(card.name, cache_dir=cache_dir)

    if not isinstance(adata, AnnData):
        raise TypeError(
            f"Expected AnnData but got {type(adata).__name__} for "
            f"dataset {card.name!r}."
        )

    adata.uns["dataset_card"] = {
        "name": card.name,
        "description": card.description,
        "perturbation_column": card.perturbation_column,
        "perturbation_type": card.perturbation_type,
        "use_case": card.use_case,
        "organism": card.organism,
        "reference": card.reference,
    }
    return adata


# ------------------------------------------------------------------
# Per-dataset convenience functions (auto-generated)
# ------------------------------------------------------------------


def _make_loader(name: str):  # noqa: ANN202
    """Create a dataset-specific loader function."""
    card = _DATASET_REGISTRY[name]

    def _loader(
        *,
        cache_dir: str | None = None,
        token: str | None = None,
    ) -> AnnData:
        return load(name, cache_dir=cache_dir, token=token)

    _loader.__name__ = name
    _loader.__qualname__ = name
    _loader.__doc__ = f"""{card.description}

    Downloads from ``{card.repo}`` via :class:`~embpy.pp.HFHandler`.

    Parameters
    ----------
    cache_dir
        Local cache directory.  Defaults to the Hugging Face Hub cache.
    token
        Optional Hugging Face API token.

    Returns
    -------
    :class:`~anndata.AnnData` with ``adata.uns["dataset_card"]``
    containing the dataset metadata.

    See Also
    --------
    embpy.dt.load : Generic loader accepting any dataset name.
    """
    return _loader


def _register_loaders() -> list[str]:
    """Populate module namespace with per-dataset loader functions."""
    names: list[str] = []
    for name in _DATASET_REGISTRY:
        globals()[name] = _make_loader(name)
        names.append(name)
    return names


_registered = _register_loaders()

__all__ = [
    "DatasetCard",
    "info",
    "list_datasets",
    "load",
    *_registered,
]
