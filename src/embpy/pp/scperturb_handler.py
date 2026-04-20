"""scPerturb dataset handler for harmonized single-cell perturbation data.

Downloads ready-to-use :class:`~anndata.AnnData` files from the
`scPerturb <http://projects.sanderlab.org/scperturb/>`_ Zenodo repository.

Quick start::

    import embpy

    embpy.pp.list_scperturb_datasets()
    card = embpy.pp.scperturb_info("NormanWeissman2019")
    adata = embpy.pp.load_scperturb("NormanWeissman2019")

References
----------
Peidli et al., *scPerturb: Information Resource for Harmonized
Single-Cell Perturbation Data*, bioRxiv 2022.08.20.504663.
https://doi.org/10.1101/2022.08.20.504663
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

_ZENODO_RNASEQ_RECORD = "7041849"
_ZENODO_ATACSEQ_RECORD = "7058382"
_ZENODO_API = "https://zenodo.org/api/records"

_DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "embpy", "scperturb",
)


@dataclass(frozen=True)
class ScPerturbDatasetCard:
    """Metadata for a scPerturb dataset."""

    name: str
    filename: str
    description: str
    modality: Literal["rna", "atac"] = "rna"
    perturbation_column: str = "perturbation"
    perturbation_type: str = "genetic"
    organism: str = "human"
    cell_line: str = ""
    reference: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Registry -- curated metadata for the main scPerturb datasets
# ------------------------------------------------------------------

_SCPERTURB_REGISTRY: dict[str, ScPerturbDatasetCard] = {
    "AdamsonWeissman2016": ScPerturbDatasetCard(
        name="AdamsonWeissman2016",
        filename="AdamsonWeissman2016_GSM2396858_rna.h5ad",
        description=(
            "CRISPRi Perturb-seq targeting UPR and erythroid "
            "differentiation genes in K562 cells (Adamson et al., 2016)."
        ),
        perturbation_type="genetic",
        cell_line="K562",
        reference="https://doi.org/10.1016/j.cell.2016.11.048",
    ),
    "DatlingerBock2017": ScPerturbDatasetCard(
        name="DatlingerBock2017",
        filename="DatlingerBock2017_GSE92872_rna.h5ad",
        description=(
            "CROP-seq targeting immune signalling regulators "
            "in Jurkat T cells (Datlinger et al., 2017)."
        ),
        perturbation_type="genetic",
        cell_line="Jurkat",
        reference="https://doi.org/10.1038/nmeth.4177",
    ),
    "DixitRegev2016": ScPerturbDatasetCard(
        name="DixitRegev2016",
        filename="DixitRegev2016_GSE90063_rna.h5ad",
        description=(
            "Perturb-seq targeting transcription factors in K562 and "
            "bone marrow dendritic cells (Dixit et al., 2016)."
        ),
        perturbation_type="genetic",
        cell_line="K562/BMDCs",
        reference="https://doi.org/10.1016/j.cell.2016.11.038",
    ),
    "FrangiehIzar2021": ScPerturbDatasetCard(
        name="FrangiehIzar2021",
        filename="FrangiehIzar2021_GSE168620_rna.h5ad",
        description=(
            "Perturb-CITE-seq of melanoma and T-cell co-cultures "
            "with CRISPR perturbations (Frangieh et al., 2021)."
        ),
        perturbation_type="genetic",
        cell_line="melanoma/T-cell",
        reference="https://doi.org/10.1038/s41588-021-00779-1",
    ),
    "GasperiniShendure2019": ScPerturbDatasetCard(
        name="GasperiniShendure2019",
        filename="GasperiniShendure2019_at_scale_rna.h5ad",
        description=(
            "CRISPRi at scale targeting enhancers in K562 cells "
            "(Gasperini et al., 2019)."
        ),
        perturbation_type="genetic",
        cell_line="K562",
        reference="https://doi.org/10.1016/j.cell.2018.11.029",
    ),
    "GehringPachter2019": ScPerturbDatasetCard(
        name="GehringPachter2019",
        filename="GehringPachter2019_GSE135497_rna.h5ad",
        description=(
            "Highly multiplexed single-cell RNA-seq with drug "
            "perturbations (Gehring et al., 2019)."
        ),
        perturbation_type="chemical",
        cell_line="MCF10A",
        reference="https://doi.org/10.1038/s41587-019-0392-8",
    ),
    "NormanWeissman2019": ScPerturbDatasetCard(
        name="NormanWeissman2019",
        filename="NormanWeissman2019_filtered_rna.h5ad",
        description=(
            "CRISPRa combinatorial Perturb-seq in K562 cells, "
            "exploring gene combinations (Norman et al., 2019)."
        ),
        perturbation_type="genetic",
        cell_line="K562",
        reference="https://doi.org/10.1126/science.aax4438",
    ),
    "PapaleandrouSchraivogel2019": ScPerturbDatasetCard(
        name="PapaleandrouSchraivogel2019",
        filename="PapaleandrouSchraivogel2019_GSE135497_rna.h5ad",
        description=(
            "TAP-seq for targeted Perturb-seq with increased sensitivity "
            "(Schraivogel et al., 2020)."
        ),
        perturbation_type="genetic",
        cell_line="K562",
        reference="https://doi.org/10.1038/s41592-020-0837-5",
    ),
    "ReplogleWeissman2022_K562": ScPerturbDatasetCard(
        name="ReplogleWeissman2022_K562",
        filename="ReplogleWeissman2022_K562_gwps_rna.h5ad",
        description=(
            "Genome-scale CRISPRi Perturb-seq in K562 cells "
            "(Replogle et al., 2022)."
        ),
        perturbation_type="genetic",
        cell_line="K562",
        reference="https://doi.org/10.1016/j.cell.2022.05.013",
    ),
    "ReplogleWeissman2022_RPE1": ScPerturbDatasetCard(
        name="ReplogleWeissman2022_RPE1",
        filename="ReplogleWeissman2022_RPE1_rna.h5ad",
        description=(
            "Genome-scale CRISPRi Perturb-seq in RPE1 cells "
            "(Replogle et al., 2022)."
        ),
        perturbation_type="genetic",
        cell_line="RPE1",
        reference="https://doi.org/10.1016/j.cell.2022.05.013",
    ),
    "SchraivogelSteinmetz2020": ScPerturbDatasetCard(
        name="SchraivogelSteinmetz2020",
        filename="SchraivogelSteinmetz2020_GSE168620_rna.h5ad",
        description=(
            "TAP-seq for targeted single-cell perturbation screens "
            "(Schraivogel et al., 2020)."
        ),
        perturbation_type="genetic",
        cell_line="K562/iPSC",
        reference="https://doi.org/10.1038/s41592-020-0837-5",
    ),
    "TianLuo2019": ScPerturbDatasetCard(
        name="TianLuo2019",
        filename="TianLuo2019_GSE133344_rna.h5ad",
        description=(
            "Large-scale CRISPRi screen in K562 cells targeting "
            "essential gene regulatory circuits (Tian et al., 2019)."
        ),
        perturbation_type="genetic",
        cell_line="K562",
        reference="https://doi.org/10.1016/j.neuron.2019.07.014",
    ),
    "UrsuHein2022": ScPerturbDatasetCard(
        name="UrsuHein2022",
        filename="UrsuHein2022_GSE196584_rna.h5ad",
        description=(
            "Massively parallel phenotyping of coding variants via "
            "Perturb-seq (Ursu et al., 2022)."
        ),
        perturbation_type="genetic",
        cell_line="HEK293T",
        reference="https://doi.org/10.1038/s41587-022-01563-y",
    ),
}


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------


def list_scperturb_datasets() -> list[str]:
    """Return the names of all registered scPerturb datasets.

    Returns
    -------
    list[str]
        Sorted list of dataset names.
    """
    return sorted(_SCPERTURB_REGISTRY.keys())


def scperturb_info(dataset: str) -> ScPerturbDatasetCard:
    """Return the :class:`ScPerturbDatasetCard` for a dataset.

    Parameters
    ----------
    dataset
        Name of the dataset (e.g. ``"NormanWeissman2019"``).
    """
    if dataset not in _SCPERTURB_REGISTRY:
        raise ValueError(
            f"Unknown scPerturb dataset {dataset!r}.  "
            f"Available: {list_scperturb_datasets()}"
        )
    return _SCPERTURB_REGISTRY[dataset]


# ------------------------------------------------------------------
# Loader
# ------------------------------------------------------------------


def _zenodo_download_url(record_id: str, filename: str) -> str:
    """Build a direct download URL for a Zenodo record file."""
    return f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"


def load_scperturb(
    dataset: str,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    force_download: bool = False,
) -> Any:
    """Download a scPerturb dataset and return it as AnnData.

    Files are cached locally so subsequent calls load from disk.

    Parameters
    ----------
    dataset
        Friendly name (e.g. ``"NormanWeissman2019"``).
        See :func:`list_scperturb_datasets`.
    cache_dir
        Directory for caching downloaded files.
        Defaults to ``~/.cache/embpy/scperturb/``.
    force_download
        Re-download even if the file is already cached.

    Returns
    -------
    :class:`~anndata.AnnData`
        The loaded dataset with ``adata.uns["scperturb_card"]``
        containing dataset metadata.
    """
    import anndata as ad

    card = scperturb_info(dataset)
    cache = Path(cache_dir or _DEFAULT_CACHE_DIR)
    cache.mkdir(parents=True, exist_ok=True)
    local_path = cache / card.filename

    if local_path.exists() and not force_download:
        logger.info("Loading cached scPerturb dataset from %s", local_path)
    else:
        record = (
            _ZENODO_RNASEQ_RECORD
            if card.modality == "rna"
            else _ZENODO_ATACSEQ_RECORD
        )
        url = _zenodo_download_url(record, card.filename)
        logger.info(
            "Downloading scPerturb dataset '%s' from Zenodo ...", card.name,
        )
        _download_file(url, local_path)

    adata = ad.read_h5ad(local_path)
    adata.uns["scperturb_card"] = {
        "name": card.name,
        "description": card.description,
        "modality": card.modality,
        "perturbation_column": card.perturbation_column,
        "perturbation_type": card.perturbation_type,
        "organism": card.organism,
        "cell_line": card.cell_line,
        "reference": card.reference,
    }
    return adata


def _download_file(url: str, dest: Path) -> None:
    """Stream-download a file with a progress indicator."""
    import urllib.request

    logger.info("Downloading %s -> %s", url, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, str(tmp))
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
