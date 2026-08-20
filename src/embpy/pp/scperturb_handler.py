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
        filename="AdamsonWeissman2016_GSM2406675_10X001.h5ad",
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
        filename="DatlingerBock2017.h5ad",
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
        filename="DixitRegev2016.h5ad",
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
        filename="FrangiehIzar2021_RNA.h5ad",
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
        filename="GasperiniShendure2019_atscale.h5ad",
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
        filename="GehringPachter2019.h5ad",
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
        filename="NormanWeissman2019_filtered.h5ad",
        description=(
            "CRISPRa combinatorial Perturb-seq in K562 cells, "
            "exploring gene combinations (Norman et al., 2019)."
        ),
        perturbation_type="genetic",
        cell_line="K562",
        reference="https://doi.org/10.1126/science.aax4438",
    ),
    "ReplogleWeissman2022_K562": ScPerturbDatasetCard(
        name="ReplogleWeissman2022_K562",
        filename="ReplogleWeissman2022_K562_gwps.h5ad",
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
        filename="ReplogleWeissman2022_rpe1.h5ad",
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
        filename="SchraivogelSteinmetz2020_TAP_SCREEN__chromosome_8_screen.h5ad",
        description=(
            "TAP-seq for targeted single-cell perturbation screens "
            "(Schraivogel et al., 2020)."
        ),
        perturbation_type="genetic",
        cell_line="K562/iPSC",
        reference="https://doi.org/10.1038/s41592-020-0837-5",
    ),
    "TianKampmann2019": ScPerturbDatasetCard(
        name="TianKampmann2019",
        filename="TianKampmann2019_iPSC.h5ad",
        description=(
            "CRISPRi screen in iPSC-derived neurons targeting genes "
            "essential for neuronal survival (Tian et al., 2019)."
        ),
        perturbation_type="genetic",
        cell_line="iPSC-neuron",
        reference="https://doi.org/10.1016/j.neuron.2019.07.014",
    ),
    "ReplogleWeissman2022_K562_essential": ScPerturbDatasetCard(
        name="ReplogleWeissman2022_K562_essential",
        filename="ReplogleWeissman2022_K562_essential.h5ad",
        description=(
            "CRISPRi Perturb-seq over essential genes in K562 cells "
            "(Replogle et al., 2022). Much smaller than the genome-wide "
            "screen, so the better default of the two."
        ),
        perturbation_type="genetic",
        cell_line="K562",
        reference="https://doi.org/10.1016/j.cell.2022.05.013",
    ),
    "ShifrutMarson2018": ScPerturbDatasetCard(
        name="ShifrutMarson2018",
        filename="ShifrutMarson2018.h5ad",
        description=(
            "SLICE CRISPR screen in primary human T cells identifying "
            "regulators of proliferation (Shifrut et al., 2018)."
        ),
        perturbation_type="genetic",
        cell_line="primary T cell",
        reference="https://doi.org/10.1016/j.cell.2018.10.024",
    ),
    "SrivatsanTrapnell2020_sciplex3": ScPerturbDatasetCard(
        name="SrivatsanTrapnell2020_sciplex3",
        filename="SrivatsanTrapnell2020_sciplex3.h5ad",
        description=(
            "sci-Plex nuclear-hashing screen of 188 compounds across three "
            "cancer cell lines (Srivatsan et al., 2020). Chemical rather "
            "than genetic, so it is the natural partner for the ChEMBL "
            "annotation layer."
        ),
        perturbation_type="chemical",
        cell_line="A549/K562/MCF7",
        reference="https://doi.org/10.1126/science.aax6234",
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
        _download_file(url, local_path, record_id=record)

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


def _record_filenames(record_id: str) -> list[str]:
    """List the files a Zenodo record actually holds.

    Only used to turn a 404 into something actionable, so any failure here is
    swallowed -- a diagnostic that itself raises is worse than the bare 404.
    """
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"{_ZENODO_API}/{record_id}", timeout=30) as resp:
            payload = json.load(resp)
        return sorted(f["key"] for f in payload.get("files", []))
    except Exception:  # noqa: BLE001
        return []


def _download_file(url: str, dest: Path, record_id: str | None = None) -> None:
    """Stream-download a file with a progress indicator."""
    import urllib.error
    import urllib.request

    logger.info("Downloading %s -> %s", url, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, str(tmp))
        tmp.rename(dest)
    except urllib.error.HTTPError as exc:
        if tmp.exists():
            tmp.unlink()
        if exc.code == 404 and record_id:
            # Every filename in this registry was once wrong, and the only
            # symptom was `HTTPError: 404` naming a URL -- which looks like
            # Zenodo being down rather than embpy asking for a file that has
            # never existed. Name the near-misses so the difference is
            # obvious.
            available = _record_filenames(record_id)
            wanted = dest.name
            stem = wanted.split("_")[0].split(".")[0].lower()
            close = [f for f in available if f.lower().startswith(stem[:8])]
            raise FileNotFoundError(
                f"Zenodo record {record_id} has no file named {wanted!r}. "
                + (
                    f"Closest matches: {close}. "
                    if close
                    else f"The record holds {len(available)} files. "
                )
                + "This is a stale entry in embpy's scPerturb registry, not a "
                "network problem -- please report it."
            ) from exc
        raise
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
