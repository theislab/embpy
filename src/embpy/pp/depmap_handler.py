"""DepMap dataset handler for cancer dependency data.

Provides loaders that read DepMap 24Q4 (or compatible) CSV files from a
local directory and return tidy :class:`~anndata.AnnData` objects with
cell-line metadata joined from ``Model.csv``.

Quick start::

    import embpy

    adata = embpy.pp.load_depmap("CRISPRGeneEffect", data_dir="path/to/depmap_data")
    embpy.pp.list_depmap_datasets()

The DepMap data must be pre-downloaded (see the download script or
`Figshare <https://figshare.com/articles/dataset/DepMap_24Q4_Public/27993248>`_).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from anndata import AnnData

logger = logging.getLogger(__name__)

_GENE_COL_PATTERN = re.compile(r"^(.+?)\s*\((\d+)\)$")


@dataclass(frozen=True)
class DepMapDatasetCard:
    """Metadata for a registered DepMap dataset file."""

    name: str
    filename: str
    description: str
    data_type: Literal["crispr", "expression", "copy_number", "mutation", "metadata", "other"]
    index_col: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

_DEPMAP_REGISTRY: dict[str, DepMapDatasetCard] = {
    "CRISPRGeneEffect": DepMapDatasetCard(
        name="CRISPRGeneEffect",
        filename="CRISPRGeneEffect.csv",
        description=(
            "CERES gene-effect scores from genome-wide CRISPR-Cas9 knockout "
            "screens. Negative values indicate dependency (essential genes)."
        ),
        data_type="crispr",
    ),
    "CRISPRGeneDependency": DepMapDatasetCard(
        name="CRISPRGeneDependency",
        filename="CRISPRGeneDependency.csv",
        description=(
            "Probability that a gene is dependent in each cell line, "
            "derived from CRISPRGeneEffect via a likelihood model."
        ),
        data_type="crispr",
    ),
    "CRISPRGeneEffectUncorrected": DepMapDatasetCard(
        name="CRISPRGeneEffectUncorrected",
        filename="CRISPRGeneEffectUncorrected.csv",
        description="CERES gene-effect scores before copy-number correction.",
        data_type="crispr",
    ),
    "OmicsExpressionProteinCodingGenesTPMLogp1": DepMapDatasetCard(
        name="OmicsExpressionProteinCodingGenesTPMLogp1",
        filename="OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        description="log2(TPM+1) expression for protein-coding genes from RNA-seq.",
        data_type="expression",
    ),
    "OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected": DepMapDatasetCard(
        name="OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected",
        filename="OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv",
        description="Batch-corrected log2(TPM+1) expression for protein-coding genes.",
        data_type="expression",
    ),
    "OmicsCNGene": DepMapDatasetCard(
        name="OmicsCNGene",
        filename="OmicsCNGene.csv",
        description="Gene-level relative copy number from WGS/WES.",
        data_type="copy_number",
    ),
    "OmicsAbsoluteCNGene": DepMapDatasetCard(
        name="OmicsAbsoluteCNGene",
        filename="OmicsAbsoluteCNGene.csv",
        description="Gene-level absolute copy number.",
        data_type="copy_number",
    ),
    "OmicsSomaticMutationsMatrixDamaging": DepMapDatasetCard(
        name="OmicsSomaticMutationsMatrixDamaging",
        filename="OmicsSomaticMutationsMatrixDamaging.csv",
        description="Binary matrix of damaging somatic mutations per gene.",
        data_type="mutation",
    ),
    "OmicsSomaticMutationsMatrixHotspot": DepMapDatasetCard(
        name="OmicsSomaticMutationsMatrixHotspot",
        filename="OmicsSomaticMutationsMatrixHotspot.csv",
        description="Binary matrix of hotspot somatic mutations per gene.",
        data_type="mutation",
    ),
    "ScreenGeneEffect": DepMapDatasetCard(
        name="ScreenGeneEffect",
        filename="ScreenGeneEffect.csv",
        description="Per-screen gene-effect scores (not collapsed across screens).",
        data_type="crispr",
    ),
    "ScreenGeneDependency": DepMapDatasetCard(
        name="ScreenGeneDependency",
        filename="ScreenGeneDependency.csv",
        description="Per-screen gene dependency probabilities.",
        data_type="crispr",
    ),
}


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _parse_gene_columns(columns: pd.Index) -> pd.DataFrame:
    """Parse DepMap-style gene columns ``"SYMBOL (ENTREZ)"`` into a DataFrame.

    Returns a DataFrame with columns ``gene_symbol``, ``entrez_id``, and
    ``raw_column``, indexed by the raw column string.
    """
    records: list[dict[str, Any]] = []
    for col in columns:
        col_str = str(col)
        m = _GENE_COL_PATTERN.match(col_str)
        if m:
            records.append(
                {"raw_column": col_str, "gene_symbol": m.group(1), "entrez_id": int(m.group(2))}
            )
        else:
            records.append(
                {"raw_column": col_str, "gene_symbol": col_str, "entrez_id": np.nan}
            )
    return pd.DataFrame(records).set_index("raw_column")


def _load_model_metadata(data_dir: str) -> pd.DataFrame:
    """Load ``Model.csv`` and index by ``ModelID``."""
    model_path = os.path.join(data_dir, "Model.csv")
    if not os.path.isfile(model_path):
        logger.warning(
            "Model.csv not found in %s – cell-line metadata will be empty.", data_dir
        )
        return pd.DataFrame()
    df = pd.read_csv(model_path, index_col="ModelID", low_memory=False)
    return df


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def list_depmap_datasets() -> list[str]:
    """Return the names of all registered DepMap datasets."""
    return list(_DEPMAP_REGISTRY.keys())


def depmap_info(dataset: str) -> DepMapDatasetCard:
    """Return the :class:`DepMapDatasetCard` for a registered dataset.

    Parameters
    ----------
    dataset
        Name of the dataset (e.g. ``"CRISPRGeneEffect"``).
    """
    if dataset not in _DEPMAP_REGISTRY:
        raise ValueError(
            f"Unknown DepMap dataset {dataset!r}. "
            f"Available: {list_depmap_datasets()}"
        )
    return _DEPMAP_REGISTRY[dataset]


def load_depmap(
    dataset: str,
    data_dir: str,
    *,
    join_metadata: bool = True,
) -> AnnData:
    """Load a DepMap dataset as an :class:`~anndata.AnnData`.

    The returned object has:

    * ``.X`` — the data matrix (cell lines × genes)
    * ``.obs`` — cell-line metadata from ``Model.csv`` (if available)
    * ``.var`` — gene annotations (``gene_symbol``, ``entrez_id``)
    * ``.uns["depmap_card"]`` — dataset metadata

    Parameters
    ----------
    dataset
        Name of the dataset (see :func:`list_depmap_datasets`).
    data_dir
        Path to the directory containing the downloaded DepMap CSV files.
    join_metadata
        If ``True`` (default), join cell-line metadata from ``Model.csv``
        into ``.obs``.

    Returns
    -------
    :class:`~anndata.AnnData`
    """
    card = depmap_info(dataset)
    csv_path = os.path.join(data_dir, card.filename)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"DepMap file not found: {csv_path}. "
            f"Download it from https://depmap.org/portal/download/ "
            f"or run the download_depmap.py script."
        )

    logger.info("Loading DepMap '%s' from %s …", card.name, csv_path)
    df = pd.read_csv(csv_path, index_col=card.index_col, low_memory=False)
    df.index = df.index.astype(str)
    df.index.name = "ModelID"

    var_df = _parse_gene_columns(df.columns)

    X = df.values.astype(np.float32)
    obs_df = pd.DataFrame(index=pd.Index(df.index, name="ModelID"))

    if join_metadata:
        model_df = _load_model_metadata(data_dir)
        if not model_df.empty:
            shared = obs_df.index.intersection(model_df.index)
            obs_df = model_df.loc[shared].copy()
            X = X[np.isin(df.index, shared)]
            var_df_aligned = var_df
            logger.info(
                "Joined metadata for %d / %d cell lines.", len(shared), len(df)
            )
        else:
            var_df_aligned = var_df
    else:
        var_df_aligned = var_df

    adata = AnnData(
        X=X,
        obs=obs_df,
        var=var_df_aligned,
    )

    adata.uns["depmap_card"] = {
        "name": card.name,
        "filename": card.filename,
        "description": card.description,
        "data_type": card.data_type,
        "release": "DepMap 24Q4 Public",
    }

    logger.info(
        "Loaded %s: %d cell lines × %d genes",
        card.name,
        adata.n_obs,
        adata.n_vars,
    )
    return adata
