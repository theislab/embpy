"""Perturbation metadata annotation via pertpy.

Wraps :mod:`pertpy.metadata` to annotate AnnData objects with prior
knowledge about genetic and small-molecule perturbations:

* **Drug / small-molecule metadata** – Mechanism of Action (MOA),
  drug targets, compound information (ChEMBL, DGIdb, PharmGKB).
* **Gene annotation** – Ensembl / HGNC / GO annotation.
* **Cell-line metadata** – DepMap, Cancerrxgene, GDSC drug sensitivity.
* **Drug response** – GDSC IC50 values.

All functions are thin wrappers that:
1. Validate the input AnnData.
2. Delegate to the corresponding :mod:`pertpy.metadata` class.
3. Store results back into ``adata.obs``, ``adata.var``, or ``adata.obsm``.

Requires the optional ``pertpy`` dependency::

    pip install pertpy
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd
from anndata import AnnData

logger = logging.getLogger(__name__)


def _import_pertpy():
    """Lazily import pertpy and return the module.

    Raises
    ------
    ImportError
        If pertpy is not installed.
    """
    try:
        import pertpy as _pt  # type: ignore[import-not-found]

        return _pt
    except ImportError:
        raise ImportError(
            "pertpy is required for metadata annotation. "
            "Install it with: pip install pertpy"
        ) from None


# =====================================================================
# Drug / Small-Molecule Metadata
# =====================================================================


def annotate_drugs(
    adata: AnnData,
    source: Literal["chembl", "dgidb", "pharmgkb"] = "chembl",
    copy: bool = False,
) -> AnnData:
    """Annotate genes by their involvement in drug targets.

    Adds a ``drug`` column to ``adata.var`` indicating which genes are
    known drug targets according to the chosen database.

    Parameters
    ----------
    adata
        AnnData object.  Gene names in ``adata.var_names`` must be in
        HGNC symbol format.
    source
        Drug–target database to query: ``"chembl"``, ``"dgidb"``, or
        ``"pharmgkb"``.
    copy
        If ``True``, return a modified copy instead of annotating in-place.

    Returns
    -------
    AnnData with ``adata.var["drug"]`` populated.
    """
    pt = _import_pertpy()
    logger.info("Annotating drug targets from '%s'…", source)
    drug = pt.md.Drug()
    return drug.annotate(adata, source=source, copy=copy)


def lookup_moa(
    query_drugs: Sequence[str] | None = None,
    target_list: Sequence[str] | None = None,
) -> None:
    """Print a summary of available Mechanism of Action (MOA) annotations.

    Parameters
    ----------
    query_drugs
        Optional drug names to check how many are present in the MOA
        metadata.
    target_list
        Optional molecular targets to check against MOA entries.
    """
    pt = _import_pertpy()
    drug = pt.md.Drug()
    lu = drug.lookup()
    lu.available_moa(query_id_list=query_drugs, target_list=target_list)


def lookup_drug_annotation(
    source: Literal["chembl", "dgidb", "pharmgkb"] = "chembl",
    query_ids: Sequence[str] | None = None,
    query_id_type: Literal["target", "compound", "disease"] = "target",
) -> None:
    """Print a summary of available drug annotation metadata.

    Parameters
    ----------
    source
        Database to query.
    query_ids
        Identifier list to check overlap.
    query_id_type
        Type of identifiers: ``"target"``, ``"compound"``, or
        ``"disease"`` (pharmgkb only).
    """
    pt = _import_pertpy()
    drug = pt.md.Drug()
    lu = drug.lookup()
    lu.available_drug_annotation(
        drug_annotation_source=source,
        query_id_list=query_ids,
        query_id_type=query_id_type,
    )


def lookup_compounds(
    query_compounds: Sequence[str] | None = None,
    query_id_type: Literal["name", "cid"] = "name",
) -> None:
    """Print a summary of available compound metadata.

    Parameters
    ----------
    query_compounds
        Compound names or CIDs to check.
    query_id_type
        ``"name"`` or ``"cid"``.
    """
    pt = _import_pertpy()
    drug = pt.md.Drug()
    lu = drug.lookup()
    lu.available_compounds(
        query_id_list=query_compounds, query_id_type=query_id_type,
    )


# =====================================================================
# Gene Annotation
# =====================================================================


def annotate_genes(
    adata: AnnData,
    reference_id: Literal[
        "gene_id", "ensembl_gene_id", "hgnc_id", "hgnc_symbol"
    ] = "ensembl_gene_id",
    query_ids: Sequence[str] | None = None,
) -> None:
    """Print a summary of gene annotation metadata and check overlap.

    Uses the pertpy ``LookUp`` to show what gene annotations are
    available (GO terms, Ensembl IDs, HGNC symbols, etc.) and how many
    of the provided identifiers are found.

    Parameters
    ----------
    adata
        AnnData whose ``var_names`` or a ``var`` column contains gene IDs.
    reference_id
        Identifier type used in the metadata.
    query_ids
        Gene identifiers to check.  Defaults to ``adata.var_names``.
    """
    pt = _import_pertpy()
    if query_ids is None:
        query_ids = list(adata.var_names)
    cl = pt.md.CellLine()
    lu = cl.lookup()
    lu.available_genes_annotation(
        reference_id=reference_id, query_id_list=query_ids,
    )


# =====================================================================
# Cell-Line Metadata
# =====================================================================


def annotate_cell_lines(
    adata: AnnData,
    query_id: str = "DepMap_ID",
    reference_id: str = "ModelID",
    cell_line_source: Literal["DepMap", "Cancerrxgene"] = "DepMap",
    fetch: Sequence[str] | None = None,
) -> AnnData:
    """Annotate observations with cell-line metadata from DepMap / GDSC.

    Adds metadata columns (e.g. cell line name, disease, age) to
    ``adata.obs`` by matching on a cell-line identifier column.

    Parameters
    ----------
    adata
        AnnData with a cell-line identifier column in ``.obs``.
    query_id
        Column name in ``adata.obs`` containing cell-line identifiers.
    reference_id
        Matching identifier type in the metadata (e.g. ``"ModelID"``).
    cell_line_source
        ``"DepMap"`` or ``"Cancerrxgene"``.
    fetch
        Specific metadata columns to add.  If ``None``, all available
        columns are added.

    Returns
    -------
    AnnData with additional columns in ``.obs``.
    """
    pt = _import_pertpy()
    logger.info("Annotating cell-line metadata from '%s'…", cell_line_source)
    cl = pt.md.CellLine()
    kwargs: dict[str, Any] = {
        "query_id": query_id,
        "reference_id": reference_id,
    }
    if fetch is not None:
        kwargs["fetch"] = list(fetch)
    return cl.annotate(adata, **kwargs)


def lookup_cell_lines(
    cell_line_source: Literal["DepMap", "Cancerrxgene"] = "DepMap",
    reference_id: str = "ModelID",
    query_ids: Sequence[str] | None = None,
) -> None:
    """Print a summary of available cell-line metadata and check overlap.

    Parameters
    ----------
    cell_line_source
        Source database.
    reference_id
        Identifier type in the metadata.
    query_ids
        Cell-line identifiers to check.
    """
    pt = _import_pertpy()
    cl = pt.md.CellLine()
    lu = cl.lookup()
    lu.available_cell_lines(
        cell_line_source=cell_line_source,
        reference_id=reference_id,
        query_id_list=query_ids,
    )


# =====================================================================
# Bulk RNA Expression
# =====================================================================


def annotate_bulk_rna(
    adata: AnnData,
    cell_line_source: Literal["broad", "sanger"] = "broad",
    query_id: str = "DepMap_ID",
) -> AnnData:
    """Annotate with bulk RNA-seq expression from DepMap (Broad/Sanger).

    The expression matrix is stored in
    ``adata.obsm["bulk_rna_{source}"]``.

    Parameters
    ----------
    adata
        AnnData with a cell-line identifier column in ``.obs``.
    cell_line_source
        ``"broad"`` or ``"sanger"``.
    query_id
        Column in ``adata.obs`` with the cell-line identifier.

    Returns
    -------
    AnnData with bulk RNA expression in ``.obsm``.
    """
    pt = _import_pertpy()
    logger.info("Annotating bulk RNA from '%s'…", cell_line_source)
    cl = pt.md.CellLine()
    return cl.annotate_bulk_rna(
        adata, cell_line_source=cell_line_source, query_id=query_id,
    )


# =====================================================================
# Drug Response (GDSC)
# =====================================================================


def annotate_drug_response(
    adata: AnnData,
    gdsc_dataset: Literal[1, 2] = 1,
    query_id: str = "SangerModelID",
    reference_id: str = "sanger_model_id",
) -> AnnData:
    """Annotate observations with GDSC drug sensitivity (IC50).

    Adds an ``ln_ic50`` column to ``adata.obs`` for each
    cell-line × drug combination.

    Parameters
    ----------
    adata
        AnnData with cell-line identifiers and perturbation labels.
    gdsc_dataset
        GDSC version (1 or 2).
    query_id
        Column in ``adata.obs`` with the cell-line identifier.
    reference_id
        Matching identifier type in GDSC.

    Returns
    -------
    AnnData with ``ln_ic50`` in ``.obs``.
    """
    pt = _import_pertpy()
    logger.info("Annotating GDSC%d drug response…", gdsc_dataset)
    cl = pt.md.CellLine()
    return cl.annotate_from_gdsc(
        adata,
        gdsc_dataset=gdsc_dataset,
        query_id=query_id,
        reference_id=reference_id,
    )


def lookup_drug_response(
    gdsc_dataset: Literal[1, 2] = 1,
    reference_id: Literal[
        "cell_line_name", "sanger_model_id", "cosmic_id"
    ] = "cell_line_name",
    query_ids: Sequence[str] | None = None,
    reference_perturbation: Literal["drug_name", "drug_id"] = "drug_name",
    query_perturbation_list: Sequence[str] | None = None,
) -> None:
    """Print a summary of available GDSC drug response data.

    Parameters
    ----------
    gdsc_dataset
        GDSC version.
    reference_id
        Cell-line identifier type in GDSC.
    query_ids
        Cell-line identifiers to check.
    reference_perturbation
        Perturbation identifier type in GDSC.
    query_perturbation_list
        Perturbation names to check.
    """
    pt = _import_pertpy()
    cl = pt.md.CellLine()
    lu = cl.lookup()
    lu.available_drug_response(
        gdsc_dataset=gdsc_dataset,
        reference_id=reference_id,
        query_id_list=query_ids,
        reference_perturbation=reference_perturbation,
        query_perturbation_list=query_perturbation_list,
    )


# =====================================================================
# Protein Expression
# =====================================================================


def lookup_protein_expression(
    reference_id: Literal["model_name", "model_id"] = "model_name",
    query_ids: Sequence[str] | None = None,
) -> None:
    """Print a summary of available protein expression data.

    Parameters
    ----------
    reference_id
        Identifier type in the metadata.
    query_ids
        Cell-line identifiers to check.
    """
    pt = _import_pertpy()
    cl = pt.md.CellLine()
    lu = cl.lookup()
    lu.available_protein_expression(
        reference_id=reference_id, query_id_list=query_ids,
    )


# =====================================================================
# Convenience: annotate perturbation metadata (auto-detect)
# =====================================================================


def annotate_perturbation(
    adata: AnnData,
    perturbation_col: str = "perturbation",
    perturbation_type: Literal["genetic", "drug", "auto"] = "auto",
    drug_source: Literal["chembl", "dgidb", "pharmgkb"] = "chembl",
    copy: bool = False,
) -> AnnData:
    """One-call metadata annotation for perturbation datasets.

    Inspects the perturbation column to decide whether the dataset
    contains **genetic** or **drug** perturbations and dispatches to
    the appropriate pertpy annotator.

    For **drug** perturbations, ``adata.var`` gets a ``drug`` column
    from :func:`annotate_drugs`.

    For **genetic** perturbations, gene annotations are looked up and
    a summary is printed.

    Parameters
    ----------
    adata
        AnnData with a perturbation column in ``.obs``.
    perturbation_col
        Column in ``adata.obs`` with perturbation identifiers.
    perturbation_type
        ``"genetic"``, ``"drug"``, or ``"auto"`` (infer from column
        values).
    drug_source
        Database for drug annotation (used when type is drug).
    copy
        If ``True``, return a copy.

    Returns
    -------
    Annotated AnnData.
    """
    _import_pertpy()

    if perturbation_col not in adata.obs.columns:
        raise KeyError(
            f"Column '{perturbation_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    if perturbation_type == "auto":
        col_series: pd.Series = adata.obs[perturbation_col]  # type: ignore[assignment]
        perturbation_type = _infer_perturbation_type(col_series)
        logger.info("Auto-detected perturbation type: %s", perturbation_type)

    if copy:
        adata = adata.copy()

    if perturbation_type == "drug":
        adata = annotate_drugs(adata, source=drug_source, copy=False)
        logger.info("Drug annotation complete (adata.var['drug']).")
    else:
        annotate_genes(adata)
        logger.info("Gene annotation lookup complete.")

    return adata


def _infer_perturbation_type(
    series: pd.Series,
) -> Literal["genetic", "drug"]:
    """Heuristic to decide if perturbations are genetic or drug-based.

    * If most values look like gene symbols (uppercase, 2-10 chars,
      alphanumeric) → ``"genetic"``.
    * Otherwise → ``"drug"``.
    """
    import re

    gene_pattern = re.compile(r"^[A-Z][A-Z0-9]{1,15}$")
    unique_vals = series.dropna().unique()
    if len(unique_vals) == 0:
        return "drug"
    n_gene_like = sum(1 for v in unique_vals if gene_pattern.match(str(v)))
    fraction = n_gene_like / len(unique_vals)
    return "genetic" if fraction > 0.5 else "drug"
