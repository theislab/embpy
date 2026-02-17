"""Tests for embpy.tl.metadata – pertpy metadata annotation wrappers.

We load the module directly from file to avoid the heavy ``embpy``
package import chain (transformers, torch, etc.).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

# ---------------------------------------------------------------------------
# Load metadata module directly from file (avoids embpy.__init__ imports)
# ---------------------------------------------------------------------------

_MOD_PATH = Path(__file__).resolve().parent.parent / "src" / "embpy" / "tl" / "metadata.py"
_spec = importlib.util.spec_from_file_location("embpy.tl.metadata", _MOD_PATH)
assert _spec is not None and _spec.loader is not None
metadata = importlib.util.module_from_spec(_spec)
sys.modules["embpy.tl.metadata"] = metadata
_spec.loader.exec_module(metadata)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_pt():
    """Build a mock pertpy module with Drug and CellLine stubs."""
    return MagicMock()


IMPORT_PATH = "embpy.tl.metadata._import_pertpy"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def adata_drug() -> AnnData:
    """AnnData mimicking a drug-perturbation experiment."""
    obs = pd.DataFrame(
        {
            "perturbation": ["Afatinib", "Trametinib", "control", "Navitoclax"],
            "DepMap_ID": ["ACH-000001", "ACH-000002", "ACH-000003", "ACH-000004"],
            "SangerModelID": ["SIDM001", "SIDM002", "SIDM003", "SIDM004"],
        }
    )
    var = pd.DataFrame(index=pd.Index(["EGFR", "BRAF", "TP53", "KRAS"]))
    X = np.random.default_rng(42).random((4, 4)).astype(np.float32)
    return AnnData(X=X, obs=obs, var=var)


@pytest.fixture()
def adata_genetic() -> AnnData:
    """AnnData mimicking a genetic-perturbation experiment."""
    obs = pd.DataFrame(
        {
            "perturbation": ["TP53", "EGFR", "BRAF", "KRAS", "MYC"],
            "DepMap_ID": [f"ACH-{i:06d}" for i in range(5)],
        }
    )
    var = pd.DataFrame(index=pd.Index(["GeneA", "GeneB", "GeneC"]))
    X = np.random.default_rng(0).random((5, 3)).astype(np.float32)
    return AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


class TestPertpyImportGuard:
    """Tests that functions raise ImportError when pertpy is missing."""

    def test_annotate_drugs_requires_pertpy(self, adata_drug):
        with patch(IMPORT_PATH, side_effect=ImportError("pertpy is required")):
            with pytest.raises(ImportError, match="pertpy is required"):
                metadata.annotate_drugs(adata_drug)

    def test_lookup_moa_requires_pertpy(self):
        with patch(IMPORT_PATH, side_effect=ImportError("pertpy is required")):
            with pytest.raises(ImportError, match="pertpy is required"):
                metadata.lookup_moa()

    def test_annotate_cell_lines_requires_pertpy(self, adata_drug):
        with patch(IMPORT_PATH, side_effect=ImportError("pertpy is required")):
            with pytest.raises(ImportError, match="pertpy is required"):
                metadata.annotate_cell_lines(adata_drug)

    def test_annotate_drug_response_requires_pertpy(self, adata_drug):
        with patch(IMPORT_PATH, side_effect=ImportError("pertpy is required")):
            with pytest.raises(ImportError, match="pertpy is required"):
                metadata.annotate_drug_response(adata_drug)

    def test_annotate_perturbation_requires_pertpy(self, adata_drug):
        with patch(IMPORT_PATH, side_effect=ImportError("pertpy is required")):
            with pytest.raises(ImportError, match="pertpy is required"):
                metadata.annotate_perturbation(adata_drug)

    def test_annotate_bulk_rna_requires_pertpy(self, adata_drug):
        with patch(IMPORT_PATH, side_effect=ImportError("pertpy is required")):
            with pytest.raises(ImportError, match="pertpy is required"):
                metadata.annotate_bulk_rna(adata_drug)

    def test_lookup_protein_expression_requires_pertpy(self):
        with patch(IMPORT_PATH, side_effect=ImportError("pertpy is required")):
            with pytest.raises(ImportError, match="pertpy is required"):
                metadata.lookup_protein_expression()


# ---------------------------------------------------------------------------
# Drug annotation
# ---------------------------------------------------------------------------


class TestAnnotateDrugs:
    """Tests for annotate_drugs wrapper."""

    def test_delegates_to_pertpy(self, adata_drug):
        mock_pt = _make_mock_pt()
        mock_pt.md.Drug.return_value.annotate.return_value = adata_drug

        with patch(IMPORT_PATH, return_value=mock_pt):
            result = metadata.annotate_drugs(adata_drug, source="chembl", copy=False)

            mock_pt.md.Drug.assert_called_once()
            mock_pt.md.Drug.return_value.annotate.assert_called_once_with(
                adata_drug,
                source="chembl",
                copy=False,
            )
            assert result is adata_drug

    def test_source_dgidb(self, adata_drug):
        mock_pt = _make_mock_pt()
        mock_pt.md.Drug.return_value.annotate.return_value = adata_drug

        with patch(IMPORT_PATH, return_value=mock_pt):
            metadata.annotate_drugs(adata_drug, source="dgidb")
            mock_pt.md.Drug.return_value.annotate.assert_called_once_with(
                adata_drug,
                source="dgidb",
                copy=False,
            )

    def test_source_pharmgkb(self, adata_drug):
        mock_pt = _make_mock_pt()
        mock_pt.md.Drug.return_value.annotate.return_value = adata_drug

        with patch(IMPORT_PATH, return_value=mock_pt):
            metadata.annotate_drugs(adata_drug, source="pharmgkb")
            mock_pt.md.Drug.return_value.annotate.assert_called_once_with(
                adata_drug,
                source="pharmgkb",
                copy=False,
            )


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


class TestLookups:
    """Tests for the various lookup_* wrappers."""

    def test_lookup_moa(self):
        mock_pt = _make_mock_pt()
        with patch(IMPORT_PATH, return_value=mock_pt):
            metadata.lookup_moa(query_drugs=["Afatinib"], target_list=["EGFR"])
            mock_pt.md.Drug.return_value.lookup.return_value.available_moa.assert_called_once_with(
                query_id_list=["Afatinib"],
                target_list=["EGFR"],
            )

    def test_lookup_drug_annotation(self):
        mock_pt = _make_mock_pt()
        with patch(IMPORT_PATH, return_value=mock_pt):
            metadata.lookup_drug_annotation(source="chembl", query_ids=["EGFR"])
            mock_pt.md.Drug.return_value.lookup.return_value.available_drug_annotation.assert_called_once_with(
                drug_annotation_source="chembl",
                query_id_list=["EGFR"],
                query_id_type="target",
            )

    def test_lookup_compounds(self):
        mock_pt = _make_mock_pt()
        with patch(IMPORT_PATH, return_value=mock_pt):
            metadata.lookup_compounds(query_compounds=["Ibuprofen"])
            mock_pt.md.Drug.return_value.lookup.return_value.available_compounds.assert_called_once_with(
                query_id_list=["Ibuprofen"],
                query_id_type="name",
            )

    def test_lookup_cell_lines(self):
        mock_pt = _make_mock_pt()
        with patch(IMPORT_PATH, return_value=mock_pt):
            metadata.lookup_cell_lines(query_ids=["ACH-000001"])
            mock_pt.md.CellLine.return_value.lookup.return_value.available_cell_lines.assert_called_once_with(
                cell_line_source="DepMap",
                reference_id="ModelID",
                query_id_list=["ACH-000001"],
            )

    def test_lookup_drug_response(self):
        mock_pt = _make_mock_pt()
        with patch(IMPORT_PATH, return_value=mock_pt):
            metadata.lookup_drug_response(gdsc_dataset=2)
            mock_pt.md.CellLine.return_value.lookup.return_value.available_drug_response.assert_called_once_with(
                gdsc_dataset=2,
                reference_id="cell_line_name",
                query_id_list=None,
                reference_perturbation="drug_name",
                query_perturbation_list=None,
            )

    def test_lookup_protein_expression(self):
        mock_pt = _make_mock_pt()
        with patch(IMPORT_PATH, return_value=mock_pt):
            metadata.lookup_protein_expression(query_ids=["A549"])
            mock_pt.md.CellLine.return_value.lookup.return_value.available_protein_expression.assert_called_once_with(
                reference_id="model_name",
                query_id_list=["A549"],
            )


# ---------------------------------------------------------------------------
# Cell-line annotation
# ---------------------------------------------------------------------------


class TestAnnotateCellLines:
    """Tests for annotate_cell_lines wrapper."""

    def test_delegates_to_pertpy(self, adata_drug):
        mock_pt = _make_mock_pt()
        mock_pt.md.CellLine.return_value.annotate.return_value = adata_drug

        with patch(IMPORT_PATH, return_value=mock_pt):
            result = metadata.annotate_cell_lines(
                adata_drug,
                query_id="DepMap_ID",
                reference_id="ModelID",
                fetch=["CellLineName", "Age"],
            )

            mock_pt.md.CellLine.return_value.annotate.assert_called_once_with(
                adata_drug,
                query_id="DepMap_ID",
                reference_id="ModelID",
                fetch=["CellLineName", "Age"],
            )
            assert result is adata_drug

    def test_no_fetch_param(self, adata_drug):
        mock_pt = _make_mock_pt()
        mock_pt.md.CellLine.return_value.annotate.return_value = adata_drug

        with patch(IMPORT_PATH, return_value=mock_pt):
            metadata.annotate_cell_lines(adata_drug)
            mock_pt.md.CellLine.return_value.annotate.assert_called_once_with(
                adata_drug,
                query_id="DepMap_ID",
                reference_id="ModelID",
            )


# ---------------------------------------------------------------------------
# Bulk RNA
# ---------------------------------------------------------------------------


class TestAnnotateBulkRna:
    """Tests for annotate_bulk_rna wrapper."""

    def test_delegates_broad(self, adata_drug):
        mock_pt = _make_mock_pt()
        mock_pt.md.CellLine.return_value.annotate_bulk_rna.return_value = adata_drug

        with patch(IMPORT_PATH, return_value=mock_pt):
            result = metadata.annotate_bulk_rna(adata_drug, cell_line_source="broad")
            mock_pt.md.CellLine.return_value.annotate_bulk_rna.assert_called_once_with(
                adata_drug,
                cell_line_source="broad",
                query_id="DepMap_ID",
            )
            assert result is adata_drug


# ---------------------------------------------------------------------------
# Drug response
# ---------------------------------------------------------------------------


class TestAnnotateDrugResponse:
    """Tests for annotate_drug_response wrapper."""

    def test_delegates_gdsc1(self, adata_drug):
        mock_pt = _make_mock_pt()
        mock_pt.md.CellLine.return_value.annotate_from_gdsc.return_value = adata_drug

        with patch(IMPORT_PATH, return_value=mock_pt):
            result = metadata.annotate_drug_response(adata_drug, gdsc_dataset=1)
            mock_pt.md.CellLine.return_value.annotate_from_gdsc.assert_called_once_with(
                adata_drug,
                gdsc_dataset=1,
                query_id="SangerModelID",
                reference_id="sanger_model_id",
            )
            assert result is adata_drug


# ---------------------------------------------------------------------------
# Gene annotation
# ---------------------------------------------------------------------------


class TestAnnotateGenes:
    """Tests for annotate_genes lookup wrapper."""

    def test_defaults_to_var_names(self, adata_drug):
        mock_pt = _make_mock_pt()
        with patch(IMPORT_PATH, return_value=mock_pt):
            metadata.annotate_genes(adata_drug)
            mock_pt.md.CellLine.return_value.lookup.return_value.available_genes_annotation.assert_called_once_with(
                reference_id="ensembl_gene_id",
                query_id_list=["EGFR", "BRAF", "TP53", "KRAS"],
            )

    def test_custom_query_ids(self, adata_drug):
        mock_pt = _make_mock_pt()
        with patch(IMPORT_PATH, return_value=mock_pt):
            metadata.annotate_genes(adata_drug, query_ids=["ENSG00000141510"])
            mock_pt.md.CellLine.return_value.lookup.return_value.available_genes_annotation.assert_called_once_with(
                reference_id="ensembl_gene_id",
                query_id_list=["ENSG00000141510"],
            )


# ---------------------------------------------------------------------------
# Convenience: annotate_perturbation
# ---------------------------------------------------------------------------


class TestAnnotatePerturbation:
    """Tests for the high-level annotate_perturbation wrapper."""

    def test_auto_detects_drug(self, adata_drug):
        mock_pt = _make_mock_pt()
        mock_pt.md.Drug.return_value.annotate.return_value = adata_drug

        with patch(IMPORT_PATH, return_value=mock_pt):
            result = metadata.annotate_perturbation(adata_drug, perturbation_type="auto")
            mock_pt.md.Drug.return_value.annotate.assert_called_once()
            assert result is adata_drug

    def test_auto_detects_genetic(self, adata_genetic):
        mock_pt = _make_mock_pt()
        with patch(IMPORT_PATH, return_value=mock_pt):
            result = metadata.annotate_perturbation(
                adata_genetic,
                perturbation_type="auto",
            )
            mock_pt.md.CellLine.return_value.lookup.return_value.available_genes_annotation.assert_called_once()
            assert result is adata_genetic

    def test_explicit_drug_type(self, adata_drug):
        mock_pt = _make_mock_pt()
        mock_pt.md.Drug.return_value.annotate.return_value = adata_drug

        with patch(IMPORT_PATH, return_value=mock_pt):
            metadata.annotate_perturbation(adata_drug, perturbation_type="drug")
            mock_pt.md.Drug.return_value.annotate.assert_called_once()

    def test_explicit_genetic_type(self, adata_drug):
        mock_pt = _make_mock_pt()
        with patch(IMPORT_PATH, return_value=mock_pt):
            metadata.annotate_perturbation(adata_drug, perturbation_type="genetic")
            mock_pt.md.CellLine.return_value.lookup.return_value.available_genes_annotation.assert_called_once()

    def test_missing_column_raises(self, adata_drug):
        mock_pt = _make_mock_pt()
        with patch(IMPORT_PATH, return_value=mock_pt):
            with pytest.raises(KeyError, match="no_such_column"):
                metadata.annotate_perturbation(
                    adata_drug,
                    perturbation_col="no_such_column",
                )

    def test_copy_flag(self, adata_drug):
        mock_pt = _make_mock_pt()
        mock_pt.md.Drug.return_value.annotate.side_effect = lambda a, **kw: a

        with patch(IMPORT_PATH, return_value=mock_pt):
            result = metadata.annotate_perturbation(
                adata_drug,
                perturbation_type="drug",
                copy=True,
            )
            assert result is not adata_drug
            assert result.shape == adata_drug.shape


# ---------------------------------------------------------------------------
# Perturbation type inference
# ---------------------------------------------------------------------------


class TestInferPerturbationType:
    """Tests for _infer_perturbation_type heuristic."""

    def test_gene_symbols(self):
        series = pd.Series(["TP53", "EGFR", "BRAF", "KRAS", "MYC"])
        assert metadata._infer_perturbation_type(series) == "genetic"

    def test_drug_names(self):
        series = pd.Series(["Afatinib", "Trametinib", "Navitoclax"])
        assert metadata._infer_perturbation_type(series) == "drug"

    def test_mixed_leans_drug(self):
        series = pd.Series(["Afatinib", "control", "TP53", "Navitoclax"])
        assert metadata._infer_perturbation_type(series) == "drug"

    def test_empty_series(self):
        series = pd.Series([], dtype=str)
        assert metadata._infer_perturbation_type(series) == "drug"

    def test_all_nan(self):
        series = pd.Series([None, None, None])
        assert metadata._infer_perturbation_type(series) == "drug"

    def test_gene_ids_uppercase(self):
        series = pd.Series(["ABCB1", "SLC6A3", "CYP2D6", "BRCA1", "PTEN"])
        assert metadata._infer_perturbation_type(series) == "genetic"
