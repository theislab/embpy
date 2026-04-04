"""Tests for embpy.resources.cellline_annotator.CellLineAnnotator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from embpy.resources.cellline_annotator import CellLineAnnotator


@pytest.fixture
def annotator():
    return CellLineAnnotator(rate_limit_delay=0)


CELLOSAURUS_SEARCH = {
    "resultList": {
        "result": [{"accession": "CVCL_0023"}],
    },
}

CELLOSAURUS_ENTRY = {
    "cellLine": {
        "nameList": [{"value": "A549"}],
        "category": "Cancer cell line",
        "speciesList": [{"value": "Homo sapiens"}],
        "derivedFromSite": [{"value": "Lung"}],
        "diseaseList": [{"value": "Non-small cell lung carcinoma"}],
        "xrefList": [
            {"database": "ATCC", "accession": "CCL-185"},
            {"database": "CCLE", "accession": "A549_LUNG"},
        ],
        "registrationProblemList": [],
    },
}

DEPMAP_RESPONSE = {
    "depmap_id": "ACH-000681",
    "cell_line_name": "A549",
    "lineage": "Lung",
    "lineage_subtype": "NSCLC adenocarcinoma",
    "primary_disease": "Lung Cancer",
    "disease_subtype": "Non-Small Cell Lung Cancer",
    "growth_pattern": "Adherent",
    "culture_medium": "RPMI",
    "sex": "Male",
    "source": "ATCC",
}

CMP_RESPONSE = {
    "data": [{
        "id": "SIDM00001",
        "attributes": {
            "model_name": "A549",
            "tissue": "Lung",
            "cancer_type": "Lung Carcinoma",
            "cancer_type_detail": "Non-Small Cell Lung Carcinoma",
            "model_type": "Cell Line",
            "msi_status": "MSS",
            "ploidy": "3.2",
            "mutational_burden": "5.1",
            "growth_properties": "Adherent",
        },
    }],
}


class TestIdentifierHelpers:
    def test_cellosaurus_id(self):
        assert CellLineAnnotator._is_cellosaurus_id("CVCL_0023") is True
        assert CellLineAnnotator._is_cellosaurus_id("A549") is False

    def test_depmap_id(self):
        assert CellLineAnnotator._is_depmap_id("ACH-000681") is True
        assert CellLineAnnotator._is_depmap_id("A549") is False
        assert CellLineAnnotator._is_depmap_id("ACH-12345") is False


class TestCellosaurusSource:
    @patch("embpy.resources.cellline_annotator._get_json")
    def test_get_cellosaurus_info(self, mock_get, annotator):
        mock_get.side_effect = [CELLOSAURUS_SEARCH, CELLOSAURUS_ENTRY]
        info = annotator.get_cellosaurus_info("A549")
        assert info["accession"] == "CVCL_0023"
        assert info["species"] == "Homo sapiens"
        assert info["tissue"] == "Lung"
        assert info["disease"] == "Non-small cell lung carcinoma"
        assert info["cross_references"]["ATCC"] == "CCL-185"
        assert info["is_problematic"] is False

    @patch("embpy.resources.cellline_annotator._get_json", return_value=None)
    def test_cellosaurus_not_found(self, mock_get, annotator):
        info = annotator.get_cellosaurus_info("NONEXISTENT_CELL_LINE")
        assert info == {}


class TestDepMapSource:
    @patch("embpy.resources.cellline_annotator._get_json", return_value=DEPMAP_RESPONSE)
    def test_get_depmap_info(self, mock_get, annotator):
        info = annotator.get_depmap_info("A549")
        assert info["depmap_id"] == "ACH-000681"
        assert info["lineage"] == "Lung"
        assert info["primary_disease"] == "Lung Cancer"
        assert info["growth_pattern"] == "Adherent"

    @patch("embpy.resources.cellline_annotator._get_json", return_value=None)
    def test_depmap_not_found(self, mock_get, annotator):
        info = annotator.get_depmap_info("NONEXISTENT")
        assert info == {}


class TestPassportsSource:
    @patch("embpy.resources.cellline_annotator._get_json", return_value=CMP_RESPONSE)
    def test_get_passports_info(self, mock_get, annotator):
        info = annotator.get_passports_info("A549")
        assert info["model_name"] == "A549"
        assert info["cancer_type"] == "Lung Carcinoma"
        assert info["model_type"] == "Cell Line"
        assert info["msi_status"] == "MSS"

    @patch("embpy.resources.cellline_annotator._get_json", return_value={"data": []})
    def test_passports_not_found(self, mock_get, annotator):
        info = annotator.get_passports_info("NONEXISTENT")
        assert info == {}


class TestAnnotateCombined:
    @patch("embpy.resources.cellline_annotator._get_json")
    def test_annotate_all_sources(self, mock_get, annotator):
        mock_get.side_effect = [
            CELLOSAURUS_SEARCH, CELLOSAURUS_ENTRY,
            DEPMAP_RESPONSE,
            CMP_RESPONSE,
        ]
        ann = annotator.annotate("A549")
        assert ann["name"] == "A549"
        assert ann["cellosaurus_id"] == "CVCL_0023"
        assert ann["depmap_id"] == "ACH-000681"
        assert ann["tissue"] == "Lung"
        assert ann["lineage"] == "Lung"
        assert "cellosaurus" in ann["sources"]
        assert "depmap" in ann["sources"]
        assert "passports" in ann["sources"]

    @patch("embpy.resources.cellline_annotator._get_json")
    def test_annotate_single_source(self, mock_get, annotator):
        mock_get.side_effect = [CELLOSAURUS_SEARCH, CELLOSAURUS_ENTRY]
        ann = annotator.annotate("A549", sources=["cellosaurus"])
        assert "cellosaurus" in ann["sources"]
        assert "depmap" not in ann["sources"]


class TestAnnotateAdata:
    @patch("embpy.resources.cellline_annotator._get_json")
    def test_annotate_adata(self, mock_get, annotator):
        mock_get.side_effect = [
            CELLOSAURUS_SEARCH, CELLOSAURUS_ENTRY,
            DEPMAP_RESPONSE, CMP_RESPONSE,
        ] * 2

        obs = pd.DataFrame(
            {"cell_line": ["A549", "A549", "A549"]},
            index=pd.Index(["c1", "c2", "c3"]),
        )
        adata = AnnData(obs=obs)

        result = annotator.annotate_adata(adata, column="cell_line")
        assert "cellline_tissue" in result.obs.columns
        assert "cellline_disease" in result.obs.columns
        assert "cellline_annotations" in result.uns


class TestWikipediaSource:
    @patch("embpy.resources.cellline_annotator._get_json")
    def test_get_wikipedia_info(self, mock_get, annotator):
        mock_get.return_value = {
            "extract": "HeLa is an immortal cell line derived from cervical cancer cells.",
            "type": "standard",
        }
        text = annotator.get_wikipedia_info("HeLa")
        assert "immortal cell line" in text

    @patch("embpy.resources.cellline_annotator._get_json")
    def test_wikipedia_fallback_cell_line_suffix(self, mock_get, annotator):
        mock_get.side_effect = [
            {"extract": "", "type": "disambiguation"},
            {"extract": "K562 is a chronic myeloid leukemia cell line."},
        ]
        text = annotator.get_wikipedia_info("K562")
        assert "myeloid" in text.lower()

    @patch("embpy.resources.cellline_annotator._get_json", return_value=None)
    def test_wikipedia_not_found(self, mock_get, annotator):
        text = annotator.get_wikipedia_info("NONEXISTENT_CELL_XYZ")
        assert text == ""


class TestTextDescription:
    @patch("embpy.resources.cellline_annotator._get_json")
    def test_get_text_description(self, mock_get, annotator):
        wiki_response = {"extract": "A549 cells are adenocarcinomic alveolar basal epithelial cells.", "type": "standard"}
        mock_get.side_effect = [
            CELLOSAURUS_SEARCH, CELLOSAURUS_ENTRY,
            DEPMAP_RESPONSE, CMP_RESPONSE,
            wiki_response,
        ]
        text = annotator.get_text_description("A549")
        assert "A549" in text
        assert "cell line" in text.lower()
        assert "Lung" in text
        assert "Homo sapiens" in text
        assert "adenocarcinomic" in text


class TestTextResolverCellLine:
    def test_detect_depmap_id(self):
        from embpy.resources.text_resolver import TextResolver

        assert TextResolver._detect_entity_type("ACH-000681") == "cellline"

    def test_detect_cellosaurus_id(self):
        from embpy.resources.text_resolver import TextResolver

        assert TextResolver._detect_entity_type("CVCL_0023") == "cellline"

    def test_gene_not_cellline(self):
        from embpy.resources.text_resolver import TextResolver

        assert TextResolver._detect_entity_type("TP53") == "gene"
