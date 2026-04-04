"""Tests for embpy.resources.text_resolver.TextResolver."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from embpy.resources.text_resolver import TextResolver


@pytest.fixture
def resolver():
    return TextResolver(organism="human", rate_limit_delay=0)


MYGENE_HIT = {
    "hits": [{
        "symbol": "TP53",
        "name": "tumor protein p53",
        "summary": "This gene encodes a tumor suppressor protein.",
        "type_of_gene": "protein-coding",
        "pathway": {
            "reactome": [
                {"id": "R-HSA-123", "name": "Apoptosis"},
                {"id": "R-HSA-456", "name": "Cell Cycle"},
            ],
        },
    }],
}

NCBI_SEARCH = {"esearchresult": {"idlist": ["7157"]}}
NCBI_SUMMARY = {
    "result": {
        "7157": {
            "name": "TP53",
            "description": "tumor protein p53",
            "nomenclaturesymbol": "TP53",
            "nomenclaturename": "tumor protein p53",
            "summary": "Encodes a tumor suppressor.",
        }
    }
}

ENSEMBL_LOOKUP = {
    "display_name": "TP53",
    "description": "tumor protein p53 [Source:HGNC Symbol;Acc:HGNC:11998]",
    "biotype": "protein_coding",
}

UNIPROT_ENTRY = {
    "proteinDescription": {
        "recommendedName": {"fullName": {"value": "Cellular tumor antigen p53"}},
    },
    "comments": [
        {"commentType": "FUNCTION", "texts": [{"value": "Acts as a tumor suppressor."}]},
        {
            "commentType": "SUBCELLULAR LOCATION",
            "subcellularLocations": [{"location": {"value": "Nucleus"}}],
        },
        {"commentType": "DISEASE", "texts": [{"value": "Li-Fraumeni syndrome"}]},
    ],
}

WIKIPEDIA_RESPONSE = {
    "extract": "p53, also known as TP53, is a tumor suppressor protein.",
    "type": "standard",
}

PUBCHEM_RESPONSE = {
    "InformationList": {
        "Information": [
            {"Description": "Aspirin is a nonsteroidal anti-inflammatory drug."},
        ],
    },
}


class TestMyGeneSource:
    @patch("embpy.resources.text_resolver._get_json", return_value=MYGENE_HIT)
    def test_fetch_mygene(self, mock_get, resolver):
        text = resolver._fetch_mygene("TP53")
        assert "TP53" in text
        assert "tumor protein p53" in text
        assert "tumor suppressor" in text
        assert "Apoptosis" in text

    @patch("embpy.resources.text_resolver._get_json", return_value={"hits": []})
    def test_empty_hits(self, mock_get, resolver):
        text = resolver._fetch_mygene("FAKEGENE")
        assert text == ""


class TestNCBISource:
    @patch("embpy.resources.text_resolver._get_json")
    def test_fetch_ncbi(self, mock_get, resolver):
        mock_get.side_effect = [NCBI_SEARCH, NCBI_SUMMARY]
        text = resolver._fetch_ncbi("TP53")
        assert "TP53" in text
        assert "tumor suppressor" in text.lower() or "tumor protein" in text.lower()

    @patch("embpy.resources.text_resolver._get_json", return_value=None)
    def test_ncbi_no_results(self, mock_get, resolver):
        assert resolver._fetch_ncbi("FAKEGENE") == ""


class TestEnsemblSource:
    @patch("embpy.resources.text_resolver._get_json", return_value=ENSEMBL_LOOKUP)
    def test_fetch_ensembl(self, mock_get, resolver):
        text = resolver._fetch_ensembl("TP53")
        assert "TP53" in text
        assert "protein_coding" in text
        assert "tumor protein p53" in text

    @patch("embpy.resources.text_resolver._get_json", return_value=None)
    def test_ensembl_not_found(self, mock_get, resolver):
        assert resolver._fetch_ensembl("FAKEGENE") == ""


class TestUniProtSource:
    @patch("embpy.resources.text_resolver._get_json", return_value=UNIPROT_ENTRY)
    def test_fetch_uniprot_by_accession(self, mock_get, resolver):
        text = resolver._fetch_uniprot("P04637")
        assert "tumor" in text.lower()
        assert "Nucleus" in text

    @patch("embpy.resources.text_resolver._get_json", return_value=None)
    def test_uniprot_not_found(self, mock_get, resolver):
        with patch("embpy.resources.protein_resolver.ProteinResolver") as mock_pr_cls:
            mock_pr_cls.return_value.resolve_uniprot_id.return_value = None
            assert resolver._fetch_uniprot("FAKEGENE") == ""


class TestWikipediaSource:
    @patch("embpy.resources.text_resolver._get_json", return_value=WIKIPEDIA_RESPONSE)
    def test_fetch_wikipedia(self, mock_get, resolver):
        text = resolver._fetch_wikipedia("TP53")
        assert "tumor suppressor" in text

    @patch("embpy.resources.text_resolver._get_json", return_value=None)
    def test_wikipedia_not_found(self, mock_get, resolver):
        assert resolver._fetch_wikipedia("xyznonexistent") == ""

    @patch("embpy.resources.text_resolver._get_json")
    def test_wikipedia_disambiguation_fallback(self, mock_get, resolver):
        mock_get.side_effect = [
            {"extract": "", "type": "disambiguation"},
            WIKIPEDIA_RESPONSE,
        ]
        text = resolver._fetch_wikipedia("TP53")
        assert "tumor suppressor" in text


class TestPubChemSource:
    @patch("embpy.resources.text_resolver._get_json", return_value=PUBCHEM_RESPONSE)
    def test_fetch_pubchem(self, mock_get, resolver):
        text = resolver._fetch_pubchem("aspirin")
        assert "anti-inflammatory" in text.lower()

    @patch("embpy.resources.text_resolver._get_json", return_value=None)
    def test_pubchem_not_found(self, mock_get, resolver):
        assert resolver._fetch_pubchem("xyzfakecompound") == ""


class TestGetDescription:
    @patch("embpy.resources.text_resolver._get_json", return_value=MYGENE_HIT)
    def test_gene_auto_detect(self, mock_get, resolver):
        descs = resolver.get_description("TP53", entity_type="auto")
        assert "mygene" in descs

    def test_source_filtering(self, resolver):
        with patch.object(resolver, "_fetch_mygene", return_value="mygene text"):
            descs = resolver.get_gene_description("TP53", sources=["mygene"])
            assert "mygene" in descs
            assert "ncbi" not in descs


class TestCombinedDescription:
    def test_section_headers(self, resolver):
        with patch.object(resolver, "_fetch_mygene", return_value="MyGene desc"), \
             patch.object(resolver, "_fetch_wikipedia", return_value="Wiki desc"):
            text = resolver.get_combined_description("TP53", entity_type="gene",
                                                      sources=["mygene", "wikipedia"])
            assert "[MYGENE]" in text
            assert "[WIKIPEDIA]" in text

    def test_empty_results(self, resolver):
        with patch.object(resolver, "_fetch_mygene", return_value=""), \
             patch.object(resolver, "_fetch_wikipedia", return_value=""):
            text = resolver.get_combined_description("FAKEGENE", entity_type="gene",
                                                      sources=["mygene", "wikipedia"])
            assert "No description" in text

    def test_custom_template(self, resolver):
        with patch.object(resolver, "_fetch_mygene", return_value="desc1"), \
             patch.object(resolver, "_fetch_wikipedia", return_value="desc2"):
            text = resolver.get_combined_description(
                "TP53", entity_type="gene",
                sources=["mygene", "wikipedia"],
                template="Gene={identifier}. MG={mygene}. WK={wikipedia}.",
            )
            assert "Gene=TP53" in text
            assert "MG=desc1" in text


class TestEntityDetection:
    def test_smiles_detected(self):
        assert TextResolver._detect_entity_type("CC(=O)Oc1ccccc1C(O)=O") == "molecule"

    def test_ensembl_gene(self):
        assert TextResolver._detect_entity_type("ENSG00000141510") == "gene"

    def test_mouse_ensembl(self):
        assert TextResolver._detect_entity_type("ENSMUSG00000059552") == "gene"

    def test_uniprot_accession(self):
        assert TextResolver._detect_entity_type("P04637") == "protein"

    def test_gene_symbol(self):
        assert TextResolver._detect_entity_type("TP53") == "gene"

    def test_short_symbol(self):
        assert TextResolver._detect_entity_type("MYC") == "gene"
