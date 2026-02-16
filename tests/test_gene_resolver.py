"""Tests for GeneResolver with mocked HTTP calls and pyensembl."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from embpy.resources.gene_resolver import GeneResolver


@pytest.fixture
def resolver():
    """GeneResolver with pyensembl disabled (API-only mode)."""
    r = GeneResolver.__new__(GeneResolver)
    r.ensembl = None
    r.mart_file = None
    r.chrom_folder = None
    r.release_version = 109
    r.species = "human"
    return r


class TestGetDnaSequence:
    def test_by_symbol(self, resolver):
        lookup_mock = MagicMock()
        lookup_mock.status_code = 200
        lookup_mock.raise_for_status = MagicMock()
        lookup_mock.json.return_value = {"id": "ENSG00000141510"}

        seq_mock = MagicMock()
        seq_mock.status_code = 200
        seq_mock.raise_for_status = MagicMock()
        seq_mock.text = "ACGTACGTACGT"

        with patch("embpy.resources.gene_resolver.requests.get", side_effect=[lookup_mock, seq_mock]):
            result = resolver.get_dna_sequence("TP53", "symbol")
            assert result == "ACGTACGTACGT"

    def test_by_ensembl_id(self, resolver):
        lookup_mock = MagicMock()
        lookup_mock.status_code = 200
        lookup_mock.raise_for_status = MagicMock()
        lookup_mock.json.return_value = {"id": "ENSG00000141510"}

        seq_mock = MagicMock()
        seq_mock.status_code = 200
        seq_mock.raise_for_status = MagicMock()
        seq_mock.text = "ACGTACGTACGT"

        with patch("embpy.resources.gene_resolver.requests.get", side_effect=[lookup_mock, seq_mock]):
            result = resolver.get_dna_sequence("ENSG00000141510", "ensembl_id")
            assert result == "ACGTACGTACGT"

    def test_unsupported_id_type(self, resolver):
        result = resolver.get_dna_sequence("TP53", "invalid_type")
        assert result is None

    def test_api_error_returns_none(self, resolver):
        import requests

        mock = MagicMock()
        mock.raise_for_status.side_effect = requests.RequestException("API down")

        with patch("embpy.resources.gene_resolver.requests.get", return_value=mock):
            result = resolver.get_dna_sequence("TP53", "symbol")
            assert result is None

    def test_no_ensembl_id_resolved(self, resolver):
        mock = MagicMock()
        mock.status_code = 200
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {}

        with patch("embpy.resources.gene_resolver.requests.get", return_value=mock):
            result = resolver.get_dna_sequence("TP53", "symbol")
            assert result is None


class TestGetProteinSequence:
    def test_by_uniprot_id(self, resolver):
        fasta_mock = MagicMock()
        fasta_mock.status_code = 200
        fasta_mock.raise_for_status = MagicMock()
        fasta_mock.text = ">sp|P04637|P53_HUMAN\nMEEPQSDPSVEPPLSQ\nETFSDLWKLLPENNVL"

        with patch("embpy.resources.gene_resolver.requests.get", return_value=fasta_mock):
            result = resolver.get_protein_sequence("P04637", "uniprot_id")
            assert result is not None
            assert result.startswith("MEEPQ")
            assert "\n" not in result

    def test_by_symbol(self, resolver):
        query_mock = MagicMock()
        query_mock.status_code = 200
        query_mock.raise_for_status = MagicMock()
        query_mock.json.return_value = {"hits": [{"uniprot": {"Swiss-Prot": "P04637"}}]}

        fasta_mock = MagicMock()
        fasta_mock.status_code = 200
        fasta_mock.raise_for_status = MagicMock()
        fasta_mock.text = ">sp|P04637\nMTEYKLVVVGAGGVGKS"

        with patch(
            "embpy.resources.gene_resolver.requests.get",
            side_effect=[query_mock, fasta_mock],
        ):
            result = resolver.get_protein_sequence("TP53", "symbol")
            assert result is not None
            assert result.startswith("MTEYKLVVVG")

    def test_no_uniprot_found(self, resolver):
        mock = MagicMock()
        mock.status_code = 200
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"hits": [{"uniprot": {}}]}

        with patch("embpy.resources.gene_resolver.requests.get", return_value=mock):
            result = resolver.get_protein_sequence("FAKE", "symbol")
            assert result is None


class TestGetGeneDescription:
    def test_success(self, resolver):
        mock = MagicMock()
        mock.status_code = 200
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {
            "hits": [
                {
                    "symbol": "TP53",
                    "name": "tumor protein p53",
                    "summary": "This gene encodes a tumor suppressor protein.",
                }
            ]
        }

        with patch("embpy.resources.gene_resolver.requests.get", return_value=mock):
            result = resolver.get_gene_description("TP53", "symbol")
            assert result is not None
            assert "TP53" in result
            assert "tumor" in result.lower()

    def test_no_hits(self, resolver):
        mock = MagicMock()
        mock.status_code = 200
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"hits": []}

        with patch("embpy.resources.gene_resolver.requests.get", return_value=mock):
            result = resolver.get_gene_description("FAKEGENE", "symbol")
            assert result is None

    def test_unsupported_id_type(self, resolver):
        result = resolver.get_gene_description("TP53", "invalid_type")
        assert result is None


class TestSymbolToEnsembl:
    def test_via_mygene(self, resolver):
        mock = MagicMock()
        mock.status_code = 200
        mock.ok = True
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"hits": [{"ensembl": {"gene": "ENSG00000141510"}}]}

        with patch("embpy.resources.gene_resolver.requests.get", return_value=mock):
            result = resolver.symbol_to_ensembl("TP53")
            assert result == "ENSG00000141510"

    def test_no_result_returns_none(self, resolver):
        mock = MagicMock()
        mock.status_code = 200
        mock.ok = False
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"hits": []}

        with patch("embpy.resources.gene_resolver.requests.get", return_value=mock):
            result = resolver.symbol_to_ensembl("FAKEGENE")
            assert result is None


class TestEnsemblToSymbol:
    def test_via_mygene(self, resolver):
        mock = MagicMock()
        mock.status_code = 200
        mock.ok = True
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"hits": [{"symbol": "TP53"}]}

        with patch("embpy.resources.gene_resolver.requests.get", return_value=mock):
            result = resolver.ensembl_to_symbol("ENSG00000141510")
            assert result == "TP53"

    def test_strips_version(self, resolver):
        mock = MagicMock()
        mock.status_code = 200
        mock.ok = True
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"hits": [{"symbol": "TP53"}]}

        with patch("embpy.resources.gene_resolver.requests.get", return_value=mock):
            result = resolver.ensembl_to_symbol("ENSG00000141510.12")
            assert result == "TP53"


class TestBatchMappings:
    def test_symbols_to_ensembl_batch(self, resolver):
        resolver.symbol_to_ensembl = MagicMock(side_effect=["ENSG1", "ENSG2"])
        result = resolver.symbols_to_ensembl_batch(["TP53", "BRCA1"])
        assert result == {"TP53": "ENSG1", "BRCA1": "ENSG2"}

    def test_ensembl_to_symbols_batch(self, resolver):
        resolver.ensembl_to_symbol = MagicMock(side_effect=["TP53", "BRCA1"])
        result = resolver.ensembl_to_symbols_batch(["ENSG1", "ENSG2"])
        assert result == {"ENSG1": "TP53", "ENSG2": "BRCA1"}


class TestGetLocalDnaSequence:
    def test_requires_mart_file(self, resolver):
        with pytest.raises(ValueError, match="mart_file"):
            resolver.get_local_dna_sequence("TP53", "symbol")
