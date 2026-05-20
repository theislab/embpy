"""Tests for embpy.resources.protein_resolver -- ProteinResolver."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from embpy.resources.protein_resolver import ProteinResolver


# =====================================================================
# Fixtures
# =====================================================================

FAKE_ACCESSION = "P04637"
FAKE_CANONICAL_SEQ = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPS"
FAKE_ISOFORM_FASTA = (
    ">sp|P04637|P53_HUMAN Cellular tumor antigen p53\n"
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPS\n"
    ">sp|P04637-2|P53_HUMAN Isoform 2\n"
    "MEEPQSDPSVEPPLSQETFSDLWKLLP\n"
    ">sp|P04637-3|P53_HUMAN Isoform 3\n"
    "MEEPQSDPSVEPPL\n"
)


@pytest.fixture
def resolver():
    return ProteinResolver(organism="human", request_timeout=5)


def _mock_mygene_response(accession=FAKE_ACCESSION):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "hits": [{"uniprot": {"Swiss-Prot": accession}}],
    }
    return resp


def _mock_fasta_response(fasta_text=None):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    if fasta_text is None:
        fasta_text = f">sp|{FAKE_ACCESSION}|P53_HUMAN\n{FAKE_CANONICAL_SEQ}\n"
    resp.text = fasta_text
    return resp


def _mock_uniprot_search_response(accession=FAKE_ACCESSION):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "results": [{"primaryAccession": accession}],
    }
    return resp


# =====================================================================
# TestResolveUniprotId
# =====================================================================


class TestResolveUniprotId:
    def test_uniprot_id_passthrough(self, resolver):
        result = resolver.resolve_uniprot_id("P04637", id_type="uniprot_id")
        assert result == "P04637"

    def test_uniprot_id_strips_isoform_suffix(self, resolver):
        result = resolver.resolve_uniprot_id("P04637-2", id_type="uniprot_id")
        assert result == "P04637"

    @patch("embpy.resources.protein_resolver.requests.get")
    def test_symbol_via_mygene(self, mock_get, resolver):
        mock_get.return_value = _mock_mygene_response()
        result = resolver.resolve_uniprot_id("TP53", id_type="symbol")
        assert result == FAKE_ACCESSION

    @patch("embpy.resources.protein_resolver.requests.get")
    def test_ensembl_via_mygene(self, mock_get, resolver):
        mock_get.return_value = _mock_mygene_response()
        result = resolver.resolve_uniprot_id(
            "ENSG00000141510", id_type="ensembl_id",
        )
        assert result == FAKE_ACCESSION

    @patch("embpy.resources.protein_resolver.requests.get")
    def test_mygene_miss_falls_back_to_uniprot_search(self, mock_get, resolver):
        empty_mygene = MagicMock()
        empty_mygene.status_code = 200
        empty_mygene.raise_for_status = MagicMock()
        empty_mygene.json.return_value = {"hits": []}

        mock_get.side_effect = [
            empty_mygene,
            _mock_uniprot_search_response(),
        ]
        result = resolver.resolve_uniprot_id("TP53", id_type="symbol")
        assert result == FAKE_ACCESSION

    @patch("embpy.resources.protein_resolver.requests.get")
    def test_caching(self, mock_get, resolver):
        mock_get.return_value = _mock_mygene_response()
        r1 = resolver.resolve_uniprot_id("TP53", id_type="symbol")
        r2 = resolver.resolve_uniprot_id("TP53", id_type="symbol")
        assert r1 == r2
        assert mock_get.call_count == 1

    @patch("embpy.resources.protein_resolver.requests.get")
    def test_both_fail_returns_none(self, mock_get, resolver):
        fail_resp = MagicMock()
        fail_resp.raise_for_status.side_effect = Exception("API error")
        mock_get.return_value = fail_resp
        result = resolver.resolve_uniprot_id("FAKEGENE", id_type="symbol")
        assert result is None


# =====================================================================
# TestGetCanonicalSequence
# =====================================================================


class TestGetCanonicalSequence:
    @patch("embpy.resources.protein_resolver.requests.get")
    def test_returns_sequence(self, mock_get, resolver):
        mock_get.side_effect = [
            _mock_mygene_response(),
            _mock_fasta_response(),
        ]
        seq = resolver.get_canonical_sequence("TP53", id_type="symbol")
        assert seq == FAKE_CANONICAL_SEQ

    @patch("embpy.resources.protein_resolver.requests.get")
    def test_unresolvable_returns_none(self, mock_get, resolver):
        fail = MagicMock()
        fail.raise_for_status.side_effect = Exception("fail")
        mock_get.return_value = fail
        seq = resolver.get_canonical_sequence("FAKEGENE", id_type="symbol")
        assert seq is None


# =====================================================================
# TestGetIsoforms
# =====================================================================


class TestGetIsoforms:
    @patch("embpy.resources.protein_resolver.requests.get")
    def test_returns_dict_with_isoforms(self, mock_get, resolver):
        mock_get.side_effect = [
            _mock_mygene_response(),
            _mock_fasta_response(FAKE_ISOFORM_FASTA),
        ]
        isoforms = resolver.get_isoforms("TP53", id_type="symbol")
        assert isinstance(isoforms, dict)
        assert len(isoforms) == 3
        assert "P04637" in isoforms
        assert "P04637-2" in isoforms
        assert "P04637-3" in isoforms

    @patch("embpy.resources.protein_resolver.requests.get")
    def test_exclude_canonical(self, mock_get, resolver):
        mock_get.side_effect = [
            _mock_mygene_response(),
            _mock_fasta_response(FAKE_ISOFORM_FASTA),
        ]
        isoforms = resolver.get_isoforms(
            "TP53", id_type="symbol", include_canonical=False,
        )
        assert "P04637" not in isoforms
        assert len(isoforms) == 2

    @patch("embpy.resources.protein_resolver.requests.get")
    def test_unresolvable_returns_empty(self, mock_get, resolver):
        fail = MagicMock()
        fail.raise_for_status.side_effect = Exception("fail")
        mock_get.return_value = fail
        result = resolver.get_isoforms("FAKEGENE", id_type="symbol")
        assert result == {}


# =====================================================================
# TestParseFasta
# =====================================================================


class TestParseFasta:
    def test_single_entry(self):
        fasta = ">sp|P04637|P53_HUMAN\nMEEPQ\nSDPSV\n"
        result = ProteinResolver._parse_multi_fasta(fasta)
        assert result == {"P04637": "MEEPQSDPSV"}

    def test_multiple_entries(self):
        result = ProteinResolver._parse_multi_fasta(FAKE_ISOFORM_FASTA)
        assert len(result) == 3
        assert result["P04637"] == FAKE_CANONICAL_SEQ
        assert result["P04637-2"] == "MEEPQSDPSVEPPLSQETFSDLWKLLP"
        assert result["P04637-3"] == "MEEPQSDPSVEPPL"

    def test_exclude_canonical(self):
        result = ProteinResolver._parse_multi_fasta(
            FAKE_ISOFORM_FASTA, include_canonical=False,
        )
        assert "P04637" not in result
        assert len(result) == 2

    def test_empty_fasta(self):
        result = ProteinResolver._parse_multi_fasta("")
        assert result == {}


# =====================================================================
# TestBatchMethods
# =====================================================================


class TestBatchMethods:
    @patch("embpy.resources.protein_resolver.requests.get")
    def test_canonical_batch(self, mock_get, resolver):
        mock_get.side_effect = [
            _mock_mygene_response(),
            _mock_fasta_response(),
            _mock_mygene_response("Q9Y6K1"),
            _mock_fasta_response(">sp|Q9Y6K1|FAKE\nACDEFG\n"),
        ]
        results = resolver.get_canonical_sequences_batch(
            ["TP53", "BRCA1"], id_type="symbol",
        )
        assert len(results) == 2

    @patch("embpy.resources.protein_resolver.requests.get")
    def test_isoforms_batch(self, mock_get, resolver):
        mock_get.side_effect = [
            _mock_mygene_response(),
            _mock_fasta_response(FAKE_ISOFORM_FASTA),
            _mock_mygene_response("Q9Y6K1"),
            _mock_fasta_response(">sp|Q9Y6K1|FAKE\nACDEFG\n"),
        ]
        results = resolver.get_isoforms_batch(
            ["TP53", "BRCA1"], id_type="symbol",
        )
        assert len(results) == 2
        assert isinstance(results["TP53"], dict)
