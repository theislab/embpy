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
# TestNonHumanResolution
#
# Cross-species work resolves lowercase, non-mammalian symbols through
# Ensembl-style organism names ("danio_rerio"), and both of those used to
# break resolution in ways that were silent.
# =====================================================================


def _empty_mygene_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"hits": []}
    return resp


def _empty_search_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"results": []}
    return resp


class TestNonHumanResolution:
    @patch("embpy.resources.protein_resolver.requests.get")
    def test_lowercase_zebrafish_symbol_without_a_reviewed_entry(self, mock_get):
        """Regression: zebrafish `cdk1` has no Swiss-Prot entry.

        The reviewed-only query returned nothing and the ortholog was dropped,
        which cost the cross-species notebook six of eight zebrafish proteins.
        Verified against UniProt: cdk1, mapk1, jun and casp3 are TrEMBL-only in
        taxon 7955, while gapdh, src, tp53 and ldha are reviewed -- which is why
        the failure looked random rather than systematic.
        """
        resolver = ProteinResolver(organism="danio_rerio", request_timeout=5)
        mock_get.side_effect = [
            _empty_mygene_response(),                    # MyGene has no Swiss-Prot
            _empty_search_response(),                    # reviewed:true -> nothing
            _mock_uniprot_search_response("Q7T3L7"),     # unreviewed -> TrEMBL hit
        ]
        assert resolver.resolve_uniprot_id("cdk1", id_type="symbol") == "Q7T3L7"

    @patch("embpy.resources.protein_resolver.requests.get")
    def test_reviewed_is_tried_first_and_short_circuits(self, mock_get):
        """A reviewed entry must win, and must not cost a second request.

        Dropping `reviewed:true` altogether would also have fixed zebrafish, at
        the cost of silently degrading the common case: an unreviewed search for
        human TP53 returns the fragment K7PPA8 rather than canonical P04637.
        """
        resolver = ProteinResolver(organism="human", request_timeout=5)
        mock_get.side_effect = [
            _empty_mygene_response(),
            _mock_uniprot_search_response("P04637"),
        ]
        assert resolver.resolve_uniprot_id("TP53", id_type="symbol") == "P04637"
        assert mock_get.call_count == 2, "a reviewed hit must not trigger the fallback"

        first_query = mock_get.call_args_list[1][1]["params"]["query"]
        assert "reviewed:true" in first_query
        assert "organism_id:9606" in first_query

    @patch("embpy.resources.protein_resolver.requests.get")
    def test_fallback_query_drops_only_the_reviewed_filter(self, mock_get):
        resolver = ProteinResolver(organism="danio_rerio", request_timeout=5)
        mock_get.side_effect = [
            _empty_mygene_response(),
            _empty_search_response(),
            _mock_uniprot_search_response("Q7T3L7"),
        ]
        resolver.resolve_uniprot_id("cdk1", id_type="symbol")

        fallback_query = mock_get.call_args_list[2][1]["params"]["query"]
        assert "reviewed:true" not in fallback_query
        assert "organism_id:7955" in fallback_query
        assert "gene_exact:cdk1" in fallback_query

    @patch("embpy.resources.protein_resolver.requests.get")
    def test_mygene_receives_a_taxid_not_an_ensembl_name(self, mock_get):
        """MyGene 400s on "danio_rerio", and the error was swallowed at debug.

        That silently disabled the MyGene leg for every cross-species lookup,
        so everything fell through to the UniProt search without any sign.
        """
        resolver = ProteinResolver(organism="danio_rerio", request_timeout=5)
        mock_get.side_effect = [
            _empty_mygene_response(),
            _mock_uniprot_search_response("Q7T3L7"),
        ]
        resolver.resolve_uniprot_id("cdk1", id_type="symbol")

        species = mock_get.call_args_list[0][1]["params"]["species"]
        assert species == 7955, f"MyGene needs a taxid or common name, got {species!r}"

    @patch("embpy.resources.protein_resolver.requests.get")
    def test_unknown_organism_still_gives_up_cleanly(self, mock_get):
        resolver = ProteinResolver(organism="tyrannosaurus_rex", request_timeout=5)
        mock_get.side_effect = [_empty_mygene_response()]
        assert resolver.resolve_uniprot_id("trex1", id_type="symbol") is None


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

    @patch("embpy.resources.protein_resolver.requests.get")
    def test_queries_the_search_endpoint_not_single_entry_retrieval(
        self, mock_get, resolver,
    ):
        """Regression: `includeIsoform` is a search-only parameter.

        On the single-entry route (``/uniprotkb/{accession}.fasta``) UniProt
        ignores it and returns the canonical sequence alone, which silently
        reduced ``isoform="all"`` to one vector. Verified live against P04637:
        retrieval yields 1 record, search yields 9. The other tests here mock
        ``requests.get`` wholesale and so pass on either URL -- this one pins
        the request itself.
        """
        mock_get.side_effect = [
            _mock_mygene_response(),
            _mock_fasta_response(FAKE_ISOFORM_FASTA),
        ]
        resolver.get_isoforms("TP53", id_type="symbol")

        url, kwargs = mock_get.call_args_list[-1][0][0], mock_get.call_args_list[-1][1]
        params = kwargs["params"]
        assert url.endswith("/uniprotkb/search"), (
            f"isoforms must come from the search endpoint, got {url!r}"
        )
        assert params["includeIsoform"] == "true"
        assert params["format"] == "fasta"
        assert FAKE_ACCESSION in params["query"]


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
