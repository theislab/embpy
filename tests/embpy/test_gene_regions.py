"""Tests for gene region (exon/intron) extraction in GeneResolver."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from embpy.resources.gene_resolver import GeneResolver

_GENE_LOOKUP_RESPONSE = {
    "id": "ENSG00000141510",
    "display_name": "TP53",
    "strand": 1,
    "seq_region_name": "17",
    "Transcript": [
        {
            "id": "ENST00000269305",
            "is_canonical": 1,
            "Exon": [
                {"id": "ENSE00001657961", "start": 7676594, "end": 7676700},
                {"id": "ENSE00003625790", "start": 7676800, "end": 7676900},
                {"id": "ENSE00003518480", "start": 7677000, "end": 7677100},
            ],
        },
        {
            "id": "ENST00000610623",
            "is_canonical": 0,
            "Exon": [
                {"id": "ENSE00003786076", "start": 7676594, "end": 7676650},
            ],
        },
    ],
}


def _mock_lookup_response():
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status = MagicMock()
    mock.json.return_value = _GENE_LOOKUP_RESPONSE
    return mock


def _mock_sequence_response(seq: str = "ACGTACGT"):
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status = MagicMock()
    mock.text = seq
    return mock


@pytest.fixture()
def resolver():
    return GeneResolver(auto_download=False)


class TestGetGeneRegions:
    @patch("embpy.resources.gene_resolver.requests.get")
    def test_returns_exons(self, mock_get: MagicMock, resolver: GeneResolver) -> None:
        mock_get.side_effect = [
            _mock_lookup_response(),
            _mock_sequence_response("EXON1SEQ"),
            _mock_sequence_response("EXON2SEQ"),
            _mock_sequence_response("EXON3SEQ"),
        ]
        regions = resolver.get_gene_regions("TP53", region="exons")
        assert regions is not None
        assert len(regions) == 3
        assert regions[0]["sequence"] == "EXON1SEQ"
        assert regions[0]["id"] == "ENSE00001657961"
        assert regions[1]["start"] == 7676800

    @patch("embpy.resources.gene_resolver.requests.get")
    def test_returns_introns(self, mock_get: MagicMock, resolver: GeneResolver) -> None:
        mock_get.side_effect = [
            _mock_lookup_response(),
            _mock_sequence_response("INTRON1"),
            _mock_sequence_response("INTRON2"),
        ]
        regions = resolver.get_gene_regions("TP53", region="introns")
        assert regions is not None
        assert len(regions) == 2
        assert regions[0]["id"] == "intron_1"
        assert regions[0]["start"] == 7676701
        assert regions[0]["end"] == 7676799
        assert regions[1]["start"] == 7676901
        assert regions[1]["end"] == 7676999

    @patch("embpy.resources.gene_resolver.requests.get")
    def test_specific_transcript(self, mock_get: MagicMock, resolver: GeneResolver) -> None:
        mock_get.side_effect = [
            _mock_lookup_response(),
            _mock_sequence_response("SINGLE_EXON"),
        ]
        regions = resolver.get_gene_regions("TP53", region="exons", transcript_id="ENST00000610623")
        assert regions is not None
        assert len(regions) == 1

    @patch("embpy.resources.gene_resolver.requests.get")
    def test_missing_transcript_returns_none(self, mock_get: MagicMock, resolver: GeneResolver) -> None:
        mock_get.side_effect = [_mock_lookup_response()]
        regions = resolver.get_gene_regions("TP53", region="exons", transcript_id="ENST_NONEXISTENT")
        assert regions is None

    @patch("embpy.resources.gene_resolver.requests.get")
    def test_ensembl_id_input(self, mock_get: MagicMock, resolver: GeneResolver) -> None:
        mock_get.side_effect = [
            _mock_lookup_response(),
            _mock_sequence_response("EX1"),
            _mock_sequence_response("EX2"),
            _mock_sequence_response("EX3"),
        ]
        regions = resolver.get_gene_regions("ENSG00000141510", id_type="ensembl_id", region="exons")
        assert regions is not None
        assert len(regions) == 3
        url_called = mock_get.call_args_list[0][0][0]
        assert "/lookup/id/" in url_called

    @patch("embpy.resources.gene_resolver.requests.get")
    def test_strand_info_preserved(self, mock_get: MagicMock, resolver: GeneResolver) -> None:
        mock_get.side_effect = [
            _mock_lookup_response(),
            _mock_sequence_response("SEQ"),
            _mock_sequence_response("SEQ"),
            _mock_sequence_response("SEQ"),
        ]
        regions = resolver.get_gene_regions("TP53", region="exons")
        assert regions is not None
        for r in regions:
            assert r["strand"] == 1


class TestGetGeneRegionSequence:
    @patch("embpy.resources.gene_resolver.requests.get")
    def test_concatenates_exons(self, mock_get: MagicMock, resolver: GeneResolver) -> None:
        mock_get.side_effect = [
            _mock_lookup_response(),
            _mock_sequence_response("AAA"),
            _mock_sequence_response("CCC"),
            _mock_sequence_response("GGG"),
        ]
        seq = resolver.get_gene_region_sequence("TP53", region="exons")
        assert seq == "AAACCCGGG"

    @patch("embpy.resources.gene_resolver.requests.get")
    def test_concatenates_introns(self, mock_get: MagicMock, resolver: GeneResolver) -> None:
        mock_get.side_effect = [
            _mock_lookup_response(),
            _mock_sequence_response("TTT"),
            _mock_sequence_response("AAA"),
        ]
        seq = resolver.get_gene_region_sequence("TP53", region="introns")
        assert seq == "TTTAAA"

    @patch("embpy.resources.gene_resolver.requests.get")
    def test_returns_none_on_failure(self, mock_get: MagicMock, resolver: GeneResolver) -> None:
        import requests

        mock_get.side_effect = requests.RequestException("Connection error")
        seq = resolver.get_gene_region_sequence("TP53", region="exons")
        assert seq is None


class TestFetchRegionSequence:
    @patch("embpy.resources.gene_resolver.requests.get")
    def test_builds_correct_url(self, mock_get: MagicMock, resolver: GeneResolver) -> None:
        mock_get.return_value = _mock_sequence_response("ACGT")
        seq = resolver._fetch_region_sequence("17", 100, 200, 1)
        assert seq == "ACGT"
        url = mock_get.call_args[0][0]
        assert "17:100..200:1" in url

    @patch("embpy.resources.gene_resolver.requests.get")
    def test_returns_none_on_error(self, mock_get: MagicMock, resolver: GeneResolver) -> None:
        import requests

        mock_get.side_effect = requests.RequestException("timeout")
        seq = resolver._fetch_region_sequence("17", 100, 200, 1)
        assert seq is None
