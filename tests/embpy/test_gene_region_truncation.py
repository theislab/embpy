"""A failed exon fetch must not silently shorten the gene.

``_fetch_region_sequence`` catches ``requests.RequestException``, logs a warning
and returns ``None``. ``get_gene_regions`` used to append only the regions that
came back, so a transient Ensembl timeout returned a *truncated transcript* with
no error at all -- observed in the wild as TUBB coming back as 2 exons / 321 bp
instead of 4 exons / ~2,500 bp. The resulting embedding has the right shape, a
plausible norm and sensible-looking neighbours, and is wrong.

Failing loudly is the only safe behaviour: a caller can retry a ``None``, but it
cannot detect a short sequence.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from embpy.resources.gene_resolver import GeneResolver

_LOOKUP = {
    "id": "ENSG00000196230",
    "display_name": "TUBB",
    "strand": 1,
    "seq_region_name": "6",
    "Transcript": [
        {
            "id": "ENST00000327892",
            "is_canonical": 1,
            "Exon": [
                {"id": "e1", "start": 100, "end": 199},
                {"id": "e2", "start": 300, "end": 399},
                {"id": "e3", "start": 500, "end": 599},
                {"id": "e4", "start": 700, "end": 799},
            ],
        }
    ],
}


def _lookup_response():
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status = MagicMock()
    mock.json.return_value = _LOOKUP
    return mock


def _sequence_response(seq: str):
    mock = MagicMock()
    mock.status_code = 200
    mock.raise_for_status = MagicMock()
    mock.text = seq
    return mock


def _persistent(script):
    """``requests.get`` side effect where a failing request keeps failing.

    Ensembl requests are retried, so a timed-out exon is requested again; a
    real outage fails every attempt. Responses in ``script`` are served in
    order, except that a URL that raised once raises again on each retry.
    """
    queue = list(script)
    failing: dict[str, BaseException] = {}

    def side_effect(url, *args, **kwargs):
        if url in failing:
            raise failing[url]
        item = queue.pop(0)
        if isinstance(item, BaseException):
            failing[url] = item
            raise item
        return item

    return side_effect


@pytest.fixture(autouse=True)
def _no_backoff_sleep():
    """Retries back off with ``time.sleep``; skip the waiting in these tests."""
    with patch("embpy.retry.time.sleep"):
        yield


@pytest.fixture()
def resolver():
    return GeneResolver(auto_download=False)


class TestExonFetchFailure:
    @patch("embpy.resources.gene_resolver.requests.get")
    def test_timeout_on_one_exon_returns_none_not_a_short_gene(
        self, mock_get: MagicMock, resolver: GeneResolver
    ) -> None:
        mock_get.side_effect = _persistent([
            _lookup_response(),
            _sequence_response("AAAA"),
            requests.exceptions.ReadTimeout("read timed out"),
            _sequence_response("CCCC"),
            _sequence_response("GGGG"),
        ])
        regions = resolver.get_gene_regions("TUBB", region="exons")
        assert regions is None, (
            "a timed-out exon was dropped and the remaining exons returned as if "
            "they were the whole gene"
        )

    @patch("embpy.resources.gene_resolver.requests.get")
    def test_concatenated_sequence_is_none_rather_than_truncated(
        self, mock_get: MagicMock, resolver: GeneResolver
    ) -> None:
        mock_get.side_effect = _persistent([
            _lookup_response(),
            _sequence_response("AAAA"),
            _sequence_response("TTTT"),
            requests.exceptions.ReadTimeout("read timed out"),
            _sequence_response("GGGG"),
        ])
        seq = resolver.get_gene_region_sequence("TUBB", region="exons")
        assert seq is None
        # The specific regression: 8 bp of a 16 bp transcript looked like success.
        assert seq != "AAAATTTT"

    @patch("embpy.resources.gene_resolver.requests.get")
    def test_complete_fetch_still_succeeds(
        self, mock_get: MagicMock, resolver: GeneResolver
    ) -> None:
        """The guard must not reject a healthy fetch."""
        mock_get.side_effect = [
            _lookup_response(),
            _sequence_response("AAAA"),
            _sequence_response("TTTT"),
            _sequence_response("CCCC"),
            _sequence_response("GGGG"),
        ]
        regions = resolver.get_gene_regions("TUBB", region="exons")
        assert regions is not None
        assert len(regions) == 4
        assert "".join(str(r["sequence"]) for r in regions) == "AAAATTTTCCCCGGGG"

    @patch("embpy.resources.gene_resolver.requests.get")
    def test_intron_fetch_failure_also_returns_none(
        self, mock_get: MagicMock, resolver: GeneResolver
    ) -> None:
        mock_get.side_effect = _persistent([
            _lookup_response(),
            _sequence_response("AAAA"),
            requests.exceptions.ReadTimeout("read timed out"),
            _sequence_response("CCCC"),
        ])
        assert resolver.get_gene_regions("TUBB", region="introns") is None

    @patch("embpy.resources.gene_resolver.requests.get")
    def test_failure_is_logged_at_error_level(
        self, mock_get: MagicMock, resolver: GeneResolver, caplog
    ) -> None:
        """A warning nothing reads was the reason this went unnoticed for so long."""
        mock_get.side_effect = _persistent([
            _lookup_response(),
            requests.exceptions.ReadTimeout("read timed out"),
            _sequence_response("TTTT"),
            _sequence_response("CCCC"),
            _sequence_response("GGGG"),
        ])
        with caplog.at_level("ERROR"):
            resolver.get_gene_regions("TUBB", region="exons")
        assert any("Incomplete exon set" in r.message for r in caplog.records)
