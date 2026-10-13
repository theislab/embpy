"""Tests for retrying transient Ensembl REST failures in the gene resolver.

``requests.get`` and ``time.sleep`` are mocked: no network, no real waiting.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import requests

from embpy.resources.gene.resolver import GeneResolver, _ensembl_get

URL = "https://rest.ensembl.org/lookup/symbol/human/EGFR?expand=1"


def _response(status: int, body: bytes = b"{}", headers: dict | None = None) -> requests.Response:
    """A real ``requests.Response`` so ``raise_for_status`` / ``ok`` behave."""
    resp = requests.Response()
    resp.status_code = status
    resp.reason = "mock"
    resp.url = URL
    resp._content = body
    if headers:
        resp.headers.update(headers)
    return resp


@pytest.fixture
def mock_get():
    with patch("embpy.resources.gene.resolver.requests.get") as m:
        yield m


@pytest.fixture
def sleeps():
    """Capture backoff waits instead of sleeping (``retry_with_backoff`` calls ``time.sleep``)."""
    with patch("embpy.retry.time.sleep") as m:
        yield m


def _waits(sleeps) -> list[float]:
    return [c.args[0] for c in sleeps.call_args_list]


class TestEnsemblGetRetry:
    def test_two_500s_then_success(self, mock_get, sleeps):
        mock_get.side_effect = [_response(500), _response(500), _response(200, b'{"id": "ENSG1"}')]

        resp = _ensembl_get(URL)

        assert resp.status_code == 200
        assert resp.json() == {"id": "ENSG1"}
        assert mock_get.call_count == 3
        assert _waits(sleeps) == [1.0, 2.0]

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_persistent_5xx_gives_up_and_returns_last_response(self, mock_get, sleeps, status):
        mock_get.side_effect = lambda *a, **k: _response(status)

        resp = _ensembl_get(URL)

        # Same error path as before the retries: the caller gets the failing
        # response back and its own ``raise_for_status()`` raises.
        assert resp.status_code == status
        with pytest.raises(requests.HTTPError):
            resp.raise_for_status()
        assert mock_get.call_count == 5
        assert _waits(sleeps) == [1.0, 2.0, 4.0, 8.0]

    def test_max_retries_is_total_attempts(self, mock_get, sleeps):
        mock_get.side_effect = lambda *a, **k: _response(500)

        resp = _ensembl_get(URL, max_retries=2, base_delay=0.25)

        assert resp.status_code == 500
        assert mock_get.call_count == 2
        assert _waits(sleeps) == [0.25]

    @pytest.mark.parametrize("status", [400, 404])
    def test_permanent_4xx_is_not_retried(self, mock_get, sleeps, status):
        mock_get.return_value = _response(status)

        resp = _ensembl_get(URL)

        assert resp.status_code == status
        assert mock_get.call_count == 1
        sleeps.assert_not_called()

    def test_success_is_not_retried(self, mock_get, sleeps):
        mock_get.return_value = _response(200)

        assert _ensembl_get(URL).status_code == 200
        assert mock_get.call_count == 1
        sleeps.assert_not_called()

    def test_429_without_retry_after_uses_backoff(self, mock_get, sleeps):
        mock_get.side_effect = [_response(429), _response(429), _response(200)]

        assert _ensembl_get(URL).status_code == 200
        assert _waits(sleeps) == [1.0, 2.0]

    def test_429_honours_retry_after(self, mock_get, sleeps):
        mock_get.side_effect = [_response(429, headers={"Retry-After": "3"}), _response(200)]

        assert _ensembl_get(URL).status_code == 200
        assert _waits(sleeps) == [3.5]

    def test_503_honours_retry_after(self, mock_get, sleeps):
        mock_get.side_effect = [_response(503, headers={"Retry-After": "4"}), _response(200)]

        assert _ensembl_get(URL).status_code == 200
        assert _waits(sleeps) == [4.5]

    def test_retry_after_is_capped(self, mock_get, sleeps):
        mock_get.side_effect = [_response(429, headers={"Retry-After": "86400"}), _response(200)]

        assert _ensembl_get(URL).status_code == 200
        assert _waits(sleeps) == [60.0]

    def test_retry_after_http_date_falls_back_to_backoff(self, mock_get, sleeps):
        date = "Wed, 21 Oct 2026 07:28:00 GMT"
        mock_get.side_effect = [_response(503, headers={"Retry-After": date}), _response(200)]

        assert _ensembl_get(URL).status_code == 200
        assert _waits(sleeps) == [1.0]

    @pytest.mark.parametrize("exc", [requests.ConnectionError("reset"), requests.Timeout("slow")])
    def test_connection_error_or_timeout_then_success(self, mock_get, sleeps, exc):
        mock_get.side_effect = [exc, _response(200)]

        assert _ensembl_get(URL).status_code == 200
        assert mock_get.call_count == 2
        assert _waits(sleeps) == [1.0]

    @pytest.mark.parametrize("exc", [requests.ConnectionError("reset"), requests.Timeout("slow")])
    def test_persistent_connection_error_is_reraised(self, mock_get, sleeps, exc):
        mock_get.side_effect = exc

        with pytest.raises(type(exc)):
            _ensembl_get(URL)
        assert mock_get.call_count == 5
        assert len(_waits(sleeps)) == 4

    def test_non_transient_request_exception_is_not_retried(self, mock_get, sleeps):
        mock_get.side_effect = requests.exceptions.InvalidURL("bad url")

        with pytest.raises(requests.exceptions.InvalidURL):
            _ensembl_get(URL)
        assert mock_get.call_count == 1
        sleeps.assert_not_called()


@pytest.fixture
def resolver(monkeypatch, tmp_path):
    """GeneResolver in API-only mode (no pyensembl, no local genome)."""
    monkeypatch.setenv("EMBPY_CACHE", str(tmp_path))
    r = GeneResolver.__new__(GeneResolver)
    r.ensembl = None
    r.species = "human"
    r._genome_fasta = None
    r._genome_dir = None
    return r


class TestGetDnaSequenceRetry:
    """The failure mode from the HPC: one Ensembl 500 used to mean ``None``."""

    def test_recovers_from_transient_500s(self, resolver, mock_get, sleeps):
        mock_get.side_effect = [
            _response(500),
            _response(500),
            _response(200, b'{"id": "ENSG00000146648"}'),
            _response(200, b"ACGTACGT\n"),
        ]

        assert resolver.get_dna_sequence("ENSG00000146648", "ensembl_id") == "ACGTACGT"
        assert mock_get.call_count == 4

    def test_persistent_500_returns_none(self, resolver, mock_get, sleeps, caplog):
        mock_get.side_effect = lambda *a, **k: _response(500)

        assert resolver.get_dna_sequence("ENSG00000146648", "ensembl_id") is None
        assert mock_get.call_count == 5
        assert "Error fetching DNA sequence from Ensembl" in caplog.text

    def test_404_returns_none_without_retry(self, resolver, mock_get, sleeps):
        mock_get.return_value = _response(404)

        assert resolver.get_dna_sequence("ENSG00000146648", "ensembl_id") is None
        assert mock_get.call_count == 1
        sleeps.assert_not_called()
