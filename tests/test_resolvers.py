"""Comprehensive unit tests for resilient gene resolution, timeouts, retries, and offline fallbacks."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, call, patch

import pytest
import requests

from embpy.errors import GeneResolutionError
from embpy.mcp.tools import map_gene_identifiers, resolve_identifiers
from embpy.resources.gene.resolver import (
    GeneResolver,
    _ensembl_get,
    _ensembl_post,
    _get_ensembl_offline_warning,
)


def _make_response(status: int, json_data: dict | None = None, text: str = "", headers: dict | None = None) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp.url = "https://rest.ensembl.org/mock"
    if json_data is not None:
        import json
        resp._content = json.dumps(json_data).encode("utf-8")
        resp.headers["Content-Type"] = "application/json"
    else:
        resp._content = text.encode("utf-8")
    if headers:
        resp.headers.update(headers)
    return resp


@pytest.fixture
def temp_resolver(tmp_path, monkeypatch):
    """Clean GeneResolver instance using a temporary directory cache and no local pyensembl."""
    monkeypatch.setenv("EMBPY_CACHE", str(tmp_path))
    resolver = GeneResolver.__new__(GeneResolver)
    resolver.mart_file = None
    resolver.chrom_folder = None
    resolver.release_version = 110
    resolver.species = "human"
    resolver.ensembl = None
    resolver._genome_fasta = None
    resolver._genome_dir = None
    resolver._cache_file = tmp_path / "gene_id_cache.json"
    resolver._symbol_cache = {}
    resolver._ensembl_cache = {}
    resolver._last_diagnostics = None
    return resolver


class TestEnsemblResilience:
    """Test explicit timeouts, retries with backoff, and chunked requests."""

    def test_ensembl_post_success(self):
        with patch("embpy.resources.gene.resolver.requests.post") as mock_post:
            mock_post.return_value = _make_response(200, {"TP53": {"id": "ENSG00000141510"}})
            resp = _ensembl_post(
                "https://rest.ensembl.org/lookup/symbol/homo_sapiens",
                json_data={"symbols": ["TP53"]},
                timeout=10,
            )
            assert resp.status_code == 200
            assert resp.json() == {"TP53": {"id": "ENSG00000141510"}}
            mock_post.assert_called_once()
            _, kwargs = mock_post.call_args
            assert kwargs.get("timeout") == 10

    def test_ensembl_post_retries_on_504_gateway_timeout(self):
        with patch("embpy.resources.gene.resolver.requests.post") as mock_post, \
             patch("embpy.retry.time.sleep") as mock_sleep:
            mock_post.side_effect = [
                _make_response(504),
                _make_response(504),
                _make_response(200, {"EGFR": {"id": "ENSG00000146648"}}),
            ]
            resp = _ensembl_post(
                "https://rest.ensembl.org/lookup/symbol/homo_sapiens",
                json_data={"symbols": ["EGFR"]},
                max_retries=3,
                base_delay=0.1,
            )
            assert resp.status_code == 200
            assert resp.json()["EGFR"]["id"] == "ENSG00000146648"
            assert mock_post.call_count == 3
            assert mock_sleep.call_count == 2

    def test_ensembl_post_retries_on_connection_timeout(self):
        with patch("embpy.resources.gene.resolver.requests.post") as mock_post, \
             patch("embpy.retry.time.sleep") as mock_sleep:
            mock_post.side_effect = [
                requests.exceptions.Timeout("Connection timed out"),
                _make_response(200, {"BRCA1": {"id": "ENSG00000012048"}}),
            ]
            resp = _ensembl_post(
                "https://rest.ensembl.org/lookup/symbol/homo_sapiens",
                json_data={"symbols": ["BRCA1"]},
                max_retries=3,
                base_delay=0.1,
            )
            assert resp.status_code == 200
            assert resp.json()["BRCA1"]["id"] == "ENSG00000012048"
            assert mock_post.call_count == 2
            assert mock_sleep.call_count == 1

    def test_ensembl_post_respects_retry_after(self):
        with patch("embpy.resources.gene.resolver.requests.post") as mock_post, \
             patch("embpy.retry.time.sleep") as mock_sleep:
            mock_post.side_effect = [
                _make_response(429, headers={"Retry-After": "2.0"}),
                _make_response(200, {"CD4": {"id": "ENSG00000010610"}}),
            ]
            resp = _ensembl_post(
                "https://rest.ensembl.org/lookup/symbol/homo_sapiens",
                json_data={"symbols": ["CD4"]},
                max_retries=3,
            )
            assert resp.status_code == 200
            assert mock_sleep.call_count == 1
            waited = mock_sleep.call_args[0][0]
            assert waited >= 2.0  # Respects Retry-After header

    def test_chunked_batch_lookup_splits_over_1000(self, temp_resolver):
        symbols = [f"GENE_{i}" for i in range(2500)]
        with patch("embpy.resources.gene.resolver._ensembl_post") as mock_post:
            def side_effect(url, json_data, **kwargs):
                sym_list = json_data.get("symbols", [])
                return _make_response(200, {s: {"id": f"ENSG_{s}"} for s in sym_list})

            mock_post.side_effect = side_effect
            res = temp_resolver.symbols_to_ensembl_batch(symbols, chunk_size=1000)

            assert len(res) == 2500
            assert res["GENE_0"] == "ENSG_GENE_0"
            assert res["GENE_2499"] == "ENSG_GENE_2499"
            assert mock_post.call_count == 3  # Chunks: 1000, 1000, 500


class TestFallbackAndDiagnostics:
    """Test graceful offline fallback, diagnostics reporting, and strict error raising."""

    def test_fallback_to_unmapped_on_complete_network_failure(self, temp_resolver, caplog):
        with patch("embpy.resources.gene.resolver._ensembl_post") as mock_post, \
             patch("embpy.resources.gene.resolver.requests.get") as mock_get:
            mock_post.side_effect = requests.exceptions.ConnectionError("Network disconnected")
            mock_get.side_effect = requests.exceptions.ConnectionError("Network disconnected")

            genes = ["UNMAPPED_1", "UNMAPPED_2"]
            result, diag = temp_resolver.symbols_to_ensembl_batch(
                genes,
                fallback_to_unmapped=True,
                return_diagnostics=True,
            )

            # Original gene symbols are retained
            assert result["UNMAPPED_1"] == "UNMAPPED_1"
            assert result["UNMAPPED_2"] == "UNMAPPED_2"
            assert diag["status"] == "error"
            assert diag["unresolved_count"] == 2
            assert diag["mapping_rate"] == 0.0
            assert "pip install \"embpy[genome]\"" in diag["action_required"]
            assert "pyensembl install" in diag["action_required"]

            # Informative warning logged
            assert any("Ensembl REST API" in record.message for record in caplog.records)

    def test_strict_mode_raises_gene_resolution_error(self, temp_resolver):
        with patch("embpy.resources.gene.resolver._ensembl_post") as mock_post, \
             patch("embpy.resources.gene.resolver.requests.get") as mock_get:
            mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
            mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

            with pytest.raises(GeneResolutionError) as exc_info:
                temp_resolver.symbols_to_ensembl_batch(["FATAL_GENE"], strict=True)

            err = exc_info.value
            assert "FATAL_GENE" in err.failed_genes
            assert err.exit_code == 15
            assert "pip install \"embpy[genome]\"" in str(err)

    def test_offline_warning_message_content(self):
        warning = _get_ensembl_offline_warning("homo_sapiens", 110, reason="timed out")
        assert "Warning: Ensembl REST API timed out." in warning
        assert 'pip install "embpy[genome]"' in warning
        assert "pyensembl install --release 110 --species homo_sapiens" in warning


class TestLocalBackendAndCaching:
    """Test local pyensembl bypass and persistent session caching."""

    def test_pyensembl_backend_bypasses_network(self, temp_resolver):
        mock_ensembl = MagicMock()
        mock_gene = MagicMock()
        mock_gene.gene_id = "ENSG00000141510"
        mock_gene.gene_name = "TP53"
        mock_ensembl.genes_by_name.return_value = [mock_gene]
        mock_ensembl.gene_by_id.return_value = mock_gene
        temp_resolver.ensembl = mock_ensembl

        with patch("embpy.resources.gene.resolver.requests.post") as mock_post, \
             patch("embpy.resources.gene.resolver.requests.get") as mock_get:
            res_sym = temp_resolver.symbols_to_ensembl_batch(["TP53"])
            assert res_sym["TP53"] == "ENSG00000141510"

            res_id = temp_resolver.ensembl_to_symbols_batch(["ENSG00000141510"])
            assert res_id["ENSG00000141510"] == "TP53"

            # No network calls made
            mock_post.assert_not_called()
            mock_get.assert_not_called()

    def test_session_cache_prevents_duplicate_network_calls(self, temp_resolver):
        with patch("embpy.resources.gene.resolver._ensembl_post") as mock_post:
            mock_post.return_value = _make_response(200, {"MYC": {"id": "ENSG00000136997"}})

            # First call populates cache
            res1 = temp_resolver.symbols_to_ensembl_batch(["MYC"])
            assert res1["MYC"] == "ENSG00000136997"
            assert mock_post.call_count == 1

            # Second call hits cache without network
            res2 = temp_resolver.symbols_to_ensembl_batch(["MYC"])
            assert res2["MYC"] == "ENSG00000136997"
            assert mock_post.call_count == 1  # Still 1


class TestMCPToolIntegration:
    """Test resolve_identifiers tool under network failure and partial success scenarios."""

    def test_resolve_identifiers_partial_success_on_timeout(self):
        with patch("embpy.resources.gene.resolver.requests.get") as mock_get, \
             patch("embpy.resources.gene.resolver._ensembl_post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("Ensembl gateway timeout")
            mock_get.side_effect = requests.exceptions.Timeout("MyGene timeout")

            res = resolve_identifiers(identifiers=["TP53", "UNKNOWN_XYZ"], entity_type="gene")

            # Must return structured payload with partial_success or error handling
            assert res["status"] in ("partial_success", "success")
            assert "unresolved_count" in res
            assert "mapping_rate" in res
            assert "warning" in res or "action_required" in res

    def test_resolve_identifiers_returns_structured_payload_on_network_crash(self):
        with patch("embpy.resources.gene.resolver.GeneResolver.symbols_to_ensembl_batch") as mock_batch:
            mock_batch.side_effect = GeneResolutionError(
                "Ensembl connection failed",
                failed_genes=["GENE1", "GENE2"],
            )
            res = map_gene_identifiers(identifiers=["GENE1", "GENE2"], entity_type="gene")

            assert res["status"] == "partial_success"
            assert res["mapping_rate"] == 0.0
            assert res["unresolved_count"] == 2
            assert "Ensembl REST API timed out on 2 genes" in res["warning"]
            assert "Install embpy[genome]" in res["action_required"]
