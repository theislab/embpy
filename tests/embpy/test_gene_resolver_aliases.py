"""Tests for the resolve_symbol alias chain.

We patch ``requests.get`` (used by every chain step that hits an HTTP
endpoint) so the test is hermetic. We assert:

* Step order: pyensembl -> HGNC -> Ensembl REST retry -> MyGene.
* Successful AARS -> AARS1 mapping via HGNC.
* Negative results are cached so a second call never re-issues the
  network requests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")

from embpy.resources.gene._alias_resolver import (  # noqa: E402
    AliasCache,
    Resolution,
    resolve_symbol_chain,
)


@dataclass
class _FakeResponse:
    status_code: int
    payload: dict | None = None

    def json(self) -> dict:
        if self.payload is None:
            raise ValueError("no body")
        return self.payload

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


@pytest.fixture
def cache(tmp_path: Path) -> AliasCache:
    return AliasCache(path=tmp_path / "symbol_resolution.json")


def _make_hgnc_alias_response(symbol: str) -> _FakeResponse:
    return _FakeResponse(
        status_code=200,
        payload={"response": {"docs": [{"symbol": symbol}]}},
    )


def _make_hgnc_miss_response() -> _FakeResponse:
    return _FakeResponse(status_code=200, payload={"response": {"docs": []}})


def _make_ensembl_ok(symbol: str) -> _FakeResponse:
    return _FakeResponse(status_code=200, payload={"display_name": symbol})


def _make_400() -> _FakeResponse:
    return _FakeResponse(status_code=400, payload={"error": "bad request"})


def test_aars_resolves_via_hgnc(monkeypatch, cache: AliasCache) -> None:
    calls: list[str] = []

    def _get(url, **kwargs):
        calls.append(url)
        if url.startswith("https://rest.genenames.org/fetch/symbol/AARS"):
            # The "fetch" endpoint returns no docs for an alias, so the
            # chain falls back to alias_symbol.
            return _make_hgnc_miss_response()
        if url.startswith(
            "https://rest.genenames.org/search/alias_symbol/AARS"
        ):
            return _make_hgnc_alias_response("AARS1")
        if url.startswith(
            "https://rest.ensembl.org/lookup/symbol/human/AARS1"
        ):
            return _make_ensembl_ok("AARS1")
        if url.startswith("https://mygene.info/v3/query"):
            return _FakeResponse(status_code=200, payload={"hits": []})
        return _make_400()

    monkeypatch.setattr(requests, "get", _get)
    res = resolve_symbol_chain(
        "AARS", organism="human", ensembl=None, cache=cache,
    )
    assert isinstance(res, Resolution)
    assert res.approved_symbol == "AARS1"
    # pyensembl missing (None), HGNC fetch miss, HGNC alias hit, Ensembl confirm.
    # We do not assert the exact chain length to keep this resilient
    # to future telemetry tweaks; we only assert ordering.
    steps = [step for step, _ in res.chain]
    assert steps[0] == "pyensembl"
    assert "hgnc" in steps
    assert "ensembl_retry" in steps
    # We made network calls.
    assert any("genenames.org" in u for u in calls)


def test_chain_falls_back_to_mygene(monkeypatch, cache: AliasCache) -> None:
    def _get(url, **kwargs):
        if "genenames.org" in url or "ensembl" in url:
            return _make_400()
        if "mygene.info" in url:
            return _FakeResponse(
                status_code=200,
                payload={"hits": [{"symbol": "FOO", "taxid": 9606}]},
            )
        return _make_400()

    monkeypatch.setattr(requests, "get", _get)
    res = resolve_symbol_chain(
        "anything", organism="human", ensembl=None, cache=cache,
    )
    assert res.approved_symbol == "FOO"
    assert res.source == "mygene"


def test_negative_result_is_cached(monkeypatch, cache: AliasCache) -> None:
    n_calls = {"count": 0}

    def _get(url, **kwargs):
        n_calls["count"] += 1
        return _make_400()

    monkeypatch.setattr(requests, "get", _get)
    first = resolve_symbol_chain(
        "zzz_not_a_gene", organism="human", ensembl=None, cache=cache,
    )
    assert first.approved_symbol is None
    after_first = n_calls["count"]
    assert after_first > 0

    # Second call must hit the cache only -- zero network requests.
    second = resolve_symbol_chain(
        "zzz_not_a_gene", organism="human", ensembl=None, cache=cache,
    )
    assert second.approved_symbol is None
    assert n_calls["count"] == after_first


def test_positive_result_is_cached(monkeypatch, cache: AliasCache) -> None:
    n_calls = {"count": 0}

    def _get(url, **kwargs):
        n_calls["count"] += 1
        if "rest.genenames.org/fetch/symbol" in url:
            return _make_hgnc_alias_response("AARS1")
        if "rest.ensembl.org/lookup/symbol/human/AARS1" in url:
            return _make_ensembl_ok("AARS1")
        return _make_400()

    monkeypatch.setattr(requests, "get", _get)
    first = resolve_symbol_chain(
        "AARS", organism="human", ensembl=None, cache=cache,
    )
    assert first.approved_symbol == "AARS1"
    after_first = n_calls["count"]
    resolve_symbol_chain(
        "AARS", organism="human", ensembl=None, cache=cache,
    )
    assert n_calls["count"] == after_first


def test_pyensembl_step_wins_when_available(monkeypatch, cache: AliasCache) -> None:
    class _FakeGene:
        gene_name = "TP53"

    class _FakeRelease:
        def genes_by_name(self, sym):
            return [_FakeGene()]

    n_calls = {"count": 0}

    def _get(url, **kwargs):
        n_calls["count"] += 1
        return _make_400()

    monkeypatch.setattr(requests, "get", _get)
    res = resolve_symbol_chain(
        "TP53", organism="human", ensembl=_FakeRelease(), cache=cache,
    )
    assert res.approved_symbol == "TP53"
    assert res.source == "pyensembl"
    # No HTTP calls were issued.
    assert n_calls["count"] == 0


def test_resolve_symbol_without_cache_resolves_again(monkeypatch, tmp_path: Path) -> None:
    """use_cache=False ignores the cached answer; the fresh answer then replaces it."""
    from embpy.resources.gene.resolver import GeneResolver

    monkeypatch.setenv("EMBPY_CACHE", str(tmp_path))
    stale = {"input_symbol": "OLDNAME", "organism": "human", "approved_symbol": "STALE1", "source": "hgnc", "chain": []}
    (tmp_path / "symbol_resolution.json").write_text(
        json.dumps({"version": 1, "entries": {"human::OLDNAME": stale}}), encoding="utf-8"
    )

    class _Release:
        calls = 0

        def genes_by_name(self, sym):
            _Release.calls += 1
            return [type("Gene", (), {"gene_name": "FRESH1"})()]

    def _no_network(url, **kwargs):
        raise AssertionError(f"network access in a test: {url}")

    monkeypatch.setattr(requests, "get", _no_network)
    resolver = GeneResolver.__new__(GeneResolver)
    resolver.ensembl = _Release()

    assert resolver.resolve_symbol("OLDNAME") == "STALE1"  # use_cache=True answers from the cache
    assert _Release.calls == 0
    assert resolver.resolve_symbol("OLDNAME", use_cache=False) == "FRESH1"
    assert _Release.calls == 1
    assert resolver.resolve_symbol("OLDNAME") == "FRESH1"
    assert _Release.calls == 1

    resolver._alias_cache.flush()
    saved = json.loads((tmp_path / "symbol_resolution.json").read_text(encoding="utf-8"))
    assert saved["entries"]["human::OLDNAME"]["approved_symbol"] == "FRESH1"


def _offline(url, **kwargs):
    raise requests.ConnectionError(f"offline: {url}")


def test_offline_refresh_keeps_the_cached_answer(monkeypatch, tmp_path: Path) -> None:
    """A fresh resolution that only met service failures does not replace a good answer."""
    from embpy.resources.gene.resolver import GeneResolver

    monkeypatch.setenv("EMBPY_CACHE", str(tmp_path))
    good = {
        "input_symbol": "AARS",
        "organism": "human",
        "approved_symbol": "AARS1",
        "source": "hgnc",
        "chain": [["pyensembl", "no_pyensembl"], ["hgnc", "ok"]],
    }
    cache_file = tmp_path / "symbol_resolution.json"
    cache_file.write_text(json.dumps({"version": 1, "entries": {"human::AARS": good}}), encoding="utf-8")
    monkeypatch.setattr(requests, "get", _offline)  # HGNC, Ensembl and MyGene.info all unreachable
    resolver = GeneResolver.__new__(GeneResolver)
    resolver.ensembl = None  # no pyensembl

    assert resolver.resolve_symbol("AARS", use_cache=False) is None  # the fresh attempt failed
    assert resolver.resolve_symbol("AARS") == "AARS1"
    resolver._alias_cache.flush()
    assert json.loads(cache_file.read_text(encoding="utf-8"))["entries"]["human::AARS"] == good


def test_offline_miss_is_not_cached(monkeypatch, cache: AliasCache) -> None:
    """Only service failures: nothing is stored, so the next call asks again."""
    monkeypatch.setattr(requests, "get", _offline)
    res = resolve_symbol_chain("NEWNAME", organism="human", ensembl=None, cache=cache)
    assert res.approved_symbol is None
    assert cache.get("NEWNAME", "human") is None
