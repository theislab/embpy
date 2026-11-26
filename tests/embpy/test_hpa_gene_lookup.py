"""HPA antibody lookups resolve a gene symbol through embpy's GeneResolver."""

from __future__ import annotations

import logging
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

from embpy.resources.gene import resolver as resolver_mod
from embpy.resources.gene.resolver import GeneResolver
from embpy.resources.morphology import hpa

_FAKE_ID = "ENSG90000000007"


def _fake_symbol_lookup(symbol, organism, **kwargs):
    return {"status": "ok", "symbol": symbol, "ensembl_id": _FAKE_ID, "ensembl_ids": [_FAKE_ID], "candidates": [symbol]}


@pytest.fixture
def mygene_lookup(monkeypatch, tmp_path):
    """GeneResolver's MyGene.info lookup, faked; no other way to resolve a symbol."""
    monkeypatch.setenv("EMBPY_CACHE", str(tmp_path))
    # Without pyensembl (no Ensembl download) and without the mygene fallback package,
    # only GeneResolver's MyGene.info lookup can answer.
    monkeypatch.setitem(sys.modules, "pyensembl", None)
    monkeypatch.setitem(sys.modules, "mygene", None)
    no_network = AssertionError("network access in a test")
    with (
        patch.object(resolver_mod, "_mygene_exact_symbol_lookup", side_effect=_fake_symbol_lookup) as lookup,
        patch.object(resolver_mod.requests, "get", side_effect=no_network),
        patch.object(resolver_mod.requests, "post", side_effect=no_network),
    ):
        yield lookup


@pytest.fixture
def resolver_builds(monkeypatch) -> list[dict]:
    """Keyword arguments of every GeneResolver built during the test."""
    builds: list[dict] = []
    real_init = GeneResolver.__init__

    def counting_init(self, *args, **kwargs):
        builds.append(kwargs)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(GeneResolver, "__init__", counting_init)
    return builds


@pytest.mark.parametrize(
    ("resolve", "expected"),
    [
        (hpa._resolve_ensembl_id, _FAKE_ID),
        (hpa._resolve_ensembl_id_quiet, (_FAKE_ID, "GeneResolver")),
    ],
)
def test_symbol_is_resolved_with_gene_resolver(resolve, expected, mygene_lookup):
    assert resolve("NOTASEEDGENE") == expected
    mygene_lookup.assert_called_once_with("NOTASEEDGENE", "human")


def test_two_lookups_build_one_resolver(mygene_lookup, resolver_builds):
    assert hpa._resolve_ensembl_id("GENEA") == _FAKE_ID
    assert hpa._resolve_ensembl_id_quiet("GENEB") == (_FAKE_ID, "GeneResolver")
    assert resolver_builds == [{"species": "human"}]
    assert mygene_lookup.call_count == 2


def test_each_thread_builds_its_own_resolver(mygene_lookup, resolver_builds):
    # pyensembl's SQLite connection only works in the thread that opened it.
    hpa._resolve_ensembl_id("GENEA")
    worker = threading.Thread(target=hpa._resolve_ensembl_id, args=("GENEB",))
    worker.start()
    worker.join()
    hpa._resolve_ensembl_id("GENEC")
    assert resolver_builds == [{"species": "human"}, {"species": "human"}]
    assert mygene_lookup.call_count == 3


def _mygene_misses(symbol, organism, **kwargs):
    if symbol == "AMBIG1":
        return {"status": "ambiguous", "symbol": None, "ensembl_id": None, "candidates": ["GENEX", "GENEY"]}
    return {"status": "miss", "symbol": None, "ensembl_id": None, "candidates": []}


@pytest.fixture
def unresolvable(mygene_lookup):
    """MyGene.info knows no gene (or several for AMBIG1); Ensembl answers 'not found'."""
    mygene_lookup.side_effect = _mygene_misses
    not_found = MagicMock(ok=False, status_code=400)
    with patch.object(resolver_mod, "_ensembl_get", return_value=not_found):
        yield mygene_lookup


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_quiet_lookup_logs_no_warnings(unresolvable, caplog):
    with caplog.at_level(logging.DEBUG):
        assert hpa._resolve_ensembl_id_quiet("NOTAGENE") == (None, None)
        assert hpa._resolve_ensembl_id_quiet("AMBIG1") == (None, None)
    assert _warnings(caplog) == []


def test_quiet_lookup_does_not_silence_other_threads(unresolvable, caplog):
    inside, done = threading.Event(), threading.Event()

    def slow_misses(symbol, organism, **kwargs):
        if symbol == "QUIETGENE":
            inside.set()  # the quiet lookup is running ...
            done.wait(30)  # ... until the other thread's lookup has finished
        return _mygene_misses(symbol, organism)

    unresolvable.side_effect = slow_misses
    quiet = threading.Thread(target=hpa._resolve_ensembl_id_quiet, args=("QUIETGENE",))
    with caplog.at_level(logging.WARNING):
        quiet.start()
        assert inside.wait(30)
        assert hpa._resolve_ensembl_id("LOUDGENE") is None
        done.set()
        quiet.join(30)
    assert any("LOUDGENE" in m for m in _warnings(caplog))
    assert not any("QUIETGENE" in m for m in _warnings(caplog))
