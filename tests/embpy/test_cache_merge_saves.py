"""Saving a gene cache merges into the file: no writer erases entries another writer saved.

Every test keeps its caches in the per-test ``EMBPY_CACHE`` folder and fakes the
network, so nothing here touches the user's cache or needs a connection.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from embpy.resources.gene import _alias_resolver as alias_mod
from embpy.resources.gene import _cache_files
from embpy.resources.gene import resolver as resolver_mod
from embpy.resources.gene.resolver import GeneResolver, default_gene_cache_path
from embpy.resources.morphology import hpa


def _fake_id(symbol: str) -> str:
    return "ENSG9" + format(zlib.crc32(symbol.encode()), "010d")


def _fake_lookup(symbol, organism, **kwargs):
    eid = _fake_id(symbol)
    return {"status": "ok", "symbol": symbol, "ensembl_id": eid, "ensembl_ids": [eid], "candidates": [symbol]}


@pytest.fixture
def offline_lookups(monkeypatch):
    """Symbols resolve through a fake MyGene.info; Ensembl's bulk lookup finds nothing; no network."""
    monkeypatch.setitem(sys.modules, "pyensembl", None)
    monkeypatch.setitem(sys.modules, "mygene", None)
    nothing_found = MagicMock(ok=True, status_code=200)
    nothing_found.json.return_value = {}
    no_network = AssertionError("network access in a test")
    with (
        patch.object(resolver_mod, "_mygene_exact_symbol_lookup", side_effect=_fake_lookup),
        patch.object(resolver_mod, "_ensembl_post", return_value=nothing_found),
        patch.object(resolver_mod.requests, "get", side_effect=no_network),
        patch.object(resolver_mod.requests, "post", side_effect=no_network),
    ):
        yield


def _saved_symbols() -> dict[str, str]:
    data = json.loads(default_gene_cache_path().read_text(encoding="utf-8"))
    return data["symbols"]


def test_shared_resolver_keeps_entries_other_resolvers_saved(offline_lookups):
    shared = resolver_mod._shared_gene_resolver("human")
    shared.symbol_to_ensembl("GENEA")  # waits in the long-lived resolver
    other = GeneResolver(species="human")
    other.symbols_to_ensembl_batch(["GENEB", "GENEC"])  # saved when the batch ends
    del other
    third = GeneResolver(species="human")
    third.symbols_to_ensembl_batch(["GENED"])
    del third
    resolver_mod._save_unsaved_gene_caches()  # what runs at exit
    saved = _saved_symbols()
    assert {f"human:GENE{x}": _fake_id(f"GENE{x}") for x in "ABCD"}.items() <= saved.items()


def test_thread_pool_lookups_keep_every_entry(offline_lookups):
    genes = [f"GENE{i}" for i in range(40)]
    ready = threading.Barrier(4)

    def work(gene):
        if gene in genes[:4]:
            ready.wait(timeout=60)  # every worker has its own resolver before any of them saves
        return hpa._resolve_ensembl_id_quiet(gene)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(work, genes))
    assert all(eid == _fake_id(g) for g, (eid, _) in zip(genes, results, strict=True))
    gc.collect()
    resolver_mod._save_unsaved_gene_caches()
    saved = _saved_symbols()
    missing = sorted(set(genes) - {k.split(":", 1)[1] for k in saved})
    assert missing == []


class _Release:
    def genes_by_name(self, name):
        return [type("Gene", (), {"gene_name": name + "_APPROVED"})()]


def test_long_lived_alias_cache_keeps_entries_other_caches_saved():
    long_lived = alias_mod.AliasCache(path=alias_mod.default_cache_path())  # like GeneResolver._alias_cache
    alias_mod.resolve_symbol_chain("X1", ensembl=_Release(), cache=long_lived)  # waits in the long-lived cache
    alias_mod.resolve_symbol_chain("Y1", ensembl=_Release())  # its own cache, saved on return
    alias_mod.resolve_symbol_chain("Z1", ensembl=_Release())
    alias_mod._save_unsaved_alias_caches()  # what runs at exit
    entries = json.loads(alias_mod.default_cache_path().read_text(encoding="utf-8"))["entries"]
    assert {"human::X1", "human::Y1", "human::Z1"} <= entries.keys()


def test_save_brings_in_entries_other_resolvers_saved(offline_lookups):
    first = GeneResolver(species="human")
    second = GeneResolver(species="human")
    first.symbol_to_ensembl("GENEA")
    first.flush_cache()
    second.symbol_to_ensembl("GENEB")
    second.flush_cache()
    # The second save read GENEA from the file, so this resolver now knows it without a lookup.
    with patch.object(resolver_mod, "_mygene_exact_symbol_lookup", side_effect=AssertionError("looked up again")):
        assert second.symbol_to_ensembl("GENEA") == _fake_id("GENEA")


def _lock_file() -> Path:
    cache = default_gene_cache_path()
    return cache.with_name(cache.name + ".lock")


def test_save_holds_a_lock_file_next_to_the_cache(offline_lookups):
    resolver = GeneResolver(species="human")
    resolver.symbol_to_ensembl("GENEA")
    held_while_writing = []
    real_replace = os.replace

    def _replace(src, dst, *args, **kwargs):
        if Path(dst) == default_gene_cache_path():
            held_while_writing.append(_lock_file().exists())
        return real_replace(src, dst, *args, **kwargs)

    with patch.object(os, "replace", _replace):
        resolver.flush_cache()
    assert held_while_writing == [True]
    assert not _lock_file().exists()


def test_save_takes_over_a_lock_left_by_a_killed_process(offline_lookups, caplog):
    resolver = GeneResolver(species="human")
    resolver.symbol_to_ensembl("GENEA")
    lock = _lock_file()
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("12345", encoding="utf-8")
    old = time.time() - 120
    os.utime(lock, (old, old))
    with caplog.at_level(logging.WARNING):
        resolver.flush_cache()
    assert "human:GENEA" in _saved_symbols()
    assert not lock.exists()
    assert not [r for r in caplog.records if "lock" in r.getMessage().lower()]


def test_save_goes_ahead_without_a_lock_another_process_keeps(offline_lookups, caplog, monkeypatch):
    monkeypatch.setattr(_cache_files, "LOCK_TIMEOUT", 0.3)
    resolver = GeneResolver(species="human")
    resolver.symbol_to_ensembl("GENEA")
    lock = _lock_file()
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("12345", encoding="utf-8")  # fresh: another process is saving
    with caplog.at_level(logging.WARNING):
        resolver.flush_cache()
    assert "human:GENEA" in _saved_symbols()
    assert lock.exists()  # not ours to remove
    assert any(r.levelno == logging.WARNING and str(lock) in r.getMessage() for r in caplog.records)
