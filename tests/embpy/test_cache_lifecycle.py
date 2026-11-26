"""How the gene caches create, replace and clean up files, and save at exit or in child processes.

Every test keeps its caches in ``tmp_path`` (or the per-test ``EMBPY_CACHE``) and
fakes the network.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import embpy
from embpy.resources.gene import _alias_resolver as alias_mod
from embpy.resources.gene import resolver as resolver_mod
from embpy.resources.gene._alias_resolver import AliasCache, Resolution
from embpy.resources.gene.resolver import GeneResolver


def _fake_lookup(symbol, organism, **kwargs):
    eid = "ENSG9" + symbol[4:].zfill(10)
    return {"status": "ok", "symbol": symbol, "ensembl_id": eid, "ensembl_ids": [eid], "candidates": [symbol]}


def _resolver_with_a_lookup(cache_file: Path) -> GeneResolver:
    r = GeneResolver.__new__(GeneResolver)
    r.ensembl = None
    r.species = "human"
    r._cache_file = cache_file
    r._ensure_cache()
    with (
        patch.object(resolver_mod, "_mygene_exact_symbol_lookup", side_effect=_fake_lookup),
        patch.object(resolver_mod.requests, "get", side_effect=AssertionError("network access in a test")),
    ):
        r.symbol_to_ensembl("GENE1")
    return r


def _alias_cache_with_an_entry(cache_file: Path) -> AliasCache:
    cache = AliasCache(path=cache_file)
    cache.put(Resolution(input_symbol="ALIAS1", organism="human", approved_symbol="GENE1", source="hgnc"))
    return cache


def _no_umask(*args):
    raise AssertionError("os.umask called: changing the umask races with other threads")


# --- (a) temporary files are created like any new file; no umask reads ----------------------


def test_gene_cache_save_does_not_touch_the_umask(tmp_path, caplog):
    cache_file = tmp_path / "cache" / "gene_id_cache.json"
    resolver = _resolver_with_a_lookup(cache_file)
    with patch.object(os, "umask", side_effect=_no_umask), caplog.at_level(logging.WARNING):
        resolver.flush_cache()
    assert "human:GENE1" in json.loads(cache_file.read_text(encoding="utf-8"))["symbols"]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_alias_cache_save_does_not_touch_the_umask(tmp_path, caplog):
    cache_file = tmp_path / "cache" / "symbol_resolution.json"
    cache = _alias_cache_with_an_entry(cache_file)
    with patch.object(os, "umask", side_effect=_no_umask), caplog.at_level(logging.WARNING):
        cache.flush()
    assert "human::ALIAS1" in json.loads(cache_file.read_text(encoding="utf-8"))["entries"]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_replacing_a_cache_file_keeps_its_mode(tmp_path):
    cache_file = tmp_path / "cache" / "gene_id_cache.json"
    resolver = _resolver_with_a_lookup(cache_file)
    resolver.flush_cache()
    os.chmod(cache_file, 0o600)
    resolver._symbol_cache[("human", "GENE2")] = "ENSG90000000002"
    resolver._save_cache()
    assert stat.S_IMODE(cache_file.stat().st_mode) == 0o600


# --- (b) exit hooks cope with registries changed by other threads meanwhile ------------------


class _ChangingSet(set):
    """A set whose first iteration fails as if another thread had changed it meanwhile."""

    failed = False

    def __iter__(self):
        if not self.failed:
            self.failed = True
            raise RuntimeError("Set changed size during iteration")
        return super().__iter__()


class _ChangingDict(dict):
    """A dict whose first ``values()`` fails as if another thread had changed it meanwhile."""

    failed = False

    def values(self):
        if not self.failed:
            self.failed = True
            raise RuntimeError("dictionary changed size during iteration")
        return super().values()


def test_gene_exit_hook_copes_with_a_registry_changed_meanwhile(tmp_path, monkeypatch):
    cache_file = tmp_path / "cache" / "gene_id_cache.json"
    resolver = _resolver_with_a_lookup(cache_file)
    monkeypatch.setattr(resolver_mod, "_RESOLVERS_WITH_UNSAVED_CACHE", _ChangingSet([resolver]))
    resolver_mod._save_unsaved_gene_caches()
    assert "human:GENE1" in json.loads(cache_file.read_text(encoding="utf-8"))["symbols"]


def test_alias_exit_hook_copes_with_a_registry_changed_meanwhile(tmp_path, monkeypatch):
    monkeypatch.setattr(alias_mod, "_UNSAVED_ALIAS_CACHES", _ChangingDict())
    cache_file = tmp_path / "cache" / "symbol_resolution.json"
    cache = _alias_cache_with_an_entry(cache_file)
    assert alias_mod._UNSAVED_ALIAS_CACHES  # registered by put
    alias_mod._save_unsaved_alias_caches()
    assert "human::ALIAS1" in json.loads(cache_file.read_text(encoding="utf-8"))["entries"]
    del cache


# --- (c) child processes that end without exit hooks save each new entry ----------------------

_CHILD_SCRIPT = """
import multiprocessing
import os
import sys
from pathlib import Path
from unittest.mock import patch

from embpy.resources.gene import resolver as mod

RESOLVER = None


def make(cache_file):
    r = mod.GeneResolver.__new__(mod.GeneResolver)
    r.ensembl = None
    r.species = "human"
    r._cache_file = Path(cache_file)
    r._ensure_cache()
    return r


def lookup(symbol, organism, **kwargs):
    eid = "ENSG9" + symbol[4:].zfill(10)
    return {"status": "ok", "symbol": symbol, "ensembl_id": eid, "ensembl_ids": [eid], "candidates": [symbol]}


def child(cache_file):
    # fork: the resolver the parent built; spawn/forkserver: one built in the child
    r = RESOLVER if RESOLVER is not None else make(cache_file)
    with (
        patch.object(mod, "_mygene_exact_symbol_lookup", side_effect=lookup),
        patch.object(mod.requests, "get", side_effect=AssertionError("network access in a test")),
    ):
        for i in range(3):
            r.symbol_to_ensembl(f"GENE{i}")
    os._exit(0)  # how fork and forkserver children end: no exit hooks run


if __name__ == "__main__":
    method, cache_file = sys.argv[1], sys.argv[2]
    RESOLVER = make(cache_file)
    process = multiprocessing.get_context(method).Process(target=child, args=(cache_file,))
    process.start()
    process.join(240)
    sys.exit(process.exitcode)
"""


@pytest.mark.parametrize("method", ["fork", "forkserver", "spawn"])
def test_lookups_in_a_child_process_are_saved_before_it_ends(tmp_path, method):
    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"start method {method!r} is not available here")
    script = tmp_path / "child_lookups.py"
    script.write_text(_CHILD_SCRIPT, encoding="utf-8")
    cache_file = tmp_path / "cache" / "gene_id_cache.json"
    env = dict(os.environ)
    src = str(Path(embpy.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH")) if p)
    done = subprocess.run(
        [sys.executable, str(script), method, str(cache_file)], env=env, capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, done.stderr
    assert cache_file.exists(), "the child's lookups were never written"
    saved = json.loads(cache_file.read_text(encoding="utf-8"))["symbols"]
    assert {f"human:GENE{i}" for i in range(3)} <= saved.keys()


# --- (d) temporary files left by killed processes are removed ----------------------------------


def _age(path: Path, seconds: float) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then))


@pytest.mark.parametrize("kind", ["gene", "alias"])
def test_save_removes_old_temporary_files_of_its_cache(tmp_path, kind):
    name = "gene_id_cache.json" if kind == "gene" else "symbol_resolution.json"
    cache_file = tmp_path / "cache" / name
    cache_file.parent.mkdir(parents=True)
    left_behind = cache_file.parent / f".{name}.4242-0a1b2c.tmp"  # from a process killed while saving
    being_written = cache_file.parent / f".{name}.4243-3d4e5f.tmp"  # another process saving right now
    other_cache = cache_file.parent / ".other.json.4244-6a7b8c.tmp"
    for leftover in (left_behind, being_written, other_cache):
        leftover.write_text("{", encoding="utf-8")
    _age(left_behind, 2 * 3600)
    _age(other_cache, 2 * 3600)
    if kind == "gene":
        _resolver_with_a_lookup(cache_file).flush_cache()
    else:
        _alias_cache_with_an_entry(cache_file).flush()
    assert not left_behind.exists()
    assert being_written.exists()
    assert other_cache.exists()
