"""The symbol alias cache (symbol_resolution.json) is written rarely, through temporary files of its own.

Every test keeps the cache in ``tmp_path``; nothing here needs the network.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import embpy
from embpy.resources.gene._alias_resolver import AliasCache, Resolution, resolve_symbol_chain


def _resolution(i: int) -> Resolution:
    return Resolution(
        input_symbol=f"ALIAS{i}",
        organism="human",
        approved_symbol=f"GENE{i}",
        source="hgnc",
        chain=(("pyensembl", "miss"), ("hgnc", "ok")),
    )


@contextmanager
def _record_writes(cache_file: Path) -> Iterator[list[Path]]:
    """List the file each write of ``cache_file`` moves into place."""
    real_replace = os.replace
    moved: list[Path] = []

    def _replace(src, dst, *args, **kwargs):
        if Path(dst) == cache_file:
            moved.append(Path(src))
        return real_replace(src, dst, *args, **kwargs)

    with patch.object(os, "replace", _replace):
        yield moved


def _entries(cache_file: Path) -> dict:
    return json.loads(cache_file.read_text(encoding="utf-8"))["entries"]


def test_puts_are_saved_every_n_new_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBPY_GENE_CACHE_SAVE_EVERY", "10")
    cache_file = tmp_path / "cache" / "symbol_resolution.json"
    cache = AliasCache(path=cache_file)
    with _record_writes(cache_file) as writes:
        for i in range(25):
            cache.put(_resolution(i))
    assert len(writes) == 2
    saved = _entries(cache_file)
    assert {f"human::ALIAS{i}" for i in range(20)} == saved.keys()


def test_flush_writes_pending_entries_once(tmp_path):
    cache_file = tmp_path / "cache" / "symbol_resolution.json"
    cache = AliasCache(path=cache_file)
    with _record_writes(cache_file) as writes:
        for i in range(3):
            cache.put(_resolution(i))
        assert writes == []
        cache.flush()
        cache.flush()  # nothing pending any more
    assert len(writes) == 1
    reloaded = AliasCache(path=cache_file)
    assert reloaded.get("ALIAS2", "human") == _resolution(2)


def test_dropped_cache_writes_its_pending_entries(tmp_path):
    cache_file = tmp_path / "cache" / "symbol_resolution.json"
    cache = AliasCache(path=cache_file)
    cache.put(_resolution(7))
    assert not cache_file.exists()
    del cache
    gc.collect()
    assert _entries(cache_file)["human::ALIAS7"]["approved_symbol"] == "GENE7"


_EXIT_SCRIPT = """
import sys
from pathlib import Path

from embpy.resources.gene._alias_resolver import AliasCache, Resolution

cache_file = Path(sys.argv[1])
cache = AliasCache(path=cache_file)
for i in range(3):
    cache.put(Resolution(input_symbol=f"ALIAS{i}", organism="human", approved_symbol=f"GENE{i}", source="hgnc"))
print("written before exit" if cache_file.exists() else "not written before exit")
"""


def test_pending_entries_are_written_when_the_process_exits(tmp_path):
    cache_file = tmp_path / "cache" / "symbol_resolution.json"
    env = dict(os.environ)
    src = str(Path(embpy.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH")) if p)
    env["EMBPY_CACHE"] = str(tmp_path / "embpy-cache")
    done = subprocess.run(
        [sys.executable, "-c", _EXIT_SCRIPT, str(cache_file)], env=env, capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "not written before exit"
    assert {f"human::ALIAS{i}" for i in range(3)} == _entries(cache_file).keys()


def test_each_write_uses_its_own_temporary_file_in_the_cache_folder(tmp_path):
    cache_file = tmp_path / "cache" / "symbol_resolution.json"
    first, second = AliasCache(path=cache_file), AliasCache(path=cache_file)
    with _record_writes(cache_file) as writes:
        first.put(_resolution(1))
        first.flush()
        first.put(_resolution(2))
        first.flush()
        second.put(_resolution(3))
        second.flush()
    assert len(writes) == 3
    assert len(set(writes)) == 3
    assert all(src.parent == cache_file.parent for src in writes)
    assert [p.name for p in cache_file.parent.iterdir()] == [cache_file.name]


def test_failed_write_logs_a_warning_and_leaves_no_temporary_file(tmp_path, caplog):
    cache_file = tmp_path / "cache" / "symbol_resolution.json"
    cache = AliasCache(path=cache_file)
    cache.put(_resolution(1))
    cache.flush()
    before = cache_file.read_bytes()
    with patch.object(os, "replace", side_effect=OSError("disk full")), caplog.at_level(logging.DEBUG):
        cache.put(_resolution(2))
        cache.flush()  # must not raise
    assert any(r.levelno == logging.WARNING and "symbol_resolution.json" in r.getMessage() for r in caplog.records)
    assert [p.name for p in cache_file.parent.iterdir()] == [cache_file.name]
    assert cache_file.read_bytes() == before
    # The entry is still pending and is written by the next flush.
    cache.flush()
    assert "human::ALIAS2" in _entries(cache_file)


def test_chain_without_a_cache_saves_its_result_right_away(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBPY_CACHE", str(tmp_path))

    class _Release:
        def genes_by_name(self, name):
            return [type("Gene", (), {"gene_name": "TP53"})()]

    res = resolve_symbol_chain("P53", organism="human", ensembl=_Release())
    assert res.approved_symbol == "TP53"
    assert _entries(tmp_path / "symbol_resolution.json")["human::P53"]["approved_symbol"] == "TP53"
