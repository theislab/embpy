"""The gene id cache is written rarely, and each write goes through a temporary file of its own.

Every test points the cache at ``tmp_path`` and answers MyGene.info lookups from
a fake, so nothing here needs the network or touches the user's cache.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import embpy
from embpy.pp.basic import PerturbationProcessor
from embpy.resources.gene import resolver as resolver_mod
from embpy.resources.gene.resolver import _GENE_CACHE_VERSION, GeneResolver


def _fake_id(symbol: str) -> str:
    """Made-up human Ensembl id of the test gene ``GENE<n>``."""
    return f"ENSG9{int(symbol[4:]):010d}"


def _fake_symbol(ensembl_id: str) -> str:
    """Made-up approved symbol of the test id ``ENSG8<n>``."""
    return f"REV{int(ensembl_id[5:])}"


def _fake_symbol_lookup(symbol: str, organism: str, **kwargs) -> dict:
    eid = _fake_id(symbol)
    return {"status": "ok", "symbol": symbol, "ensembl_id": eid, "ensembl_ids": [eid], "candidates": [symbol]}


@contextmanager
def _fake_network() -> Iterator[None]:
    """MyGene.info answers from the fakes above; any real request fails."""
    no_network = AssertionError("network access in a test")
    with (
        patch.object(resolver_mod, "_mygene_exact_symbol_lookup", side_effect=_fake_symbol_lookup),
        patch.object(resolver_mod, "_mygene_symbol_for_ensembl", side_effect=lambda eid, org, **kw: _fake_symbol(eid)),
        patch.object(resolver_mod.requests, "get", side_effect=no_network),
        patch.object(resolver_mod.requests, "post", side_effect=no_network),
    ):
        yield


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


def _make_resolver(cache_file: Path) -> GeneResolver:
    r = GeneResolver.__new__(GeneResolver)
    r.ensembl = None
    r.mart_file = None
    r.chrom_folder = None
    r.release_version = 109
    r.species = "human"
    r._genome_fasta = None
    r._genome_dir = None
    r._cache_file = cache_file
    r._ensure_cache()
    return r


def _read(cache_file: Path) -> dict:
    return json.loads(cache_file.read_text(encoding="utf-8"))


@pytest.fixture
def resolver(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBPY_CACHE", str(tmp_path / "embpy-cache"))
    return _make_resolver(tmp_path / "cache" / "gene_id_cache.json")


def test_normalize_gene_names_writes_the_cache_a_few_times_for_500_symbols(resolver):
    symbols = [f"GENE{i}" for i in range(500)]
    processor = PerturbationProcessor(gene_resolver=resolver, drug_resolver=MagicMock())
    with _fake_network(), _record_writes(resolver._cache_file) as writes:
        result = processor.normalize_gene_names(symbols)
    assert result == {s: _fake_id(s) for s in symbols}
    assert len(writes) <= 5


def test_single_lookups_save_every_n_new_entries(resolver, monkeypatch):
    monkeypatch.setattr(resolver_mod, "_GENE_CACHE_SAVE_EVERY", 10)
    with _fake_network(), _record_writes(resolver._cache_file) as writes:
        for i in range(25):
            resolver.symbol_to_ensembl(f"GENE{i}")
    assert len(writes) == 2
    saved = _read(resolver._cache_file)["symbols"]
    assert {f"human:GENE{i}" for i in range(20)} <= saved.keys()
    assert "human:GENE20" not in saved


def test_save_threshold_can_be_set_with_an_environment_variable(resolver, monkeypatch):
    monkeypatch.setenv("EMBPY_GENE_CACHE_SAVE_EVERY", "10")
    with _fake_network(), _record_writes(resolver._cache_file) as writes:
        for i in range(25):
            resolver.symbol_to_ensembl(f"GENE{i}")
    assert len(writes) == 2


@pytest.mark.parametrize("value", ["0", "-3", "often"])
def test_invalid_save_threshold_falls_back_to_the_default(resolver, monkeypatch, caplog, value):
    monkeypatch.setattr(resolver_mod, "_GENE_CACHE_SAVE_EVERY", 10)
    monkeypatch.setenv("EMBPY_GENE_CACHE_SAVE_EVERY", value)
    with _fake_network(), _record_writes(resolver._cache_file) as writes, caplog.at_level(logging.WARNING):
        for i in range(25):
            resolver.symbol_to_ensembl(f"GENE{i}")
    assert len(writes) == 2
    assert any("EMBPY_GENE_CACHE_SAVE_EVERY" in r.getMessage() for r in caplog.records)


def test_batch_calls_write_the_cache_at_most_once(resolver):
    symbols = [f"GENE{i}" for i in range(50)]
    ids = [f"ENSG8{i:010d}" for i in range(50)]
    # Ensembl's bulk endpoint finds nothing, so every entry goes through a single lookup.
    nothing_found = MagicMock(ok=True, status_code=200)
    nothing_found.json.return_value = {}
    with _fake_network(), patch.object(resolver_mod, "_ensembl_post", return_value=nothing_found):
        with _record_writes(resolver._cache_file) as writes:
            assert resolver.symbols_to_ensembl_batch(symbols) == {s: _fake_id(s) for s in symbols}
        assert len(writes) == 1
        with _record_writes(resolver._cache_file) as writes:
            assert resolver.ensembl_to_symbols_batch(ids) == {e: _fake_symbol(e) for e in ids}
        assert len(writes) == 1
        # Answered from the cache: nothing new, so nothing is written.
        with _record_writes(resolver._cache_file) as writes:
            resolver.symbols_to_ensembl_batch(symbols)
            resolver.ensembl_to_symbols_batch(ids)
        assert writes == []


def test_flush_cache_writes_pending_entries_once(resolver):
    with _fake_network(), _record_writes(resolver._cache_file) as writes:
        for i in range(3):
            resolver.symbol_to_ensembl(f"GENE{i}")
        assert writes == []
        resolver.flush_cache()
        resolver.flush_cache()  # nothing pending any more
    assert len(writes) == 1
    saved = _read(resolver._cache_file)
    assert saved["version"] == _GENE_CACHE_VERSION
    assert {f"human:GENE{i}": _fake_id(f"GENE{i}") for i in range(3)}.items() <= saved["symbols"].items()
    assert saved["ensembl"][f"human:{_fake_id('GENE0')}"] == "GENE0"

    # The file reads back as before: a new resolver answers from it.
    reloaded = _make_resolver(resolver._cache_file)
    reloaded._load_cache()
    with patch.object(resolver_mod.requests, "get", side_effect=AssertionError("network access in a test")):
        assert reloaded.symbol_to_ensembl("GENE1") == _fake_id("GENE1")
        assert reloaded.ensembl_to_symbol(_fake_id("GENE2")) == "GENE2"


def test_dropped_resolver_writes_its_pending_entries(tmp_path):
    cache_file = tmp_path / "cache" / "gene_id_cache.json"
    r = _make_resolver(cache_file)
    with _fake_network():
        r.symbol_to_ensembl("GENE7")
    assert not cache_file.exists()
    del r
    gc.collect()
    assert _read(cache_file)["symbols"]["human:GENE7"] == _fake_id("GENE7")


def _child_env(tmp_path: Path) -> dict[str, str]:
    """Environment for a child Python that imports this embpy and keeps off the user's cache."""
    env = dict(os.environ)
    src = str(Path(embpy.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH")) if p)
    env["EMBPY_CACHE"] = str(tmp_path / "embpy-cache")
    return env


_EXIT_SCRIPT = """
import sys
from pathlib import Path
from unittest.mock import patch

from embpy.resources.gene import resolver as mod

cache_file = Path(sys.argv[1])
r = mod.GeneResolver.__new__(mod.GeneResolver)
r.ensembl = None
r.species = "human"
r._cache_file = cache_file
r._ensure_cache()


def lookup(symbol, organism, **kwargs):
    eid = "ENSG9" + symbol[4:].zfill(10)
    return {"status": "ok", "symbol": symbol, "ensembl_id": eid, "ensembl_ids": [eid], "candidates": [symbol]}


with (
    patch.object(mod, "_mygene_exact_symbol_lookup", side_effect=lookup),
    patch.object(mod.requests, "get", side_effect=AssertionError("network access in a test")),
):
    for i in range(3):
        r.symbol_to_ensembl(f"GENE{i}")
print("written before exit" if cache_file.exists() else "not written before exit")
"""


def test_pending_entries_are_written_when_the_process_exits(tmp_path):
    cache_file = tmp_path / "cache" / "gene_id_cache.json"
    done = subprocess.run(
        [sys.executable, "-c", _EXIT_SCRIPT, str(cache_file)],
        env=_child_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "not written before exit"
    saved = _read(cache_file)["symbols"]
    assert {f"human:GENE{i}": _fake_id(f"GENE{i}") for i in range(3)}.items() <= saved.items()


def test_each_save_uses_its_own_temporary_file_in_the_cache_folder(resolver):
    other = _make_resolver(resolver._cache_file)
    with _record_writes(resolver._cache_file) as writes:
        resolver._save_cache()
        resolver._save_cache()
        other._save_cache()
    assert len(writes) == 3
    assert len(set(writes)) == 3
    assert all(src.parent == resolver._cache_file.parent for src in writes)
    assert [p.name for p in resolver._cache_file.parent.iterdir()] == [resolver._cache_file.name]


def test_failed_save_logs_a_warning_and_leaves_no_temporary_file(resolver, caplog):
    resolver._save_cache()
    before = resolver._cache_file.read_bytes()
    resolver._symbol_cache[("human", "GENE1")] = _fake_id("GENE1")
    with patch.object(os, "replace", side_effect=OSError("disk full")), caplog.at_level(logging.DEBUG):
        resolver._save_cache()
    assert any(r.levelno == logging.WARNING and "gene cache" in r.getMessage() for r in caplog.records)
    assert [p.name for p in resolver._cache_file.parent.iterdir()] == [resolver._cache_file.name]
    assert resolver._cache_file.read_bytes() == before


def test_saved_cache_file_gets_the_permissions_of_a_new_file(resolver):
    resolver._save_cache()
    umask = os.umask(0)
    os.umask(umask)
    assert stat.S_IMODE(resolver._cache_file.stat().st_mode) == 0o666 & ~umask


_SAVE_SCRIPT = """
import sys
import time
from pathlib import Path

from embpy.resources.gene.resolver import GeneResolver

cache_file, tag, digit, size, flags = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])
r = GeneResolver.__new__(GeneResolver)
r._cache_file = cache_file
r._ensure_cache()
for i in range(size):
    r._symbol_cache[("human", f"{tag}{i}")] = f"ENSG{digit}{i:010d}"
(flags / f"ready-{tag}").touch()
while not (flags / "go").exists():
    time.sleep(0.001)
for _ in range(60):
    r._save_cache()
"""

# Writer tag -> (id digit, number of entries). The files differ in length, so
# two writes mixed into one file show up as broken JSON or as both tags.
_WRITERS = {"A": ("1", 3000), "B": ("2", 1000)}


def _check_one_writers_file(text: str) -> None:
    saved = json.loads(text)
    assert saved["version"] == _GENE_CACHE_VERSION
    tags = {key.split(":", 1)[1][0] for key in saved["symbols"]}
    assert len(tags) == 1, f"entries of both writers in one file: {sorted(tags)}"
    (tag,) = tags
    digit, size = _WRITERS[tag]
    assert saved["symbols"] == {f"human:{tag}{i}": f"ENSG{digit}{i:010d}" for i in range(size)}


def test_two_processes_saving_at_once_leave_a_valid_cache_file(tmp_path):
    cache_file = tmp_path / "cache" / "gene_id_cache.json"
    flags = tmp_path / "flags"
    flags.mkdir()
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _SAVE_SCRIPT, str(cache_file), tag, digit, str(size), str(flags)],
            env=_child_env(tmp_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for tag, (digit, size) in _WRITERS.items()
    ]
    try:
        # Both writers start saving together, once both are ready.
        deadline = time.monotonic() + 300
        while not all((flags / f"ready-{tag}").exists() for tag in _WRITERS):
            if any(p.poll() is not None for p in procs) or time.monotonic() > deadline:
                break
            time.sleep(0.01)
        (flags / "go").touch()
        # Every version of the file a reader sees meanwhile is one writer's complete file.
        while any(p.poll() is None for p in procs) and time.monotonic() < deadline:
            try:
                text = cache_file.read_text(encoding="utf-8")
            except OSError:  # not written yet, or being replaced (Windows)
                text = None
            if text is not None:
                _check_one_writers_file(text)
            time.sleep(0.005)
        outputs = [p.communicate(timeout=300) for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    assert [p.returncode for p in procs] == [0, 0], outputs
    _check_one_writers_file(cache_file.read_text(encoding="utf-8"))
    assert [p.name for p in cache_file.parent.iterdir()] == [cache_file.name]
