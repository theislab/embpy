"""Fixtures for the whole test suite (tests/embpy/conftest.py adds the embpy-specific ones)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_embpy_cache(tmp_path_factory, monkeypatch):
    """Give every test an empty ``EMBPY_CACHE`` folder of its own.

    embpy keeps its caches (gene ids, symbol aliases, Reactome files, genomes,
    ...) under ``$EMBPY_CACHE``, ``~/.cache/embpy`` by default. Pointing it at a
    fresh temporary folder means no test reads or writes the user's real cache,
    and no test sees entries another test left behind. Child processes started
    by a test inherit the setting. ``EMBPY_GENE_CACHE_SAVE_EVERY`` is cleared so
    a value set in the user's shell does not change how often caches are saved.
    """
    monkeypatch.setenv("EMBPY_CACHE", str(tmp_path_factory.mktemp("embpy-cache")))
    monkeypatch.delenv("EMBPY_GENE_CACHE_SAVE_EVERY", raising=False)
