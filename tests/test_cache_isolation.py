"""Tests never write embpy caches into the user's real cache folder."""

from __future__ import annotations

import os
from pathlib import Path

from embpy.resources.gene._alias_resolver import default_cache_path
from embpy.resources.gene.resolver import default_gene_cache_path


def test_embpy_cache_is_a_temporary_folder_of_this_test(tmp_path_factory):
    cache = Path(os.environ["EMBPY_CACHE"])
    assert tmp_path_factory.getbasetemp() in cache.parents
    assert Path.home() / ".cache" / "embpy" not in (cache, *cache.parents)
    assert default_gene_cache_path().parent == cache
    assert default_cache_path().parent == cache


def test_each_test_gets_an_empty_cache_folder():
    cache = Path(os.environ["EMBPY_CACHE"])
    assert not cache.exists() or not any(cache.iterdir())
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "left_by_this_test.json").write_text("{}", encoding="utf-8")


def test_a_file_left_by_another_test_is_not_seen():
    assert not (Path(os.environ["EMBPY_CACHE"]) / "left_by_this_test.json").exists()


def test_save_threshold_setting_does_not_leak_into_tests():
    assert "EMBPY_GENE_CACHE_SAVE_EVERY" not in os.environ
