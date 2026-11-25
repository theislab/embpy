"""HPA antibody lookups resolve a gene symbol through embpy's GeneResolver."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from embpy.resources.gene import resolver as resolver_mod
from embpy.resources.morphology import hpa

_FAKE_ID = "ENSG90000000007"


def _fake_symbol_lookup(symbol, organism, **kwargs):
    return {"status": "ok", "symbol": symbol, "ensembl_id": _FAKE_ID, "ensembl_ids": [_FAKE_ID], "candidates": [symbol]}


@pytest.mark.parametrize(
    ("resolve", "expected"),
    [
        (hpa._resolve_ensembl_id, _FAKE_ID),
        (hpa._resolve_ensembl_id_quiet, (_FAKE_ID, "GeneResolver")),
    ],
)
def test_symbol_is_resolved_with_gene_resolver(resolve, expected, monkeypatch, tmp_path):
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
        assert resolve("NOTASEEDGENE") == expected
    lookup.assert_called_once_with("NOTASEEDGENE", "human")
