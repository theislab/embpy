"""MyGene.info lookups must be exact, species-checked, and never cache aliases as symbols."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from embpy.resources.gene._alias_resolver import _step_mygene
from embpy.resources.gene.resolver import (
    _GENE_CACHE_VERSION,
    GeneResolver,
    _mygene_exact_symbol_lookup,
    ensembl_id_matches_species,
)

_TP53 = {"symbol": "TP53", "taxid": 9606, "ensembl": {"gene": "ENSG00000141510"}}
_PURPL = {"symbol": "PURPL", "taxid": 9606, "ensembl": {"gene": "ENSG00000250337"}}


def _response(hits: list[dict], status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.ok = status == 200
    resp.status_code = status
    resp.json.return_value = {"hits": hits}
    return resp


def _mygene(table: dict[str, list[dict]]):
    """Fake ``requests.get`` answering MyGene queries from ``table`` by their ``q``."""

    def _get(url, params=None, **kwargs):
        # A free-text query is what used to turn 'P53' into PURPL: rank it first
        # so any regression to free text would pick the wrong gene.
        return _response(table.get(params["q"], [_PURPL]))

    return _get


@pytest.fixture
def resolver(tmp_path):
    r = GeneResolver.__new__(GeneResolver)
    r.ensembl = None
    r.mart_file = None
    r.chrom_folder = None
    r.release_version = 109
    r.species = "human"
    r._genome_fasta = None
    r._genome_dir = None
    r._cache_file = tmp_path / "gene_id_cache.json"
    r._ensure_cache()
    return r


def test_alias_resolves_to_approved_symbol_not_free_text_hit():
    table = {'symbol:"P53"': [], 'alias:"P53"': [_TP53]}
    with patch("embpy.resources.gene.resolver.requests.get", side_effect=_mygene(table)):
        hit = _mygene_exact_symbol_lookup("P53", "human")
    assert hit["status"] == "ok"
    assert hit["symbol"] == "TP53"
    assert hit["ensembl_id"] == "ENSG00000141510"


def test_alias_shared_by_several_genes_is_ambiguous():
    hits = [
        {"symbol": "CRYBG1", "taxid": 9606},
        {"symbol": "SLC45A2", "taxid": 9606},
        {"symbol": "AURKB", "taxid": 9606},
    ]
    table = {'symbol:"AIM1"': [], 'alias:"AIM1"': hits}
    with patch("embpy.resources.gene.resolver.requests.get", side_effect=_mygene(table)):
        hit = _mygene_exact_symbol_lookup("AIM1", "human")
    assert hit["status"] == "ambiguous"
    assert hit["candidates"] == ["AURKB", "CRYBG1", "SLC45A2"]


def test_hits_from_another_species_are_rejected():
    mouse_hit = {"symbol": "Trp53", "taxid": 10090, "ensembl": {"gene": "ENSMUSG00000059552"}}
    table = {'symbol:"Trp53"': [mouse_hit], 'alias:"Trp53"': [mouse_hit]}
    with patch("embpy.resources.gene.resolver.requests.get", side_effect=_mygene(table)):
        hit = _mygene_exact_symbol_lookup("Trp53", "human")
    assert hit["status"] == "miss"


def test_several_ensembl_ids_defer_to_ensembl_for_the_primary_one():
    hla = {"symbol": "HLA-A", "taxid": 9606, "ensembl": [{"gene": "ENSG00000235657"}, {"gene": "ENSG00000206503"}]}
    table = {'symbol:"HLA-A"': [hla]}
    with patch("embpy.resources.gene.resolver.requests.get", side_effect=_mygene(table)):
        hit = _mygene_exact_symbol_lookup("HLA-A", "human")
    assert hit["status"] == "ok"
    assert hit["symbol"] == "HLA-A"
    assert hit["ensembl_id"] is None


def test_service_failure_is_reported_as_error_not_miss():
    with patch("embpy.resources.gene.resolver.requests.get", return_value=_response([], status=503)):
        hit = _mygene_exact_symbol_lookup("TP53", "human")
    assert hit["status"].startswith("error")


def test_alias_chain_mygene_step_uses_exact_lookup():
    table = {'symbol:"P53"': [], 'alias:"P53"': [_TP53]}
    with patch("embpy.resources.gene.resolver.requests.get", side_effect=_mygene(table)):
        assert _step_mygene("P53", "human") == ("TP53", "ok")


def test_symbol_to_ensembl_caches_approved_symbol_not_the_alias(resolver):
    table = {'symbol:"P53"': [], 'alias:"P53"': [_TP53]}
    resolver._symbol_cache.clear()
    resolver._ensembl_cache.clear()
    with patch("embpy.resources.gene.resolver.requests.get", side_effect=_mygene(table)):
        assert resolver.symbol_to_ensembl("P53") == "ENSG00000141510"
    # Reverse lookup is answered from cache, with the approved symbol.
    with patch("embpy.resources.gene.resolver.requests.get", side_effect=AssertionError("no network")):
        assert resolver.ensembl_to_symbol("ENSG00000141510") == "TP53"
    saved = json.loads(resolver._cache_file.read_text(encoding="utf-8"))
    assert saved["ensembl"]["human:ENSG00000141510"] == "TP53"
    assert "P53" not in saved["ensembl"].values()


def test_batch_diagnostics_report_ambiguous_aliases(resolver):
    hits = [{"symbol": "SEPTIN2", "taxid": 9606}, {"symbol": "SEPTIN6", "taxid": 9606}]
    table = {'symbol:"SEPT2"': [], 'alias:"SEPT2"': hits}
    not_found = MagicMock(ok=False, status_code=400)
    empty_post = MagicMock(ok=True, status_code=200)
    empty_post.json.return_value = {}
    with (
        patch("embpy.resources.gene.resolver.requests.get", side_effect=_mygene(table)),
        patch("embpy.resources.gene.resolver._ensembl_get", return_value=not_found),
        patch("embpy.resources.gene.resolver._ensembl_post", return_value=empty_post),
    ):
        result, diag = resolver.symbols_to_ensembl_batch(["SEPT2"], return_diagnostics=True)
    assert result["SEPT2"] is None
    assert diag["ambiguous"] == {"SEPT2": ["SEPTIN2", "SEPTIN6"]}


def test_legacy_cache_with_alias_reverse_entry_is_ignored(resolver):
    resolver._cache_file.write_text(
        json.dumps(
            {
                "version": 1,
                "symbols": {"human:P53": "ENSG00000141510", "human:FOO1": "ENSG00000000003"},
                "ensembl": {"human:ENSG00000141510": "P53", "human:ENSG00000000003": "FOO1"},
            }
        ),
        encoding="utf-8",
    )
    resolver._symbol_cache = {}
    resolver._ensembl_cache = {}
    resolver._load_cache()
    assert resolver._ensembl_cache[("human", "ENSG00000141510")] == "TP53"
    assert ("human", "ENSG00000000003") not in resolver._ensembl_cache
    assert ("human", "P53") not in resolver._symbol_cache


def test_current_cache_cannot_override_seeded_symbols(resolver):
    resolver._cache_file.write_text(
        json.dumps(
            {
                "version": _GENE_CACHE_VERSION,
                "symbols": {"human:FOO1": "ENSG00000000003"},
                "ensembl": {"human:ENSG00000141510": "P53", "human:ENSG00000000003": "TSPAN6"},
            }
        ),
        encoding="utf-8",
    )
    resolver._symbol_cache = {}
    resolver._ensembl_cache = {}
    resolver._load_cache()
    assert resolver._ensembl_cache[("human", "ENSG00000141510")] == "TP53"
    assert resolver._ensembl_cache[("human", "ENSG00000000003")] == "TSPAN6"


@pytest.mark.parametrize(
    ("eid", "organism", "expected"),
    [
        ("ENSG00000141510", "human", True),
        ("ENSMUSG00000059552", "human", False),
        ("ENSMUSG00000059552", "mouse", True),
        ("ENSG00000141510", "mouse", False),
        ("ENSDARG00000035559", "zebrafish", True),
    ],
)
def test_ensembl_id_species_check(eid, organism, expected):
    assert ensembl_id_matches_species(eid, organism) is expected


class _SharedSymbolEnsembl:
    """Fake pyensembl where two Ensembl ids carry the same symbol."""

    FORWARD, OTHER = "ENSG00000900001", "ENSG00000900002"

    def genes_by_name(self, name):
        return [MagicMock(gene_id=self.FORWARD, gene_name="SHARED1")] if name == "SHARED1" else []

    def gene_by_id(self, ens):
        return MagicMock(gene_name="SHARED1") if ens in (self.FORWARD, self.OTHER) else None


@pytest.mark.parametrize("batch", [False, True])
def test_reverse_lookup_does_not_change_forward_result(resolver, batch):
    """symbol -> id stays the same after id -> symbol finds another id with that symbol."""
    ens = _SharedSymbolEnsembl()
    resolver.ensembl = ens

    def forward():
        if batch:
            return resolver.symbols_to_ensembl_batch(["SHARED1"])["SHARED1"]
        return resolver.symbol_to_ensembl("SHARED1")

    assert forward() == ens.FORWARD
    if batch:
        assert resolver.ensembl_to_symbols_batch([ens.OTHER])[ens.OTHER] == "SHARED1"
    else:
        assert resolver.ensembl_to_symbol(ens.OTHER) == "SHARED1"
    assert forward() == ens.FORWARD

    # The same answer after saving and reloading the cache, without pyensembl.
    resolver._save_cache()
    reloaded = GeneResolver.__new__(GeneResolver)
    reloaded.ensembl = None
    reloaded.species = "human"
    reloaded._cache_file = resolver._cache_file
    reloaded._ensure_cache()
    reloaded._load_cache()
    assert reloaded._symbol_cache[("human", "SHARED1")] == ens.FORWARD
    assert reloaded._ensembl_cache[("human", ens.OTHER)] == "SHARED1"
