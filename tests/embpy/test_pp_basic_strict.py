"""``strict`` and ``fallback_to_unmapped`` reach the gene resolver and are never dropped silently."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from embpy.errors import GeneResolutionError
from embpy.pp.basic import PerturbationProcessor


class _StrictPathBug:
    """A resolver whose strict code path has a bug that raises TypeError."""

    def __init__(self):
        self.calls: list[dict] = []

    def symbol_to_ensembl(self, symbol, organism="human", *, fallback_to_unmapped=False, strict=False):
        self.calls.append({"symbol": symbol, "strict": strict})
        if strict:
            raise TypeError("bug in the strict path")
        return None


class _OldStyleResolver:
    """A custom resolver whose symbol_to_ensembl takes only the symbol and organism."""

    def __init__(self, answers: dict[str, str]):
        self.answers = answers
        self.calls: list[tuple] = []

    def symbol_to_ensembl(self, symbol, organism="human"):
        self.calls.append((symbol, organism))
        return self.answers.get(symbol)


def _processor(resolver) -> PerturbationProcessor:
    return PerturbationProcessor(gene_resolver=resolver, drug_resolver=MagicMock())


@pytest.mark.parametrize("method", ["normalize_gene_names", "resolve_identifiers"])
def test_type_error_inside_the_resolver_propagates_with_strict(method):
    resolver = _StrictPathBug()
    with pytest.raises(TypeError, match="bug in the strict path"):
        getattr(_processor(resolver), method)(["TP53"], strict=True)
    assert resolver.calls == [{"symbol": "TP53", "strict": True}]


@pytest.mark.parametrize("method", ["normalize_gene_names", "resolve_identifiers"])
def test_resolver_without_a_strict_option_still_fails_unmapped_genes_in_strict_mode(method):
    resolver = _OldStyleResolver({"TP53": "ENSG00000141510"})
    with pytest.raises(GeneResolutionError) as err:
        getattr(_processor(resolver), method)(["TP53", "NOTAGENE"], strict=True)
    assert err.value.failed_genes == ["NOTAGENE"]


def test_resolver_without_options_gets_symbol_and_organism_and_unmapped_fallback():
    resolver = _OldStyleResolver({"TP53": "ENSG00000141510"})
    result = _processor(resolver).normalize_gene_names(
        ["TP53", "NOTAGENE"], organism="mouse", fallback_to_unmapped=True
    )
    assert result == {"TP53": "ENSG00000141510", "NOTAGENE": "NOTAGENE"}
    assert resolver.calls == [("TP53", "mouse"), ("NOTAGENE", "mouse")]
