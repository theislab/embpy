"""The JUMP gene-canonicalisation fallback was dead code.

``_resolve_gene_jump_fallback`` built its resolver with
``GeneResolver(organism="human")``, but the constructor takes ``species=``. The
resulting ``TypeError`` was raised on the first statement inside the ``try`` and
swallowed by a bare ``except Exception: pass``, so neither the Ensembl
canonicalisation nor the MyGene alias lookup below it ever ran -- every gene that
needed an alternative symbol silently resolved to no rows.

The bug is invisible from the outside (the function is *supposed* to be able to
return nothing), so these tests assert on the call itself.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("torch")

from embpy.embedder import _resolve_gene_jump_fallback  # noqa: E402
from embpy.resources.gene.resolver import GeneResolver  # noqa: E402


class TestResolverIsConstructible:
    def test_constructor_rejects_organism_kwarg(self) -> None:
        """Pin the spelling the bug got wrong, so a rename cannot resurrect it."""
        with pytest.raises(TypeError):
            GeneResolver(organism="human")  # type: ignore[call-arg]

    def test_constructor_accepts_species_kwarg(self) -> None:
        assert GeneResolver(species="human", auto_download=False) is not None


class TestFallbackReachesTheResolver:
    @patch("embpy.resources.gene_resolver.GeneResolver")
    @patch("embpy.resources.jump_metadata.get_jump_item_location_metadata")
    def test_resolver_is_constructed_with_species(
        self, mock_jump: MagicMock, mock_resolver_cls: MagicMock
    ) -> None:
        mock_jump.return_value = []
        mock_resolver_cls.return_value.symbol_to_ensembl.return_value = None

        _resolve_gene_jump_fallback("TP53")

        assert mock_resolver_cls.called, (
            "the resolver was never constructed -- the TypeError is being swallowed "
            "again and the whole fallback is dead"
        )
        _, kwargs = mock_resolver_cls.call_args
        assert "organism" not in kwargs
        assert kwargs.get("species") == "human"

    @patch("embpy.resources.gene_resolver.GeneResolver")
    @patch("embpy.resources.jump_metadata.get_jump_item_location_metadata")
    def test_canonical_symbol_is_looked_up(
        self, mock_jump: MagicMock, mock_resolver_cls: MagicMock
    ) -> None:
        """The path past the construction must actually execute."""
        resolver = mock_resolver_cls.return_value
        resolver.symbol_to_ensembl.return_value = "ENSG00000141510"
        resolver.ensembl_to_symbol.return_value = "TP53"

        _resolve_gene_jump_fallback("TUMOUR_PROTEIN_53")

        resolver.symbol_to_ensembl.assert_called_once_with("TUMOUR_PROTEIN_53")
        resolver.ensembl_to_symbol.assert_called_once_with("ENSG00000141510")

    @patch("embpy.resources.gene_resolver.GeneResolver")
    @patch("embpy.resources.jump_metadata.get_jump_item_location_metadata")
    def test_canonical_rename_reports_its_source(
        self, mock_jump: MagicMock, mock_resolver_cls: MagicMock
    ) -> None:
        resolver = mock_resolver_cls.return_value
        resolver.symbol_to_ensembl.return_value = "ENSG00000141510"
        resolver.ensembl_to_symbol.return_value = "TP53"
        mock_jump.return_value = [{"well": "A01"}]

        rows, source = _resolve_gene_jump_fallback("TUMOUR_PROTEIN_53", return_source=True)

        assert rows == [{"well": "A01"}]
        assert source is not None and "TP53" in source
