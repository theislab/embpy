"""``gene_n_ppi_partners`` counts what was fetched, and the fetch is capped.

STRING is queried with ``limit=DEFAULT_PPI_PARTNERS`` and Open Targets with
``size=DEFAULT_DISEASE_ASSOCIATIONS``, so for any well-studied gene the summary
columns report the cap rather than a measurement. Measured across a panel of
eight genes, ``gene_n_ppi_partners`` was exactly 10 every time.

A near-constant column named like a count invites regressing on it, so the
saturated rows are flagged.
"""

from __future__ import annotations

from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from embpy.resources.gene.annotator import (
    DEFAULT_DISEASE_ASSOCIATIONS,
    DEFAULT_PPI_PARTNERS,
    GeneAnnotator,
)

GENES = ["TP53", "SMALLGENE"]


def _annotations() -> dict:
    return {
        # saturated: exactly as many partners/diseases as were asked for
        "TP53": {
            "pathways": {"reactome": ["a", "b", "c"]},
            "ppi_partners": [{"p": i} for i in range(DEFAULT_PPI_PARTNERS)],
            "disease_associations": [
                {"d": i} for i in range(DEFAULT_DISEASE_ASSOCIATIONS)
            ],
            "transcription_factors": [],
            "tissue_expression": [{"tissue_name": "lung"}],
        },
        # genuinely below the cap
        "SMALLGENE": {
            "pathways": {"reactome": ["a"]},
            "ppi_partners": [{"p": 0}, {"p": 1}],
            "disease_associations": [{"d": 0}],
            "transcription_factors": [],
            "tissue_expression": [],
        },
    }


@pytest.fixture()
def adata() -> ad.AnnData:
    return ad.AnnData(
        X=np.zeros((len(GENES), 1), dtype=np.float32),
        obs=pd.DataFrame({"symbol": GENES}, index=pd.Index(GENES)),
        var=pd.DataFrame(index=["placeholder"]),
    )


class TestSaturationFlags:
    def _annotate(self, adata: ad.AnnData) -> ad.AnnData:
        annotator = GeneAnnotator()
        with patch.object(GeneAnnotator, "annotate_batch", return_value=_annotations()):
            return annotator.annotate_adata(adata, column="symbol")

    def test_flags_are_written(self, adata: ad.AnnData) -> None:
        out = self._annotate(adata)
        assert "gene_n_ppi_partners_at_limit" in out.obs
        assert "gene_n_disease_assoc_at_limit" in out.obs

    def test_saturated_gene_is_flagged(self, adata: ad.AnnData) -> None:
        out = self._annotate(adata)
        assert bool(out.obs.loc["TP53", "gene_n_ppi_partners_at_limit"]) is True
        assert bool(out.obs.loc["TP53", "gene_n_disease_assoc_at_limit"]) is True

    def test_unsaturated_gene_is_not_flagged(self, adata: ad.AnnData) -> None:
        out = self._annotate(adata)
        assert bool(out.obs.loc["SMALLGENE", "gene_n_ppi_partners_at_limit"]) is False
        assert bool(out.obs.loc["SMALLGENE", "gene_n_disease_assoc_at_limit"]) is False

    def test_limits_are_recorded_for_the_reader(self, adata: ad.AnnData) -> None:
        out = self._annotate(adata)
        limits = out.uns["gene_annotation_limits"]
        assert limits["gene_n_ppi_partners"] == DEFAULT_PPI_PARTNERS
        assert limits["gene_n_disease_assoc"] == DEFAULT_DISEASE_ASSOCIATIONS

    def test_pathway_count_is_not_flagged_because_it_is_not_capped(
        self, adata: ad.AnnData
    ) -> None:
        """Only the paginated sources saturate; gene_n_pathways is a real count."""
        out = self._annotate(adata)
        assert "gene_n_pathways_at_limit" not in out.obs
        assert out.obs.loc["TP53", "gene_n_pathways"] == 3


class TestRequestSizesAreNamedConstants:
    def test_getters_default_to_the_documented_caps(self) -> None:
        import inspect

        ppi = inspect.signature(GeneAnnotator.get_protein_interactions)
        disease = inspect.signature(GeneAnnotator.get_disease_associations)
        assert ppi.parameters["n_partners"].default == DEFAULT_PPI_PARTNERS
        assert disease.parameters["top_n"].default == DEFAULT_DISEASE_ASSOCIATIONS
