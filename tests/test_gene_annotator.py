"""Tests for embpy.resources.gene_annotator -- GeneAnnotator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embpy.resources.gene_annotator import GeneAnnotator


@pytest.fixture
def annotator():
    return GeneAnnotator(organism="human", rate_limit_delay=0)


# =====================================================================
# Pathways (mocked MyGene)
# =====================================================================


class TestPathways:
    @patch("embpy.resources.gene_annotator._get_json")
    def test_get_pathways(self, mock_get, annotator):
        mock_get.return_value = {
            "hits": [{
                "pathway": {
                    "reactome": [{"id": "R-HSA-123", "name": "Apoptosis"}],
                    "kegg": {"id": "hsa04210", "name": "Apoptosis"},
                    "wikipathways": [{"id": "WP254", "name": "Apoptosis"}],
                },
            }],
        }
        result = annotator.get_pathways("TP53")
        assert len(result["reactome"]) == 1
        assert len(result["kegg"]) == 1
        assert len(result["wikipathways"]) == 1

    @patch("embpy.resources.gene_annotator._get_json")
    def test_no_pathways(self, mock_get, annotator):
        mock_get.return_value = {"hits": [{}]}
        result = annotator.get_pathways("FAKEGENE")
        assert result["reactome"] == []


# =====================================================================
# Tissue expression (mocked GTEx)
# =====================================================================


class TestTissueExpression:
    @patch("embpy.resources.gene_annotator._get_json")
    def test_get_tissue_expression(self, mock_get, annotator):
        mock_get.side_effect = [
            {"hits": [{"ensembl": {"gene": "ENSG00000141510"}}]},
            {"data": [
                {"tissueSiteDetailId": "Brain", "tissueSiteDetail": "Brain - Cortex", "median": 15.2, "numSamples": 100},
                {"tissueSiteDetailId": "Liver", "tissueSiteDetail": "Liver", "median": 5.1, "numSamples": 80},
            ]},
        ]
        tissues = annotator.get_tissue_expression("TP53")
        assert len(tissues) == 2
        assert tissues[0]["median_tpm"] >= tissues[1]["median_tpm"]

    @patch("embpy.resources.gene_annotator._get_json")
    def test_unresolvable_gene(self, mock_get, annotator):
        mock_get.return_value = {"hits": []}
        tissues = annotator.get_tissue_expression("FAKEGENE")
        assert tissues == []


# =====================================================================
# PPI partners (mocked STRING)
# =====================================================================


class TestProteinInteractions:
    @patch("embpy.resources.gene_annotator._get_text")
    @patch("embpy.resources.gene_annotator._get_json")
    def test_get_ppi(self, mock_json, mock_text, annotator):
        mock_json.return_value = {"hits": [{"symbol": "TP53"}]}
        mock_text.return_value = (
            "stringId_A\tstringId_B\tpreferredName_A\tpreferredName_B\tscore\tnscore\tfscore\tpscore\tescore\tdscore\ttscore\n"
            "9606.ENSP1\t9606.ENSP2\tTP53\tMDM2\t999\t0\t0\t0\t900\t800\t950\n"
            "9606.ENSP1\t9606.ENSP3\tTP53\tBRCA1\t850\t0\t0\t0\t700\t600\t800\n"
        )
        partners = annotator.get_protein_interactions("TP53", n_partners=5)
        assert len(partners) == 2
        assert partners[0]["partner"] == "MDM2"
        assert partners[0]["combined_score"] == 999


# =====================================================================
# Disease associations (mocked Open Targets)
# =====================================================================


class TestDiseaseAssociations:
    @patch("embpy.resources.gene_annotator._post_json")
    @patch("embpy.resources.gene_annotator._get_json")
    def test_get_diseases(self, mock_json, mock_post, annotator):
        mock_json.return_value = {"hits": [{"ensembl": {"gene": "ENSG00000141510"}}]}
        mock_post.return_value = {
            "data": {
                "target": {
                    "associatedDiseases": {
                        "rows": [
                            {"disease": {"id": "EFO_001", "name": "Breast cancer"}, "score": 0.95},
                            {"disease": {"id": "EFO_002", "name": "Lung cancer"}, "score": 0.88},
                        ],
                    },
                },
            },
        }
        diseases = annotator.get_disease_associations("TP53")
        assert len(diseases) == 2
        assert diseases[0]["disease_name"] == "Breast cancer"


# =====================================================================
# Annotate (aggregation)
# =====================================================================


class TestAnnotate:
    @patch("embpy.resources.gene_annotator._get_json")
    def test_pathways_only(self, mock_get, annotator):
        mock_get.return_value = {"hits": [{"pathway": {"reactome": [{"id": "R1", "name": "P1"}]}}]}
        result = annotator.annotate("TP53", sources="pathways")
        assert "pathways" in result

    @patch("embpy.resources.gene_annotator._get_json")
    def test_ensembl_id_resolution(self, mock_get, annotator):
        mock_get.return_value = {"hits": [{"symbol": "TP53"}]}
        symbol = annotator._resolve_symbol("ENSG00000141510")
        assert symbol == "TP53"


# =====================================================================
# annotate_adata
# =====================================================================


class TestAnnotateAdata:
    @patch("embpy.resources.gene_annotator._get_json")
    @patch("embpy.resources.gene_annotator._post_json")
    @patch("embpy.resources.gene_annotator._get_text")
    def test_annotate_adata(self, mock_text, mock_post, mock_json, annotator):
        import pandas as pd
        from anndata import AnnData

        mock_json.return_value = {"hits": [{"pathway": {}, "ensembl": {"gene": "ENSG00000141510"}}]}
        mock_post.return_value = {"data": {"target": {"associatedDiseases": {"rows": []}}}}
        mock_text.return_value = ""

        adata = AnnData(
            obs=pd.DataFrame({"gene": ["TP53", "BRCA1"]}),
        )
        adata.obs.index = ["c0", "c1"]

        result = annotator.annotate_adata(
            adata, column="gene", sources="pathways",
        )
        assert "gene_n_pathways" in result.obs.columns
        assert "gene_annotations" in result.uns
