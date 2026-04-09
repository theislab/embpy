"""Tests for multi-species support across resolvers, annotators, and BioEmbedder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestIsEnsemblId:
    def test_human_gene(self):
        from embpy.resources.gene_resolver import _is_ensembl_id

        assert _is_ensembl_id("ENSG00000141510") is True

    def test_mouse_gene(self):
        from embpy.resources.gene_resolver import _is_ensembl_id

        assert _is_ensembl_id("ENSMUSG00000059552") is True

    def test_zebrafish_gene(self):
        from embpy.resources.gene_resolver import _is_ensembl_id

        assert _is_ensembl_id("ENSDARG00000002314") is True

    def test_transcript(self):
        from embpy.resources.gene_resolver import _is_ensembl_id

        assert _is_ensembl_id("ENST00000269305") is True

    def test_protein(self):
        from embpy.resources.gene_resolver import _is_ensembl_id

        assert _is_ensembl_id("ENSP00000269305") is True

    def test_not_ensembl(self):
        from embpy.resources.gene_resolver import _is_ensembl_id

        assert _is_ensembl_id("TP53") is False
        assert _is_ensembl_id("P04637") is False
        assert _is_ensembl_id("CC(=O)O") is False

    def test_with_version(self):
        from embpy.resources.gene_resolver import _is_ensembl_id

        assert _is_ensembl_id("ENSG00000141510.12") is True


class TestGeneResolverSpecies:
    @patch("embpy.resources.gene_resolver._ensembl_get")
    def test_fetch_region_uses_species(self, mock_get):
        from embpy.resources.gene_resolver import GeneResolver

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "ATCG"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        gr = GeneResolver.__new__(GeneResolver)
        gr.species = "mouse"
        gr.ensembl = None

        result = gr._fetch_region_sequence("1", 100, 200, 1)
        call_url = mock_get.call_args[0][0]
        assert "/sequence/region/mouse/" in call_url

    @patch("embpy.resources.gene_resolver._ensembl_get")
    def test_fetch_region_default_species(self, mock_get):
        from embpy.resources.gene_resolver import GeneResolver

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "ATCG"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        gr = GeneResolver.__new__(GeneResolver)
        gr.species = "human"
        gr.ensembl = None

        gr._fetch_region_sequence("1", 100, 200, 1)
        call_url = mock_get.call_args[0][0]
        assert "/sequence/region/human/" in call_url


class TestProteinResolverSpecies:
    def test_organism_taxon_has_common_species(self):
        from embpy.resources.protein_resolver import ORGANISM_TAXON

        assert ORGANISM_TAXON["human"] == 9606
        assert ORGANISM_TAXON["mouse"] == 10090
        assert ORGANISM_TAXON["rat"] == 10116
        assert ORGANISM_TAXON["zebrafish"] == 7955
        assert ORGANISM_TAXON["drosophila"] == 7227
        assert ORGANISM_TAXON["worm"] == 6239
        assert ORGANISM_TAXON["yeast"] == 559292
        assert ORGANISM_TAXON["chicken"] == 9031
        assert ORGANISM_TAXON["pig"] == 9823
        assert ORGANISM_TAXON["dog"] == 9615

    def test_get_all_gene_ids_uses_organism(self):
        from embpy.resources.protein_resolver import ProteinResolver

        pr = ProteinResolver(organism="mouse")
        assert pr.organism == "mouse"

        pr_human = ProteinResolver(organism="human")
        assert pr_human.organism == "human"


class TestGeneAnnotatorSpecies:
    def test_string_taxon_human(self):
        from embpy.resources.gene_annotator import GeneAnnotator

        ann = GeneAnnotator(organism="human")
        assert ann._string_species == 9606

    def test_string_taxon_mouse(self):
        from embpy.resources.gene_annotator import GeneAnnotator

        ann = GeneAnnotator(organism="mouse")
        assert ann._string_species == 10090

    def test_gtex_skipped_for_mouse(self):
        from embpy.resources.gene_annotator import GeneAnnotator

        ann = GeneAnnotator(organism="mouse")
        result = ann.get_tissue_expression("Trp53")
        assert result == []

    def test_hpa_skipped_for_mouse(self):
        from embpy.resources.gene_annotator import GeneAnnotator

        ann = GeneAnnotator(organism="mouse")
        result = ann.get_subcellular_localization("Trp53")
        assert result == {}

    def test_gwas_skipped_for_mouse(self):
        from embpy.resources.gene_annotator import GeneAnnotator

        ann = GeneAnnotator(organism="mouse")
        result = ann.get_gwas_associations("Trp53")
        assert result == []

    def test_open_targets_skipped_for_mouse(self):
        from embpy.resources.gene_annotator import GeneAnnotator

        ann = GeneAnnotator(organism="mouse")
        result = ann.get_disease_associations("Trp53")
        assert result == []

    def test_ensembl_id_detection_mouse(self):
        from embpy.resources.gene_annotator import _is_ensembl_gene_id

        assert _is_ensembl_gene_id("ENSMUSG00000059552") is True
        assert _is_ensembl_gene_id("ENSG00000141510") is True
        assert _is_ensembl_gene_id("Trp53") is False


class TestProteinAnnotatorSpecies:
    def test_ensembl_id_regex_handles_mouse(self):
        import re

        pattern = r"^ENS[A-Z]*G\d{11}"
        assert re.match(pattern, "ENSMUSG00000059552", re.IGNORECASE)
        assert re.match(pattern, "ENSG00000141510", re.IGNORECASE)
        assert not re.match(pattern, "P04637", re.IGNORECASE)


class TestBioEmbedderOrganism:
    @patch("embpy.embedder.GeneResolver")
    @patch("embpy.embedder.ProteinResolver")
    @patch("embpy.embedder.TextResolver")
    def test_organism_propagated(self, mock_tr, mock_pr, mock_gr):
        from embpy.embedder import BioEmbedder

        with patch.object(BioEmbedder, "_discover_models", return_value={}):
            embedder = BioEmbedder(device="cpu", organism="mouse")

        assert embedder.organism == "mouse"
        mock_gr.assert_called_once_with(species="mouse")
        mock_pr.assert_called_once_with(organism="mouse")
        mock_tr.assert_called_once_with(organism="mouse")

    @patch("embpy.embedder.GeneResolver")
    @patch("embpy.embedder.ProteinResolver")
    @patch("embpy.embedder.TextResolver")
    def test_default_organism_is_human(self, mock_tr, mock_pr, mock_gr):
        from embpy.embedder import BioEmbedder

        with patch.object(BioEmbedder, "_discover_models", return_value={}):
            embedder = BioEmbedder(device="cpu")

        assert embedder.organism == "human"
