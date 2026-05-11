"""Tests for local indexed genome/proteome access in GeneResolver and ProteinResolver."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestGeneResolverDownloadGenome:
    @patch("embpy.resources.gene_resolver.GeneResolver.__init__", return_value=None)
    def test_download_genome_unsupported_species(self, mock_init):
        from embpy.resources.gene_resolver import GeneResolver

        gr = GeneResolver.__new__(GeneResolver)
        gr.species = "platypus"
        gr.release_version = 109
        gr._genome_fasta = None
        gr._genome_dir = None

        with pytest.raises(ValueError, match="Unsupported species"):
            gr.download_genome()

    def test_species_assembly_mapping(self):
        from embpy.resources.gene_resolver import GeneResolver

        assert "human" in GeneResolver._SPECIES_ASSEMBLY
        assert "mouse" in GeneResolver._SPECIES_ASSEMBLY
        assert GeneResolver._SPECIES_ASSEMBLY["human"][1] == "GRCh38"
        assert GeneResolver._SPECIES_ASSEMBLY["mouse"][1] == "GRCm39"


class TestGeneResolverLocalIndexedSequence:
    @pytest.fixture
    def resolver_with_genome(self, tmp_path):
        """Create a GeneResolver with a tiny mock genome FASTA."""
        import pysam
        from embpy.resources.gene_resolver import GeneResolver

        fa_path = tmp_path / "test_genome.fa"
        fa_path.write_text(
            ">1\n"
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG\n"
            "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG\n"
            ">17\n"
            "NNNATCGATCGAAATTTCCCGGGAAATTTCCCGGGNNNN\n"
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG\n"
        )
        pysam.faidx(str(fa_path))

        gr = GeneResolver.__new__(GeneResolver)
        gr.species = "human"
        gr.release_version = 109
        gr._genome_fasta = pysam.FastaFile(str(fa_path))
        gr._genome_dir = tmp_path
        gr.ensembl = MagicMock()

        mock_gene = MagicMock()
        mock_gene.contig = "1"
        mock_gene.start = 5
        mock_gene.end = 20
        mock_gene.strand = "+"
        mock_gene.gene_id = "ENSG00000000001"

        mock_exon1 = MagicMock()
        mock_exon1.start = 5
        mock_exon1.end = 10
        mock_exon2 = MagicMock()
        mock_exon2.start = 15
        mock_exon2.end = 20
        mock_gene.exons = [mock_exon1, mock_exon2]

        gr.ensembl.gene_by_id.return_value = mock_gene
        return gr

    def test_full_sequence(self, resolver_with_genome):
        seq = resolver_with_genome._get_local_indexed_sequence("ENSG00000000001", region="full")
        assert seq is not None
        assert len(seq) == 16
        assert all(c in "ACGTacgtNn" for c in seq)

    def test_exon_sequences(self, resolver_with_genome):
        seq = resolver_with_genome._get_local_indexed_sequence("ENSG00000000001", region="exons")
        assert seq is not None
        assert len(seq) > 0

    def test_intron_sequences(self, resolver_with_genome):
        seq = resolver_with_genome._get_local_indexed_sequence("ENSG00000000001", region="introns")
        assert seq is not None

    def test_unknown_gene_returns_none(self, resolver_with_genome):
        resolver_with_genome.ensembl.gene_by_id.side_effect = ValueError("Not found")
        seq = resolver_with_genome._get_local_indexed_sequence("ENSG99999999999")
        assert seq is None

    def test_unknown_chrom_returns_none(self, resolver_with_genome):
        mock_gene = MagicMock()
        mock_gene.contig = "chrUn_gl000220"
        mock_gene.start = 1
        mock_gene.end = 100
        mock_gene.strand = "+"
        mock_gene.gene_id = "ENSG00000000002"
        resolver_with_genome.ensembl.gene_by_id.return_value = mock_gene

        seq = resolver_with_genome._get_local_indexed_sequence("ENSG00000000002")
        assert seq is None


class TestGeneResolverReverseComplement:
    def test_reverse_complement(self):
        from embpy.resources.gene_resolver import GeneResolver

        assert GeneResolver._reverse_complement("ATCG") == "CGAT"
        assert GeneResolver._reverse_complement("AAAA") == "TTTT"
        assert GeneResolver._reverse_complement("GCGC") == "GCGC"
        assert GeneResolver._reverse_complement("") == ""


class TestGeneResolverLocalFirstFallback:
    @patch("embpy.resources.gene_resolver.GeneResolver.__init__", return_value=None)
    def test_get_dna_sequence_tries_local_first(self, mock_init):
        from embpy.resources.gene_resolver import GeneResolver

        gr = GeneResolver.__new__(GeneResolver)
        gr.species = "human"
        gr._genome_fasta = MagicMock()
        gr._genome_dir = Path("/tmp")
        gr.ensembl = MagicMock()

        with patch.object(gr, "_load_genome_if_available", return_value=True), \
             patch.object(gr, "symbol_to_ensembl", return_value="ENSG00000141510"), \
             patch.object(gr, "_get_local_indexed_sequence", return_value="ATCGATCG"):
            seq = gr.get_dna_sequence("TP53", id_type="symbol", organism="human")
            assert seq == "ATCGATCG"

    @patch("embpy.resources.gene_resolver.GeneResolver.__init__", return_value=None)
    def test_falls_back_to_api_when_no_genome(self, mock_init):
        from embpy.resources.gene_resolver import GeneResolver

        gr = GeneResolver.__new__(GeneResolver)
        gr.species = "human"
        gr._genome_fasta = None
        gr._genome_dir = None
        gr.ensembl = None

        with patch.object(gr, "_load_genome_if_available", return_value=False), \
             patch("embpy.resources.gene_resolver._ensembl_get") as mock_api:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"id": "ENSG00000141510"}
            mock_resp.text = "ATCGATCG"
            mock_resp.raise_for_status = MagicMock()
            mock_api.return_value = mock_resp

            seq = gr.get_dna_sequence("TP53", id_type="symbol", organism="human")
            assert mock_api.called


class TestProteinResolverDownloadProteome:
    def test_proteome_mapping_has_species(self):
        from embpy.resources.protein_resolver import ProteinResolver

        assert "human" in ProteinResolver._UNIPROT_PROTEOME
        assert "mouse" in ProteinResolver._UNIPROT_PROTEOME
        assert ProteinResolver._UNIPROT_PROTEOME["human"][1] == "9606"

    def test_unsupported_species_raises(self):
        from embpy.resources.protein_resolver import ProteinResolver

        pr = ProteinResolver(organism="platypus")
        with pytest.raises(ValueError, match="Unsupported"):
            pr.download_proteome()


class TestProteinResolverLocalLookup:
    @pytest.fixture
    def resolver_with_proteome(self):
        from embpy.resources.protein_resolver import ProteinResolver

        pr = ProteinResolver(organism="human")
        pr._local_proteome = {
            "P04637": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPS",
            "P38398": "MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTK",
        }
        pr._local_gene_to_acc = {
            "TP53": "P04637",
            "BRCA1": "P38398",
        }
        return pr

    def test_local_lookup_by_symbol(self, resolver_with_proteome):
        seq = resolver_with_proteome._get_local_protein_sequence("TP53", "symbol")
        assert seq is not None
        assert seq.startswith("MEEPQ")

    def test_local_lookup_by_accession(self, resolver_with_proteome):
        seq = resolver_with_proteome._get_local_protein_sequence("P04637", "uniprot_id")
        assert seq is not None

    def test_local_lookup_unknown(self, resolver_with_proteome):
        seq = resolver_with_proteome._get_local_protein_sequence("FAKEGENE", "symbol")
        assert seq is None

    def test_get_canonical_tries_local_first(self, resolver_with_proteome):
        with patch.object(resolver_with_proteome, "_load_proteome_if_available", return_value=True):
            seq = resolver_with_proteome.get_canonical_sequence("TP53", id_type="symbol")
            assert seq is not None
            assert "MEEPQ" in seq
