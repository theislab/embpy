"""Tests for GeneResolver with mocked HTTP calls and pyensembl."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from embpy.resources.gene_resolver import GeneResolver, _looks_like_smiles, detect_identifier_type


@pytest.fixture
def resolver():
    """GeneResolver with pyensembl disabled (API-only mode)."""
    r = GeneResolver.__new__(GeneResolver)
    r.ensembl = None
    r.mart_file = None
    r.chrom_folder = None
    r.release_version = 109
    r.species = "human"
    return r


class TestGetDnaSequence:
    def test_by_symbol(self, resolver):
        lookup_mock = MagicMock()
        lookup_mock.status_code = 200
        lookup_mock.raise_for_status = MagicMock()
        lookup_mock.json.return_value = {"id": "ENSG00000141510"}

        seq_mock = MagicMock()
        seq_mock.status_code = 200
        seq_mock.raise_for_status = MagicMock()
        seq_mock.text = "ACGTACGTACGT"

        with patch("embpy.resources.gene_resolver.requests.get", side_effect=[lookup_mock, seq_mock]):
            result = resolver.get_dna_sequence("TP53", "symbol")
            assert result == "ACGTACGTACGT"

    def test_by_ensembl_id(self, resolver):
        lookup_mock = MagicMock()
        lookup_mock.status_code = 200
        lookup_mock.raise_for_status = MagicMock()
        lookup_mock.json.return_value = {"id": "ENSG00000141510"}

        seq_mock = MagicMock()
        seq_mock.status_code = 200
        seq_mock.raise_for_status = MagicMock()
        seq_mock.text = "ACGTACGTACGT"

        with patch("embpy.resources.gene_resolver.requests.get", side_effect=[lookup_mock, seq_mock]):
            result = resolver.get_dna_sequence("ENSG00000141510", "ensembl_id")
            assert result == "ACGTACGTACGT"

    def test_unsupported_id_type(self, resolver):
        result = resolver.get_dna_sequence("TP53", "invalid_type")
        assert result is None

    def test_api_error_returns_none(self, resolver):
        import requests

        mock = MagicMock()
        mock.raise_for_status.side_effect = requests.RequestException("API down")

        with patch("embpy.resources.gene_resolver.requests.get", return_value=mock):
            result = resolver.get_dna_sequence("TP53", "symbol")
            assert result is None

    def test_no_ensembl_id_resolved(self, resolver):
        mock = MagicMock()
        mock.status_code = 200
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {}

        with patch("embpy.resources.gene_resolver.requests.get", return_value=mock):
            result = resolver.get_dna_sequence("TP53", "symbol")
            assert result is None


class TestGetProteinSequence:
    def test_by_uniprot_id(self, resolver):
        fasta_mock = MagicMock()
        fasta_mock.status_code = 200
        fasta_mock.raise_for_status = MagicMock()
        fasta_mock.text = ">sp|P04637|P53_HUMAN\nMEEPQSDPSVEPPLSQ\nETFSDLWKLLPENNVL"

        with patch("embpy.resources.gene_resolver.requests.get", return_value=fasta_mock):
            result = resolver.get_protein_sequence("P04637", "uniprot_id")
            assert result is not None
            assert result.startswith("MEEPQ")
            assert "\n" not in result

    def test_by_symbol(self, resolver):
        query_mock = MagicMock()
        query_mock.status_code = 200
        query_mock.raise_for_status = MagicMock()
        query_mock.json.return_value = {"hits": [{"uniprot": {"Swiss-Prot": "P04637"}}]}

        fasta_mock = MagicMock()
        fasta_mock.status_code = 200
        fasta_mock.raise_for_status = MagicMock()
        fasta_mock.text = ">sp|P04637\nMTEYKLVVVGAGGVGKS"

        with patch(
            "embpy.resources.gene_resolver.requests.get",
            side_effect=[query_mock, fasta_mock],
        ):
            result = resolver.get_protein_sequence("TP53", "symbol")
            assert result is not None
            assert result.startswith("MTEYKLVVVG")

    def test_no_uniprot_found(self, resolver):
        mock = MagicMock()
        mock.status_code = 200
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"hits": [{"uniprot": {}}]}

        with patch("embpy.resources.gene_resolver.requests.get", return_value=mock):
            result = resolver.get_protein_sequence("FAKE", "symbol")
            assert result is None


class TestGetGeneDescription:
    def test_success(self, resolver):
        mock = MagicMock()
        mock.status_code = 200
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {
            "hits": [
                {
                    "symbol": "TP53",
                    "name": "tumor protein p53",
                    "summary": "This gene encodes a tumor suppressor protein.",
                }
            ]
        }

        with patch("embpy.resources.gene_resolver.requests.get", return_value=mock):
            result = resolver.get_gene_description("TP53", "symbol")
            assert result is not None
            assert "TP53" in result
            assert "tumor" in result.lower()

    def test_no_hits(self, resolver):
        mock = MagicMock()
        mock.status_code = 200
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"hits": []}

        with patch("embpy.resources.gene_resolver.requests.get", return_value=mock):
            result = resolver.get_gene_description("FAKEGENE", "symbol")
            assert result is None

    def test_unsupported_id_type(self, resolver):
        result = resolver.get_gene_description("TP53", "invalid_type")
        assert result is None


class TestSymbolToEnsembl:
    def test_via_mygene(self, resolver):
        mock = MagicMock()
        mock.status_code = 200
        mock.ok = True
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"hits": [{"ensembl": {"gene": "ENSG00000141510"}}]}

        with patch("embpy.resources.gene_resolver.requests.get", return_value=mock):
            result = resolver.symbol_to_ensembl("TP53")
            assert result == "ENSG00000141510"

    def test_no_result_returns_none(self, resolver):
        mock = MagicMock()
        mock.status_code = 200
        mock.ok = False
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"hits": []}

        with patch("embpy.resources.gene_resolver.requests.get", return_value=mock):
            result = resolver.symbol_to_ensembl("FAKEGENE")
            assert result is None


class TestEnsemblToSymbol:
    def test_via_mygene(self, resolver):
        mock = MagicMock()
        mock.status_code = 200
        mock.ok = True
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"hits": [{"symbol": "TP53"}]}

        with patch("embpy.resources.gene_resolver.requests.get", return_value=mock):
            result = resolver.ensembl_to_symbol("ENSG00000141510")
            assert result == "TP53"

    def test_strips_version(self, resolver):
        mock = MagicMock()
        mock.status_code = 200
        mock.ok = True
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"hits": [{"symbol": "TP53"}]}

        with patch("embpy.resources.gene_resolver.requests.get", return_value=mock):
            result = resolver.ensembl_to_symbol("ENSG00000141510.12")
            assert result == "TP53"


class TestBatchMappings:
    def test_symbols_to_ensembl_batch(self, resolver):
        resolver.symbol_to_ensembl = MagicMock(side_effect=["ENSG1", "ENSG2"])
        result = resolver.symbols_to_ensembl_batch(["TP53", "BRCA1"])
        assert result == {"TP53": "ENSG1", "BRCA1": "ENSG2"}

    def test_ensembl_to_symbols_batch(self, resolver):
        resolver.ensembl_to_symbol = MagicMock(side_effect=["TP53", "BRCA1"])
        result = resolver.ensembl_to_symbols_batch(["ENSG1", "ENSG2"])
        assert result == {"ENSG1": "TP53", "ENSG2": "BRCA1"}


class TestGetLocalDnaSequence:
    def test_requires_mart_file(self, resolver):
        with pytest.raises(ValueError, match="mart_file"):
            resolver.get_local_dna_sequence("TP53", "symbol")


# =====================================================================
# _looks_like_smiles  (private helper, tested here since it lives in gene_resolver)
# =====================================================================
class TestLooksLikeSmiles:
    def test_true_for_special_chars(self):
        assert _looks_like_smiles("CC(=O)O") is True
        assert _looks_like_smiles("CC[O-]") is True
        assert _looks_like_smiles("C/C=C/C") is True

    def test_true_for_ring_closure(self):
        assert _looks_like_smiles("c1ccccc1") is True

    def test_false_for_simple_letters(self):
        assert _looks_like_smiles("CCO") is False

    def test_false_for_gene_symbols(self):
        assert _looks_like_smiles("TP53") is False
        assert _looks_like_smiles("BRCA1") is False

    def test_false_for_short(self):
        assert _looks_like_smiles("CC") is False

    def test_false_for_ensembl(self):
        assert _looks_like_smiles("ENSG00000141510") is False


# =====================================================================
# detect_identifier_type  (unified: DNA, Ensembl, symbol, SMILES, protein)
# =====================================================================
class TestDetectIdentifierType:
    def test_ensembl_id(self):
        assert detect_identifier_type("ENSG00000141510") == "ensembl_id"

    def test_ensembl_id_with_version(self):
        assert detect_identifier_type("ENSG00000141510.12") == "ensembl_id"

    def test_dna_sequence(self):
        assert detect_identifier_type("ACGTACGTACGTACGTACGTACGT") == "dna_sequence"

    def test_short_dna_is_symbol(self):
        assert detect_identifier_type("ACGT") == "symbol"

    def test_gene_symbol(self):
        assert detect_identifier_type("TP53") == "symbol"

    def test_gene_symbol_brca1(self):
        assert detect_identifier_type("BRCA1") == "symbol"

    def test_mixed_case_dna(self):
        assert detect_identifier_type("acgtACGTacgtACGTnnnn") == "dna_sequence"

    def test_smiles_with_special_chars(self):
        assert detect_identifier_type("CC(=O)O") == "smiles"

    def test_smiles_with_ring(self):
        assert detect_identifier_type("c1ccccc1") == "smiles"

    def test_protein_sequence(self):
        assert detect_identifier_type("MTEYKLVVVGAGGVGKSALT") == "protein_sequence"

    def test_short_protein_is_symbol(self):
        assert detect_identifier_type("MTEYK") == "symbol"

    def test_simple_letters_no_special_is_symbol(self):
        # "CCO" – no SMILES special chars, too short for protein/DNA
        assert detect_identifier_type("CCO") == "symbol"


# =====================================================================
# load_sequences_from_biomart
# =====================================================================
class TestLoadSequencesFromBiomart:
    def _make_mart_csv(self, tmpdir):
        df = pd.DataFrame(
            {
                "Gene stable ID": ["ENSG001", "ENSG002"],
                "Chromosome/scaffold name": ["1", "1"],
                "Gene start (bp)": [10, 50],
                "Gene end (bp)": [20, 60],
                "Gene type": ["protein_coding", "lncRNA"],
            }
        )
        path = os.path.join(tmpdir, "mart.csv")
        df.to_csv(path, index=False)
        return path

    def _make_chr_fasta(self, tmpdir):
        chrom_dir = os.path.join(tmpdir, "genome")
        os.makedirs(chrom_dir, exist_ok=True)
        fasta_path = os.path.join(chrom_dir, "chr1.fa")
        with open(fasta_path, "w") as f:
            f.write(">chr1\n")
            f.write("A" * 100 + "\n")
        return chrom_dir

    def test_loads_all_genes(self, resolver):
        with tempfile.TemporaryDirectory() as tmpdir:
            mart = self._make_mart_csv(tmpdir)
            genome = self._make_chr_fasta(tmpdir)
            seqs = resolver.load_sequences_from_biomart(mart_file=mart, chrom_folder=genome)
            assert len(seqs) == 2
            assert "ENSG001" in seqs
            assert len(seqs["ENSG001"]) == 11  # 10..20 inclusive → 1-based [9:20] = 11 chars

    def test_filters_by_biotype(self, resolver):
        with tempfile.TemporaryDirectory() as tmpdir:
            mart = self._make_mart_csv(tmpdir)
            genome = self._make_chr_fasta(tmpdir)
            seqs = resolver.load_sequences_from_biomart(mart_file=mart, chrom_folder=genome, biotype="protein_coding")
            assert len(seqs) == 1
            assert "ENSG001" in seqs

    def test_raises_without_mart_file(self, resolver):
        with pytest.raises(ValueError, match="mart_file"):
            resolver.load_sequences_from_biomart()

    def test_missing_chr_skips(self, resolver):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame(
                {
                    "Gene stable ID": ["ENSG999"],
                    "Chromosome/scaffold name": ["99"],
                    "Gene start (bp)": [10],
                    "Gene end (bp)": [20],
                }
            )
            mart = os.path.join(tmpdir, "mart.csv")
            df.to_csv(mart, index=False)
            genome = os.path.join(tmpdir, "genome")
            os.makedirs(genome, exist_ok=True)
            seqs = resolver.load_sequences_from_biomart(mart_file=mart, chrom_folder=genome)
            assert len(seqs) == 0


# =====================================================================
# load_genes_from_adata
# =====================================================================
class TestLoadGenesFromAdata:
    def test_explicit_column(self, resolver):
        import anndata as ad

        with tempfile.TemporaryDirectory() as tmpdir:
            var = pd.DataFrame({"my_genes": ["TP53", "BRCA1"]}, index=["g0", "g1"])
            adata = ad.AnnData(np.zeros((3, 2)), var=var)
            path = os.path.join(tmpdir, "test.h5ad")
            adata.write_h5ad(path)
            genes = resolver.load_genes_from_adata(path, column="my_genes")
            assert genes == ["TP53", "BRCA1"]

    def test_auto_detects_ensembl_id(self, resolver):
        import anndata as ad

        with tempfile.TemporaryDirectory() as tmpdir:
            var = pd.DataFrame({"ensembl_id": ["ENSG001", "ENSG002"]}, index=["g0", "g1"])
            adata = ad.AnnData(np.zeros((3, 2)), var=var)
            path = os.path.join(tmpdir, "test.h5ad")
            adata.write_h5ad(path)
            genes = resolver.load_genes_from_adata(path)
            assert genes == ["ENSG001", "ENSG002"]

    def test_auto_detects_gene_name(self, resolver):
        import anndata as ad

        with tempfile.TemporaryDirectory() as tmpdir:
            var = pd.DataFrame({"gene_name": ["TP53", "BRCA1"]}, index=["g0", "g1"])
            adata = ad.AnnData(np.zeros((3, 2)), var=var)
            path = os.path.join(tmpdir, "test.h5ad")
            adata.write_h5ad(path)
            genes = resolver.load_genes_from_adata(path)
            assert genes == ["TP53", "BRCA1"]

    def test_falls_back_to_index(self, resolver):
        import anndata as ad

        with tempfile.TemporaryDirectory() as tmpdir:
            var = pd.DataFrame(index=["GeneA", "GeneB"])
            adata = ad.AnnData(np.zeros((3, 2)), var=var)
            path = os.path.join(tmpdir, "test.h5ad")
            adata.write_h5ad(path)
            genes = resolver.load_genes_from_adata(path)
            assert genes == ["GeneA", "GeneB"]


# =====================================================================
# get_protein_sequences_batch
# =====================================================================
class TestGetProteinSequencesBatch:
    def test_batch_returns_dict(self, resolver):
        resolver.get_protein_sequence = MagicMock(side_effect=["MSEQ1", "MSEQ2", None])
        result = resolver.get_protein_sequences_batch(["A", "B", "C"], id_type="ensembl_id")
        assert result == {"A": "MSEQ1", "B": "MSEQ2"}
        assert "C" not in result

    def test_empty_input(self, resolver):
        resolver.get_protein_sequence = MagicMock()
        result = resolver.get_protein_sequences_batch([])
        assert result == {}


# =====================================================================
# get_all_local_protein_sequences
# =====================================================================
class TestGetAllLocalProteinSequences:
    def test_reads_from_mart_and_batches(self, resolver):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame(
                {
                    "Gene stable ID": ["ENSG001", "ENSG002"],
                    "Gene type": ["protein_coding", "protein_coding"],
                }
            )
            mart = os.path.join(tmpdir, "mart.csv")
            df.to_csv(mart, index=False)

            resolver.get_protein_sequence = MagicMock(side_effect=["MSEQ1", "MSEQ2"])
            result = resolver.get_all_local_protein_sequences(mart_file=mart)
            assert len(result) == 2

    def test_raises_without_mart(self, resolver):
        with pytest.raises(ValueError, match="mart_file"):
            resolver.get_all_local_protein_sequences()
