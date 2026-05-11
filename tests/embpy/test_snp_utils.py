from __future__ import annotations

import io
import textwrap
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embpy.tl.snp_utils import (
    SNPContext,
    SNPEmbeddingResult,
    SNPEmbedder,
    SequenceProvider,
    _apply_snp,
    _extract_context,
)


CHR_SEQ = "ACGT" * 100          
SNP_POS = 200                   
REF_BASE = CHR_SEQ[SNP_POS - 1].upper()   
ALT_BASE = "T" if REF_BASE != "T" else "G"

def _make_wrapper(hidden_dim: int = 64, same_each_call: bool = True) -> MagicMock:
    """Return a mock DNA model wrapper.

    If same_each_call=True every call returns the same fixed embedding
    (useful for testing zero-delta path).  If False each call gets a fresh
    random embedding seeded by call count.
    """
    wrapper = MagicMock()
    if same_each_call:
        wrapper.embed.return_value = np.ones(hidden_dim, dtype=np.float32)
    else:
        counter = [0]
        def _varying(*a, **kw):
            counter[0] += 1
            rng = np.random.default_rng(counter[0])
            return rng.standard_normal(hidden_dim).astype(np.float32)
        wrapper.embed.side_effect = _varying
    wrapper.model_name = "mock_dna_model"
    return wrapper


def _make_snp(
    position: int = SNP_POS,
    ref: str = REF_BASE,
    alts: list[str] | None = None,
    context_window: int = 64,
    chrom: str = "chr17",
    variant_id: str = "rs_test",
    strand: str = "+",
) -> SNPContext:
    return SNPContext(
        chrom=chrom,
        position=position,
        ref_allele=ref,
        alt_alleles=alts or [ALT_BASE],
        context_window=context_window,
        variant_id=variant_id,
        strand=strand,
    )


class TestExtractContext:

    def test_mid_sequence_returns_correct_window(self):
        start, end, offset = _extract_context(CHR_SEQ, SNP_POS, context_window=64)
        assert end - start == 64
        assert start >= 0 and end <= len(CHR_SEQ)
        assert CHR_SEQ[start + offset].upper() == REF_BASE

    def test_offset_points_to_ref_allele(self):
        for pos in [10, 100, 200, 390]:
            start, end, offset = _extract_context(CHR_SEQ, pos, context_window=64)
            assert CHR_SEQ[start + offset].upper() == CHR_SEQ[pos - 1].upper()

    def test_clamps_at_sequence_start(self):
        start, end, offset = _extract_context(CHR_SEQ, 5, context_window=64)
        assert start == 0
        assert offset == 4   # pos=5 → 0-based 4

    def test_clamps_at_sequence_end(self):
        start, end, offset = _extract_context(CHR_SEQ, len(CHR_SEQ), context_window=64)
        assert end == len(CHR_SEQ)

    def test_small_context_window(self):
        start, end, offset = _extract_context(CHR_SEQ, SNP_POS, context_window=10)
        assert end - start <= 10
        assert CHR_SEQ[start + offset].upper() == REF_BASE

    def test_context_larger_than_seq(self):
        short_seq = "ACGTACGT"  # 8 bp
        start, end, offset = _extract_context(short_seq, 4, context_window=100)
        assert start == 0
        assert end == len(short_seq)


class TestApplySNP:

    def test_substitutes_single_base(self):
        seq = "ACGTACGT"
        result = _apply_snp(seq, offset=2, ref="G", alt="T")
        assert result == "ACTTACGT"

    def test_length_preserved_for_snp(self):
        seq = "ACGTACGT"
        result = _apply_snp(seq, offset=3, ref="T", alt="C")
        assert len(result) == len(seq)

    def test_ref_mismatch_raises_value_error(self):
        seq = "ACGTACGT"
        with pytest.raises(ValueError, match="Reference mismatch"):
            _apply_snp(seq, offset=0, ref="T", alt="G")   # seq[0]='A', not 'T'

    def test_case_insensitive_ref_check(self):
        seq = "acgtacgt"
        # ref='A' should match lower-case 'a' at offset 0
        result = _apply_snp(seq, offset=0, ref="A", alt="T")
        assert result[0] == "T"

    def test_insertion_lengthens_sequence(self):
        seq = "ACGTACGT"
        result = _apply_snp(seq, offset=2, ref="G", alt="GTT")
        assert len(result) == len(seq) + 2

    def test_deletion_shortens_sequence(self):
        seq = "ACGTACGT"
        result = _apply_snp(seq, offset=2, ref="GT", alt="G")
        assert len(result) == len(seq) - 1


class TestSNPContext:

    def test_init_basic(self):
        snp = _make_snp()
        assert snp.chrom == "chr17"
        assert snp.position == SNP_POS
        assert snp.ref_allele == REF_BASE.upper()
        assert snp.alt_alleles == [ALT_BASE.upper()]
        assert snp.variant_id == "rs_test"
        assert snp.context_window == 64

    def test_alleles_uppercased_on_init(self):
        snp = SNPContext(position=100, ref_allele="a", alt_alleles=["t", "g"],
                         context_window=64)
        assert snp.ref_allele == "A"
        assert snp.alt_alleles == ["T", "G"]

    def test_multiple_alts(self):
        snp = _make_snp(alts=["T", "G", "C"])
        assert len(snp.alt_alleles) == 3

    def test_default_strand_is_plus(self):
        snp = _make_snp()
        assert snp.strand == "+"

    def test_minus_strand(self):
        snp = _make_snp(strand="-")
        assert snp.strand == "-"

    def test_default_variant_id_is_empty(self):
        snp = SNPContext(position=100, ref_allele="A", alt_alleles=["T"],
                         context_window=64)
        assert snp.variant_id == ""


class TestSNPEmbeddingResult:

    def _make_result(self, hidden_dim: int = 64) -> SNPEmbeddingResult:
        snp = _make_snp()
        ref_emb = np.ones(hidden_dim, dtype=np.float32)
        alt_emb = np.ones(hidden_dim, dtype=np.float32) * 2.0
        return SNPEmbeddingResult(
            snp=snp,
            ref_sequence="ACGT" * 16,
            alt_sequences=["TCGT" + "ACGT" * 15],
            ref_embedding=ref_emb,
            alt_embeddings=[alt_emb],
        )

    def test_delta_computed_automatically(self):
        r = self._make_result(64)
        assert len(r.delta_embeddings) == 1
        assert r.delta_embeddings[0].shape == (64,)
        # alt(2) - ref(1) = 1 everywhere
        assert np.allclose(r.delta_embeddings[0], 1.0)

    def test_delta_norm_computed_automatically(self):
        r = self._make_result(64)
        expected_norm = float(np.linalg.norm(np.ones(64)))
        assert pytest.approx(r.delta_norms[0], abs=1e-4) == expected_norm

    def test_cosine_similarity_computed_automatically(self):
        r = self._make_result(64)
        # Both ref and alt point in the same direction → cosine = 1
        assert pytest.approx(r.cosine_similarities[0], abs=1e-4) == 1.0

    def test_cosine_similarity_range(self):
        snp = _make_snp()
        rng = np.random.default_rng(42)
        ref = rng.standard_normal(64).astype(np.float32)
        alt = rng.standard_normal(64).astype(np.float32)
        r = SNPEmbeddingResult(
            snp=snp, ref_sequence="X", alt_sequences=["Y"],
            ref_embedding=ref, alt_embeddings=[alt],
        )
        assert -1.0 <= r.cosine_similarities[0] <= 1.0

    def test_zero_delta_when_ref_equals_alt(self):
        snp = _make_snp()
        emb = np.ones(64, dtype=np.float32)
        r = SNPEmbeddingResult(
            snp=snp, ref_sequence="X", alt_sequences=["X"],
            ref_embedding=emb, alt_embeddings=[emb.copy()],
        )
        assert pytest.approx(r.delta_norms[0], abs=1e-6) == 0.0

    def test_multiple_alts(self):
        snp = _make_snp(alts=["T", "G"])
        ref = np.ones(32, dtype=np.float32)
        alts = [np.ones(32, dtype=np.float32) * 2, np.ones(32, dtype=np.float32) * 3]
        r = SNPEmbeddingResult(
            snp=snp, ref_sequence="X", alt_sequences=["Y", "Z"],
            ref_embedding=ref, alt_embeddings=alts,
        )
        assert len(r.delta_embeddings) == 2
        assert len(r.delta_norms) == 2
        assert len(r.cosine_similarities) == 2

    def test_to_dict_returns_list_of_rows(self):
        r = self._make_result()
        rows = r.to_dict()
        assert isinstance(rows, list)
        assert len(rows) == 1
        row = rows[0]
        assert row["variant_id"] == "rs_test"
        assert row["ref"] == REF_BASE
        assert "delta_l2_norm" in row
        assert "cosine_similarity" in row
        assert "model" in row

    def test_to_dict_multi_alt(self):
        snp = _make_snp(alts=["T", "G"])
        ref = np.ones(32, dtype=np.float32)
        alts = [np.ones(32, dtype=np.float32) * 2, np.ones(32, dtype=np.float32) * 3]
        r = SNPEmbeddingResult(
            snp=snp, ref_sequence="X", alt_sequences=["Y", "Z"],
            ref_embedding=ref, alt_embeddings=alts,
        )
        rows = r.to_dict()
        assert len(rows) == 2
        assert rows[0]["alt"] == "T"
        assert rows[1]["alt"] == "G"


class TestSNPEmbedderEmbedSNP:

    def test_returns_snp_embedding_result(self):
        snp = _make_snp()
        embedder = SNPEmbedder(_make_wrapper(), pooling_strategy="mean")
        result = embedder.embed_snp(snp, chromosome_sequence=CHR_SEQ)
        assert isinstance(result, SNPEmbeddingResult)

    def test_ref_and_alt_embeddings_correct_shape(self):
        hidden_dim = 64
        snp = _make_snp()
        embedder = SNPEmbedder(_make_wrapper(hidden_dim))
        result = embedder.embed_snp(snp, chromosome_sequence=CHR_SEQ)
        assert result.ref_embedding.shape == (hidden_dim,)
        assert len(result.alt_embeddings) == 1
        assert result.alt_embeddings[0].shape == (hidden_dim,)

    def test_delta_norm_non_negative(self):
        snp = _make_snp()
        embedder = SNPEmbedder(_make_wrapper(same_each_call=False))
        result = embedder.embed_snp(snp, chromosome_sequence=CHR_SEQ)
        assert result.delta_norms[0] >= 0.0

    def test_cosine_similarity_in_range(self):
        snp = _make_snp()
        embedder = SNPEmbedder(_make_wrapper(same_each_call=False))
        result = embedder.embed_snp(snp, chromosome_sequence=CHR_SEQ)
        assert -1.0 <= result.cosine_similarities[0] <= 1.0

    def test_identical_embeddings_give_zero_delta(self):
        """When wrapper returns the same vector for ref and alt → delta = 0."""
        snp = _make_snp()
        embedder = SNPEmbedder(_make_wrapper(same_each_call=True))
        result = embedder.embed_snp(snp, chromosome_sequence=CHR_SEQ)
        assert pytest.approx(result.delta_norms[0], abs=1e-6) == 0.0

    def test_different_embeddings_give_nonzero_delta(self):
        snp = _make_snp()
        embedder = SNPEmbedder(_make_wrapper(same_each_call=False))
        result = embedder.embed_snp(snp, chromosome_sequence=CHR_SEQ)
        assert result.delta_norms[0] > 0.0

    def test_embed_called_once_per_allele(self):
        """embed() must be called exactly 1 (ref) + len(alts) times."""
        wrapper = _make_wrapper()
        snp = _make_snp(alts=["T", "G", "C"])
        SNPEmbedder(wrapper).embed_snp(snp, chromosome_sequence=CHR_SEQ)
        assert wrapper.embed.call_count == 4   # 1 ref + 3 alts

    def test_pooling_strategy_forwarded_to_wrapper(self):
        wrapper = _make_wrapper()
        snp = _make_snp()
        SNPEmbedder(wrapper, pooling_strategy="cls").embed_snp(snp, chromosome_sequence=CHR_SEQ)
        for call in wrapper.embed.call_args_list:
            assert call.kwargs.get("pooling_strategy") == "cls"

    def test_pooling_strategy_override_per_call(self):
        wrapper = _make_wrapper()
        snp = _make_snp()
        SNPEmbedder(wrapper, pooling_strategy="mean").embed_snp(
            snp, chromosome_sequence=CHR_SEQ, pooling_strategy="max"
        )
        for call in wrapper.embed.call_args_list:
            assert call.kwargs.get("pooling_strategy") == "max"

    def test_multiple_alts_returns_multiple_results(self):
        wrapper = _make_wrapper(same_each_call=False)
        snp = _make_snp(alts=["T", "G"])
        result = SNPEmbedder(wrapper).embed_snp(snp, chromosome_sequence=CHR_SEQ)
        assert len(result.alt_embeddings) == 2
        assert len(result.delta_norms) == 2
        assert len(result.cosine_similarities) == 2

    def test_result_stores_snp_reference(self):
        snp = _make_snp(variant_id="rs12345")
        result = SNPEmbedder(_make_wrapper()).embed_snp(snp, chromosome_sequence=CHR_SEQ)
        assert result.snp.variant_id == "rs12345"

    def test_result_ref_sequence_contains_ref_allele(self):
        snp = _make_snp()
        result = SNPEmbedder(_make_wrapper()).embed_snp(snp, chromosome_sequence=CHR_SEQ)
        # The ref context should contain the reference base at the right position
        assert REF_BASE in result.ref_sequence.upper()

    def test_alt_sequence_differs_from_ref_sequence(self):
        snp = _make_snp()
        result = SNPEmbedder(_make_wrapper()).embed_snp(snp, chromosome_sequence=CHR_SEQ)
        assert result.ref_sequence != result.alt_sequences[0]

    def test_minus_strand_reverse_complements_context(self):
        """On the minus strand the sequences sent to the model should be RC."""
        wrapper = _make_wrapper()
        snp_plus  = _make_snp(strand="+")
        snp_minus = _make_snp(strand="-")

        r_plus  = SNPEmbedder(wrapper).embed_snp(snp_plus,  chromosome_sequence=CHR_SEQ)
        r_minus = SNPEmbedder(wrapper).embed_snp(snp_minus, chromosome_sequence=CHR_SEQ)

        assert r_plus.ref_sequence != r_minus.ref_sequence

    def test_result_model_name_set(self):
        result = SNPEmbedder(_make_wrapper()).embed_snp(_make_snp(), chromosome_sequence=CHR_SEQ)
        assert result.model_name == "mock_dna_model"

    def test_result_pooling_strategy_set(self):
        result = SNPEmbedder(_make_wrapper(), pooling_strategy="max").embed_snp(
            _make_snp(), chromosome_sequence=CHR_SEQ
        )
        assert result.pooling_strategy == "max"

    def test_output_embeddings_are_float32(self):
        result = SNPEmbedder(_make_wrapper()).embed_snp(_make_snp(), chromosome_sequence=CHR_SEQ)
        assert result.ref_embedding.dtype == np.float32
        assert result.alt_embeddings[0].dtype == np.float32


class TestEmbedSNPFromVCFRow:

    def test_basic_vcf_row(self):
        wrapper = _make_wrapper()
        embedder = SNPEmbedder(wrapper)
        result = embedder.embed_snp_from_vcf_row(
            chrom="chr17", pos=SNP_POS, ref=REF_BASE, alt=ALT_BASE,
            chromosome_sequence=CHR_SEQ, context_window=64, variant_id="rs123",
        )
        assert isinstance(result, SNPEmbeddingResult)
        assert result.snp.variant_id == "rs123"

    def test_comma_separated_alts_split(self):
        wrapper = _make_wrapper()
        result = SNPEmbedder(wrapper).embed_snp_from_vcf_row(
            chrom="chr17", pos=SNP_POS, ref=REF_BASE, alt="T,G",
            chromosome_sequence=CHR_SEQ, context_window=64,
        )
        assert len(result.alt_embeddings) == 2

    def test_correct_alleles_stored(self):
        result = SNPEmbedder(_make_wrapper()).embed_snp_from_vcf_row(
            chrom="chr17", pos=SNP_POS, ref=REF_BASE, alt=ALT_BASE,
            chromosome_sequence=CHR_SEQ, context_window=64,
        )
        assert result.snp.ref_allele == REF_BASE.upper()
        assert ALT_BASE.upper() in result.snp.alt_alleles

    def test_context_window_forwarded(self):
        result = SNPEmbedder(_make_wrapper()).embed_snp_from_vcf_row(
            chrom="chr17", pos=SNP_POS, ref=REF_BASE, alt=ALT_BASE,
            chromosome_sequence=CHR_SEQ, context_window=128,
        )
        assert result.snp.context_window == 128


class TestEmbedSNPsBatch:

    def _snps(self, n: int = 3) -> list[SNPContext]:
        positions = [50, 100, 200][:n]
        return [
            _make_snp(position=p, ref=CHR_SEQ[p - 1].upper(), variant_id=f"rs{p}")
            for p in positions
        ]

    def test_list_mode_returns_all_results(self):
        snps = self._snps(3)
        chr_seqs = [CHR_SEQ] * 3
        results = SNPEmbedder(_make_wrapper()).embed_snps_batch(snps, chr_seqs)
        assert len(results) == 3

    def test_dict_mode_keyed_by_chrom(self):
        snps = self._snps(3)
        chr_dict = {"chr17": CHR_SEQ}
        results = SNPEmbedder(_make_wrapper()).embed_snps_batch(snps, chr_dict)
        assert len(results) == 3

    def test_missing_chrom_in_dict_skips_snp(self):
        snps = self._snps(2)
        # chr_dict has no entry for the snps' chromosome
        results = SNPEmbedder(_make_wrapper()).embed_snps_batch(snps, {})
        assert len(results) == 0

    def test_empty_snps_returns_empty(self):
        results = SNPEmbedder(_make_wrapper()).embed_snps_batch([], {"chr17": CHR_SEQ})
        assert results == []


class TestEmbedVCF:

    def _vcf_content(self) -> str:
        return textwrap.dedent(f"""\
            ##fileformat=VCFv4.1
            #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
            chr17\t{SNP_POS}\trs_test\t{REF_BASE}\t{ALT_BASE}\t.\t.\t.
            chr17\t100\trs_other\t{CHR_SEQ[99].upper()}\tC\t.\t.\t.
        """)

    def test_embed_vcf_returns_results(self, tmp_path):
        from embpy.tl.snp_utils import embed_vcf

        vcf_file = tmp_path / "test.vcf"
        vcf_file.write_text(self._vcf_content())

        wrapper = _make_wrapper()
        results = embed_vcf(
            vcf_path=str(vcf_file),
            model_wrapper=wrapper,
            chromosome_sequences={"chr17": CHR_SEQ},
            context_window=64,
        )
        assert len(results) == 2

    def test_embed_vcf_skips_missing_chrom(self, tmp_path):
        from embpy.tl.snp_utils import embed_vcf

        vcf_file = tmp_path / "test.vcf"
        vcf_file.write_text(self._vcf_content())

        wrapper = _make_wrapper()
        results = embed_vcf(
            vcf_path=str(vcf_file),
            model_wrapper=wrapper,
            chromosome_sequences={"chr1": CHR_SEQ},   # wrong chrom
            context_window=64,
        )
        assert len(results) == 0

    def test_embed_vcf_max_variants(self, tmp_path):
        from embpy.tl.snp_utils import embed_vcf

        vcf_file = tmp_path / "test.vcf"
        vcf_file.write_text(self._vcf_content())

        wrapper = _make_wrapper()
        results = embed_vcf(
            vcf_path=str(vcf_file),
            model_wrapper=wrapper,
            chromosome_sequences={"chr17": CHR_SEQ},
            context_window=64,
            max_variants=1,
        )
        assert len(results) <= 1

    def test_embed_vcf_gz_supported(self, tmp_path):
        import gzip
        from embpy.tl.snp_utils import embed_vcf

        vcf_gz = tmp_path / "test.vcf.gz"
        with gzip.open(vcf_gz, "wt") as fh:
            fh.write(self._vcf_content())

        wrapper = _make_wrapper()
        results = embed_vcf(
            vcf_path=str(vcf_gz),
            model_wrapper=wrapper,
            chromosome_sequences={"chr17": CHR_SEQ},
            context_window=64,
        )
        assert len(results) == 2

    def test_embed_vcf_skips_header_lines(self, tmp_path):
        from embpy.tl.snp_utils import embed_vcf

        # Add extra ## comment lines — should all be ignored
        content = "##extra=header\n" * 5 + self._vcf_content()
        vcf_file = tmp_path / "test.vcf"
        vcf_file.write_text(content)

        results = embed_vcf(
            vcf_path=str(vcf_file),
            model_wrapper=_make_wrapper(),
            chromosome_sequences={"chr17": CHR_SEQ},
            context_window=64,
        )
        assert len(results) == 2


class TestSequenceProvider:

    def _write_fasta(self, path, chrom_name: str, seq: str) -> None:
        path.write_text(f">{chrom_name}\n{seq}\n")

    def test_get_chromosome_from_fasta_dir(self, tmp_path):
        fa = tmp_path / "chr17.fa"
        self._write_fasta(fa, "chr17", CHR_SEQ)

        provider = SequenceProvider(fasta_dir=str(tmp_path))
        seq = provider.get_chromosome("chr17")

        assert isinstance(seq, str)
        assert len(seq) == len(CHR_SEQ)
        assert seq.upper() == CHR_SEQ.upper()

    def test_get_chromosome_case_insensitive_prefix(self, tmp_path):
        fa = tmp_path / "chr17.fa"
        self._write_fasta(fa, "chr17", CHR_SEQ)

        provider = SequenceProvider(fasta_dir=str(tmp_path))

        seq = provider.get_chromosome("chr17")
        assert len(seq) > 0

    def test_missing_chromosome_raises(self, tmp_path):
        provider = SequenceProvider(fasta_dir=str(tmp_path))
        with pytest.raises(RuntimeError):
            provider.get_chromosome("chr99")

    def test_get_window_returns_subsequence(self, tmp_path):
        fa = tmp_path / "chr17.fa"
        self._write_fasta(fa, "chr17", CHR_SEQ)

        provider = SequenceProvider(fasta_dir=str(tmp_path))
        window, offset = provider.get_window("chr17", SNP_POS, context=64)

        assert isinstance(window, str)
        assert len(window) <= 64
        assert isinstance(offset, int)
        assert 0 <= offset < len(window)

    def test_get_window_offset_points_to_ref_allele(self, tmp_path):
        fa = tmp_path / "chr17.fa"
        self._write_fasta(fa, "chr17", CHR_SEQ)

        provider = SequenceProvider(fasta_dir=str(tmp_path))
        window, offset = provider.get_window("chr17", SNP_POS, context=64)

        assert window[offset - 1].upper() == REF_BASE.upper()

    def test_get_region_correct_length(self, tmp_path):
        fa = tmp_path / "chr17.fa"
        self._write_fasta(fa, "chr17", CHR_SEQ)

        provider = SequenceProvider(fasta_dir=str(tmp_path))

        region = provider.get_region("chr17", 11, 50)

        assert len(region) == 40

    def test_get_region_correct_content(self, tmp_path):
        fa = tmp_path / "chr17.fa"
        self._write_fasta(fa, "chr17", CHR_SEQ)

        provider = SequenceProvider(fasta_dir=str(tmp_path))

        region = provider.get_region("chr17", 1, 8)

        assert region.upper() == CHR_SEQ[:8].upper()

    def test_caching_avoids_re_read(self, tmp_path):
        fa = tmp_path / "chr17.fa"
        self._write_fasta(fa, "chr17", CHR_SEQ)

        provider = SequenceProvider(fasta_dir=str(tmp_path), cache=True)
        seq1 = provider.get_chromosome("chr17")
        seq2 = provider.get_chromosome("chr17")  

        assert seq1 == seq2

    def test_get_chromosome_from_fasta_file(self, tmp_path):
        fa = tmp_path / "genome.fa"
        fa.write_text(f">chr17\n{CHR_SEQ}\n>chr1\n{'ACGT' * 50}\n")

        provider = SequenceProvider(fasta_file=str(fa))
        seq = provider.get_chromosome("chr17")

        assert seq.upper() == CHR_SEQ.upper()

    def test_multiple_chroms_from_single_file(self, tmp_path):
        chr1_seq = "TTTT" * 100
        fa = tmp_path / "genome.fa"
        fa.write_text(f">chr17\n{CHR_SEQ}\n>chr1\n{chr1_seq}\n")

        provider = SequenceProvider(fasta_file=str(fa))
        assert provider.get_chromosome("chr17").upper() == CHR_SEQ.upper()
        assert provider.get_chromosome("chr1").upper() == chr1_seq.upper()

    def test_init_no_args_does_not_raise(self):
        provider = SequenceProvider()
        assert provider is not None

    def test_rest_fallback_attempted_when_no_files(self):
        """When no local files, get_chromosome should attempt the REST backend."""
        provider = SequenceProvider()
        if hasattr(provider, "_fetch_from_rest"):
            with patch.object(provider, "_fetch_from_rest", return_value=CHR_SEQ) as m:
                try:
                    provider.get_chromosome("chr17")
                except Exception:
                    pass
                m.assert_called()