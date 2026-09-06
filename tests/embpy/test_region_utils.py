from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from embpy.tl.genomics.region_utils import (
    BORZOI_ISM_SHUFFLE_PRESETS,
    MotifHit,
    RegionContext,
    RegionEffectResult,
    RegionEmbedder,
    apply_haplotype,
    dinucleotide_shuffle,
    perturb_interval,
    region_effect_table,
    uniform_random_sequence,
)

RNG = np.random.default_rng(20260903)


def _random_dna(n: int, rng: np.random.Generator | None = None) -> str:
    r = rng or RNG
    return "".join(r.choice(("A", "C", "G", "T"), size=n))


def _dinuc_counts(seq: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in range(len(seq) - 1):
        out[seq[i : i + 2]] = out.get(seq[i : i + 2], 0) + 1
    return out


# ======================================================================= primitives
class TestDinucleotideShuffle:
    def test_preserves_length_and_dinucleotide_composition(self):
        seq = _random_dna(600)
        for seed in range(8):
            shuf = dinucleotide_shuffle(seq, np.random.default_rng(seed))
            assert len(shuf) == len(seq)
            assert _dinuc_counts(shuf) == _dinuc_counts(seq), f"dinucleotide counts changed (seed={seed})"

    def test_preserves_mononucleotide_composition(self):
        seq = _random_dna(400)
        shuf = dinucleotide_shuffle(seq, np.random.default_rng(1))
        assert sorted(shuf) == sorted(seq)

    def test_preserves_first_and_last_base(self):
        # The Eulerian-path construction fixes both endpoints; downstream bin alignment
        # relies on the interval's flanks being untouched.
        seq = _random_dna(300)
        shuf = dinucleotide_shuffle(seq, np.random.default_rng(3))
        assert shuf[0] == seq[0]
        assert shuf[-1] == seq[-1]

    def test_actually_shuffles(self):
        seq = _random_dna(500)
        variants = {dinucleotide_shuffle(seq, np.random.default_rng(s)) for s in range(6)}
        assert len(variants) > 1, "shuffling produced an identical sequence every time"
        assert seq not in variants or len(variants) > 1

    def test_reproducible_for_a_given_seed(self):
        seq = _random_dna(200)
        a = dinucleotide_shuffle(seq, np.random.default_rng(42))
        b = dinucleotide_shuffle(seq, np.random.default_rng(42))
        assert a == b

    @pytest.mark.parametrize("seq", ["", "A", "AC", "AAAAAAAA", "CGCGCGCGCG"])
    def test_degenerate_inputs_do_not_raise(self, seq):
        out = dinucleotide_shuffle(seq, np.random.default_rng(0))
        assert len(out) == len(seq)
        if len(seq) > 1:
            assert _dinuc_counts(out) == _dinuc_counts(seq)

    def test_cpg_content_is_preserved(self):
        # The reason to prefer a dinucleotide over a mononucleotide shuffle: CpG density
        # is itself predictive of regulatory activity (and of the CNS2-type CpG islands
        # this module is aimed at), so it must survive the perturbation.
        seq = "CG" * 150 + _random_dna(200)
        shuf = dinucleotide_shuffle(seq, np.random.default_rng(7))
        assert shuf.count("CG") == seq.count("CG")


class TestUniformRandomSequence:
    def test_length_and_alphabet(self):
        s = uniform_random_sequence(250, np.random.default_rng(0))
        assert len(s) == 250
        assert set(s) <= {"A", "C", "G", "T"}

    def test_zero_and_negative_length(self):
        assert uniform_random_sequence(0, np.random.default_rng(0)) == ""
        assert uniform_random_sequence(-5, np.random.default_rng(0)) == ""


class TestPerturbInterval:
    def test_preserves_total_length_and_flanks(self):
        seq = _random_dna(1000)
        for mode in ("dinuc_shuffle", "uniform_random"):
            out = perturb_interval(seq, 400, 500, mode, np.random.default_rng(0))
            assert len(out) == len(seq)
            assert out[:400] == seq[:400]
            assert out[500:] == seq[500:]

    def test_dinuc_mode_preserves_window_composition(self):
        seq = _random_dna(800)
        out = perturb_interval(seq, 200, 400, "dinuc_shuffle", np.random.default_rng(0))
        assert sorted(out[200:400]) == sorted(seq[200:400])

    def test_clips_to_sequence_bounds(self):
        seq = _random_dna(100)
        out = perturb_interval(seq, -20, 500, "uniform_random", np.random.default_rng(0))
        assert len(out) == 100

    def test_empty_interval_is_a_noop(self):
        seq = _random_dna(100)
        assert perturb_interval(seq, 50, 50, "uniform_random", np.random.default_rng(0)) == seq
        assert perturb_interval(seq, 60, 50, "uniform_random", np.random.default_rng(0)) == seq

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown interval perturbation mode"):
            perturb_interval(_random_dna(50), 10, 20, "nonsense", np.random.default_rng(0))

    def test_mask_n_warns_because_wrappers_encode_n_as_a(self, caplog):
        # Regression guard for a real trap: both wrappers use ALPHABET_MAP.get(base, 0),
        # so 'N' is encoded as 'A' and a "mask" silently becomes a poly-A tract.
        seq = _random_dna(100)
        with caplog.at_level("WARNING"):
            out = perturb_interval(seq, 20, 30, "mask_n", np.random.default_rng(0))
        assert out[20:30] == "N" * 10
        assert "encoded as 'A'" in caplog.text


class TestApplyHaplotype:
    def test_applies_multiple_substitutions(self):
        seq = "AAAACCCCGGGGTTTT"
        out = apply_haplotype(seq, [(0, "A", "G"), (4, "C", "T"), (12, "T", "A")])
        assert out == "GAAATCCCGGGGATTT"

    def test_reference_mismatch_raises_by_default(self):
        with pytest.raises(ValueError, match="expected reference"):
            apply_haplotype("AAAACCCC", [(0, "G", "T")])

    def test_reference_mismatch_can_be_forced(self, caplog):
        with caplog.at_level("WARNING"):
            out = apply_haplotype("AAAACCCC", [(0, "G", "T")], strict=False)
        assert out.startswith("T")
        assert "expected reference" in caplog.text

    def test_offsets_stay_valid_with_length_changing_alleles(self):
        # Applied right-to-left, so an indel at a low offset must not shift a later one.
        seq = "AAAACCCCGGGG"
        out = apply_haplotype(seq, [(0, "A", "AGGG"), (8, "G", "T")])
        assert out == "AGGGAAACCCCTGGG"

    def test_empty_substitution_list_is_a_noop(self):
        assert apply_haplotype("ACGT", []) == "ACGT"


# ==================================================================== RegionContext
class TestRegionContext:
    def test_centres_window_on_interval_midpoint(self):
        r = RegionContext(chrom="chrX", start=49_260_470, end=49_261_090, context_window=524_288)
        midpoint = (49_260_470 + 49_261_090) // 2
        assert r.window_start == midpoint - 524_288 // 2
        assert r.length == 620
        assert r.offset_in_window == 49_260_470 - r.window_start

    def test_explicit_window_start_is_respected(self):
        r = RegionContext(chrom="chr6", start=1000, end=1100, context_window=2048, window_start=500)
        assert r.window_start == 500
        assert r.offset_in_window == 500

    def test_window_start_is_clamped_at_zero(self):
        r = RegionContext(chrom="chr1", start=100, end=200, context_window=524_288)
        assert r.window_start == 0

    def test_requires_end_after_start(self):
        with pytest.raises(ValueError, match="end > start"):
            RegionContext(chrom="chr1", start=500, end=500)

    def test_str_includes_id_and_span(self):
        r = RegionContext(chrom="chrX", start=10, end=20, region_id="FOXP3_CNS2", context_window=64)
        assert "FOXP3_CNS2" in str(r) and "chrX:10-20" in str(r) and "10 bp" in str(r)


# ======================================================================= stub model
class StubProfileModel:
    """A deterministic stand-in for Borzoi/Scooby.

    Emits a profile whose signal is a fixed function of the sequence content in a
    designated "functional" window, so ablating that window measurably lowers the
    prediction while ablating anywhere else does not. That makes the effect direction
    and the bin-restriction logic testable without a checkpoint.
    """

    SEQUENCE_LENGTH = 4096
    BIN_SIZE = 32
    model_name = "stub"
    n_tracks = 2

    def __init__(self, functional_slice=slice(2000, 2100), motif: str = "GATA"):
        self.model = object()  # satisfies the "is it loaded?" check
        self.functional_slice = functional_slice
        self.motif = motif
        self.n_calls = 0

    @property
    def profile_offset_bp(self) -> int:
        return 0

    @property
    def num_bins(self) -> int:
        return self.SEQUENCE_LENGTH // self.BIN_SIZE

    def get_track_metadata(self):
        return pd.DataFrame({"identifier": ["stub:RNA", "stub:ATAC"]})

    def predict_profile(self, input: str, **kwargs):
        self.n_calls += 1
        seq = input.upper()
        window = seq[self.functional_slice]
        # Signal is driven by motif occurrences in the functional window.
        strength = 10.0 * window.count(self.motif) + 1.0
        prof = np.full((self.n_tracks, self.num_bins), 0.1, dtype=np.float64)
        lo = self.functional_slice.start // self.BIN_SIZE
        hi = max(lo + 1, self.functional_slice.stop // self.BIN_SIZE)
        prof[:, lo:hi] = strength
        return prof


@pytest.fixture
def stub_window():
    """A 4096 bp window with 6 GATA motifs inside the functional slice."""
    rng = np.random.default_rng(11)
    seq = list(_random_dna(4096, rng))
    for i in range(6):
        pos = 2000 + i * 15
        seq[pos : pos + 4] = list("GATA")
    return "".join(seq)


# ================================================================= RegionEmbedder
class TestRegionEmbedderConstruction:
    def test_rejects_a_model_without_predict_profile(self):
        class NoProfile:
            model = object()

        with pytest.raises(TypeError, match="does not implement predict_profile"):
            RegionEmbedder(NoProfile())

    def test_warns_when_model_is_not_loaded(self, caplog):
        class Unloaded(StubProfileModel):
            def __init__(self):
                super().__init__()
                self.model = None

        with caplog.at_level("WARNING"):
            RegionEmbedder(Unloaded())
        assert "does not appear to be loaded" in caplog.text


class TestPredictRegionEffect:
    def test_ablating_the_functional_window_reduces_signal(self, stub_window):
        model = StubProfileModel()
        emb = RegionEmbedder(model, seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100,
                               context_window=4096, window_start=0, region_id="functional")
        res = emb.predict_region_effect(region, stub_window, mode="uniform_random", n_replicates=6)
        assert res.mean_effect.shape == (2,)
        assert (res.mean_effect < 0).all(), "ablating the driver window should lower predicted signal"
        # Borzoi's attribution convention is the opposite sign.
        assert np.allclose(res.attribution_scores, -res.mean_effect)

    def test_ablating_an_inert_window_has_no_effect(self, stub_window):
        model = StubProfileModel()
        emb = RegionEmbedder(model, seed=0)
        region = RegionContext(chrom="chr1", start=200, end=300,
                               context_window=4096, window_start=0, region_id="inert")
        res = emb.predict_region_effect(region, stub_window, mode="uniform_random", n_replicates=4)
        assert np.allclose(res.mean_effect, 0.0), "a window the model ignores must score ~0"

    def test_replicate_count_and_shapes(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        res = emb.predict_region_effect(region, stub_window, n_replicates=5)
        assert res.effect_scores.shape == (5, 2)
        assert res.n_replicates == 5
        assert res.sd_effect.shape == (2,)

    def test_profiles_are_dropped_unless_requested(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        assert emb.predict_region_effect(region, stub_window, n_replicates=3).alt_profiles == []
        kept = emb.predict_region_effect(region, stub_window, n_replicates=3, keep_profiles=True)
        assert len(kept.alt_profiles) == 3

    def test_reproducible_given_a_seed(self, stub_window):
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        a = RegionEmbedder(StubProfileModel(), seed=123).predict_region_effect(
            region, stub_window, n_replicates=4)
        b = RegionEmbedder(StubProfileModel(), seed=123).predict_region_effect(
            region, stub_window, n_replicates=4)
        assert np.allclose(a.effect_scores, b.effect_scores)

    def test_mask_n_is_forced_to_one_replicate(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        res = emb.predict_region_effect(region, stub_window, mode="mask_n", n_replicates=10)
        assert res.n_replicates == 1
        assert res.sd_effect.shape == (2,) and np.allclose(res.sd_effect, 0.0)

    def test_bin_restriction_changes_the_statistic(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        whole = emb.predict_region_effect(region, stub_window, n_replicates=4)
        # Bins far from the functional window carry only background, so restricting to
        # them must wash the effect out.
        far = emb.predict_region_effect(region, stub_window, n_replicates=4, bin_indices=[0, 1, 2, 3])
        assert abs(whole.mean_effect[0]) > abs(far.mean_effect[0])
        assert far.bin_indices is not None and len(far.bin_indices) == 4

    def test_zero_replicates_rejected(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        with pytest.raises(ValueError, match="n_replicates must be >= 1"):
            emb.predict_region_effect(region, stub_window, n_replicates=0)

    def test_interval_outside_window_raises(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=90_000, end=90_100,
                               context_window=4096, window_start=0)
        with pytest.raises(ValueError, match="outside the fetched window"):
            emb.predict_region_effect(region, stub_window, n_replicates=1)

    def test_track_names_are_propagated(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        res = emb.predict_region_effect(region, stub_window, n_replicates=2)
        assert res.track_names == ["stub:RNA", "stub:ATAC"]

    def test_predict_kwargs_are_forwarded(self, stub_window):
        seen = {}

        class Recording(StubProfileModel):
            def predict_profile(self, input, **kwargs):
                seen.update(kwargs)
                return super().predict_profile(input)

        emb = RegionEmbedder(Recording(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        emb.predict_region_effect(region, stub_window, n_replicates=1,
                                  cell_embeddings=np.zeros((3, 14)), aggregate="pseudobulk")
        assert "cell_embeddings" in seen and seen["aggregate"] == "pseudobulk"


class TestPredictHaplotypeEffect:
    def test_single_variant_haplotype_matches_a_direct_substitution(self, stub_window):
        # Destroying one GATA motif must lower the signal, and the haplotype path with a
        # single variant must agree with doing the substitution by hand.
        model = StubProfileModel()
        emb = RegionEmbedder(model, seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100,
                               context_window=4096, window_start=0)
        pos1 = 2001  # 1-based -> offset 2000, the 'G' of the first GATA
        res = emb.predict_haplotype_effect(region, stub_window, [(pos1, "G", "T")])

        manual = stub_window[:2000] + "T" + stub_window[2001:]
        ref = model.predict_profile(stub_window)
        alt = model.predict_profile(manual)
        expected = np.log2((alt.sum(axis=1) + 1.0) / (ref.sum(axis=1) + 1.0))
        assert np.allclose(res.effect_scores[0], expected)
        assert (res.effect_scores[0] < 0).all()

    def test_joint_haplotype_is_stronger_than_any_single_variant(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        variants = [(2000 + i * 15 + 1, "G", "T") for i in range(6)]
        joint = emb.predict_haplotype_effect(region, stub_window, variants)
        singles = [emb.predict_haplotype_effect(region, stub_window, [v]).mean_effect[0] for v in variants]
        assert joint.mean_effect[0] < min(singles), (
            "the joint haplotype should exceed every individual variant -- this is the "
            "non-additivity that per-SNP scoring cannot see"
        )

    def test_deterministic_so_sd_is_zero(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        res = emb.predict_haplotype_effect(region, stub_window, [(2001, "G", "T")])
        assert res.n_replicates == 1
        assert np.allclose(res.sd_effect, 0.0)
        assert res.mode.startswith("haplotype[n=1]")

    def test_empty_variant_list_raises(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        with pytest.raises(ValueError, match="at least one variant"):
            emb.predict_haplotype_effect(region, stub_window, [])

    def test_variant_outside_window_raises(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        with pytest.raises(ValueError, match="outside the"):
            emb.predict_haplotype_effect(region, stub_window, [(999_999, "G", "T")])

    def test_reference_mismatch_raises(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        # `predict_haplotype_effect` aggregates mismatches across all variants
        # and raises the summary message; "expected reference" is what the
        # per-variant `apply_haplotype` path emits (asserted separately above).
        with pytest.raises(ValueError, match="do not match the reference base"):
            emb.predict_haplotype_effect(region, stub_window, [(2001, "C", "T")])


class TestIsmShuffleScan:
    def test_localises_the_functional_window(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=1900, end=2200,
                               context_window=4096, window_start=0, region_id="scan")
        out = emb.ism_shuffle_scan(region, stub_window, mode="uniform_random",
                                   window_size=10, n_shuffles=3, stride=10)
        pos, attr = out["positions"], out["attribution"]
        assert len(pos) == 30
        assert attr.shape == (30, 2)
        # The peak attribution must fall inside the motif-bearing stretch (2000-2090).
        peak = pos[np.argmax(attr[:, 0])]
        assert 1990 <= peak <= 2100, f"attribution peak at {peak}, expected inside 1990-2100"

    def test_stride_defaults_to_window_size(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        out = emb.ism_shuffle_scan(region, stub_window, window_size=20, n_shuffles=2)
        assert len(out["positions"]) == 5  # 100 bp / stride 20

    def test_sign_conventions_are_negatives_of_each_other(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2060, context_window=4096, window_start=0)
        out = emb.ism_shuffle_scan(region, stub_window, window_size=10, n_shuffles=2, stride=20)
        assert np.allclose(out["attribution"], -out["log2fc"])

    def test_bad_stride_raises(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        with pytest.raises(ValueError, match="stride must be >= 1"):
            emb.ism_shuffle_scan(region, stub_window, stride=0)

    def test_forward_pass_count_is_as_documented(self, stub_window):
        model = StubProfileModel()
        emb = RegionEmbedder(model, seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        emb.ism_shuffle_scan(region, stub_window, window_size=10, n_shuffles=3, stride=10)
        assert model.n_calls == 10 * 3 + 1  # positions * shuffles + one reference


class TestPredictMotifEffect:
    def _hits(self):
        return [MotifHit(motif_id="GATA1.H12CORE.1.PSM.A", tf_name="GATA1",
                         start=2000 + i * 15, end=2004 + i * 15) for i in range(6)]

    def test_ablating_all_sites_of_a_tf_reduces_signal(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        out = emb.predict_motif_effect(self._hits(), region, stub_window, n_replicates=4)
        assert set(out) == {"GATA1"}
        assert (out["GATA1"].mean_effect < 0).all()
        assert out["GATA1"].mode.startswith("motif_ablate[uniform_random, n_sites=6]")

    def test_grouping_by_tf_beats_any_single_site(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        grouped = emb.predict_motif_effect(self._hits(), region, stub_window, n_replicates=4)
        per_site = emb.predict_motif_effect(self._hits(), region, stub_window,
                                            n_replicates=4, group_by_tf=False)
        assert len(per_site) == 6
        assert grouped["GATA1"].mean_effect[0] < min(r.mean_effect[0] for r in per_site.values())

    def test_hits_outside_the_window_are_dropped_with_a_warning(self, stub_window, caplog):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        hits = self._hits() + [MotifHit(motif_id="X", tf_name="FAR", start=900_000, end=900_010)]
        with caplog.at_level("WARNING"):
            out = emb.predict_motif_effect(hits, region, stub_window, n_replicates=2)
        assert "FAR" not in out
        assert "outside the" in caplog.text

    def test_empty_hit_list_returns_empty(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        assert emb.predict_motif_effect([], region, stub_window) == {}

    def test_scooby_defaults_are_ten_replicates(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        out = emb.predict_motif_effect(self._hits(), region, stub_window)
        assert out["GATA1"].n_replicates == 10  # scooby's published setting


class TestResultsTable:
    def test_tidy_table_has_one_row_per_track(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100,
                               context_window=4096, window_start=0, region_id="CNS2")
        res = emb.predict_region_effect(region, stub_window, n_replicates=3)
        df = region_effect_table([res])
        assert len(df) == 2
        assert {"region_id", "track", "mean_log2fc", "sd_log2fc", "attribution", "seed"} <= set(df.columns)
        assert set(df.region_id) == {"CNS2"}

    def test_accepts_a_dict_of_results(self, stub_window):
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2100, context_window=4096, window_start=0)
        out = emb.predict_motif_effect(
            [MotifHit(motif_id="m", tf_name="GATA1", start=2000, end=2004)],
            region, stub_window, n_replicates=2,
        )
        df = region_effect_table(out)
        assert len(df) == 2 and set(df.region_id) == {"GATA1"}


class TestPublishedPresets:
    def test_borzoi_presets_match_the_paper(self):
        # Borzoi Methods, "Window-shuffled ISM": dinucleotide shuffle with M=7, N=24 for
        # enhancer saliency; uniform random with M=5, N=24 for promoters, splice sites
        # and polyadenylation sites.
        assert BORZOI_ISM_SHUFFLE_PRESETS["enhancer"] == ("dinuc_shuffle", 7, 24)
        assert BORZOI_ISM_SHUFFLE_PRESETS["promoter"] == ("uniform_random", 5, 24)
        assert BORZOI_ISM_SHUFFLE_PRESETS["splice_site"] == ("uniform_random", 5, 24)
        assert BORZOI_ISM_SHUFFLE_PRESETS["polyadenylation_site"] == ("uniform_random", 5, 24)

    def test_presets_are_usable_as_scan_arguments(self, stub_window):
        mode, m, n = BORZOI_ISM_SHUFFLE_PRESETS["enhancer"]
        emb = RegionEmbedder(StubProfileModel(), seed=0)
        region = RegionContext(chrom="chr1", start=2000, end=2021, context_window=4096, window_start=0)
        out = emb.ism_shuffle_scan(region, stub_window, mode=mode, window_size=m, n_shuffles=n, stride=7)
        assert out["attribution"].shape[0] == 3


def _toy_wrapper():
    """A deterministic stand-in for a genomic model: no GPU, no weights, no downloads.

    predict_profile returns a profile that depends on the sequence in a
    position-dependent, NON reverse-complement-invariant way -- like the real models.
    """
    import numpy as np

    class ToyWrapper:
        BIN_SIZE = 32
        profile_offset_bp = 0
        model_name = "toy"

        def predict_profile(self, seq, **kwargs):
            arr = np.frombuffer(seq.encode(), dtype=np.uint8).astype(float)
            n_bins = len(arr) // self.BIN_SIZE
            binned = arr[: n_bins * self.BIN_SIZE].reshape(n_bins, self.BIN_SIZE).sum(1)
            # weight by position so the profile is not palindromic
            ramp = np.linspace(1.0, 2.0, n_bins)
            return (binned * ramp)[None, :]

    return ToyWrapper()


def test_minus_strand_scores_the_same_locus_as_plus_strand():
    """Strand must not change the score of the same genomic interval.

    Bin indices are always forward-oriented (genomic_to_bin_indices has no strand
    argument), so reverse-complementing the input would make them address the wrong
    bins. Strand belongs in the output tracks, not the input orientation.
    """
    import numpy as np
    from embpy.tl.genomics import RegionContext, RegionEmbedder

    seq = "AAAC" * 4096
    emb = RegionEmbedder(_toy_wrapper(), seed=0)
    common = dict(chrom="chr1", start=1000, end=1000 + len(seq), context_window=len(seq),
                  window_start=0)
    plus = emb._predict(seq, RegionContext(strand="+", **common))
    minus = emb._predict(seq, RegionContext(strand="-", **common))
    np.testing.assert_allclose(
        plus, minus, rtol=0, atol=0,
        err_msg="strand changed the profile for the same locus; bin_indices would now "
                "address the wrong bins")


def test_single_substitution_haplotype_equals_single_variant_score():
    """A one-variant haplotype IS a single-variant score; the two APIs must agree.

    These were once reported as differing by 5x. They do not: that report came from a
    minus-strand SNPContext hitting the bug above. This pins them together.
    """
    import numpy as np
    from embpy.tl.genomics import (RegionContext, RegionEmbedder, SNPContext, SNPEmbedder)

    seq = "AAAC" * 4096
    pos_1based, ref, alt = 2049, seq[2048], "T" if seq[2048] != "T" else "G"
    w = _toy_wrapper()
    bins = list(range(10, 40))

    region = RegionContext(chrom="chr1", start=0, end=len(seq), context_window=len(seq),
                           window_start=0)
    hap = RegionEmbedder(w, seed=0).predict_haplotype_effect(
        region, seq, variants=[(pos_1based, ref, alt)], bin_indices=bins, strict=False)

    snp = SNPContext(chrom="chr1", position=pos_1based, ref_allele=ref, alt_alleles=[alt],
                     context_window=len(seq), variant_id="toy")
    var = SNPEmbedder(w).predict_variant_effect(snp, seq, bin_indices=bins)

    np.testing.assert_allclose(
        np.ravel(hap.effect_scores), np.ravel(var.effect_scores[0]), rtol=1e-9, atol=1e-9,
        err_msg="predict_haplotype_effect and predict_variant_effect disagree on a "
                "single-substitution haplotype")

def test_reference_mismatch_error_diagnoses_off_by_one():
    """An off-by-one lands on a neighbouring base; the error should say so."""
    import pytest
    from embpy.tl.genomics.snp_utils import _apply_snp

    seq = "AAAGTTT"          # the G sits at 0-based offset 3
    _apply_snp(seq, 3, "G", "C")               # correct offset: no error

    with pytest.raises(ValueError) as e:
        _apply_snp(seq, 4, "G", "C")           # one too far right
    msg = str(e.value)
    assert "off-by-one" in msg, msg
    assert "1-BASED" in msg and "0-BASED" in msg, msg


def test_haplotype_reports_reference_mismatch_rate():
    """strict=False must report HOW MANY substitutions failed, not warn once per variant."""
    import logging
    import pytest
    from embpy.tl.genomics import RegionContext, RegionEmbedder

    seq = "AAAC" * 4096
    emb = RegionEmbedder(_toy_wrapper(), seed=0)
    region = RegionContext(chrom="chr1", start=0, end=len(seq), context_window=len(seq),
                           window_start=0)
    # three variants, two of which declare the wrong reference base
    variants = [(1, "A", "G"), (2, "C", "G"), (3, "C", "G")]

    with pytest.raises(ValueError) as e:
        emb.predict_haplotype_effect(region, seq, variants=variants, bin_indices=[1, 2],
                                     strict=True)
    assert "do not match the reference base" in str(e.value)

    caplog = logging.getLogger("embpy.tl.genomics.region_utils")
    records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())  # type: ignore[assignment]
    caplog.addHandler(handler)
    try:
        emb.predict_haplotype_effect(region, seq, variants=variants, bin_indices=[1, 2],
                                     strict=False)
    finally:
        caplog.removeHandler(handler)
    assert any("2 of 3 substitutions" in m for m in records), records
