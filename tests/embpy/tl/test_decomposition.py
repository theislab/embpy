"""Tests for the redundant / unique / synergy decomposition of fusion gains.

The load-bearing test is ``TestRecoversPlantedParts``: targets built from a shared
latent, one private latent per view and a product of the private latents must give
back each part's share of the variance.
"""

from __future__ import annotations

import numpy as np
import pytest

from embpy.tl import fusion_decomposition, fusion_decomposition_curve


def planted(rng, n=800, beta=1.0, a1=0.5, a2=0.25, gamma=0.0, noise=0.25):
    """Views (z, u1) and (z, u2), each rotated; y = shared + private + interaction + noise.

    With unit-variance latents the parts have variances beta, a1, a2, gamma and noise,
    so each expected share is that variance over their sum.
    """
    z, u1, u2 = rng.normal(size=(3, n, 2))
    X1 = np.hstack([z, u1]) @ np.linalg.qr(rng.normal(size=(4, 4)))[0]
    X2 = np.hstack([z, u2]) @ np.linalg.qr(rng.normal(size=(4, 4)))[0]
    y = (
        np.sqrt(beta) * z[:, 0]
        + np.sqrt(a1) * u1[:, 0]
        + np.sqrt(a2) * u2[:, 0]
        + np.sqrt(gamma) * u1[:, 1] * u2[:, 1]
        + np.sqrt(noise) * rng.normal(size=n)
    )
    total = beta + a1 + a2 + gamma + noise
    return (
        X1,
        X2,
        y,
        {"redundant": beta / total, "unique_x1": a1 / total, "unique_x2": a2 / total, "synergy": gamma / total},
    )


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestRecoversPlantedParts:
    def test_additive_target_has_no_synergy(self, rng):
        X1, X2, y, truth = planted(rng)
        df = fusion_decomposition(X1, X2, y, pre_components=None, interaction_components=4, n_boot=200)
        for part in ("redundant", "unique_x1", "unique_x2"):
            assert df.loc[part, "estimate"] == pytest.approx(truth[part], abs=0.06)
        assert abs(df.loc["synergy", "estimate"]) < 0.03
        assert df.loc["synergy", "ci_low"] <= 0.0 <= df.loc["synergy", "ci_high"] + 0.01

    def test_interaction_shows_up_as_synergy(self, rng):
        # n = 2000: the product term is heavy-tailed, so the redundant share varies by
        # about 0.03 between samples of 800 (no bias: means match the truth).
        X1, X2, y, truth = planted(rng, n=2000, gamma=0.5)
        df = fusion_decomposition(X1, X2, y, pre_components=None, interaction_components=4, n_boot=200)
        assert df.loc["synergy", "estimate"] == pytest.approx(truth["synergy"], abs=0.07)
        assert df.loc["synergy", "ci_low"] > 0.0
        assert df.loc["redundant", "estimate"] == pytest.approx(truth["redundant"], abs=0.06)

    def test_parts_sum_to_early_fusion(self, rng):
        X1, X2, y, _ = planted(rng, n=400, gamma=0.3)
        est = fusion_decomposition(X1, X2, y, n_boot=0)["estimate"]
        total = est["redundant"] + est["unique_x1"] + est["unique_x2"] + est["synergy"]
        assert total == pytest.approx(est["r2_early"], abs=1e-12)

    def test_stacking_counts_the_shared_part_twice(self, rng):
        """Averaging two trained single-view models loses to a jointly trained additive model."""
        X1, X2, y, _ = planted(rng, beta=1.0, a1=0.5, a2=0.5)
        df = fusion_decomposition(X1, X2, y, pre_components=None, n_boot=200)
        assert df.loc["stacking_gap", "estimate"] > 0.02
        assert df.loc["stacking_gap", "ci_low"] > 0.0

    def test_redundant_views_gain_nothing_from_fusion(self, rng):
        X1, X2, y, _ = planted(rng, beta=1.0, a1=0.0, a2=0.0)
        est = fusion_decomposition(X1, X2, y, pre_components=None, n_boot=0)["estimate"]
        assert abs(est["late_gain"]) < 0.02
        assert est["redundant"] > 0.7


class TestOptions:
    def test_rbf_kernel_runs_and_fits(self, rng):
        X1, X2, y, _ = planted(rng, n=400)
        df = fusion_decomposition(X1, X2, y, kernel="rbf", n_boot=0)
        assert df.loc["r2_late", "estimate"] > 0.5

    def test_groups_stay_in_one_fold(self, rng):
        X1, X2, y, _ = planted(rng, n=400)
        groups = np.repeat(np.arange(100), 4)
        df = fusion_decomposition(X1, X2, y, groups=groups, n_boot=50)
        folds_per_group = df.attrs["errors"].groupby("group")["fold"].nunique()
        assert (folds_per_group == 1).all()

    def test_multi_output_target(self, rng):
        X1, X2, y, _ = planted(rng, n=300)
        Y = np.c_[y, -y, rng.normal(size=300)]
        df = fusion_decomposition(X1, X2, Y, n_boot=0)
        assert 0.0 < df.loc["r2_late", "estimate"] < 1.0

    def test_intervals_bracket_the_estimate(self, rng):
        X1, X2, y, _ = planted(rng, n=300)
        df = fusion_decomposition(X1, X2, y, n_boot=200)
        assert (df["ci_low"] <= df["estimate"] + 1e-12).all()
        assert (df["estimate"] <= df["ci_high"] + 1e-12).all()

    def test_chosen_parameters_are_reported(self, rng):
        X1, X2, y, _ = planted(rng, n=300)
        folds = fusion_decomposition(X1, X2, y, n_boot=0).attrs["folds"]
        assert len(folds) == 5
        assert {"alpha_late", "late_weight_x1", "interaction_weight", "stack_a", "n_train"} <= set(folds[0])

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"kernel": "poly"}, "kernel must be"),
            ({"interaction_weights": (0.5, 1.0)}, "must contain 0"),
            ({"late_weights": (0.25, 0.5)}, "must contain 0 and 1"),
        ],
    )
    def test_invalid_options_raise(self, kwargs, match, rng):
        X1, X2, y, _ = planted(rng, n=60)
        with pytest.raises(ValueError, match=match):
            fusion_decomposition(X1, X2, y, **kwargs)

    def test_row_mismatch_raises(self, rng):
        X1, X2, y, _ = planted(rng, n=60)
        with pytest.raises(ValueError, match="same rows"):
            fusion_decomposition(X1, X2[:50], y)


class TestCurve:
    def test_tidy_rows_for_every_fraction_and_seed(self, rng):
        X1, X2, y, _ = planted(rng, n=300, gamma=0.5)
        df = fusion_decomposition_curve(X1, X2, y, fractions=(0.3, 1.0), seeds=(0, 1))
        assert set(df["fraction"]) == {0.3, 1.0}
        assert set(df["seed"]) == {0, 1}
        assert {"redundant", "unique_x1", "unique_x2", "synergy"} <= set(df["quantity"])
        n_train = df.groupby("fraction")["n_train"].mean()
        assert n_train[0.3] < n_train[1.0]

    def test_full_fraction_matches_the_single_call(self, rng):
        X1, X2, y, _ = planted(rng, n=200)
        curve = fusion_decomposition_curve(X1, X2, y, fractions=(1.0,), seeds=(0,))
        single = fusion_decomposition(X1, X2, y, n_boot=0)["estimate"]
        got = curve.set_index("quantity")["estimate"]
        assert got["synergy"] == pytest.approx(single["synergy"], abs=1e-12)

    def test_synergy_needs_more_data_than_the_additive_parts(self, rng):
        X1, X2, y, _ = planted(rng, n=600, gamma=0.5)
        df = fusion_decomposition_curve(X1, X2, y, fractions=(0.1, 1.0), seeds=(0,))
        s = df[df["quantity"] == "synergy"].set_index("fraction")["estimate"]
        assert s[1.0] > s[0.1]

    def test_invalid_fraction_raises(self, rng):
        X1, X2, y, _ = planted(rng, n=60)
        with pytest.raises(ValueError, match="fractions"):
            fusion_decomposition_curve(X1, X2, y, fractions=(0.0, 1.0))
