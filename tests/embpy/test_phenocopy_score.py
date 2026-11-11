"""phenocopy_score must run on ordinary inputs and reward predictions that keep the neighbourhoods."""

from __future__ import annotations

import numpy as np

from embpy.tl.metrics import phenocopy_score


def _profiles(seed: int = 0) -> np.ndarray:
    # Four groups of three perturbations with shared responses, so each has close neighbours.
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(4, 30))
    return np.repeat(centers, 3, axis=0) + rng.normal(scale=0.2, size=(12, 30))


def test_auroc_scores_are_finite_and_rank_good_predictions_higher():
    true = _profiles()
    good = true + np.random.default_rng(1).normal(scale=0.05, size=true.shape)
    shuffled = true[np.random.default_rng(2).permutation(len(true))]

    good_scores = phenocopy_score(true, good, n_pca_components=None, mad_thresholds=(1,), recall_ks=(2,))
    bad_scores = phenocopy_score(true, shuffled, n_pca_components=None, mad_thresholds=(1,), recall_ks=(2,))

    assert np.isfinite(good_scores["phenocopy_auroc_mad_1"])
    assert good_scores["phenocopy_auroc_mad_1"] > bad_scores["phenocopy_auroc_mad_1"]
    assert good_scores["phenocopy_recall_2"] > bad_scores["phenocopy_recall_2"]
