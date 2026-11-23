"""Redundant, unique and synergistic gains of fusing two views, measured by prediction.

Two views ``x1``, ``x2`` of the same entities can carry information about a target
``y`` in three ways: both views carry it (redundant), one view carries it (unique),
or only the pair carries it (synergy). This module measures each part as held-out
explained variance, from four predictors of **one nested model family** fitted on
the same entities and folds:

* ``x1``, ``x2`` -- one view each;
* ``late`` -- an additive model ``g1(x1) + g2(x2)`` trained jointly: it can combine
  what each view knows but cannot let the views interact;
* ``early`` -- ``late`` plus interactions between the views.

With held-out explained variances ``R1``, ``R2``, ``R_late``, ``R_early``::

    redundant   B  = R1 + R2 - R_late
    unique      U1 = R_late - R2         U2 = R_late - R1
    synergy     S  = R_early - R_late

These are the variance analogues of the partial information decomposition (Williams &
Beer 2010). They are exact when the target is a sum of a shared part, a private part
per view and an interaction part with independent latents, and estimates otherwise.
Late fusion in this sense exploits private information; early fusion (or
cross-attention) is what can exploit synergy.

A fifth predictor, ``stacked``, combines the two single-view predictions with two
weights fitted on inner out-of-fold predictions -- the usual "late fusion" of trained
models. It counts the shared part twice, so ``R_late - R_stacked`` shows what joint
training recovers.

Model family
------------
Kernel ridge regression with one kernel per view, computed on a train-fitted PCA of
the view (scaled to unit mean squared norm):

* ``kernel="linear"`` -- ``K_m = P_m P_m^T``. The interaction kernel is the linear
  kernel of all products of the leading ``interaction_components`` PCs of the two
  views, ``(Q_1 Q_1^T) * (Q_2 Q_2^T)`` elementwise.
* ``kernel="rbf"`` -- Gaussian kernel with the median-distance bandwidth of the
  training entities. The interaction kernel is ``K_1 * K_2``, a Gaussian kernel on
  the concatenation.

``late`` uses ``v K_1 + (1 - v) K_2`` and ``early`` adds ``w K_int``. Because the
weight grids contain ``v in {0, 1}`` and ``w = 0``, each predictor's model class
contains the previous one: late can fall back to a single view and early to late.
The ridge penalty and the weights are chosen by inner cross-validation on squared
error, grouped when ``groups`` is given. Unsupervised preprocessing (PCA, scaling,
bandwidth) is fitted once per outer training set; targets never leave their fold.

Scores
------
Explained variance is pooled over all held-out entities and every output column:
``R = 1 - sum (y - y_hat)^2 / sum (y - mean_train)^2``, with each fold's own training
mean. Confidence intervals come from a bootstrap over held-out entities (or over
groups), applied to all predictors jointly, so differences are paired.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

__all__ = ["fusion_decomposition", "fusion_decomposition_curve"]

_ALPHAS = tuple(float(a) for a in np.logspace(-3, 3, 13))
_LATE_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
_INTERACTION_WEIGHTS = (0.0, 0.25, 1.0, 4.0)
_PREDICTORS = ("x1", "x2", "late", "stacked", "early")


# ---------------------------------------------------------------------------
# inputs and splits
# ---------------------------------------------------------------------------


def _as_2d(a: Any, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return arr


def _check_inputs(X1: Any, X2: Any, y: Any, groups: Any) -> tuple[np.ndarray, ...]:
    A, B, Y = _as_2d(X1, "X1"), _as_2d(X2, "X2"), _as_2d(y, "y")
    n = A.shape[0]
    if B.shape[0] != n or Y.shape[0] != n:
        raise ValueError(f"X1, X2 and y must have the same rows: got {n}, {B.shape[0]}, {Y.shape[0]}.")
    g = np.arange(n).astype(str) if groups is None else np.asarray(groups).astype(str)
    if g.shape[0] != n:
        raise ValueError(f"groups has {g.shape[0]} entries but X1 has {n} rows.")
    return A, B, Y, g


def _splits(groups: np.ndarray, n_splits: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group-aware folds; singleton groups give shuffled KFold."""
    from sklearn.model_selection import GroupKFold, KFold

    idx = np.arange(groups.shape[0])
    if len(np.unique(groups)) == groups.shape[0]:
        return list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(idx))
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"Need at least {n_splits} groups for {n_splits} folds, got {len(np.unique(groups))}.")
    # GroupKFold is deterministic; shuffle group labels so the seed matters.
    rng = np.random.default_rng(random_state)
    labels = np.unique(groups)
    relabel = dict(zip(labels, rng.permutation(len(labels)), strict=True))
    return list(GroupKFold(n_splits=n_splits).split(idx, groups=np.array([relabel[x] for x in groups])))


def _subsample(train: np.ndarray, groups: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    if fraction >= 1.0:
        return train
    labels = np.unique(groups[train])
    keep = max(2, int(round(fraction * len(labels))))
    chosen = set(rng.choice(labels, size=keep, replace=False))
    return train[np.array([groups[i] in chosen for i in train])]


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def _project(Xtr: np.ndarray, Xte: np.ndarray, width: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Train-fitted PCA scores, scaled to unit mean squared norm on the training rows."""
    mu = Xtr.mean(axis=0)
    A, B = Xtr - mu, Xte - mu
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        raise ValueError("A view has zero variance on the training entities.")
    keep = int(np.sum(s > s[0] * 1e-10))
    if width is not None:
        keep = min(keep, int(width))
    V = Vt[:keep].T
    Ptr, Pte = A @ V, B @ V
    scale = math.sqrt(float((Ptr**2).sum(axis=1).mean()))
    return Ptr / scale, Pte / scale


def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d = (A**2).sum(axis=1)[:, None] + (B**2).sum(axis=1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


def _view_kernels(
    Ptr: np.ndarray, Pte: np.ndarray, kernel: str, interaction_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train-train and test-train kernels of one view, and its interaction factor."""
    if kernel == "linear":
        Ktr, Kte = Ptr @ Ptr.T, Pte @ Ptr.T
        q = min(interaction_components, Ptr.shape[1])
        Qtr, Qte = Ptr[:, :q], Pte[:, :q]
        qs = math.sqrt(float((Qtr**2).sum(axis=1).mean()))
        Qtr, Qte = Qtr / qs, Qte / qs
        return Ktr, Kte, Qtr @ Qtr.T, Qte @ Qtr.T
    if kernel == "rbf":
        Dtr = _sq_dists(Ptr, Ptr)
        off = Dtr[~np.eye(Dtr.shape[0], dtype=bool)]
        med = float(np.median(off)) if off.size else 1.0
        gamma = 1.0 / med if med > 0 else 1.0
        Ktr, Kte = np.exp(-gamma * Dtr), np.exp(-gamma * _sq_dists(Pte, Ptr))
        return Ktr, Kte, Ktr, Kte
    raise ValueError(f"kernel must be 'linear' or 'rbf', got {kernel!r}.")


# ---------------------------------------------------------------------------
# kernel ridge with inner selection
# ---------------------------------------------------------------------------


def _select_alpha(
    K: np.ndarray,
    Y: np.ndarray,
    inner: Sequence[tuple[np.ndarray, np.ndarray]],
    alphas: Sequence[float],
    want_oof: bool,
) -> tuple[float, float, np.ndarray | None]:
    """Best ridge penalty by inner-fold squared error; optionally its out-of-fold predictions."""
    sse = np.zeros(len(alphas))
    oof = np.zeros_like(Y) if want_oof else None
    per_split = []
    for tr, va in inner:
        mean = Y[tr].mean(axis=0)
        lam, E = np.linalg.eigh(K[np.ix_(tr, tr)])
        lam = np.clip(lam, 0.0, None)
        Z = E.T @ (Y[tr] - mean)
        M = K[np.ix_(va, tr)] @ E
        preds = []
        for a_i, alpha in enumerate(alphas):
            pred = mean + (M / (lam + alpha)) @ Z
            sse[a_i] += float(((Y[va] - pred) ** 2).sum())
            preds.append(pred)
        per_split.append((va, preds))
    best = int(np.argmin(sse))
    if want_oof:
        for va, preds in per_split:
            oof[va] = preds[best]
    return float(alphas[best]), float(sse[best]), oof


def _fit_predict(K: np.ndarray, Kte: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    mean = Y.mean(axis=0)
    coef = np.linalg.solve(K + alpha * np.eye(K.shape[0]), Y - mean)
    return mean + Kte @ coef


def _stacking_weights(Y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[float, float]:
    """Least-squares weights ``a, b`` for ``y - mean ~ a (p1 - mean) + b (p2 - mean)``."""
    mean = Y.mean(axis=0)
    t, u, v = (Y - mean).ravel(), (p1 - mean).ravel(), (p2 - mean).ravel()
    G = np.array([[u @ u, u @ v], [u @ v, v @ v]])
    rhs = np.array([u @ t, v @ t])
    a, b = np.linalg.lstsq(G, rhs, rcond=None)[0]
    return float(a), float(b)


def _ladder_fold(
    A: np.ndarray,
    B: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    kernel: str,
    pre_components: int | None,
    interaction_components: int,
    inner_cv: int,
    alphas: Sequence[float],
    late_weights: Sequence[float],
    interaction_weights: Sequence[float],
    random_state: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit the five predictors on ``train``; return test predictions and chosen parameters."""
    P1tr, P1te = _project(A[train], A[test], pre_components)
    P2tr, P2te = _project(B[train], B[test], pre_components)
    K1, K1te, I1, I1te = _view_kernels(P1tr, P1te, kernel, interaction_components)
    K2, K2te, I2, I2te = _view_kernels(P2tr, P2te, kernel, interaction_components)
    Kint, Kint_te = I1 * I2, I1te * I2te
    scale = float(np.mean(np.diag(Kint)))
    Kint, Kint_te = Kint / scale, Kint_te / scale

    Ytr = Y[train]
    inner = _splits(groups[train], inner_cv, random_state)

    cache: dict[tuple[float, float], tuple[float, float, np.ndarray | None]] = {}

    def select(v: float, w: float, want_oof: bool = False) -> tuple[float, float, np.ndarray | None]:
        key = (v, w)
        if key not in cache or (want_oof and cache[key][2] is None):
            K = v * K1 + (1.0 - v) * K2 + w * Kint
            cache[key] = _select_alpha(K, Ytr, inner, alphas, want_oof)
        return cache[key]

    def kernels(v: float, w: float) -> tuple[np.ndarray, np.ndarray]:
        return v * K1 + (1.0 - v) * K2 + w * Kint, v * K1te + (1.0 - v) * K2te + w * Kint_te

    a1, _, oof1 = select(1.0, 0.0, want_oof=True)
    a2, _, oof2 = select(0.0, 0.0, want_oof=True)
    v_best = min(late_weights, key=lambda v: select(v, 0.0)[1])
    w_best = min(interaction_weights, key=lambda w: select(v_best, w)[1])
    a_late, a_early = select(v_best, 0.0)[0], select(v_best, w_best)[0]

    pred = {
        name: _fit_predict(*kernels(v, w), Ytr, alpha)
        for name, v, w, alpha in (
            ("x1", 1.0, 0.0, a1),
            ("x2", 0.0, 0.0, a2),
            ("late", v_best, 0.0, a_late),
            ("early", v_best, w_best, a_early),
        )
    }
    sa, sb = _stacking_weights(Ytr, oof1, oof2)
    mean = Ytr.mean(axis=0)
    pred["stacked"] = mean + sa * (pred["x1"] - mean) + sb * (pred["x2"] - mean)
    params = {
        "alpha_x1": a1,
        "alpha_x2": a2,
        "alpha_late": a_late,
        "alpha_early": a_early,
        "late_weight_x1": v_best,
        "interaction_weight": w_best,
        "stack_a": sa,
        "stack_b": sb,
        "n_train": int(train.shape[0]),
    }
    return pred, params


def _run(
    A: np.ndarray,
    B: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    *,
    cv: int,
    train_fraction: float,
    subsample_seed: int,
    random_state: int,
    **fold_kwargs: Any,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Per-entity held-out squared errors for every predictor, plus each fold's parameters."""
    rng = np.random.default_rng(subsample_seed)
    rows, params = [], []
    for fold, (train, test) in enumerate(_splits(groups, cv, random_state)):
        sub = _subsample(train, groups, train_fraction, rng)
        pred, p = _ladder_fold(A, B, Y, groups, sub, test, random_state=random_state, **fold_kwargs)
        base = Y[sub].mean(axis=0)
        frame = pd.DataFrame(
            {"entity": test, "group": groups[test], "fold": fold, "sst": ((Y[test] - base) ** 2).sum(axis=1)}
        )
        for name in _PREDICTORS:
            frame[f"sse_{name}"] = ((Y[test] - pred[name]) ** 2).sum(axis=1)
        rows.append(frame)
        params.append({"fold": fold, **p})
    return pd.concat(rows, ignore_index=True), params


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------


def _quantities(sse: dict[str, np.ndarray], sst: np.ndarray) -> dict[str, np.ndarray]:
    """All reported quantities from summed errors; works on scalars and bootstrap vectors."""
    r = {name: 1.0 - sse[name] / sst for name in _PREDICTORS}
    best_single = np.maximum(r["x1"], r["x2"])
    return {
        "r2_x1": r["x1"],
        "r2_x2": r["x2"],
        "r2_late": r["late"],
        "r2_stacked": r["stacked"],
        "r2_early": r["early"],
        "redundant": r["x1"] + r["x2"] - r["late"],
        "unique_x1": r["late"] - r["x2"],
        "unique_x2": r["late"] - r["x1"],
        "synergy": r["early"] - r["late"],
        "late_gain": r["late"] - best_single,
        "early_gain": r["early"] - best_single,
        "stacking_gap": r["late"] - r["stacked"],
    }


def _summarise(errors: pd.DataFrame, n_boot: int, level: float, random_state: int) -> pd.DataFrame:
    by_group = errors.groupby("group")[["sst", *[f"sse_{p}" for p in _PREDICTORS]]].sum()
    point = _quantities({p: by_group[f"sse_{p}"].sum() for p in _PREDICTORS}, by_group["sst"].sum())
    out = pd.DataFrame({"estimate": {k: float(v) for k, v in point.items()}})
    out.index.name = "quantity"
    if n_boot > 0:
        rng = np.random.default_rng(random_state)
        m = by_group.shape[0]
        counts = rng.multinomial(m, np.full(m, 1.0 / m), size=n_boot)  # (n_boot, m)
        boot = _quantities(
            {p: counts @ by_group[f"sse_{p}"].to_numpy() for p in _PREDICTORS}, counts @ by_group["sst"].to_numpy()
        )
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        out["ci_low"] = [float(np.quantile(boot[q], lo)) for q in out.index]
        out["ci_high"] = [float(np.quantile(boot[q], hi)) for q in out.index]
    return out


def _fold_kwargs(
    kernel: str,
    pre_components: int | None,
    interaction_components: int,
    inner_cv: int,
    alphas: Sequence[float],
    late_weights: Sequence[float],
    interaction_weights: Sequence[float],
) -> dict[str, Any]:
    if kernel not in {"linear", "rbf"}:
        raise ValueError(f"kernel must be 'linear' or 'rbf', got {kernel!r}.")
    if 0.0 not in interaction_weights:
        raise ValueError("interaction_weights must contain 0, so early fusion contains late fusion.")
    if not {0.0, 1.0} <= set(late_weights):
        raise ValueError("late_weights must contain 0 and 1, so late fusion contains each single view.")
    return {
        "kernel": kernel,
        "pre_components": pre_components,
        "interaction_components": int(interaction_components),
        "inner_cv": int(inner_cv),
        "alphas": tuple(alphas),
        "late_weights": tuple(late_weights),
        "interaction_weights": tuple(interaction_weights),
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def fusion_decomposition(
    X1: Any,
    X2: Any,
    y: Any,
    *,
    kernel: Literal["linear", "rbf"] = "linear",
    groups: Any = None,
    cv: int = 5,
    inner_cv: int = 3,
    pre_components: int | None = 64,
    interaction_components: int = 16,
    alphas: Sequence[float] = _ALPHAS,
    late_weights: Sequence[float] = _LATE_WEIGHTS,
    interaction_weights: Sequence[float] = _INTERACTION_WEIGHTS,
    n_boot: int = 1000,
    level: float = 0.95,
    random_state: int = 0,
) -> pd.DataFrame:
    """Split the held-out gain of fusing two views into redundant, unique and synergy parts.

    Parameters
    ----------
    X1, X2
        The two views, ``(n_entities, d1)`` and ``(n_entities, d2)``, rows aligned.
    y
        Target, ``(n_entities,)`` or ``(n_entities, n_outputs)`` (for example a
        response profile per entity). Scores pool all outputs.
    kernel
        ``"linear"`` (ridge, with bilinear interactions of the leading PCs) or
        ``"rbf"``. See the module docstring.
    groups
        Optional group label per entity. Outer and inner folds keep each group whole,
        and the bootstrap resamples groups. Use it for related entities (for example
        compound groups sharing a scaffold).
    cv, inner_cv
        Outer folds for scoring, inner folds for selecting penalty and weights.
    pre_components
        PCA width per view, fitted on each training set. ``None`` keeps the full rank.
    interaction_components
        Leading PCs per view whose products form the linear interaction kernel.
    alphas, late_weights, interaction_weights
        Selection grids. ``late_weights`` must contain 0 and 1 and
        ``interaction_weights`` must contain 0, which keeps the predictors nested.
    n_boot, level
        Bootstrap replicates and interval level. ``n_boot=0`` skips the intervals.
    random_state
        Seed for folds and bootstrap.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``quantity``: ``r2_x1``, ``r2_x2``, ``r2_late``, ``r2_stacked``,
        ``r2_early``, ``redundant``, ``unique_x1``, ``unique_x2``, ``synergy``,
        ``late_gain`` and ``early_gain`` (over the better single view) and
        ``stacking_gap`` (``r2_late - r2_stacked``). Columns ``estimate`` and, with a
        bootstrap, ``ci_low`` and ``ci_high``. ``df.attrs`` holds ``folds`` (the chosen
        parameters per fold), ``errors`` (per-entity held-out squared errors) and
        ``n_entities``.

    Examples
    --------
    >>> import numpy as np
    >>> from embpy.tl import fusion_decomposition
    >>> rng = np.random.default_rng(0)
    >>> z, u1, u2 = rng.normal(size=(3, 400, 2))
    >>> X1, X2 = np.hstack([z, u1]), np.hstack([z, u2])
    >>> y = z[:, 0] + u1[:, 0] + 0.3 * rng.normal(size=400)  # shared + private to X1
    >>> df = fusion_decomposition(X1, X2, y, n_boot=0)
    >>> bool(df.loc["redundant", "estimate"] > 0.3 and df.loc["unique_x1", "estimate"] > 0.3)
    True
    """
    A, B, Y, g = _check_inputs(X1, X2, y, groups)
    kw = _fold_kwargs(
        kernel, pre_components, interaction_components, inner_cv, alphas, late_weights, interaction_weights
    )
    errors, folds = _run(
        A, B, Y, g, cv=cv, train_fraction=1.0, subsample_seed=random_state, random_state=random_state, **kw
    )
    out = _summarise(errors, n_boot, level, random_state)
    out.attrs.update({"folds": folds, "errors": errors, "n_entities": int(A.shape[0])})
    return out


def fusion_decomposition_curve(
    X1: Any,
    X2: Any,
    y: Any,
    *,
    fractions: Sequence[float] = (0.1, 0.2, 0.4, 0.7, 1.0),
    seeds: Sequence[int] = (0, 1, 2),
    kernel: Literal["linear", "rbf"] = "linear",
    groups: Any = None,
    cv: int = 5,
    inner_cv: int = 3,
    pre_components: int | None = 64,
    interaction_components: int = 16,
    alphas: Sequence[float] = _ALPHAS,
    late_weights: Sequence[float] = _LATE_WEIGHTS,
    interaction_weights: Sequence[float] = _INTERACTION_WEIGHTS,
    random_state: int = 0,
) -> pd.DataFrame:
    """The decomposition as a learning curve: every part against training-set size.

    Each outer training set is subsampled to a fraction of its groups (entities when
    ``groups`` is None), the ladder is refitted, and the full held-out fold is scored.
    The test folds are the same at every fraction and seed, so the curves are paired.
    A plateau says how much of each part the views hold. Where a part becomes
    positive says how much data it takes to use it: synergy has the most parameters
    and usually starts last.

    Parameters
    ----------
    fractions
        Fractions of each training set's groups, in ``(0, 1]``.
    seeds
        Subsampling seeds; the outer folds stay fixed across seeds.
    **others
        As in :func:`fusion_decomposition`.

    Returns
    -------
    pandas.DataFrame
        Tidy rows with ``fraction``, ``seed``, ``n_train`` (mean training entities over
        folds), ``quantity`` and ``estimate``.
    """
    A, B, Y, g = _check_inputs(X1, X2, y, groups)
    if any(not 0 < f <= 1 for f in fractions):
        raise ValueError(f"fractions must be in (0, 1], got {list(fractions)}.")
    kw = _fold_kwargs(
        kernel, pre_components, interaction_components, inner_cv, alphas, late_weights, interaction_weights
    )
    rows = []
    for seed in seeds:
        for fraction in sorted(fractions):
            errors, folds = _run(
                A, B, Y, g, cv=cv, train_fraction=fraction, subsample_seed=seed, random_state=random_state, **kw
            )
            summary = _summarise(errors, 0, 0.95, random_state)
            n_train = float(np.mean([f["n_train"] for f in folds]))
            for quantity, value in summary["estimate"].items():
                rows.append(
                    {"fraction": fraction, "seed": seed, "n_train": n_train, "quantity": quantity, "estimate": value}
                )
    return pd.DataFrame(rows)
