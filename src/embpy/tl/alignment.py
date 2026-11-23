"""Representation-alignment metrics: how similarly do two embeddings organise the same entities?

The question this module answers is "do these two representations encode the same
structure?" -- across models, across layers of one model, or between an embedding
and a reference space.

Ordinal metrics (TSI, QSI)
--------------------------
Both are built from a single primitive: comparing which of two distances is larger.

* ``TSI`` -- over triplets ``(i, j, k)``, the fraction of times both representations
  agree on whether ``j`` or ``k`` is closer to ``i``. Anchored, so it probes *local*
  neighbourhood structure.
* ``QSI`` -- over quadruplets ``(i, j, k, l)``, the fraction of times both agree on
  whether ``d(i,j) > d(k,l)``. No shared anchor, so it probes *global* geometry.

Two properties make them well suited to single-cell data:

* **Fixed, interpretable null.** Unrelated representations score ~0.5, because an
  arbitrary ordinal comparison agrees half the time. Identical ones score 1.0. CKA's
  null instead depends on dimensionality and spectrum, so "low CKA" has no absolute
  meaning.
* **Outlier robustness.** CKA is a ratio of Frobenius norms, so a handful of
  large-magnitude rows dominate it. Real single-cell data always has such rows --
  dying cells, doublets, ambient-RNA artefacts. See ``tests/embpy/tl/test_alignment.py``,
  where a pure rotation corrupted with 2% outliers keeps TSI ~0.97 while linear CKA
  collapses to ~0.03.

Both families share the invariances Kornblith et al. (2019) argued for -- translation,
isotropic scaling, orthogonal transformation -- but not invariance to arbitrary
invertible linear maps.

Scales
------
``tsi``, ``qsi``, ``mutual_knn``, ``cknna`` and ``procrustes_similarity`` take a
scale ``k``. At scale ``k`` only comparisons inside each entity's ``k`` nearest
neighbours count:

* ``tsi(k=...)`` -- triplets ``(i, j, l)`` with ``j, l`` among the ``k`` nearest of ``i``.
* ``qsi(k=...)`` -- pairs of pairs drawn from ``{(i, j): j in N_k(i)}``.
* ``cknna(k=...)`` -- CKA over pairs that are mutual ``k``-nearest neighbours
  (Huh et al. 2024, *The Platonic Representation Hypothesis*).
* ``procrustes_similarity(k=...)`` -- Procrustes fit on each neighbourhood, averaged.

Neighbour lists come from one space, so ``neighbors`` chooses ``"x"``, ``"y"`` or
``"both"`` (the mean of the two). ``k = n - 1`` (or ``k=None``) is the global value.
:func:`alignment_profile` evaluates every metric over a grid of ``k`` for one pair.

Fitted linear comparators
-------------------------
* ``procrustes_similarity`` -- the best orthogonal map from one space to the other,
  ``||X^T Y||_* / (||X||_F ||Y||_F)`` after centring. 1.0 for a rotation plus
  isotropic scaling; sensitive to how variance is spread across directions.
* ``cca_sharing`` -- held-out canonical correlations. Fitted on training folds and
  scored on held-out entities, so it counts shared linear directions without the
  in-sample inflation of a fit on all rows. Invariant to invertible linear maps up to
  the PCA truncation and the ridge.

Implementation note
-------------------
The metrics are defined over O(N^3) triplets / O(N^4) quadruplets, but both reduce
exactly to a Kendall-tau concordance count, which runs in O(N^2 log N):

* TSI -- for each anchor ``i``, the agreement fraction over pairs ``(j, k)`` is the
  concordance between the distance vectors ``d_X(i, .)`` and ``d_Y(i, .)``.
* QSI -- the agreement fraction over pairs of pairs is the concordance between the
  two flattened (condensed) distance matrices.

Concordance is recovered from ``scipy.stats.kendalltau`` (tau-b) together with the
tie counts, so tied distances are handled exactly rather than approximately. This
reduction is verified against the naive triple/quadruple loops -- including a
tie-heavy discrete case -- in the test suite.

References
----------
Soares, Gawade, Dittadi & Szczurek. *Scalable and Interpretable Representation
Alignment with Ordinal Similarity* (arXiv:2606.16379). Reference implementation:
https://github.com/diogosoares22/ordinal-similarity-metrics -- consulted for the
metric definitions; the numeric core here is an independent Kendall-tau derivation.

Kornblith, Norouzi, Lee & Hinton. *Similarity of Neural Network Representations
Revisited* (ICML 2019) -- linear CKA.

Schönemann. *A generalized solution of the orthogonal Procrustes problem*
(Psychometrika 1966) -- orthogonal Procrustes.

Hotelling. *Relations between two sets of variates* (Biometrika 1936) -- CCA.

Huh, Cheung, Wang & Isola. *The Platonic Representation Hypothesis* (ICML 2024) --
mutual kNN and CKNNA; reference code https://github.com/minyoungg/platonic-rep.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau

__all__ = [
    "alignment_matrix",
    "alignment_profile",
    "block_to_attention_index",
    "block_to_hidden_state_index",
    "cca_sharing",
    "cknna",
    "linear_cka",
    "mutual_knn",
    "procrustes_similarity",
    "qsi",
    "rank_layers",
    "sample_size_for",
    "tsi",
]

#: Above these sizes ``method="auto"`` switches from the exact path to the
#: guaranteed sampling estimator. TSI holds only O(N) memory per anchor, so it
#: tolerates more rows than QSI, which materialises the full condensed distance
#: matrix (N*(N-1)/2 entries) and sorts it.
_AUTO_EXACT_MAX_N_TSI = 4000
_AUTO_EXACT_MAX_N_QSI = 1500
#: The same QSI budget expressed as a number of compared pairs, for scale ``k``.
_AUTO_EXACT_MAX_PAIRS_QSI = _AUTO_EXACT_MAX_N_QSI * (_AUTO_EXACT_MAX_N_QSI - 1) // 2

#: Default scales for :func:`alignment_profile`; values >= n - 1 are dropped and
#: n - 1 (the global value) is always appended.
_DEFAULT_K_GRID = (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)

_NEIGHBOUR_SOURCES: dict[str, tuple[str, ...]] = {"x": ("x",), "y": ("y",), "both": ("x", "y")}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_matrix(X: Any, name: str) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features), got shape {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf; clean it before measuring alignment.")
    return arr


def _check_pair(X: Any, Y: Any) -> tuple[np.ndarray, np.ndarray]:
    Xa, Ya = _as_matrix(X, "X"), _as_matrix(Y, "Y")
    if Xa.shape[0] != Ya.shape[0]:
        raise ValueError(
            f"X and Y must describe the same entities in the same order: got {Xa.shape[0]} and {Ya.shape[0]} rows."
        )
    if Xa.shape[0] < 3:
        raise ValueError(f"At least 3 entities are required, got {Xa.shape[0]}.")
    return Xa, Ya


def _distance_matrix(X: np.ndarray, metric: str | Callable[..., float]) -> np.ndarray:
    """Full square distance matrix under an arbitrary metric.

    ``metric`` is forwarded to :func:`scipy.spatial.distance.pdist`, so any of its
    string metrics or a caller-supplied callable ``f(u, v) -> float`` works.
    """
    return squareform(pdist(X, metric=metric), checks=False)


def _condensed(X: np.ndarray, metric: str | Callable[..., float]) -> np.ndarray:
    return pdist(X, metric=metric)


def _tie_pairs(v: np.ndarray) -> int:
    """Number of unordered pairs that are tied in ``v``."""
    _, counts = np.unique(v, return_counts=True)
    return int((counts * (counts - 1) // 2).sum())


def _concordance(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of unordered pairs (p, q) with ``sign(a_p-a_q) == sign(b_p-b_q)``.

    Ties count as agreement only when *both* sequences are tied on that pair, which
    matches the ordinal predicate ``sign(...) in {-1, 0, 1}``.
    """
    m = a.shape[0]
    n0 = m * (m - 1) // 2
    if n0 == 0:
        return float("nan")

    tau = kendalltau(a, b).statistic
    if not np.isfinite(tau):
        # kendalltau returns nan when a sequence is constant: then every pair is
        # tied in that sequence, and agreement is exactly the both-tied fraction.
        stacked = np.unique(np.stack([a, b], axis=1), axis=0, return_counts=True)[1]
        return float((stacked * (stacked - 1) // 2).sum() / n0)

    n1, n2 = _tie_pairs(a), _tie_pairs(b)
    _, joint_counts = np.unique(np.stack([a, b], axis=1), axis=0, return_counts=True)
    n_both = int((joint_counts * (joint_counts - 1) // 2).sum())

    c_minus_d = tau * math.sqrt((n0 - n1) * (n0 - n2))
    c_plus_d = n0 - n1 - n2 + n_both
    concordant = (c_minus_d + c_plus_d) / 2.0
    return float((concordant + n_both) / n0)


def sample_size_for(epsilon: float, delta: float) -> int:
    """Samples needed so the estimate is within ``epsilon`` with probability ``1 - delta``.

    ``n = ceil( log(2/delta) / (2 * epsilon**2) )`` (Hoeffding). Note this is
    **independent of N**, which is what makes the sampling path practical on large
    datasets: eps=0.05, delta=0.05 needs 738 samples whether you have 10^3 or 10^6 cells.
    """
    if not 0 < epsilon < 1:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}.")
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}.")
    return int(math.ceil(math.log(2.0 / delta) / (2.0 * epsilon**2)))


def _resolve_n_samples(epsilon: float | None, delta: float | None, n_samples: int | None) -> int:
    if n_samples is not None:
        if epsilon is not None or delta is not None:
            raise ValueError("Pass either n_samples or (epsilon, delta), not both.")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
        return int(n_samples)
    if epsilon is None and delta is None:
        epsilon, delta = 0.05, 0.05  # 738 samples
    if epsilon is None or delta is None:
        raise ValueError("epsilon and delta must be given together.")
    return sample_size_for(epsilon, delta)


def _resolve_method(method: str, n: int, exact_max: int) -> str:
    if method not in {"auto", "exact", "approx"}:
        raise ValueError(f"method must be 'auto', 'exact' or 'approx', got {method!r}.")
    if method != "auto":
        return method
    return "exact" if n <= exact_max else "approx"


def _resolve_sources(neighbors: str) -> tuple[str, ...]:
    if neighbors not in _NEIGHBOUR_SOURCES:
        raise ValueError(f"neighbors must be 'x', 'y' or 'both', got {neighbors!r}.")
    return _NEIGHBOUR_SOURCES[neighbors]


def _resolve_k(k: int | None, n: int, minimum: int) -> int:
    """Scale ``k`` as an int; ``None`` means every other entity (the global value)."""
    if k is None:
        return n - 1
    k = int(k)
    if not minimum <= k <= n - 1:
        raise ValueError(f"k must be in [{minimum}, n-1] = [{minimum}, {n - 1}], got {k}.")
    return k


def _neighbour_order(D: np.ndarray) -> np.ndarray:
    """Row ``i`` lists every other entity by increasing distance from ``i``.

    Ties are broken by index (stable sort), so neighbour sets are deterministic.
    """
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1, kind="stable")[:, : D.shape[0] - 1]


def _orders_for(sources: Sequence[str], DX: np.ndarray, DY: np.ndarray) -> dict[str, np.ndarray]:
    return {s: _neighbour_order(DX if s == "x" else DY) for s in sources}


def _tsi_at_k(
    DX: np.ndarray,
    DY: np.ndarray,
    order: np.ndarray,
    k: int,
    resolved: str,
    size: int | None,
    random_state: int,
) -> float:
    """TSI over triplets whose two compared entities lie in the anchor's ``k`` nearest."""
    n = DX.shape[0]
    if resolved == "exact":
        nb = order[:, :k]
        rows = np.arange(n)[:, None]
        dx, dy = DX[rows, nb], DY[rows, nb]
        return float(np.nanmean([_concordance(dx[i], dy[i]) for i in range(n)]))
    rng = np.random.default_rng(random_state)
    i = rng.integers(0, n, size=size)
    a = rng.integers(0, k, size=size)
    b = rng.integers(0, k - 1, size=size)
    b = b + (b >= a)  # two distinct neighbour positions
    j1, j2 = order[i, a], order[i, b]
    agree = np.sign(DX[i, j1] - DX[i, j2]) == np.sign(DY[i, j1] - DY[i, j2])
    return float(agree.mean())


def _pairs_within_k(order: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Unordered pairs ``(i, j)`` with ``j`` among the ``k`` nearest of ``i``, each once."""
    n = order.shape[0]
    i = np.repeat(np.arange(n, dtype=np.int64), k)
    j = order[:, :k].ravel().astype(np.int64)
    codes = np.unique(np.minimum(i, j) * n + np.maximum(i, j))
    return codes // n, codes % n


def _qsi_at_k(
    DX: np.ndarray,
    DY: np.ndarray,
    order: np.ndarray,
    k: int,
    method: str,
    size: int | None,
    random_state: int,
) -> float:
    """QSI over pairs of pairs drawn from the ``k``-nearest-neighbour pairs."""
    a, b = _pairs_within_k(order, k)
    dx, dy = DX[a, b], DY[a, b]
    m = dx.shape[0]
    if m < 2:
        return float("nan")
    if _resolve_method(method, m, _AUTO_EXACT_MAX_PAIRS_QSI) == "exact":
        return _concordance(dx, dy)
    rng = np.random.default_rng(random_state)
    p = rng.integers(0, m, size=size)
    q = rng.integers(0, m - 1, size=size)
    q = q + (q >= p)
    return float((np.sign(dx[p] - dx[q]) == np.sign(dy[p] - dy[q])).mean())


# ---------------------------------------------------------------------------
# ordinal metrics
# ---------------------------------------------------------------------------


def tsi(
    X: Any,
    Y: Any,
    *,
    k: int | None = None,
    neighbors: Literal["x", "y", "both"] = "both",
    metric: str | Callable[..., float] = "euclidean",
    method: Literal["auto", "exact", "approx"] = "auto",
    epsilon: float | None = None,
    delta: float | None = None,
    n_samples: int | None = None,
    random_state: int = 0,
) -> float:
    """Triplet Similarity Index between two representations of the same entities.

    Over triplets ``(i, j, k)``, the fraction where ``X`` and ``Y`` agree on whether
    ``j`` or ``k`` is closer to ``i``. Anchored, so it measures agreement of *local*
    neighbourhood structure.

    With a scale ``k``, only triplets whose two compared entities are among the ``k``
    nearest neighbours of the anchor count. Small ``k`` asks whether close
    neighbourhoods are ordered alike; ``k = n - 1`` (or ``None``) is the global value.

    Parameters
    ----------
    X, Y
        Arrays of shape ``(n_entities, n_features)`` describing the **same entities
        in the same order**. Feature dimensions may differ.
    k
        Scale: number of nearest neighbours per anchor, in ``[2, n - 1]``. ``None``
        (default) compares against every other entity.
    neighbors
        Which space defines the neighbour lists when ``k`` is given: ``"x"``, ``"y"``
        or ``"both"`` (default; the mean of the two, which keeps the index symmetric).
        Ignored when ``k`` is ``None``.
    metric
        Distance passed to :func:`scipy.spatial.distance.pdist` -- any of its string
        metrics (``"euclidean"``, ``"cosine"``, ``"correlation"``, ...) or a callable
        ``f(u, v) -> float``. Cosine and correlation are common for single-cell data.
    method
        ``"exact"`` computes the full O(N^2 log N) value; ``"approx"`` uses uniform
        triplet sampling with a Hoeffding guarantee; ``"auto"`` (default) picks exact
        for N <= 4000 and approx above.
    epsilon, delta
        Sampling accuracy: the estimate is within ``epsilon`` of the exact value with
        probability at least ``1 - delta``. Defaults to ``0.05, 0.05`` when the approx
        path runs without an explicit ``n_samples``.
    n_samples
        Explicit triplet count, mutually exclusive with ``(epsilon, delta)``.
    random_state
        Seed for the sampling path. Ignored when exact.

    Returns
    -------
    float
        Agreement fraction in ``[0, 1]``. **0.5 is the null** (unrelated
        representations); 1.0 means every distance ordering is preserved.

    Examples
    --------
    >>> import numpy as np
    >>> from embpy.tl import tsi
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 16))
    >>> Q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    >>> float(round(tsi(X, X @ Q), 6))  # rotation preserves all distances
    1.0
    """
    Xa, Ya = _check_pair(X, Y)
    n = Xa.shape[0]
    resolved = _resolve_method(method, n, _AUTO_EXACT_MAX_N_TSI)

    if k is not None:
        kk = _resolve_k(k, n, minimum=2)
        sources = _resolve_sources(neighbors)
        size = None if resolved == "exact" else _resolve_n_samples(epsilon, delta, n_samples)
        DX, DY = _distance_matrix(Xa, metric), _distance_matrix(Ya, metric)
        orders = _orders_for(sources, DX, DY)
        return float(np.mean([_tsi_at_k(DX, DY, orders[s], kk, resolved, size, random_state) for s in sources]))

    if resolved == "exact":
        DX, DY = _distance_matrix(Xa, metric), _distance_matrix(Ya, metric)
        keep = ~np.eye(n, dtype=bool)
        scores = [_concordance(DX[i][keep[i]], DY[i][keep[i]]) for i in range(n)]
        return float(np.nanmean(scores))

    size = _resolve_n_samples(epsilon, delta, n_samples)
    rng = np.random.default_rng(random_state)
    DX, DY = _distance_matrix(Xa, metric), _distance_matrix(Ya, metric)
    i = rng.integers(0, n, size=size)
    j = rng.integers(0, n - 1, size=size)
    k = rng.integers(0, n - 2, size=size)
    j = j + (j >= i)  # j != i
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    k = k + (k >= lo) + ((k + (k >= lo)) >= hi)  # k != i, j
    agree = np.sign(DX[i, j] - DX[i, k]) == np.sign(DY[i, j] - DY[i, k])
    return float(agree.mean())


def qsi(
    X: Any,
    Y: Any,
    *,
    k: int | None = None,
    neighbors: Literal["x", "y", "both"] = "both",
    metric: str | Callable[..., float] = "euclidean",
    method: Literal["auto", "exact", "approx"] = "auto",
    epsilon: float | None = None,
    delta: float | None = None,
    n_samples: int | None = None,
    random_state: int = 0,
) -> float:
    """Quadruplet Similarity Index between two representations of the same entities.

    Over quadruplets ``(i, j, k, l)``, the fraction where ``X`` and ``Y`` agree on
    whether ``d(i,j) > d(k,l)``. There is no shared anchor, so unlike :func:`tsi`
    this probes *global* geometry rather than local neighbourhoods.

    Parameters are identical to :func:`tsi`, except that ``"auto"`` switches to
    sampling above N = 1500, because the exact path materialises and sorts the full
    condensed distance matrix (``N*(N-1)/2`` entries).

    The pairs are drawn from all unordered index pairs; the two pairs of a quadruplet
    are required to be distinct from each other but may share an index.

    With a scale ``k`` (in ``[1, n - 1]``), the pairs are restricted to close pairs
    ``{(i, j): j among the k nearest of i}``, each counted once, so the index asks
    whether both spaces agree on which *close* pairs are closer. ``"auto"`` then
    samples once that set exceeds the exact budget above. ``k = n - 1`` is global.

    Returns
    -------
    float
        Agreement fraction in ``[0, 1]``, null 0.5, identical 1.0.
    """
    Xa, Ya = _check_pair(X, Y)
    n = Xa.shape[0]
    resolved = _resolve_method(method, n, _AUTO_EXACT_MAX_N_QSI)

    if k is not None:
        kk = _resolve_k(k, n, minimum=1)
        sources = _resolve_sources(neighbors)
        size = None if method == "exact" else _resolve_n_samples(epsilon, delta, n_samples)
        DX, DY = _distance_matrix(Xa, metric), _distance_matrix(Ya, metric)
        orders = _orders_for(sources, DX, DY)
        return float(np.mean([_qsi_at_k(DX, DY, orders[s], kk, method, size, random_state) for s in sources]))

    if resolved == "exact":
        return _concordance(_condensed(Xa, metric), _condensed(Ya, metric))

    size = _resolve_n_samples(epsilon, delta, n_samples)
    rng = np.random.default_rng(random_state)
    dX, dY = _condensed(Xa, metric), _condensed(Ya, metric)
    m = dX.shape[0]
    p = rng.integers(0, m, size=size)
    q = rng.integers(0, m - 1, size=size)
    q = q + (q >= p)  # q != p
    agree = np.sign(dX[p] - dX[q]) == np.sign(dY[p] - dY[q])
    return float(agree.mean())


# ---------------------------------------------------------------------------
# comparators
# ---------------------------------------------------------------------------


def linear_cka(X: Any, Y: Any) -> float:
    """Linear Centered Kernel Alignment between two representations.

    Provided as a comparator to the ordinal metrics rather than as the recommended
    default. It is invariant to translation, isotropic scaling and orthogonal
    transformation, but has two properties worth knowing:

    * its null value depends on dimensionality and spectrum, so a "low" score has no
      absolute meaning (unlike TSI, whose null is 0.5);
    * it is a ratio of Frobenius norms and therefore **highly sensitive to outliers** --
      a rotation corrupted with 2% extreme rows drops it to ~0.03 while TSI stays ~0.97.

    Returns
    -------
    float
        Similarity in ``[0, 1]``; 1.0 for representations equal up to the invariances
        above.
    """
    Xa, Ya = _check_pair(X, Y)
    Xc = Xa - Xa.mean(axis=0, keepdims=True)
    Yc = Ya - Ya.mean(axis=0, keepdims=True)
    hsic = float(np.linalg.norm(Yc.T @ Xc, ord="fro") ** 2)
    norm_x = float(np.linalg.norm(Xc.T @ Xc, ord="fro"))
    norm_y = float(np.linalg.norm(Yc.T @ Yc, ord="fro"))
    if norm_x == 0.0 or norm_y == 0.0:
        return float("nan")
    return hsic / (norm_x * norm_y)


def mutual_knn(
    X: Any,
    Y: Any,
    *,
    k: int = 10,
    metric: str | Callable[..., float] = "euclidean",
) -> float:
    """Mean overlap of k-nearest-neighbour sets between two representations.

    A purely local metric: for each entity, the fraction of its ``k`` nearest
    neighbours in ``X`` that are also among its ``k`` nearest in ``Y``. Complements
    the ordinal metrics -- ``mutual_knn`` sees only the neighbour *set*, while
    :func:`tsi` also sees the ordering within it.

    Returns
    -------
    float
        Mean overlap in ``[0, 1]``. The chance level is roughly ``k / (n - 1)``, so
        unlike TSI the null moves with ``k`` and ``n``.
    """
    Xa, Ya = _check_pair(X, Y)
    n = Xa.shape[0]
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must be in [1, n-1] = [1, {n - 1}], got {k}.")

    def neighbours(mat: np.ndarray) -> np.ndarray:
        d = _distance_matrix(mat, metric)
        np.fill_diagonal(d, np.inf)
        return np.argpartition(d, kth=k - 1, axis=1)[:, :k]

    nx, ny = neighbours(Xa), neighbours(Ya)
    shared = [len(set(nx[i]).intersection(ny[i])) for i in range(n)]
    return float(np.mean(shared) / k)


def _mutual_knn_from_orders(order_x: np.ndarray, order_y: np.ndarray, k: int) -> float:
    nx, ny = order_x[:, :k], order_y[:, :k]
    return float(np.mean([np.intersect1d(nx[i], ny[i], assume_unique=True).size for i in range(nx.shape[0])]) / k)


def _hsic_unbiased(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC estimator (Song et al. 2012, eq. 5), diagonals ignored."""
    m = K.shape[0]
    Kt, Lt = K.copy(), L.copy()
    np.fill_diagonal(Kt, 0.0)
    np.fill_diagonal(Lt, 0.0)
    value = (
        float((Kt * Lt.T).sum())
        + float(Kt.sum()) * float(Lt.sum()) / ((m - 1) * (m - 2))
        - 2.0 * float(Kt.sum(axis=0) @ Lt.sum(axis=1)) / (m - 2)
    )
    return value / (m * (m - 3))


def _hsic_biased(K: np.ndarray, L: np.ndarray) -> float:
    """Biased HSIC, ``tr(K H L H)`` with the centring matrix ``H`` (the original CKA numerator)."""
    Kc = K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()
    return float((Kc * L.T).sum())


def _similarity_order(G: np.ndarray) -> np.ndarray:
    """Row ``i`` lists every other entity by decreasing inner product with ``i`` (ties by index)."""
    S = -G.copy()
    np.fill_diagonal(S, np.inf)
    return np.argsort(S, axis=1, kind="stable")[:, : G.shape[0] - 1]


def _knn_mask(order: np.ndarray, k: int) -> np.ndarray:
    n = order.shape[0]
    mask = np.zeros((n, n))
    mask[np.arange(n)[:, None], order[:, :k]] = 1.0
    return mask


def _cknna_from_grams(
    K: np.ndarray,
    L: np.ndarray,
    order_k: np.ndarray,
    order_l: np.ndarray,
    k: int,
    unbiased: bool,
) -> float:
    hsic = _hsic_unbiased if unbiased else _hsic_biased
    mask_k, mask_l = _knn_mask(order_k, k), _knn_mask(order_l, k)
    mask = mask_k * mask_l
    kl = hsic(mask * K, mask * L)
    kk = hsic(mask_k * K, mask_k * K)
    ll = hsic(mask_l * L, mask_l * L)
    if not kk * ll > 0:
        return float("nan")
    return float(kl / math.sqrt(kk * ll))


def _gram(X: np.ndarray, normalize: bool) -> np.ndarray:
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("A row has zero norm and cannot be L2-normalized; pass normalize=False.")
        X = X / norms
    return X @ X.T


def cknna(
    X: Any,
    Y: Any,
    *,
    k: int | None = None,
    normalize: bool = True,
    unbiased: bool = True,
) -> float:
    """Centered Kernel Nearest-Neighbor Alignment (Huh et al. 2024).

    CKA restricted to mutual nearest neighbours: with linear kernels ``K = X X^T`` and
    ``L = Y Y^T``, only pairs ``(i, j)`` where ``j`` is among the ``k`` nearest of ``i``
    in **both** spaces enter the HSIC numerator; each denominator uses its own
    space's neighbours. It is the kernel-valued counterpart of :func:`mutual_knn`:
    neighbourhood overlap weighted by how strongly the shared neighbours co-vary.

    The definition follows the reference implementation of *The Platonic
    Representation Hypothesis* (``platonic-rep``, ``metrics.py``): neighbours by inner
    product, unbiased HSIC, ratio ``HSIC(K, L) / sqrt(HSIC(K, K) HSIC(L, L))``. Two
    differences: ``k=None`` means ``n - 1`` (where it reduces to unbiased linear CKA),
    and no ``1e-6`` is added to the denominator.

    Parameters
    ----------
    X, Y
        Arrays of shape ``(n_entities, n_features)`` for the same entities in the same
        order. Feature dimensions may differ.
    k
        Number of nearest neighbours, in ``[2, n - 1]``. ``None`` uses every other
        entity.
    normalize
        L2-normalize each row first (default), so neighbours are by cosine similarity,
        as in the Platonic-representation experiments.
    unbiased
        Use the unbiased HSIC estimator (default) rather than ``tr(K H L H)``.

    Returns
    -------
    float
        Alignment, 1.0 for representations equal up to rotation (and, with
        ``normalize``, per-row scale); about 0 for unrelated ones. ``NaN`` when a
        self-similarity term is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from embpy.tl import cknna
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 8))
    >>> Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    >>> float(round(cknna(X, X @ Q, k=10), 6))
    1.0
    """
    Xa, Ya = _check_pair(X, Y)
    n = Xa.shape[0]
    if n < 4:
        raise ValueError(f"CKNNA needs at least 4 entities for the unbiased HSIC, got {n}.")
    kk = _resolve_k(k, n, minimum=2)
    K, L = _gram(Xa, normalize), _gram(Ya, normalize)
    return _cknna_from_grams(K, L, _similarity_order(K), _similarity_order(L), kk, unbiased)


# ---------------------------------------------------------------------------
# fitted linear comparators
# ---------------------------------------------------------------------------


def _nuclear_cross(A: np.ndarray, B: np.ndarray) -> float:
    """Nuclear norm of ``A.T @ B`` for matrices with the same rows.

    When the feature dimensions are large relative to the rows, the norm is computed
    in the row space: with ``A = U_a S_a V_a^T`` and ``B = U_b S_b V_b^T``,
    ``||A^T B||_* = ||S_a U_a^T U_b S_b||_*`` because ``V_a``, ``V_b`` have
    orthonormal columns. The cost then depends on the rows, not on the features.
    """
    n = A.shape[0]
    if A.shape[1] * B.shape[1] <= n * n:
        return float(np.linalg.svd(A.T @ B, compute_uv=False).sum())
    Ua, sa, _ = np.linalg.svd(A, full_matrices=False)
    Ub, sb, _ = np.linalg.svd(B, full_matrices=False)
    return float(np.linalg.svd((Ua * sa).T @ (Ub * sb), compute_uv=False).sum())


def _procrustes_core(A: np.ndarray, B: np.ndarray) -> float:
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    norm_a, norm_b = float(np.linalg.norm(A)), float(np.linalg.norm(B))
    if norm_a == 0.0 or norm_b == 0.0:
        return float("nan")
    return _nuclear_cross(A, B) / (norm_a * norm_b)


def _procrustes_at_k(
    Xa: np.ndarray,
    Ya: np.ndarray,
    order: np.ndarray,
    k: int,
    n_anchors: int | None,
    random_state: int,
) -> float:
    n = Xa.shape[0]
    if k >= n - 1:
        return _procrustes_core(Xa, Ya)
    if n_anchors is None or n_anchors >= n:
        anchors = np.arange(n)
    else:
        anchors = np.random.default_rng(random_state).choice(n, size=n_anchors, replace=False)
    values = []
    for i in anchors:
        idx = np.concatenate(([i], order[i, :k]))
        values.append(_procrustes_core(Xa[idx], Ya[idx]))
    return float(np.nanmean(values))


def procrustes_similarity(
    X: Any,
    Y: Any,
    *,
    k: int | None = None,
    neighbors: Literal["x", "y", "both"] = "both",
    metric: str | Callable[..., float] = "euclidean",
    n_anchors: int | None = None,
    random_state: int = 0,
) -> float:
    """Orthogonal Procrustes similarity between two representations of the same entities.

    After centring, the best orthogonal (or semi-orthogonal, when the dimensions
    differ) map from ``X`` to ``Y`` explains
    ``||X^T Y||_* / (||X||_F ||Y||_F)`` of the joint scale. The value is in
    ``[0, 1]`` and equals 1.0 exactly when ``Y`` is a rotation of ``X`` up to
    isotropic scaling. Unlike the ordinal metrics it is a fitted, in-sample quantity:
    with few rows and many dimensions even unrelated spaces score above zero, so
    compare against a permutation null (:func:`alignment_profile` with ``null=True``).

    Parameters
    ----------
    X, Y
        Arrays of shape ``(n_entities, n_features)`` for the same entities in the same
        order. Feature dimensions may differ.
    k
        Scale. ``None`` (default) fits one map to all entities. With ``k`` in
        ``[2, n - 1]``, a separate map is fitted to each entity's neighbourhood (the
        entity and its ``k`` nearest neighbours, each neighbourhood centred on its
        own mean) and the similarities are averaged: local linear agreement.
    neighbors
        Which space defines the neighbourhoods: ``"x"``, ``"y"`` or ``"both"``
        (default; the mean of the two). Ignored when ``k`` is ``None``.
    metric
        Distance used to find the neighbours, as in :func:`tsi`.
    n_anchors
        Number of neighbourhoods to fit at scale ``k``. ``None`` (default) fits one
        per entity; a smaller number samples anchors uniformly, which bounds the cost
        at large ``k``.
    random_state
        Seed for the anchor sample.

    Returns
    -------
    float
        Similarity in ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from embpy.tl import procrustes_similarity
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 8))
    >>> Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    >>> float(round(procrustes_similarity(X, 3.0 * X @ Q + 1.0), 6))
    1.0
    """
    Xa, Ya = _check_pair(X, Y)
    n = Xa.shape[0]
    kk = _resolve_k(k, n, minimum=2)
    if kk == n - 1:
        return _procrustes_core(Xa, Ya)
    sources = _resolve_sources(neighbors)
    orders = _orders_for(sources, _distance_matrix(Xa, metric), _distance_matrix(Ya, metric))
    return float(np.mean([_procrustes_at_k(Xa, Ya, orders[s], kk, n_anchors, random_state) for s in sources]))


def _pca_basis(M: np.ndarray, width: int | None, rank_tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Leading right singular vectors and singular values of a centred matrix."""
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        raise ValueError("A view has zero variance on the training entities.")
    rank = int(np.sum(s > s[0] * rank_tol))
    keep = rank if width is None else min(int(width), rank)
    return Vt[:keep].T, s[:keep]


def _fit_cca(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    pre_components: int | None,
    regularization: float,
    rank_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Regularized CCA on training rows: means, projections and canonical correlations.

    Each view is centred and reduced by PCA to ``pre_components``. The covariance
    ridge is ``regularization`` times the mean retained eigenvalue, the convention of
    :class:`embpy.tl.fusion.CCAFusion`.
    """
    mx, my = Xtr.mean(axis=0), Ytr.mean(axis=0)
    Xc, Yc = Xtr - mx, Ytr - my
    Vx, sx = _pca_basis(Xc, pre_components, rank_tol)
    Vy, sy = _pca_basis(Yc, pre_components, rank_tol)
    dof = Xtr.shape[0] - 1
    var_x, var_y = sx**2 / dof, sy**2 / dof
    wx = 1.0 / np.sqrt(var_x + regularization * var_x.mean())
    wy = 1.0 / np.sqrt(var_y + regularization * var_y.mean())
    cross = wx[:, None] * ((Xc @ Vx).T @ (Yc @ Vy) / dof) * wy[None, :]
    left, corr, right_t = np.linalg.svd(cross, full_matrices=False)
    Px = Vx @ (wx[:, None] * left)
    Py = Vy @ (wy[:, None] * right_t.T)
    return mx, my, Px, Py, corr


def _columnwise_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    denom = np.sqrt((A**2).sum(axis=0) * (B**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (A * B).sum(axis=0) / denom, np.nan)


def _heldout_canonical_correlations(
    Xa: np.ndarray,
    Ya: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    n_components: int | None,
    pre_components: int | None,
    regularization: float,
    rank_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-fold held-out and training canonical correlations, shape ``(folds, comps)``."""
    test_r, train_r = [], []
    for train, test in splits:
        mx, my, Px, Py, corr = _fit_cca(Xa[train], Ya[train], pre_components, regularization, rank_tol)
        test_r.append(_columnwise_corr((Xa[test] - mx) @ Px, (Ya[test] - my) @ Py))
        train_r.append(corr)
    width = min(r.shape[0] for r in test_r)
    if n_components is not None:
        width = min(width, int(n_components))
    return np.stack([r[:width] for r in test_r]), np.stack([r[:width] for r in train_r])


def cca_sharing(
    X: Any,
    Y: Any,
    *,
    n_components: int | None = None,
    pre_components: int | None = 50,
    regularization: float = 0.1,
    cv: int = 5,
    groups: Any = None,
    n_permutations: int = 0,
    alpha: float = 0.05,
    rank_tol: float = 1e-7,
    random_state: int = 0,
) -> pd.DataFrame:
    """Held-out canonical correlations: how many linear directions two spaces share.

    CCA is fitted on training folds and every canonical pair is correlated on the
    held-out entities, so the result does not inflate with dimension the way an
    in-sample fit does. Each view is centred and reduced by PCA to
    ``pre_components`` within the training fold.

    Parameters
    ----------
    X, Y
        Arrays of shape ``(n_entities, n_features)`` for the same entities in the same
        order. Feature dimensions may differ.
    n_components
        Number of canonical pairs to report. Defaults to all retained.
    pre_components
        PCA width per view, fitted on training entities. ``None`` keeps the full rank,
        which is only sensible when the dimension is well below the training size.
    regularization
        Covariance ridge as a multiple of the mean retained eigenvalue. Zero gives
        ordinary CCA.
    cv
        Number of folds.
    groups
        Optional group label per entity. When given, folds never split a group
        (:class:`sklearn.model_selection.GroupKFold`); otherwise folds are shuffled
        with ``random_state``.
    n_permutations
        Number of row permutations of ``Y`` used as a null. Each one refits the whole
        pipeline on the same folds. Zero (default) skips the null.
    alpha
        A component is significant when its held-out correlation exceeds the
        ``1 - alpha`` quantile of its null.
    rank_tol
        Relative singular-value threshold for the retained rank of each view.
    random_state
        Seed for the folds and the permutations.

    Returns
    -------
    pandas.DataFrame
        One row per canonical pair (indexed from 1) with ``r`` (mean held-out
        correlation over folds), ``r_sd`` (its spread over folds) and ``r_train``
        (mean training correlation of the regularized fit). With permutations, also
        ``null_q`` and ``significant``. ``df.attrs`` holds the summaries:
        ``sum_r2`` (sum of squared positive held-out correlations: shared
        dimensions' worth of linear signal), ``mean_r2`` (``sum_r2`` over the number
        of pairs), ``n_significant`` (leading pairs above the null, or ``None``),
        ``n_folds`` and ``n_entities``.

    Examples
    --------
    >>> import numpy as np
    >>> from embpy.tl import cca_sharing
    >>> rng = np.random.default_rng(0)
    >>> z = rng.normal(size=(300, 2))
    >>> X = np.hstack([z, rng.normal(size=(300, 4))])
    >>> Y = np.hstack([z, rng.normal(size=(300, 4))])
    >>> df = cca_sharing(X, Y, pre_components=6)
    >>> bool(df["r"].iloc[1] > 0.9 > 0.3 > df["r"].iloc[2])
    True
    """
    from sklearn.model_selection import GroupKFold, KFold

    Xa, Ya = _check_pair(X, Y)
    n = Xa.shape[0]
    if cv < 2:
        raise ValueError(f"cv must be at least 2, got {cv}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if groups is not None:
        group_arr = np.asarray(groups)
        if group_arr.shape[0] != n:
            raise ValueError(f"groups has {group_arr.shape[0]} entries but X has {n} rows.")
        splits = list(GroupKFold(n_splits=cv).split(Xa, groups=group_arr))
    else:
        splits = list(KFold(n_splits=cv, shuffle=True, random_state=random_state).split(Xa))

    test_r, train_r = _heldout_canonical_correlations(
        Xa, Ya, splits, n_components, pre_components, regularization, rank_tol
    )
    width = test_r.shape[1]
    out = pd.DataFrame(
        {
            "r": test_r.mean(axis=0),
            "r_sd": test_r.std(axis=0, ddof=1) if len(splits) > 1 else np.nan,
            "r_train": train_r.mean(axis=0),
        },
        index=pd.RangeIndex(1, width + 1, name="component"),
    )

    n_significant = None
    if n_permutations > 0:
        rng = np.random.default_rng(random_state)
        null = np.full((n_permutations, width), np.nan)
        for p in range(n_permutations):
            perm_r, _ = _heldout_canonical_correlations(
                Xa, Ya[rng.permutation(n)], splits, width, pre_components, regularization, rank_tol
            )
            null[p, : perm_r.shape[1]] = perm_r.mean(axis=0)
        out["null_q"] = np.nanquantile(null, 1 - alpha, axis=0)
        out["significant"] = out["r"] > out["null_q"]
        leading = out["significant"].to_numpy()
        n_significant = int(width if leading.all() else np.argmin(leading))

    positive = np.clip(out["r"].to_numpy(), 0.0, None)
    out.attrs.update(
        {
            "sum_r2": float(np.sum(positive**2)),
            "mean_r2": float(np.mean(positive**2)),
            "n_significant": n_significant,
            "n_folds": len(splits),
            "n_entities": n,
        }
    )
    return out


def _cca_mean_r2(X: Any, Y: Any, **kwargs: Any) -> float:
    """Scalar CCA summary for :func:`alignment_matrix`: mean squared held-out correlation."""
    return float(cca_sharing(X, Y, **kwargs).attrs["mean_r2"])


# ---------------------------------------------------------------------------
# many-way comparison
# ---------------------------------------------------------------------------

_METRICS: dict[str, Callable[..., float]] = {
    "tsi": tsi,
    "qsi": qsi,
    "cka": linear_cka,
    "linear_cka": linear_cka,
    "mutual_knn": mutual_knn,
    "procrustes": procrustes_similarity,
    "cca": _cca_mean_r2,
    "cknna": cknna,
}

#: Metrics defined without a distance; ``alignment_matrix`` must not pass one.
_DISTANCE_FREE = (linear_cka, _cca_mean_r2, cknna)


def alignment_matrix(
    embeddings: Mapping[str, Any],
    *,
    metric: Literal["tsi", "qsi", "cka", "linear_cka", "mutual_knn", "procrustes", "cca", "cknna"] = "tsi",
    distance: str | Callable[..., float] = "euclidean",
    keys: Sequence[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Pairwise alignment between several representations of the same entities.

    The direct way to ask "do these foundation models encode the same information?" --
    embed one common set of entities with each model, then pass the resulting matrices
    here.

    Parameters
    ----------
    embeddings
        Mapping of label -> array, each ``(n_entities, n_features)``. All must
        describe the same entities in the same order; feature dimensions may differ.
    metric
        Which alignment metric to apply: ``"tsi"``, ``"qsi"``, ``"cka"`` /
        ``"linear_cka"``, ``"mutual_knn"``, ``"procrustes"``, ``"cknna"``, or ``"cca"``
        (the mean squared held-out canonical correlation from :func:`cca_sharing`).
    distance
        The *distance* used inside the metric (``"euclidean"``, ``"cosine"``,
        ``"correlation"``, or a callable). Kept separate from ``metric`` because that
        name is taken by the metric selector. Ignored by ``linear_cka`` and ``cca``,
        which are defined on inner products rather than distances, and by ``cknna``,
        whose neighbours come from inner products.
    keys
        Restrict and order the comparison. Defaults to all keys in insertion order.
    **kwargs
        Forwarded to the metric (``k=`` for ``mutual_knn``; ``k=``, ``neighbors=``,
        ``method=``, ``epsilon=``/``delta=``, ``n_samples=``, ``random_state=`` for
        TSI/QSI; ``k=``, ``n_anchors=`` for Procrustes; ``pre_components=``,
        ``cv=``, ``groups=`` for CCA). With ``neighbors="x"`` or ``"y"`` a scaled
        metric is not symmetric; the matrix then reports the upper-triangle value
        (row label as ``X``) in both cells.

    Returns
    -------
    pandas.DataFrame
        Symmetric labelled matrix with 1.0 on the diagonal, ready for
        :func:`seaborn.heatmap` or the ``embpy.pl`` helpers.

    Examples
    --------
    >>> import numpy as np
    >>> from embpy.tl import alignment_matrix
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(50, 8))
    >>> df = alignment_matrix({"a": X, "b": X + 1.0})  # translation-invariant
    >>> bool(np.isclose(df.loc["a", "b"], 1.0))
    True
    """
    if metric not in _METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Choose from {sorted(_METRICS)}.")
    fn = _METRICS[metric]
    # linear_cka and cca are defined on inner products and take no distance argument.
    call_kwargs = dict(kwargs) if fn in _DISTANCE_FREE else {"metric": distance, **kwargs}

    labels = list(keys) if keys is not None else list(embeddings)
    missing = [lbl for lbl in labels if lbl not in embeddings]
    if missing:
        raise KeyError(f"Labels not present in embeddings: {missing}.")
    if len(labels) < 2:
        raise ValueError(f"Need at least 2 representations to compare, got {len(labels)}.")

    mats = {lbl: _as_matrix(embeddings[lbl], lbl) for lbl in labels}
    n_rows = {lbl: m.shape[0] for lbl, m in mats.items()}
    if len(set(n_rows.values())) > 1:
        raise ValueError(f"All representations must cover the same entities; got row counts {n_rows}.")

    out = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels, dtype=float)
    for a_i in range(len(labels)):
        for b_i in range(a_i + 1, len(labels)):
            la, lb = labels[a_i], labels[b_i]
            value = float(fn(mats[la], mats[lb], **call_kwargs))
            out.iloc[a_i, b_i] = value
            out.iloc[b_i, a_i] = value
    return out


def _default_k_grid(n: int) -> list[int]:
    return [k for k in _DEFAULT_K_GRID if k < n - 1] + [n - 1]


def _profile_values(
    Xa: np.ndarray,
    Ya: np.ndarray,
    DX: np.ndarray,
    DY: np.ndarray,
    ks: Sequence[int],
    metrics: Sequence[str],
    sources: Sequence[str],
    method: str,
    size: int,
    n_anchors: int | None,
    random_state: int,
) -> dict[tuple[str, int, str], float]:
    """All scaled metrics for one pair, keyed by ``(metric, k, neighbours)``."""
    n = Xa.shape[0]
    orders = _orders_for(("x", "y"), DX, DY)
    tsi_method = _resolve_method(method, n, _AUTO_EXACT_MAX_N_TSI)
    if "cknna" in metrics:
        K, L = _gram(Xa, True), _gram(Ya, True)
        sim_orders = (_similarity_order(K), _similarity_order(L))
    out: dict[tuple[str, int, str], float] = {}
    for k in ks:
        global_k = k == n - 1
        for name in metrics:
            if name == "mutual_knn":
                if not global_k:  # at k = n - 1 every set is everything: trivially 1
                    out[(name, k, "x,y")] = _mutual_knn_from_orders(orders["x"], orders["y"], k)
                continue
            if name == "cknna":
                out[(name, k, "all" if global_k else "x,y")] = _cknna_from_grams(K, L, *sim_orders, k, True)
                continue
            per_source = {}
            for s in ("x",) if global_k else sources:
                if name == "tsi":
                    per_source[s] = _tsi_at_k(DX, DY, orders[s], k, tsi_method, size, random_state)
                elif name == "qsi":
                    per_source[s] = _qsi_at_k(DX, DY, orders[s], k, method, size, random_state)
                else:  # procrustes
                    per_source[s] = _procrustes_at_k(Xa, Ya, orders[s], k, n_anchors, random_state)
            if global_k:
                out[(name, k, "all")] = per_source["x"]
                continue
            for s, v in per_source.items():
                out[(name, k, s)] = v
            if len(per_source) > 1:
                out[(name, k, "mean")] = float(np.mean(list(per_source.values())))
    return out


def alignment_profile(
    X: Any,
    Y: Any,
    *,
    ks: Sequence[int] | None = None,
    metrics: Sequence[str] = ("tsi", "qsi", "mutual_knn", "cknna", "procrustes"),
    neighbors: Literal["x", "y", "both"] = "both",
    distance: str | Callable[..., float] = "euclidean",
    cka: bool = True,
    cca: bool = True,
    cca_kwargs: Mapping[str, Any] | None = None,
    n_anchors: int | None = 100,
    method: Literal["auto", "exact", "approx"] = "auto",
    epsilon: float = 0.01,
    delta: float = 0.05,
    null: bool = False,
    random_state: int = 0,
) -> pd.DataFrame:
    """Alignment of one pair of representations at every scale, as one tidy table.

    Local metrics (TSI, QSI, mutual kNN, CKNNA, Procrustes) are evaluated over a grid
    of neighbourhood sizes ``k``, from close neighbourhoods to ``k = n - 1`` (global).
    Global comparators (linear CKA, held-out CCA) are added as single rows. Distance
    matrices and neighbour lists are computed once per space and shared by all
    metrics.

    Parameters
    ----------
    X, Y
        Arrays of shape ``(n_entities, n_features)`` for the same entities in the same
        order. Feature dimensions may differ.
    ks
        Scales to evaluate, each in ``[2, n - 1]``. Defaults to 5, 10, 20, 50, 100,
        200, 500, ... below ``n - 1``, plus ``n - 1``.
    metrics
        Any of ``"tsi"``, ``"qsi"``, ``"mutual_knn"``, ``"cknna"``, ``"procrustes"``.
    neighbors
        Which space defines the neighbour lists: ``"x"``, ``"y"`` or ``"both"``
        (default). With ``"both"`` the table holds the ``x``, ``y`` and ``mean``
        rows. Mutual kNN and CKNNA always use both lists (CKNNA by cosine
        similarity on L2-normalized rows, as in :func:`cknna`).
    distance
        Distance for the neighbour lists and the ordinal metrics, as in :func:`tsi`.
    cka, cca
        Add linear CKA and held-out CCA summary rows.
    cca_kwargs
        Forwarded to :func:`cca_sharing` (``pre_components``, ``cv``, ``groups``,
        ``n_permutations``, ...).
    n_anchors
        Neighbourhoods fitted per scale for local Procrustes (see
        :func:`procrustes_similarity`). ``None`` fits one per entity.
    method, epsilon, delta
        Exact or sampled evaluation of TSI and QSI, as in :func:`tsi`. The sampling
        accuracy defaults to ``epsilon=0.01`` (18,445 samples), tighter than the
        single-call default, because profiles are read across scales.
    null
        Also evaluate every metric with the rows of ``Y`` permuted once, in a
        ``null`` column. TSI and QSI have a fixed null of 0.5; mutual kNN,
        Procrustes and CKA do not, so their values only read against this column.
    random_state
        Seed for sampling, anchors, the CCA folds and the null permutation.

    Returns
    -------
    pandas.DataFrame
        Columns ``metric``, ``k`` (``NaN`` for global-only rows), ``k_frac``
        (``k / (n - 1)``), ``neighbors`` (``"x"``, ``"y"``, ``"mean"``, ``"x,y"`` for
        mutual kNN and CKNNA, ``"all"`` at ``k = n - 1`` and for global rows), ``value`` and,
        with ``null=True``, ``null``. CCA contributes ``cca_sum_r2``,
        ``cca_mean_r2``, ``cca_r1`` and, with permutations, ``cca_n_significant``.

    Examples
    --------
    >>> import numpy as np
    >>> from embpy.tl import alignment_profile
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(80, 6))
    >>> df = alignment_profile(X, X @ np.linalg.qr(rng.normal(size=(6, 6)))[0], cca=False)
    >>> sorted(df["metric"].unique())
    ['cknna', 'linear_cka', 'mutual_knn', 'procrustes', 'qsi', 'tsi']
    >>> bool(np.allclose(df["value"], 1.0))
    True
    """
    Xa, Ya = _check_pair(X, Y)
    n = Xa.shape[0]
    unknown = sorted(set(metrics) - {"tsi", "qsi", "mutual_knn", "cknna", "procrustes"})
    if unknown:
        raise ValueError(f"Unknown scaled metrics {unknown}. Choose from tsi, qsi, mutual_knn, cknna, procrustes.")
    sources = _resolve_sources(neighbors)
    grid = _default_k_grid(n) if ks is None else sorted({_resolve_k(k, n, minimum=2) for k in ks})
    size = _resolve_n_samples(epsilon, delta, None)

    DX, DY = _distance_matrix(Xa, distance), _distance_matrix(Ya, distance)
    values = _profile_values(Xa, Ya, DX, DY, grid, metrics, sources, method, size, n_anchors, random_state)
    if cka:
        values[("linear_cka", -1, "all")] = linear_cka(Xa, Ya)
    if cca:
        res = cca_sharing(Xa, Ya, **{"random_state": random_state, **dict(cca_kwargs or {})})
        values[("cca_sum_r2", -1, "all")] = res.attrs["sum_r2"]
        values[("cca_mean_r2", -1, "all")] = res.attrs["mean_r2"]
        values[("cca_r1", -1, "all")] = float(res["r"].iloc[0])
        if res.attrs["n_significant"] is not None:
            values[("cca_n_significant", -1, "all")] = float(res.attrs["n_significant"])

    null_values: dict[tuple[str, int, str], float] = {}
    if null:
        perm = np.random.default_rng(random_state).permutation(n)
        Yp, DYp = Ya[perm], DY[np.ix_(perm, perm)]
        null_values = _profile_values(Xa, Yp, DX, DYp, grid, metrics, sources, method, size, n_anchors, random_state)
        if cka:
            null_values[("linear_cka", -1, "all")] = linear_cka(Xa, Yp)
        if cca:
            res_p = cca_sharing(Xa, Yp, **{"random_state": random_state, **dict(cca_kwargs or {}), "n_permutations": 0})
            null_values[("cca_sum_r2", -1, "all")] = res_p.attrs["sum_r2"]
            null_values[("cca_mean_r2", -1, "all")] = res_p.attrs["mean_r2"]
            null_values[("cca_r1", -1, "all")] = float(res_p["r"].iloc[0])

    rows = []
    for (name, k, nb), value in values.items():
        row = {
            "metric": name,
            "k": float(k) if k >= 0 else np.nan,
            "k_frac": k / (n - 1) if k >= 0 else np.nan,
            "neighbors": nb,
            "value": value,
        }
        if null:
            row["null"] = null_values.get((name, k, nb), np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# layer indexing -- the two extractors do NOT share a convention
# ---------------------------------------------------------------------------
#
# ``BaseModelWrapper.extract_hidden_states`` returns ``n_blocks + 1`` tensors:
# index 0 is the embedding-layer output, and index ``b + 1`` is the output of
# transformer block ``b``. ``BaseModelWrapper.extract_attention`` returns exactly
# ``n_blocks`` tensors, so index ``b`` is block ``b`` -- there is no embedding entry
# (an embedding layer computes no attention).
#
# Anything joining the two must convert explicitly, or an off-by-one shows up as a
# silent misattribution: "layer 5 attention" paired with block 4's hidden state.
# Use the two helpers below and state which convention you report.


def block_to_hidden_state_index(block: int) -> int:
    """Index into :meth:`extract_hidden_states` output for transformer ``block``.

    Blocks are 0-based, so block 0 -- the first transformer block -- is hidden-state
    index 1; index 0 is the embedding layer, which is not a block.
    """
    if block < 0:
        raise ValueError(f"block must be non-negative, got {block}. Resolve negatives first.")
    return block + 1


def block_to_attention_index(block: int) -> int:
    """Index into :meth:`extract_attention` output for transformer ``block``.

    Identity, because attention has one entry per block and no embedding entry. It
    exists so call sites name the convention instead of assuming it.
    """
    if block < 0:
        raise ValueError(f"block must be non-negative, got {block}. Resolve negatives first.")
    return block


# ---------------------------------------------------------------------------
# layer selection
# ---------------------------------------------------------------------------


def _probe_score(
    X: np.ndarray,
    y: np.ndarray,
    probe: str,
    cv: int,
    random_state: int,
) -> float:
    """Cross-validated score of a simple probe predicting ``y`` from ``X``."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y)
    classification = y.dtype.kind in "USObu" or (y.dtype.kind in "iu" and len(np.unique(y)) <= max(2, len(y) // 10))
    if probe == "ridge":
        estimator = LogisticRegression(max_iter=1000) if classification else Ridge(alpha=1.0, random_state=random_state)
    elif probe == "logistic":
        estimator = LogisticRegression(max_iter=1000)
        classification = True
    else:
        raise ValueError(f"probe must be 'ridge' or 'logistic', got {probe!r}.")

    folds = int(min(cv, len(y) if not classification else np.bincount(_encode(y)).min()))
    if folds < 2:
        return float("nan")
    pipe = make_pipeline(StandardScaler(), estimator)
    scoring = "accuracy" if classification else "r2"
    scores = cross_val_score(pipe, X, y, cv=folds, scoring=scoring)
    return float(np.mean(scores))


def _encode(y: np.ndarray) -> np.ndarray:
    _, inv = np.unique(y, return_inverse=True)
    return inv


def rank_layers(
    layer_embeddings: Mapping[int, Any] | Any,
    target: Any = None,
    *,
    inputs: Any = None,
    probe: str = "ridge",
    metric: Literal["tsi", "qsi", "cka", "linear_cka", "mutual_knn"] = "tsi",
    distance: str | Callable[..., float] = "euclidean",
    layers: Sequence[int] | None = None,
    cv: int = 5,
    random_state: int = 0,
    **metric_kwargs: Any,
) -> pd.DataFrame:
    """Score every layer of a model so you can pick one from data instead of by default.

    The last layer of a foundation model is routinely *not* the best for transfer --
    it is specialised toward the pretraining objective -- so "which layer should I
    cache?" deserves an answer measured on your own task.

    Parameters
    ----------
    layer_embeddings
        Either a mapping ``{layer_index: (n_entities, n_features) array}`` -- exactly
        what :meth:`~embpy.models.base.BaseModelWrapper.embed_all_layers` returns --
        or a model wrapper, in which case ``inputs`` must be given and
        ``embed_all_layers`` is called for you.
    target
        Optional per-entity label to probe for: numeric for regression, categorical
        for classification. When omitted, the probe and target columns are skipped
        and only the structural comparisons are reported.
    inputs
        Token tensor for the wrapper form, forwarded to ``embed_all_layers``.
    probe
        ``"ridge"`` (default; switches to logistic regression for categorical
        targets) or ``"logistic"``.
    metric, distance
        Alignment metric and the distance used inside it, as in
        :func:`alignment_matrix`.
    layers
        Restrict to these layer indices. Defaults to all, in sorted order.
    cv
        Cross-validation folds for the probe.

    Returns
    -------
    pandas.DataFrame
        One row per layer, indexed by layer, with columns:

        ``probe_score``
            Cross-validated R^2 (regression) or accuracy (classification). Absent
            when ``target`` is None.
        ``to_final``
            Alignment with the last layer -- low means this layer carries something
            the final representation has discarded.
        ``to_target``
            Alignment between the layer's geometry and the target's. Absent when
            ``target`` is None.
        ``to_previous``
            Alignment with the preceding layer. High means the layer is largely
            redundant, so caching it buys little over its neighbour.

    Notes
    -----
    Layer indices are reported exactly as they appear in ``layer_embeddings``. For
    ``embed_all_layers`` that is the hidden-state convention, where **index 0 is the
    embedding layer and index b+1 is transformer block b** -- which is *not* the
    convention used by :meth:`~embpy.models.base.BaseModelWrapper.extract_attention`.
    Convert with :func:`block_to_hidden_state_index` /
    :func:`block_to_attention_index` before joining the two.

    Examples
    --------
    >>> import numpy as np
    >>> from embpy.tl import rank_layers
    >>> rng = np.random.default_rng(0)
    >>> embs = {i: rng.normal(size=(60, 8)) for i in range(4)}
    >>> df = rank_layers(embs)
    >>> list(df.columns)
    ['to_final', 'to_previous']
    """
    if not isinstance(layer_embeddings, Mapping):
        wrapper = layer_embeddings
        if inputs is None:
            raise ValueError(
                "When passing a model wrapper, also pass inputs= (the token tensor) so embed_all_layers can be called."
            )
        layer_embeddings = wrapper.embed_all_layers(inputs)

    if not layer_embeddings:
        raise ValueError("layer_embeddings is empty; nothing to rank.")

    idx = sorted(layer_embeddings) if layers is None else list(layers)
    missing = [i for i in idx if i not in layer_embeddings]
    if missing:
        raise KeyError(f"Layers not present in layer_embeddings: {missing}.")

    mats = {i: _as_matrix(layer_embeddings[i], f"layer {i}") for i in idx}
    final = mats[idx[-1]]

    fn = _METRICS[metric]
    align_kw = dict(metric_kwargs) if fn is linear_cka else {"metric": distance, **metric_kwargs}

    y = None if target is None else np.asarray(target)
    if y is not None and y.shape[0] != final.shape[0]:
        raise ValueError(f"target has {y.shape[0]} entries but the embeddings have {final.shape[0]} rows.")
    target_space = None
    if y is not None and y.dtype.kind in "fiu":
        target_space = y.reshape(len(y), -1).astype(np.float64)

    rows: list[dict[str, float]] = []
    for pos, i in enumerate(idx):
        row: dict[str, float] = {}
        if y is not None:
            row["probe_score"] = _probe_score(mats[i], y, probe, cv, random_state)
        row["to_final"] = 1.0 if i == idx[-1] else float(fn(mats[i], final, **align_kw))
        if target_space is not None:
            row["to_target"] = float(fn(mats[i], target_space, **align_kw))
        row["to_previous"] = float("nan") if pos == 0 else float(fn(mats[i], mats[idx[pos - 1]], **align_kw))
        rows.append(row)

    return pd.DataFrame(rows, index=pd.Index(idx, name="layer"))
