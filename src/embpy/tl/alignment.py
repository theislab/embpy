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
    "linear_cka",
    "mutual_knn",
    "qsi",
    "sample_size_for",
    "tsi",
]

#: Above these sizes ``method="auto"`` switches from the exact path to the
#: guaranteed sampling estimator. TSI holds only O(N) memory per anchor, so it
#: tolerates more rows than QSI, which materialises the full condensed distance
#: matrix (N*(N-1)/2 entries) and sorts it.
_AUTO_EXACT_MAX_N_TSI = 4000
_AUTO_EXACT_MAX_N_QSI = 1500


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


# ---------------------------------------------------------------------------
# ordinal metrics
# ---------------------------------------------------------------------------


def tsi(
    X: Any,
    Y: Any,
    *,
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

    Parameters
    ----------
    X, Y
        Arrays of shape ``(n_entities, n_features)`` describing the **same entities
        in the same order**. Feature dimensions may differ.
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

    Returns
    -------
    float
        Agreement fraction in ``[0, 1]``, null 0.5, identical 1.0.
    """
    Xa, Ya = _check_pair(X, Y)
    n = Xa.shape[0]
    resolved = _resolve_method(method, n, _AUTO_EXACT_MAX_N_QSI)

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


# ---------------------------------------------------------------------------
# many-way comparison
# ---------------------------------------------------------------------------

_METRICS: dict[str, Callable[..., float]] = {
    "tsi": tsi,
    "qsi": qsi,
    "cka": linear_cka,
    "linear_cka": linear_cka,
    "mutual_knn": mutual_knn,
}


def alignment_matrix(
    embeddings: Mapping[str, Any],
    *,
    metric: Literal["tsi", "qsi", "cka", "linear_cka", "mutual_knn"] = "tsi",
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
        ``"linear_cka"``, or ``"mutual_knn"``.
    distance
        The *distance* used inside the metric (``"euclidean"``, ``"cosine"``,
        ``"correlation"``, or a callable). Kept separate from ``metric`` because that
        name is taken by the metric selector. Ignored by ``linear_cka``, which is
        defined on inner products rather than distances.
    keys
        Restrict and order the comparison. Defaults to all keys in insertion order.
    **kwargs
        Forwarded to the metric (``k=`` for ``mutual_knn``; ``method=``,
        ``epsilon=``/``delta=``, ``n_samples=``, ``random_state=`` for TSI/QSI).

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
    # linear_cka is defined on inner products and takes no distance argument.
    call_kwargs = dict(kwargs) if fn is linear_cka else {"metric": distance, **kwargs}

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
