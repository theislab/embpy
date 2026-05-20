"""Annotation-aware diagnostics for embedding panels.

These plots quantify *whether* an embedding actually separates a known
categorical annotation (gene functional group, GO term, top tissue, etc.).
They complement :mod:`embpy.pl.embedding_space` (qualitative scatter plots)
and :mod:`embpy.pl.heatmaps` (cross-model similarity).

All functions consume an :class:`anndata.AnnData` with the embedding in
``adata.obsm[obsm_key]`` and a categorical label column in ``adata.obs``,
matching the convention used elsewhere in :mod:`embpy.pl`.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.figure import Figure


def within_vs_between_similarity(
    adata: AnnData,
    label_key: str,
    obsm_key: str | None = None,
    metric: str = "cosine",
    ax: Any = None,
    title: str | None = None,
) -> dict[str, float]:
    """Boxplot of pairwise similarity within vs between categories.

    The basic sanity check for *"is the embedding informative for this
    annotation?"*. Within-category similarity should be systematically
    higher than between-category if the embedding captures the labelled
    biology.

    Parameters
    ----------
    adata
        AnnData with embeddings in ``.obsm`` and a categorical label in ``.obs``.
    label_key
        Column in ``adata.obs`` with the categorical label. Rows whose label
        is missing (``None`` / ``NaN``) are dropped from the comparison.
    obsm_key
        Embedding key. Defaults to the first discovered embedding.
    metric
        Currently only ``"cosine"`` is supported.
    ax
        Optional matplotlib ``Axes`` to draw into.
    title
        Plot title.

    Returns
    -------
    Dict with ``mean_within``, ``mean_between``, ``n_within``, ``n_between``,
    and the Mann-Whitney U one-sided p-value (``p_value``).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    from ._helpers import _get_embedding_keys

    if obsm_key is None:
        keys = _get_embedding_keys(adata)
        if not keys:
            raise ValueError("No embedding keys found in adata.obsm.")
        obsm_key = keys[0]
    if label_key not in adata.obs.columns:
        raise KeyError(f"'{label_key}' not in adata.obs.")

    if metric != "cosine":
        raise ValueError("Only metric='cosine' is implemented.")

    X = np.asarray(adata.obsm[obsm_key])
    S = cosine_similarity(X)

    cats = adata.obs[label_key].to_numpy()
    valid_lbl = pd.notna(cats)
    iu = np.triu_indices(len(cats), k=1)
    same = cats[iu[0]] == cats[iu[1]]
    valid = valid_lbl[iu[0]] & valid_lbl[iu[1]]
    s_within = S[iu][same & valid]
    s_between = S[iu][(~same) & valid]

    created = ax is None
    fig, ax = (plt.subplots(figsize=(6, 4.5)) if created else (ax.figure, ax))
    parts = ax.boxplot([s_within, s_between], widths=0.55, patch_artist=True,
                       tick_labels=[f"within (n={len(s_within)})",
                                    f"between (n={len(s_between)})"])
    for patch, c in zip(parts["boxes"], ["#4C72B0", "#DD8452"]):
        patch.set_facecolor(c); patch.set_alpha(0.85)
    ax.set_ylabel(f"{metric} similarity")
    ax.set_title(title or f"Within- vs between-{label_key} similarity",
                 fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    p_value = float("nan")
    if len(s_within) and len(s_between):
        from scipy.stats import mannwhitneyu
        _, p_value = mannwhitneyu(s_within, s_between, alternative="greater")
        ax.text(0.5, 0.95, f"Mann-Whitney p = {p_value:.2e}",
                ha="center", va="top", transform=ax.transAxes, fontsize=9)
    if created:
        fig.tight_layout()

    return {
        "mean_within": float(np.mean(s_within)) if len(s_within) else float("nan"),
        "mean_between": float(np.mean(s_between)) if len(s_between) else float("nan"),
        "n_within": int(len(s_within)),
        "n_between": int(len(s_between)),
        "p_value": float(p_value),
    }


def category_centroid_similarity(
    adata: AnnData,
    label_key: str,
    obsm_key: str | None = None,
    ax: Any = None,
    title: str | None = None,
    annot: bool = True,
    cmap: str = "RdBu_r",
) -> pd.DataFrame:
    """Heatmap of pairwise cosine similarity between per-category mean embeddings.

    Cells with the same diagonal label should appear bright (1.0); off-diagonal
    structure reveals which categories the embedding model conflates.

    Parameters
    ----------
    adata
        AnnData.
    label_key
        Column in ``adata.obs`` with the categorical label.
    obsm_key
        Embedding key. Defaults to the first discovered embedding.
    ax
        Optional matplotlib ``Axes``.
    title
        Plot title.
    annot
        Whether to annotate each cell with its similarity value.
    cmap
        Colormap for the heatmap.

    Returns
    -------
    DataFrame ``(n_categories x n_categories)`` of cosine similarities.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    from ._helpers import _get_embedding_keys

    if obsm_key is None:
        keys = _get_embedding_keys(adata)
        if not keys:
            raise ValueError("No embedding keys found in adata.obsm.")
        obsm_key = keys[0]
    if label_key not in adata.obs.columns:
        raise KeyError(f"'{label_key}' not in adata.obs.")

    X = np.asarray(adata.obsm[obsm_key])
    cats = adata.obs[label_key].to_numpy()
    unique = pd.Series(cats).dropna().unique().tolist()
    if not unique:
        raise ValueError(f"No labelled rows in adata.obs[{label_key!r}].")

    centroids = np.stack([X[cats == c].mean(axis=0) for c in unique])
    S = cosine_similarity(centroids)
    df = pd.DataFrame(S, index=unique, columns=unique)

    created = ax is None
    fig, ax = (
        plt.subplots(figsize=(1.0 + 0.6 * len(unique), 0.8 + 0.5 * len(unique)))
        if created else (ax.figure, ax)
    )
    sns.heatmap(df, ax=ax, cmap=cmap, vmin=-1, vmax=1, center=0,
                annot=annot, fmt=".2f", square=True, linewidths=0.4,
                cbar_kws={"label": "cosine"})
    ax.set_title(title or f"{label_key} centroid similarity", fontweight="bold")
    if created:
        fig.tight_layout()
    return df


def knn_label_purity(
    adata: AnnData,
    label_key: str,
    obsm_key: str | None = None,
    k: int = 5,
    metric: str = "cosine",
    ax: Any = None,
    title: str | None = None,
) -> dict[str, float]:
    """Per-category mean fraction of *k* nearest neighbors sharing the label.

    A scalar diagnostic of *neighborhood coherence*. Reported alongside the
    random baseline :math:`\\sum_c p_c^2` (where :math:`p_c` is the category
    frequency), drawn as a dashed vertical line on the bar chart.

    Parameters
    ----------
    adata
        AnnData.
    label_key
        Column in ``adata.obs`` with the categorical label.
    obsm_key
        Embedding key. Defaults to the first discovered embedding.
    k
        Number of nearest neighbors per observation.
    metric
        Distance metric passed to :class:`sklearn.neighbors.NearestNeighbors`.
    ax
        Optional matplotlib ``Axes``.
    title
        Plot title.

    Returns
    -------
    Dict ``{category: mean_purity}`` plus two summary keys
    ``"__overall__"`` (mean across all observations) and ``"__baseline__"``
    (random expectation).
    """
    from sklearn.neighbors import NearestNeighbors

    from ._helpers import _get_embedding_keys

    if obsm_key is None:
        keys = _get_embedding_keys(adata)
        if not keys:
            raise ValueError("No embedding keys found in adata.obsm.")
        obsm_key = keys[0]
    if label_key not in adata.obs.columns:
        raise KeyError(f"'{label_key}' not in adata.obs.")

    X = np.asarray(adata.obsm[obsm_key])
    cats = adata.obs[label_key].to_numpy()
    valid = pd.notna(cats)
    if int(valid.sum()) < k + 1:
        raise ValueError(
            f"Need >= k+1 ({k + 1}) labelled rows; got {int(valid.sum())}."
        )

    Xv = X[valid]; cv = cats[valid]
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(Xv)
    _, idx = nn.kneighbors(Xv)
    purity = np.array(
        [(cv[neigh[1:]] == cv[i]).mean() for i, neigh in enumerate(idx)]
    )
    by_cat = pd.Series(purity, index=cv).groupby(level=0).mean().sort_values()

    sizes = pd.Series(cv).value_counts(normalize=True)
    baseline = float((sizes ** 2).sum())

    created = ax is None
    fig, ax = (
        plt.subplots(figsize=(7, 0.4 * len(by_cat) + 1.5))
        if created else (ax.figure, ax)
    )
    bars = ax.barh(by_cat.index.astype(str), by_cat.values,
                   color=sns.color_palette("tab20", n_colors=len(by_cat)))
    ax.axvline(baseline, color="black", lw=1, ls="--", alpha=0.6,
               label=f"random baseline = {baseline:.2f}")
    for b, v in zip(bars, by_cat.values):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"mean k-NN label purity (k={k})")
    ax.set_title(title or f"k-NN coherence of '{label_key}' on {obsm_key}",
                 fontweight="bold", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    if created:
        fig.tight_layout()

    out = {str(k_): float(v) for k_, v in by_cat.to_dict().items()}
    out["__baseline__"] = baseline
    out["__overall__"] = float(purity.mean())
    return out
