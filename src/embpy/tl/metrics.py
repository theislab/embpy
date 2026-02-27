"""Evaluation metrics for perturbation prediction.

Provides standalone metric functions, a scanpy DEG wrapper, and biological
evaluation tools for comparing predicted vs. observed expression.

**Regression metrics** (operate on numpy arrays):

- :func:`mse`, :func:`r2`, :func:`mean_correlation`
- :func:`delta_l2` — L2-norm shift from control
- :func:`compute_metrics` — aggregate dictionary of all the above

**Biological metrics** (expression-level):

- :func:`gene_r2`, :func:`perturbation_r2`
- :func:`frac_correct_direction`

**Differential expression** (scanpy wrappers + comparison):

- :func:`rank_genes_groups` — thin wrapper around ``sc.tl.rank_genes_groups``
- :func:`get_deg_dataframe` — extract a filtered DEG table
- :func:`deg_overlap` — Jaccard / precision / recall of top DEG lists
- :func:`deg_direction_agreement` — fraction of shared DEGs with matching sign
- :func:`compare_deg` — high-level function running both comparisons
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_scanpy():
    """Lazily import scanpy, raising a clear error if missing."""
    try:
        import scanpy as sc

        return sc
    except ImportError as e:
        raise ImportError(
            "scanpy is required for this function but is not installed. Install it with:  pip install 'embpy[scanpy]'"
        ) from e


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    result = pearsonr(a.ravel(), b.ravel())
    return float(result[0])  # type: ignore[arg-type]


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    result = spearmanr(a.ravel(), b.ravel())
    return float(result[0])  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════
# Regression metrics
# ═══════════════════════════════════════════════════════════════════════════


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error between true and predicted values."""
    return float(mean_squared_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² score, averaged per perturbation for 2-D expression matrices.

    For 2-D inputs, R² is computed per row (perturbation) and averaged,
    skipping constant rows.  This answers: "for each perturbation, how
    well does the predicted expression profile match the true one?"
    """
    if y_true.ndim == 2:
        scores: list[float] = []
        for i in range(y_true.shape[0]):
            if np.std(y_true[i]) == 0:
                continue
            scores.append(float(r2_score(y_true[i], y_pred[i])))
        return float(np.mean(scores)) if scores else 0.0
    return float(r2_score(y_true, y_pred))


def mean_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = "pearson",
) -> float:
    """Mean correlation between true and predicted values.

    For 2-D inputs (expression matrices), correlation is computed per row
    (perturbation) and averaged — i.e. "how well does the predicted
    expression signature match the true one for each perturbation?"

    Parameters
    ----------
    y_true
        True values (1-D scalar target or 2-D expression matrix).
    y_pred
        Predicted values (same shape).
    method
        ``"pearson"`` or ``"spearman"``.
    """
    corr_fn = _safe_pearson if method == "pearson" else _safe_spearman
    if y_true.ndim == 2:
        rs = [corr_fn(y_true[i], y_pred[i]) for i in range(y_true.shape[0])]
        return float(np.mean(rs)) if rs else 0.0
    return corr_fn(y_true, y_pred)


def delta_l2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mean_control: np.ndarray,
) -> dict[str, float]:
    """Delta-L2 metrics measuring change from control.

    Computes the L2 norm of the difference from the control mean for
    both true and predicted expression, then reports the MAE and Pearson
    correlation between these norms.

    Parameters
    ----------
    y_true
        True expression matrix ``(n_samples, n_genes)``.
    y_pred
        Predicted expression matrix (same shape).
    mean_control
        Mean control expression vector ``(n_genes,)``.

    Returns
    -------
    Dictionary with keys ``"delta_l2_mae"`` and ``"delta_l2_pearson"``.
    """
    if y_true.ndim != 2:
        return {"delta_l2_mae": float("nan"), "delta_l2_pearson": float("nan")}

    delta_true = np.linalg.norm(y_true - mean_control, axis=1)
    delta_pred = np.linalg.norm(y_pred - mean_control, axis=1)

    mae = float(np.mean(np.abs(delta_pred - delta_true)))
    r = _safe_pearson(delta_true, delta_pred)
    return {"delta_l2_mae": mae, "delta_l2_pearson": r}


# ---------------------------------------------------------------------------
# Aggregate regression metrics (used by the benchmark module)
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mean_control: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all regression metrics for a prediction task.

    Parameters
    ----------
    y_true
        Ground-truth target (1-D or 2-D).
    y_pred
        Predicted target (same shape as *y_true*).
    mean_control
        Mean control expression vector (only used for delta-L2 when
        *y_true* is 2-D).

    Returns
    -------
    Dictionary mapping metric names to float values.
    """
    metrics: dict[str, float] = {}
    metrics["mse"] = mse(y_true, y_pred)
    metrics["r2"] = r2(y_true, y_pred)
    metrics["pearson"] = mean_correlation(y_true, y_pred, method="pearson")
    metrics["spearman"] = mean_correlation(y_true, y_pred, method="spearman")

    if y_true.ndim == 2 and mean_control is not None:
        metrics.update(delta_l2(y_true, y_pred, mean_control))
    else:
        metrics["delta_l2_mae"] = float("nan")
        metrics["delta_l2_pearson"] = float("nan")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Biological metrics — expression-level
# ═══════════════════════════════════════════════════════════════════════════


def gene_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Per-gene R², averaged across genes.

    Measures how well predicted expression tracks each gene's variation
    across perturbations.

    Parameters
    ----------
    y_true
        True expression matrix ``(n_perturbations, n_genes)``.
    y_pred
        Predicted expression matrix (same shape).
    """
    if y_true.ndim != 2:
        raise ValueError("gene_r2 requires 2-D expression matrices.")
    scores: list[float] = []
    for j in range(y_true.shape[1]):
        if np.std(y_true[:, j]) == 0:
            continue
        scores.append(float(r2_score(y_true[:, j], y_pred[:, j])))
    return float(np.mean(scores)) if scores else 0.0


def frac_correct_direction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mean_control: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """Fraction of genes with correct direction of change from control.

    For each gene, the sign of ``(expression − control)`` is compared
    between true and predicted.  Returns the fraction that agree.

    Parameters
    ----------
    y_true
        True expression matrix ``(n_perturbations, n_genes)``.
    y_pred
        Predicted expression matrix (same shape).
    mean_control
        Mean control expression vector ``(n_genes,)``.
    threshold
        Minimum absolute change to consider.  Entries where both true and
        predicted absolute change are below this value are excluded.
    """
    if y_true.ndim != 2:
        raise ValueError("frac_correct_direction requires 2-D expression matrices.")

    diff_true = y_true - mean_control
    diff_pred = y_pred - mean_control

    sign_true = np.sign(diff_true)
    sign_pred = np.sign(diff_pred)

    if threshold > 0:
        mask = (np.abs(diff_true) >= threshold) | (np.abs(diff_pred) >= threshold)
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(sign_true[mask] == sign_pred[mask]))

    return float(np.mean(sign_true == sign_pred))


# ═══════════════════════════════════════════════════════════════════════════
# Scanpy wrappers — Differential Expression
# ═══════════════════════════════════════════════════════════════════════════


def rank_genes_groups(
    adata: AnnData,
    groupby: str,
    reference: str = "rest",
    method: str = "wilcoxon",
    n_genes: int | None = None,
    key_added: str = "rank_genes_groups",
    **kwargs: Any,
) -> AnnData:
    """Run differential expression analysis via scanpy.

    Thin wrapper around :func:`scanpy.tl.rank_genes_groups` with
    sensible defaults for perturbation screens.

    Parameters
    ----------
    adata
        AnnData with expression in ``.X`` and group labels in ``.obs``.
    groupby
        Column in ``.obs`` defining groups (e.g. ``"perturbation"``).
    reference
        Reference group for comparison.  ``"rest"`` compares each group
        against all others.
    method
        Statistical test: ``"wilcoxon"``, ``"t-test"``,
        ``"t-test_overestim_var"``, ``"logreg"``.
    n_genes
        Number of genes to report per group.  ``None`` reports all.
    key_added
        Key in ``adata.uns`` for the results.
    **kwargs
        Additional arguments passed to ``scanpy.tl.rank_genes_groups``.

    Returns
    -------
    AnnData with DEG results stored in ``uns[key_added]``.
    """
    sc = _require_scanpy()

    if groupby not in adata.obs.columns:
        raise KeyError(f"'{groupby}' not found in adata.obs. Available: {list(adata.obs.columns)}")

    sc_kwargs: dict[str, Any] = {
        "groupby": groupby,
        "reference": reference,
        "method": method,
        "key_added": key_added,
    }
    if n_genes is not None:
        sc_kwargs["n_genes"] = n_genes
    sc_kwargs.update(kwargs)

    sc.tl.rank_genes_groups(adata, **sc_kwargs)
    logger.info(
        "Ranked genes by '%s' (method=%s, reference=%s), stored in uns['%s'].",
        groupby,
        method,
        reference,
        key_added,
    )
    return adata


def get_deg_dataframe(
    adata: AnnData,
    group: str,
    key: str = "rank_genes_groups",
    n_top: int | None = None,
    pval_cutoff: float | None = 0.05,
    log2fc_min: float | None = None,
) -> pd.DataFrame:
    """Extract a DEG table for a specific group from scanpy results.

    Parameters
    ----------
    adata
        AnnData with :func:`rank_genes_groups` results in ``uns``.
    group
        Group name to extract (must be in the results).
    key
        Key in ``adata.uns`` for the DEG results.
    n_top
        Maximum number of top genes to return.  ``None`` returns all
        that pass the filters.
    pval_cutoff
        Maximum adjusted p-value.  ``None`` disables filtering.
    log2fc_min
        Minimum absolute log₂ fold change.  ``None`` disables filtering.

    Returns
    -------
    DataFrame with columns
    ``["gene", "score", "logfoldchange", "pval", "pval_adj"]``,
    sorted by absolute score descending.
    """
    sc = _require_scanpy()

    if key not in adata.uns:
        raise KeyError(f"'{key}' not found in adata.uns. Run rank_genes_groups() first.")

    result: pd.DataFrame = sc.get.rank_genes_groups_df(adata, group=group, key=key)

    if pval_cutoff is not None and "pvals_adj" in result.columns:
        result = result.loc[result["pvals_adj"] <= pval_cutoff]

    if log2fc_min is not None and "logfoldchanges" in result.columns:
        result = result.loc[result["logfoldchanges"].abs() >= log2fc_min]

    if n_top is not None:
        result = result.head(n_top)

    rename_map = {
        "names": "gene",
        "scores": "score",
        "logfoldchanges": "logfoldchange",
        "pvals": "pval",
        "pvals_adj": "pval_adj",
    }
    result = result.rename(columns={k: v for k, v in rename_map.items() if k in result.columns})

    return result.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Biological metrics — DEG-based
# ═══════════════════════════════════════════════════════════════════════════


def deg_overlap(
    genes_true: Sequence[str],
    genes_pred: Sequence[str],
) -> dict[str, float]:
    """Overlap between two gene lists.

    Parameters
    ----------
    genes_true
        Reference (ground-truth) gene list.
    genes_pred
        Predicted gene list.

    Returns
    -------
    Dictionary with ``"jaccard"``, ``"precision"``, ``"recall"``,
    and ``"n_shared"``.
    """
    set_true = set(genes_true)
    set_pred = set(genes_pred)

    intersection = len(set_true & set_pred)
    union = len(set_true | set_pred)

    return {
        "jaccard": intersection / union if union > 0 else 0.0,
        "precision": intersection / len(set_pred) if set_pred else 0.0,
        "recall": intersection / len(set_true) if set_true else 0.0,
        "n_shared": float(intersection),
    }


def deg_direction_agreement(
    deg_true: pd.DataFrame,
    deg_pred: pd.DataFrame,
    gene_col: str = "gene",
    lfc_col: str = "logfoldchange",
) -> dict[str, float]:
    """Fraction of shared DEGs with matching direction of change.

    Parameters
    ----------
    deg_true
        DEG table for the true expression (as returned by
        :func:`get_deg_dataframe`).
    deg_pred
        DEG table for the predicted expression.
    gene_col
        Column holding gene names.
    lfc_col
        Column holding log fold change values.

    Returns
    -------
    Dictionary with ``"direction_agreement"`` and ``"n_shared"``.
    """
    shared = set(deg_true[gene_col]) & set(deg_pred[gene_col])
    if not shared:
        return {"direction_agreement": float("nan"), "n_shared": 0.0}

    lfc_true = dict(zip(deg_true[gene_col], deg_true[lfc_col], strict=True))
    lfc_pred = dict(zip(deg_pred[gene_col], deg_pred[lfc_col], strict=True))

    agree = sum(1 for g in shared if np.sign(lfc_true[g]) == np.sign(lfc_pred[g]))

    return {
        "direction_agreement": agree / len(shared),
        "n_shared": float(len(shared)),
    }


def compare_deg(
    adata_true: AnnData,
    adata_pred: AnnData,
    groupby: str,
    reference: str = "rest",
    n_top_genes: int = 50,
    method: str = "wilcoxon",
    groups: list[str] | None = None,
) -> pd.DataFrame:
    """Compare DEGs between true and predicted expression across groups.

    High-level function that runs :func:`rank_genes_groups` on both
    AnnData objects, then computes :func:`deg_overlap` and
    :func:`deg_direction_agreement` for every perturbation group.

    Parameters
    ----------
    adata_true
        AnnData with true (observed) expression in ``.X``.
    adata_pred
        AnnData with predicted expression in ``.X``.  Must have the
        same ``.obs`` metadata as *adata_true*.
    groupby
        Column in ``.obs`` defining perturbation groups.
    reference
        Reference group (e.g. ``"control"``).
    n_top_genes
        Number of top DEGs to compare per group.
    method
        Statistical test for DEG analysis.
    groups
        Subset of groups to evaluate.  ``None`` evaluates all
        non-reference groups.

    Returns
    -------
    DataFrame with one row per group and columns ``["group", "jaccard",
    "precision", "recall", "n_shared", "direction_agreement"]``.
    """
    key_true = "_embpy_deg_true"
    key_pred = "_embpy_deg_pred"

    rank_genes_groups(
        adata_true,
        groupby=groupby,
        reference=reference,
        method=method,
        n_genes=n_top_genes,
        key_added=key_true,
    )
    rank_genes_groups(
        adata_pred,
        groupby=groupby,
        reference=reference,
        method=method,
        n_genes=n_top_genes,
        key_added=key_pred,
    )

    if groups is None:
        all_groups = adata_true.obs[groupby].unique().tolist()
        groups = sorted(g for g in all_groups if str(g) != reference)

    rows: list[dict[str, Any]] = []
    for grp in groups:
        grp_str = str(grp)
        try:
            df_true = get_deg_dataframe(
                adata_true,
                group=grp_str,
                key=key_true,
                n_top=n_top_genes,
                pval_cutoff=None,
            )
            df_pred = get_deg_dataframe(
                adata_pred,
                group=grp_str,
                key=key_pred,
                n_top=n_top_genes,
                pval_cutoff=None,
            )
        except (KeyError, ValueError, IndexError):
            logger.warning("Could not extract DEGs for group '%s', skipping.", grp)
            continue

        overlap = deg_overlap(df_true["gene"].tolist(), df_pred["gene"].tolist())
        direction = deg_direction_agreement(df_true, df_pred)

        rows.append(
            {
                "group": grp,
                **overlap,
                "direction_agreement": direction["direction_agreement"],
            }
        )

    adata_true.uns.pop(key_true, None)
    adata_pred.uns.pop(key_pred, None)

    return pd.DataFrame(rows)
