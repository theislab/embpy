"""Cross-modal alignment between protein embeddings and transcriptomic phenotypes.

Evaluates whether protein sequence embeddings encode enough functional biology
to predict which genetic perturbations produce similar transcriptomic responses.

Public API
----------
- :func:`protein_phenotype_alignment` — full alignment scorecard for one or
  more protein embedding models vs. a single-cell phenotype space.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _pseudobulk_dual(
    adata: AnnData,
    perturbation_col: str,
    sc_obsm_key: str,
    protein_obsm_key: str,
    control_col: str | None,
    control_ids: set[str] | None,
) -> AnnData:
    """Pseudobulk-aggregate both obsm keys in a single pass.

    Groups observations by *perturbation_col*, computes the per-group mean
    for each embedding space, and returns a compact AnnData where every row
    is a unique perturbation.  Both embedding keys are preserved in the
    returned object.

    Parameters
    ----------
    adata
        Cell-level AnnData with replicate observations per perturbation.
    perturbation_col
        Column in ``.obs`` identifying perturbation groups.
    sc_obsm_key, protein_obsm_key
        Keys in ``.obsm`` for the two embedding spaces.
    control_col
        Optional column that marks control observations.
    control_ids
        Values in *control_col* that identify controls to exclude.

    Returns
    -------
    AnnData with one row per perturbation and both obsm keys populated.
    """
    import anndata as ad

    from embpy.tl.similarity import _get_embedding

    # filter controls
    mask = np.ones(adata.n_obs, dtype=bool)
    if control_col is not None and control_ids is not None:
        if control_col not in adata.obs.columns:
            raise KeyError(f"control_col '{control_col}' not in adata.obs.")
        mask = ~adata.obs[control_col].isin(control_ids).values

    adata_filt = adata[mask]
    groups = adata_filt.obs[perturbation_col].astype(str).values
    # lexicographic order for reproducibility
    unique_groups = sorted(set(groups))

    X_sc = _get_embedding(adata_filt, sc_obsm_key)
    X_prot = _get_embedding(adata_filt, protein_obsm_key)

    mean_sc = np.array([X_sc[groups == g].mean(axis=0) for g in unique_groups], dtype=np.float64)
    mean_prot = np.array([X_prot[groups == g].mean(axis=0) for g in unique_groups], dtype=np.float64)

    adata_pb = ad.AnnData(obs=pd.DataFrame(index=unique_groups))
    adata_pb.obs_names = unique_groups
    adata_pb.obs[perturbation_col] = unique_groups
    adata_pb.obsm[sc_obsm_key] = mean_sc
    adata_pb.obsm[protein_obsm_key] = mean_prot

    logger.info(
        "Pseudobulk: %d cells → %d perturbation groups (col='%s', %d controls excluded).",
        adata_filt.n_obs,
        len(unique_groups),
        perturbation_col,
        int((~mask).sum()),
    )
    return adata_pb


def _cross_modal_phenocopy(
    emb_sc: np.ndarray,
    emb_prot: np.ndarray,
    n_pca_components: int | None,
    mad_thresholds: Sequence[int],
    recall_ks: Sequence[int],
) -> dict[str, float]:
    """Compute cross-modal phenocopy AUROC and Recall@K.

    Measures whether protein-embedding similarity predicts transcriptomic
    phenotype similarity.  Adapts the PRESAGE phenocopy framework to work
    across two spaces that may have different dimensionalities.

    Algorithm
    ---------
    1. Independently PCA-reduce each embedding space.
    2. Compute pairwise cosine-similarity matrices within each space.
    3. For each perturbation, define *true phenocopy neighbors* in scRNA-seq
       space using MAD thresholds, then evaluate whether protein similarity
       recovers these neighbors (AUROC) or the top-K matches (Recall@K).

    Parameters
    ----------
    emb_sc
        Mean scRNA-seq embedding per perturbation, shape
        ``(n_perturbations, d_sc)``.  Serves as the **ground-truth** space.
    emb_prot
        Protein embedding per perturbation, shape
        ``(n_perturbations, d_prot)``.  Serves as the **predictor** space.
    n_pca_components
        Number of PCA components applied independently to each matrix before
        computing similarities.  ``None`` skips PCA.
    mad_thresholds
        MAD multipliers used to define phenocopy neighbor thresholds.
    recall_ks
        Values of *k* for Recall@K.

    Returns
    -------
    Dict with keys ``"cross_modal_auroc_mad_{t}"`` and
    ``"cross_modal_recall_{k}"``.
    """
    n = emb_sc.shape[0]

    if n_pca_components is not None:
        n_comp_sc = min(n_pca_components, n, emb_sc.shape[1])
        n_comp_prot = min(n_pca_components, n, emb_prot.shape[1])
        sc_reduced = PCA(n_components=n_comp_sc, random_state=0).fit_transform(emb_sc)
        prot_reduced = PCA(n_components=n_comp_prot, random_state=0).fit_transform(emb_prot)
    else:
        sc_reduced = emb_sc
        prot_reduced = emb_prot

    sim_sc = cosine_similarity(sc_reduced)    # ground-truth similarity
    sim_prot = cosine_similarity(prot_reduced)  # predicted similarity

    metrics: dict[str, float] = {}

    # Precompute boolean mask for excluding self (reused across iterations)
    all_idx = np.arange(n)

    # AUROC at multiple MAD thresholds
    for t in mad_thresholds:
        aurocs: list[float] = []
        for i in range(n):
            # exclude self by boolean slicing — avoids passing ±inf to roc_auc_score
            keep = all_idx != i
            sc_no_self = sim_sc[i, keep]
            prot_no_self = sim_prot[i, keep]

            mad = float(median_abs_deviation(sc_no_self))
            if mad == 0.0:
                continue
            neighbors = sc_no_self > (float(np.median(sc_no_self)) + t * mad)
            n_pos = int(neighbors.sum())
            if n_pos == 0 or n_pos == n - 1:
                continue
            aurocs.append(float(roc_auc_score(neighbors, prot_no_self)))
        metrics[f"cross_modal_auroc_mad_{t}"] = float(np.mean(aurocs)) if aurocs else float("nan")

    # Recall@K
    for k in recall_ks:
        if k >= n:
            metrics[f"cross_modal_recall_{k}"] = float("nan")
            continue
        recalls: list[float] = []
        for i in range(n):
            keep = all_idx != i
            sc_no_self = sim_sc[i, keep]
            prot_no_self = sim_prot[i, keep]
            sc_topk = set(np.argsort(sc_no_self)[-k:])
            prot_topk = set(np.argsort(prot_no_self)[-k:])
            recalls.append(len(sc_topk & prot_topk) / k)
        metrics[f"cross_modal_recall_{k}"] = float(np.mean(recalls))

    return metrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def protein_phenotype_alignment(
    adata: AnnData,
    protein_obsm_key: str,
    sc_obsm_key: str,
    *,
    perturbation_col: str | None = None,
    k: int = 15,
    n_pca_components: int | None = 50,
    mad_thresholds: Sequence[int] = (1, 2, 3, 5),
    recall_ks: Sequence[int] = (5, 10, 20, 50),
    control_col: str | None = None,
    control_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Measure alignment between protein embeddings and transcriptomic phenotypes.

    Tests the core embpy assumption: proteins that are similar in embedding
    space, when knocked out, produce similar transcriptomic responses.  Runs
    three complementary metrics at perturbation level:

    1. **Mantel test** — Spearman correlation between pairwise distance
       matrices in protein space and scRNA-seq phenotype space.
    2. **kNN Jaccard overlap** — fraction of each perturbation's k nearest
       protein-space neighbors that are also phenotype-space neighbors.
    3. **Cross-modal phenocopy** — AUROC and Recall@K measuring how well
       protein similarity rankings recover true phenocopy neighbors
       (defined by MAD thresholds in scRNA-seq space).

    Parameters
    ----------
    adata
        AnnData containing both embeddings.  If *perturbation_col* is
        ``None``, each row is assumed to be one perturbation (already
        pseudobulked).  If *perturbation_col* is provided, cell-level
        replicates are automatically pseudobulk-averaged per group.
    protein_obsm_key
        Key in ``.obsm`` for the protein embedding (predictor space).
    sc_obsm_key
        Key in ``.obsm`` for the scRNA-seq / transcriptomic embedding
        (ground-truth phenotype space).
    perturbation_col
        Column in ``.obs`` identifying perturbation groups.  When provided,
        the function pseudobulk-aggregates both embedding spaces before
        computing metrics.  Leave ``None`` if *adata* is already at
        perturbation level.
    k
        Number of nearest neighbours for kNN Jaccard overlap.
    n_pca_components
        Number of PCA components applied independently to each embedding
        space before computing similarities in the cross-modal phenocopy
        step.  ``None`` disables PCA.
    mad_thresholds
        MAD multipliers that define what counts as a true phenocopy
        neighbour in scRNA-seq space.
    recall_ks
        Values of *k* for Recall@K in the cross-modal phenocopy metric.
    control_col
        Column in ``.obs`` that marks control observations (only used when
        *perturbation_col* is set).
    control_ids
        Set of values in *control_col* that identify controls to exclude
        before pseudobulk aggregation.

    Returns
    -------
    Dictionary with the following keys:

    Scalars (global metrics):

    - ``"mantel_rho"`` / ``"mantel_pval"`` — Mantel test result.
    - ``"knn_jaccard_mean"`` — Mean Jaccard overlap across all perturbations.
    - ``"cross_modal_auroc_mad_{t}"`` for each *t* in *mad_thresholds*.
    - ``"cross_modal_recall_{k}"`` for each *k* in *recall_ks*.
    - ``"n_perturbations"`` — number of perturbations evaluated.

    Arrays (per-perturbation):

    - ``"knn_jaccard_per_perturbation"`` — 1-D float64 array, one value
      per perturbation row in the (aggregated) AnnData.
    - ``"perturbation_names"`` — list of perturbation identifiers matching
      the array above.

    Metadata:

    - ``"protein_obsm_key"``, ``"sc_obsm_key"`` — keys used.

    Examples
    --------
    Perturbation-level AnnData (already pseudobulked):

    >>> results = tl.protein_phenotype_alignment(
    ...     adata_pb,
    ...     protein_obsm_key="X_esm2_650M",
    ...     sc_obsm_key="X_scgpt",
    ... )
    >>> print(f"Mantel ρ = {results['mantel_rho']:.3f}")
    >>> print(f"kNN Jaccard (k=15) = {results['knn_jaccard_mean']:.3f}")
    >>> print(f"AUROC (MAD 1) = {results['cross_modal_auroc_mad_1']:.3f}")

    Cell-level AnnData with replicates (pseudobulk on the fly):

    >>> results = tl.protein_phenotype_alignment(
    ...     adata,
    ...     protein_obsm_key="X_esm2_650M",
    ...     sc_obsm_key="X_scgpt",
    ...     perturbation_col="gene",
    ...     control_col="is_control",
    ...     control_ids={True},
    ... )

    Compare multiple protein models on the same scRNA-seq phenotype space:

    >>> models = ["esm2_8M", "esm2_650M", "esm2_15B", "prott5_xl"]
    >>> scorecards = {
    ...     m: tl.protein_phenotype_alignment(adata_pb, f"X_{m}", "X_scgpt")
    ...     for m in models
    ... }
    >>> pd.DataFrame(
    ...     {m: {k: v for k, v in sc.items() if isinstance(v, float)}
    ...      for m, sc in scorecards.items()}
    ... ).T
    """
    from embpy.tl.similarity import _get_embedding, compute_knn_overlap, cross_modal_mantel

    # ---- pseudobulk if cell-level adata is provided ----------------------
    if perturbation_col is not None:
        if perturbation_col not in adata.obs.columns:
            raise KeyError(
                f"'{perturbation_col}' not in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        adata_eval = _pseudobulk_dual(
            adata,
            perturbation_col=perturbation_col,
            sc_obsm_key=sc_obsm_key,
            protein_obsm_key=protein_obsm_key,
            control_col=control_col,
            control_ids=control_ids,
        )
    else:
        # validate both keys exist before proceeding
        _get_embedding(adata, protein_obsm_key)
        _get_embedding(adata, sc_obsm_key)
        adata_eval = adata

    n = adata_eval.n_obs
    if n < 3:
        raise ValueError(
            f"Need at least 3 perturbations for alignment metrics, got {n}. "
            "Check your data or perturbation_col / control filtering."
        )

    logger.info(
        "protein_phenotype_alignment: n=%d, protein='%s', sc='%s', k=%d",
        n, protein_obsm_key, sc_obsm_key, k,
    )

    # ---- 1. Mantel test --------------------------------------------------
    mantel_rho, mantel_pval = cross_modal_mantel(adata_eval, protein_obsm_key, sc_obsm_key)

    # ---- 2. kNN Jaccard overlap ------------------------------------------
    jaccard_per_pert, knn_jaccard_mean = compute_knn_overlap(
        adata_eval, protein_obsm_key, sc_obsm_key, k=k,
    )

    # ---- 3. Cross-modal phenocopy ----------------------------------------
    emb_sc = np.asarray(adata_eval.obsm[sc_obsm_key], dtype=np.float64)
    emb_prot = np.asarray(adata_eval.obsm[protein_obsm_key], dtype=np.float64)

    phenocopy_metrics = _cross_modal_phenocopy(
        emb_sc=emb_sc,
        emb_prot=emb_prot,
        n_pca_components=n_pca_components,
        mad_thresholds=mad_thresholds,
        recall_ks=recall_ks,
    )

    # ---- assemble output -------------------------------------------------
    results: dict[str, Any] = {
        "mantel_rho": mantel_rho,
        "mantel_pval": mantel_pval,
        "knn_jaccard_mean": knn_jaccard_mean,
        **phenocopy_metrics,
        "n_perturbations": n,
        "knn_jaccard_per_perturbation": jaccard_per_pert,
        "perturbation_names": list(adata_eval.obs_names),
        "protein_obsm_key": protein_obsm_key,
        "sc_obsm_key": sc_obsm_key,
    }

    logger.info(
        "Alignment summary — Mantel ρ=%.4f, kNN Jaccard=%.4f, "
        "AUROC(MAD1)=%.4f, Recall@%d=%.4f",
        mantel_rho,
        knn_jaccard_mean,
        phenocopy_metrics.get(f"cross_modal_auroc_mad_{mad_thresholds[0]}", float("nan")),
        recall_ks[0],
        phenocopy_metrics.get(f"cross_modal_recall_{recall_ks[0]}", float("nan")),
    )

    return results


def alignment_summary(results: dict[str, Any]) -> pd.DataFrame:
    """Format :func:`protein_phenotype_alignment` results as a tidy DataFrame.

    Extracts the scalar metrics from the dict returned by
    :func:`protein_phenotype_alignment` — useful for comparing multiple
    protein models side-by-side.

    Parameters
    ----------
    results
        Single result dict, or a mapping ``{model_name: result_dict}``
        for multi-model comparison.

    Returns
    -------
    :class:`~pandas.DataFrame`

    - Single-result mode: one-column DataFrame indexed by metric name.
    - Multi-model mode: rows = models, columns = metrics.

    Examples
    --------
    >>> tl.alignment_summary(results)
    >>> tl.alignment_summary({"esm2_650M": r1, "prott5_xl": r2})
    """
    def _scalars(d: dict[str, Any]) -> dict[str, float]:
        return {k: v for k, v in d.items() if isinstance(v, float)}

    # single result dict
    if all(not isinstance(v, dict) for v in results.values()):
        return pd.DataFrame.from_dict(_scalars(results), orient="index", columns=["value"])

    # multi-model dict-of-dicts
    return pd.DataFrame({model: _scalars(res) for model, res in results.items()}).T
