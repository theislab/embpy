"""Embedding benchmark module.

Evaluate how informative a perturbation embedding is for a prediction task
(e.g. predicting IC50 or gene expression) using standard ML regressors.

The main entry point is :func:`benchmark_embeddings`, which supports two
modes:

- ``mode="quick"`` -- single train/test split with default hyper-parameters.
  Fast, suitable for exploration.
- ``mode="rigorous"`` -- 5-fold cross-validation with randomised
  hyper-parameter search.  Slower, but provides mean +/- std metrics and
  tuned models.  Suitable for papers and final comparisons.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin

_DEFAULT_MODELS = ["linear", "ridge", "knn", "random_forest"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_target(adata: AnnData, target: str) -> np.ndarray:
    """Extract the prediction target as a dense numpy array.

    Parameters
    ----------
    adata
        Source AnnData.
    target
        ``"X"`` for the expression matrix, or a column name in ``.obs``.

    Returns
    -------
    2-D array when ``target="X"`` (shape ``n_obs x n_vars``),
    1-D array for a scalar ``.obs`` column.
    """
    if target == "X":
        X = adata.X
        if X is None:
            raise ValueError("adata.X is None — cannot use target='X'.")
        if sp.issparse(X):
            return np.asarray(X.todense())  # type: ignore[union-attr]
        return np.asarray(X)

    if target not in adata.obs.columns:
        raise KeyError(
            f"Target column '{target}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    vals = adata.obs[target].values
    if not np.issubdtype(vals.dtype, np.number):
        raise ValueError(
            f"Target column '{target}' must be numeric, got dtype={vals.dtype}."
        )
    return np.asarray(vals, dtype=np.float64)


def _build_model(name: str) -> RegressorMixin:
    """Instantiate an sklearn regressor by short name."""
    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "knn":
        return KNeighborsRegressor(n_neighbors=5)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
    raise ValueError(
        f"Unknown model '{name}'. Choose from: {_DEFAULT_MODELS}"
    )


def _build_param_grid(name: str) -> dict[str, list[Any]]:
    """Return a hyper-parameter grid for :class:`RandomizedSearchCV`."""
    if name == "linear":
        return {"fit_intercept": [True]}
    if name == "ridge":
        return {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    if name == "knn":
        return {
            "n_neighbors": [3, 5, 7, 11, 15],
            "weights": ["uniform", "distance"],
            "metric": ["cosine", "euclidean"],
        }
    if name == "random_forest":
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 50],
            "min_samples_split": [2, 5, 10],
        }
    raise ValueError(f"Unknown model '{name}'.")


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mean_control: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute evaluation metrics for a single regressor.

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
    is_multi = y_true.ndim == 2
    metrics: dict[str, float] = {}

    # -- MSE --
    metrics["mse"] = float(mean_squared_error(y_true, y_pred))

    # -- R² --
    if is_multi:
        r2_per_output = []
        for j in range(y_true.shape[1]):
            if np.std(y_true[:, j]) == 0:
                continue
            r2_per_output.append(r2_score(y_true[:, j], y_pred[:, j]))
        metrics["r2"] = float(np.mean(r2_per_output)) if r2_per_output else 0.0
    else:
        metrics["r2"] = float(r2_score(y_true, y_pred))

    # -- Pearson & Spearman --
    if is_multi:
        pearson_per_sample = []
        spearman_per_sample = []
        for i in range(y_true.shape[0]):
            if np.std(y_true[i]) == 0 or np.std(y_pred[i]) == 0:
                continue
            r_p, _ = pearsonr(y_true[i], y_pred[i])
            r_s, _ = spearmanr(y_true[i], y_pred[i])
            pearson_per_sample.append(r_p)
            spearman_per_sample.append(r_s)
        metrics["pearson"] = float(np.mean(pearson_per_sample)) if pearson_per_sample else 0.0
        metrics["spearman"] = float(np.mean(spearman_per_sample)) if spearman_per_sample else 0.0
    else:
        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            r_p, _ = pearsonr(y_true.ravel(), y_pred.ravel())
            r_s, _ = spearmanr(y_true.ravel(), y_pred.ravel())
            metrics["pearson"] = float(r_p)
            metrics["spearman"] = float(r_s)
        else:
            metrics["pearson"] = 0.0
            metrics["spearman"] = 0.0

    # -- Delta L2 (expression + controls only) --
    if is_multi and mean_control is not None:
        delta_actual = np.linalg.norm(y_true - mean_control, axis=1)
        delta_pred = np.linalg.norm(y_pred - mean_control, axis=1)
        metrics["delta_l2_mae"] = float(np.mean(np.abs(delta_pred - delta_actual)))
        if np.std(delta_actual) > 0 and np.std(delta_pred) > 0:
            r_delta, _ = pearsonr(delta_actual, delta_pred)
            metrics["delta_l2_pearson"] = float(r_delta)
        else:
            metrics["delta_l2_pearson"] = 0.0
    else:
        metrics["delta_l2_mae"] = float("nan")
        metrics["delta_l2_pearson"] = float("nan")

    return metrics


def _generate_embeddings(
    adata: AnnData,
    perturbation_column: str,
    perturbation_type: str,
    embedding_model: str,
    id_type: str = "gene_name",
    organism: str = "human",
) -> str:
    """Generate embeddings via BioEmbedder and store them in *adata.obsm*.

    Returns the obsm key where embeddings were stored.
    """
    from embpy.embedder import BioEmbedder
    from embpy.pp.basic import PerturbationProcessor

    embedder = BioEmbedder(device="auto")
    pp = PerturbationProcessor(embedder=embedder)

    if perturbation_column == "obs_names":
        identifiers = list(adata.obs_names)
    else:
        identifiers = list(adata.obs[perturbation_column].astype(str))

    obsm_key = f"X_{embedding_model}"

    if perturbation_type == "genetic":
        tmp = pp.build_embedding_matrix(
            identifiers=identifiers,
            model=embedding_model,
            id_type=id_type,
            organism=organism,
        )
        adata.obsm[obsm_key] = tmp.obsm[list(tmp.obsm.keys())[0]]
    elif perturbation_type == "chemical":
        tmp = pp.build_molecule_embedding_matrix(
            identifiers=identifiers,
            model=embedding_model,
        )
        adata.obsm[obsm_key] = tmp.obsm[list(tmp.obsm.keys())[0]]
    else:
        raise ValueError(
            f"Unknown perturbation_type '{perturbation_type}'. "
            "Choose 'genetic' or 'chemical'."
        )

    logger.info("Generated embeddings with '%s', stored in obsm['%s'].", embedding_model, obsm_key)
    return obsm_key


# ---------------------------------------------------------------------------
# Quick mode (single train/test split)
# ---------------------------------------------------------------------------


def _run_quick(
    X_emb: np.ndarray,
    y: np.ndarray,
    model_names: list[str],
    mean_control: np.ndarray | None,
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    """Run the quick (single-split) benchmark."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_emb, y, test_size=test_size, random_state=random_state,
    )
    logger.info("Quick split: %d train, %d test.", X_train.shape[0], X_test.shape[0])

    rows: list[dict[str, object]] = []
    for model_name in model_names:
        reg = _build_model(model_name)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        if y_pred.ndim == 1 and y_test.ndim == 2:
            y_pred = y_pred.reshape(-1, 1)

        mets = _compute_metrics(y_test, y_pred, mean_control=mean_control)
        mets["model"] = model_name
        rows.append(mets)
        logger.info("  %s → MSE=%.4f  R²=%.4f  Pearson=%.4f", model_name, mets["mse"], mets["r2"], mets["pearson"])

    return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------------------
# Rigorous mode (5-fold CV + randomised grid search)
# ---------------------------------------------------------------------------


def _run_rigorous(
    X_emb: np.ndarray,
    y: np.ndarray,
    model_names: list[str],
    mean_control: np.ndarray | None,
    cv: int = 5,
    n_iter: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run the rigorous (CV + hyper-parameter search) benchmark.

    For each model:
    1. ``RandomizedSearchCV`` finds the best hyper-parameters using
       ``neg_mean_squared_error`` on *cv* folds.
    2. A second manual *cv*-fold loop with the best estimator computes
       all custom metrics per fold (MSE, R², Pearson, Spearman, delta L2).
    3. Mean +/- std are reported for every metric.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    rows: list[dict[str, object]] = []
    for model_name in model_names:
        logger.info("Rigorous benchmark: %s (%d-fold CV, n_iter=%d)", model_name, cv, n_iter)

        base = _build_model(model_name)
        param_grid = _build_param_grid(model_name)

        actual_n_iter = min(n_iter, _grid_size(param_grid))

        search = RandomizedSearchCV(
            base,
            param_distributions=param_grid,
            n_iter=actual_n_iter,
            cv=cv,
            scoring="neg_mean_squared_error",
            random_state=random_state,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(X_emb, y)
        best_params = search.best_params_
        logger.info("  Best params: %s", best_params)

        fold_metrics: list[dict[str, float]] = []
        for train_idx, test_idx in kf.split(X_emb):
            X_tr, X_te = X_emb[train_idx], X_emb[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            estimator = _build_model(model_name)
            estimator.set_params(**best_params)
            estimator.fit(X_tr, y_tr)
            y_pred = estimator.predict(X_te)
            if y_pred.ndim == 1 and y_te.ndim == 2:
                y_pred = y_pred.reshape(-1, 1)

            fold_metrics.append(_compute_metrics(y_te, y_pred, mean_control=mean_control))

        fold_df = pd.DataFrame(fold_metrics)
        row: dict[str, object] = {"model": model_name}
        for col in fold_df.columns:
            vals = fold_df[col].dropna()
            if len(vals) > 0:
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std())
            else:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_std"] = float("nan")
        row["best_params"] = best_params
        rows.append(row)

        logger.info(
            "  %s → R²=%.4f±%.4f  Pearson=%.4f±%.4f",
            model_name,
            row.get("r2_mean", 0), row.get("r2_std", 0),
            row.get("pearson_mean", 0), row.get("pearson_std", 0),
        )

    return pd.DataFrame(rows).set_index("model")


def _grid_size(param_grid: dict[str, list[Any]]) -> int:
    """Compute the total number of combinations in a parameter grid."""
    size = 1
    for vals in param_grid.values():
        size *= len(vals)
    return size


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def benchmark_embeddings(
    adata: AnnData,
    perturbation_column: str,
    perturbation_type: str,
    target: str = "X",
    embedding_model: str | None = None,
    obsm_key: str | None = None,
    control_column: str | None = None,
    control_value: str | None = None,
    models: list[str] | None = None,
    mode: str = "quick",
    reduce_dim: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    id_type: str = "gene_name",
    organism: str = "human",
) -> pd.DataFrame:
    """Benchmark how informative an embedding is for a prediction task.

    Given an AnnData with perturbation identifiers and a prediction target
    (a scalar ``.obs`` column like IC50, or the full expression matrix ``.X``),
    this function trains multiple regressors on the embedding vectors and
    reports standard evaluation metrics.

    Parameters
    ----------
    adata
        AnnData object.  Must contain perturbation identifiers in ``.obs``
        (or ``.obs_names``), and either a numeric target column in ``.obs``
        or an expression matrix in ``.X``.
    perturbation_column
        Column in ``.obs`` with perturbation names, or the literal string
        ``"obs_names"`` to use the AnnData index.
    perturbation_type
        ``"genetic"`` or ``"chemical"``.  Used when generating embeddings
        on the fly.
    target
        ``"X"`` to predict the expression matrix, or the name of a numeric
        column in ``.obs`` (e.g. ``"ic50"``).
    embedding_model
        Model name from the ``MODEL_REGISTRY`` (e.g. ``"esm2_650M"``).
        If provided and *obsm_key* is ``None``, embeddings are generated
        automatically.
    obsm_key
        Key in ``.obsm`` pointing to pre-computed embeddings.  Takes
        priority over *embedding_model*.
    control_column
        Column in ``.obs`` that identifies control observations
        (e.g. ``"condition"``).
    control_value
        Value in *control_column* marking control rows (e.g. ``"DMSO"``).
    models
        Which regressors to benchmark.  Subset of
        ``["linear", "ridge", "knn", "random_forest"]``.
        Defaults to all four.
    mode
        ``"quick"`` for a single train/test split with default
        hyper-parameters, or ``"rigorous"`` for 5-fold cross-validation
        with randomised hyper-parameter search.
    reduce_dim
        If set, apply PCA to this many components before benchmarking.
    test_size
        Fraction of observations held out for testing (default 0.2).
        Only used in ``mode="quick"``.
    random_state
        Random seed for reproducibility.
    id_type
        Identifier type passed to the gene resolver (e.g.
        ``"gene_name"``, ``"ensembl_id"``).
    organism
        Organism string for the gene resolver.

    Returns
    -------
    DataFrame with one row per regressor.

    - **Quick mode** columns: ``mse``, ``r2``, ``pearson``, ``spearman``,
      ``delta_l2_mae``, ``delta_l2_pearson``.
    - **Rigorous mode** columns: ``<metric>_mean``, ``<metric>_std`` for
      each metric, plus ``best_params`` (dict of tuned hyper-parameters).

    The DataFrame is also stored in ``adata.uns["benchmark_results"]``.
    """
    if mode not in ("quick", "rigorous"):
        raise ValueError(f"Unknown mode '{mode}'. Choose 'quick' or 'rigorous'.")

    models = models or list(_DEFAULT_MODELS)
    for m in models:
        if m not in _DEFAULT_MODELS:
            raise ValueError(f"Unknown model '{m}'. Choose from: {_DEFAULT_MODELS}")

    # --- Validate perturbation_column ---
    if perturbation_column != "obs_names" and perturbation_column not in adata.obs.columns:
        raise KeyError(
            f"perturbation_column '{perturbation_column}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    # --- Validate target ---
    if target != "X" and target not in adata.obs.columns:
        raise KeyError(
            f"Target '{target}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    # --- Obtain embeddings ---
    if obsm_key is not None:
        if obsm_key not in adata.obsm:
            raise KeyError(f"obsm_key '{obsm_key}' not found in adata.obsm.")
        emb_key = obsm_key
    elif embedding_model is not None:
        emb_key = _generate_embeddings(
            adata,
            perturbation_column=perturbation_column,
            perturbation_type=perturbation_type,
            embedding_model=embedding_model,
            id_type=id_type,
            organism=organism,
        )
    else:
        raise ValueError("Provide either 'obsm_key' (pre-computed) or 'embedding_model' (generate).")

    # --- Optional dimensionality reduction ---
    if reduce_dim is not None:
        from embpy.pp.basic import reduce_embeddings

        reduced_key = f"{emb_key}_pca{reduce_dim}"
        reduce_embeddings(adata, obsm_key=emb_key, n_components=reduce_dim, output_key=reduced_key)
        emb_key = reduced_key
        logger.info("Applied PCA → %d dims, stored in obsm['%s'].", reduce_dim, emb_key)

    # --- Extract features and target ---
    X_emb = np.asarray(adata.obsm[emb_key], dtype=np.float64)
    y = _get_target(adata, target)

    # --- Identify controls ---
    has_controls = control_column is not None and control_value is not None
    mean_control: np.ndarray | None = None

    if has_controls:
        if control_column not in adata.obs.columns:
            raise KeyError(f"control_column '{control_column}' not found in adata.obs.")
        ctrl_mask = adata.obs[control_column].astype(str) == str(control_value)
        pert_mask = ~ctrl_mask

        if ctrl_mask.sum() == 0:
            raise ValueError(
                f"No control observations found where "
                f"obs['{control_column}'] == '{control_value}'."
            )

        if target == "X":
            mean_control = np.mean(y[ctrl_mask.values], axis=0)

        X_emb = X_emb[pert_mask.values]
        y = y[pert_mask.values]
        logger.info(
            "Separated %d controls and %d perturbations.",
            ctrl_mask.sum(), pert_mask.sum(),
        )

    # --- Run the selected mode ---
    if mode == "quick":
        results = _run_quick(
            X_emb, y,
            model_names=models,
            mean_control=mean_control,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        results = _run_rigorous(
            X_emb, y,
            model_names=models,
            mean_control=mean_control,
            random_state=random_state,
        )

    adata.uns["benchmark_results"] = results
    return results
