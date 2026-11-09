"""Embedding benchmark module.

Evaluate how informative a perturbation embedding is for a prediction task
(e.g. predicting IC50 or gene expression) using standard ML regressors.

The main entry point is :func:`benchmark_embeddings`, which supports two
modes:

- ``mode="quick"`` -- single train/test split with default hyper-parameters.
  Fast, suitable for exploration.
- ``mode="rigorous"`` -- 5-fold cross-validation with nested randomised
  hyper-parameter search.  Slower, but provides mean +/- std metrics and
  tuned models.  Suitable for papers and final comparisons.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupShuffleSplit, KFold, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from embpy.tl.metrics import compute_metrics as _compute_metrics

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin

# Supported regressors. All rely only on scikit-learn (a core dependency), so
# the benchmark runs on a lightweight install. (An xgboost backend exists in
# `_build_model` but is intentionally not in this set -- it needs the optional
# `xgboost` package, so it is not offered as a default/validated choice.)
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
        raise KeyError(f"Target column '{target}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")
    vals = adata.obs[target].values
    if not np.issubdtype(vals.dtype, np.number):
        raise ValueError(f"Target column '{target}' must be numeric, got dtype={vals.dtype}.")
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
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError(
                "xgboost is required for the 'xgboost' model but is not installed. "
                "Install it with:  pip install xgboost"
            ) from e
        return XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=0,
            n_jobs=-1,
            verbosity=0,
        )
    raise ValueError(f"Unknown model '{name}'. Choose from: {_DEFAULT_MODELS}")


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
    if name == "xgboost":
        return {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.3],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
        }
    raise ValueError(f"Unknown model '{name}'.")


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
        raise ValueError(f"Unknown perturbation_type '{perturbation_type}'. Choose 'genetic' or 'chemical'.")

    logger.info("Generated embeddings with '%s', stored in obsm['%s'].", embedding_model, obsm_key)
    return obsm_key


# ---------------------------------------------------------------------------
# Quick mode (single train/test split)
# ---------------------------------------------------------------------------


def _splits(n, *, groups=None, mode="rigorous", cv=5, test_size=0.2, random_state=42):
    """Generate paired folds, keeping all rows of a group together."""
    rows = np.arange(n)
    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != n or pd.isna(groups).any():
            raise ValueError("groups must provide one nonmissing label per entity.")
        groups = groups.astype(str)
    if mode == "quick":
        if groups is None:
            return [train_test_split(rows, test_size=test_size, random_state=random_state)]
        return list(
            GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state).split(rows, groups=groups)
        )
    if mode != "rigorous":
        raise ValueError("mode must be 'quick' or 'rigorous'.")
    if groups is None:
        return list(KFold(n_splits=cv, shuffle=True, random_state=random_state).split(rows))
    unique = np.unique(groups)
    folds = KFold(n_splits=cv, shuffle=True, random_state=random_state).split(unique)
    return [
        (np.flatnonzero(np.isin(groups, unique[tr])), np.flatnonzero(np.isin(groups, unique[te]))) for tr, te in folds
    ]


def _fit_fold(
    X,
    y,
    train,
    test,
    *,
    model_name,
    transformer=None,
    groups=None,
    tune=False,
    cv=3,
    n_iter=20,
    random_state=42,
    n_jobs=1,
    param_grid=None,
    fusion_param_grid=None,
    memory=None,
):
    """Fit preprocessing and tune the readout entirely inside an outer training fold."""
    pipeline = Pipeline(
        [
            ("features", "passthrough" if transformer is None else clone(transformer)),
            ("regressor", _build_model(model_name)),
        ],
        memory=memory,
    )
    parameters = {}
    if tune:
        train_groups = None if groups is None else np.asarray(groups)[train]
        available = len(train) if train_groups is None else len(np.unique(train_groups))
        inner_cv = min(cv, available)
        if inner_cv < 2:
            raise ValueError("Hyperparameter tuning requires at least two training entities/groups.")
        inner = _splits(len(train), groups=train_groups, cv=inner_cv, random_state=random_state)
        grid = _build_param_grid(model_name) if param_grid is None else param_grid
        grid = {"regressor__" + key: values for key, values in grid.items()}
        grid.update({"features__fusion__" + key: values for key, values in (fusion_param_grid or {}).items()})
        search = RandomizedSearchCV(
            pipeline,
            grid,
            n_iter=min(n_iter, _grid_size(grid)),
            cv=inner,
            scoring="neg_mean_squared_error",
            random_state=random_state,
            n_jobs=n_jobs,
            error_score="raise",
        )
        search.fit(X[train], y[train])
        pipeline = search.best_estimator_
        parameters = {
            key.removeprefix("regressor__").removeprefix("features__"): value
            for key, value in search.best_params_.items()
        }
    else:
        pipeline.fit(X[train], y[train])
    prediction = pipeline.predict(X[test])
    if prediction.ndim == 1 and y.ndim == 2:
        prediction = prediction.reshape(-1, 1)
    return pipeline, prediction, parameters


def _run_quick(X_emb, y, model_names, mean_control, test_size, random_state, transformer=None, groups=None, n_jobs=1):
    """Run a single held-out evaluation with preprocessing fit on training rows."""
    train, test = _splits(len(y), groups=groups, mode="quick", test_size=test_size, random_state=random_state)[0]
    rows = []
    for name in model_names:
        _, prediction, _ = _fit_fold(X_emb, y, train, test, model_name=name, transformer=transformer, n_jobs=n_jobs)
        rows.append({"model": name, **_compute_metrics(y[test], prediction, mean_control=mean_control)})
    return pd.DataFrame(rows).set_index("model")


def _run_rigorous(
    X_emb, y, model_names, mean_control, cv=5, n_iter=20, random_state=42, transformer=None, groups=None, n_jobs=1
):
    """Run nested CV; both preprocessing and tuning exclude each outer test fold."""
    splits = _splits(len(y), groups=groups, cv=cv, random_state=random_state)
    rows = []
    for name in model_names:
        metrics, params = [], []
        for train, test in splits:
            _, prediction, best = _fit_fold(
                X_emb,
                y,
                train,
                test,
                model_name=name,
                transformer=transformer,
                groups=groups,
                tune=True,
                cv=3,
                n_iter=n_iter,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            metrics.append(_compute_metrics(y[test], prediction, mean_control=mean_control))
            params.append(best)
        frame = pd.DataFrame(metrics)
        row = {"model": name, "best_params": params}
        for col in frame:
            row[col + "_mean"] = float(frame[col].mean())
            row[col + "_std"] = float(frame[col].std())
        rows.append(row)
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
    *,
    obsm_keys: Sequence[str] | None = None,
    fusion: Any = None,
    groupby: str | None = None,
    n_jobs: int = 1,
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
        If set, fit PCA inside each training fold. No globally fitted PCA is
        written to obsm. Use the fusion operator separately for exploratory plots.
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

    obsm_keys
        Multiple precomputed views; combined with ``fusion`` (plain concatenation
        by default). Mutually exclusive with obsm_key and embedding_model.
    fusion
        Unfitted BaseFusion operator, including CCA, GCCA or Procrustes.
        Cloned inside each training fold. For fusion-parameter grids or
        per-configuration view subsets, use benchmark_fusion instead.
    groupby
        Optional obs column whose groups must stay together in all outer and
        inner splits. Recommended for repeated drugs, genes, donors or batches.
    n_jobs
        Parallelism for inner hyperparameter search; defaults to one.

    Returns
    -------
    DataFrame with one row per regressor.

    - **Quick mode** columns: ``mse``, ``r2``, ``pearson``, ``spearman``,
      ``delta_l2_mae``, ``delta_l2_pearson``.
    - **Rigorous mode** columns: ``<metric>_mean``, ``<metric>_std`` for
      each metric, plus ``best_params`` (one parameter dictionary per outer fold).

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
            f"perturbation_column '{perturbation_column}' not found in adata.obs. Available: {list(adata.obs.columns)}"
        )

    # --- Validate target ---
    if target != "X" and target not in adata.obs.columns:
        raise KeyError(f"Target '{target}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")

    # Prepare raw views only; all fitted transforms belong inside the pipeline.
    from embpy.tl.fusion import ConcatFusion, PCAFusion, views_from_anndata
    from embpy.tl.fusion._benchmark import MatrixFusion

    if obsm_keys is not None:
        if obsm_key is not None or embedding_model is not None:
            raise ValueError("Use obsm_keys or a single obsm_key/embedding_model, not both.")
        keys = list(obsm_keys) if not isinstance(obsm_keys, str) else obsm_keys
    elif obsm_key is not None:
        keys = [obsm_key]
    elif embedding_model is not None:
        keys = [_generate_embeddings(adata, perturbation_column, perturbation_type, embedding_model, id_type, organism)]
    else:
        raise ValueError("Provide obsm_key, obsm_keys, or embedding_model.")
    if fusion is not None and reduce_dim is not None:
        raise ValueError("Specify a fusion operator or reduce_dim, not both; use PCAFusion for joint PCA.")
    views = views_from_anndata(adata, keys)
    if fusion is None and reduce_dim is not None:
        fusion = PCAFusion(n_components=reduce_dim, normalization="zscore", random_state=random_state)
    elif fusion is None and len(views) > 1:
        fusion = ConcatFusion()
    transformer = (
        None if fusion is None else MatrixFusion(fusion, tuple(views), tuple(v.n_dims for v in views.values()))
    )
    X_emb = np.concatenate([v.matrix for v in views.values()], axis=1)
    y = _get_target(adata, target)
    groups = None if groupby is None else np.asarray(adata.obs[groupby])
    if not np.isfinite(y).all():
        raise ValueError("Benchmark targets must be finite.")

    # --- Identify controls ---
    has_controls = control_column is not None and control_value is not None
    mean_control: np.ndarray | None = None

    if has_controls:
        if control_column not in adata.obs.columns:
            raise KeyError(f"control_column '{control_column}' not found in adata.obs.")
        ctrl_mask = adata.obs[control_column].astype(str) == str(control_value)
        pert_mask = ~ctrl_mask

        if ctrl_mask.sum() == 0:
            raise ValueError(f"No control observations found where obs['{control_column}'] == '{control_value}'.")

        if target == "X":
            mean_control = np.mean(y[ctrl_mask.values], axis=0)

        X_emb = X_emb[pert_mask.values]
        y = y[pert_mask.values]
        if groups is not None:
            groups = groups[pert_mask.values]
        logger.info(
            "Separated %d controls and %d perturbations.",
            ctrl_mask.sum(),
            pert_mask.sum(),
        )

    # --- Run the selected mode ---
    if mode == "quick":
        results = _run_quick(
            X_emb,
            y,
            model_names=models,
            mean_control=mean_control,
            test_size=test_size,
            random_state=random_state,
            transformer=transformer,
            groups=groups,
            n_jobs=n_jobs,
        )
    else:
        results = _run_rigorous(
            X_emb,
            y,
            model_names=models,
            mean_control=mean_control,
            random_state=random_state,
            transformer=transformer,
            groups=groups,
            n_jobs=n_jobs,
        )

    adata.uns["benchmark_results"] = results
    adata.uns["benchmark_config"] = {
        "fit_scope": "training_fold",
        "groupby": groupby,
        "obsm_keys": list(keys),
        "nested_cv": mode == "rigorous",
    }
    return results


@dataclass
class FusionBenchmarkResult:
    """Fold metrics, entity metrics and explicit splits from a paired benchmark.

    ``scores`` has one row per configuration/seed/fold. ``entity_scores`` keeps
    out-of-fold metrics for paired downstream comparisons, without retaining
    large prediction matrices. ``splits`` records every entity's partition.
    Fold standard deviations are descriptive, not confidence intervals.
    """

    scores: pd.DataFrame
    entity_scores: pd.DataFrame
    splits: pd.DataFrame
    coverage: dict[str, Any]
    config: dict[str, Any]


def _indexed_values(values, ids, name):
    if not isinstance(values, (pd.Series, pd.DataFrame)):
        raise TypeError(f"{name} must be a pandas Series/DataFrame indexed by canonical entity ID.")
    values = values.copy()
    values.index = values.index.astype(str)
    if not values.index.is_unique:
        raise ValueError(f"{name} has duplicate entity IDs.")
    if not set(ids).issubset(values.index):
        raise ValueError(f"{name} is missing values for some aligned entity IDs.")
    values = values.reindex(ids)
    if values.isna().to_numpy().any():
        raise ValueError(f"{name} contains missing values.")
    return values.to_numpy()


def _fusion_inputs(views, labels, groups, join):
    from embpy.tl.fusion import align_views

    aligned, coverage = align_views(views, join=join)
    ids = next(iter(aligned.values())).entity_ids
    values = _indexed_values(labels, ids, "targets")
    group_values = None if groups is None else _indexed_values(groups, ids, "groups")
    if group_values is not None and group_values.ndim != 1:
        raise ValueError("groups must be a Series, not a DataFrame.")
    return aligned, ids, values, group_values, coverage


def _fusion_configs(views, fusions, include_single, single_normalization, single_n_components, random_state):
    from embpy.tl.fusion import BaseFusion, ConcatFusion, FusionConfig, PCAFusion

    configs = {}
    if include_single:
        for name in views:
            operator = (
                ConcatFusion(normalization=single_normalization)
                if single_n_components is None
                else PCAFusion(
                    n_components=single_n_components,
                    normalization=single_normalization,
                    random_state=random_state,
                )
            )
            configs["single:" + name] = ({name: views[name]}, operator, {})
    if fusions is None:
        fusions = {"concat_l2": ConcatFusion(normalization="l2")}
    for name, specification in fusions.items():
        if not isinstance(name, str) or not name or name in configs:
            raise ValueError("Fusion configuration names must be nonempty and distinct from single-view baselines.")
        specification = specification if isinstance(specification, FusionConfig) else FusionConfig(specification)
        operator = specification.operator
        if not isinstance(operator, BaseFusion):
            raise TypeError("Feature benchmarks require a BaseFusion operator.")
        names = list(views) if specification.views is None else list(specification.views)
        if not names or len(set(names)) != len(names) or not set(names).issubset(views):
            raise ValueError("FusionConfig.views must name a nonempty, unique subset of supplied views.")
        grid = dict(specification.param_grid or {})
        if not set(grid).issubset(operator.get_params()):
            raise ValueError("FusionConfig.param_grid contains unknown operator parameters.")
        configs[name] = ({key: views[key] for key in names}, operator, grid)
    if not configs:
        raise ValueError("At least one benchmark configuration is required.")
    return configs


def _split_records(ids, groups, seed, fold, train, test):
    return [
        {
            "seed": seed,
            "fold": fold,
            "entity_id": ids[i],
            "group": ids[i] if groups is None else str(groups[i]),
            "partition": partition,
        }
        for partition, rows in (("train", train), ("test", test))
        for i in rows
    ]


def benchmark_fusion(
    views: Mapping[str, Any],
    targets: pd.Series | pd.DataFrame,
    *,
    fusions: Mapping[str, Any] | None = None,
    groups: pd.Series | None = None,
    include_single: bool = True,
    single_normalization: str = "l2",
    single_n_components: int | None = None,
    join: str = "strict",
    model: str = "ridge",
    cv: int = 5,
    seeds: Sequence[int] = (0,),
    tune: bool = True,
    param_grid: dict[str, list[Any]] | None = None,
    n_jobs: int = 1,
    cache_dir: str | None = None,
) -> FusionBenchmarkResult:
    """Compare feature fusions on identical held-out entities using nested CV.

    Inputs are named EmbeddingResult views and targets indexed by canonical ID.
    Use ``groups`` to keep compound aliases, donors or other related entities
    together in both outer and inner folds. Every learned preprocessing step
    is fitted inside the relevant training fold. Targets are never normalized
    implicitly. ``tune=True`` selects readout parameters by three-fold inner
    CV on negative MSE, including operator parameters supplied in FusionConfig.
    FusionConfig can select a view subset; every configuration still uses the
    same cohort aligned across all supplied views. ``cache_dir`` optionally
    caches training transforms across readout candidates (caller manages its
    lifetime). Include a separate final holdout if selecting
    the best configuration for a downstream claim.

    The single-view baselines use the specified normalization, optionally PCA
    to the same width as a PCAFusion comparator. Full concatenations retain
    their larger width, which is reported in ``scores.output_dim``. No fit is
    performed on the supplied operator instances; all fits use clones.
    """
    from embpy.tl.fusion._benchmark import MatrixFusion

    aligned, ids, y, group_values, coverage = _fusion_inputs(views, targets, groups, join)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).all() or y.ndim not in (1, 2):
        raise ValueError("Targets must be finite numeric values.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique.")
    configs = _fusion_configs(aligned, fusions, include_single, single_normalization, single_n_components, seeds[0])
    if not tune and any(grid for _, _, grid in configs.values()):
        raise ValueError("Fusion parameter grids require tune=True.")
    packed = {
        name: np.concatenate([v.matrix for v in parts.values()], axis=1) for name, (parts, _, _) in configs.items()
    }
    scores, entity_scores, split_rows = [], [], []
    for seed in seeds:
        for fold, (train, test) in enumerate(_splits(len(ids), groups=group_values, cv=cv, random_state=seed)):
            split_rows.extend(_split_records(ids, group_values, seed, fold, train, test))
            for name, (parts, operator, fusion_grid) in configs.items():
                started = time.perf_counter()
                adapter = MatrixFusion(operator, tuple(parts), tuple(v.n_dims for v in parts.values()))
                fitted, prediction, params = _fit_fold(
                    packed[name],
                    y,
                    train,
                    test,
                    model_name=model,
                    transformer=adapter,
                    groups=group_values,
                    tune=tune,
                    cv=3,
                    random_state=seed,
                    n_jobs=n_jobs,
                    param_grid=param_grid,
                    fusion_param_grid=fusion_grid,
                    memory=cache_dir,
                )
                elapsed = time.perf_counter() - started
                metrics = _compute_metrics(y[test], prediction)
                scores.append(
                    {
                        "configuration": name,
                        "seed": seed,
                        "fold": fold,
                        "model": model,
                        "n_train": len(train),
                        "n_test": len(test),
                        "fit_predict_seconds": elapsed,
                        "output_dim": fitted.named_steps["features"].fusion_.n_features_out_,
                        "projection_coefficients": fitted.named_steps["features"].fusion_.n_projection_coefficients_,
                        "best_params": params,
                        **metrics,
                    }
                )
                error = (y[test] - prediction) ** 2
                per_row_mse = error if error.ndim == 1 else error.mean(axis=1)
                for position, i in enumerate(test):
                    row = {
                        "configuration": name,
                        "seed": seed,
                        "fold": fold,
                        "entity_id": ids[i],
                        "mse": float(per_row_mse[position]),
                    }
                    if y.ndim == 2:
                        truth = y[i] - y[i].mean()
                        pred = prediction[position] - prediction[position].mean()
                        norm = np.linalg.norm(truth) * np.linalg.norm(pred)
                        row["pearson"] = float(truth @ pred / norm) if norm else 0.0
                    entity_scores.append(row)
                logger.info("Fusion %s seed=%s fold=%s pearson=%.4f", name, seed, fold, metrics["pearson"])
    return FusionBenchmarkResult(
        pd.DataFrame(scores),
        pd.DataFrame(entity_scores),
        pd.DataFrame(split_rows),
        coverage,
        {
            "task": "regression",
            "cv": cv,
            "seeds": list(seeds),
            "model": model,
            "tune": tune,
            "readout_param_grid": (_build_param_grid(model) if param_grid is None else param_grid) if tune else None,
            "selection_metric": "neg_mean_squared_error" if tune else None,
            "max_search_candidates": 20 if tune else None,
            "inner_cv": 3 if tune else None,
            "fit_scope": "training_fold",
            "grouped": groups is not None,
            "configurations": {
                name: {"views": list(parts), "parameters": op.get_params(), "param_grid": grid}
                for name, (parts, op, grid) in configs.items()
            },
        },
    )


def _tune_retrieval(operator, grid, parts, ids, labels, train, groups, seed, k):
    """Select fusion parameters by grouped inner precision@k, never outer queries."""
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.model_selection import ParameterGrid

    from embpy.tl.fusion._views import subset_view

    if not grid:
        return {}, None
    train_groups = None if groups is None else groups[train]
    available = len(train) if groups is None else len(np.unique(train_groups))
    if available < 2:
        raise ValueError("Retrieval tuning requires at least two training entities/groups.")
    inner = _splits(len(train), groups=train_groups, cv=min(3, available), random_state=seed)
    best_params, best_score = {}, -np.inf
    for params in ParameterGrid(grid):
        scores = []
        for reference, query in inner:
            reference, query = train[reference], train[query]
            reference_views = {name: subset_view(view, [ids[i] for i in reference]) for name, view in parts.items()}
            query_views = {name: subset_view(view, [ids[i] for i in query]) for name, view in parts.items()}
            fitted = clone(operator).set_params(**params).fit(reference_views)
            similarity = cosine_similarity(
                fitted.transform(query_views).matrix.astype(float),
                fitted.transform(reference_views).matrix.astype(float),
            )
            width = min(k, len(reference))
            order = np.argsort(-similarity, axis=1, kind="stable")[:, :width]
            relevant = labels[query, None] == labels[reference][None, :]
            scores.append(np.take_along_axis(relevant, order, axis=1).mean())
        score = float(np.mean(scores))
        if score > best_score:
            best_params, best_score = params, score
    return best_params, best_score


def benchmark_fusion_retrieval(
    views: Mapping[str, Any],
    labels: pd.Series,
    *,
    fusions: Mapping[str, Any] | None = None,
    groups: pd.Series | None = None,
    include_single: bool = True,
    join: str = "strict",
    cv: int = 5,
    seeds: Sequence[int] = (0,),
    k: int = 10,
    late_weights: Mapping[str, float] | None = None,
    tune: bool = False,
    single_n_components: int | None = None,
) -> FusionBenchmarkResult:
    """Compare cosine retrieval, score averaging and rank averaging on paired folds.

    Each test entity queries training entities only. Relevance means exactly
    equal categorical labels. Operators never use labels. With ``tune=True``,
    FusionConfig grids are selected by mean precision@k across three grouped
    inner folds; only outer-training labels participate in selection.
    Precision and hit rate include queries with no relevant training candidate;
    recall for those queries is undefined and reported as NaN. Coverage is
    reported explicitly. The effective k is min(k, number of training entities).
    Ties in the final retrieval order follow the reference entity order.

    This CPU reference benchmark materializes query-by-reference score matrices;
    use a bounded cohort. It never constructs a full registry-scale similarity
    matrix. The standalone fusion helpers can also operate on query batches.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    from embpy.tl.fusion import fuse_ranks, fuse_scores
    from embpy.tl.fusion._views import subset_view

    aligned, ids, values, group_values, coverage = _fusion_inputs(views, labels, groups, join)
    if values.ndim != 1:
        raise ValueError("Retrieval labels must be a Series of scalar categories.")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique.")
    configs = _fusion_configs(aligned, fusions, include_single, "l2", single_n_components, seeds[0])
    if not tune and any(grid for _, _, grid in configs.values()):
        raise ValueError("Fusion parameter grids require tune=True.")
    if set(configs) & {"score_mean", "rank_mean"}:
        raise ValueError("score_mean and rank_mean are reserved retrieval configuration names.")
    rows, entity_rows, split_rows = [], [], []
    for seed in seeds:
        for fold, (train, test) in enumerate(_splits(len(ids), groups=group_values, cv=cv, random_state=seed)):
            split_rows.extend(_split_records(ids, group_values, seed, fold, train, test))
            reference_ids, query_ids = [ids[i] for i in train], [ids[i] for i in test]
            relevant = values[test, None] == values[train][None, :]
            total_relevant = relevant.sum(axis=1)
            effective_k = min(k, len(train))

            def record(
                name,
                matrix,
                elapsed,
                *,
                fitted=None,
                best_params=None,
                inner_score=None,
                effective_k=effective_k,
                relevant=relevant,
                total_relevant=total_relevant,
                seed=seed,
                fold=fold,
                query_ids=query_ids,
                train=train,
                test=test,
            ):
                order = np.argsort(-matrix, axis=1, kind="stable")[:, :effective_k]
                hits = np.take_along_axis(relevant, order, axis=1).sum(axis=1)
                recall = np.divide(hits, total_relevant, out=np.full(len(test), np.nan), where=total_relevant > 0)
                rows.append(
                    {
                        "configuration": name,
                        "seed": seed,
                        "fold": fold,
                        "k": effective_k,
                        "n_train": len(train),
                        "n_test": len(test),
                        "n_queries_with_relevant": int((total_relevant > 0).sum()),
                        "precision_at_k": float(np.mean(hits / effective_k)),
                        "hit_rate_at_k": float(np.mean(hits > 0)),
                        "recall_at_k": float(np.mean(recall[total_relevant > 0]))
                        if (total_relevant > 0).any()
                        else np.nan,
                        "fit_predict_seconds": elapsed,
                        "output_dim": None if fitted is None else fitted.n_features_out_,
                        "projection_coefficients": 0 if fitted is None else fitted.n_projection_coefficients_,
                        "best_params": best_params or {},
                        "inner_precision_at_k": inner_score,
                    }
                )
                for j, key in enumerate(query_ids):
                    entity_rows.append(
                        {
                            "configuration": name,
                            "seed": seed,
                            "fold": fold,
                            "entity_id": key,
                            "precision_at_k": float(hits[j] / effective_k),
                            "recall_at_k": float(recall[j]),
                            "hit_at_k": int(hits[j] > 0),
                            "n_relevant": int(total_relevant[j]),
                        }
                    )

            for name, (parts, operator, grid) in configs.items():
                started = time.perf_counter()
                train_views = {key: subset_view(view, reference_ids) for key, view in parts.items()}
                test_views = {key: subset_view(view, query_ids) for key, view in parts.items()}
                params, inner_score = _tune_retrieval(operator, grid, parts, ids, values, train, group_values, seed, k)
                fitted = clone(operator).set_params(**params).fit(train_views)
                matrix = cosine_similarity(
                    fitted.transform(test_views).matrix.astype(np.float64),
                    fitted.transform(train_views).matrix.astype(np.float64),
                )
                record(
                    name,
                    matrix,
                    time.perf_counter() - started,
                    fitted=fitted,
                    best_params=params,
                    inner_score=inner_score,
                )
            started = time.perf_counter()
            scores = {
                key: pd.DataFrame(
                    cosine_similarity(view.matrix[test].astype(np.float64), view.matrix[train].astype(np.float64)),
                    index=query_ids,
                    columns=reference_ids,
                )
                for key, view in aligned.items()
            }
            base_elapsed = time.perf_counter() - started
            for name, combine in (("score_mean", fuse_scores), ("rank_mean", fuse_ranks)):
                started = time.perf_counter()
                matrix = combine(scores, weights=late_weights).to_numpy()
                record(name, matrix, base_elapsed + time.perf_counter() - started)
    return FusionBenchmarkResult(
        pd.DataFrame(rows),
        pd.DataFrame(entity_rows),
        pd.DataFrame(split_rows),
        coverage,
        {
            "task": "retrieval",
            "cv": cv,
            "seeds": list(seeds),
            "k": k,
            "grouped": groups is not None,
            "fit_scope": "training_fold",
            "relevance": "exact_label_match",
            "late_weights": late_weights,
            "tune": tune,
            "inner_cv": 3 if tune else None,
            "selection_metric": "precision_at_k" if tune else None,
            "configurations": {
                name: {"views": list(parts), "parameters": op.get_params(), "param_grid": grid}
                for name, (parts, op, grid) in configs.items()
            },
        },
    )
