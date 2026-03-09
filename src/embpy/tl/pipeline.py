"""scIB-style benchmarking pipeline.

Automates the full workflow of generating embeddings for a given use case
(protein, gene, small molecule, etc.), applying PCA, training simple
prediction models, and computing evaluation metrics across all embedding
models relevant to the use case.

The main entry point is :func:`run_pipeline`.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from anndata import AnnData

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Use-case → embedding model registry
# ---------------------------------------------------------------------------

_USE_CASE_REGISTRY: dict[str, dict[str, Any]] = {
    "protein": {
        "perturbation_type": "genetic",
        "models": [
            "esm2_8M",
            "esm2_35M",
            "esm2_150M",
            "esm2_650M",
            "esmc_300m",
            "esmc_600m",
            "prot_t5_xl",
        ],
    },
    "gene": {
        "perturbation_type": "genetic",
        "models": [
            # Enformer / Borzoi
            "enformer_human_rough",
            "borzoi_v0",
            "borzoi_v1",
            # Evo v1 / v1.5 (require pip install embpy[evo])
            "evo1_8k",
            "evo1_131k",
            "evo1.5_8k",
            "evo1_crispr",
            "evo1_transposon",
            # Evo2 (require pip install embpy[evo2])
            "evo2_7b",
            "evo2_40b",
            "evo2_7b_base",
            "evo2_1b_base",
        ],
    },
    "small_molecule": {
        "perturbation_type": "chemical",
        "models": [
            "chemberta2MTR",
            "chemberta2MLM",
            "molformer_base",
            # RDKit fingerprints
            "rdkit_fp",
            "morgan_fp",
            "morgan_count_fp",
            "maccs_fp",
            "atom_pair_fp",
            "atom_pair_count_fp",
            "torsion_fp",
            "torsion_count_fp",
            # GNN-based (optional deps)
            "minimol",
            "mhg_gnn",
            "mole",
        ],
    },
    "text": {
        "perturbation_type": "genetic",
        "models": [
            "minilm_l6_v2",
            "bert_base_uncased",
        ],
    },
}

_DEFAULT_PREDICTION_MODELS = ["ridge", "knn", "xgboost"]


def list_use_cases() -> list[str]:
    """Return the available use-case names."""
    return list(_USE_CASE_REGISTRY.keys())


def list_embedding_models(use_case: str) -> list[str]:
    """Return the default embedding models for a use case.

    Parameters
    ----------
    use_case
        One of ``"protein"``, ``"gene"``, ``"small_molecule"``, ``"text"``.
    """
    if use_case not in _USE_CASE_REGISTRY:
        raise ValueError(f"Unknown use_case '{use_case}'. Choose from: {list_use_cases()}")
    return list(_USE_CASE_REGISTRY[use_case]["models"])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    adata: AnnData,
    perturbation_column: str,
    use_case: Literal["protein", "gene", "small_molecule", "text"],
    target: str = "X",
    embedding_models: Sequence[str] | Literal["all"] | None = None,
    prediction_models: Sequence[str] | None = None,
    reduce_dim: int | None = 50,
    mode: Literal["quick", "rigorous"] = "quick",
    control_column: str | None = None,
    control_value: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    id_type: str = "gene_name",
    organism: str = "human",
    skip_failures: bool = True,
) -> pd.DataFrame:
    """Run an scIB-style benchmarking pipeline.

    For each embedding model relevant to the *use_case*, this function:

    1. Generates embeddings via :class:`~embpy.BioEmbedder`.
    2. Optionally applies PCA for dimensionality reduction.
    3. Trains each prediction model (ridge, KNN, XGBoost, ...).
    4. Computes evaluation metrics.
    5. Consolidates all results into a single comparison table.

    Parameters
    ----------
    adata
        AnnData with perturbation identifiers in ``.obs`` and a
        prediction target (either ``.X`` or a numeric ``.obs`` column).
    perturbation_column
        Column in ``.obs`` holding perturbation names, or ``"obs_names"``
        to use the AnnData index.
    use_case
        Determines which embedding models to run.  One of ``"protein"``,
        ``"gene"``, ``"small_molecule"``, ``"text"``.
    target
        ``"X"`` to predict the full expression matrix, or the name of a
        numeric column in ``.obs`` (e.g. ``"ic50"``).
    embedding_models
        Which embedding models to benchmark.  Three options:

        - ``None`` (default) — the user **must** provide a list; raises
          if omitted, so the choice is always explicit.
        - ``"all"`` — run every model registered for the use case.
          Call :func:`list_embedding_models` to see what's included.
        - A list of model name strings — run only those models.
    prediction_models
        Which regressors to train.  Defaults to
        ``["ridge", "knn", "xgboost"]``.
    reduce_dim
        Number of PCA components.  ``None`` skips dimensionality reduction.
    mode
        ``"quick"`` for a single train/test split, ``"rigorous"`` for
        cross-validated hyperparameter search.
    control_column
        Column in ``.obs`` identifying control observations.
    control_value
        Value in *control_column* marking control rows.
    test_size
        Fraction held out for testing (``"quick"`` mode only).
    random_state
        Random seed.
    id_type
        Identifier type for gene resolution.
    organism
        Organism string for gene resolution.
    skip_failures
        If ``True`` (default), log a warning and skip embedding models
        that fail (e.g. due to missing optional dependencies) instead
        of raising.

    Returns
    -------
    DataFrame with one row per (embedding_model, prediction_model)
    combination and columns for each metric.  Also stored in
    ``adata.uns["pipeline_results"]``.
    """
    from .benchmark import benchmark_embeddings

    if use_case not in _USE_CASE_REGISTRY:
        raise ValueError(f"Unknown use_case '{use_case}'. Choose from: {list_use_cases()}")

    registry = _USE_CASE_REGISTRY[use_case]
    perturbation_type: str = registry["perturbation_type"]

    if embedding_models is None:
        raise ValueError(
            "embedding_models is required. Pass a list of model names "
            '(e.g. ["esm2_650M", "esmc_300m"]) or "all" to run every '
            f"model for the '{use_case}' use case. "
            f"Available: {registry['models']}"
        )
    if embedding_models == "all":
        emb_models = list(registry["models"])
    else:
        emb_models = list(embedding_models)

    pred_models = list(prediction_models) if prediction_models is not None else list(_DEFAULT_PREDICTION_MODELS)

    logger.info(
        "Pipeline: use_case='%s', %d embedding models, %d prediction models, mode='%s'",
        use_case,
        len(emb_models),
        len(pred_models),
        mode,
    )

    all_results: list[pd.DataFrame] = []

    for emb_model in emb_models:
        logger.info("── Embedding model: %s", emb_model)
        t0 = time.time()

        try:
            result = benchmark_embeddings(
                adata,
                perturbation_column=perturbation_column,
                perturbation_type=perturbation_type,
                target=target,
                embedding_model=emb_model,
                control_column=control_column,
                control_value=control_value,
                models=pred_models,
                mode=mode,
                reduce_dim=reduce_dim,
                test_size=test_size,
                random_state=random_state,
                id_type=id_type,
                organism=organism,
            )
        except Exception:
            if skip_failures:
                logger.warning("   Failed for '%s', skipping.", emb_model, exc_info=True)
                continue
            raise

        elapsed = time.time() - t0
        result = result.reset_index()
        result.insert(0, "embedding", emb_model)
        result["time_s"] = round(elapsed, 1)
        all_results.append(result)
        logger.info("   Done in %.1fs", elapsed)

    if not all_results:
        logger.warning("Pipeline produced no results — all embedding models failed or list was empty.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    adata.uns["pipeline_results"] = combined
    logger.info(
        "Pipeline complete: %d rows (%d embeddings × %d models).",
        len(combined),
        len(all_results),
        len(pred_models),
    )
    return combined
