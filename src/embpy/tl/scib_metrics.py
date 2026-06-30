"""scIB metrics for single-cell foundation-model embeddings.

This module wraps `scib <https://scib.readthedocs.io/>`_ so the embeddings
produced by :meth:`embpy.BioEmbedder.embed` (``entity_type="cell"``) for the
single-cell foundation models (scGPT, Geneformer, UCE, scFoundation, ...) can
be **scored and compared** against a known cell-type label, and optionally
against a batch covariate.

Typical workflow
----------------
1. Embed the same AnnData with several single-cell models, each landing in its
   own ``.obsm`` slot::

       for model in ["scgpt", "geneformer", "uce", "pca"]:
           adata = embedder.embed(adata, entity_type="cell", model=model,
                                  output="anndata", key=f"X_{model}")

2. Score and compare them with scIB::

       from embpy import tl
       report = tl.compute_scib_metrics(
           adata,
           embedding_keys=["X_scgpt", "X_geneformer", "X_uce", "X_pca"],
           label_key="cell_type",
           batch_key="batch",   # optional
       )

``report`` is a tidy :class:`pandas.DataFrame` (one row per embedding) with the
individual scIB metrics plus the aggregate *bio-conservation*,
*batch-correction* and *total* scores using scIB's standard 0.6 / 0.4 weighting.

``scib`` is an **optional** dependency (it pulls scanpy and friends). Install it
with ``pip install embpy[scib]`` (or ``pip install scib``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from embpy.errors import DependencyError

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)

# Bio-conservation metrics (need only a cell-type label) and batch-correction
# metrics (need a batch covariate). The aggregate scores follow scIB's standard
# weighting: total = 0.6 * bio-conservation + 0.4 * batch-correction.
_BIO_METRICS = ("nmi", "ari", "asw_label", "isolated_label_asw", "clisi")
_BATCH_METRICS = ("asw_batch", "graph_conn", "ilisi", "kbet")


def _load_scib():
    """Import scib lazily, raising a friendly error if it is missing."""
    try:
        import scib  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - exercised via DependencyError
        raise DependencyError("scib", feature="scIB embedding metrics") from e
    return scib


def _load_scanpy():
    try:
        import scanpy as sc
    except ImportError as e:  # pragma: no cover
        raise DependencyError("scanpy", feature="scIB embedding metrics") from e
    return sc


def compute_scib_metrics(
    adata: AnnData,
    embedding_keys: str | list[str],
    label_key: str,
    batch_key: str | None = None,
    *,
    cluster_resolution_range: tuple[float, float, float] = (0.1, 2.0, 0.1),
    n_neighbors: int = 15,
    verbose: bool = False,
) -> pd.DataFrame:
    """Score one or more cell embeddings with scIB and return a comparison table.

    Each embedding in ``embedding_keys`` is evaluated independently: a kNN graph
    is built on that ``.obsm`` representation, Leiden clustering is optimised
    against the label (for NMI/ARI), and the scIB metric battery is computed.

    Parameters
    ----------
    adata
        AnnData holding the cells. Every key in ``embedding_keys`` must be a
        matrix in ``adata.obsm`` with one row per cell.
    embedding_keys
        One ``.obsm`` key, or a list of keys, to score and compare. These are
        typically the ``X_<model>`` slots written by
        :meth:`embpy.BioEmbedder.embed`.
    label_key
        Column in ``adata.obs`` with the ground-truth cell-type labels. Drives
        every bio-conservation metric.
    batch_key
        Optional column in ``adata.obs`` with a batch covariate. When given, the
        batch-correction metrics (ASW-batch, graph connectivity, iLISI, kBET)
        and the aggregate *total* score are added.
    cluster_resolution_range
        ``(start, stop, step)`` passed to scIB's optimal-resolution search used
        for NMI/ARI clustering.
    n_neighbors
        Neighbours for the per-embedding kNN graph.
    verbose
        Forward scanpy/scib chatter to stdout.

    Returns
    -------
    pandas.DataFrame
        One row per embedding key, columns for each computed metric plus
        ``bio_conservation``, ``batch_correction`` (NaN when ``batch_key`` is
        ``None``) and ``total``. Higher is better for every column. Sorted by
        ``total`` descending.

    Raises
    ------
    embpy.errors.DependencyError
        If ``scib`` (or ``scanpy``) is not installed.
    KeyError
        If ``label_key``/``batch_key`` are absent from ``adata.obs`` or an
        embedding key is absent from ``adata.obsm``.
    """
    # Validate user input before importing the (optional, heavy) scib stack so
    # obvious mistakes fail fast with a clear message.
    if isinstance(embedding_keys, str):
        embedding_keys = [embedding_keys]
    if label_key not in adata.obs:
        raise KeyError(f"label_key {label_key!r} not in adata.obs (have: {list(adata.obs.columns)}).")
    if batch_key is not None and batch_key not in adata.obs:
        raise KeyError(f"batch_key {batch_key!r} not in adata.obs (have: {list(adata.obs.columns)}).")
    missing = [k for k in embedding_keys if k not in adata.obsm]
    if missing:
        raise KeyError(f"embedding key(s) {missing} not in adata.obsm (have: {list(adata.obsm)}).")

    scib = _load_scib()
    sc = _load_scanpy()

    rows: dict[str, dict[str, float]] = {}
    for key in embedding_keys:
        logger.info("Scoring embedding %r with scIB ...", key)
        rows[key] = _score_one_embedding(
            adata,
            embed_key=key,
            label_key=label_key,
            batch_key=batch_key,
            scib=scib,
            sc=sc,
            cluster_resolution_range=cluster_resolution_range,
            n_neighbors=n_neighbors,
            verbose=verbose,
        )

    report = pd.DataFrame.from_dict(rows, orient="index")
    report = _add_aggregate_scores(report, has_batch=batch_key is not None)
    return report.sort_values("total", ascending=False)


def _score_one_embedding(
    adata: AnnData,
    *,
    embed_key: str,
    label_key: str,
    batch_key: str | None,
    scib,
    sc,
    cluster_resolution_range: tuple[float, float, float],
    n_neighbors: int,
    verbose: bool,
) -> dict[str, float]:
    """Compute the scIB metric battery for a single ``.obsm`` embedding."""
    # Work on a shallow copy so the per-embedding neighbours graph / clustering
    # does not leak between embeddings or mutate the caller's AnnData.
    ad = adata.copy()
    sc.pp.neighbors(ad, use_rep=embed_key, n_neighbors=n_neighbors)

    out: dict[str, float] = {}

    # --- Bio conservation (label only) -------------------------------------
    # NMI/ARI need a clustering optimised against the label.
    scib.metrics.cluster_optimal_resolution(
        ad,
        label_key=label_key,
        cluster_key="scib_cluster",
        resolutions=_resolutions(cluster_resolution_range),
        verbose=verbose,
    )
    out["nmi"] = float(scib.metrics.nmi(ad, "scib_cluster", label_key))
    out["ari"] = float(scib.metrics.ari(ad, "scib_cluster", label_key))
    out["asw_label"] = float(scib.metrics.silhouette(ad, label_key, embed_key))
    out["isolated_label_asw"] = float(
        scib.metrics.isolated_labels(ad, label_key, batch_key, embed_key, cluster=False, verbose=verbose)
    )
    out["clisi"] = float(scib.metrics.clisi_graph(ad, label_key, type_="embed", use_rep=embed_key))

    # --- Batch correction (needs a batch covariate) ------------------------
    if batch_key is not None:
        out["asw_batch"] = float(
            scib.metrics.silhouette_batch(ad, batch_key, label_key, embed_key, verbose=verbose)
        )
        out["graph_conn"] = float(scib.metrics.graph_connectivity(ad, label_key))
        out["ilisi"] = float(scib.metrics.ilisi_graph(ad, batch_key, type_="embed", use_rep=embed_key))
        try:
            out["kbet"] = float(scib.metrics.kBET(ad, batch_key, label_key, type_="embed", embed=embed_key))
        except Exception as e:  # kBET is the most fragile metric; never fail the whole report.
            logger.warning("kBET failed for %r (%s); reporting NaN.", embed_key, e)
            out["kbet"] = float("nan")

    return out


def _resolutions(rng: tuple[float, float, float]) -> list[float]:
    start, stop, step = rng
    return [round(r, 4) for r in np.arange(start, stop + step / 2, step)]


def _add_aggregate_scores(report: pd.DataFrame, *, has_batch: bool) -> pd.DataFrame:
    """Add bio_conservation / batch_correction / total columns (scIB weighting)."""
    bio_cols = [c for c in _BIO_METRICS if c in report.columns]
    report["bio_conservation"] = report[bio_cols].mean(axis=1, skipna=True)

    if has_batch:
        batch_cols = [c for c in _BATCH_METRICS if c in report.columns]
        report["batch_correction"] = report[batch_cols].mean(axis=1, skipna=True)
        report["total"] = 0.6 * report["bio_conservation"] + 0.4 * report["batch_correction"]
    else:
        report["batch_correction"] = float("nan")
        report["total"] = report["bio_conservation"]
    return report
