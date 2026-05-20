"""Clustering and neighbor graph tools.

Supports both CPU (scanpy) and GPU (rapids_singlecell) backends.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from anndata import AnnData


def _get_embedding(adata: AnnData, obsm_key: str) -> np.ndarray:
    """Extract an embedding matrix from *adata.obsm* with validation."""
    if obsm_key not in adata.obsm:
        raise KeyError(
            f"'{obsm_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    X = np.asarray(adata.obsm[obsm_key], dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(
            f"Expected 2-D array in adata.obsm['{obsm_key}'], got shape {X.shape}."
        )
    return X


def _require_scanpy():
    try:
        import scanpy as sc
        return sc
    except ImportError as e:
        raise ImportError(
            "scanpy is required for this function. "
            "Install with: pip install scanpy"
        ) from e


def _require_rapids():
    try:
        import rapids_singlecell as rsc
        return rsc
    except ImportError as e:
        raise ImportError(
            "rapids_singlecell is required for GPU backend. "
            "Install with: pip install rapids-singlecell"
        ) from e


def find_nearest_neighbors(
    adata: AnnData,
    obsm_key: str,
    n_neighbors: int = 15,
    metric: str = "cosine",
    backend: Literal["cpu", "gpu"] = "cpu",
) -> AnnData:
    """Compute nearest-neighbor graph on embeddings.

    Parameters
    ----------
    adata
        AnnData with embedding vectors in ``obsm[obsm_key]``.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    n_neighbors
        Number of nearest neighbors.
    metric
        Distance metric for neighbor computation.
    backend
        ``"cpu"`` uses scanpy, ``"gpu"`` uses rapids_singlecell.

    Returns
    -------
    AnnData with neighbor graph stored in ``obsp``.
    """
    if backend == "gpu":
        rsc = _require_rapids()
        rsc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=obsm_key, metric=metric)
    else:
        sc = _require_scanpy()
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=obsm_key, metric=metric)

    logging.info(
        "Computed %d-nearest-neighbor graph on '%s' (metric=%s, backend=%s).",
        n_neighbors, obsm_key, metric, backend,
    )
    return adata


def leiden(
    adata: AnnData,
    obsm_key: str,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    key_added: str = "leiden",
    backend: Literal["cpu", "gpu"] = "cpu",
) -> AnnData:
    """Run Leiden community detection on an embedding space.

    Parameters
    ----------
    adata
        AnnData with embedding vectors in ``obsm[obsm_key]``.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    resolution
        Resolution parameter for Leiden (higher = more clusters).
    n_neighbors
        Number of neighbors for the graph.
    key_added
        Column name in ``adata.obs`` for cluster labels.
    backend
        ``"cpu"`` uses scanpy, ``"gpu"`` uses rapids_singlecell.

    Returns
    -------
    AnnData with cluster labels in ``obs[key_added]``.
    """
    if backend == "gpu":
        rsc = _require_rapids()
        rsc.pp.neighbors(adata, use_rep=obsm_key, n_neighbors=n_neighbors)
        rsc.tl.leiden(adata, resolution=resolution, key_added=key_added)
    else:
        sc = _require_scanpy()
        sc.pp.neighbors(adata, use_rep=obsm_key, n_neighbors=n_neighbors)
        sc.tl.leiden(adata, resolution=resolution, key_added=key_added)

    n_clusters = adata.obs[key_added].nunique()
    logging.info(
        "Leiden clustering on '%s' (resolution=%.2f, backend=%s): "
        "%d clusters, stored in obs['%s'].",
        obsm_key, resolution, backend, n_clusters, key_added,
    )
    return adata


def cluster_embeddings(
    adata: AnnData,
    obsm_key: str,
    method: str = "leiden",
    resolution: float = 1.0,
    n_clusters: int | None = None,
    n_neighbors: int = 15,
    key_added: str = "cluster",
    backend: Literal["cpu", "gpu"] = "cpu",
) -> AnnData:
    """Cluster embeddings using Leiden, k-means, or spectral clustering.

    Parameters
    ----------
    adata
        AnnData with embedding vectors in ``obsm[obsm_key]``.
    obsm_key
        Key in ``.obsm`` holding the embedding matrix.
    method
        ``"leiden"``, ``"kmeans"``, or ``"spectral"``.
    resolution
        Resolution for Leiden.
    n_clusters
        Number of clusters (k-means / spectral). Ignored for Leiden.
    n_neighbors
        Number of neighbors for Leiden graph.
    key_added
        Column in ``adata.obs`` for cluster labels.
    backend
        ``"cpu"`` or ``"gpu"`` (only affects Leiden).

    Returns
    -------
    AnnData with cluster labels in ``obs[key_added]``.
    """
    X = _get_embedding(adata, obsm_key)

    if method == "leiden":
        return leiden(
            adata, obsm_key=obsm_key, resolution=resolution,
            n_neighbors=n_neighbors, key_added=key_added, backend=backend,
        )

    if method == "kmeans":
        from sklearn.cluster import KMeans
        k = n_clusters or 10
        labels = KMeans(n_clusters=k, random_state=0, n_init="auto").fit_predict(X)
        adata.obs[key_added] = [str(l) for l in labels]
        adata.obs[key_added] = adata.obs[key_added].astype("category")
        logging.info("KMeans clustering (k=%d) stored in obs['%s'].", k, key_added)
        return adata

    if method == "spectral":
        from sklearn.cluster import SpectralClustering
        k = n_clusters or 10
        labels = SpectralClustering(
            n_clusters=k, random_state=0, affinity="nearest_neighbors",
        ).fit_predict(X)
        adata.obs[key_added] = [str(l) for l in labels]
        adata.obs[key_added] = adata.obs[key_added].astype("category")
        logging.info("Spectral clustering (k=%d) stored in obs['%s'].", k, key_added)
        return adata

    raise ValueError(
        f"Unknown clustering method '{method}'. Choose from: leiden, kmeans, spectral."
    )


def cluster_annotation_enrichment(
    adata: AnnData,
    cluster_key: str,
    annotation_key: str,
    top_k: int = 5,
):
    """Top fold-enriched annotation terms per cluster.

    For each cluster (values of ``adata.obs[cluster_key]``) and each term
    appearing in ``adata.obs[annotation_key]`` (a string label *or* a list /
    tuple / set of multi-label terms), report the in-cluster frequency and
    the global frequency, ranked by fold-enrichment.

    A fast, dependency-free first pass for *"what does each cluster
    represent biologically?"*. For rigorous statistics on a 20 000-gene
    panel, plug the per-cluster gene lists into ``gseapy.enrichr`` or a
    Fisher exact test.

    Parameters
    ----------
    adata
        AnnData with cluster labels and annotation column in ``.obs``.
    cluster_key
        Column in ``adata.obs`` with cluster labels.
    annotation_key
        Column in ``adata.obs`` whose values are either:
          * a single string per row (single-label), or
          * a list / tuple / set of strings per row (multi-label,
            e.g. all GO terms a gene belongs to).
    top_k
        Number of top-enriched terms returned per cluster.

    Returns
    -------
    Long-format ``pandas.DataFrame`` with columns
    ``cluster, term, in_cluster, overall, in_freq, overall_freq, enrichment``.
    """
    import pandas as pd
    from collections import Counter

    if cluster_key not in adata.obs.columns:
        raise KeyError(f"'{cluster_key}' not in adata.obs.")
    if annotation_key not in adata.obs.columns:
        raise KeyError(f"'{annotation_key}' not in adata.obs.")

    def _terms_of(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set, frozenset)):
            return [str(t) for t in value if t is not None]
        if isinstance(value, float) and np.isnan(value):
            return []
        return [str(value)]

    rows = adata.obs[[cluster_key, annotation_key]].copy()
    overall_counts: Counter = Counter()
    for v in rows[annotation_key]:
        overall_counts.update(set(_terms_of(v)))
    n_total = len(rows)

    out_rows = []
    for c, sub in rows.groupby(cluster_key, observed=True):
        cluster_counts: Counter = Counter()
        for v in sub[annotation_key]:
            cluster_counts.update(set(_terms_of(v)))
        n_cluster = max(1, len(sub))
        for term, k_in in cluster_counts.items():
            in_freq = k_in / n_cluster
            ov_freq = overall_counts[term] / max(1, n_total)
            enrichment = in_freq / ov_freq if ov_freq > 0 else float("inf")
            out_rows.append({
                "cluster": c, "term": term,
                "in_cluster": int(k_in),
                "overall": int(overall_counts[term]),
                "in_freq": round(in_freq, 3),
                "overall_freq": round(ov_freq, 3),
                "enrichment": round(enrichment, 2),
            })
    df = pd.DataFrame(out_rows)
    if df.empty:
        return df
    return (
        df.sort_values(["cluster", "enrichment", "in_cluster"],
                       ascending=[True, False, False])
          .groupby("cluster", as_index=False)
          .head(top_k)
          .reset_index(drop=True)
    )
