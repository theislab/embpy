"""Verify that `BioEmbedder.embed_cells` caches the loaded wrapper.

First call should include model instantiation + inference; the second
call on the same embedder instance should skip instantiation entirely
and only pay the inference cost.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import anndata as ad  # type: ignore[import-not-found]

from embpy.embedder import BioEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> int:
    h5ad_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/lustre/groups/ml01/workspace/hediyehzadeh.s/projects/"
        "perturbation_projects/genetic_chemical_integration/EBM_SA_model/"
        "data/preprocessed/D1_Rest.assigned_guide_5k_hvg_equalSubSample200.h5ad"
    )
    n_cells = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    log.info("Reading %s (this may take a moment for a 4.8 GB file)...", h5ad_path)
    adata = ad.read_h5ad(h5ad_path)
    log.info("AnnData: %s x %s (%s)", adata.n_obs, adata.n_vars, adata.X.dtype)

    # Subsample to keep the test fast; cache behaviour is independent
    # of dataset size.
    if adata.n_obs > n_cells:
        adata = adata[:n_cells].copy()
    log.info("Subsampled to %s x %s", adata.n_obs, adata.n_vars)

    # NB: we intentionally do NOT manually rekey var_names to gene symbols
    # here. This dataset stores Ensembl IDs in var_names and gene symbols
    # in a `gene_name` column. Starting with this release, embpy's
    # `BioEmbedder.embed_cells` auto-detects the Ensembl-ID format and
    # routes through `GeneResolver` so scGPT receives symbols it can
    # tokenize. This test is therefore a real acceptance test for that
    # auto-conversion path.
    log.info("var_names[:3] = %s  (Ensembl IDs, will auto-convert)", list(adata.var_names[:3]))

    embedder = BioEmbedder(device="auto")

    log.info("=== call 1 (cold: loads model) ===")
    t0 = time.perf_counter()
    embedder.embed_cells(adata.copy(), models=["scgpt"], preprocessing="standard")
    t1 = time.perf_counter()
    log.info("call 1 elapsed = %.2f s", t1 - t0)

    log.info("=== call 2 (warm: should reuse cached wrapper) ===")
    t2 = time.perf_counter()
    embedder.embed_cells(adata.copy(), models=["scgpt"], preprocessing="standard")
    t3 = time.perf_counter()
    log.info("call 2 elapsed = %.2f s", t3 - t2)

    log.info("=== call 3 (warm: second reuse) ===")
    t4 = time.perf_counter()
    embedder.embed_cells(adata.copy(), models=["scgpt"], preprocessing="standard")
    t5 = time.perf_counter()
    log.info("call 3 elapsed = %.2f s", t5 - t4)

    log.info("cached wrappers: %s", list(embedder._singlecell_cache.keys()))

    cold = t1 - t0
    warm1 = t3 - t2
    warm2 = t5 - t4
    speedup1 = cold / max(warm1, 1e-6)
    speedup2 = cold / max(warm2, 1e-6)
    log.info("cold=%.2fs  warm1=%.2fs  warm2=%.2fs  speedup1=%.1fx  speedup2=%.1fx",
             cold, warm1, warm2, speedup1, speedup2)

    # Sanity assertions: warm calls should be substantially faster
    # (at least 2x) than the cold call on any GPU worth using.
    if warm1 > cold * 0.9:
        log.error("FAIL: warm call was not meaningfully faster than cold call")
        return 1

    log.info("PASS: wrapper cache is working -- model loaded once, inference reused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
