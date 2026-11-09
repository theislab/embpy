"""Cluster benchmark of CCA, MAXVAR GCCA and Procrustes on the audited LINCS cohort.

Reuse the Stage-1 NPZ and provenance rather than rebuilding or downloading data.
All candidates in a comparison have identical views, entities and output width.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from embpy.io.result import EmbeddingProvenance, EmbeddingResult
from embpy.scripts.benchmark_fusion_lincs import _save_json, _sha, _write_result
from embpy.tl.benchmark import FusionBenchmarkResult, benchmark_fusion, benchmark_fusion_retrieval
from embpy.tl.fusion import CCAFusion, ConcatFusion, FusionConfig, GCCAFusion, PCAFusion, ProcrustesFusion

VIEW_NAMES = ("morgan1024_raw", "molformer_cls", "chemberta2MTR_cls", "chemberta2MLM_mean")


def configurations(components=64, pre_components=128):
    """Return prespecified pair/four-view comparisons; do not select on test scores."""
    pair, multi = VIEW_NAMES[:2], VIEW_NAMES
    grid = {"regularization": [0.01, 0.1, 1.0]}
    return {
        "morgan_l2": FusionConfig(ConcatFusion(normalization="l2"), (VIEW_NAMES[0],)),
        "pair_pca": FusionConfig(PCAFusion(components, normalization="l2"), pair),
        "pair_cca": FusionConfig(CCAFusion(components, pre_components=pre_components), pair, grid),
        "pair_procrustes": FusionConfig(ProcrustesFusion(components, reference=VIEW_NAMES[0]), pair),
        "multi_pca": FusionConfig(PCAFusion(components, normalization="l2"), multi),
        "multi_gcca": FusionConfig(GCCAFusion(components, pre_components=pre_components), multi, grid),
        "multi_procrustes": FusionConfig(ProcrustesFusion(components, reference=VIEW_NAMES[0]), multi),
    }


def run(cohort_dir, output, *, seeds, cv, components, pre_components, task):
    """Run and checkpoint each seed on a prepared cohort."""
    with np.load(cohort_dir / "cohort.npz", allow_pickle=False) as stored:
        ids = tuple(stored["entity_ids"].tolist())
        views = {
            key: EmbeddingResult(stored[key], ids, "molecule", "canonical_smiles", EmbeddingProvenance(key))
            for key in VIEW_NAMES
        }
        target = pd.DataFrame(stored["target"], index=ids)
        labels = pd.Series(stored["labels"], index=ids)
        groups = pd.Series(stored["groups"], index=ids)
    operators = configurations(components, pre_components)
    all_results = []
    for seed in seeds:
        seed_output = output / f"seed_{seed}"
        seed_output.mkdir(parents=True, exist_ok=True)
        common = {
            "fusions": operators,
            "groups": groups,
            "cv": cv,
            "seeds": (seed,),
            "tune": True,
            "single_n_components": components,
        }
        if task == "regression":
            # Short-lived cluster cache avoids repeating fusion fits for every Ridge alpha.
            with tempfile.TemporaryDirectory(prefix="embpy-fusion-", dir=os.getenv("TMPDIR")) as cache:
                result = benchmark_fusion(
                    views,
                    target,
                    **common,
                    param_grid={"alpha": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]},
                    cache_dir=cache,
                    n_jobs=1,
                )
        else:
            result = benchmark_fusion_retrieval(views, labels, **common, k=10)
        _write_result(result, seed_output, task)
        all_results.append(result)
    combined = FusionBenchmarkResult(
        pd.concat([r.scores for r in all_results], ignore_index=True),
        pd.concat([r.entity_scores for r in all_results], ignore_index=True),
        pd.concat([r.splits for r in all_results], ignore_index=True),
        all_results[0].coverage,
        {**all_results[0].config, "seeds": list(seeds)},
    )
    _write_result(combined, output, task)


def main():
    """Parse cluster benchmark arguments and record source/data provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=["regression", "retrieval"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--pre-components", type=int, default=128)
    args = parser.parse_args()
    provenance = args.cohort_dir / "cohort_provenance.json"
    if not provenance.exists():
        parser.error("A prepared cohort and its provenance file are required.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()
    run(
        args.cohort_dir,
        args.output_dir,
        seeds=args.seeds,
        cv=args.cv,
        components=args.components,
        pre_components=args.pre_components,
        task=args.task,
    )
    import sklearn

    source_root = Path(__file__).parents[1]
    files = [*sorted((source_root / "tl" / "fusion").glob("*.py")), source_root / "tl" / "benchmark.py", Path(__file__)]
    _save_json(
        args.output_dir / f"execution_{args.task}.json",
        {
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "host": platform.node(),
            "elapsed_seconds": time.time() - started,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "cohort_path": str(args.cohort_dir / "cohort.npz"),
            "cohort_sha256": _sha(args.cohort_dir / "cohort.npz"),
            "cohort_provenance_sha256": _sha(provenance),
            "source_sha256": {str(path.relative_to(source_root)): _sha(path) for path in files},
            "components": args.components,
            "pre_components": args.pre_components,
            "seeds": args.seeds,
            "cv": args.cv,
        },
    )


if __name__ == "__main__":
    main()
