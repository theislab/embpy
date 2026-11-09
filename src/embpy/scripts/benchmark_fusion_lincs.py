"""Run the Stage-1 LINCS fusion validation on a compute node.

This is a new structure-level benchmark, not a reproduction of the previous
signature-level LINCS notebook. Neural features come from named CSV feature
columns. Classical descriptors are regenerated without population scaling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from embpy.io.result import EmbeddingProvenance, EmbeddingResult

logger = logging.getLogger(__name__)


def _sha(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def _save_json(path, value):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    path.write_text(json.dumps(value, indent=2, default=convert) + "\n")


def _csv_view(path, wanted, *, conflicting="raise"):
    if conflicting not in {"raise", "drop"}:
        raise ValueError("conflicting must be 'raise' or 'drop'.")
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    id_column = "smiles" if "smiles" in columns else columns[0]
    features = [c for c in columns if c.startswith("e_") and c[2:].isdigit()]
    if not features:
        raise ValueError(f"No explicitly named embedding features found in {path}")
    features.sort(key=lambda key: int(key[2:]))
    selected = []
    for block in pd.read_csv(
        path, usecols=[id_column, *features], chunksize=4096, dtype=dict.fromkeys(features, np.float32)
    ):
        block = block.loc[block[id_column].isin(wanted)].set_index(id_column)
        if len(block):
            selected.append(block[features])
    if not selected:
        raise ValueError(f"No cohort entities found in {path}")
    table = pd.concat(selected)
    conflicts = {}
    if not table.index.is_unique:
        # Identical duplicate records are safe to collapse; conflicting ones are not.
        for key, duplicate in table[table.index.duplicated(keep=False)].groupby(level=0):
            if not np.allclose(duplicate.values, duplicate.values[0], rtol=0, atol=0):
                if conflicting == "raise":
                    raise ValueError(f"Conflicting duplicate embedding for {key!r} in {path}")
                conflicts[key] = float(np.nanmax(np.abs(duplicate.values - duplicate.values[0])))
        table = table.loc[~table.index.isin(conflicts)]
        table = table.loc[~table.index.duplicated()]
    finite = np.isfinite(table.to_numpy()).all(axis=1)
    return table.loc[finite], {
        "path": str(path),
        "sha256": _sha(path),
        "dimensions": len(features),
        "excluded_metadata_columns": [c for c in columns if c not in features and c != id_column],
        "nonfinite_rows": int((~finite).sum()),
        "matched_rows": int(finite.sum()),
        "conflicting_duplicate_policy": conflicting,
        "conflicting_duplicate_max_abs_differences": conflicts,
    }


def _compound_groups(obs):
    parent = {key: key for key in obs.canonical_smiles.unique()}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for _, frame in obs.groupby("pert_iname", observed=True):
        structures = frame.canonical_smiles.unique()
        anchor = find(structures[0])
        for key in structures[1:]:
            parent[find(key)] = anchor
    return {key: find(key) for key in parent}


def prepare(input_h5ad, embedding_root, output):
    """Build a modest, explicitly audited cohort using only cluster-side files."""
    import h5py
    import rdkit
    from anndata.io import read_elem
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdFingerprintGenerator

    with h5py.File(input_h5ad) as handle:
        obs = read_elem(handle["obs"])
        required = ["canonical_smiles", "pert_iname", "moa", "target"]
        for col in required:
            obs[col] = obs[col].astype(str)
        valid = np.ones(len(obs), dtype=bool)
        for col in required:
            valid &= ~obs[col].str.lower().isin({"nan", "unknown", "none", "-666", ""}).to_numpy()
        selected = obs.loc[valid, required].copy()
        ids = sorted(selected.canonical_smiles.unique())
        lookup = {key: i for i, key in enumerate(ids)}
        codes = selected.canonical_smiles.map(lookup).to_numpy()
        source_rows = np.flatnonzero(valid)
        counts = np.bincount(codes, minlength=len(ids))
        gene_names = read_elem(handle["var"]).index.astype(str).tolist()
        sums = np.zeros((len(ids), len(gene_names)), dtype=np.float64)
        X = handle["X"]
        if not isinstance(X, h5py.Dataset):
            raise ValueError("This LINCS preparation expects the existing dense Level-5 matrix.")
        for start in range(0, len(obs), 4096):
            left, right = np.searchsorted(source_rows, [start, start + 4096])
            if right > left:
                block = X[start : start + 4096][source_rows[left:right] - start]
                np.add.at(sums, codes[left:right], block)
    targets = pd.DataFrame(sums / counts[:, None], index=ids, columns=gene_names)
    groups = _compound_groups(selected)
    label_sets = selected.groupby("canonical_smiles", observed=True).moa.agg(lambda x: sorted(set(x)))
    labels = pd.Series({key: values[0] if len(values) == 1 else None for key, values in label_sets.items()})
    tables, sources = {}, {}
    for family, folder in [
        ("chemberta2MLM", "chemberta2MLM"),
        ("chemberta2MTR", "chemberta2MTR"),
        ("molformer", "molformer_base"),
    ]:
        for pooling in ("cls", "max", "mean"):
            name = f"{family}_{pooling}"
            path = embedding_root / "drug_embeddings" / folder / f"embeddings_{pooling}.csv"
            logger.info("Reading named feature columns: %s", path)
            tables[name], sources[name] = _csv_view(path, set(ids), conflicting="drop")

    # Legacy Morgan and RDKit CSVs were globally standardized and their fit
    # statistics are unavailable. Recompute these inexpensive features instead.
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fingerprints, descriptors, descriptor_ids, excluded = [], [], [], []
    descriptor_names = None
    for key in ids:
        mol = Chem.MolFromSmiles(key)
        if mol is None:
            excluded.append({"entity_id": key, "reason": "invalid_smiles"})
            continue
        calculated = Descriptors.CalcMolDescriptors(mol, missingVal=np.nan)
        if descriptor_names is None:
            descriptor_names = sorted(calculated)
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray([calculated[k] for k in descriptor_names], dtype=np.float32)
        if not np.isfinite(values).all():
            excluded.append({"entity_id": key, "reason": "nonfinite_raw_descriptor"})
            continue
        fingerprints.append(generator.GetFingerprintAsNumPy(mol))
        descriptors.append(values)
        descriptor_ids.append(key)
    tables["morgan1024_raw"] = pd.DataFrame(fingerprints, index=descriptor_ids)
    tables["rdkit_descriptors_raw"] = pd.DataFrame(descriptors, index=descriptor_ids, columns=descriptor_names)
    sources["morgan1024_raw"] = {
        "generated_from": "canonical_smiles",
        "radius": 2,
        "n_bits": 1024,
        "rdkit_version": rdkit.__version__,
    }
    sources["rdkit_descriptors_raw"] = {
        "generated_from": "canonical_smiles",
        "descriptor_names": descriptor_names,
        "rdkit_version": rdkit.__version__,
    }
    common = set(ids)
    for table in tables.values():
        common &= set(table.index)
    common &= set(labels.dropna().index)
    common = sorted(common)
    if len(common) < 100:
        raise ValueError("Too few complete, unambiguously annotated structures for the benchmark.")
    arrays = {name: table.loc[common].to_numpy(dtype=np.float32) for name, table in tables.items()}
    arrays.update(
        target=targets.loc[common].to_numpy(dtype=np.float32),
        entity_ids=np.asarray(common),
        labels=labels.loc[common].to_numpy(dtype=str),
        groups=np.asarray([groups[key] for key in common]),
    )
    np.savez_compressed(output / "cohort.npz", **arrays)
    provenance = {
        "input_h5ad": str(input_h5ad),
        "input_sha256": _sha(input_h5ad),
        "loaded_signatures": len(obs),
        "annotated_signatures": len(selected),
        "annotated_structures": len(ids),
        "complete_structures": len(common),
        "compound_groups": len(set(arrays["groups"])),
        "genes": len(gene_names),
        "retained_signatures": int(selected.canonical_smiles.isin(common).sum()),
        "excluded_structure_ids": sorted(set(ids) - set(common)),
        "descriptor_exclusions": excluded,
        "multiple_moa_structures": int(labels.isna().sum()),
        "sources": sources,
        "aggregation": "Mean Level-5 signature per canonical SMILES across the observed cell lines, doses and times. Equal weight per signature within structure, equal weight per structure at evaluation.",
        "split_unit": "Connected components of SMILES linked by compound name; no component spans train/test or inner folds.",
        "corrections": [
            "Use e_* CSV columns only; row_index is not a ChemBERTa feature.",
            "Regenerate Morgan and RDKit features; do not reuse globally standardized legacy CSVs.",
        ],
        "limitations": [
            "This predicts pooled structure-level profiles, not context-specific responses.",
            "Classical features and splitting differ from the old LINCS notebook; results are not directly comparable.",
            "Embedding pretraining overlap with downstream entities is not audited here.",
        ],
    }
    _save_json(output / "cohort_provenance.json", provenance)
    logger.info("Prepared %d structures in %d compound groups", len(common), provenance["compound_groups"])


def _write_result(result, output, prefix):
    result.scores.to_csv(output / f"{prefix}_folds.csv", index=False)
    result.entity_scores.to_csv(output / f"{prefix}_entities.csv", index=False)
    result.splits.to_csv(output / f"{prefix}_splits.csv", index=False)
    _save_json(output / f"{prefix}_config.json", {"coverage": result.coverage, **result.config})
    numeric = result.scores.select_dtypes(include="number").columns.difference(["seed", "fold"])
    summary = result.scores.groupby("configuration")[numeric].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary.to_csv(output / f"{prefix}_summary.csv")


def run(output, *, seeds, cv, components, task):
    """Run paired feature and retrieval comparisons on the prepared cohort."""
    from embpy.tl.benchmark import benchmark_fusion, benchmark_fusion_retrieval
    from embpy.tl.fusion import ConcatFusion, PCAFusion

    with np.load(output / "cohort.npz", allow_pickle=False) as stored:
        ids = tuple(stored["entity_ids"].tolist())
        views = {
            key: EmbeddingResult(stored[key], ids, "molecule", "canonical_smiles", EmbeddingProvenance(key))
            for key in stored.files
            if key not in {"target", "entity_ids", "labels", "groups"}
        }
        target = pd.DataFrame(stored["target"], index=ids)
        labels = pd.Series(stored["labels"], index=ids)
        groups = pd.Series(stored["groups"], index=ids)
    family_weights = {key: 1 / 3 if key.startswith(("chemberta", "molformer")) else 1.0 for key in views}
    operators = {
        "concat_raw": ConcatFusion(),
        "concat_zscore": ConcatFusion(normalization="zscore"),
        "concat_l2": ConcatFusion(normalization="l2"),
        "concat_zscore_balanced": ConcatFusion(normalization="zscore", balance_blocks=True),
        "concat_l2_family_weights": ConcatFusion(normalization="l2", weights=family_weights),
        "pca_l2": PCAFusion(components, normalization="l2"),
        "pca_zscore_balanced": PCAFusion(components, normalization="zscore", balance_blocks=True),
    }
    if task in {"both", "regression"}:
        result = benchmark_fusion(
            views,
            target,
            fusions=operators,
            groups=groups,
            cv=cv,
            seeds=seeds,
            tune=True,
            param_grid={"alpha": [0.1, 1.0, 10.0, 100.0]},
            n_jobs=1,
        )
        _write_result(result, output, "regression")
        # Matched-width single-view controls are separate, named experiments.
        fixed = benchmark_fusion(
            views,
            target,
            fusions={},
            groups=groups,
            cv=cv,
            seeds=seeds,
            single_n_components=components,
            tune=True,
            param_grid={"alpha": [0.1, 1.0, 10.0, 100.0]},
            n_jobs=1,
        )
        _write_result(fixed, output, "regression_single_pca")
        from embpy.tl.metrics import compute_metrics

        baseline_rows = []
        for (seed, fold), split in result.splits.groupby(["seed", "fold"]):
            training = target.loc[split.loc[split.partition == "train", "entity_id"]].to_numpy()
            testing = target.loc[split.loc[split.partition == "test", "entity_id"]].to_numpy()
            prediction = np.broadcast_to(training.mean(axis=0), testing.shape)
            baseline_rows.append(
                {"seed": seed, "fold": fold, "configuration": "training_mean", **compute_metrics(testing, prediction)}
            )
        pd.DataFrame(baseline_rows).to_csv(output / "regression_mean_baseline.csv", index=False)
    if task in {"both", "retrieval"}:
        result = benchmark_fusion_retrieval(views, labels, fusions=operators, groups=groups, cv=cv, seeds=seeds, k=10)
        _write_result(result, output, "retrieval")


def main():
    """Command-line entry point for cluster execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-cohort", action="store_true")
    parser.add_argument("--task", choices=["both", "regression", "retrieval"], default="both")
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--components", type=int, default=64)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not args.reuse_cohort:
        prepare(args.input_h5ad, args.embedding_root, args.output_dir)
    elif not (args.output_dir / "cohort_provenance.json").exists():
        raise ValueError("--reuse-cohort requires a prepared cohort and its provenance file.")
    if not args.prepare_only:
        run(args.output_dir, seeds=tuple(args.seeds), cv=args.cv, components=args.components, task=args.task)
    import sklearn

    _save_json(
        args.output_dir / f"execution_{'prepare' if args.prepare_only else args.task}.json",
        {
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "host": platform.node(),
            "elapsed_seconds": time.time() - started,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "seeds": args.seeds,
            "cv": args.cv,
            "components": args.components,
            "code_sha256": _sha(Path(__file__)),
        },
    )


if __name__ == "__main__":
    main()
