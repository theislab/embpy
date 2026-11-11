"""Audit saved Stage-2 fold metrics, cohort coverage and source fingerprints on the cluster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from embpy.scripts.benchmark_fusion_lincs import _save_json, _sha


def verify(cohort_dir, output, source_root):
    """Reaggregate saved entity scores and compare explicit split/provenance records."""
    with np.load(cohort_dir / "cohort.npz", allow_pickle=False) as cohort:
        ids = cohort["entity_ids"].astype(str)
        groups = dict(zip(ids, cohort["groups"].astype(str), strict=True))
        target = pd.DataFrame(cohort["target"].astype(float), index=ids)
    reference = pd.read_csv(cohort_dir / "regression_splits.csv")
    checks = {}
    for task, expected_configs in (("regression", 11), ("retrieval", 13)):
        folds = pd.read_csv(output / f"{task}_folds.csv")
        entities = pd.read_csv(output / f"{task}_entities.csv")
        splits = pd.read_csv(output / f"{task}_splits.csv")
        execution = json.loads((output / f"execution_{task}.json").read_text())
        assert execution["slurm_job_id"], "Real benchmark must have Slurm provenance."
        assert execution["cohort_sha256"] == _sha(cohort_dir / "cohort.npz")
        assert execution["cohort_provenance_sha256"] == _sha(cohort_dir / "cohort_provenance.json")
        for name, digest in execution["source_sha256"].items():
            # macOS archive sidecars are metadata, never imported Python modules.
            # They need not exist when verifying against a clean Git checkout.
            if Path(name).name.startswith("._"):
                continue
            assert _sha(source_root / name) == digest, name
        assert_frame_equal(splits, reference)
        assert folds.configuration.nunique() == expected_configs
        assert len(folds) == expected_configs * 9
        assert not folds.duplicated(["configuration", "seed", "fold"]).any()
        assert not entities.duplicated(["configuration", "seed", "entity_id"]).any()
        for _, rows in entities.groupby(["configuration", "seed"]):
            assert set(rows.entity_id) == set(ids)
        for _, split in splits.groupby(["seed", "fold"]):
            assert len(split) == len(ids)
            assert split.entity_id.is_unique
            training = split.loc[split.partition == "train"]
            testing = split.loc[split.partition == "test"]
            assert not set(training.group) & set(testing.group)
            assert (split.entity_id.map(groups) == split.group).all()
        feature_rows = folds.loc[~folds.configuration.isin(["score_mean", "rank_mean", "morgan_l2"])]
        assert feature_rows.output_dim.eq(64).all()
        assert folds.loc[folds.configuration == "morgan_l2", "output_dim"].eq(1024).all()
        metrics = ["mse", "pearson"] if task == "regression" else ["precision_at_k", "recall_at_k"]
        assert np.isfinite(entities[["mse", "pearson"] if task == "regression" else ["precision_at_k"]]).all().all()
        if task == "retrieval":
            assert entities.precision_at_k.between(0, 1).all()
            assert entities.loc[entities.n_relevant == 0, "recall_at_k"].isna().all()
        aggregate = entities.groupby(["configuration", "seed", "fold"])[metrics].mean().sort_index()
        saved = folds.set_index(["configuration", "seed", "fold"])[metrics].sort_index()
        assert_allclose(aggregate, saved, rtol=1e-10, atol=1e-12, equal_nan=True)
        summary = pd.read_csv(output / f"{task}_summary.csv", index_col=0)
        for metric in metrics:
            expected = folds.groupby("configuration")[metric].agg(["mean", "std"])
            for statistic in ("mean", "std"):
                assert_allclose(summary.loc[expected.index, f"{metric}_{statistic}"], expected[statistic], atol=1e-12)
        checks[task] = {
            "slurm_job_id": execution["slurm_job_id"],
            "fold_rows": len(folds),
            "entity_rows": len(entities),
            "configurations": expected_configs,
            "reaggregated_metrics": metrics,
            "stage1_splits_identical": True,
            "source_hashes_match": True,
        }
    # Independent response-mean baseline: no fusion or regression model involved.
    baseline = []
    for (seed, fold), split in reference.groupby(["seed", "fold"]):
        train = target.loc[split.loc[split.partition == "train", "entity_id"]].to_numpy()
        test = target.loc[split.loc[split.partition == "test", "entity_id"]].to_numpy()
        prediction = train.mean(axis=0)
        truth = test - test.mean(axis=1, keepdims=True)
        centered = prediction - prediction.mean()
        correlations = (truth @ centered) / (np.linalg.norm(truth, axis=1) * np.linalg.norm(centered))
        baseline.append(
            {
                "seed": seed,
                "fold": fold,
                "mse": float(((test - prediction) ** 2).mean()),
                "pearson": float(correlations.mean()),
            }
        )
    pd.DataFrame(baseline).to_csv(output / "training_mean_baseline.csv", index=False)
    checks["structures"], checks["compound_groups"] = len(ids), len(set(groups.values()))
    checks["verification_scope"] = "Saved metric reaggregation and provenance; does not independently refit all models."
    checks["verifier_sha256"] = _sha(Path(__file__))
    _save_json(output / "verification.json", checks)
    print(json.dumps(checks, indent=2))


def plot_summary(output):
    """Render verified fold means and descriptive SD; no significance annotations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "morgan_l2": "Morgan L2 (1,024D)",
        "single:morgan1024_raw": "Morgan + PCA64",
        "single:molformer_cls": "MoLFormer CLS + PCA64",
        "single:chemberta2MTR_cls": "ChemBERTa MTR CLS + PCA64",
        "single:chemberta2MLM_mean": "ChemBERTa MLM mean + PCA64",
        "pair_pca": "Two views: joint PCA",
        "pair_cca": "Two views: CCA",
        "pair_procrustes": "Two views: Procrustes",
        "multi_pca": "Four views: joint PCA",
        "multi_gcca": "Four views: GCCA",
        "multi_procrustes": "Four views: Procrustes",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), sharey=True)
    for ax, task, metric, title in zip(
        axes,
        ("regression", "retrieval"),
        ("pearson", "precision_at_k"),
        ("Expression prediction", "MoA retrieval"),
        strict=True,
    ):
        summary = pd.read_csv(output / f"{task}_summary.csv", index_col=0)
        for position, name in enumerate(labels):
            row = summary.loc[name]
            color = "#14836e" if "procrustes" in name else "#8547ac" if name.endswith("cca") else "#3675b5"
            if name.startswith("single:") or name == "morgan_l2":
                color = "#555555"
            ax.errorbar(
                row[f"{metric}_mean"],
                position,
                xerr=row[f"{metric}_std"],
                fmt="o",
                color=color,
                capsize=3,
                markersize=6,
                elinewidth=1.4,
            )
        ax.set_title(title, fontsize=13, pad=14)
        ax.set_xlabel(
            "Pearson across genes (higher is better)" if task == "regression" else "Precision@10 (higher is better)"
        )
        ax.set_yticks(range(len(labels)), labels.values())
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.axhline(4.5, color="#cccccc", linewidth=0.6)
        ax.axhline(7.5, color="#cccccc", linewidth=0.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    fig.suptitle("Stage 2 fusion on LINCS", fontsize=17, y=0.98)
    fig.text(
        0.5,
        0.925,
        "Matched input views and 64D outputs; uncompressed Morgan shown as an extra control",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.025,
        "Points: means of 9 grouped outer folds. Whiskers: fold SD, not confidence intervals.\n"
        "1,183 structures in 929 compound groups. Late score/rank ensembles are reported separately.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))
    fig.savefig(output / "overview.png", dpi=180)
    fig.savefig(output / "overview.pdf")
    plt.close(fig)


def main():
    """Verify the fixed nine-fold Stage-2 LINCS run and write an audit record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--plot", action="store_true", help="Render verified summary means and fold SD.")
    args = parser.parse_args()
    verify(args.cohort_dir, args.output_dir, args.source_root)
    if args.plot:
        plot_summary(args.output_dir)


if __name__ == "__main__":
    main()
