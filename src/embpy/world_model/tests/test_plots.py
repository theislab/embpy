"""Tests that the plotting helpers actually write PNG + SVG to disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from embpy.world_model.evaluation.plots import (
    plot_baseline_comparison,
    plot_deg_overlap_bar,
    plot_per_perturbation_metric,
    plot_pred_vs_real_scatter,
)


pytest.importorskip("matplotlib")


def test_plot_pred_vs_real_scatter_creates_files(tmp_path: Path):
    pytest.importorskip("anndata")
    import anndata as ad  # noqa: PLC0415

    rng = np.random.default_rng(0)
    n_cells = 20
    n_genes = 8
    real = ad.AnnData(
        X=rng.normal(size=(n_cells, n_genes)).astype(np.float32),
        obs={"perturbation": ["P0"] * 10 + ["P1"] * 10},
    )
    real.var_names = [f"g{i}" for i in range(n_genes)]
    pred = ad.AnnData(
        X=rng.normal(size=(n_cells, n_genes)).astype(np.float32),
        obs=real.obs.copy(),
    )
    pred.var_names = list(real.var_names)

    out = tmp_path / "scatter.png"
    plot_pred_vs_real_scatter(real, pred, out)
    assert out.exists()
    assert out.with_suffix(".svg").exists()


def test_plot_per_perturbation_metric_creates_files(tmp_path: Path):
    df = pd.DataFrame({"perturbation": ["P0", "P1", "P2"], "r2": [0.1, 0.2, 0.3]})
    out = tmp_path / "perpert_r2.png"
    plot_per_perturbation_metric(df, out, metric="r2")
    assert out.exists()
    assert out.with_suffix(".svg").exists()


def test_plot_per_perturbation_metric_skips_when_empty(tmp_path: Path):
    df = pd.DataFrame({"perturbation": [], "r2": []})
    out = tmp_path / "empty.png"
    plot_per_perturbation_metric(df, out, metric="r2")
    assert not out.exists()


def test_plot_deg_overlap_bar_creates_files(tmp_path: Path):
    df = pd.DataFrame({"perturbation": ["P0", "P1"], "deg_overlap@50": [0.1, 0.4]})
    out = tmp_path / "deg_overlap.png"
    plot_deg_overlap_bar(df, out)
    assert out.exists()
    assert out.with_suffix(".svg").exists()


def test_plot_baseline_comparison_creates_files(tmp_path: Path):
    df = pd.DataFrame({
        "name": ["world_model", "identity", "control_mean"],
        "mse": [0.1, 0.5, 0.4],
        "r2": [0.7, -0.1, -0.05],
    })
    out = tmp_path / "comparison.png"
    plot_baseline_comparison(df, out, metrics=["mse", "r2"])
    assert out.exists()
    assert out.with_suffix(".svg").exists()
