"""benchmark_perturbations: the data each cell-eval score is computed on, and where reports go.

Every test writes a reference and a prediction .h5ad with gene expression in ``.X`` and
embeddings in ``.obsm``, and checks the scores against values recomputed here with numpy.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from embpy.mcp import benchmark_perturbations

pytest.importorskip("cell_eval", reason="cell-eval is not installed")

PERTS = ["ctrl", "A", "B", "C", "D"]
N_PER = 10
N_GENES = 12

# Embedding centres per perturbation. SWAPPED predicts B's centre for A and A's for B.
CENTERS = dict(zip(PERTS, np.random.default_rng(7).normal(size=(len(PERTS), 6))))
SWAPPED = {**CENTERS, "A": CENTERS["B"], "B": CENTERS["A"]}
NARROW = dict(zip(PERTS, np.random.default_rng(8).normal(size=(len(PERTS), 4))))


def _adata(prefix: str, seed: int, embeddings: dict[str, dict[str, np.ndarray]]) -> AnnData:
    """Log-normalized expression in which each perturbation raises its own three genes."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, _ in enumerate(PERTS):
        mean = np.full(N_GENES, 1.0)
        if i:
            mean[(i - 1) * 3 : i * 3] += 1.5
        rows.append(mean + rng.uniform(0.0, 0.1, size=(N_PER, N_GENES)))
    adata = AnnData(X=np.vstack(rows).astype(np.float32))
    adata.obs_names = [f"{prefix}_{i}" for i in range(adata.n_obs)]
    adata.var_names = [f"g{j}" for j in range(N_GENES)]
    adata.obs["perturbation"] = np.repeat(PERTS, N_PER)
    for key, centers in embeddings.items():
        adata.obsm[key] = np.vstack(
            [centers[p] + rng.normal(scale=0.05, size=(N_PER, len(centers[p]))) for p in PERTS]
        ).astype(np.float32)
    return adata


@pytest.fixture
def write_pair(tmp_path):
    """Write a reference and a prediction file and return their paths (reference first).

    The expression is the same on every call: the prediction gets each perturbation's
    effect right, so expression-based discrimination scores are 1 everywhere. Only the
    embeddings, given as {key: centres per perturbation}, change between calls.
    """

    def write(name: str = "pair", real_emb=None, pred_emb=None) -> tuple[str, str]:
        folder = tmp_path / name
        folder.mkdir()
        real_path, pred_path = str(folder / "real.h5ad"), str(folder / "pred.h5ad")
        _adata("real", 1, real_emb if real_emb is not None else {"X_good": CENTERS}).write_h5ad(real_path)
        _adata("pred", 2, pred_emb if pred_emb is not None else {"X_good": CENTERS}).write_h5ad(pred_path)
        return real_path, pred_path

    return write


def _bench(real: str, pred: str, **kwargs) -> dict:
    return benchmark_perturbations(path=real, path_pred=pred, control_pert="ctrl", **kwargs)


# ---------------------------------------------------------------------------
# 1. A requested embedding must be in both files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["minimal", "anndata"])
@pytest.mark.parametrize("keys", ["X_DOES_NOT_EXIST", ["X_good", "X_DOES_NOT_EXIST"]])
def test_missing_embedding_is_an_error(write_pair, profile, keys):
    real, pred = write_pair()
    res = _bench(real, pred, embedding_keys=keys, profile=profile)
    assert res["status"] == "error", res.get("evaluated_keys")
    assert res["error_type"] == "KeyError"
    assert res["missing_keys"] == ["X_DOES_NOT_EXIST"]
    assert res["available_keys"] == ["X_good"]
    assert "X_DOES_NOT_EXIST" in res["error"] and "Available embedding: X_good" in res["error"]
    assert real in res["error"] and pred in res["error"]


def test_embedding_missing_from_the_reference_file_is_an_error(write_pair):
    real, pred = write_pair(real_emb={"X_other": CENTERS}, pred_emb={"X_good": CENTERS, "X_other": CENTERS})
    res = _bench(real, pred, embedding_keys="X_good")
    assert res["status"] == "error", res.get("evaluated_keys")
    assert res["missing_keys"] == ["X_good"]
    assert real in res["error"] and pred not in res["error"].split("Available")[0]
    assert "Available embedding: X_other" in res["error"]


def test_missing_embedding_in_a_split_dataset_is_an_error(tmp_path):
    import anndata

    both = anndata.concat(
        {"observed": _adata("real", 1, {"X_good": CENTERS}), "predicted": _adata("pred", 2, {"X_good": CENTERS})},
        label="split",
    )
    path = str(tmp_path / "split.h5ad")
    both.write_h5ad(path)
    res = benchmark_perturbations(
        path=path, split_col="split", pred_label="predicted", ref_label="observed",
        control_pert="ctrl", embedding_keys="X_DOES_NOT_EXIST",
    )
    assert res["status"] == "error", res.get("evaluated_keys")
    assert "X_DOES_NOT_EXIST" in res["error"] and "Available embedding: X_good" in res["error"]


def test_embedding_widths_must_match_between_files(write_pair):
    real, pred = write_pair(real_emb={"X_good": NARROW}, pred_emb={"X_good": CENTERS})
    res = _bench(real, pred, embedding_keys="X_good", profile="anndata")
    assert res["status"] == "error", res.get("evaluated_keys")
    assert res["error_type"] == "ValueError"
    assert "X_good" in res["error"] and "6 columns" in res["error"] and "4" in res["error"]


# ---------------------------------------------------------------------------
# Scores recomputed from the files
# ---------------------------------------------------------------------------

EXPRESSION = "gene expression (.X)"


def _read(path: str) -> AnnData:
    import anndata

    return anndata.read_h5ad(path)


def _means(adata: AnnData, rep: str | None) -> dict[str, np.ndarray]:
    """Mean of each perturbation's cells, on .X (rep=None) or on .obsm[rep]."""
    matrix = np.asarray(adata.X if rep is None else adata.obsm[rep], dtype=np.float64)
    labels = adata.obs["perturbation"].to_numpy()
    return {p: matrix[labels == p].mean(axis=0) for p in PERTS}


def _discrimination(real: AnnData, pred: AnnData, rep: str | None, metric: str) -> dict[str, float]:
    """cell-eval's discrimination score, recomputed.

    Per perturbation: 1 - rank / n, where rank is the place of its own observed effect when
    all observed effects are sorted by distance to its predicted effect.
    """
    r, q = _means(real, rep), _means(pred, rep)
    names = sorted(p for p in PERTS if p != "ctrl")
    observed = np.vstack([r[p] - r["ctrl"] for p in names])
    scores = {}
    for i, p in enumerate(names):
        predicted = q[p] - q["ctrl"]
        if metric == "cosine":
            distances = 1 - observed @ predicted / (np.linalg.norm(observed, axis=1) * np.linalg.norm(predicted))
        else:  # l2
            distances = np.linalg.norm(observed - predicted, axis=1)
        rank = int(np.flatnonzero(np.argsort(distances) == i)[0])
        scores[p] = 1 - rank / len(names)
    return scores


def _column(res: dict, name: str) -> dict[str, float]:
    return {row["perturbation"]: row[name] for row in res["results"]}


# ---------------------------------------------------------------------------
# 2. Metric settings of one call do not carry over to the next
# ---------------------------------------------------------------------------


def test_a_later_call_scores_expression_not_an_earlier_calls_embedding(write_pair):
    real, pred = write_pair(pred_emb={"X_good": SWAPPED})
    first = _bench(real, pred, embedding_keys="X_good", profile="anndata")
    assert first["status"] == "success", first
    later = _bench(real, pred, profile="anndata")
    assert later["status"] == "success", later
    expected = _discrimination(_read(real), _read(pred), None, "cosine")
    assert expected == {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0}
    assert _column(later, "discrimination_score_cosine") == pytest.approx(expected)


def test_cell_eval_does_not_keep_metric_configs(write_pair, tmp_path):
    import cell_eval

    from embpy.tl import cell_eval as run_cell_eval_suite

    real, pred = write_pair()
    stored = cell_eval.metrics_registry.get_metric("discrimination_score_cosine")
    before = dict(stored.kwargs or {})
    run_cell_eval_suite(
        _read(pred), _read(real), control_pert="ctrl", profile="anndata",
        metric_configs={"discrimination_score_cosine": {"embed_key": "X_good"}},
        skip_de=True, outdir=str(tmp_path / "cell-eval"),
    )
    after = dict(stored.kwargs or {})
    assert after.get("embed_key") != "X_good"
    assert after == before


def _registry_settings() -> dict:
    import copy

    import cell_eval

    return {name: copy.deepcopy(info.kwargs) for name, info in cell_eval.metrics_registry.metrics.items()}


def _cosine_only(**kwargs) -> dict:
    """Run embpy.tl.cell_eval with discrimination_score_cosine as the only metric."""
    import cell_eval

    from embpy.tl import cell_eval as run_cell_eval_suite

    others = cell_eval.metrics_registry.list_metrics(cell_eval.MetricType.ANNDATA_PAIR)
    results, _ = run_cell_eval_suite(
        control_pert="ctrl", profile="anndata", skip_de=True,
        skip_metrics=[m for m in others if m != "discrimination_score_cosine"], **kwargs,
    )
    return dict(zip(results["perturbation"], results["discrimination_score_cosine"]))


def test_overlapping_cell_eval_calls_leave_the_registry_as_before(write_pair, tmp_path, monkeypatch):
    """Two calls on worker threads, as the MCP server runs tools, must not restore each other's settings.

    The patched save and restore steps force the interleaving that broke the registry: B saves
    while A's embed_key is applied, A restores, then B restores what it saved.
    """
    import threading

    import cell_eval

    import embpy.tl.metrics as tl_metrics

    real, pred = write_pair(real_emb={"X_a": CENTERS, "X_b": CENTERS}, pred_emb={"X_a": SWAPPED, "X_b": CENTERS})
    before = _registry_settings()
    save, restore = tl_metrics._stored_metric_kwargs, tl_metrics._restore_metric_kwargs
    a_computed, b_saved, a_restored = threading.Event(), threading.Event(), threading.Event()

    def saving(*args):
        saved = save(*args)
        if threading.current_thread().name == "B":
            b_saved.set()
        return saved

    def restoring(*args):
        name = threading.current_thread().name
        if name == "A":
            a_computed.set()
            b_saved.wait(timeout=3)  # B saves here unless the calls take turns
            restore(*args)
            a_restored.set()
        else:
            if name == "B":
                a_restored.wait(timeout=30)
            restore(*args)

    monkeypatch.setattr(tl_metrics, "_stored_metric_kwargs", saving)
    monkeypatch.setattr(tl_metrics, "_restore_metric_kwargs", restoring)
    scores, failures = {}, []

    def score(key):
        try:
            if key == "X_b":
                a_computed.wait(timeout=60)
            scores[key] = _cosine_only(
                adata_pred=_read(pred), adata_real=_read(real), outdir=str(tmp_path / key),
                metric_configs={"discrimination_score_cosine": {"embed_key": key}},
            )
        except Exception as exc:  # noqa: BLE001 -- reported by the assertion below
            failures.append(exc)

    threads = [threading.Thread(target=score, args=(key,), name=name) for name, key in (("A", "X_a"), ("B", "X_b"))]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert not failures and set(scores) == {"X_a", "X_b"}, failures
        real_ad, pred_ad = _read(real), _read(pred)
        for key in ("X_a", "X_b"):
            assert scores[key] == pytest.approx(_discrimination(real_ad, pred_ad, key, "cosine")), key
        assert _registry_settings() == before
        # A later call without metric_configs scores .X, not the embedding of either call.
        later = _cosine_only(adata_pred=_read(pred), adata_real=_read(real), outdir=str(tmp_path / "later"))
        assert later == pytest.approx(_discrimination(real_ad, pred_ad, None, "cosine"))
    finally:
        for name, kwargs in before.items():
            cell_eval.metrics_registry.metrics[name].kwargs = kwargs


def test_a_cell_eval_call_that_raises_restores_the_registry_and_lets_the_next_call_run(
    write_pair, tmp_path, monkeypatch
):
    import threading

    import cell_eval

    real, pred = write_pair()
    before = _registry_settings()

    def failing(self, *args, metric_configs=None, **kwargs):
        # compute() applies metric_configs to the registry first, then fails
        for name, config in (metric_configs or {}).items():
            cell_eval.metrics_registry.update_metric_kwargs(name, config)
        raise RuntimeError("compute failed")

    with monkeypatch.context() as patch:
        patch.setattr(cell_eval.MetricsEvaluator, "compute", failing)
        with pytest.raises(RuntimeError, match="compute failed"):
            _cosine_only(
                adata_pred=_read(pred), adata_real=_read(real), outdir=str(tmp_path / "failing"),
                metric_configs={"discrimination_score_cosine": {"embed_key": "X_good"}},
            )
    assert _registry_settings() == before

    scores = []
    thread = threading.Thread(
        target=lambda: scores.append(
            _cosine_only(adata_pred=_read(pred), adata_real=_read(real), outdir=str(tmp_path / "next"))
        )
    )
    thread.start()
    thread.join(timeout=60)
    assert scores, "the next call did not finish: the failed call kept the registry locked"


# ---------------------------------------------------------------------------
# 3. Each score says which data it was computed on
# ---------------------------------------------------------------------------


def test_expression_only_profile_reports_the_embedding_as_not_scored(write_pair):
    real, pred = write_pair()
    plain = _bench(real, pred)
    asked = _bench(real, pred, embedding_keys="X_good")
    assert asked["status"] == "success", asked
    assert asked["evaluated_keys"] == []
    assert asked["unscored_keys"] == ["X_good"]
    for hint in ("minimal", "anndata", "full", "discrimination_score_cosine"):
        assert hint in asked["warning"]
    assert set(asked["scored_on"]) >= {"mse", "mae", "pearson_delta", "discrimination_score_l1"}
    assert set(asked["scored_on"].values()) == {EXPRESSION}
    assert asked["results"] == plain["results"]
    # mse of A: mean over genes of (mean predicted - mean observed expression)^2
    r, q = _means(_read(real), None), _means(_read(pred), None)
    assert _column(asked, "mse")["A"] == pytest.approx(np.mean((q["A"] - r["A"]) ** 2), rel=1e-4)


def test_embedding_metrics_read_the_embedding(write_pair):
    right = write_pair("right", pred_emb={"X_good": CENTERS})
    swapped = write_pair("swapped", pred_emb={"X_good": SWAPPED})
    results = {}
    for name, (real, pred) in {"right": right, "swapped": swapped}.items():
        res = _bench(real, pred, embedding_keys="X_good", profile="anndata")
        assert res["status"] == "success", res
        assert res["evaluated_keys"] == ["X_good"]
        assert res["scored_on"]["discrimination_score_cosine[X_good]"] == "embedding X_good (.obsm['X_good'])"
        assert res["scored_on"]["discrimination_score_l2[X_good]"] == "embedding X_good (.obsm['X_good'])"
        assert res["scored_on"]["discrimination_score_cosine"] == EXPRESSION
        assert res["scored_on"]["mse"] == EXPRESSION
        real_ad, pred_ad = _read(real), _read(pred)
        for metric in ("cosine", "l2"):
            assert _column(res, f"discrimination_score_{metric}[X_good]") == pytest.approx(
                _discrimination(real_ad, pred_ad, "X_good", metric)
            )
            assert _column(res, f"discrimination_score_{metric}") == pytest.approx(
                _discrimination(real_ad, pred_ad, None, metric)
            )
        results[name] = res

    # Same expression, different predicted embedding: only the embedding columns change.
    expression_columns = [c for c, data in results["right"]["scored_on"].items() if data == EXPRESSION]
    for column in expression_columns:
        assert _column(results["right"], column) == _column(results["swapped"], column), column
    assert set(_column(results["right"], "discrimination_score_cosine[X_good]").values()) == {1.0}
    assert _column(results["swapped"], "discrimination_score_cosine[X_good]")["A"] < 1.0


def test_each_requested_embedding_is_scored_separately(write_pair):
    real, pred = write_pair(
        real_emb={"X_good": CENTERS, "X_alt": CENTERS},
        pred_emb={"X_good": CENTERS, "X_alt": SWAPPED},
    )
    res = _bench(real, pred, embedding_keys=["X_good", "X_alt"], profile="anndata")
    assert res["status"] == "success", res
    assert res["evaluated_keys"] == ["X_good", "X_alt"]
    real_ad, pred_ad = _read(real), _read(pred)
    for key in ("X_good", "X_alt"):
        assert res["scored_on"][f"discrimination_score_cosine[{key}]"] == f"embedding {key} (.obsm['{key}'])"
        assert _column(res, f"discrimination_score_cosine[{key}]") == pytest.approx(
            _discrimination(real_ad, pred_ad, key, "cosine")
        )
    assert _column(res, "discrimination_score_cosine[X_good]") != _column(res, "discrimination_score_cosine[X_alt]")


def test_an_embedding_cell_eval_cannot_score_is_an_error(write_pair):
    """cell-eval logs a metric that raises and leaves its column out; that is not a score."""
    real, pred = write_pair()
    broken = _read(pred)
    broken.obsm["X_good"][0, 0] = np.nan
    broken.write_h5ad(pred)
    res = _bench(real, pred, embedding_keys="X_good", profile="anndata")
    assert res["status"] == "error", res.get("evaluated_keys")
    assert "X_good" in res["error"] and "NaN" in res["error"]
    assert "discrimination_score_cosine" in res["error"]


# ---------------------------------------------------------------------------
# 4. Reports are written to output_dir, and nowhere else
# ---------------------------------------------------------------------------


def test_reports_are_written_to_output_dir(write_pair, tmp_path, monkeypatch):
    real, pred = write_pair()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    out = tmp_path / "reports"
    res = _bench(real, pred, embedding_keys="X_good", profile="anndata", output_dir=str(out))
    assert res["status"] == "success", res
    assert os.listdir(cwd) == []
    expected = {"results.csv", "agg_results.csv", "real_de.csv", "pred_de.csv"}
    assert set(os.listdir(out)) == expected
    assert res["output_dir"] == str(out)
    assert sorted(res["report_files"]) == sorted(str(out / name) for name in expected)
    table = pd.read_csv(out / "results.csv")
    assert set(res["scored_on"]) <= set(table.columns)
    assert list(table["perturbation"]) == [row["perturbation"] for row in res["results"]]
    assert list(table["discrimination_score_cosine[X_good]"]) == pytest.approx(
        [row["discrimination_score_cosine[X_good]"] for row in res["results"]]
    )


def test_nothing_is_written_without_output_dir(write_pair, tmp_path, monkeypatch):
    real, pred = write_pair()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    res = _bench(real, pred)
    assert res["status"] == "success", res
    assert os.listdir(cwd) == []
    assert res["output_dir"] is None and res["report_files"] == []
