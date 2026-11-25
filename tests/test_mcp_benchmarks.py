"""benchmark_perturbations: the data each cell-eval score is computed on, and where reports go.

Every test writes a reference and a prediction .h5ad with gene expression in ``.X`` and
embeddings in ``.obsm``, and checks the scores against values recomputed here with numpy.
"""

from __future__ import annotations

import numpy as np
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
