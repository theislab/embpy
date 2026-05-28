from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from world_model.scripts.assemble_ready_adata import main


def test_assemble_ready_adata_attaches_multiple_action_embeddings(tmp_path):
    state_path = tmp_path / "state.h5ad"
    out_path = tmp_path / "ready.h5ad"
    adata = ad.AnnData(
        X=np.zeros((3, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"perturbation": ["control", "A", "B"]},
            index=["c0", "c1", "c2"],
        ),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    adata.obsm["X_stack"] = np.ones((3, 4), dtype=np.float32)
    adata.write_h5ad(state_path)

    genept = tmp_path / "genept.npz"
    np.savez(
        genept,
        symbols=np.asarray(["A", "B"], dtype=object),
        embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        statuses=np.asarray(["RESOLVED", "RESOLVED"], dtype=object),
    )
    mini = tmp_path / "mini.npz"
    np.savez(
        mini,
        symbols=np.asarray(["A"], dtype=object),
        embeddings=np.asarray([[5.0]], dtype=np.float32),
        statuses=np.asarray(["RESOLVED"], dtype=object),
    )

    rc = main(
        [
            "--input-h5ad",
            str(state_path),
            "--output-h5ad",
            str(out_path),
            "--state-obsm-key",
            "X_stack",
            "--perturbation-key",
            "perturbation",
            "--embedding",
            f"genept={genept}",
            "--embedding",
            f"minilm={mini}",
        ]
    )
    assert rc == 0

    out = ad.read_h5ad(out_path)
    assert "X_stack" in out.obsm
    assert out.obsm["X_pert_genept"].shape == (3, 2)
    assert out.obsm["X_pert_minilm"].shape == (3, 1)
    np.testing.assert_allclose(out.obsm["X_pert_genept"][1], [1.0, 2.0])
    np.testing.assert_allclose(out.obsm["X_pert_genept"][2], [3.0, 4.0])
    np.testing.assert_allclose(out.obsm["X_pert_minilm"][1], [5.0])
    np.testing.assert_allclose(out.obsm["X_pert_minilm"][2], [0.0])
    assert out.obs["X_pert_genept_status"].tolist() == ["CONTROL", "RESOLVED", "RESOLVED"]
    assert out.obs["X_pert_minilm_status"].tolist() == ["CONTROL", "RESOLVED", "UNRESOLVED"]
    assert "X_pert_genept" in out.uns["world_model_action_embeddings"]
    assert "X_pert_genept" in out.uns["perturbations"]
