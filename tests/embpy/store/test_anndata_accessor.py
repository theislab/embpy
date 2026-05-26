from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from embpy.io.result import EmbeddingProvenance, EmbeddingResult
from embpy.store import EmbeddingStore


def _adata() -> AnnData:
    obs = pd.DataFrame(
        {
            "perturbation": ["g1", "g1", "g2", "g2", "g1+g2", "g1+g2"],
            "target": ["g1", "g1", "g2", "g2", "combo", "combo"],
            "score": [1.0, 1.2, 3.0, 2.8, 2.0, 2.2],
        },
        index=[f"cell{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    return AnnData(X=np.arange(18, dtype=np.float32).reshape(6, 3), obs=obs, var=var)


def _obs_embedding() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.5, 0.5],
            [0.45, 0.55],
        ],
        dtype=np.float32,
    )


def _gene_result() -> EmbeddingResult:
    return EmbeddingResult(
        matrix=np.array([[1.0, 0.0], [0.0, 1.0], [0.25, 0.25]], dtype=np.float32),
        entity_ids=("g1", "g2", "g3"),
        entity_type="gene",
        id_scheme="symbol",
        provenance=EmbeddingProvenance(model="toy_gene"),
    )


def test_accessor_initializes_registry_and_registers_obs_var_embeddings():
    adata = _adata()
    x_before = adata.X.copy()

    adata.embpy.register_embedding(
        "X_cells",
        _obs_embedding(),
        entity_ids=adata.obs_names,
        entity_type="cell",
        id_scheme="obs_name",
    )
    adata.embpy.register_embedding("X_gene", result=_gene_result())

    assert "embpy" in adata.uns
    assert "X_cells" in adata.obsm
    assert "X_gene" in adata.varm
    assert np.array_equal(adata.X, x_before)
    assert set(adata.embpy.list_embeddings()["key"]) == {"X_cells", "X_gene"}


def test_aggregate_neighbors_correlate_and_compare_embeddings():
    adata = _adata()
    adata.embpy.register_embedding(
        "X_a",
        _obs_embedding(),
        entity_ids=adata.obs_names,
        entity_type="cell",
        id_scheme="obs_name",
    )
    adata.embpy.register_embedding(
        "X_b",
        _obs_embedding()[:, ::-1],
        entity_ids=adata.obs_names,
        entity_type="cell",
        id_scheme="obs_name",
    )

    agg = adata.embpy.aggregate("X_a", by="perturbation")
    neigh = adata.embpy.neighbors("X_a", query="cell0", k=2)
    corr = adata.embpy.correlate("X_a", phenotype="score")
    comparison = adata.embpy.compare_embeddings(["X_a", "X_b"])

    assert agg.shape == (3, 2)
    assert neigh.iloc[0]["neighbor_id"] == "cell1"
    assert {"pearson", "spearman", "n_pairs"} <= set(corr.columns)
    assert comparison.loc[0, "embedding_a"] == "X_a"
    assert "knn_jaccard" in comparison.columns


def test_score_activity_wraps_existing_metric():
    adata = _adata()
    adata.embpy.register_embedding(
        "X_cells",
        _obs_embedding(),
        entity_ids=adata.obs_names,
        entity_type="cell",
        id_scheme="obs_name",
    )

    out = adata.embpy.score_activity("X_cells", perturbation_col="perturbation", chunk_size=3)

    assert {"perturbation", "mean_ap", "n_wells"} <= set(out.columns)


def test_setup_conditions_compile_actions_single_and_multi_target():
    adata = _adata()
    store = EmbeddingStore.from_results(_gene_result())
    adata.embpy.register_store(store)
    adata.embpy.setup_conditions("perturbation", control_values=["DMSO"])

    table = adata.embpy.compile_actions(
        target_embedding="gene:toy_gene",
        perturbation_key="perturbation",
    )

    assert "X_embpy_action" in adata.obsm
    assert table.loc["g1+g2"].to_numpy().tolist() == pytest.approx([0.5, 0.5])
    assert adata.obsm["X_embpy_action"].shape == (adata.n_obs, 2)


def test_compile_actions_uses_relation_and_errors_on_missing_targets():
    adata = _adata()
    store = EmbeddingStore.from_results(_gene_result())
    adata.embpy.register_store(store)
    adata.embpy.register_relation(
        "perturbation_targets_gene",
        pd.DataFrame({"source_id": ["g1", "g2", "g1+g2"], "target_id": ["g1", "g2", "missing"]}),
        source_type="perturbation",
        target_type="gene",
    )

    with pytest.raises(ValueError, match="no target embeddings"):
        adata.embpy.compile_actions(
            relation="perturbation_targets_gene",
            target_embedding="gene:toy_gene",
            perturbation_key="perturbation",
        )


def test_splits_by_target_prevent_group_overlap_and_torch_dataset():
    adata = _adata()
    store = EmbeddingStore.from_results(_gene_result())
    adata.embpy.register_store(store)
    adata.embpy.compile_actions(target_embedding="gene:toy_gene", perturbation_key="perturbation")

    splits = adata.embpy.make_splits(by="target", random_state=1)
    labels = adata.obs["target"].astype(str).to_numpy()
    train_targets = set(labels[splits["train"]])
    test_targets = set(labels[splits["test"]])
    dataset = adata.embpy.make_torch_dataset(split=splits["train"])

    assert train_targets.isdisjoint(test_targets)
    sample = dataset[0]
    assert {"state", "action", "target", "obs_index", "obs_name"} <= set(sample)


def test_prompt_context_is_compact_and_prompt_ready():
    adata = _adata()
    store = EmbeddingStore.from_results(_gene_result())
    adata.embpy.register_store(store)
    adata.embpy.register_relation(
        "perturbation_targets_gene",
        pd.DataFrame({"source_id": ["g1", "g2", "g1+g2"], "target_id": ["g1", "g2", "g1"]}),
        source_type="perturbation",
        target_type="gene",
    )
    adata.embpy.register_embedding(
        "X_cells",
        _obs_embedding(),
        entity_ids=adata.obs_names,
        entity_type="cell",
        id_scheme="obs_name",
    )
    adata.embpy.setup_conditions("perturbation", control_values=[])
    adata.embpy.compile_actions(target_embedding="gene:toy_gene", perturbation_key="perturbation")
    adata.embpy.make_splits(by="target", random_state=0)

    context = adata.embpy.prompt_context()
    markdown = adata.embpy.prompt_context(output="markdown")
    payload = json.dumps(context)

    assert context["anndata"]["n_obs"] == adata.n_obs
    assert context["embeddings"][0]["key"] == "X_cells"
    assert "gene:toy_gene" in context["stores"][0]["keys"]
    assert context["relations"][0]["n_edges"] == 3
    assert context["conditions"]["n_table"]["obs_id"] == adata.n_obs
    assert context["actions"]["X_embpy_action"]["status_counts"] == {"RESOLVED": 3}
    assert context["actions"]["X_embpy_action"]["n_condition_vectors"]["dim_0"] == 3
    assert context["splits"]["n_indices"]["train"] > 0
    assert "condition_vectors" not in context["actions"]["X_embpy_action"]
    assert "statuses" not in context["actions"]["X_embpy_action"]
    assert len(payload) < 8000
    assert "AnnData:" in markdown
    assert "`X_embpy_action`" in markdown
    assert "## Conditions" in markdown
    assert "## Splits" in markdown


def test_plotting_wrappers_return_figures():
    import matplotlib

    matplotlib.use("Agg")
    adata = _adata()
    adata.embpy.register_embedding(
        "X_cells",
        _obs_embedding(),
        entity_ids=adata.obs_names,
        entity_type="cell",
        id_scheme="obs_name",
    )

    fig = adata.embpy.plot_embedding("X_cells", method="pca", color="perturbation")
    sim_fig = adata.embpy.plot_similarity("X_cells")
    diag_fig = adata.embpy.plot_diagnostics(["X_cells"])

    assert fig.axes
    assert sim_fig.axes
    assert diag_fig.axes


def test_compile_actions_records_status_and_handles_control_and_unresolved():
    from embpy.store.actions import control_sentinel_vector

    obs = pd.DataFrame(
        {"perturbation": ["g1", "g1", "non-targeting", "g2", "ghost", "ghost"]},
        index=[f"c{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    adata = AnnData(X=np.zeros((6, 3), dtype=np.float32), obs=obs, var=var)
    store = EmbeddingStore.from_results(_gene_result())
    adata.embpy.register_store(store)
    # control_values flow from setup_conditions into compile_actions by default.
    adata.embpy.setup_conditions("perturbation", control_values=["non-targeting"])

    table = adata.embpy.compile_actions(
        target_embedding="gene:toy_gene",
        perturbation_key="perturbation",
        on_unresolved="zero",
    )

    sentinel = control_sentinel_vector(2, seed=0)
    assert table.loc["non-targeting"].to_numpy().tolist() == pytest.approx(sentinel.tolist())
    assert table.loc["ghost"].to_numpy().tolist() == [0.0, 0.0]
    assert adata.obsm["X_embpy_action"].shape == (6, 2)

    meta = adata.uns["embpy"]["actions"]["X_embpy_action"]
    assert meta["statuses"]["non-targeting"] == "CONTROL"
    assert meta["statuses"]["ghost"] == "UNRESOLVED"
    assert meta["statuses"]["g1"] == "RESOLVED"
    assert (meta["n_control"], meta["n_unresolved"], meta["n_resolved"]) == (1, 1, 2)
    assert meta["missing_targets"]["ghost"] == ["ghost"]
