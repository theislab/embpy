from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from embpy.embedder import BioEmbedder


def _bare_embedder() -> BioEmbedder:
    emb = BioEmbedder.__new__(BioEmbedder)
    emb.organism = "human"
    emb.gene_resolver = SimpleNamespace(
        symbols_to_ensembl_batch=lambda ids, organism="human": {
            "TP53": "ENSG00000141510",
            "MYC": "ENSG00000136997",
        },
        get_gene_sequences=lambda biotype="protein_coding": {
            "ENSG00000141510": "AAAA",
            "ENSG00000136997": "CCCC",
        },
    )
    emb.protein_resolver = SimpleNamespace()
    return emb


def test_embed_list_of_genes_to_table_uses_canonical_ids(monkeypatch):
    emb = _bare_embedder()

    def fake_embed_genes_batch(**kwargs):
        return [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ]

    monkeypatch.setattr(emb, "embed_genes_batch", fake_embed_genes_batch)
    out = emb.embed(["TP53", "MYC"], entity_type="gene", model="toy", output="table")

    assert list(out.index) == ["ENSG00000141510", "ENSG00000136997"]
    assert out.index.name == "ensembl_gene_id"
    assert list(out.columns) == ["gene_symbol", "dim_0", "dim_1"]
    assert out.loc["ENSG00000141510", "gene_symbol"] == "TP53"


def test_embed_list_of_genes_to_payload_includes_canonical_ids_and_model(monkeypatch):
    emb = _bare_embedder()

    def fake_embed_genes_batch(**kwargs):
        return [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ]

    monkeypatch.setattr(emb, "embed_genes_batch", fake_embed_genes_batch)
    out = emb.embed(["TP53", "MYC"], entity_type="gene", model="toy", output="payload")

    assert out["entity_ids"] == ["ENSG00000141510", "ENSG00000136997"]
    assert out["id_scheme"] == "ensembl_gene_id"
    assert out["model"]["name"] == "toy"
    assert out["provenance"]["extra"]["n_requested_inputs"] == 2
    assert out["matrix"].shape == (2, 2)


def test_embed_list_of_genes_attach_to_uns(monkeypatch):
    from anndata import AnnData

    emb = _bare_embedder()

    def fake_embed_genes_batch(**kwargs):
        return [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ]

    monkeypatch.setattr(emb, "embed_genes_batch", fake_embed_genes_batch)
    target = AnnData(X=np.ones((3, 4), dtype=np.float32))
    out = emb.embed(
        ["TP53", "MYC"],
        entity_type="gene",
        model="toy",
        output="anndata",
        target=target,
        attach_to="uns",
        key="X_pert_toy",
    )

    payload = out.uns["perturbations"]["X_pert_toy"]
    assert payload["entity_ids"] == ["ENSG00000141510", "ENSG00000136997"]
    assert "X_pert_toy" not in out.obsm


def test_embed_multi_model_to_anndata(monkeypatch):
    emb = _bare_embedder()

    def fake_embed_genes_batch(**kwargs):
        scale = 1.0 if kwargs["model"] == "m1" else 10.0
        return [
            np.array([scale, scale + 1], dtype=np.float32),
            np.array([scale + 2, scale + 3], dtype=np.float32),
        ]

    monkeypatch.setattr(emb, "embed_genes_batch", fake_embed_genes_batch)
    out = emb.embed(["TP53", "MYC"], entity_type="gene", model=["m1", "m2"])

    assert set(out.varm.keys()) == {"X_emb__gene__m1__pool_mean", "X_emb__gene__m2__pool_mean"}
    assert list(out.var_names) == ["ENSG00000141510", "ENSG00000136997"]
    assert out.X.nnz == 0
    meta = out.uns["embeddings"]["X_emb__gene__m1__pool_mean"]["provenance"]["extra"]
    assert meta["n_requested_inputs"] == 2
    assert meta["duplicate_canonical_ids_dropped"] == 0


def test_embed_gene_perturbations_to_anndata_obsm(monkeypatch):
    emb = _bare_embedder()

    def fake_embed_genes_batch(**kwargs):
        return [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ]

    monkeypatch.setattr(emb, "embed_genes_batch", fake_embed_genes_batch)
    out = emb.embed(
        ["TP53", "MYC"],
        entity_type="gene",
        model="toy",
        output="anndata",
        is_perturbation=True,
        key="X_pert_toy",
    )

    assert "X_pert_toy" in out.obsm
    assert "X_pert_toy" not in out.varm
    assert list(out.obs_names) == ["ENSG00000141510", "ENSG00000136997"]
    meta = out.uns["embeddings"]["X_pert_toy"]["provenance"]["extra"]
    assert meta["is_perturbation"] is True


def test_embed_gene_perturbations_align_to_symbol_obs_names(monkeypatch):
    from anndata import AnnData

    emb = _bare_embedder()

    def fake_embed_genes_batch(**kwargs):
        return [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ]

    monkeypatch.setattr(emb, "embed_genes_batch", fake_embed_genes_batch)
    target = AnnData(X=np.zeros((2, 1), dtype=np.float32))
    target.obs_names = ["TP53", "MYC"]
    target.var_names = ["placeholder"]

    out = emb.embed(
        target,
        entity_type="gene",
        model="toy",
        output="anndata",
        anndata_axis="obs",
        is_perturbation=True,
        key="X_pert_toy",
    )

    assert np.array_equal(out.obsm["X_pert_toy"], np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    assert list(out.obs["gene_symbol"]) == ["TP53", "MYC"]


def test_embed_whole_genome_uses_ensembl_ids_and_prefetched_sequences(monkeypatch):
    emb = _bare_embedder()
    seen = {}

    def fake_embed_genes_batch(**kwargs):
        seen.update(kwargs)
        assert kwargs["id_type"] == "ensembl_id"
        assert kwargs["prefetched_sequences"] == {
            "ENSG00000141510": "AAAA",
            "ENSG00000136997": "CCCC",
        }
        return [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ]

    monkeypatch.setattr(emb, "embed_genes_batch", fake_embed_genes_batch)
    out = emb.embed(entity_type="gene", model="toy", whole_genome=True, output="table")

    assert list(out.index) == ["ENSG00000141510", "ENSG00000136997"]
    assert seen["fetch_all_dna"] is True


def test_embed_static_hf_gene_embeddings_to_table(monkeypatch):
    emb = _bare_embedder()

    def fail_if_model_inference_runs(**kwargs):
        raise AssertionError("static HF lookup should not call embed_genes_batch")

    def fake_static_table(*args, **kwargs):
        return {
            "ids": ["TP53", "MYC"],
            "matrix": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "id_key": "symbols",
            "id_type": "symbol",
        }

    monkeypatch.setattr(emb, "embed_genes_batch", fail_if_model_inference_runs)
    monkeypatch.setattr(emb, "_load_static_embedding_table", fake_static_table)

    out = emb.embed(
        ["TP53", "MYC"],
        entity_type="gene",
        model="genept",
        output="table",
    )

    assert list(out.index) == ["ENSG00000141510", "ENSG00000136997"]
    assert list(out.columns) == ["gene_symbol", "dim_0", "dim_1"]
    assert np.array_equal(out[["dim_0", "dim_1"]].to_numpy(), np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_embed_static_hf_warns_and_drops_missing_genes(monkeypatch):
    emb = _bare_embedder()

    def fake_static_table(*args, **kwargs):
        return {
            "ids": ["TP53"],
            "matrix": np.array([[1.0, 2.0]], dtype=np.float32),
            "id_key": "symbols",
            "id_type": "symbol",
        }

    monkeypatch.setattr(emb, "_load_static_embedding_table", fake_static_table)

    with pytest.warns(UserWarning, match="available for 1/2 requested gene"):
        out = emb.embed(
            ["TP53", "MYC"],
            entity_type="gene",
            model="genept",
            output="table",
        )

    assert list(out.index) == ["ENSG00000141510"]


def test_embed_static_hf_gene_perturbations_to_obsm(monkeypatch):
    emb = _bare_embedder()

    def fake_static_table(*args, **kwargs):
        return {
            "ids": ["TP53", "MYC"],
            "matrix": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "id_key": "symbols",
            "id_type": "symbol",
        }

    monkeypatch.setattr(emb, "_load_static_embedding_table", fake_static_table)

    out = emb.embed(
        ["TP53", "MYC"],
        entity_type="gene",
        model="genept",
        embedding_source="static",
        output="anndata",
        is_perturbation=True,
        key="X_pert_genept",
    )

    assert "X_pert_genept" in out.obsm
    assert "X_pert_genept" not in out.varm
    assert list(out.obs_names) == ["ENSG00000141510", "ENSG00000136997"]
    meta = out.uns["embeddings"]["X_pert_genept"]["provenance"]["extra"]
    assert meta["embedding_source"] == "hf"
    assert meta["hf_repo_id"] == "theislab/Embpy_Data"
    assert meta["is_perturbation"] is True
