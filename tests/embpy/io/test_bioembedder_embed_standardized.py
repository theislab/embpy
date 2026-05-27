from __future__ import annotations

from types import SimpleNamespace

import numpy as np

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

    payload = out.uns["embpy"]["perturbations"]["X_pert_toy"]
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
    meta = out.uns["embpy"]["embeddings"]["X_emb__gene__m1__pool_mean"]["provenance"]["extra"]
    assert meta["n_requested_inputs"] == 2
    assert meta["duplicate_canonical_ids_dropped"] == 0


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
