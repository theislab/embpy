"""Legacy loader against the REAL heterogeneous artifacts (read-only).

Covers each id-convention from the un-harmonized data dir:
    genept                -> unnamed index = Ensembl ids
    molformer             -> unnamed index = raw SMILES, e_* dim cols
    morgan                -> 'smiles' column, offset numeric dim cols
    new_combined_rdkit    -> name + canonical_smiles, feat_* dim cols
plus the {ids, embeddings} npz layout via a synthetic file (no network).

The big drug files have ~142k rows; we pass max_rows to keep tests fast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from embpy.io.legacy import load_legacy_embedding
from embpy.io.result import EmbeddingResult

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "embeddings"

GENEPT = DATA / "gene_embeddings" / "genept" / "embeddings_3072.csv"
MOLFORMER = DATA / "drug_embeddings" / "molformer_base" / "embeddings_mean.csv"
MORGAN = DATA / "drug_embeddings" / "morgan_fingerprint" / "fingerprint_1024.csv"
RDKIT = DATA / "new_combined_rdkit_embeddings.csv"


def _valid(res: EmbeddingResult, entity_type: str, id_scheme: str):
    assert isinstance(res, EmbeddingResult)
    assert res.entity_type == entity_type
    assert res.id_scheme == id_scheme
    assert res.n_entities > 0 and res.n_dims > 0
    assert res.matrix.dtype == np.float32


@pytest.mark.skipif(not GENEPT.exists(), reason="genept file not present")
def test_genept_unnamed_ensembl_index():
    res = load_legacy_embedding(GENEPT, entity_type="gene", max_rows=200)
    _valid(res, "gene", "ensembl_gene_id")
    assert all(i.startswith("ENSG") for i in res.entity_ids)
    assert res.n_dims == 3072  # 0..3071 stripped to dim_*


@pytest.mark.skipif(not MOLFORMER.exists(), reason="molformer file not present")
def test_molformer_unnamed_smiles_index():
    res = load_legacy_embedding(MOLFORMER, entity_type="molecule", max_rows=40)
    _valid(res, "molecule", "canonical_smiles")
    # e_* dim columns were stripped; canonicalized SMILES are unique.
    assert len(set(res.entity_ids)) == res.n_entities


@pytest.mark.skipif(not MORGAN.exists(), reason="morgan file not present")
def test_morgan_smiles_column_detected():
    res = load_legacy_embedding(MORGAN, entity_type="molecule", max_rows=40)
    _valid(res, "molecule", "canonical_smiles")


@pytest.mark.skipif(not RDKIT.exists(), reason="rdkit combined file not present")
def test_rdkit_canonical_smiles_id_with_name_alias():
    res = load_legacy_embedding(RDKIT, entity_type="molecule")
    _valid(res, "molecule", "canonical_smiles")
    # 'name' column kept as a display alias, never the key.
    assert res.aliases is not None
    assert any("name" in v for v in res.aliases.values())


def test_npz_ids_embeddings_layout(tmp_path):
    # Synthetic {gene_ids, embeddings} npz (the esm2 layout) -- entity_type
    # 'sequence' keeps ids verbatim so no network resolution is needed.
    p = tmp_path / "emb.npz"
    np.savez(
        p,
        gene_ids=np.array(["AAA", "BBB", "CCC"], dtype=object),
        embeddings=np.arange(9, dtype=np.float32).reshape(3, 3),
        model=np.array("toy"),
    )
    res = load_legacy_embedding(p, entity_type="sequence")
    _valid(res, "sequence", "sequence")
    assert res.entity_ids == ("AAA", "BBB", "CCC")
    assert res.matrix.shape == (3, 3)


def test_unknown_entity_type_raises(tmp_path):
    p = tmp_path / "x.npz"
    np.savez(p, ids=np.array(["a"], dtype=object), embeddings=np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match=r"entity_type must be one of"):
        load_legacy_embedding(p, entity_type="cell_line")  # type: ignore[arg-type]
