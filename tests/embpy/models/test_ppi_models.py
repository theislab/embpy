"""Tests for embpy.models.ppi_models – PrecomputedPPIWrapper."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py", reason="h5py not installed")

from embpy.models.ppi_models import PrecomputedPPIWrapper, _fetch_gene_names

# =====================================================================
# Helpers
# =====================================================================

_SPECIES = 9606
_DIM_FUNCTIONAL = 512
_DIM_NODE2VEC = 128
_N_PROTEINS = 5
_PROTEIN_IDS = [
    f"{_SPECIES}.ENSP{i:05d}" for i in range(1, _N_PROTEINS + 1)
]
_GENE_NAMES = ["TP53", "BRCA1", "EGFR", "KRAS", "MYC"]


def _make_h5(path: str, dim: int) -> None:
    """Create a minimal HDF5 file matching the precomputed format."""
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "embeddings",
            data=np.random.randn(_N_PROTEINS, dim).astype(np.float16),
        )
        f.create_dataset(
            "proteins",
            data=np.array(_PROTEIN_IDS, dtype=object),
        )
        f.create_group("metadata")


def _mock_string_response() -> str:
    """TSV response from STRING get_string_ids for our fake proteins."""
    header = "queryIndex\tqueryItem\tstringId\tncbiTaxonId\ttaxonName\tpreferredName\tannotation\n"
    rows = []
    for i, pid in enumerate(_PROTEIN_IDS):
        acc = pid.split(".", 1)[1]
        rows.append(
            f"{i}\t{acc}\t{pid}\t{_SPECIES}\tHomo sapiens\t{_GENE_NAMES[i]}\ttest\n"
        )
    return header + "".join(rows)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture()
def data_dir(tmp_path):
    """Create a temp directory with both functional and node2vec H5 files."""
    func_dir = tmp_path / "functional_embeddings" / "functional_emb"
    func_dir.mkdir(parents=True)
    _make_h5(str(func_dir / f"{_SPECIES}.h5"), _DIM_FUNCTIONAL)

    n2v_dir = tmp_path / "node2vec" / "node2vec"
    n2v_dir.mkdir(parents=True)
    _make_h5(str(n2v_dir / f"{_SPECIES}.h5"), _DIM_NODE2VEC)

    return str(tmp_path)


@pytest.fixture()
def loaded_wrapper(data_dir):
    """A PrecomputedPPIWrapper loaded with functional embeddings."""
    with patch("embpy.models.ppi_models.requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.text = _mock_string_response()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        w = PrecomputedPPIWrapper(
            data_dir=data_dir,
            species=_SPECIES,
            embedding_type="functional",
        )
        w.load(torch.device("cpu"))
    return w


# =====================================================================
# Construction
# =====================================================================


class TestConstruction:
    def test_invalid_embedding_type(self, data_dir) -> None:
        with pytest.raises(ValueError, match="Unknown embedding_type"):
            PrecomputedPPIWrapper(
                data_dir=data_dir,
                embedding_type="invalid",  # type: ignore[arg-type]
            )

    def test_missing_data_dir_raises(self) -> None:
        w = PrecomputedPPIWrapper(data_dir=None)
        with pytest.raises(ValueError, match="data_dir must be provided"):
            w.load(torch.device("cpu"))

    def test_missing_h5_file_raises(self, tmp_path) -> None:
        w = PrecomputedPPIWrapper(
            data_dir=str(tmp_path),
            species=99999,
        )
        with pytest.raises(FileNotFoundError, match="HDF5 file not found"):
            w.load(torch.device("cpu"))


# =====================================================================
# Loading
# =====================================================================


class TestLoad:
    def test_loads_functional(self, loaded_wrapper) -> None:
        assert loaded_wrapper.num_proteins == _N_PROTEINS
        assert loaded_wrapper.embedding_dim == _DIM_FUNCTIONAL
        assert len(loaded_wrapper.available_genes) == _N_PROTEINS
        assert len(loaded_wrapper.available_proteins) == _N_PROTEINS

    def test_loads_node2vec(self, data_dir) -> None:
        with patch("embpy.models.ppi_models.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.text = _mock_string_response()
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            w = PrecomputedPPIWrapper(
                data_dir=data_dir,
                species=_SPECIES,
                embedding_type="node2vec",
            )
            w.load(torch.device("cpu"))

        assert w.embedding_dim == _DIM_NODE2VEC
        assert w.num_proteins == _N_PROTEINS

    def test_gene_names_available(self, loaded_wrapper) -> None:
        for gene in _GENE_NAMES:
            assert gene in loaded_wrapper.available_genes

    def test_protein_ids_available(self, loaded_wrapper) -> None:
        for pid in _PROTEIN_IDS:
            assert pid in loaded_wrapper.available_proteins


# =====================================================================
# Embed single
# =====================================================================


class TestEmbed:
    def test_embed_by_gene(self, loaded_wrapper) -> None:
        emb = loaded_wrapper.embed("TP53")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (_DIM_FUNCTIONAL,)
        assert emb.dtype == np.float32

    def test_embed_by_protein_id(self, loaded_wrapper) -> None:
        pid = _PROTEIN_IDS[0]
        emb = loaded_wrapper.embed(pid)
        assert emb.shape == (_DIM_FUNCTIONAL,)

    def test_embed_unknown_raises(self, loaded_wrapper) -> None:
        with pytest.raises(ValueError, match="not found"):
            loaded_wrapper.embed("NONEXISTENT_GENE")

    def test_embed_before_load_raises(self, data_dir) -> None:
        w = PrecomputedPPIWrapper(data_dir=data_dir)
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed("TP53")

    def test_different_genes_may_differ(self, loaded_wrapper) -> None:
        emb1 = loaded_wrapper.embed("TP53")
        emb2 = loaded_wrapper.embed("BRCA1")
        assert emb1.shape == emb2.shape


# =====================================================================
# Embed batch
# =====================================================================


class TestEmbedBatch:
    def test_batch(self, loaded_wrapper) -> None:
        embs = loaded_wrapper.embed_batch(["TP53", "BRCA1", "EGFR"])
        assert len(embs) == 3
        for emb in embs:
            assert emb.shape == (_DIM_FUNCTIONAL,)
            assert emb.dtype == np.float32

    def test_missing_gene_returns_zeros(self, loaded_wrapper) -> None:
        embs = loaded_wrapper.embed_batch(["TP53", "NONEXISTENT"])
        assert embs[0].shape == (_DIM_FUNCTIONAL,)
        assert np.allclose(embs[1], 0.0)

    def test_batch_before_load_raises(self, data_dir) -> None:
        w = PrecomputedPPIWrapper(data_dir=data_dir)
        with pytest.raises(RuntimeError, match="not loaded"):
            w.embed_batch(["TP53"])


# =====================================================================
# Properties
# =====================================================================


class TestProperties:
    def test_embedding_dim_before_load(self) -> None:
        w = PrecomputedPPIWrapper(data_dir="/nonexistent")
        assert w.embedding_dim == 0

    def test_num_proteins_before_load(self) -> None:
        w = PrecomputedPPIWrapper(data_dir="/nonexistent")
        assert w.num_proteins == 0

    def test_model_type(self, loaded_wrapper) -> None:
        assert loaded_wrapper.model_type == "ppi"


# =====================================================================
# _fetch_gene_names (mocked)
# =====================================================================


class TestFetchGeneNames:
    @patch("embpy.models.ppi_models.requests.post")
    def test_maps_accessions(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.text = _mock_string_response()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        accessions = [pid.split(".", 1)[1] for pid in _PROTEIN_IDS]
        result = _fetch_gene_names(accessions, species=_SPECIES)
        assert len(result) == _N_PROTEINS
        for acc, gene in zip(accessions, _GENE_NAMES):
            assert result[acc] == gene

    @patch("embpy.models.ppi_models.requests.post")
    def test_batching(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.text = _mock_string_response()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        accessions = [pid.split(".", 1)[1] for pid in _PROTEIN_IDS]
        _fetch_gene_names(accessions, species=_SPECIES, batch_size=2)
        assert mock_post.call_count == 3  # ceil(5/2)
