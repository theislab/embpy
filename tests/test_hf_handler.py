"""Tests for embpy.pp.hf_handler — HFHandler with mocked HF Hub calls."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from embpy.pp.hf_handler import HFHandler, _read_tabular, _read_to_anndata


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def handler():
    return HFHandler("user/test-repo", token="fake-token")


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_adata():
    obs = pd.DataFrame({"drug": ["aspirin", "ibuprofen"], "dose": [10.0, 20.0]})
    var = pd.DataFrame({"gene_name": ["TP53", "BRCA1"]}, index=pd.Index(["ENSG1", "ENSG2"]))
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    return AnnData(X=X, obs=obs, var=var)


# =====================================================================
# File-format readers (unit-level, no mocking needed)
# =====================================================================


class TestReadToAnndata:
    def test_csv(self, tmp_dir):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=pd.Index(["r0", "r1"]))
        path = tmp_dir / "data.csv"
        df.to_csv(path)
        adata = _read_to_anndata(path)
        assert isinstance(adata, AnnData)
        assert adata.shape == (2, 2)

    def test_parquet(self, tmp_dir):
        df = pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0]})
        path = tmp_dir / "data.parquet"
        df.to_parquet(path)
        adata = _read_to_anndata(path)
        assert isinstance(adata, AnnData)

    def test_h5ad(self, tmp_dir, sample_adata):
        path = tmp_dir / "data.h5ad"
        sample_adata.write_h5ad(path)
        adata = _read_to_anndata(path)
        assert isinstance(adata, AnnData)
        assert adata.shape == sample_adata.shape

    def test_unsupported_extension(self, tmp_dir):
        path = tmp_dir / "data.json"
        path.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            _read_to_anndata(path)


class TestReadTabular:
    def test_csv(self, tmp_dir):
        df = pd.DataFrame({"a": [1], "b": [2]})
        path = tmp_dir / "t.csv"
        df.to_csv(path, index=False)
        result = _read_tabular(path)
        assert list(result.columns) == ["a", "b"]

    def test_parquet(self, tmp_dir):
        df = pd.DataFrame({"a": [1], "b": [2]})
        path = tmp_dir / "t.parquet"
        df.to_parquet(path, index=False)
        result = _read_tabular(path)
        assert list(result.columns) == ["a", "b"]

    def test_unsupported(self, tmp_dir):
        path = tmp_dir / "t.txt"
        path.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported tabular extension"):
            _read_tabular(path)


# =====================================================================
# HFHandler — constructor & repr
# =====================================================================


class TestHFHandlerInit:
    def test_init(self, handler):
        assert handler.repo_name == "user/test-repo"
        assert handler.token == "fake-token"

    def test_repr(self, handler):
        assert "user/test-repo" in repr(handler)


# =====================================================================
# Repository helpers
# =====================================================================


class TestCreateRepo:
    def test_create_repo_calls_api(self, handler):
        handler._api.create_repo = MagicMock(return_value="https://hf.co/datasets/user/test-repo")
        url = handler.create_repo(private=True)
        handler._api.create_repo.assert_called_once_with(
            "user/test-repo",
            repo_type="dataset",
            exist_ok=True,
            private=True,
        )
        assert "user/test-repo" in url


class TestListFiles:
    @patch("embpy.pp.hf_handler.list_repo_files")
    def test_list_files(self, mock_list, handler):
        mock_list.return_value = ["raw/tahoe.h5ad", "embeddings/model.parquet"]
        result = handler.list_files()
        assert result == ["raw/tahoe.h5ad", "embeddings/model.parquet"]
        mock_list.assert_called_once_with("user/test-repo", repo_type="dataset", token="fake-token")


# =====================================================================
# Upload
# =====================================================================


class TestUploadFile:
    def test_upload_file(self, handler):
        handler._api.upload_file = MagicMock()
        handler.upload_file("/tmp/foo.h5ad", "raw/tahoe.h5ad")
        handler._api.upload_file.assert_called_once_with(
            path_or_fileobj="/tmp/foo.h5ad",
            path_in_repo="raw/tahoe.h5ad",
            repo_id="user/test-repo",
            repo_type="dataset",
        )


class TestUploadDatasets:
    def test_uploads_h5ad(self, handler, tmp_dir):
        f = tmp_dir / "tahoe.h5ad"
        f.write_text("fake")
        handler.upload_file = MagicMock()
        handler.upload_datasets({"tahoe": str(f)})
        handler.upload_file.assert_called_once_with(f, "raw/tahoe.h5ad")

    def test_uploads_csv(self, handler, tmp_dir):
        f = tmp_dir / "lincs.csv"
        f.write_text("a,b\n1,2")
        handler.upload_file = MagicMock()
        handler.upload_datasets({"lincs": str(f)})
        handler.upload_file.assert_called_once_with(f, "raw/lincs.csv")

    def test_skips_missing(self, handler, caplog):
        handler.upload_file = MagicMock()
        handler.upload_datasets({"missing": "/nonexistent/file.h5ad"})
        handler.upload_file.assert_not_called()

    def test_skips_unsupported_format(self, handler, tmp_dir, caplog):
        f = tmp_dir / "data.json"
        f.write_text("{}")
        handler.upload_file = MagicMock()
        handler.upload_datasets({"bad": str(f)})
        handler.upload_file.assert_not_called()


class TestUploadAnndata:
    def test_upload_anndata(self, handler, sample_adata):
        handler.upload_file = MagicMock()
        handler.upload_anndata(sample_adata, "test_ds")
        handler.upload_file.assert_called_once()
        args = handler.upload_file.call_args
        assert args[0][1] == "raw/test_ds.h5ad"
        assert Path(args[0][0]).suffix == ".h5ad"


class TestUploadMetadata:
    def test_upload_dataframe(self, handler):
        handler.upload_file = MagicMock()
        df = pd.DataFrame({"drug": ["aspirin"], "smiles": ["CC(=O)O"]})
        handler.upload_metadata(df)
        handler.upload_file.assert_called_once()
        assert handler.upload_file.call_args[0][1] == "metadata/perturbations.parquet"

    def test_upload_path(self, handler, tmp_dir):
        handler.upload_file = MagicMock()
        f = tmp_dir / "meta.parquet"
        f.write_text("fake")
        handler.upload_metadata(str(f))
        handler.upload_file.assert_called_once_with(str(f), "metadata/perturbations.parquet")


class TestUploadEmbeddings:
    def test_uploads_parquet_and_npy(self, handler, tmp_dir):
        (tmp_dir / "model_a.parquet").write_text("fake")
        (tmp_dir / "model_b.npy").write_bytes(b"\x00")
        (tmp_dir / "model_b_index.csv").write_text("id\na\nb")
        (tmp_dir / "ignore.txt").write_text("skip")

        handler.upload_file = MagicMock()
        handler.upload_embeddings(tmp_dir)

        uploaded = {c[0][1] for c in handler.upload_file.call_args_list}
        assert "embeddings/model_a.parquet" in uploaded
        assert "embeddings/model_b.npy" in uploaded
        assert "embeddings/model_b_index.csv" in uploaded
        assert "embeddings/ignore.txt" not in uploaded

    def test_raises_if_dir_missing(self, handler):
        with pytest.raises(FileNotFoundError):
            handler.upload_embeddings("/nonexistent/dir")


class TestUploadEmbeddingArray:
    def test_parquet_format(self, handler):
        handler.upload_file = MagicMock()
        emb = np.random.randn(3, 8).astype(np.float32)
        ids = ["CCO", "O=C=O", "c1ccccc1"]
        handler.upload_embedding_array("test_model", emb, ids, fmt="parquet")
        handler.upload_file.assert_called_once()
        assert handler.upload_file.call_args[0][1] == "embeddings/test_model.parquet"

    def test_npy_format(self, handler):
        handler.upload_file = MagicMock()
        emb = np.random.randn(2, 4).astype(np.float32)
        ids = ["CCO", "O=C=O"]
        handler.upload_embedding_array("mdl", emb, ids, fmt="npy")
        assert handler.upload_file.call_count == 2
        paths = {c[0][1] for c in handler.upload_file.call_args_list}
        assert "embeddings/mdl.npy" in paths
        assert "embeddings/mdl_index.csv" in paths

    def test_length_mismatch(self, handler):
        emb = np.zeros((3, 4))
        with pytest.raises(ValueError, match="Length mismatch"):
            handler.upload_embedding_array("m", emb, ["a", "b"])

    def test_unsupported_format(self, handler):
        emb = np.zeros((2, 4))
        with pytest.raises(ValueError, match="Unsupported format"):
            handler.upload_embedding_array("m", emb, ["a", "b"], fmt="hdf5")


# =====================================================================
# Discovery
# =====================================================================


class TestAvailableDatasets:
    @patch("embpy.pp.hf_handler.list_repo_files")
    def test_mixed_formats(self, mock_list, handler):
        mock_list.return_value = [
            "raw/tahoe.h5ad",
            "raw/lincs.csv",
            "raw/gdsc.parquet",
            "embeddings/model.parquet",
            "README.md",
        ]
        result = handler.available_datasets()
        assert sorted(result) == ["gdsc", "lincs", "tahoe"]


class TestAvailableEmbeddings:
    @patch("embpy.pp.hf_handler.list_repo_files")
    def test_detects_formats_and_ignores_index(self, mock_list, handler):
        mock_list.return_value = [
            "embeddings/chemberta.parquet",
            "embeddings/minimol.npy",
            "embeddings/minimol_index.csv",
            "embeddings/mole.npz",
            "raw/tahoe.h5ad",
        ]
        result = handler.available_embeddings()
        assert sorted(result) == ["chemberta", "minimol", "mole"]


# =====================================================================
# Download — datasets
# =====================================================================


class TestResolveRawFilename:
    @patch("embpy.pp.hf_handler.list_repo_files")
    def test_prefers_h5ad(self, mock_list, handler):
        mock_list.return_value = ["raw/ds.h5ad", "raw/ds.csv"]
        assert handler._resolve_raw_filename("ds") == "raw/ds.h5ad"

    @patch("embpy.pp.hf_handler.list_repo_files")
    def test_fallback_to_csv(self, mock_list, handler):
        mock_list.return_value = ["raw/ds.csv"]
        assert handler._resolve_raw_filename("ds") == "raw/ds.csv"

    @patch("embpy.pp.hf_handler.list_repo_files")
    def test_not_found(self, mock_list, handler):
        mock_list.return_value = ["raw/other.h5ad"]
        with pytest.raises(FileNotFoundError, match="not found in repo"):
            handler._resolve_raw_filename("missing")


class TestDownloadDataset:
    def test_h5ad_as_anndata(self, handler, tmp_dir, sample_adata):
        h5ad_path = tmp_dir / "tahoe.h5ad"
        sample_adata.write_h5ad(h5ad_path)

        handler._resolve_raw_filename = MagicMock(return_value="raw/tahoe.h5ad")
        handler._download_file = MagicMock(return_value=h5ad_path)

        result = handler.download_dataset("tahoe")
        assert isinstance(result, AnnData)
        assert result.shape == sample_adata.shape

    def test_csv_as_anndata(self, handler, tmp_dir):
        csv_path = tmp_dir / "lincs.csv"
        pd.DataFrame({"A": [1, 2]}, index=pd.Index(["r0", "r1"])).to_csv(csv_path)

        handler._resolve_raw_filename = MagicMock(return_value="raw/lincs.csv")
        handler._download_file = MagicMock(return_value=csv_path)

        result = handler.download_dataset("lincs", as_anndata=True)
        assert isinstance(result, AnnData)

    def test_csv_as_dataframe(self, handler, tmp_dir):
        csv_path = tmp_dir / "lincs.csv"
        pd.DataFrame({"A": [1, 2]}).to_csv(csv_path, index=False)

        handler._resolve_raw_filename = MagicMock(return_value="raw/lincs.csv")
        handler._download_file = MagicMock(return_value=csv_path)

        result = handler.download_dataset("lincs", as_anndata=False)
        assert isinstance(result, pd.DataFrame)
        assert "A" in result.columns


class TestDownloadAllDatasets:
    @patch("embpy.pp.hf_handler.list_repo_files")
    def test_downloads_all(self, mock_list, handler, tmp_dir, sample_adata):
        mock_list.return_value = ["raw/a.h5ad", "raw/b.h5ad"]

        path_a = tmp_dir / "a.h5ad"
        path_b = tmp_dir / "b.h5ad"
        sample_adata.write_h5ad(path_a)
        sample_adata.write_h5ad(path_b)

        def fake_download(filename, cache_dir=None):
            if "a.h5ad" in filename:
                return path_a
            return path_b

        handler._download_file = MagicMock(side_effect=fake_download)

        result = handler.download_all_datasets()
        assert set(result.keys()) == {"a", "b"}
        assert all(isinstance(v, AnnData) for v in result.values())


# =====================================================================
# Download — partial (obs / var)
# =====================================================================


class TestDownloadObs:
    def test_returns_obs(self, handler, tmp_dir, sample_adata):
        path = tmp_dir / "ds.h5ad"
        sample_adata.write_h5ad(path)
        handler._resolve_raw_filename = MagicMock(return_value="raw/ds.h5ad")
        handler._download_file = MagicMock(return_value=path)

        obs = handler.download_obs("ds")
        assert isinstance(obs, pd.DataFrame)
        assert "drug" in obs.columns
        assert "dose" in obs.columns

    def test_column_filter(self, handler, tmp_dir, sample_adata):
        path = tmp_dir / "ds.h5ad"
        sample_adata.write_h5ad(path)
        handler._resolve_raw_filename = MagicMock(return_value="raw/ds.h5ad")
        handler._download_file = MagicMock(return_value=path)

        obs = handler.download_obs("ds", columns=["drug"])
        assert list(obs.columns) == ["drug"]

    def test_warns_on_missing_column(self, handler, tmp_dir, sample_adata, caplog):
        path = tmp_dir / "ds.h5ad"
        sample_adata.write_h5ad(path)
        handler._resolve_raw_filename = MagicMock(return_value="raw/ds.h5ad")
        handler._download_file = MagicMock(return_value=path)

        import logging

        with caplog.at_level(logging.WARNING):
            obs = handler.download_obs("ds", columns=["drug", "nonexistent"])
        assert list(obs.columns) == ["drug"]


class TestDownloadVar:
    def test_returns_var(self, handler, tmp_dir, sample_adata):
        path = tmp_dir / "ds.h5ad"
        sample_adata.write_h5ad(path)
        handler._resolve_raw_filename = MagicMock(return_value="raw/ds.h5ad")
        handler._download_file = MagicMock(return_value=path)

        var = handler.download_var("ds")
        assert isinstance(var, pd.DataFrame)
        assert "gene_name" in var.columns


# =====================================================================
# Download — metadata
# =====================================================================


class TestDownloadMetadata:
    def test_reads_parquet(self, handler, tmp_dir):
        df = pd.DataFrame({"drug": ["aspirin"], "smiles": ["CC(=O)O"]})
        path = tmp_dir / "perturbations.parquet"
        df.to_parquet(path, index=False)

        handler._download_file = MagicMock(return_value=path)
        result = handler.download_metadata()
        assert list(result.columns) == ["drug", "smiles"]
        assert len(result) == 1


# =====================================================================
# Download — embeddings
# =====================================================================


class TestResolveEmbeddingFilename:
    @patch("embpy.pp.hf_handler.list_repo_files")
    def test_prefers_parquet(self, mock_list, handler):
        mock_list.return_value = [
            "embeddings/model.parquet",
            "embeddings/model.npy",
        ]
        assert handler._resolve_embedding_filename("model") == "embeddings/model.parquet"

    @patch("embpy.pp.hf_handler.list_repo_files")
    def test_fallback_to_npy(self, mock_list, handler):
        mock_list.return_value = ["embeddings/model.npy"]
        assert handler._resolve_embedding_filename("model") == "embeddings/model.npy"

    @patch("embpy.pp.hf_handler.list_repo_files")
    def test_not_found(self, mock_list, handler):
        mock_list.return_value = []
        with pytest.raises(FileNotFoundError, match="not found"):
            handler._resolve_embedding_filename("missing")


class TestDownloadEmbeddingParquet:
    def test_returns_dataframe(self, handler, tmp_dir):
        df = pd.DataFrame({
            "smiles": ["CCO", "O"],
            "embedding": [[0.1, 0.2], [0.3, 0.4]],
        })
        path = tmp_dir / "chemberta.parquet"
        df.to_parquet(path, index=False)

        handler._resolve_embedding_filename = MagicMock(return_value="embeddings/chemberta.parquet")
        handler._download_file = MagicMock(return_value=path)

        result = handler.download_embedding("chemberta")
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["smiles", "embedding"]
        assert len(result) == 2


class TestDownloadEmbeddingNpy:
    def test_with_index(self, handler, tmp_dir):
        arr = np.random.randn(3, 8).astype(np.float32)
        arr_path = tmp_dir / "minimol.npy"
        idx_path = tmp_dir / "minimol_index.csv"
        np.save(arr_path, arr)
        pd.DataFrame({"id": ["smi1", "smi2", "smi3"]}).to_csv(idx_path, index=False)

        call_map = {
            "embeddings/minimol.npy": arr_path,
            "embeddings/minimol_index.csv": idx_path,
        }
        handler._resolve_embedding_filename = MagicMock(return_value="embeddings/minimol.npy")
        handler._download_file = MagicMock(side_effect=lambda f, **kw: call_map[f])

        result = handler.download_embedding("minimol")
        assert isinstance(result, dict)
        assert result["embeddings"].shape == (3, 8)
        assert list(result["index"]["id"]) == ["smi1", "smi2", "smi3"]

    def test_without_index(self, handler, tmp_dir):
        arr = np.random.randn(2, 4).astype(np.float32)
        arr_path = tmp_dir / "model.npy"
        np.save(arr_path, arr)

        handler._resolve_embedding_filename = MagicMock(return_value="embeddings/model.npy")

        def fake_download(filename, **kw):
            if filename.endswith(".npy"):
                return arr_path
            raise FileNotFoundError("no index")

        handler._download_file = MagicMock(side_effect=fake_download)

        result = handler.download_embedding("model")
        assert isinstance(result, dict)
        assert result["embeddings"].shape == (2, 4)
        assert len(result["index"]) == 2


class TestDownloadEmbeddingNpz:
    def test_returns_dict(self, handler, tmp_dir):
        path = tmp_dir / "mole.npz"
        np.savez(path, embeddings=np.zeros((5, 16)), ids=np.array(["a", "b", "c", "d", "e"]))

        handler._resolve_embedding_filename = MagicMock(return_value="embeddings/mole.npz")
        handler._download_file = MagicMock(return_value=path)

        result = handler.download_embedding("mole")
        assert isinstance(result, dict)
        assert "embeddings" in result
        assert result["embeddings"].shape == (5, 16)


class TestDownloadAllEmbeddings:
    @patch("embpy.pp.hf_handler.list_repo_files")
    def test_downloads_all(self, mock_list, handler, tmp_dir):
        mock_list.return_value = [
            "embeddings/a.parquet",
            "embeddings/b.parquet",
        ]

        df = pd.DataFrame({"smiles": ["CCO"], "embedding": [[0.1]]})
        path_a = tmp_dir / "a.parquet"
        path_b = tmp_dir / "b.parquet"
        df.to_parquet(path_a, index=False)
        df.to_parquet(path_b, index=False)

        def fake_download(filename, **kw):
            if "a.parquet" in filename:
                return path_a
            return path_b

        handler._download_file = MagicMock(side_effect=fake_download)

        result = handler.download_all_embeddings()
        assert set(result.keys()) == {"a", "b"}
        assert all(isinstance(v, pd.DataFrame) for v in result.values())
