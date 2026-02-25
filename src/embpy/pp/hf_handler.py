"""Hugging Face dataset handler for perturbation data.

Provides :class:`HFHandler`, a high-level wrapper around ``huggingface_hub``
for uploading and downloading perturbation datasets, metadata, and
pre-computed embeddings stored on a Hugging Face dataset repository.

Supported file formats
~~~~~~~~~~~~~~~~~~~~~~

* **AnnData** (``.h5ad``) — preferred for perturbation data.
* **CSV** (``.csv``) — tabular data; converted to AnnData on download when
  possible.
* **Parquet** (``.parquet``) — efficient columnar storage for metadata and
  embeddings.
* **NumPy tensors** (``.npy`` / ``.npz``) — raw embedding arrays with an
  accompanying row-index file (``_index.csv``) that maps rows to
  identifiers.

Users can operate on **specific datasets** or **all datasets** in the repo.
Partial downloads (e.g. only ``.obs``, ``.var``, or embedding columns) are
supported so that large expression matrices need not be fetched when only
metadata is needed.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from anndata import AnnData
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_RAW_EXTENSIONS = frozenset({".h5ad", ".csv", ".parquet"})
_EMBEDDING_EXTENSIONS = frozenset({".parquet", ".npy", ".npz"})


# ------------------------------------------------------------------
# File-format readers
# ------------------------------------------------------------------


def _read_to_anndata(path: Path) -> AnnData:
    """Read a local file into an AnnData, dispatching on extension."""
    import anndata as ad

    ext = path.suffix.lower()
    if ext == ".h5ad":
        return ad.read_h5ad(path)
    if ext == ".csv":
        df = pd.read_csv(path, index_col=0)
        return AnnData(df)
    if ext == ".parquet":
        df = pd.read_parquet(path)
        if df.columns[0] == df.index.name or df.columns[0] in ("index", ""):
            df = df.set_index(df.columns[0])
        return AnnData(df)
    raise ValueError(f"Unsupported file extension {ext!r} for AnnData conversion.")


def _read_tabular(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported tabular extension {ext!r}. Use .csv or .parquet.")


# ------------------------------------------------------------------
# Handler class
# ------------------------------------------------------------------


class HFHandler:
    """Upload and download perturbation data from a Hugging Face dataset repo.

    The handler is **dataset-agnostic**: any ``.h5ad``, ``.csv``, or
    ``.parquet`` file placed under ``raw/`` is treated as a dataset.
    There is no hardcoded list of expected dataset names — the repo
    contents are discovered at runtime via :meth:`available_datasets`.

    Parameters
    ----------
    repo_name
        Hugging Face dataset repository identifier, e.g.
        ``"your-username/perturbation-embeddings"``.
    token
        Optional HF API token.  When ``None`` the token cached by
        ``huggingface-cli login`` is used.

    Examples
    --------
    >>> hf = HFHandler("user/perturbation-embeddings")
    >>> hf.upload_datasets({"tahoe": "data/tahoe.h5ad"})
    >>> adata = hf.download_dataset("tahoe")
    """

    def __init__(self, repo_name: str, token: str | None = None) -> None:
        self.repo_name = repo_name
        self.token = token
        self._api = HfApi(token=token)

    # ------------------------------------------------------------------
    # Repository helpers
    # ------------------------------------------------------------------

    def create_repo(self, *, private: bool = False) -> str:
        """Create the dataset repo if it does not already exist.

        Returns
        -------
        URL of the created (or existing) repository.
        """
        url = self._api.create_repo(
            self.repo_name,
            repo_type="dataset",
            exist_ok=True,
            private=private,
        )
        logger.info("Repo ready: %s", url)
        return str(url)

    def list_files(self) -> list[str]:
        """List all files in the remote repository."""
        return list(list_repo_files(self.repo_name, repo_type="dataset", token=self.token))

    # ══════════════════════════════════════════════════════════════════
    # Upload
    # ══════════════════════════════════════════════════════════════════

    def upload_file(self, local_path: str | Path, path_in_repo: str) -> None:
        """Upload a single file to the repo."""
        self._api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=self.repo_name,
            repo_type="dataset",
        )
        logger.info("Uploaded %s → %s", local_path, path_in_repo)

    # ── raw datasets ─────────────────────────────────────────────────

    def upload_datasets(
        self,
        dataset_paths: dict[str, str | Path],
    ) -> None:
        """Upload dataset files to ``raw/`` in the repo.

        Accepts ``.h5ad`` (recommended), ``.csv``, and ``.parquet`` files.
        The original extension is preserved in the repo.  Any dataset
        name is valid — there is no fixed list.

        Parameters
        ----------
        dataset_paths
            Mapping of dataset name to local file path, e.g.
            ``{"tahoe": "/data/tahoe.h5ad", "my_screen": "/data/screen.csv"}``.
        """
        for name, path in dataset_paths.items():
            p = Path(path)
            if not p.exists():
                logger.warning("Skipping %s — file not found: %s", name, p)
                continue
            if p.suffix.lower() not in _RAW_EXTENSIONS:
                logger.warning(
                    "Skipping %s — unsupported format %s (use .h5ad, .csv, or .parquet)",
                    name,
                    p.suffix,
                )
                continue
            self.upload_file(p, f"raw/{name}{p.suffix}")

    def upload_anndata(self, adata: AnnData, name: str) -> None:
        """Write an AnnData to a temporary ``.h5ad`` and upload to ``raw/``.

        Parameters
        ----------
        adata
            AnnData object to upload.
        name
            Dataset name used as the filename stem in the repo.
        """
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            adata.write_h5ad(f.name)
            self.upload_file(f.name, f"raw/{name}.h5ad")

    # ── metadata ─────────────────────────────────────────────────────

    def upload_metadata(self, metadata: pd.DataFrame | str | Path) -> None:
        """Upload a perturbation metadata table to ``metadata/perturbations.parquet``.

        Parameters
        ----------
        metadata
            A DataFrame, or path to an existing Parquet/CSV file.
        """
        if isinstance(metadata, pd.DataFrame):
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                metadata.to_parquet(f.name, index=False)
                self.upload_file(f.name, "metadata/perturbations.parquet")
        else:
            self.upload_file(metadata, "metadata/perturbations.parquet")

    # ── embeddings ───────────────────────────────────────────────────

    def upload_embeddings(self, embeddings_dir: str | Path) -> None:
        """Upload embedding files from a local directory to ``embeddings/``.

        Supports ``.parquet``, ``.npy``, and ``.npz`` files.

        Parameters
        ----------
        embeddings_dir
            Directory containing embedding files (one per model).
        """
        emb_dir = Path(embeddings_dir)
        if not emb_dir.is_dir():
            raise FileNotFoundError(f"Embeddings directory not found: {emb_dir}")
        for fpath in sorted(emb_dir.iterdir()):
            if fpath.suffix.lower() in _EMBEDDING_EXTENSIONS or fpath.name.endswith("_index.csv"):
                self.upload_file(fpath, f"embeddings/{fpath.name}")

    def upload_embedding_array(
        self,
        model_key: str,
        embeddings: np.ndarray,
        identifiers: Sequence[str],
        *,
        fmt: str = "parquet",
    ) -> None:
        """Upload an embedding matrix directly from numpy arrays.

        Parameters
        ----------
        model_key
            Model name used as the filename stem.
        embeddings
            2-D array of shape ``(n_molecules, emb_dim)``.
        identifiers
            Row identifiers (e.g. SMILES strings), same length as
            *embeddings*.
        fmt
            Output format: ``"parquet"`` (default) stores identifiers and
            embedding vectors together; ``"npy"`` saves the array as
            ``.npy`` with a companion ``_index.csv``.
        """
        if embeddings.shape[0] != len(identifiers):
            raise ValueError(
                f"Length mismatch: embeddings has {embeddings.shape[0]} rows "
                f"but {len(identifiers)} identifiers given."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            if fmt == "parquet":
                df = pd.DataFrame({
                    "smiles": list(identifiers),
                    "embedding": [row.tolist() for row in embeddings],
                })
                out = tmp / f"{model_key}.parquet"
                df.to_parquet(out, index=False)
                self.upload_file(out, f"embeddings/{model_key}.parquet")
            elif fmt == "npy":
                arr_path = tmp / f"{model_key}.npy"
                idx_path = tmp / f"{model_key}_index.csv"
                np.save(arr_path, embeddings)
                pd.DataFrame({"id": list(identifiers)}).to_csv(idx_path, index=False)
                self.upload_file(arr_path, f"embeddings/{model_key}.npy")
                self.upload_file(idx_path, f"embeddings/{model_key}_index.csv")
            else:
                raise ValueError(f"Unsupported format {fmt!r}. Use 'parquet' or 'npy'.")

    # ══════════════════════════════════════════════════════════════════
    # Download
    # ══════════════════════════════════════════════════════════════════

    def _download_file(self, filename: str, cache_dir: str | Path | None = None) -> Path:
        """Download a single file from the repo, returning the local path."""
        local = hf_hub_download(
            repo_id=self.repo_name,
            filename=filename,
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir else None,
            token=self.token,
        )
        return Path(local)

    # ── discovery ────────────────────────────────────────────────────

    def available_datasets(self) -> list[str]:
        """Return names of dataset files in ``raw/``.

        Recognises ``.h5ad``, ``.csv``, and ``.parquet`` extensions.
        Any dataset name is valid — the list is built dynamically from
        the repo contents.
        """
        out: list[str] = []
        for f in self.list_files():
            if not f.startswith("raw/"):
                continue
            stem = f.removeprefix("raw/")
            for ext in (".h5ad", ".csv", ".parquet"):
                if stem.endswith(ext):
                    out.append(stem.removesuffix(ext))
                    break
        return out

    def available_embeddings(self) -> list[str]:
        """Return model keys whose embeddings are in the repo.

        Detects ``.parquet``, ``.npy``, and ``.npz`` files under
        ``embeddings/``, ignoring companion ``_index.csv`` files.
        """
        out: list[str] = []
        for f in self.list_files():
            if not f.startswith("embeddings/"):
                continue
            stem = f.removeprefix("embeddings/")
            if stem.endswith("_index.csv"):
                continue
            for ext in (".parquet", ".npy", ".npz"):
                if stem.endswith(ext):
                    out.append(stem.removesuffix(ext))
                    break
        return out

    # ── datasets ─────────────────────────────────────────────────────

    def _resolve_raw_filename(self, name: str) -> str:
        """Find the actual filename in ``raw/`` for a given dataset name."""
        files = self.list_files()
        for ext in (".h5ad", ".parquet", ".csv"):
            candidate = f"raw/{name}{ext}"
            if candidate in files:
                return candidate
        raise FileNotFoundError(
            f"Dataset {name!r} not found in repo. "
            f"Available: {self.available_datasets()}"
        )

    def download_dataset(
        self,
        name: str,
        *,
        as_anndata: bool = True,
        cache_dir: str | Path | None = None,
    ) -> AnnData | pd.DataFrame:
        """Download a single dataset.

        Parameters
        ----------
        name
            Dataset name (e.g. ``"tahoe"``).  Any name present in the
            repo under ``raw/`` is accepted.
        as_anndata
            If ``True`` (default), the file is loaded as an
            :class:`~anndata.AnnData` regardless of its on-disk format.
            If ``False``, ``.csv`` and ``.parquet`` files are returned as
            DataFrames.
        cache_dir
            Local cache directory.  Defaults to the HF Hub cache.

        Returns
        -------
        AnnData (preferred) or DataFrame depending on *as_anndata* and
        file format.
        """
        remote = self._resolve_raw_filename(name)
        path = self._download_file(remote, cache_dir=cache_dir)
        logger.info("Reading %s", path)

        if as_anndata:
            return _read_to_anndata(path)
        ext = path.suffix.lower()
        if ext == ".h5ad":
            import anndata as ad

            return ad.read_h5ad(path)
        return _read_tabular(path)

    def download_all_datasets(
        self,
        *,
        as_anndata: bool = True,
        cache_dir: str | Path | None = None,
    ) -> dict[str, AnnData | pd.DataFrame]:
        """Download all dataset files in ``raw/``.

        Returns
        -------
        Dictionary mapping dataset name to AnnData (or DataFrame).
        """
        names = self.available_datasets()
        logger.info("Found %d datasets: %s", len(names), names)
        return {
            name: self.download_dataset(name, as_anndata=as_anndata, cache_dir=cache_dir)
            for name in names
        }

    # ── partial: obs / var ───────────────────────────────────────────

    def download_obs(
        self,
        name: str,
        *,
        columns: Sequence[str] | None = None,
        cache_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        """Download only the ``.obs`` DataFrame from a dataset.

        Parameters
        ----------
        name
            Dataset name.
        columns
            Subset of ``.obs`` columns to return.  ``None`` returns all.
        cache_dir
            Local cache directory.

        Returns
        -------
        The observation metadata as a DataFrame.
        """
        result = self.download_dataset(name, as_anndata=True, cache_dir=cache_dir)
        if not isinstance(result, AnnData):
            raise TypeError(f"Cannot extract .obs — dataset {name!r} is not AnnData-compatible.")
        df = pd.DataFrame(result.obs)
        if columns is not None:
            missing = set(columns) - set(df.columns)
            if missing:
                logger.warning("Columns not found in .obs: %s", missing)
            df = pd.DataFrame(df[[c for c in columns if c in df.columns]])
        return df

    def download_var(
        self,
        name: str,
        *,
        columns: Sequence[str] | None = None,
        cache_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        """Download only the ``.var`` DataFrame from a dataset.

        Parameters
        ----------
        name
            Dataset name.
        columns
            Subset of ``.var`` columns to return.  ``None`` returns all.
        cache_dir
            Local cache directory.

        Returns
        -------
        The variable metadata as a DataFrame.
        """
        result = self.download_dataset(name, as_anndata=True, cache_dir=cache_dir)
        if not isinstance(result, AnnData):
            raise TypeError(f"Cannot extract .var — dataset {name!r} is not AnnData-compatible.")
        df = pd.DataFrame(result.var)
        if columns is not None:
            missing = set(columns) - set(df.columns)
            if missing:
                logger.warning("Columns not found in .var: %s", missing)
            df = pd.DataFrame(df[[c for c in columns if c in df.columns]])
        return df

    # ── metadata ─────────────────────────────────────────────────────

    def download_metadata(
        self,
        *,
        cache_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        """Download the perturbation metadata table.

        Returns
        -------
        DataFrame from ``metadata/perturbations.parquet``.
        """
        path = self._download_file("metadata/perturbations.parquet", cache_dir=cache_dir)
        return pd.read_parquet(path)

    # ── embeddings ───────────────────────────────────────────────────

    def _resolve_embedding_filename(self, model_key: str) -> str:
        """Find the actual embedding filename for a model key."""
        files = self.list_files()
        for ext in (".parquet", ".npy", ".npz"):
            candidate = f"embeddings/{model_key}{ext}"
            if candidate in files:
                return candidate
        raise FileNotFoundError(
            f"Embeddings for {model_key!r} not found. "
            f"Available: {self.available_embeddings()}"
        )

    def download_embedding(
        self,
        model_key: str,
        *,
        cache_dir: str | Path | None = None,
    ) -> pd.DataFrame | dict[str, Any]:
        """Download embeddings for a single model.

        Parameters
        ----------
        model_key
            Model name, e.g. ``"chemberta2MTR"``.
        cache_dir
            Local cache directory.

        Returns
        -------
        For ``.parquet`` files: DataFrame with columns
        ``["smiles", "embedding"]``.

        For ``.npy`` files: a dict ``{"embeddings": np.ndarray,
        "index": pd.DataFrame}`` where *index* contains the row
        identifiers from the companion ``_index.csv``.

        For ``.npz`` files: a dict with all stored arrays.
        """
        remote = self._resolve_embedding_filename(model_key)
        path = self._download_file(remote, cache_dir=cache_dir)
        ext = path.suffix.lower()

        if ext == ".parquet":
            return pd.read_parquet(path)

        if ext == ".npy":
            arr = np.load(path)
            idx_remote = f"embeddings/{model_key}_index.csv"
            try:
                idx_path = self._download_file(idx_remote, cache_dir=cache_dir)
                index = pd.read_csv(idx_path)
            except Exception:  # noqa: BLE001
                logger.warning("No index file for %s — returning array only.", model_key)
                index = pd.DataFrame({"id": np.arange(arr.shape[0])})
            return {"embeddings": arr, "index": index}

        if ext == ".npz":
            return dict(np.load(path))  # type: ignore[arg-type]

        raise ValueError(f"Unsupported embedding format: {ext}")

    def download_all_embeddings(
        self,
        *,
        cache_dir: str | Path | None = None,
    ) -> dict[str, pd.DataFrame | dict[str, Any]]:
        """Download all embedding files.

        Returns
        -------
        Dictionary mapping model key to its embeddings (DataFrame or dict).
        """
        keys = self.available_embeddings()
        logger.info("Found %d embedding models: %s", len(keys), keys)
        return {k: self.download_embedding(k, cache_dir=cache_dir) for k in keys}

    # ══════════════════════════════════════════════════════════════════
    # Convenience
    # ══════════════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        return f"HFHandler(repo_name={self.repo_name!r})"
