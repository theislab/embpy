"""Disk + in-memory cell-embedding cache for foundation backbones.

Layout on disk::

    {cache_dir}/{backbone}/{ckpt_hash}/{dataset_hash}.npz

Each NPZ stores:

* ``embeddings`` -- ``(n_cells, embedding_dim)`` float32
* ``meta``       -- 0-d object array holding a JSON string with the
  dataclass :class:`CacheMeta`. Stored this way (rather than as
  separate keys) so a single ``np.load`` returns everything and the
  metadata survives ``np.savez`` round-tripping.

Atomicity
---------
``save_cached`` writes to a ``*.tmp`` sibling then ``os.replace``s it
into place. If :mod:`filelock` is available it is used to serialise
concurrent writers (multiple SLURM array tasks racing on the same
NPZ); when unavailable we fall back to lock-free atomic rename, which
on POSIX is itself atomic per ``rename(2)``. The remaining race window
is the tiny gap between two writers detecting *no file exists* and
both running ``encode(...)``; both succeed but only one wins the
rename. We accept that wasted compute rather than adding a hard
dependency on :mod:`filelock`.

CLI inspect
-----------
::

    python -m embpy.world_model.models.encoders.backbones.cache \\
        --inspect outputs/_cache/state_backbone
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = "1"


@dataclass
class CacheMeta:
    backbone: str
    ckpt_hash: str
    dataset_hash: str
    adata_hash: str
    n_cells: int
    embedding_dim: int
    created_at: str
    version: str = CACHE_FORMAT_VERSION


def _key_path(cache_dir: Path, backbone: str, ckpt_hash: str, dataset_hash: str) -> Path:
    return Path(cache_dir) / backbone / ckpt_hash / f"{dataset_hash}.npz"


def cache_path_for(
    cache_dir: str | os.PathLike[str],
    *,
    backbone: str,
    ckpt_hash: str,
    dataset_hash: str,
) -> Path:
    """Return the NPZ path that would store the given key."""
    return _key_path(Path(cache_dir), backbone, ckpt_hash, dataset_hash)


def load_cached(
    cache_dir: str | os.PathLike[str],
    *,
    backbone: str,
    ckpt_hash: str,
    dataset_hash: str,
) -> tuple[np.ndarray, CacheMeta] | None:
    """Return ``(embeddings, meta)`` if cached, else ``None``."""
    path = _key_path(Path(cache_dir), backbone, ckpt_hash, dataset_hash)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            meta_json = str(np.asarray(data["meta"]).item())
        meta = CacheMeta(**json.loads(meta_json))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Cache file %s is unreadable (%s); ignoring.", path, exc)
        return None
    return embeddings, meta


def save_cached(
    cache_dir: str | os.PathLike[str],
    *,
    backbone: str,
    ckpt_hash: str,
    dataset_hash: str,
    embeddings: np.ndarray,
    adata_hash: str,
) -> Path:
    """Atomically write ``embeddings`` to the cache and return the path."""
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    path = _key_path(Path(cache_dir), backbone, ckpt_hash, dataset_hash)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = CacheMeta(
        backbone=backbone,
        ckpt_hash=ckpt_hash,
        dataset_hash=dataset_hash,
        adata_hash=adata_hash,
        n_cells=int(embeddings.shape[0]),
        embedding_dim=int(embeddings.shape[1]),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    try:
        from filelock import FileLock  # type: ignore[import-not-found]

        lock_ctx: Any = FileLock(str(path) + ".lock", timeout=600)
    except ImportError:
        lock_ctx = _NullLock()

    arr = np.asarray(embeddings, dtype=np.float32)
    meta_arr = np.asarray(json.dumps(asdict(meta)), dtype=object)
    with lock_ctx:
        if path.exists():
            return path
        tmp = tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False,
        )
        tmp_name = tmp.name
        try:
            np.savez(tmp, embeddings=arr, meta=meta_arr)
            tmp.close()
            os.replace(tmp_name, path)
        except Exception:
            tmp.close()
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise
    return path


class _NullLock:
    def __enter__(self) -> "_NullLock":
        return self

    def __exit__(self, *_: Any) -> None:
        return None


def inspect_cache(cache_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Walk ``cache_dir`` and return one record per cached NPZ."""
    root = Path(cache_dir)
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    for npz in sorted(root.rglob("*.npz")):
        size = npz.stat().st_size
        try:
            with np.load(npz, allow_pickle=True) as data:
                meta_json = str(np.asarray(data["meta"]).item())
                meta = json.loads(meta_json)
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            meta = {}
        records.append(
            {
                "path": str(npz),
                "size_bytes": int(size),
                "backbone": meta.get("backbone", "?"),
                "ckpt_hash": meta.get("ckpt_hash", "?"),
                "dataset_hash": meta.get("dataset_hash", "?"),
                "n_cells": meta.get("n_cells", -1),
                "embedding_dim": meta.get("embedding_dim", -1),
                "created_at": meta.get("created_at", ""),
            }
        )
    return records


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Inspect a state-backbone embedding cache.")
    p.add_argument("--inspect", required=True, help="Path to the cache directory.")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = p.parse_args(argv)

    records = inspect_cache(args.inspect)
    total_size = sum(r["size_bytes"] for r in records)

    if args.json:
        json.dump({"records": records, "total_bytes": total_size}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if not records:
        print(f"No cached embeddings under {args.inspect}.")
        return 0
    header = f"{'BACKBONE':<10} {'N_CELLS':>10} {'DIM':>6} {'SIZE_MB':>10}  PATH"
    print(header)
    print("-" * len(header))
    for r in records:
        size_mb = r["size_bytes"] / (1024 * 1024)
        print(
            f"{r['backbone']:<10} {r['n_cells']:>10} {r['embedding_dim']:>6} "
            f"{size_mb:>10.2f}  {r['path']}"
        )
    print("-" * len(header))
    print(f"{'TOTAL':<10} {'':>10} {'':>6} {total_size / (1024 * 1024):>10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "CACHE_FORMAT_VERSION",
    "CacheMeta",
    "cache_path_for",
    "inspect_cache",
    "load_cached",
    "save_cached",
]
