"""Disk-backed cache for action embeddings.

File layout::

    {cache_dir}/{model_name}/{region}_{pooling}_{organism}.npz
    {cache_dir}/{model_name}/{region}_{pooling}_{organism}.npz.lock

Each NPZ archive carries two arrays:

* ``symbols``    : object dtype, ``(N,)``
* ``embeddings`` : float32, ``(N, D)``

Concurrency: ``fcntl.flock`` over a sibling ``.lock`` file serialises
concurrent writers within and across processes. On non-POSIX platforms
the lock is a no-op and an info-level log line documents the race
window (last writer wins; symbols are still merged so no data is
permanently lost as long as the writes are not strictly simultaneous).

Atomicity: writes go to a sibling temp file and ``os.replace`` swaps
it into place at the end -- readers therefore always see either the
old archive or the new one, never a half-written file.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingCacheKey:
    """Unique identifier for an action-embedding cache file."""

    model_name: str
    region: str = "full"
    pooling_strategy: str = "mean"
    organism: str = "human"

    def relative_path(self) -> Path:
        return Path(self.model_name) / f"{self.region}_{self.pooling_strategy}_{self.organism}.npz"


# ---------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------


@contextmanager
def _file_lock(lock_path: Path):
    """Best-effort cross-process advisory lock.

    Uses ``fcntl.flock`` on POSIX. On non-POSIX we yield without
    locking and log once per process so the user knows the cache is
    last-writer-wins.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl  # noqa: PLC0415

        fp = open(lock_path, "w")
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
            finally:
                fp.close()
    except ImportError:
        if not _file_lock._warned:  # type: ignore[attr-defined]
            logger.warning(
                "fcntl not available -- embedding cache writes are last-writer-wins."
            )
            _file_lock._warned = True  # type: ignore[attr-defined]
        yield


_file_lock._warned = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------
# Read / write helpers
# ---------------------------------------------------------------------


def _archive_path(cache_dir: str | Path, key: EmbeddingCacheKey) -> Path:
    return Path(cache_dir) / key.relative_path()


def _read_archive(path: Path) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        return [], np.zeros((0, 0), dtype=np.float32)
    archive = np.load(path, allow_pickle=True)
    symbols = [str(s) for s in archive["symbols"]]
    embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    return symbols, embeddings


def _atomic_write(path: Path, symbols: Sequence[str], embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem + ".", suffix=".tmp.npz")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez(
            tmp_path,
            symbols=np.asarray(list(symbols), dtype=object),
            embeddings=np.asarray(embeddings, dtype=np.float32),
        )
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------
# Public API used by BioEmbedderProvider
# ---------------------------------------------------------------------


def load_cached(
    cache_dir: str | Path,
    key: EmbeddingCacheKey,
    symbols: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return ``{symbol: embedding}`` for symbols already on disk."""
    path = _archive_path(cache_dir, key)
    cached_symbols, cached_emb = _read_archive(path)
    if not cached_symbols:
        return {}
    by_symbol = {s: cached_emb[i] for i, s in enumerate(cached_symbols)}
    requested = set(symbols)
    return {s: by_symbol[s] for s in requested if s in by_symbol}


def save_cached(
    cache_dir: str | Path,
    key: EmbeddingCacheKey,
    symbols: Sequence[str],
    embeddings: np.ndarray,
) -> Path:
    """Merge ``(symbols, embeddings)`` into the archive at ``cache_dir / key``.

    Existing rows for the same symbols are *overwritten* with the new
    values (the assumption is that the most recent computation wins).
    """
    if embeddings.ndim != 2 or embeddings.shape[0] != len(symbols):
        raise ValueError(
            f"embeddings shape {embeddings.shape} inconsistent with {len(symbols)} symbols"
        )
    path = _archive_path(cache_dir, key)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _file_lock(lock_path):
        old_symbols, old_emb = _read_archive(path)
        # Deduplicate: the new entries take precedence on conflict.
        merged: dict[str, np.ndarray] = {}
        if old_emb.size and old_emb.shape[1] != embeddings.shape[1]:
            logger.warning(
                "Cache at %s has dim %d, incoming dim %d; rewriting from scratch.",
                path, old_emb.shape[1], embeddings.shape[1],
            )
            old_symbols, old_emb = [], np.zeros((0, 0), dtype=np.float32)
        for i, s in enumerate(old_symbols):
            merged[s] = old_emb[i]
        for i, s in enumerate(symbols):
            merged[s] = np.asarray(embeddings[i], dtype=np.float32)
        merged_syms = sorted(merged.keys())
        merged_emb = np.stack([merged[s] for s in merged_syms], axis=0)
        _atomic_write(path, merged_syms, merged_emb)
    return path


__all__ = ["EmbeddingCacheKey", "load_cached", "save_cached"]
