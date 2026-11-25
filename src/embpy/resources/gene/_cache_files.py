"""File helpers and settings shared by embpy's gene caches (gene ids and symbol aliases).

Standard library only, so the lightweight alias resolver can use it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

SAVE_EVERY_ENV = "EMBPY_GENE_CACHE_SAVE_EVERY"
"""Environment variable: save a gene cache after this many new entries."""

_warned_values: set[str] = set()


def save_every(default: int) -> int:
    """Number of new entries after which a gene cache file is saved.

    ``EMBPY_GENE_CACHE_SAVE_EVERY`` overrides ``default`` when it holds a
    positive whole number (``1`` saves after every new entry); any other value
    is ignored with a warning.
    """
    raw = os.environ.get(SAVE_EVERY_ENV, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value >= 1:
        return value
    if raw not in _warned_values:
        _warned_values.add(raw)
        logging.warning(
            "Ignoring %s=%r: expected a positive whole number; saving every %d new entries.",
            SAVE_EVERY_ENV,
            raw,
            default,
        )
    return default


def write_json_atomically(path: Path, payload: Any, **dump_kwargs: Any) -> None:
    """Write ``payload`` as JSON to ``path`` without ever exposing a half-written file.

    The data goes to a temporary file of its own in the same folder, which then
    replaces ``path`` in one step, so readers see the old or the new file and
    processes sharing the folder (e.g. SLURM array jobs) never write into the
    same temporary file. On failure the temporary file is removed and the error
    is raised.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, **dump_kwargs)
        # mkstemp creates the file owner-only; keep the permissions of the file it replaces.
        os.chmod(tmp_name, _file_mode(path))
        _replace_file(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_name)
        raise


def _file_mode(path: Path) -> int:
    """Permission bits for a new version of ``path``: the ones it has, or a new file's."""
    try:
        return os.stat(path).st_mode & 0o777
    except OSError:
        umask = os.umask(0)
        os.umask(umask)
        return 0o666 & ~umask


def _replace_file(src: str, dst: Path) -> None:
    """``os.replace(src, dst)``, retried briefly on Windows.

    Windows refuses to replace a file while another process is replacing or
    reading it (PermissionError); that clears within milliseconds.
    """
    for attempt in range(8):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 7:
                raise
            time.sleep(0.02 * (attempt + 1))
