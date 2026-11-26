"""File helpers and settings shared by embpy's gene caches (gene ids and symbol aliases).

Standard library only, so the lightweight alias resolver can use it.
"""

from __future__ import annotations

import atexit
import contextlib
import glob
import json
import logging
import os
import secrets
import stat
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TypeVar

SAVE_EVERY_ENV = "EMBPY_GENE_CACHE_SAVE_EVERY"
"""Environment variable: save a gene cache after this many new entries."""

LOCK_TIMEOUT = 10.0
"""Seconds a save waits for another process's save of the same cache file."""

STALE_LOCK_AGE = 60.0
"""A lock file older than this many seconds was left by a process that died while saving."""

STALE_TEMP_AGE = 3600.0
"""A cache's temporary file older than this many seconds was left by a process killed while saving."""

_T = TypeVar("_T")

_warned_values: set[str] = set()

# Saves of one cache file in this process run one at a time (_FILE_LOCKS). Entries
# that a finalizer could not save because their file was being saved in this
# process wait in _ORPHANED until that save, the next one, or the exit hook writes
# them. _GUARD is re-entrant because a finalizer can run while it is held.
_GUARD = threading.RLock()
_FILE_LOCKS: dict[str, threading.Lock] = {}
_ORPHANED: dict[str, tuple[Path, int, dict[str, dict[str, Any]], dict[str, Any]]] = {}
_SAVING = threading.local()


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


def in_child_process(owner_pid: int) -> bool:
    """Whether this process may end without running exit hooks, so a cache should save every entry.

    True in a process forked from the one that made the cache (``owner_pid``)
    and in any multiprocessing child: fork and forkserver children end with
    ``os._exit``, which skips the exit hooks that save pending entries.
    """
    if os.getpid() != owner_pid:
        return True
    mp = sys.modules.get("multiprocessing")
    return mp is not None and mp.parent_process() is not None


def snapshot(take: Callable[[], list[_T]]) -> list[_T]:
    """``take()``, a copy of a collection, retried while another thread changes the collection."""
    for _ in range(100):
        with contextlib.suppress(RuntimeError):  # "... changed size during iteration"
            return take()
    return []


def merge_into_cache_file(
    path: Path,
    version: int,
    entries: dict[str, dict[str, Any]] | None,
    *,
    wait: bool = True,
    **dump_kwargs: Any,
) -> dict[str, Any] | None:
    """Merge ``entries`` into the JSON cache file at ``path`` and return what the file now holds.

    A cache file is ``{"version": version, <section>: {key: value}, ...}`` and
    ``entries`` maps sections to the entries to add or change. The file is read,
    ``entries`` are merged into what it holds (a file of another version, or one
    that is not valid JSON, starts over empty) and the result is written
    atomically. Nothing is removed, so a cache saving its new entries keeps the
    entries other caches and processes saved.

    Saves of one file run one at a time: within the process under a lock, across
    processes under a lock file next to the cache file (see :func:`_lock_file`).
    With ``wait=False`` (finalizers), a save that would wait for a save of the
    same file in this process, including one this thread is in the middle of,
    hands its entries to that save instead and returns ``None``.
    """
    path = Path(path)
    key = os.path.abspath(path)
    with _GUARD:
        lock = _FILE_LOCKS.setdefault(key, threading.Lock())
    saving = _files_being_saved()
    if key in saving or not lock.acquire(blocking=wait):
        _orphan(key, path, version, entries, dump_kwargs)
        if key in saving or not lock.acquire(blocking=False):
            return None  # the save holding the lock writes them before it lets go
        entries = None  # they are among the orphaned entries now
    saving.add(key)
    try:
        written = _merge_locked(path, key, version, entries, dump_kwargs)
    finally:
        saving.discard(key)
        lock.release()
    if key in _ORPHANED:  # handed over while this save was finishing
        merge_into_cache_file(path, version, None, wait=False, **dump_kwargs)
    return written


def _merge_locked(
    path: Path, key: str, version: int, entries: dict[str, dict[str, Any]] | None, dump_kwargs: dict[str, Any]
) -> dict[str, Any] | None:
    """The read-merge-write of :func:`merge_into_cache_file`, run under this process's lock for ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock_file(path):
        sections = _sections(_read_json(path), version)
        written = None
        while True:
            orphaned = _take_orphaned(key)
            if entries is None and orphaned is None:
                if written is not None:
                    _remove_stale_temp_files(path)
                return written
            for part in (orphaned, entries):
                for section, new in (part or {}).items():
                    sections.setdefault(section, {}).update(new)
            content = {"version": version, **sections}
            try:
                write_json_atomically(path, content, **dump_kwargs)
            except BaseException:
                _orphan(key, path, version, orphaned, dump_kwargs)  # kept for the next save
                raise
            written = content
            entries = None


@contextlib.contextmanager
def _lock_file(path: Path) -> Iterator[None]:
    """Hold ``<path>.lock`` while the cache file is read, merged and written.

    The lock file is created with ``O_CREAT | O_EXCL``, so one process holds it at
    a time; a holder is waited for up to :data:`LOCK_TIMEOUT` seconds, and a lock
    file older than :data:`STALE_LOCK_AGE` seconds is removed as left behind by a
    killed process. When the lock cannot be had, the save goes ahead without it
    after a warning: the write is still atomic, it only risks dropping the new
    entries of a save running at the same moment.
    """
    lock = path.with_name(path.name + ".lock")
    fd = None
    deadline = time.monotonic() + LOCK_TIMEOUT
    delay = 0.005
    while True:
        try:
            fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            break
        except FileExistsError:
            pass
        except PermissionError:
            if os.name != "nt":  # on Windows: the lock file is being deleted right now
                logging.warning("Could not create %s; saving %s without the lock.", lock, path)
                break
        except OSError as e:
            logging.warning("Could not create %s (%s); saving %s without the lock.", lock, e, path)
            break
        try:
            stale = time.time() - os.stat(lock).st_mtime > STALE_LOCK_AGE
        except OSError:
            continue  # released meanwhile
        if stale:
            with contextlib.suppress(OSError):
                os.remove(lock)
            continue
        if time.monotonic() >= deadline:
            logging.warning(
                "Another process has held %s for %.0f s; saving %s without the lock.", lock, LOCK_TIMEOUT, path
            )
            break
        time.sleep(delay)
        delay = min(delay * 2, 0.1)
    try:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        if fd is not None:
            os.close(fd)
            with contextlib.suppress(OSError):
                os.remove(lock)


def _read_json(path: Path) -> Any:
    """Parsed content of ``path``, or ``None`` if it does not exist or is not valid JSON."""
    for attempt in range(8):
        try:
            text = path.read_text(encoding="utf-8")
            break
        except FileNotFoundError:
            return None
        except PermissionError:
            if os.name != "nt" or attempt == 7:  # Windows: being replaced right now
                raise
            time.sleep(0.02 * (attempt + 1))
    try:
        return json.loads(text)
    except ValueError:
        logging.warning("Cache file %s is not valid JSON; it is written anew.", path)
        return None


def _sections(content: Any, version: int) -> dict[str, dict[str, Any]]:
    """Copies of the sections of a cache file's content; none if it has another version."""
    if not isinstance(content, dict) or content.get("version") != version:
        return {}
    return {name: dict(value) for name, value in content.items() if name != "version" and isinstance(value, dict)}


def _files_being_saved() -> set[str]:
    files = getattr(_SAVING, "files", None)
    if files is None:
        files = _SAVING.files = set()
    return files


def _orphan(
    key: str, path: Path, version: int, entries: dict[str, dict[str, Any]] | None, dump_kwargs: dict[str, Any]
) -> None:
    if not entries:
        return
    with _GUARD:
        waiting = _ORPHANED[key][2] if key in _ORPHANED else {}
        for section, new in entries.items():
            waiting.setdefault(section, {}).update(new)
        _ORPHANED[key] = (path, version, waiting, dump_kwargs)


def _take_orphaned(key: str) -> dict[str, dict[str, Any]] | None:
    with _GUARD:
        item = _ORPHANED.pop(key, None)
    return item[2] if item else None


def _save_orphaned_entries() -> None:
    """Exit hook: write entries that are still waiting for their cache file."""
    with _GUARD:
        waiting = snapshot(lambda: [(p, v, kw) for p, v, _, kw in _ORPHANED.values()])
    for path, version, dump_kwargs in waiting:
        with contextlib.suppress(Exception):
            merge_into_cache_file(path, version, None, **dump_kwargs)


atexit.register(_save_orphaned_entries)


def write_json_atomically(path: Path, payload: Any, **dump_kwargs: Any) -> None:
    """Write ``payload`` as JSON to ``path`` without ever exposing a half-written file.

    The data goes to a temporary file of its own in the same folder, which then
    replaces ``path`` in one step, so readers see the old or the new file and
    processes sharing the folder (e.g. SLURM array jobs) never write into the
    same temporary file. The temporary file is created like any new file (the
    umask or the folder's default ACL apply) and, when ``path`` exists, given
    its mode. On failure the temporary file is removed and the error is raised.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = _new_temp_file(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, **dump_kwargs)
        with contextlib.suppress(FileNotFoundError):
            os.chmod(tmp_name, stat.S_IMODE(os.stat(path).st_mode))
        _replace_file(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_name)
        raise


def _new_temp_file(path: Path) -> tuple[int, str]:
    """Create ``.<name>.<pid>-<random>.tmp`` next to ``path``, as an ordinary new file.

    ``os.open(..., 0o666)`` lets the system apply the umask or the folder's
    default ACL as for any new file; reading the umask would mean changing it,
    which races with other threads creating files.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        name = str(path.with_name(f".{path.name}.{os.getpid()}-{secrets.token_hex(6)}.tmp"))
        try:
            return os.open(name, flags, 0o666), name
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create a temporary file next to {path}")


def _remove_stale_temp_files(path: Path) -> None:
    """Remove temporary files of ``path`` older than :data:`STALE_TEMP_AGE` (left by killed processes)."""
    cutoff = time.time() - STALE_TEMP_AGE
    for leftover in path.parent.glob(f".{glob.escape(path.name)}.*.tmp"):
        with contextlib.suppress(OSError):
            if leftover.stat().st_mtime < cutoff:
                leftover.unlink()


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
