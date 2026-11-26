"""Resilient symbol -> approved-symbol resolution chain.

Why this module exists
----------------------

The Ensembl REST endpoint ``/lookup/symbol/human/<SYMBOL>`` returns
``400 Bad Request`` when ``<SYMBOL>`` has been aliased to a different
approved name (e.g. ``AARS`` -> ``AARS1``, ``SARS`` -> ``SARS1``,
``RARS`` -> ``RARS1``, ``WARS`` -> ``WARS1``, ``YARS`` -> ``YARS1``,
``VARS`` -> ``VARS1``, ``DARS`` -> ``DARS1``, ``EPRS`` -> ``EPRS1``,
``QARS`` -> ``QARS1``, ``KARS`` -> ``KARS1``, ``IARS`` -> ``IARS1``,
``CARS`` -> ``CARS1``, ``HARS`` -> ``HARS1``, ``LARS`` -> ``LARS1``,
``MARS`` -> ``MARS1``, ``NARS`` -> ``NARS1``, ``GARS`` -> ``GARS1``,
``TARS`` -> ``TARS1``). The pre-Part-A code path swallowed the 400 and
emitted a zero embedding row, which silently turned thousands of real
genes into zero rows for an entire dataset.

The chain implemented here is:

1. pyensembl local lookup (cheap, offline-able).
2. HGNC ``fetch/symbol`` REST -- maps an aliased / withdrawn symbol to
   the current approved one.
3. Ensembl REST retry with the approved symbol from step 2.
4. MyGene.info query (last resort: tolerant of name variants).

Both positive and negative results are cached on disk
(``~/.cache/embpy/symbol_resolution.json``) so subsequent runs of the
same dataset never re-issue the same 4xx-prone HTTPS request.

Lazy imports: the network helpers only import ``requests`` on the path
that actually needs them. The cache helpers use only the standard
library so this module is importable in environments without
``requests`` / ``pyensembl`` (the chain just degrades gracefully).
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ._cache_files import merge_into_cache_file, save_every

__all__ = [
    "ALIAS_CACHE_VERSION",
    "AliasCache",
    "Resolution",
    "default_cache_path",
    "resolve_symbol_chain",
]

logger = logging.getLogger(__name__)


ALIAS_CACHE_VERSION: int = 1
"""Bump if the cache schema changes incompatibly."""

# The file is saved every this many new entries (EMBPY_GENE_CACHE_SAVE_EVERY overrides it).
_ALIAS_CACHE_SAVE_EVERY = 500


def default_cache_path() -> Path:
    """Return ``~/.cache/embpy/symbol_resolution.json``, honouring ``EMBPY_CACHE``.

    Falls back to ``/tmp`` if ``HOME`` is unset (rare but possible on
    minimal SLURM containers). Resolution-cache files are tiny (a few
    KB) so the fallback is harmless.
    """
    env = os.environ.get("EMBPY_CACHE")
    if env:
        base = Path(env)
    else:
        home = Path.home() if os.environ.get("HOME") else Path("/tmp")
        base = home / ".cache" / "embpy"
    return base / "symbol_resolution.json"


@dataclass
class Resolution:
    """Outcome of a single :func:`resolve_symbol_chain` call.

    ``approved_symbol`` is the canonical HGNC symbol if any step in the
    chain succeeded, or ``None`` if every step failed. ``source`` names
    the step that succeeded (or ``"none"`` for a negative result), and
    ``chain`` records the per-step verdict (``"ok"``, ``"miss"``,
    ``"http_error"``, etc.) for the audit log.
    """

    input_symbol: str
    organism: str
    approved_symbol: str | None
    source: Literal["pyensembl", "hgnc", "ensembl_retry", "mygene", "none"]
    chain: tuple[tuple[str, str], ...] = field(default_factory=tuple)


@dataclass
class AliasCache:
    """File-backed JSON cache for :class:`Resolution` results.

    Saves are batched (see :meth:`flush`) and merge this cache's new entries
    into the file under a lock file, so entries other caches or processes
    saved meanwhile are kept (see
    :func:`embpy.resources.gene._cache_files.merge_into_cache_file`).
    """

    path: Path
    version: int = ALIAS_CACHE_VERSION
    _data: dict[str, dict[str, Any]] = field(default_factory=dict)
    _loaded: bool = False
    _unsaved: int = field(default=0, init=False, repr=False, compare=False)
    # The file's entries as of the last load or save; entries that differ are what a save merges in.
    _on_disk: dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)

    def _load(self) -> None:
        if self._loaded:
            return
        if not self.path.exists():
            self._data = {}
            self._on_disk = {}
            self._loaded = True
            return
        try:
            raw = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(raw, dict):
                raise ValueError("cache root is not a dict")
            if int(raw.get("version", 0)) != int(self.version):
                logger.info(
                    "AliasCache: version mismatch at %s "
                    "(found=%s, expected=%d); ignoring on disk.",
                    self.path, raw.get("version"), self.version,
                )
                self._data = {}
            else:
                entries = raw.get("entries", {})
                if isinstance(entries, dict):
                    self._data = entries
                else:
                    self._data = {}
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.warning(
                "AliasCache: could not load %s (%s); starting fresh.",
                self.path, e,
            )
            self._data = {}
        self._on_disk = dict(self._data)
        self._loaded = True

    def _key(self, symbol: str, organism: str) -> str:
        return f"{organism.lower()}::{symbol.strip()}"

    def get(self, symbol: str, organism: str) -> Resolution | None:
        self._load()
        entry = self._data.get(self._key(symbol, organism))
        if entry is None:
            return None
        try:
            return Resolution(
                input_symbol=entry["input_symbol"],
                organism=entry["organism"],
                approved_symbol=entry["approved_symbol"],
                source=entry["source"],
                chain=tuple(tuple(x) for x in entry.get("chain", [])),
            )
        except (KeyError, TypeError):
            return None

    def put(self, resolution: Resolution) -> None:
        """Add ``resolution``; the file is saved every 500 new entries (see :meth:`flush`)."""
        self._load()
        key = self._key(resolution.input_symbol, resolution.organism)
        self._data[key] = {
            "input_symbol": resolution.input_symbol,
            "organism": resolution.organism,
            "approved_symbol": resolution.approved_symbol,
            "source": resolution.source,
            "chain": [list(step) for step in resolution.chain],
        }
        self._unsaved += 1
        _UNSAVED_ALIAS_CACHES[id(self)] = self
        if self._unsaved % save_every(_ALIAS_CACHE_SAVE_EVERY) == 0:
            self._flush()

    def flush(self) -> None:
        """Write entries that are not saved yet.

        :meth:`put` saves the file every 500 new entries (the environment
        variable ``EMBPY_GENE_CACHE_SAVE_EVERY`` changes that number); entries
        still unsaved are written when the cache is garbage-collected or the
        process exits normally. Call this to save earlier.
        """
        if self._unsaved:
            self._flush()

    def _pending(self) -> dict[str, Any]:
        """Entries this cache holds that the file did not hold at the last load or save."""
        return {key: entry for key, entry in list(self._data.items()) if self._on_disk.get(key) != entry}

    def _flush(self) -> None:
        pending = self._pending()
        try:
            written = merge_into_cache_file(
                self.path, int(self.version), {"entries": pending}, indent=2, sort_keys=True
            )
        except Exception as e:  # noqa: BLE001 - a cache that cannot be saved must not break resolution
            logger.warning("AliasCache: could not save %s (%s); retrying at the next save.", self.path, e)
            return
        if written is None:  # handed to the save of this file that this thread is in the middle of
            self._on_disk.update(pending)
        else:
            entries = written.get("entries", {})
            self._data.update(entries)  # what other caches saved meanwhile
            self._on_disk = dict(entries)
        self._unsaved = 0
        _UNSAVED_ALIAS_CACHES.pop(id(self), None)

    def __del__(self) -> None:
        # A dropped cache saves what it has not saved yet (the exit hook covers
        # caches still alive when the process ends). wait=False: if this file is
        # being saved in this process right now, that save writes the entries.
        try:
            if getattr(self, "_unsaved", 0):
                merge_into_cache_file(
                    self.path, int(self.version), {"entries": self._pending()}, wait=False, indent=2, sort_keys=True
                )
        except Exception:  # noqa: BLE001 - a finalizer must not raise
            pass


# Caches holding entries their file does not have yet, by id (dataclass
# instances are not hashable). Weak references: never keeps a cache alive.
_UNSAVED_ALIAS_CACHES: weakref.WeakValueDictionary[int, AliasCache] = weakref.WeakValueDictionary()


def _save_unsaved_alias_caches() -> None:
    """Exit hook: write what each live alias cache has not saved yet."""
    for cache in list(_UNSAVED_ALIAS_CACHES.values()):
        with contextlib.suppress(Exception):
            cache.flush()


atexit.register(_save_unsaved_alias_caches)


# ----------------------------------------------------------------------
# Individual chain steps
# ----------------------------------------------------------------------


def _step_pyensembl(
    symbol: str,
    organism: str,
    *,
    ensembl: Any | None,
) -> tuple[str | None, str]:
    """Try the locally-cached pyensembl release."""
    if ensembl is None:
        return None, "no_pyensembl"
    try:
        genes = ensembl.genes_by_name(symbol)
        if not genes and symbol.upper() != symbol:
            genes = ensembl.genes_by_name(symbol.upper())
        if genes:
            name = getattr(genes[0], "gene_name", None) or symbol
            return str(name), "ok"
    except Exception as e:  # noqa: BLE001
        return None, f"error:{type(e).__name__}"
    return None, "miss"


def _step_hgnc(symbol: str) -> tuple[str | None, str]:
    """Query HGNC ``fetch/symbol`` to find the approved name."""
    try:
        import requests
    except ImportError:
        return None, "no_requests"
    url = f"https://rest.genenames.org/fetch/symbol/{symbol}"
    try:
        resp = requests.get(
            url,
            headers={"Accept": "application/json"},
            timeout=10,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("HGNC HTTP error for %s: %s", symbol, e)
        return None, f"error:{type(e).__name__}"
    if resp.status_code != 200:
        return None, f"http:{resp.status_code}"
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return None, "decode_error"
    docs = data.get("response", {}).get("docs", [])
    if not docs:
        # HGNC reports the symbol but not via "symbol"; the alias
        # endpoint is the other common case.
        return _step_hgnc_alias(symbol)
    approved = docs[0].get("symbol")
    if approved:
        return str(approved), "ok"
    return None, "miss"


def _step_hgnc_alias(symbol: str) -> tuple[str | None, str]:
    """Fallback to HGNC ``search/alias_symbol`` for withdrawn names."""
    try:
        import requests
    except ImportError:
        return None, "no_requests"
    url = f"https://rest.genenames.org/search/alias_symbol/{symbol}"
    try:
        resp = requests.get(
            url,
            headers={"Accept": "application/json"},
            timeout=10,
        )
    except Exception as e:  # noqa: BLE001
        return None, f"error:{type(e).__name__}"
    if resp.status_code != 200:
        return None, f"http:{resp.status_code}"
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return None, "decode_error"
    docs = data.get("response", {}).get("docs", [])
    if not docs:
        return None, "miss"
    approved = docs[0].get("symbol")
    if approved:
        return str(approved), "alias_ok"
    return None, "miss"


def _step_ensembl_retry(
    symbol: str,
    organism: str,
) -> tuple[str | None, str]:
    """Hit Ensembl REST one more time after HGNC remapping.

    Used as a confirmation step: if HGNC gave us an "approved" symbol
    but Ensembl still 4xxs on it, we record that fact in the chain so
    the audit log shows where the breakdown is.
    """
    try:
        from .resolver import _ensembl_get
    except ImportError:
        _ensembl_get = None

    url = f"https://rest.ensembl.org/lookup/symbol/{organism}/{symbol}"
    try:
        if _ensembl_get is not None:
            resp = _ensembl_get(url, headers={"Content-Type": "application/json"}, timeout=10)
        else:
            import requests

            resp = requests.get(url, headers={"Content-Type": "application/json"}, timeout=10)
    except Exception as e:  # noqa: BLE001
        return None, f"error:{type(e).__name__}"
    if resp.status_code != 200:
        return None, f"http:{resp.status_code}"
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return None, "decode_error"
    display_name = data.get("display_name")
    if display_name:
        return str(display_name), "ok"
    return None, "miss"


def _step_mygene(symbol: str, organism: str) -> tuple[str | None, str]:
    """MyGene.info by exact symbol, then exact alias, for the requested species.

    An alias shared by several genes returns ``(None, "ambiguous:<symbols>")``
    rather than whichever gene ranks first.
    """
    from .resolver import _mygene_exact_symbol_lookup

    hit = _mygene_exact_symbol_lookup(symbol, organism)
    if hit["status"] == "ok":
        return str(hit["symbol"]), "ok"
    if hit["status"] == "ambiguous":
        return None, "ambiguous:" + ",".join(hit["candidates"])
    return None, hit["status"]


# ----------------------------------------------------------------------
# Chain orchestrator
# ----------------------------------------------------------------------


def resolve_symbol_chain(
    symbol: str,
    *,
    organism: str = "human",
    ensembl: Any | None = None,
    cache: AliasCache | None = None,
    refresh: bool = False,
) -> Resolution:
    """Run the four-step chain and cache the result.

    Parameters
    ----------
    symbol
        Input gene symbol (case-sensitive at entry; HGNC matches
        case-sensitively in practice, so we do NOT upper-case it
        eagerly).
    organism
        Species name, e.g. ``"human"``. Lowercased into the cache key.
    ensembl
        Optional pre-initialised ``pyensembl.EnsemblRelease``. Pass the
        instance owned by :class:`GeneResolver` to share the cache.
    cache
        Optional :class:`AliasCache`; defaults to a fresh cache pointing
        at :func:`default_cache_path`. Pass the same instance across
        many calls to amortise the cache load.
    refresh
        If True, ignore a cached answer and run the chain again; the new
        result replaces the cached one.
    """
    sym = symbol.strip()
    if not sym:
        return Resolution(
            input_symbol=symbol,
            organism=organism,
            approved_symbol=None,
            source="none",
            chain=(("input", "empty"),),
        )

    if cache is None:
        # A cache made for this one call is saved before returning.
        cache = AliasCache(path=default_cache_path())
        try:
            return resolve_symbol_chain(symbol, organism=organism, ensembl=ensembl, cache=cache, refresh=refresh)
        finally:
            cache.flush()

    cached = None if refresh else cache.get(sym, organism)
    if cached is not None:
        logger.debug(
            "resolve_symbol: cache hit for %s/%s -> %s (%s)",
            sym, organism, cached.approved_symbol, cached.source,
        )
        return cached

    chain: list[tuple[str, str]] = []

    # Step 1: pyensembl
    approved, status = _step_pyensembl(sym, organism, ensembl=ensembl)
    chain.append(("pyensembl", status))
    if approved:
        res = Resolution(
            input_symbol=sym,
            organism=organism,
            approved_symbol=approved,
            source="pyensembl",
            chain=tuple(chain),
        )
        cache.put(res)
        return res

    # Step 2: HGNC fetch/symbol (+ alias_symbol fallback)
    approved, status = _step_hgnc(sym)
    chain.append(("hgnc", status))
    hgnc_symbol: str | None = approved
    if approved:
        # Step 3: confirm with Ensembl REST using the approved name.
        confirm, conf_status = _step_ensembl_retry(approved, organism)
        chain.append(("ensembl_retry", conf_status))
        final = confirm or approved
        res = Resolution(
            input_symbol=sym,
            organism=organism,
            approved_symbol=final,
            source="ensembl_retry" if confirm else "hgnc",
            chain=tuple(chain),
        )
        cache.put(res)
        return res

    # Step 4: MyGene.info
    approved, status = _step_mygene(sym, organism)
    chain.append(("mygene", status))
    if approved:
        res = Resolution(
            input_symbol=sym,
            organism=organism,
            approved_symbol=approved,
            source="mygene",
            chain=tuple(chain),
        )
        cache.put(res)
        return res

    # Everything failed: record a negative result so we never repeat
    # the same 4xx storm.
    res = Resolution(
        input_symbol=sym,
        organism=organism,
        approved_symbol=None,
        source="none",
        chain=tuple(chain),
    )
    logger.warning(
        "resolve_symbol: could not resolve %r (%s); chain=%s",
        sym, organism, chain,
    )
    cache.put(res)
    _ = hgnc_symbol  # kept for future telemetry; silences linters
    return res
