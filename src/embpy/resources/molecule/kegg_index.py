"""Offline KEGG index: KEGG COMPOUND and KEGG DRUG entries keyed by InChIKey.

Why this module exists
----------------------
KEGG's ``pubchem:`` namespace holds PubChem **substance** IDs (SIDs), not
compound IDs (CIDs); the KEGG API manual lists it as "PubChem SID". Sending a
CID to ``conv/compound/pubchem:{CID}`` therefore looks up whichever unrelated
substance happens to have SID == CID. For example CID 5291 (imatinib) comes
back as ``cpd:C02226`` (citraconic acid), because SID 5291 is KEGG's own
deposit of C02226. Every non-empty answer of that per-CID route was a wrong
compound. This module never builds such a URL.

Instead the index is built once, from KEGG's bulk tables plus PubChem:

1. ``conv/compound/pubchem`` and ``conv/drug/pubchem`` give the SID that KEGG
   deposited for each ``cpd:`` and ``dr:`` entry;
2. PubChem ``POST substance/sid/cids/JSON?cids_type=standardized`` maps each
   SID to the standardized CID;
3. PubChem ``POST compound/cid/property/InChIKey/JSON`` gives the standard
   InChIKey of each CID.

The result, ``kegg_inchikey.tsv`` (``kegg_id, namespace, sid, cid,
inchikey``), is joined locally on an RDKit InChIKey, so annotating a cohort
needs no per-structure network call. Pathway membership comes from the
``link/`` tables and names from ``list/pathway`` and ``list/pathway/hsa``, so
there is no per-pathway ``get/``.

:meth:`KeggIndex.lookup` keeps three KEGG annotations apart, because they mean
different things:

``compound_pathways``
    Reference maps (``mapNNNNN``, excluding ``map07xxx``) in which a matched
    COMPOUND or DRUG entry is itself a node (``link/pathway/compound`` and the
    non-``map07`` rows of ``link/pathway/drug``).
``drug_maps``
    ``map07xxx`` maps: structure or pharmacological class maps of KEGG DRUG
    ("Antineoplastics - protein kinase inhibitors"). They are drug classes,
    not pathways.
``target_pathways``
    Human pathways of the drug's target genes: ``link/hsa/drug`` (drug to
    ``hsa:`` genes) joined to ``link/pathway/hsa``. They describe the targets,
    not the molecule, and overlap with gene-level pathway annotation.

Every mapped entry carries a ``match_tier``: ``"exact"`` when the full
standard InChIKey matches, ``"connectivity"`` when only the first block (the
skeleton, ignoring stereo, isotopes and protonation) matches. The
connectivity tier is a weaker, flagged evidence class (dexamethasone and
betamethasone share it) and every pathway item says which tier supports it.

Licence
-------
The KEGG REST API is "only for academic use by academic users", at no more
than 3 requests per second (https://www.kegg.jp/kegg/rest/,
https://www.kegg.jp/kegg/legal.html). The tables written by
:func:`build_kegg_reference` are KEGG-derived: keep them on local or cluster
storage for research use and never commit them to a repository or ship them
as package data. :func:`build_kegg_reference` paces KEGG at
``kegg_rate`` (default 1 request/s) and PubChem at ``pubchem_rate``
(default 4 requests/s).

HTTP
----
All HTTP goes through the module-level helpers :func:`_get_text` and
:func:`_post_json` (or the ``get_text`` / ``post_json`` callables passed to
:func:`build_kegg_reference`), so a caller can route the requests through its
own transport (cache, pacing, allowlist). Exceptions raised by those callables
propagate unchanged. A ``None`` return (the helpers' 404 convention) is a
failed download and raises :class:`KeggReferenceError`; it is never read as
"no data".
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

KEGG = "https://rest.kegg.jp"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

#: KEGG bulk tables: ``{name: REST path below https://rest.kegg.jp}``. Each
#: table is written to ``<name>.tsv`` (``info`` to ``info.txt``); see
#: :data:`KEGG_TABLE_FILES`.
KEGG_TABLES: dict[str, str] = {
    "info": "info/kegg",
    "conv_compound_pubchem": "conv/compound/pubchem",
    "conv_drug_pubchem": "conv/drug/pubchem",
    "link_pathway_compound": "link/pathway/compound",
    "link_pathway_drug": "link/pathway/drug",
    "link_hsa_drug": "link/hsa/drug",
    "link_pathway_hsa": "link/pathway/hsa",
    "list_pathway": "list/pathway",
    "list_pathway_hsa": "list/pathway/hsa",
    "conv_compound_chebi": "conv/compound/chebi",
    "conv_drug_chebi": "conv/drug/chebi",
}

#: File name of each :data:`KEGG_TABLES` entry inside the reference folder.
KEGG_TABLE_FILES: dict[str, str] = {
    name: ("info.txt" if name == "info" else f"{name}.tsv") for name in KEGG_TABLES
}

#: KEGG entry to InChIKey map written by :func:`build_kegg_reference`.
INCHIKEY_FILE = "kegg_inchikey.tsv"
INCHIKEY_COLUMNS = ("kegg_id", "namespace", "sid", "cid", "inchikey")
#: Build record written next to the tables (also returned by the builder).
MANIFEST_FILE = "kegg_manifest.json"

#: KEGG documents at most 3 requests per second; faster pacing is refused.
KEGG_MAX_RATE = 3.0
#: PubChem PUG REST allows at most 5 requests per second.
PUBCHEM_MAX_RATE = 5.0

PUBCHEM_SID_CIDS_URL = f"{PUBCHEM}/substance/sid/cids/JSON?cids_type=standardized"
PUBCHEM_CID_INCHIKEY_URL = f"{PUBCHEM}/compound/cid/property/InChIKey/JSON"

KEGG_LICENCE = (
    "KEGG REST API: academic use by academic users only, at most 3 requests/s "
    "(https://www.kegg.jp/kegg/rest/, https://www.kegg.jp/kegg/legal.html). "
    "KEGG-derived tables must not be committed to a repository or redistributed."
)
PUBCHEM_LICENCE = "PubChem (NCBI) data use policy: https://www.ncbi.nlm.nih.gov/home/about/policies/"

_NAMESPACES = ("cpd", "dr")
_ENTRY_ID = {"cpd": re.compile(r"^C\d{5}$"), "dr": re.compile(r"^D\d{5}$")}
_INCHIKEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
_HSA_SUFFIX = re.compile(r"\s+-\s+Homo sapiens \(human\)\s*$")
_DRUG_MAP = re.compile(r"^map07\d{3}$")

# Row validators per two-column table: (left column, right column).
_ROW_PATTERNS: dict[str, tuple[re.Pattern[str], re.Pattern[str]]] = {
    "conv_compound_pubchem": (re.compile(r"^pubchem:\d+$"), re.compile(r"^cpd:C\d{5}$")),
    "conv_drug_pubchem": (re.compile(r"^pubchem:\d+$"), re.compile(r"^dr:D\d{5}$")),
    "link_pathway_compound": (re.compile(r"^cpd:C\d{5}$"), re.compile(r"^path:map\d{5}$")),
    "link_pathway_drug": (re.compile(r"^dr:D\d{5}$"), re.compile(r"^path:map\d{5}$")),
    "link_hsa_drug": (re.compile(r"^dr:D\d{5}$"), re.compile(r"^hsa:\S+$")),
    "link_pathway_hsa": (re.compile(r"^hsa:\S+$"), re.compile(r"^path:hsa\d{5}$")),
    "list_pathway": (re.compile(r"^(?:path:)?map\d{5}$"), re.compile(r"^.+$")),
    "list_pathway_hsa": (re.compile(r"^(?:path:)?hsa\d{5}$"), re.compile(r"^.+$")),
    "conv_compound_chebi": (re.compile(r"^chebi:\d+$"), re.compile(r"^cpd:C\d{5}$")),
    "conv_drug_chebi": (re.compile(r"^chebi:\d+$"), re.compile(r"^dr:D\d{5}$")),
}

#: Tables :meth:`KeggIndex.from_dir` cannot work without (besides the map).
_REQUIRED_TABLES = (
    "link_pathway_compound",
    "link_pathway_drug",
    "link_hsa_drug",
    "link_pathway_hsa",
    "list_pathway",
    "list_pathway_hsa",
)


class KeggReferenceError(RuntimeError):
    """A KEGG reference build failed, or a reference folder is corrupt."""


# ---------------------------------------------------------------------------
# HTTP helpers (module level so a caller can monkeypatch them)
# ---------------------------------------------------------------------------


def _get_text(url: str, params: dict | None = None, timeout: int = 30) -> str | None:
    """GET ``url`` and return the body; ``None`` on HTTP 404, raise otherwise."""
    resp = requests.get(url, params=params, timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.text


def _post_json(
    url: str,
    json_data: Any = None,
    data: Any = None,
    timeout: int = 30,
) -> Any:
    """POST to ``url`` and return the decoded JSON; ``None`` on 404, raise otherwise."""
    resp = requests.post(url, json=json_data, data=data, timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


class _Pacer:
    """Keep successive request start times at least ``1 / rate`` seconds apart."""

    def __init__(self, rate: float, clock: Callable[[], float], sleep: Callable[[float], None]) -> None:
        self.interval = 1.0 / rate
        self.clock = clock
        self.sleep = sleep
        self.last: float | None = None

    def wait(self) -> None:
        now = self.clock()
        if self.last is not None:
            due = self.last + self.interval
            if due > now:
                self.sleep(due - now)
                now = max(self.clock(), due)
        self.last = now


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_table(name: str, text: str) -> list[tuple[str, str]]:
    """Split a KEGG two-column table and validate every row against its pattern."""
    left_pat, right_pat = _ROW_PATTERNS[name]
    rows: list[tuple[str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2 or not left_pat.match(parts[0]) or not right_pat.match(parts[1]):
            raise KeggReferenceError(
                f"KEGG table {name!r}: unexpected row {lineno}: {line[:120]!r}"
            )
        rows.append((parts[0], parts[1]))
    return rows


def parse_kegg_info(text: str) -> dict[str, str]:
    """Return ``{database: release date}`` from an ``info/kegg`` response.

    ``info/kegg`` lists one line per database (name, entry count, date),
    e.g. ``pathway  588  2026/11/09``; the date is the database's last update.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^\s*([A-Za-z_]+)\s+[\d,]+\s+(\d{4}/\d{2}/\d{2})\s*$", line)
        if m:
            out.setdefault(m.group(1), m.group(2))
    return out


def kegg_release_string(info: Mapping[str, str]) -> str:
    """Compact release label from :func:`parse_kegg_info` output."""
    parts = [f"{db} {info[db]}" for db in ("pathway", "compound", "drug", "genes") if db in info]
    return "KEGG " + ", ".join(parts) if parts else "KEGG release unknown"


def _normalise_inchikey(value: str) -> str:
    key = str(value).strip().upper()
    if not _INCHIKEY.match(key):
        raise ValueError(f"not an InChIKey: {value!r}")
    return key


def _normalise_chebi(value: str | int) -> str:
    text = str(value).strip().lower()
    if text.startswith("chebi:"):
        text = text[len("chebi:"):]
    if not text.isdigit():
        raise ValueError(f"not a ChEBI ID: {value!r}")
    return f"chebi:{int(text)}"


def _chunks(items: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_kegg_reference(
    out_dir: str | os.PathLike[str],
    *,
    get_text: Callable[..., str | None] | None = None,
    post_json: Callable[..., Any] | None = None,
    kegg_rate: float = 1.0,
    pubchem_rate: float = 4.0,
    sid_chunk_size: int = 10000,
    cid_chunk_size: int = 10000,
    timeout: int = 120,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Download the KEGG bulk tables and build the KEGG entry to InChIKey map.

    Sends ``len(KEGG_TABLES)`` KEGG GETs (11) at no more than ``kegg_rate``
    requests per second, then chunked PubChem POSTs at no more than
    ``pubchem_rate`` per second: KEGG SIDs to standardized CIDs, and CIDs to
    InChIKeys. Nothing is written until every request has succeeded; the
    tables are then written as received, followed by :data:`INCHIKEY_FILE`
    and :data:`MANIFEST_FILE`.

    Parameters
    ----------
    out_dir
        Reference folder (e.g. ``REF/kegg``); created if missing.
    get_text, post_json
        Callables with the signatures of :func:`_get_text` and
        :func:`_post_json`. Default: this module's helpers, looked up at call
        time so a monkeypatched helper is used.
    kegg_rate, pubchem_rate
        Maximum request rates (requests per second). ``kegg_rate`` above
        :data:`KEGG_MAX_RATE` and ``pubchem_rate`` above
        :data:`PUBCHEM_MAX_RATE` are refused.
    sid_chunk_size, cid_chunk_size
        IDs per PubChem POST.
    timeout
        Per-request timeout passed to the callables.
    clock, sleep
        Time source and sleep function for pacing (defaults
        :func:`time.monotonic` and :func:`time.sleep`).

    Returns
    -------
    dict
        The manifest (also written to ``out_dir/kegg_manifest.json``):
        ``source``, ``release`` (from ``info/kegg``), ``kegg_release``
        (per-database dates), ``created_at``, ``licence``, ``rates``,
        ``counts``, ``pubchem_requests`` and ``files``, where ``files`` maps
        each file name, relative to ``out_dir``, to ``{url, sha256, bytes,
        rows, release, licence, fetched_at}``.

    Raises
    ------
    ValueError
        Bad rates or chunk sizes.
    KeggReferenceError
        A download returned ``None`` or an empty or malformed payload.
    Exception
        Whatever ``get_text`` / ``post_json`` raise propagates unchanged.
    """
    if not 0 < kegg_rate <= KEGG_MAX_RATE:
        raise ValueError(f"kegg_rate must be in (0, {KEGG_MAX_RATE}] requests/s, got {kegg_rate}")
    if not 0 < pubchem_rate <= PUBCHEM_MAX_RATE:
        raise ValueError(f"pubchem_rate must be in (0, {PUBCHEM_MAX_RATE}] requests/s, got {pubchem_rate}")
    if sid_chunk_size < 1 or cid_chunk_size < 1:
        raise ValueError("sid_chunk_size and cid_chunk_size must be positive")
    text_fn = get_text if get_text is not None else _get_text
    post_fn = post_json if post_json is not None else _post_json
    clock = clock if clock is not None else time.monotonic
    sleep = sleep if sleep is not None else time.sleep
    kegg_pacer = _Pacer(kegg_rate, clock, sleep)
    pubchem_pacer = _Pacer(pubchem_rate, clock, sleep)
    out = Path(out_dir)

    # 1. KEGG tables -------------------------------------------------------
    bodies: dict[str, str] = {}
    parsed: dict[str, list[tuple[str, str]]] = {}
    fetched_at: dict[str, str] = {}
    for name, path in KEGG_TABLES.items():
        url = f"{KEGG}/{path}"
        kegg_pacer.wait()
        logger.info("KEGG GET %s", url)
        text = text_fn(url, params=None, timeout=timeout)
        fetched_at[name] = _utc_now()
        if text is None:
            raise KeggReferenceError(f"KEGG table {name!r} could not be downloaded from {url} (no data returned)")
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        if not isinstance(text, str) or not text.strip():
            raise KeggReferenceError(f"KEGG table {name!r} from {url} is empty")
        bodies[name] = text
        if name != "info":
            parsed[name] = _parse_table(name, text)
    info = parse_kegg_info(bodies["info"])
    if "pathway" not in info:
        raise KeggReferenceError("info/kegg did not list the pathway database release")
    release = kegg_release_string(info)

    # 2. KEGG entries and their SIDs --------------------------------------
    entries: list[tuple[str, str, int]] = []  # (namespace, kegg_id, sid)
    for table, namespace in (("conv_compound_pubchem", "cpd"), ("conv_drug_pubchem", "dr")):
        for left, right in parsed[table]:
            entries.append((namespace, right.split(":", 1)[1], int(left.split(":", 1)[1])))
    sids = sorted({sid for _, _, sid in entries})
    if not sids:
        raise KeggReferenceError("the KEGG conv tables listed no PubChem SIDs")

    # 3. PubChem: SID -> standardized CID ---------------------------------
    pubchem_requests: list[dict[str, Any]] = []
    sid_cids: dict[int, list[int]] = {}
    for chunk in _chunks(sids, sid_chunk_size):
        pubchem_pacer.wait()
        payload = post_fn(PUBCHEM_SID_CIDS_URL, data={"sid": ",".join(map(str, chunk))}, timeout=timeout)
        pubchem_requests.append({"url": PUBCHEM_SID_CIDS_URL, "ids": len(chunk), "fetched_at": _utc_now()})
        if payload is None:
            raise KeggReferenceError(
                f"PubChem SID->CID request for {len(chunk)} SIDs ({chunk[0]}..{chunk[-1]}) returned no data"
            )
        requested = set(chunk)
        for item in _information_items(payload):
            sid = item.get("SID")
            if not isinstance(sid, int) or sid not in requested:
                raise KeggReferenceError(f"PubChem SID->CID response contains an SID that was not requested: {sid!r}")
            cids = item.get("CID") or []
            if not isinstance(cids, list) or not all(isinstance(c, int) for c in cids):
                raise KeggReferenceError(f"PubChem SID->CID response has a malformed CID list for SID {sid}")
            merged = sid_cids.setdefault(sid, [])
            merged.extend(c for c in cids if c > 0 and c not in merged)

    # 4. PubChem: CID -> InChIKey -----------------------------------------
    cids_all = sorted({c for cids in sid_cids.values() for c in cids})
    cid_key: dict[int, str] = {}
    invalid_keys = 0
    for chunk in _chunks(cids_all, cid_chunk_size):
        pubchem_pacer.wait()
        payload = post_fn(PUBCHEM_CID_INCHIKEY_URL, data={"cid": ",".join(map(str, chunk))}, timeout=timeout)
        pubchem_requests.append({"url": PUBCHEM_CID_INCHIKEY_URL, "ids": len(chunk), "fetched_at": _utc_now()})
        if payload is None:
            raise KeggReferenceError(
                f"PubChem CID->InChIKey request for {len(chunk)} CIDs ({chunk[0]}..{chunk[-1]}) returned no data"
            )
        requested = set(chunk)
        for prop in _property_items(payload):
            cid = prop.get("CID")
            if not isinstance(cid, int) or cid not in requested:
                raise KeggReferenceError(f"PubChem InChIKey response contains a CID that was not requested: {cid!r}")
            key = str(prop.get("InChIKey") or "").strip().upper()
            if not _INCHIKEY.match(key):
                invalid_keys += 1
                continue
            cid_key[cid] = key

    # 5. Join --------------------------------------------------------------
    map_rows: list[tuple[str, str, str, str, str]] = []
    for namespace, kegg_id, sid in sorted(entries, key=lambda e: (_NAMESPACES.index(e[0]), e[1], e[2])):
        cids = sid_cids.get(sid, [])
        if not cids:
            map_rows.append((kegg_id, namespace, str(sid), "", ""))
        for cid in cids:
            map_rows.append((kegg_id, namespace, str(sid), str(cid), cid_key.get(cid, "")))
    map_text = "\t".join(INCHIKEY_COLUMNS) + "\n" + "".join("\t".join(r) + "\n" for r in map_rows)

    # 6. Write -------------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for name, path in KEGG_TABLES.items():
        data = bodies[name].encode("utf-8")
        filename = KEGG_TABLE_FILES[name]
        _write_atomic(out / filename, data)
        files[filename] = {
            "url": f"{KEGG}/{path}",
            "sha256": _sha256(data),
            "bytes": len(data),
            "rows": len(parsed[name]) if name in parsed else None,
            "release": release,
            "licence": KEGG_LICENCE,
            "fetched_at": fetched_at[name],
        }
    map_data = map_text.encode("utf-8")
    _write_atomic(out / INCHIKEY_FILE, map_data)
    files[INCHIKEY_FILE] = {
        "url": None,
        "derived_from": [
            KEGG_TABLE_FILES["conv_compound_pubchem"],
            KEGG_TABLE_FILES["conv_drug_pubchem"],
            PUBCHEM_SID_CIDS_URL,
            PUBCHEM_CID_INCHIKEY_URL,
        ],
        "sha256": _sha256(map_data),
        "bytes": len(map_data),
        "rows": len(map_rows),
        "release": release,
        "licence": f"{KEGG_LICENCE} {PUBCHEM_LICENCE}",
        "fetched_at": _utc_now(),
    }
    with_cid = {(r[1], r[0]) for r in map_rows if r[3]}
    with_key = {(r[1], r[0]) for r in map_rows if r[4]}
    manifest: dict[str, Any] = {
        "source": "kegg",
        "release": release,
        "kegg_release": info,
        "created_at": _utc_now(),
        "licence": KEGG_LICENCE,
        "rates": {"kegg_requests_per_s": kegg_rate, "pubchem_requests_per_s": pubchem_rate},
        "counts": {
            "cpd_entries": sum(1 for e in entries if e[0] == "cpd"),
            "dr_entries": sum(1 for e in entries if e[0] == "dr"),
            "sids": len(sids),
            "sids_without_cid": sum(1 for s in sids if not sid_cids.get(s)),
            "cids": len(cids_all),
            "cids_without_inchikey": sum(1 for c in cids_all if c not in cid_key),
            "invalid_inchikeys": invalid_keys,
            "entries_with_cid": len(with_cid),
            "entries_with_inchikey": len(with_key),
            "distinct_inchikeys": len({r[4] for r in map_rows if r[4]}),
        },
        "pubchem_requests": pubchem_requests,
        "files": files,
    }
    _write_atomic(out / MANIFEST_FILE, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    logger.info(
        "KEGG reference %s written to %s: %d entries, %d with an InChIKey",
        release, out, len(entries), len(with_key),
    )
    return manifest


def _information_items(payload: Any) -> list[dict[str, Any]]:
    try:
        items = payload["InformationList"]["Information"]
    except (KeyError, TypeError):
        raise KeggReferenceError(f"unexpected PubChem SID->CID payload: {str(payload)[:200]}") from None
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        raise KeggReferenceError("unexpected PubChem SID->CID payload: Information is not a list of records")
    return items


def _property_items(payload: Any) -> list[dict[str, Any]]:
    try:
        items = payload["PropertyTable"]["Properties"]
    except (KeyError, TypeError):
        raise KeggReferenceError(f"unexpected PubChem property payload: {str(payload)[:200]}") from None
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        raise KeggReferenceError("unexpected PubChem property payload: Properties is not a list of records")
    return items


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

_TIER_ORDER = {"exact": 0, "connectivity": 1}


class KeggIndex:
    """KEGG COMPOUND / DRUG lookup by InChIKey, built from a reference folder.

    Load with :meth:`from_dir` on the output of :func:`build_kegg_reference`
    and query with :meth:`lookup`. No network access.
    """

    def __init__(
        self,
        map_rows: Iterable[tuple[str, str, str, str, str]],
        tables: Mapping[str, list[tuple[str, str]]],
        *,
        release: str = "KEGG release unknown",
        source_dir: str | None = None,
    ) -> None:
        self.release = release
        self.source_dir = source_dir
        # (namespace, kegg_id) -> {"sids", "cids", "inchikeys"}
        self._entries: dict[tuple[str, str], dict[str, list]] = {}
        self._by_key: dict[str, set[tuple[str, str]]] = {}
        self._by_block: dict[str, set[tuple[str, str]]] = {}
        self._by_cid: dict[int, set[tuple[str, str]]] = {}
        for kegg_id, namespace, sid, cid, key in map_rows:
            ent = (namespace, kegg_id)
            rec = self._entries.setdefault(ent, {"sids": [], "cids": [], "inchikeys": []})
            if sid and int(sid) not in rec["sids"]:
                rec["sids"].append(int(sid))
            if cid:
                if int(cid) not in rec["cids"]:
                    rec["cids"].append(int(cid))
                self._by_cid.setdefault(int(cid), set()).add(ent)
            if key:
                if key not in rec["inchikeys"]:
                    rec["inchikeys"].append(key)
                self._by_key.setdefault(key, set()).add(ent)
                self._by_block.setdefault(key[:14], set()).add(ent)

        def strip(prefixed: str) -> str:
            return prefixed.split(":", 1)[1]

        self._chebi: dict[str, set[tuple[str, str]]] = {}
        self._chebi_of: dict[tuple[str, str], list[str]] = {}
        for table, namespace in (("conv_compound_chebi", "cpd"), ("conv_drug_chebi", "dr")):
            for chebi, kegg in tables.get(table, ()):
                ent = (namespace, strip(kegg))
                self._chebi.setdefault(chebi, set()).add(ent)
                self._chebi_of.setdefault(ent, []).append(chebi.upper())

        self._entry_maps: dict[tuple[str, str], list[str]] = {}
        for table, namespace in (("link_pathway_compound", "cpd"), ("link_pathway_drug", "dr")):
            for kegg, path in tables.get(table, ()):
                self._entry_maps.setdefault((namespace, strip(kegg)), []).append(strip(path))
        self._drug_targets: dict[str, list[str]] = {}
        for drug, gene in tables.get("link_hsa_drug", ()):
            self._drug_targets.setdefault(strip(drug), []).append(gene)
        self._gene_pathways: dict[str, list[str]] = {}
        for gene, path in tables.get("link_pathway_hsa", ()):
            self._gene_pathways.setdefault(gene, []).append(strip(path))
        self._names: dict[str, str] = {}
        for table in ("list_pathway", "list_pathway_hsa"):
            for pid, name in tables.get(table, ()):
                pid = pid.split(":", 1)[1] if pid.startswith("path:") else pid
                self._names[pid] = _HSA_SUFFIX.sub("", name).strip()

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_dir(cls, path: str | os.PathLike[str], *, verify: bool = True) -> KeggIndex:
        """Load a folder written by :func:`build_kegg_reference`.

        Parameters
        ----------
        path
            The reference folder (holding :data:`INCHIKEY_FILE` and the KEGG
            tables).
        verify
            When :data:`MANIFEST_FILE` is present, check each file's sha256
            against it.

        Raises
        ------
        FileNotFoundError
            The folder, :data:`INCHIKEY_FILE` or a required link/list table is
            missing (the index was never built, or the build did not finish).
        KeggReferenceError
            A file is malformed or does not match the manifest.
        """
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"KEGG reference folder not found: {root}")
        missing = [INCHIKEY_FILE] if not (root / INCHIKEY_FILE).is_file() else []
        missing += [KEGG_TABLE_FILES[t] for t in _REQUIRED_TABLES if not (root / KEGG_TABLE_FILES[t]).is_file()]
        if missing:
            raise FileNotFoundError(
                f"KEGG reference folder {root} is incomplete (missing {', '.join(missing)}); "
                "build it with embpy.resources.molecule.kegg_index.build_kegg_reference"
            )
        manifest: dict[str, Any] | None = None
        if (root / MANIFEST_FILE).is_file():
            manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
            if verify:
                for filename, meta in (manifest.get("files") or {}).items():
                    target = root / filename
                    if not target.is_file():
                        continue
                    if _sha256(target.read_bytes()) != meta.get("sha256"):
                        raise KeggReferenceError(f"{target} does not match the sha256 in {MANIFEST_FILE}")

        tables: dict[str, list[tuple[str, str]]] = {}
        for name in _ROW_PATTERNS:
            target = root / KEGG_TABLE_FILES[name]
            if target.is_file():
                tables[name] = _parse_table(name, target.read_text(encoding="utf-8"))

        lines = (root / INCHIKEY_FILE).read_text(encoding="utf-8").splitlines()
        if not lines or tuple(lines[0].split("\t")) != INCHIKEY_COLUMNS:
            raise KeggReferenceError(f"{root / INCHIKEY_FILE}: header must be {INCHIKEY_COLUMNS}")
        map_rows: list[tuple[str, str, str, str, str]] = []
        for lineno, line in enumerate(lines[1:], start=2):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 5:
                raise KeggReferenceError(f"{root / INCHIKEY_FILE}: row {lineno} has {len(parts)} fields")
            kegg_id, namespace, sid, cid, key = parts
            if (
                namespace not in _ENTRY_ID
                or not _ENTRY_ID[namespace].match(kegg_id)
                or not sid.isdigit()
                or (cid and not cid.isdigit())
                or (key and not _INCHIKEY.match(key))
            ):
                raise KeggReferenceError(f"{root / INCHIKEY_FILE}: malformed row {lineno}: {line[:120]!r}")
            map_rows.append((kegg_id, namespace, sid, cid, key))

        if manifest and manifest.get("release"):
            release = str(manifest["release"])
        elif (root / KEGG_TABLE_FILES["info"]).is_file():
            info_text = (root / KEGG_TABLE_FILES["info"]).read_text(encoding="utf-8")
            release = kegg_release_string(parse_kegg_info(info_text))
        else:
            release = "KEGG release unknown"
        return cls(map_rows, tables, release=release, source_dir=str(root))

    def __len__(self) -> int:
        return len(self._entries)

    # -- lookup -------------------------------------------------------------

    def lookup(
        self,
        inchikey: str,
        *,
        cid: int | None = None,
        chebi_ids: Iterable[str | int] | None = None,
        connectivity: bool = True,
    ) -> dict[str, Any]:
        """Map one structure, given its standard InChIKey, to KEGG.

        Parameters
        ----------
        inchikey
            Standard InChIKey of the query structure (e.g. from RDKit).
        cid, chebi_ids
            Optional identifiers the caller resolved by other routes. KEGG
            entries reached through them are reported, never accepted: an
            entry whose InChIKey differs from ``inchikey`` is listed under
            ``mismatched`` (and gives ``identity_mismatch`` when nothing
            matched by InChIKey); one without a PubChem structure is listed
            under ``unverified``.
        connectivity
            Also match entries that share only the InChIKey connectivity
            block (first 14 characters). They are flagged with
            ``match_tier="connectivity"``.

        Returns
        -------
        dict
            ``status``: ``"success"`` (a matched entry has at least one
            compound pathway, drug map or target pathway),
            ``"kegg_entry_no_pathway"`` (matched, but no links),
            ``"identity_mismatch"`` (no InChIKey match, but ``cid`` or
            ``chebi_ids`` point at KEGG entries with a different InChIKey) or
            ``"not_in_kegg"``.
            ``entries``: ``[{kegg_id, namespace, match_tier, inchikey, sids,
            cids, chebi_ids}]``, exact matches first.
            ``compound_pathways`` / ``drug_maps`` / ``target_pathways``:
            ``[{id, name, match_tier, kegg_ids, ...}]``, one item per pathway
            with the best tier that supports it and the supporting entries of
            that tier (``compound_pathways`` items add ``namespaces``;
            ``target_pathways`` items add ``genes``).
            ``targets``: ``[{gene, entrez_id, match_tier, kegg_ids}]``.
            Also ``inchikey``, ``match_tier`` (best tier or ``None``),
            ``mismatched``, ``unverified`` and ``release``.

        Raises
        ------
        ValueError
            ``inchikey`` is not an InChIKey, or a ChEBI ID is malformed.
        """
        key = _normalise_inchikey(inchikey)
        matched: dict[tuple[str, str], str] = {ent: "exact" for ent in self._by_key.get(key, ())}
        if connectivity:
            for ent in self._by_block.get(key[:14], ()):
                matched.setdefault(ent, "connectivity")
        order = sorted(matched, key=lambda e: (_TIER_ORDER[matched[e]], _NAMESPACES.index(e[0]), e[1]))

        entries = []
        for ent in order:
            rec = self._entries[ent]
            hit = [k for k in rec["inchikeys"] if k == key] or [k for k in rec["inchikeys"] if k[:14] == key[:14]]
            entries.append({
                "kegg_id": ent[1],
                "namespace": ent[0],
                "match_tier": matched[ent],
                "primary_annotation_eligible": matched[ent] == "exact",
                "identity_evidence": "exact_standard_inchikey" if matched[ent] == "exact" else "connectivity_only_stereochemistry_unverified",
                "inchikey": hit[0] if hit else None,
                "sids": list(rec["sids"]),
                "cids": list(rec["cids"]),
                "chebi_ids": list(self._chebi_of.get(ent, [])),
            })

        compound: dict[str, dict[str, Any]] = {}
        drug_maps: dict[str, dict[str, Any]] = {}
        targets: dict[str, dict[str, Any]] = {}
        target_paths: dict[str, dict[str, Any]] = {}
        for ent in order:
            tier = matched[ent]
            namespace, kegg_id = ent
            for map_id in self._entry_maps.get(ent, ()):
                if _DRUG_MAP.match(map_id):
                    self._support(drug_maps, map_id, tier, kegg_id)
                    continue
                item = self._support(compound, map_id, tier, kegg_id, lists=("namespaces",))
                if item is not None and namespace not in item["namespaces"]:
                    item["namespaces"].append(namespace)
            if namespace != "dr":
                continue
            for gene in self._drug_targets.get(kegg_id, ()):
                if self._support(targets, gene, tier, kegg_id, named=False) is None:
                    continue
                for path_id in self._gene_pathways.get(gene, ()):
                    pitem = self._support(target_paths, path_id, tier, kegg_id, lists=("genes",))
                    if pitem is not None and gene not in pitem["genes"]:
                        pitem["genes"].append(gene)

        candidates: dict[tuple[str, str], list[str]] = {}
        if cid is not None:
            for ent in self._by_cid.get(int(cid), ()):
                candidates.setdefault(ent, []).append("cid")
        for chebi in chebi_ids or ():
            for ent in self._chebi.get(_normalise_chebi(chebi), ()):
                if "chebi" not in candidates.setdefault(ent, []):
                    candidates[ent].append("chebi")
        mismatched, unverified = [], []
        for ent in sorted(candidates, key=lambda e: (_NAMESPACES.index(e[0]), e[1])):
            if ent in matched:
                continue
            rec = self._entries.get(ent)
            row = {"kegg_id": ent[1], "namespace": ent[0], "via": candidates[ent]}
            if rec and rec["inchikeys"]:
                mismatched.append({**row, "inchikeys": list(rec["inchikeys"])})
            else:
                unverified.append({**row, "reason": "no PubChem structure for this KEGG entry"})

        compound_pathways = _sorted_items(compound)
        drug_map_items = _sorted_items(drug_maps)
        target_items = _sorted_items(target_paths)
        for item in compound_pathways:
            item["namespaces"].sort(key=_NAMESPACES.index)
        target_list = []
        for gene, item in sorted(targets.items()):
            target_list.append({
                "gene": gene,
                "entrez_id": gene.split(":", 1)[1],
                "match_tier": item["match_tier"],
                "kegg_ids": item["kegg_ids"],
            })

        if entries:
            status = "success" if (compound_pathways or drug_map_items or target_items) else "kegg_entry_no_pathway"
        else:
            status = "identity_mismatch" if mismatched else "not_in_kegg"
        best = entries[0]["match_tier"] if entries else None
        return {
            "status": status,
            "inchikey": key,
            "match_tier": best,
            "entries": entries,
            "compound_pathways": compound_pathways,
            "drug_maps": drug_map_items,
            "target_pathways": target_items,
            "targets": target_list,
            "mismatched": mismatched,
            "unverified": unverified,
            "release": self.release,
        }

    def lookup_many(self, inchikeys: Iterable[str], **kwargs: Any) -> dict[str, dict[str, Any]]:
        """:meth:`lookup` for several InChIKeys: ``{inchikey: result}``."""
        return {k: self.lookup(k, **kwargs) for k in inchikeys}

    def _support(
        self,
        bucket: dict[str, dict[str, Any]],
        item_id: str,
        tier: str,
        kegg_id: str,
        *,
        named: bool = True,
        lists: tuple[str, ...] = (),
    ) -> dict[str, Any] | None:
        """Record that ``kegg_id`` (at ``tier``) supports ``item_id``.

        Returns the item when this entry is in the item's best tier, else
        ``None`` (a connectivity entry adds nothing to an exact item).
        Entries arrive exact-first, so the first tier seen is the best.
        ``lists`` names extra list fields to create on a new item.
        """
        item = bucket.get(item_id)
        if item is None:
            item = {"id": item_id}
            if named:
                item["name"] = self._names.get(item_id, item_id)
            item.update({"match_tier": tier, "kegg_ids": []})
            item["primary_annotation_eligible"] = tier == "exact"
            item.update({field: [] for field in lists})
            bucket[item_id] = item
        elif _TIER_ORDER[tier] > _TIER_ORDER[item["match_tier"]]:
            return None
        if kegg_id not in item["kegg_ids"]:
            item["kegg_ids"].append(kegg_id)
        return item


def _sorted_items(bucket: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [bucket[k] for k in sorted(bucket)]
