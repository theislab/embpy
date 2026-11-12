"""Cell line annotation from public cell-line registries.

Aggregates metadata for a cell line from three sources:

1. **Cellosaurus** (API v2, https://api.cellosaurus.org) -- species,
   derived-from site, NCIt disease, category, sex, cross-references and
   problematic-line flags. Lines are resolved by accession first (a
   ``CVCL_`` accession, a DepMap ``ACH-`` or Passports ``SIDM`` ID via an
   exact cross-reference query, or a caller-supplied name -> accession
   mapping). A name search is only the last resort and returns
   ``"ambiguous"`` instead of guessing when several human entries carry the
   name (``PC3`` matches five human lines).
2. **DepMap** -- read from a pinned ``Model.csv`` release file only
   (e.g. DepMap Public 24Q4 on figshare). The DepMap portal API is
   bot-protected and is never called; without a file the source reports
   ``"source_unavailable"``.
3. **Cell Model Passports** (Sanger) -- the JSON:API at
   https://api.cellmodelpassports.sanger.ac.uk (there is no ``/v1``), or a
   pinned model snapshot. Unknown filters are silently ignored by that API,
   so a live answer is only accepted when ``meta.count == 1`` and the
   returned model carries the requested name or RRID.

Every per-source dict has a ``"status"`` (one of :data:`CELL_STATUSES`) and
the common keys ``accession``, ``rrid``, ``name``, ``synonyms``,
``species``, ``disease``, ``site``, ``tissue`` and ``xrefs`` (a list of
``{"database", "accession"}``), plus source-specific fields.

All HTTP traffic goes through the module-level :func:`_get_json` so callers
(e.g. an instrumented transport) can replace it. Exceptions raised by a
replaced helper propagate; the default helper's ``None`` (any HTTP or
network failure) is reported as ``"service_error"``, never as a miss.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

CELLOSAURUS_API = "https://api.cellosaurus.org"
CMP_API = "https://api.cellmodelpassports.sanger.ac.uk"
#: Where the pinned DepMap Model.csv can be obtained (DepMap Public 24Q4,
#: figshare article 27993248, CC BY 4.0). The package never downloads it.
DEPMAP_MODEL_CSV_24Q4_URL = "https://ndownloader.figshare.com/files/51065297"

ALL_SOURCES = ("cellosaurus", "depmap", "passports")
CELL_STATUSES = ("success", "no_match", "ambiguous", "source_unavailable", "service_error")
HUMAN_TAXID = "9606"

#: Fields requested for full Cellosaurus records (search endpoint).
CELLOSAURUS_FIELDS = "ac,id,sy,ox,sx,ca,di,derived-from-site,cell-type,dr,hi,problematic"
#: Lighter fields for the last-resort name search.
CELLOSAURUS_NAME_FIELDS = "ac,id,sy,ox,ca,hi"
CELLOSAURUS_ROWS = 1000
CELLOSAURUS_MAX_PAGES = 5
CELLOSAURUS_BATCH_SIZE = 100

#: Cross-reference databases whose accessions are identifiers (not names).
IDENTIFIER_XREF_DATABASES = ("DepMap", "Cell_Model_Passport", "LINCS_LDP", "LINCS_HMS")
LINCS_XREF_DATABASES = ("LINCS_LDP", "LINCS_HMS")

PASSPORTS_INCLUDE = "sample.tissue,sample.cancer_type,model_msi_status,identifiers.source"
PASSPORTS_PAGE_SIZE = 5

_MISSING_MSI = {"", "no data", "unknown", "nan", "none"}
# Keys of a stored Passports record that are recomputed or belong to a result envelope.
_FLAT_SKIP = {"model_id", "id", "names", "xrefs", "rrid", "status", "source", "query", "method", "release",
              "file_sha256", "meta_count", "candidates", "reason", "error", "accession", "name", "model_name",
              "synonyms", "depmap_id", "disease", "mutational_burden"}


def _get_json(url: str, params: dict | None = None, timeout: int = 30) -> Any:
    """GET ``url`` and decode JSON.

    Returns ``None`` on a 404, any other HTTP error, a network error or an
    undecodable body (legacy convention). Callers treat ``None`` as a
    service failure, never as "no data".
    """
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.debug("Request failed for %s: %s", url, e)
        return None


# ======================================================================
# Small helpers
# ======================================================================


def normalize_cell_name(name: Any) -> str:
    """Casefold and drop punctuation/whitespace (``"MCF-7"`` -> ``"mcf7"``)."""
    return re.sub(r"[\W_]+", "", str(name or "").casefold())


def _is_cvcl(s: str) -> bool:
    return bool(re.match(r"^CVCL_[A-Za-z0-9]+$", s or ""))


def _is_ach(s: str) -> bool:
    return bool(re.match(r"^ACH-\d{6}$", s or ""))


def _is_sidm(s: str) -> bool:
    return bool(re.match(r"^SIDM\d{5}$", s or ""))


def _canonical_cvcl(s: str) -> str:
    return "CVCL_" + s.strip()[5:].upper()


def _solr_phrase(text: str) -> str:
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _strip_common_name(label: str) -> str:
    """``"Homo sapiens (Human)"`` -> ``"Homo sapiens"``."""
    return re.sub(r"\s*\([^()]*\)\s*$", "", str(label or "")).strip()


def _dedupe(values: Iterable[Any]) -> list:
    seen: set = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    return str(value).strip()


def _cell_result(source: str, status: str, query: str, **fields: Any) -> dict[str, Any]:
    """Build a per-source dict with the common keys always present."""
    out: dict[str, Any] = {
        "status": status,
        "source": source,
        "query": query,
        "accession": "",
        "rrid": "",
        "name": "",
        "synonyms": [],
        "species": "",
        "disease": "",
        "site": "",
        "tissue": "",
        "xrefs": [],
    }
    out.update(fields)
    return out


def _service_error(source: str, query: str, reason: str, **fields: Any) -> dict[str, Any]:
    return _cell_result(source, "service_error", query, reason=reason, **fields)


# ======================================================================
# Cellosaurus parsing (API v2 JSON)
# ======================================================================


def cellosaurus_entries(data: Any) -> list | None:
    """Return ``data["Cellosaurus"]["cell-line-list"]``.

    ``None`` means the payload does not have the API v2 shape (for example
    an error body or an older schema); an empty list is a genuine no-hit.
    """
    if not isinstance(data, Mapping):
        return None
    root = data.get("Cellosaurus")
    if not isinstance(root, Mapping):
        return None
    entries = root.get("cell-line-list")
    if not isinstance(entries, list):
        return None
    return entries


def parse_cellosaurus_entry(entry: Any) -> dict[str, Any] | None:
    """Parse one Cellosaurus v2 ``cell-line-list`` entry (search or detail).

    Returns ``None`` when the entry has no primary accession.
    """
    if not isinstance(entry, Mapping):
        return None
    acc_list = [a for a in entry.get("accession-list") or [] if isinstance(a, Mapping)]
    primary = [_str(a.get("value")) for a in acc_list if a.get("type") == "primary" and a.get("value")]
    if not primary:
        return None
    secondary = [_str(a.get("value")) for a in acc_list if a.get("type") != "primary" and a.get("value")]

    names = [n for n in entry.get("name-list") or [] if isinstance(n, Mapping) and n.get("value")]
    identifiers = [_str(n["value"]) for n in names if n.get("type") == "identifier"]
    synonyms = [_str(n["value"]) for n in names if n.get("type") == "synonym"]
    name = identifiers[0] if identifiers else (_str(names[0]["value"]) if names else "")

    species_list = [
        {"taxid": _str(s.get("accession")), "label": _str(s.get("label"))}
        for s in entry.get("species-list") or []
        if isinstance(s, Mapping)
    ]

    sites = []
    for item in entry.get("derived-from-site-list") or []:
        site = item.get("site") if isinstance(item, Mapping) else None
        if isinstance(site, Mapping):
            sites.append({
                "value": _str(site.get("value")),
                "site_type": _str(site.get("site-type")),
                "uberon": [
                    _str(x.get("accession"))
                    for x in site.get("xref-list") or []
                    if isinstance(x, Mapping) and x.get("database") == "UBERON"
                ],
            })

    diseases = [
        {"database": _str(d.get("database")), "accession": _str(d.get("accession")), "label": _str(d.get("label"))}
        for d in entry.get("disease-list") or []
        if isinstance(d, Mapping)
    ]

    cell_type_raw = entry.get("cell-type")
    if isinstance(cell_type_raw, list):
        cell_type_raw = cell_type_raw[0] if cell_type_raw else ""
    cell_type = _str(cell_type_raw.get("value")) if isinstance(cell_type_raw, Mapping) else _str(cell_type_raw)

    xrefs = [
        {"database": _str(x.get("database")), "accession": _str(x.get("accession"))}
        for x in entry.get("xref-list") or []
        if isinstance(x, Mapping) and x.get("database") and x.get("accession")
    ]
    cross_references: dict[str, str] = {}
    for x in xrefs:
        cross_references.setdefault(x["database"], x["accession"])

    def ids(*databases: str) -> list[str]:
        return [x["accession"] for x in xrefs if x["database"] in databases]

    derived_from = [
        {"accession": _str(d.get("accession")), "label": _str(d.get("label"))}
        for d in entry.get("derived-from") or []
        if isinstance(d, Mapping)
    ]
    problems = [
        _str(c.get("value"))
        for c in entry.get("comment-list") or []
        if isinstance(c, Mapping) and c.get("category") == "Problematic cell line"
    ]
    msi = [
        _str(m.get("msi-value"))
        for m in entry.get("microsatellite-instability-list") or []
        if isinstance(m, Mapping) and m.get("msi-value")
    ]
    lincs_ids = ids(*LINCS_XREF_DATABASES)

    return {
        "status": "success",
        "source": "cellosaurus",
        "accession": primary[0],
        "rrid": primary[0],
        "secondary_accessions": secondary,
        "name": name,
        "synonyms": synonyms,
        "species": _strip_common_name(species_list[0]["label"]) if species_list else "",
        "species_label": species_list[0]["label"] if species_list else "",
        "species_taxids": [s["taxid"] for s in species_list],
        "species_list": species_list,
        "sex": _str(entry.get("sex")),
        "age": _str(entry.get("age")),
        "category": _str(entry.get("category")),
        "site": sites[0]["value"] if sites else "",
        "site_type": sites[0]["site_type"] if sites else "",
        "tissue": sites[0]["value"] if sites else "",
        "sites": sites,
        "cell_type": cell_type,
        "disease": diseases[0]["label"] if diseases else "",
        "diseases": diseases,
        "msi_status": msi[0] if msi else "",
        "xrefs": xrefs,
        "cross_references": cross_references,
        "depmap_ids": ids("DepMap"),
        "passports_ids": ids("Cell_Model_Passport"),
        "lincs_ids": lincs_ids,
        "has_lincs_xref": bool(lincs_ids),
        "derived_from": derived_from,
        "parent_accession": derived_from[0]["accession"] if derived_from else "",
        "parent_cell_line": derived_from[0]["label"] if derived_from else "",
        "is_problematic": bool(problems),
        "problems": problems,
    }


def _cellosaurus_candidate(rec: Mapping[str, Any], match: str) -> dict[str, Any]:
    return {
        "accession": rec["accession"],
        "name": rec["name"],
        "synonyms": list(rec["synonyms"]),
        "species": rec["species"],
        "species_taxids": list(rec["species_taxids"]),
        "category": rec["category"],
        "derived_from": [d["accession"] for d in rec["derived_from"]],
        "match": match,
    }


# ======================================================================
# Passports parsing (JSON:API)
# ======================================================================


def _jsonapi_index(included: Any) -> dict[tuple[str, str], Mapping]:
    index: dict[tuple[str, str], Mapping] = {}
    for res in included or []:
        if isinstance(res, Mapping) and res.get("type") is not None and res.get("id") is not None:
            index[(str(res["type"]), str(res["id"]))] = res
    return index


def _rel_items(resource: Mapping, name: str) -> list[Mapping]:
    rel = (resource.get("relationships") or {}).get(name) or {}
    data = rel.get("data") if isinstance(rel, Mapping) else None
    if data is None:
        return []
    if isinstance(data, Mapping):
        return [data]
    return [d for d in data if isinstance(d, Mapping)]


def _resolve(index: Mapping, ref: Mapping) -> Mapping | None:
    return index.get((str(ref.get("type")), str(ref.get("id"))))


def _guess_identifier_database(value: str) -> str:
    if _is_cvcl(value):
        return "Cellosaurus"
    if _is_ach(value):
        return "DepMap"
    return ""


def _passports_record(model: Mapping, index: Mapping) -> dict[str, Any]:
    """Flatten one JSON:API ``model`` resource with its included resources."""
    attrs = model.get("attributes") or {}
    names = [_str(n) for n in attrs.get("names") or [] if _str(n)]

    sample = None
    for ref in _rel_items(model, "sample"):
        sample = _resolve(index, ref)
    sample_attrs = (sample or {}).get("attributes") or {}
    tissue = cancer_type = ""
    if sample is not None:
        for ref in _rel_items(sample, "tissue"):
            res = _resolve(index, ref)
            tissue = _str(((res or {}).get("attributes") or {}).get("name"))
        for ref in _rel_items(sample, "cancer_type"):
            res = _resolve(index, ref)
            cancer_type = _str(((res or {}).get("attributes") or {}).get("name"))

    msi_items = []
    for ref in _rel_items(model, "model_msi_status"):
        res = _resolve(index, ref)
        if res is not None:
            a = res.get("attributes") or {}
            msi_items.append({"msi_status": _str(a.get("msi_status")), "current": a.get("current") is True})
    current = [m["msi_status"] for m in msi_items if m["current"]]
    msi = current[0] if current else ""

    xrefs = []
    for ref in _rel_items(model, "identifiers"):
        res = _resolve(index, ref)
        if res is None:
            continue
        value = _str((res.get("attributes") or {}).get("identifier"))
        if not value:
            continue
        database = ""
        for src_ref in _rel_items(res, "source"):
            src = _resolve(index, src_ref)
            database = _str(((src or {}).get("attributes") or {}).get("name"))
        xrefs.append({"database": database or _guess_identifier_database(value), "accession": value})
    return _passports_flat(
        model_id=_str(model.get("id")),
        names=names,
        xrefs=xrefs,
        tissue=tissue,
        cancer_type=cancer_type,
        cancer_type_detail=_str(sample_attrs.get("cancer_type_detail")),
        site=_str(sample_attrs.get("sample_site")),
        tissue_status=_str(sample_attrs.get("tissue_status")),
        model_type=_str(attrs.get("model_type")),
        growth_properties=_str(attrs.get("growth_properties")),
        msi_status=msi,
        msi_statuses=msi_items,
        ploidy=attrs.get("ploidy"),
        ploidy_snp6=None,
        ploidy_wes=attrs.get("ploidy_wes"),
        ploidy_wgs=attrs.get("ploidy_wgs"),
        mutations_per_mb=attrs.get("mutations_per_mb"),
        species="",
    )


def _passports_flat(*, model_id: str, names: list[str], xrefs: list[dict], **fields: Any) -> dict[str, Any]:
    rrid = next((x["accession"] for x in xrefs if _is_cvcl(x["accession"])), "")
    depmap_id = next((x["accession"] for x in xrefs if _is_ach(x["accession"])), "")
    msi = _str(fields.pop("msi_status", ""))
    cancer_type = _str(fields.get("cancer_type"))
    return {
        "model_id": model_id,
        "accession": model_id,
        "rrid": rrid,
        "depmap_id": depmap_id,
        "name": names[0] if names else "",
        "model_name": names[0] if names else "",
        "synonyms": names[1:],
        "names": names,
        "disease": cancer_type,
        "msi_status": "" if msi.casefold() in _MISSING_MSI else msi,
        "mutational_burden": fields.get("mutations_per_mb"),
        "xrefs": xrefs,
        **fields,
    }


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _passports_from_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one row of a Passports ``model_list_*.csv`` release file."""
    name = _str(row.get("model_name"))
    synonyms = [s.strip() for s in _str(row.get("synonyms")).split(";") if s.strip()]
    names = _dedupe([n for n in [name, *synonyms] if n])
    xrefs = []
    for column, database in (("RRID", "Cellosaurus (RRID)"), ("BROAD_ID", "DepMap_ID (Broad_ID)"),
                             ("CCLE_ID", "CCLE_ID"), ("COSMIC_ID", "COSMIC_ID")):
        value = _str(row.get(column))
        if value:
            xrefs.append({"database": database, "accession": value})
    msi = _str(row.get("msi_status"))
    return _passports_flat(
        model_id=_str(row.get("model_id")),
        names=names,
        xrefs=xrefs,
        tissue=_str(row.get("tissue")),
        cancer_type=_str(row.get("cancer_type")),
        cancer_type_detail=_str(row.get("cancer_type_detail")),
        site=_str(row.get("sample_site")),
        tissue_status=_str(row.get("tissue_status")),
        model_type=_str(row.get("model_type")),
        growth_properties=_str(row.get("growth_properties")),
        msi_status=msi,
        msi_statuses=[{"msi_status": msi, "current": True}] if msi else [],
        ploidy=None,
        ploidy_snp6=_float_or_none(row.get("ploidy_snp6")),
        ploidy_wes=_float_or_none(row.get("ploidy_wes")),
        ploidy_wgs=_float_or_none(row.get("ploidy_wgs")),
        mutations_per_mb=_float_or_none(row.get("mutational_burden")),
        species=_str(row.get("species")),
    )


def _passports_records_from_json(data: Any) -> list[dict[str, Any]]:
    """Flatten a Passports snapshot into records.

    Accepts a JSON:API document, a list of documents (pages), a list of
    ``model`` resources, or a list of already-flattened records.
    """
    if isinstance(data, Mapping):
        if isinstance(data.get("data"), list):
            index = _jsonapi_index(data.get("included"))
            return [_passports_record(m, index) for m in data["data"]
                    if isinstance(m, Mapping) and m.get("type", "model") == "model"]
        if isinstance(data.get("data"), Mapping):
            return _passports_records_from_json({"data": [data["data"]], "included": data.get("included")})
        for key in ("models", "records"):
            if isinstance(data.get(key), list):
                return _passports_records_from_json(data[key])
        raise ValueError("Unrecognised Cell Model Passports snapshot: expected a JSON:API document")
    if isinstance(data, list):
        if all(isinstance(d, Mapping) and "data" in d for d in data):
            merged_data, merged_included = [], []
            for doc in data:
                merged_data.extend(doc.get("data") or [])
                merged_included.extend(doc.get("included") or [])
            return _passports_records_from_json({"data": merged_data, "included": merged_included})
        if all(isinstance(d, Mapping) and d.get("type") == "model" and "attributes" in d for d in data):
            return _passports_records_from_json({"data": data, "included": []})
        if all(isinstance(d, Mapping) for d in data):
            out = []
            for d in data:
                if "model_name" in d or "RRID" in d:
                    out.append(_passports_from_csv_row(d))
                else:
                    names = [_str(n) for n in d.get("names") or [] if _str(n)]
                    xrefs = [x for x in d.get("xrefs") or [] if isinstance(x, Mapping)]
                    if d.get("rrid") and not any(_is_cvcl(_str(x.get("accession"))) for x in xrefs):
                        xrefs.append({"database": "Cellosaurus (RRID)", "accession": _str(d["rrid"])})
                    rest = {k: v for k, v in d.items() if k not in _FLAT_SKIP}
                    out.append(_passports_flat(model_id=_str(d.get("model_id") or d.get("id")), names=names,
                                               xrefs=[{"database": _str(x.get("database")),
                                                       "accession": _str(x.get("accession"))} for x in xrefs],
                                               **rest))
            return out
    raise ValueError("Unrecognised Cell Model Passports snapshot")


# ======================================================================
# Pinned release files (DepMap Model.csv, Passports snapshot)
# ======================================================================

_TABLE_CACHE: dict[tuple, Any] = {}
_TABLE_LOCK = threading.Lock()
_DEPMAP_WARNED = False


def _cached_table(kind: str, path: Path, loader: Callable[[bytes, Path], Any]) -> Any:
    """Load a pinned file once per (path, size, mtime); returns ``(table, sha256)``."""
    stat = path.stat()
    key = (kind, str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    with _TABLE_LOCK:
        if key not in _TABLE_CACHE:
            raw = path.read_bytes()
            sha256 = hashlib.sha256(raw).hexdigest()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            _TABLE_CACHE[key] = (loader(raw, path), sha256)
        return _TABLE_CACHE[key]


def _csv_rows(raw: bytes) -> list[dict[str, str]]:
    text = raw.decode("utf-8-sig")
    dialect = "excel-tab" if text.split("\n", 1)[0].count("\t") > text.split("\n", 1)[0].count(",") else "excel"
    return list(csv.DictReader(io.StringIO(text, newline=""), dialect=dialect))


class _DepMapTable:
    """Index of DepMap ``Model.csv`` rows by ModelID, RRID and names."""

    def __init__(self, rows: Iterable[Mapping[str, Any]]):
        self.rows = [{k: _str(v) for k, v in r.items() if k is not None} for r in rows]
        if self.rows and "ModelID" not in self.rows[0]:
            raise ValueError("DepMap Model.csv must have a 'ModelID' column")
        self.by_id: dict[str, list[dict]] = {}
        self.by_rrid: dict[str, list[dict]] = {}
        self.by_name: dict[str, list[dict]] = {}
        for r in self.rows:
            self.by_id.setdefault(r.get("ModelID", ""), []).append(r)
            if r.get("RRID"):
                self.by_rrid.setdefault(r["RRID"], []).append(r)
            keys = {normalize_cell_name(r.get(c, "")) for c in ("StrippedCellLineName", "CellLineName")}
            for k in keys - {""}:
                self.by_name.setdefault(k, []).append(r)


class _PassportsTable:
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records
        self.by_id: dict[str, list[dict]] = {}
        self.by_rrid: dict[str, list[dict]] = {}
        self.by_exact: dict[str, list[dict]] = {}
        self.by_norm: dict[str, list[dict]] = {}
        for r in records:
            self.by_id.setdefault(r["model_id"], []).append(r)
            if r.get("rrid"):
                self.by_rrid.setdefault(r["rrid"], []).append(r)
            for n in _dedupe(n.casefold() for n in r["names"]):
                self.by_exact.setdefault(n, []).append(r)
            for n in _dedupe(normalize_cell_name(n) for n in r["names"]):
                if n:
                    self.by_norm.setdefault(n, []).append(r)


def _load_depmap(raw: bytes, path: Path) -> _DepMapTable:
    return _DepMapTable(_csv_rows(raw))


def _load_passports(raw: bytes, path: Path) -> _PassportsTable:
    name = path.name.casefold()
    if name.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz")):
        return _PassportsTable([_passports_from_csv_row(r) for r in _csv_rows(raw)])
    return _PassportsTable(_passports_records_from_json(json.loads(raw.decode("utf-8-sig"))))


def _depmap_candidate(row: Mapping[str, str]) -> dict[str, str]:
    return {"depmap_id": row.get("ModelID", ""), "name": row.get("CellLineName", ""),
            "stripped_cell_line_name": row.get("StrippedCellLineName", ""), "rrid": row.get("RRID", "")}


def _passports_candidate(rec: Mapping[str, Any]) -> dict[str, Any]:
    return {"model_id": rec["model_id"], "names": list(rec["names"]), "rrid": rec.get("rrid", "")}


# ======================================================================
# Annotator
# ======================================================================


class CellLineAnnotator:
    """Aggregate annotations for cell lines from public registries.

    Parameters
    ----------
    rate_limit_delay
        Seconds to wait between consecutive API calls (default 0.3).
    depmap_model_csv
        Path to a pinned DepMap ``Model.csv`` (``.csv`` or ``.csv.gz``), or
        an iterable of row mappings / a DataFrame with the same columns.
        Without it :meth:`get_depmap_info` returns ``"source_unavailable"``.
    passports_models
        Optional pinned Cell Model Passports snapshot: a JSON:API
        ``/models`` document (or a list of pages), a ``model_list_*.csv``
        release file, or the equivalent in-memory object. When given,
        :meth:`get_passports_info` makes no network calls.
    cellosaurus_ids
        Optional mapping of caller names (e.g. LINCS ``cell_id``) to
        Cellosaurus accessions, consulted before any name search.
    depmap_release, passports_release
        Free-text release labels recorded in results (e.g. ``"24Q4"``).
    """

    def __init__(
        self,
        rate_limit_delay: float = 0.3,
        *,
        depmap_model_csv: Any = None,
        passports_models: Any = None,
        cellosaurus_ids: Mapping[str, str] | None = None,
        depmap_release: str | None = None,
        passports_release: str | None = None,
    ) -> None:
        self.delay = rate_limit_delay
        self.depmap_model_csv = depmap_model_csv
        self.passports_models = passports_models
        self.depmap_release = depmap_release
        self.passports_release = passports_release
        self._cellosaurus_ids: dict[str, str] = {}
        self._cellosaurus_ids_norm: dict[str, str] = {}
        for k, v in (cellosaurus_ids or {}).items():
            if v and _is_cvcl(str(v).strip()):
                self._cellosaurus_ids[str(k)] = _canonical_cvcl(str(v))
                self._cellosaurus_ids_norm.setdefault(normalize_cell_name(k), _canonical_cvcl(str(v)))
        self._depmap: tuple[_DepMapTable, str | None] | None = None
        self._passports: tuple[_PassportsTable, str | None] | None = None

    def _sleep(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    # ==================================================================
    # Identifier helpers
    # ==================================================================

    @staticmethod
    def _is_cellosaurus_id(s: str) -> bool:
        return _is_cvcl(s)

    @staticmethod
    def _is_depmap_id(s: str) -> bool:
        return _is_ach(s)

    @staticmethod
    def _is_passports_id(s: str) -> bool:
        return _is_sidm(s)

    normalize_name = staticmethod(normalize_cell_name)
    parse_cellosaurus_entry = staticmethod(parse_cellosaurus_entry)

    # ==================================================================
    # Pinned files
    # ==================================================================

    @staticmethod
    def _load_table(kind: str, source: Any, loader: Callable[[bytes, Path], Any],
                    from_rows: Callable[[Any], Any]) -> tuple[Any, str | None]:
        if isinstance(source, (str, os.PathLike)):
            return _cached_table(kind, Path(source), loader)
        if hasattr(source, "to_dict"):
            source = source.to_dict("records")
        return from_rows(source), None

    def _depmap_table(self) -> tuple[_DepMapTable, str | None] | None:
        if self.depmap_model_csv is None:
            return None
        if self._depmap is None:
            self._depmap = self._load_table("depmap", self.depmap_model_csv, _load_depmap, _DepMapTable)
        return self._depmap

    def _passports_table(self) -> tuple[_PassportsTable, str | None] | None:
        if self.passports_models is None:
            return None
        if self._passports is None:
            def from_obj(obj: Any) -> _PassportsTable:
                return _PassportsTable(_passports_records_from_json(obj))
            self._passports = self._load_table("passports", self.passports_models, _load_passports, from_obj)
        return self._passports

    @property
    def provenance(self) -> dict[str, Any]:
        """Release labels and sha256 of the pinned files in use."""
        out: dict[str, Any] = {}
        for key, source, table, release in (
            ("depmap", self.depmap_model_csv, self._depmap_table(), self.depmap_release),
            ("passports", self.passports_models, self._passports_table(), self.passports_release),
        ):
            if table is not None:
                out[key] = {
                    "path": str(source) if isinstance(source, (str, os.PathLike)) else None,
                    "sha256": table[1],
                    "release": release,
                    "records": len(table[0].rows if key == "depmap" else table[0].records),
                }
        return out

    # ==================================================================
    # Source: Cellosaurus
    # ==================================================================

    def _cellosaurus_search(self, q: str, *, fields: str, rows: int, start: int = 0) -> tuple[list, str | None]:
        """One search request. Returns ``(entries, error)``."""
        params: dict[str, Any] = {"q": q, "fields": fields, "format": "json", "rows": rows}
        if start:
            params["start"] = start
        data = _get_json(f"{CELLOSAURUS_API}/search/cell-line", params=params)
        if data is None:
            return [], "request_failed"
        entries = cellosaurus_entries(data)
        if entries is None:
            return [], "unparsed_response"
        return entries, None

    @staticmethod
    def _parse_entries(entries: list) -> list[dict[str, Any]] | None:
        parsed = [p for p in (parse_cellosaurus_entry(e) for e in entries) if p is not None]
        if entries and not parsed:
            return None
        return parsed

    def _cellosaurus_fetch(self, targets: Mapping[str, str], chunk_size: int = CELLOSAURUS_BATCH_SIZE,
                           ) -> dict[str, dict[str, Any]]:
        """Fetch ``{accession: method}`` with one ``ac:A OR ac:B ...`` search per chunk."""
        out: dict[str, dict[str, Any]] = {}
        accessions = _dedupe(_canonical_cvcl(a) for a in targets)
        chunk_size = max(1, int(chunk_size))
        for i in range(0, len(accessions), chunk_size):
            chunk = accessions[i:i + chunk_size]
            if i:
                self._sleep()
            q = " OR ".join(f"ac:{a}" for a in chunk)
            entries, error = self._cellosaurus_search(q, fields=CELLOSAURUS_FIELDS, rows=max(len(chunk), 10))
            parsed = None if error else self._parse_entries(entries)
            if error or parsed is None:
                for a in chunk:
                    out[a] = _service_error("cellosaurus", a, error or "unparsed_response", batch_size=len(chunk))
                continue
            by_acc: dict[str, dict[str, Any]] = {}
            for rec in parsed:
                by_acc.setdefault(rec["accession"], rec)
            for rec in parsed:
                for sec in rec["secondary_accessions"]:
                    by_acc.setdefault(sec, rec)
            for a in chunk:
                rec = by_acc.get(a)
                if rec is None:
                    out[a] = _cell_result("cellosaurus", "no_match", a, reason="accession_not_found")
                else:
                    out[a] = {**rec, "query": a,
                              **({} if rec["accession"] == a else {"requested_accession": a})}
        return {a: {**out[_canonical_cvcl(a)], "method": m} for a, m in targets.items()}

    def get_cellosaurus_batch(
        self, accessions: Iterable[str], chunk_size: int = CELLOSAURUS_BATCH_SIZE,
    ) -> dict[str, dict[str, Any]]:
        """Fetch many Cellosaurus records with one search per chunk.

        Each chunk is a single ``q=ac:A OR ac:B ...`` query with
        ``rows >= len(chunk)``. Records are matched back by primary (or
        secondary) accession, so the provider's relevance order does not
        matter, and accessions absent from the answer are ``"no_match"``.
        A failed or unparsable chunk marks all of its accessions
        ``"service_error"``. Names found in the constructor's
        ``cellosaurus_ids`` mapping join the batch (``method="mapping"``);
        any other input is resolved alone with :meth:`get_cellosaurus_info`.

        Returns
        -------
        ``{input: record}`` for every distinct input.
        """
        inputs = _dedupe(str(a) for a in accessions)
        targets: dict[str, tuple[str, str]] = {}
        for a in inputs:
            s = a.strip()
            if _is_cvcl(s):
                targets[a] = (_canonical_cvcl(s), "accession")
            elif not (_is_ach(s) or _is_sidm(s)):
                mapped = self._cellosaurus_ids.get(s) or self._cellosaurus_ids_norm.get(normalize_cell_name(s))
                if mapped:
                    targets[a] = (mapped, "mapping")
        fetched = self._cellosaurus_fetch({acc: "accession" for acc, _ in targets.values()}, chunk_size=chunk_size)
        out: dict[str, dict[str, Any]] = {}
        for a in inputs:
            if a in targets:
                acc, method = targets[a]
                out[a] = {**fetched[acc], "query": a, "method": method}
            else:
                if fetched or out:
                    self._sleep()
                out[a] = self.get_cellosaurus_info(a)
        return out

    def _cellosaurus_by_xref(self, database: str, value: str, query: str) -> dict[str, Any]:
        entries, error = self._cellosaurus_search(f"dr:{_solr_phrase(f'{database}; {value}')}",
                                                  fields=CELLOSAURUS_FIELDS, rows=10)
        method = f"xref:{database}"
        parsed = None if error else self._parse_entries(entries)
        if error or parsed is None:
            return _service_error("cellosaurus", query, error or "unparsed_response", method=method)
        matched = [p for p in parsed
                   if any(x["database"] == database and x["accession"] == value for x in p["xrefs"])]
        if not matched:
            return _cell_result("cellosaurus", "no_match", query, reason="xref_not_found", method=method)
        if len(matched) > 1:
            return _cell_result("cellosaurus", "ambiguous", query, method=method,
                                candidates=[_cellosaurus_candidate(p, "xref") for p in matched])
        return {**matched[0], "query": query, "method": method}

    def _cellosaurus_by_name(self, name: str, query: str, taxid: str | None) -> dict[str, Any]:
        phrase = _solr_phrase(name)
        q = f"id:{phrase} OR sy:{phrase}"
        entries: list = []
        truncated = True
        for page in range(CELLOSAURUS_MAX_PAGES):
            if page:
                self._sleep()
            batch, error = self._cellosaurus_search(q, fields=CELLOSAURUS_NAME_FIELDS, rows=CELLOSAURUS_ROWS,
                                                    start=page * CELLOSAURUS_ROWS)
            if error:
                return _service_error("cellosaurus", query, error, method="name_search")
            entries.extend(batch)
            if len(batch) < CELLOSAURUS_ROWS:
                truncated = False
                break
        parsed = self._parse_entries(entries)
        if parsed is None:
            return _service_error("cellosaurus", query, "unparsed_response", method="name_search")
        key = normalize_cell_name(name)
        matches, other_species = [], []
        for rec in parsed:
            if normalize_cell_name(rec["name"]) == key:
                match = "identifier"
            elif any(normalize_cell_name(s) == key for s in rec["synonyms"]):
                match = "synonym"
            else:
                continue
            if taxid and taxid not in rec["species_taxids"]:
                other_species.append(_cellosaurus_candidate(rec, match))
            else:
                matches.append(_cellosaurus_candidate(rec, match))
        common = {"method": "name_search", "search_rows": len(entries), "truncated": truncated,
                  "other_species_matches": other_species}
        if not matches:
            return _cell_result("cellosaurus", "no_match", query, reason="no_exact_name_match", **common)
        if len(matches) > 1:
            return _cell_result("cellosaurus", "ambiguous", query, candidates=matches, **common)
        self._sleep()
        acc = matches[0]["accession"]
        rec = self._cellosaurus_fetch({acc: "name_search"})[acc]
        return {**rec, "query": query, "name_match": matches[0]["match"], **common}

    def get_cellosaurus_info(
        self,
        name_or_accession: str,
        *,
        accession: str | None = None,
        taxid: str | None = HUMAN_TAXID,
        name_search: bool = True,
    ) -> dict[str, Any]:
        """Look up one cell line in Cellosaurus (API v2).

        Resolution order: an explicit ``accession``; a ``CVCL_`` accession;
        a DepMap ``ACH-`` or Passports ``SIDM`` ID via an exact
        cross-reference query; the ``cellosaurus_ids`` mapping given to the
        constructor; and only then a name search. The name search returns
        ``"ambiguous"`` with ``candidates`` whenever more than one entry of
        species ``taxid`` has an identifier or synonym equal to the query
        after :func:`normalize_cell_name`; it never takes a first hit.

        Returns
        -------
        Dict with ``status`` (``success``, ``no_match``, ``ambiguous`` or
        ``service_error``), ``method``, ``accession``/``rrid``, ``name``,
        ``synonyms``, ``species``, ``sex``, ``category``, ``site``/``tissue``,
        ``disease`` (first NCIt label) and ``diseases``, ``xrefs``,
        ``cross_references``, ``depmap_ids``, ``passports_ids``,
        ``lincs_ids``, ``has_lincs_xref``, ``derived_from``,
        ``parent_cell_line``, ``is_problematic`` and ``problems``.
        """
        query = str(name_or_accession or "").strip()
        acc, method = None, None
        if accession:
            acc, method = str(accession).strip(), "accession"
        elif _is_cvcl(query):
            acc, method = query, "accession"
        if acc is None and not (_is_ach(query) or _is_sidm(query)):
            mapped = self._cellosaurus_ids.get(query) or self._cellosaurus_ids_norm.get(normalize_cell_name(query))
            if mapped:
                acc, method = mapped, "mapping"
        if acc is not None:
            if not _is_cvcl(acc):
                return _cell_result("cellosaurus", "no_match", query, reason="invalid_accession", method=method)
            rec = self._cellosaurus_fetch({acc: method})[acc]
            return {**rec, "query": query, "method": method}
        if _is_ach(query):
            return self._cellosaurus_by_xref("DepMap", query, query)
        if _is_sidm(query):
            return self._cellosaurus_by_xref("Cell_Model_Passport", query, query)
        if not query:
            return _cell_result("cellosaurus", "no_match", query, reason="empty_query")
        if not name_search:
            return _cell_result("cellosaurus", "no_match", query, reason="name_search_disabled")
        return self._cellosaurus_by_name(query, query, taxid)

    def get_cellosaurus_release(self) -> dict[str, Any]:
        """Return the live Cellosaurus release (``/release-info``)."""
        data = _get_json(f"{CELLOSAURUS_API}/release-info", params={"format": "json"})
        if data is None:
            return {"status": "service_error", "reason": "request_failed"}
        release = None
        if isinstance(data, Mapping):
            release = ((data.get("Cellosaurus") or {}).get("header") or {}).get("release")
        if not isinstance(release, Mapping) or not release.get("version"):
            return {"status": "service_error", "reason": "unparsed_response"}
        return {"status": "success", "version": _str(release.get("version")), "updated": _str(release.get("updated")),
                "nb_cell_lines": _str(release.get("nb-cell-lines"))}

    # ==================================================================
    # Source: DepMap (pinned Model.csv only)
    # ==================================================================

    def _depmap_result(self, row: Mapping[str, str], query: str, method: str, sha256: str | None) -> dict[str, Any]:
        xrefs = [{"database": db, "accession": row.get(col, "")}
                 for col, db in (("RRID", "Cellosaurus"), ("SangerModelID", "Cell_Model_Passport"),
                                 ("CCLEName", "CCLE"), ("COSMICID", "COSMIC"))
                 if row.get(col, "")]
        synonyms = _dedupe(s for s in (row.get("StrippedCellLineName", ""), row.get("CCLEName", ""))
                           if s and s != row.get("CellLineName", ""))
        return _cell_result(
            "depmap", "success", query,
            method=method,
            depmap_id=row.get("ModelID", ""),
            accession=row.get("ModelID", ""),
            rrid=row.get("RRID", ""),
            name=row.get("CellLineName", ""),
            cell_line_name=row.get("CellLineName", ""),
            stripped_cell_line_name=row.get("StrippedCellLineName", ""),
            synonyms=synonyms,
            lineage=row.get("OncotreeLineage", ""),
            lineage_subtype=row.get("OncotreeSubtype", ""),
            primary_disease=row.get("OncotreePrimaryDisease", ""),
            disease=row.get("OncotreePrimaryDisease", ""),
            disease_subtype=row.get("OncotreeSubtype", ""),
            oncotree_code=row.get("OncotreeCode", ""),
            depmap_model_type=row.get("DepmapModelType", ""),
            model_type=row.get("ModelType", ""),
            tissue=row.get("OncotreeLineage", ""),
            site=row.get("SampleCollectionSite", ""),
            primary_or_metastasis=row.get("PrimaryOrMetastasis", ""),
            growth_pattern=row.get("GrowthPattern", ""),
            culture_medium=row.get("OnboardedMedia", ""),
            sex=row.get("Sex", ""),
            source_type=row.get("SourceType", ""),
            passports_id=row.get("SangerModelID", ""),
            ccle_name=row.get("CCLEName", ""),
            xrefs=xrefs,
            release=self.depmap_release,
            file_sha256=sha256,
        )

    def get_depmap_info(self, name: str, *, rrid: str | None = None) -> dict[str, Any]:
        """Look up a cell line in the pinned DepMap ``Model.csv``.

        Order: a DepMap ``ACH-`` ModelID; the ``rrid`` (Cellosaurus
        accession; a ``CVCL_`` ``name`` counts as one); then the
        punctuation-insensitive ``StrippedCellLineName``/``CellLineName``.
        Several rows for one RRID (e.g. RH-30 and SJRH-30 share CVCL_0041)
        are narrowed by name, otherwise ``"ambiguous"``. A name match whose
        RRID differs from a given ``rrid`` is rejected. No network calls are
        made; without a file the result is ``"source_unavailable"``.

        Returns
        -------
        Dict with ``status``, ``method``, ``depmap_id``, ``rrid``, ``name``
        (``CellLineName``), ``lineage``, ``lineage_subtype``,
        ``primary_disease``, ``disease_subtype``, ``growth_pattern``,
        ``sex``, ``site`` (``SampleCollectionSite``), ``tissue``
        (``OncotreeLineage``), ``xrefs`` and more.
        """
        global _DEPMAP_WARNED
        query = str(name or "").strip()
        table_sha = self._depmap_table()
        if table_sha is None:
            if not _DEPMAP_WARNED:
                logger.info("DepMap needs a pinned Model.csv (CellLineAnnotator(depmap_model_csv=...)); "
                            "the DepMap portal API is not used.")
                _DEPMAP_WARNED = True
            return _cell_result("depmap", "source_unavailable", query,
                                reason="no DepMap Model.csv configured (depmap_model_csv)")
        table, sha = table_sha
        rrid = (rrid or "").strip() or (query if _is_cvcl(query) else "")
        if rrid and _is_cvcl(rrid):
            rrid = _canonical_cvcl(rrid)
        key = normalize_cell_name(query) if not (_is_cvcl(query) or _is_ach(query)) else ""

        if _is_ach(query):
            rows = table.by_id.get(query, [])
            if len(rows) == 1:
                return self._depmap_result(rows[0], query, "model_id", sha)
            if len(rows) > 1:
                return _cell_result("depmap", "ambiguous", query, method="model_id",
                                    candidates=[_depmap_candidate(r) for r in rows])
            if not rrid:
                return _cell_result("depmap", "no_match", query, method="model_id", release=self.depmap_release)

        def by_name(rows: list[dict]) -> list[dict]:
            return [r for r in rows if key and key in {normalize_cell_name(r.get("StrippedCellLineName", "")),
                                                       normalize_cell_name(r.get("CellLineName", ""))}]

        if rrid:
            rows = table.by_rrid.get(rrid, [])
            if len(rows) == 1:
                return self._depmap_result(rows[0], query, "rrid", sha)
            if len(rows) > 1:
                narrowed = by_name(rows)
                if len(narrowed) == 1:
                    return self._depmap_result(narrowed[0], query, "rrid+name", sha)
                return _cell_result("depmap", "ambiguous", query, method="rrid", rrid=rrid,
                                    candidates=[_depmap_candidate(r) for r in rows])
        if key:
            rows = [r for r in table.by_name.get(key, []) if not (rrid and r.get("RRID") and r["RRID"] != rrid)]
            if len(rows) == 1:
                return self._depmap_result(rows[0], query, "name", sha)
            if len(rows) > 1:
                return _cell_result("depmap", "ambiguous", query, method="name",
                                    candidates=[_depmap_candidate(r) for r in rows])
        return _cell_result("depmap", "no_match", query, method="rrid" if rrid else "name", rrid=rrid,
                            release=self.depmap_release)

    # ==================================================================
    # Source: Cell Model Passports (Sanger)
    # ==================================================================

    def _passports_live(self, filters: list[dict], query: str, method: str,
                        check: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        data = _get_json(
            f"{CMP_API}/models",
            params={"filter": json.dumps(filters, separators=(",", ":")), "include": PASSPORTS_INCLUDE,
                    "page[size]": PASSPORTS_PAGE_SIZE},
        )
        if data is None:
            return _service_error("passports", query, "request_failed", method=method)
        if not isinstance(data, Mapping) or not isinstance(data.get("data"), list):
            return _service_error("passports", query, "unparsed_response", method=method)
        count = (data.get("meta") or {}).get("count")
        if not isinstance(count, int) or isinstance(count, bool):
            return _service_error("passports", query, "unparsed_response", method=method,
                                  error="JSON:API response without meta.count")
        index = _jsonapi_index(data.get("included"))
        models = [_passports_record(m, index) for m in data["data"]
                  if isinstance(m, Mapping) and m.get("type", "model") == "model"]
        if count == 0 and not models:
            return _cell_result("passports", "no_match", query, method=method, meta_count=0)
        if not models:
            return _service_error("passports", query, "unparsed_response", method=method, meta_count=count)
        if not all(check(m) for m in models):
            return _service_error(
                "passports", query, "filter_ignored", method=method, meta_count=count,
                error=f"Passports returned {count} model(s) that do not carry the requested identity "
                      f"(first: {[m['model_id'] for m in models[:3]]})",
            )
        if count == 1 and len(models) == 1:
            return _cell_result("passports", "success", query, method=method, meta_count=1,
                                release=self.passports_release, **models[0])
        return _cell_result("passports", "ambiguous", query, method=method, meta_count=count,
                            candidates=[_passports_candidate(m) for m in models])

    def _passports_offline(self, table: _PassportsTable, sha: str | None, query: str, rrid: str) -> dict[str, Any]:
        def success(rec: dict, method: str) -> dict[str, Any]:
            return _cell_result("passports", "success", query, method=method, release=self.passports_release,
                                file_sha256=sha, **rec)

        def ambiguous(recs: list[dict], method: str) -> dict[str, Any]:
            return _cell_result("passports", "ambiguous", query, method=method,
                                candidates=[_passports_candidate(r) for r in recs])

        if _is_sidm(query):
            recs = table.by_id.get(query, [])
            if len(recs) == 1:
                return success(recs[0], "model_id")
        exact_key = query.casefold() if query and query != rrid else ""
        norm_key = normalize_cell_name(query) if exact_key else ""

        def named(recs: list[dict]) -> tuple[list[dict], str]:
            hits = [r for r in recs if exact_key and exact_key in {n.casefold() for n in r["names"]}]
            if hits:
                return hits, "name"
            return [r for r in recs if norm_key and norm_key in {normalize_cell_name(n) for n in r["names"]}], \
                "normalized_name"

        if rrid:
            recs = table.by_rrid.get(rrid, [])
            if len(recs) == 1:
                return success(recs[0], "rrid")
            if len(recs) > 1:
                narrowed, how = named(recs)
                if len(narrowed) == 1:
                    return success(narrowed[0], f"rrid+{how}")
                return ambiguous(recs, "rrid")
        if exact_key:
            for recs, how in ((table.by_exact.get(exact_key, []), "name"),
                              (table.by_norm.get(norm_key, []), "normalized_name")):
                recs = [r for r in recs if not (rrid and r.get("rrid") and r["rrid"] != rrid)]
                if len(recs) == 1:
                    return success(recs[0], how)
                if len(recs) > 1:
                    return ambiguous(recs, how)
        return _cell_result("passports", "no_match", query, method="rrid" if rrid else "name", rrid=rrid,
                            release=self.passports_release)

    def get_passports_info(self, name: str, *, rrid: str | None = None) -> dict[str, Any]:
        """Look up a model in Cell Model Passports.

        With a pinned snapshot (``passports_models``) the lookup is offline:
        RRID first, then an exact (casefolded) name, then a
        punctuation-insensitive name (``method="normalized_name"``).

        Live, the JSON:API ``/models`` endpoint is queried with
        ``filter=[{"name":"identifiers","op":"any","val":{"name":"identifier","op":"eq","val":RRID}}]``
        when an RRID is known, otherwise with
        ``filter=[{"name":"names","op":"any","val":NAME}]``, including
        ``sample.tissue``, ``sample.cancer_type``, ``model_msi_status`` and
        ``identifiers.source``. The answer is accepted only when
        ``meta.count == 1`` and the model carries the RRID / exact name;
        a response whose models lack it (the API ignores unknown filters and
        then returns all 2,292 models) is ``service_error`` with
        ``reason="filter_ignored"``.

        Returns
        -------
        Dict with ``status``, ``method``, ``model_id`` (also ``accession``),
        ``rrid``, ``depmap_id``, ``name``/``model_name`` (first of
        ``names``), ``synonyms``, ``tissue``, ``cancer_type`` (also
        ``disease``), ``site``, ``model_type``, ``growth_properties``,
        ``msi_status`` (the item with ``current == True``), ``ploidy``,
        ``ploidy_wes``, ``ploidy_wgs``, ``mutations_per_mb`` and ``xrefs``.
        """
        query = str(name or "").strip()
        rrid = (rrid or "").strip() or (query if _is_cvcl(query) else "")
        if rrid and _is_cvcl(rrid):
            rrid = _canonical_cvcl(rrid)
        table_sha = self._passports_table()
        if table_sha is not None:
            return self._passports_offline(table_sha[0], table_sha[1], query, rrid)

        if _is_sidm(query):
            data = _get_json(f"{CMP_API}/models/{query}", params={"include": PASSPORTS_INCLUDE})
            if data is None:
                return _service_error("passports", query, "request_failed", method="model_id")
            model = data.get("data") if isinstance(data, Mapping) else None
            if not isinstance(model, Mapping) or _str(model.get("id")) != query:
                return _service_error("passports", query, "unparsed_response", method="model_id")
            rec = _passports_record(model, _jsonapi_index(data.get("included")))
            return _cell_result("passports", "success", query, method="model_id", release=self.passports_release,
                                **rec)
        if rrid:
            result = self._passports_live(
                [{"name": "identifiers", "op": "any", "val": {"name": "identifier", "op": "eq", "val": rrid}}],
                query, "rrid", lambda m: any(x["accession"] == rrid for x in m["xrefs"]),
            )
            if result["status"] != "no_match" or not query or query == rrid:
                return result
            self._sleep()
        if not query:
            return _cell_result("passports", "no_match", query, reason="empty_query")
        key = query.casefold()
        return self._passports_live(
            [{"name": "names", "op": "any", "val": query}],
            query, "name", lambda m: key in {n.casefold() for n in m["names"]},
        )

    # ==================================================================
    # Combined annotation
    # ==================================================================

    def annotate(
        self,
        name: str,
        sources: str | list[str] = "all",
        *,
        cellosaurus_record: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Aggregate metadata from all sources for a cell line.

        Parameters
        ----------
        name
            Cell line name or identifier.
        sources
            Which sources to query: ``"all"`` or a list like
            ``["cellosaurus", "depmap"]``.
        cellosaurus_record
            Optional pre-fetched Cellosaurus record (e.g. from
            :meth:`get_cellosaurus_batch`) used instead of a new lookup.

        Returns
        -------
        Combined annotation dict with top-level summary fields (filled only
        from sources whose ``status`` is ``"success"``) and per-source
        details under ``"sources"``. The Cellosaurus accession, once
        resolved, is passed as ``rrid`` to DepMap and Passports.
        """
        src_list = list(ALL_SOURCES) if sources == "all" else list(sources)

        result: dict[str, Any] = {
            "name": name,
            "species": "",
            "tissue": "",
            "disease": "",
            "lineage": "",
            "lineage_subtype": "",
            "growth_pattern": "",
            "model_type": "",
            "msi_status": "",
            "is_problematic": False,
            "cellosaurus_id": "",
            "depmap_id": "",
            "passports_id": "",
            "cross_references": {},
            "sources": {},
        }
        rrid = str(name).strip() if _is_cvcl(str(name).strip()) else ""

        if "cellosaurus" in src_list:
            cello = dict(cellosaurus_record) if cellosaurus_record is not None else self.get_cellosaurus_info(name)
            result["sources"]["cellosaurus"] = cello
            if cello.get("status") == "success":
                rrid = cello.get("accession", "") or rrid
                result["cellosaurus_id"] = cello.get("accession", "")
                result["species"] = cello.get("species", "")
                result["tissue"] = cello.get("tissue", "")
                result["disease"] = cello.get("disease", "")
                result["is_problematic"] = bool(cello.get("is_problematic", False))
                result["cross_references"].update(cello.get("cross_references", {}))

        if "depmap" in src_list:
            depmap = self.get_depmap_info(name, rrid=rrid or None)
            result["sources"]["depmap"] = depmap
            if depmap.get("status") == "success":
                result["depmap_id"] = depmap.get("depmap_id", "")
                result["lineage"] = depmap.get("lineage", "")
                result["lineage_subtype"] = depmap.get("lineage_subtype", "")
                result["disease"] = result["disease"] or depmap.get("primary_disease", "")
                result["growth_pattern"] = depmap.get("growth_pattern", "")

        if "passports" in src_list:
            passports = self.get_passports_info(name, rrid=rrid or None)
            result["sources"]["passports"] = passports
            if passports.get("status") == "success":
                result["passports_id"] = passports.get("model_id", "")
                result["tissue"] = result["tissue"] or passports.get("tissue", "")
                result["disease"] = result["disease"] or passports.get("cancer_type", "")
                result["model_type"] = passports.get("model_type", "")
                result["msi_status"] = passports.get("msi_status", "")
                result["growth_pattern"] = result["growth_pattern"] or passports.get("growth_properties", "")

        return result

    def annotate_batch(
        self,
        names: list[str],
        sources: str | list[str] = "all",
    ) -> list[dict[str, Any]]:
        """Annotate multiple cell lines.

        ``CVCL_`` accessions are fetched from Cellosaurus in batched
        requests (:meth:`get_cellosaurus_batch`); other names are resolved
        one by one.

        Returns
        -------
        List of annotation dicts, one per cell line.
        """
        src_list = list(ALL_SOURCES) if sources == "all" else list(sources)
        prefetched: dict[str, dict[str, Any]] = {}
        if "cellosaurus" in src_list:
            accessions = [n for n in names if _is_cvcl(str(n).strip())]
            if accessions:
                prefetched = self.get_cellosaurus_batch(accessions)
        return [self.annotate(n, sources=src_list, cellosaurus_record=prefetched.get(n)) for n in names]

    def annotate_adata(
        self,
        adata,
        column: str,
        sources: str | list[str] = "all",
    ):
        """Annotate cell lines in an AnnData object.

        Parameters
        ----------
        adata
            AnnData with cell line identifiers in ``obs[column]``.
        column
            Column name containing cell line names.
        sources
            Sources to query.

        Returns
        -------
        The AnnData with annotation columns added to ``.obs`` and
        full annotation dicts in ``.uns["cellline_annotations"]``.
        """
        if column not in adata.obs.columns:
            raise KeyError(f"Column '{column}' not in adata.obs")

        unique_names = [str(n) for n in adata.obs[column].dropna().unique().tolist()]
        logger.info("Annotating %d unique cell lines from column '%s'", len(unique_names), column)

        annotations = dict(zip(unique_names, self.annotate_batch(unique_names, sources=sources), strict=True))

        for field in ("species", "tissue", "disease", "lineage", "model_type", "msi_status"):
            vals = []
            for cl in adata.obs[column]:
                ann = annotations.get(str(cl), {})
                vals.append(ann.get(field, ""))
            adata.obs[f"cellline_{field}"] = vals

        adata.uns["cellline_annotations"] = annotations
        return adata

    # ==================================================================
    # Wikipedia
    # ==================================================================

    def get_wikipedia_info(self, name: str) -> str:
        """Fetch the Wikipedia article extract for a cell line.

        Parameters
        ----------
        name
            Cell line name (e.g. ``"HeLa"``, ``"A549"``).

        Returns
        -------
        Plain-text extract from the Wikipedia article, or ``""``
        if no article is found.
        """
        self._sleep()
        title = name.replace(" ", "_")
        data = _get_json(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
        )
        if data and data.get("extract"):
            if data.get("type") != "disambiguation":
                return data["extract"].strip()

        cell_title = f"{title}_(cell_line)"
        data = _get_json(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{cell_title}",
        )
        if data and data.get("extract"):
            return data["extract"].strip()

        cells_title = f"{title}_cells"
        data = _get_json(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{cells_title}",
        )
        if data and data.get("extract"):
            return data["extract"].strip()

        return ""

    # ==================================================================
    # Text description
    # ==================================================================

    def get_text_description(self, name: str) -> str:
        """Build a text description of a cell line for embedding.

        Combines structured metadata from Cellosaurus/DepMap/Passports
        with the Wikipedia article extract into a single text suitable
        for text embedding models.

        Parameters
        ----------
        name
            Cell line name or identifier.

        Returns
        -------
        Text description string.
        """
        ann = self.annotate(name)

        parts = []
        cl_name = ann.get("name", name)
        parts.append(f"{cl_name} is a cell line.")

        species = ann.get("species", "")
        if species:
            parts.append(f"Species: {species}.")

        tissue = ann.get("tissue", "")
        if tissue:
            parts.append(f"Derived from {tissue} tissue.")

        disease = ann.get("disease", "")
        if disease:
            parts.append(f"Disease: {disease}.")

        lineage = ann.get("lineage", "")
        if lineage:
            subtype = ann.get("lineage_subtype", "")
            if subtype:
                parts.append(f"Lineage: {lineage} ({subtype}).")
            else:
                parts.append(f"Lineage: {lineage}.")

        growth = ann.get("growth_pattern", "")
        if growth:
            parts.append(f"Growth pattern: {growth}.")

        model_type = ann.get("model_type", "")
        if model_type:
            parts.append(f"Model type: {model_type}.")

        msi = ann.get("msi_status", "")
        if msi:
            parts.append(f"Microsatellite instability status: {msi}.")

        if ann.get("is_problematic"):
            parts.append("Warning: this cell line has known problems (contamination or misidentification).")

        wiki = self.get_wikipedia_info(cl_name or name)
        if wiki:
            parts.append(wiki)

        return " ".join(parts)
