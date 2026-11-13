"""Deep ChEMBL annotation for drugs and chemical perturbations.

:class:`~embpy.resources.molecule.MoleculeAnnotator` reaches three ChEMBL
endpoints -- ``molecule``, ``activity`` and ``mechanism`` -- which covers
structure, potency and mechanism of action.  ChEMBL_37 carries a good deal
more that a perturbation screen wants to know about its compounds:

* **Development status** -- clinical phase, year of first approval, whether
  the compound is an approved drug, a clinical candidate or a preclinical
  tool compound, its routes of administration and availability.
* **Indications** -- what the drug is given for, as MeSH headings and EFO
  terms, each with the phase reached *for that indication*.
* **Safety** -- black box warnings and withdrawals, with class, country,
  year and literature references.
* **ATC classification** -- the WHO therapeutic hierarchy, levels 1--5.
* **Synonyms** -- trade names, INN, USAN and research codes, which is how
  compounds are usually labelled in a screen's metadata.
* **Metabolism** -- metabolites, the enzymes responsible, and the conversion.
* **Molecular forms** -- the parent compound and its salts.
* **Target identity** -- UniProt accession, gene symbol and ChEMBL protein
  family for each target, rather than an opaque ``CHEMBL`` accession.
* **Analogs** -- structurally similar compounds by Tanimoto similarity.

Four ChEMBL behaviours shape this module.

First, drug-level annotation hangs off the **parent** molecule, not the salt
that was dosed.  ``mechanism?molecule_chembl_id=CHEMBL941`` -- imatinib --
returns nothing at all, because the mechanism row belongs to imatinib
mesylate (``CHEMBL1642``) and names ``CHEMBL941`` only as its parent.  Every
drug-level query here resolves the molecule hierarchy first and filters on
``parent_molecule_chembl_id``, which for imatinib is the difference between
0 and 4 mechanisms, and between 52 and 134 indications.

Second, ChEMBL silently ignores filter parameters it does not recognise and
returns the unfiltered collection -- ``mechanism.json?parent_chembl_id__in=``
(a misspelling) answers with all 7,561 mechanism rows.  Field names here
follow the ChEMBL_37 schema exactly, and every filtered response is checked
row by row against the keys that were asked for
(:class:`ChEMBLFilterIgnoredError`).

Third, the REST API fails intermittently: it answered 500 to roughly half
of all requests for about a day in November 2026.  :func:`_get_json` raises
:class:`ChEMBLServiceError` on 5xx, 429, timeouts and connection errors so
that an outage is never mistaken for "no data"; only a 404 means absent.

Fourth, per-compound requests do not scale -- a full annotation costs about
15 requests per parent.  Identity is therefore resolved by exact standard
InChIKey in batches (:meth:`ChEMBLAnnotator.resolve_many`), the per-source
getters have ``*_many`` batch forms using validated ``__in`` filters, and an
offline backend (``backend="sqlite"``) answers every getter from a local
ChEMBL SQLite release with the same return shapes and no network at all.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import statistics
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

CHEMBL_HOST = "https://www.ebi.ac.uk"
CHEMBL = f"{CHEMBL_HOST}/chembl/api/data"

# Fetch caps. Counts derived from a capped fetch are flagged ``*_at_limit``
# in :meth:`ChEMBLAnnotator.annotate_adata` so a saturated column is not
# mistaken for a measurement.
DEFAULT_ACTIVITIES = 200
DEFAULT_INDICATIONS = 100
DEFAULT_WARNINGS = 50
DEFAULT_SYNONYMS = 50
DEFAULT_METABOLITES = 50
DEFAULT_FORMS = 50
DEFAULT_ANALOGS = 25
DEFAULT_ANALOG_SIMILARITY = 70
DEFAULT_TARGETS = 50

#: Keys per batched ``__in`` request.  50 InChIKeys make a URL of about
#: 1.5 KB; the 2026-11 baseline used 25 and saw at most 21 rows per batch.
DEFAULT_BATCH = 50
#: Page size for filtered collection queries (the API maximum).
PAGE_LIMIT = 1000
#: Safety stop for pagination.  A filtered query never needs this many
#: pages; reaching it means the filter was not applied.
MAX_PAGES = 50

#: ``max_phase`` in ``molecule_dictionary``. Missing does not establish a
#: preclinical development programme; keep it unknown.
MAX_PHASE_LABELS: dict[float, str] = {
    4.0: "approved",
    3.0: "phase 3",
    2.0: "phase 2",
    1.0: "phase 1",
    0.5: "early phase 1",
    -1.0: "clinical phase unknown",
}

#: ``availability_type`` in ``molecule_dictionary``.
AVAILABILITY_LABELS: dict[int, str] = {
    -2: "withdrawn",
    -1: "unknown",
    0: "discontinued",
    1: "prescription only",
    2: "over the counter",
}

#: ``chirality`` in ``molecule_dictionary``.
CHIRALITY_LABELS: dict[int, str] = {
    2: "achiral",
    1: "single enantiomer",
    0: "mixture of stereoisomers",
    -1: "unknown",
}

ChEMBLSource = Literal[
    "identity",
    "development",
    "indications",
    "safety",
    "atc",
    "synonyms",
    "mechanisms",
    "activities",
    "targets",
    "metabolism",
    "forms",
    "analogs",
    "xrefs",
]

#: Everything :meth:`ChEMBLAnnotator.annotate` queries for ``sources="all"``.
#: ``analogs`` is excluded -- a similarity search is a structure search over
#: the whole database and costs far more than a keyed lookup, so it is
#: opt-in.
ALL_SOURCES: tuple[str, ...] = (
    "identity",
    "development",
    "indications",
    "safety",
    "atc",
    "synonyms",
    "mechanisms",
    "activities",
    "targets",
    "metabolism",
    "forms",
    "xrefs",
)

_CHEMBL_ID_RE = re.compile(r"^CHEMBL\d+$", re.IGNORECASE)
_INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
# Characters that can legally appear in a SMILES string. Used only to guess
# whether an identifier is a structure or a name when RDKit is unavailable.
_SMILES_CHARS_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]()=#$:/\\.%*]+$")


# =====================================================================
# Errors and HTTP
# =====================================================================


class ChEMBLServiceError(RuntimeError):
    """The ChEMBL web service failed to answer: 5xx, 429, timeout, no connection.

    Distinct from "no data": a 404 is reported as ``None`` by
    :func:`_get_json`, an empty collection as ``[]``.

    Attributes
    ----------
    status_code
        HTTP status, or ``None`` for timeouts and connection errors.
    url
        The request URL.
    kind
        ``"http_5xx"``, ``"rate_limited"``, ``"timeout"``, ``"connection"``,
        ``"invalid_json"``, ``"no_response"`` or ``"too_many_pages"``.
    """

    status = "service_error"

    def __init__(
        self,
        status_code: int | None = None,
        url: str = "",
        message: str = "",
        *,
        kind: str = "http_5xx",
    ) -> None:
        self.status_code = status_code
        self.url = url
        self.kind = kind
        detail = message or (f"HTTP {status_code}" if status_code else kind)
        super().__init__(f"ChEMBL {kind}: {detail} ({url})" if url else f"ChEMBL {kind}: {detail}")


class ChEMBLFilterIgnoredError(RuntimeError):
    """A filtered response contained rows that do not match the filter.

    ChEMBL answers an unsupported filter with the unfiltered collection, so
    one foreign row means none of the rows can be trusted.
    """

    status = "filter_ignored"

    def __init__(self, endpoint: str, field_name: str, foreign: list[Any]) -> None:
        self.endpoint = endpoint
        self.field_name = field_name
        self.foreign = foreign
        super().__init__(
            f"ChEMBL {endpoint}: {len(foreign)} row(s) outside the requested "
            f"{field_name} values (e.g. {foreign[:3]}); filter not applied"
        )


def _get_json(url: str, params: dict | None = None, timeout: int = 30) -> dict | None:
    """GET a ChEMBL JSON resource.

    Returns ``None`` for 404 (absent) and for other 4xx client errors, which
    the API also uses for malformed filter paths.  Raises
    :class:`ChEMBLServiceError` on 5xx, 429, timeouts, connection errors and
    non-JSON bodies, so callers can tell an outage from missing data.

    The name and signature are fixed: ``embpy_repro``'s ``Transport``
    replaces this function to add caching, pacing and circuit breaking.
    """
    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except requests.Timeout as exc:
        raise ChEMBLServiceError(None, url, str(exc), kind="timeout") from exc
    except requests.ConnectionError as exc:
        raise ChEMBLServiceError(None, url, str(exc), kind="connection") from exc
    except requests.RequestException as exc:
        raise ChEMBLServiceError(None, url, str(exc), kind="connection") from exc
    code = resp.status_code
    if code == 404:
        return None
    if code == 429:
        raise ChEMBLServiceError(code, url, kind="rate_limited")
    if code >= 500:
        raise ChEMBLServiceError(code, url, kind="http_5xx")
    if code >= 400:
        logger.warning("ChEMBL client error %s for %s: %s", code, url, resp.text[:200])
        return None
    try:
        return resp.json()
    except ValueError as exc:
        raise ChEMBLServiceError(code, url, "response is not JSON", kind="invalid_json") from exc


# =====================================================================
# Decoding helpers
# =====================================================================


def _as_float(value: Any) -> float | None:
    """ChEMBL returns numeric fields as strings; ``None`` and ``""`` are common."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    f = _as_float(value)
    return None if f is None else int(f)


def _flag(value: Any) -> bool | None:
    """Decode a ChEMBL flag.

    Most flags are ``1``/``0``.  Some (``prodrug``, ``first_in_class``,
    ``orphan``) additionally use ``-1`` for "preclinical compound, so the
    question does not apply", which is not the same as "no" and is mapped
    to ``None``.
    """
    if isinstance(value, bool):
        return value
    i = _as_int(value)
    if i is None or i < 0:
        return None
    return bool(i)


def _phase_label(max_phase: Any) -> str | None:
    f = _as_float(max_phase)
    if f is None:
        return None
    return MAX_PHASE_LABELS.get(f)


def _protein_class_path(class_desc: str | None) -> list[str]:
    """Split a ``protein_class_desc`` into its hierarchy levels.

    ChEMBL joins the levels with two spaces, e.g.
    ``"enzyme  kinase  protein kinase  tk  abl"``.
    """
    if not class_desc:
        return []
    return [part.strip() for part in class_desc.split("  ") if part.strip()]


def _chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    size = max(1, int(size))
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _synonym_sort_key(syn: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Order synonyms as the REST API does (a locale-style collation).

    Letters and digits first, ignoring case and punctuation; then
    punctuation and spaces; then lower case before upper case; then type.
    Checked against all 132 synonyms of CHEMBL25, CHEMBL941 and CHEMBL1642.
    """
    name = str(syn.get("molecule_synonym") or syn.get("synonyms") or "")
    return (re.sub(r"[\W_]+", "", name).upper(), name.upper(), name.swapcase(), str(syn.get("syn_type") or ""))


def _record_inchikey(record: Mapping[str, Any]) -> str | None:
    key = (record.get("molecule_structures") or {}).get("standard_inchi_key")
    return key.upper() if isinstance(key, str) else None


def smiles_to_inchikey(smiles: str) -> str | None:
    """Standard InChIKey of a SMILES with RDKit, or ``None`` if it does not parse.

    Raises
    ------
    ImportError
        If RDKit is not installed.
    """
    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        key = Chem.MolToInchiKey(mol)
    return key or None


def _rdkit_available() -> bool:
    try:
        import rdkit  # noqa: F401
    except ImportError:
        return False
    return True


# =====================================================================
# Result containers
# =====================================================================


class BatchResult(dict):
    """Result of a ``*_many`` batch getter.

    A ``dict`` keyed by requested identifier for every identifier whose batch
    succeeded -- an identifier with no rows maps to ``[]`` (or ``None`` for a
    record that does not exist).  Identifiers whose batch failed are absent
    from the mapping and listed in :attr:`failed` instead, so a provider
    failure can never be read as "no data".  Use :meth:`status` rather than
    ``.get`` to tell the cases apart.

    Attributes
    ----------
    failed
        ``{identifier: {"status": "batch_failed" | "filter_ignored", "error": str}}``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.failed: dict[str, dict[str, str]] = {}

    def status(self, key: str) -> str:
        """``"success"``, ``"not_found"``, ``"batch_failed"``, ``"filter_ignored"`` or ``"not_requested"``."""
        if key in self:
            return "not_found" if self[key] is None else "success"
        if key in self.failed:
            return self.failed[key]["status"]
        return "not_requested"


@dataclass(frozen=True)
class ChEMBLResolution:
    """Outcome of mapping a user identifier onto ChEMBL.

    Attributes
    ----------
    identifier
        The string that was looked up.
    molecule_chembl_id
        The matched molecule, or ``None`` if nothing matched.
    parent_chembl_id
        Parent of the matched molecule in the ChEMBL hierarchy -- the form
        drug-level annotation is registered against.  Equal to
        ``molecule_chembl_id`` for a compound that is its own parent.
    pref_name
        ChEMBL preferred name, when the compound has one.
    matched_by
        How the match was made: ``"chembl_id"``, ``"standard_inchi_key"``,
        ``"smiles_flexmatch"`` (opt-in, InChIKey-verified), ``"pref_name"``,
        ``"synonym"`` or ``"search"``.
    status
        ``"matched"``, ``"no_match"``, ``"ambiguous"``,
        ``"invalid_structure"``, ``"batch_failed"`` or ``"filter_ignored"``.
        The last two come only from :meth:`ChEMBLAnnotator.resolve_many`,
        where a failed batch must not read as "not in ChEMBL".
    inchikey
        The query's standard InChIKey, for structure inputs.
    matched_inchikey
        ``standard_inchi_key`` of the matched ChEMBL record.
    candidates
        ``(molecule_chembl_id, parent_chembl_id)`` pairs when ambiguous.
    error
        Failure detail for ``batch_failed`` / ``filter_ignored``.
    """

    identifier: str
    molecule_chembl_id: str | None = None
    parent_chembl_id: str | None = None
    pref_name: str | None = None
    matched_by: str | None = None
    status: str = "no_match"
    inchikey: str | None = None
    matched_inchikey: str | None = None
    candidates: tuple[tuple[str | None, str | None], ...] = field(default_factory=tuple)
    error: str | None = None

    @property
    def found(self) -> bool:
        """Whether the identifier matched a ChEMBL molecule."""
        return self.molecule_chembl_id is not None


# =====================================================================
# Offline backend: ChEMBL SQLite release
# =====================================================================

_SQLITE_BACKENDS: dict[str, ChEMBLSqlite] = {}
_SQLITE_LOCK = threading.Lock()
# SQLite's default limit on host parameters is 999 on old builds.
_SQL_CHUNK = 500


class ChEMBLSqlite:
    """Read-only access to a ChEMBL SQLite release, shaped like the REST API.

    Every method returns rows with the field names of the corresponding
    REST resource, so :class:`ChEMBLAnnotator` parses both backends with the
    same code.  Drug-level rows (mechanisms, indications, warnings,
    metabolism) are aggregated over every molecule whose
    ``molecule_hierarchy.parent_molregno`` is the requested parent, which is
    what the REST ``parent_molecule_chembl_id`` filter does.

    The database is opened as ``file:<path>?mode=ro&immutable=1``: no write
    can happen and no lock traffic reaches a network filesystem.  Each thread
    gets its own connection.

    Parameters
    ----------
    db_path
        Path to ``chembl_<release>.db``.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"ChEMBL SQLite database not found: {self.path}")
        self.uri = f"{self.path.as_uri()}?mode=ro&immutable=1"
        self._local = threading.local()
        self.tables = {
            r["name"] for r in self.query("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        self.release = self._read_release()

    @classmethod
    def shared(cls, db_path: str | Path) -> ChEMBLSqlite:
        """One backend per database file, shared by every annotator instance."""
        key = str(Path(db_path).expanduser().resolve())
        with _SQLITE_LOCK:
            backend = _SQLITE_BACKENDS.get(key)
            if backend is None:
                backend = _SQLITE_BACKENDS[key] = cls(key)
            return backend

    # -- plumbing ---------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.uri, uri=True, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        """Run one read-only query on this thread's connection."""
        return self._conn().execute(sql, tuple(params)).fetchall()

    def query_in(self, sql: str, values: Iterable[Any], extra: Iterable[Any] = ()) -> list[sqlite3.Row]:
        """Run ``sql`` with ``{in}`` replaced by a placeholder list, in chunks."""
        values = list(dict.fromkeys(values))
        extra = tuple(extra)
        rows: list[sqlite3.Row] = []
        for chunk in _chunks(values, _SQL_CHUNK):
            marks = ",".join("?" * len(chunk))
            rows.extend(self.query(sql.replace("{in}", marks), (*chunk, *extra)))
        return rows

    def _read_release(self) -> str:
        # ``version`` lists every bundled resource (BAO, EFO, RDKit, ...);
        # the release itself is the ``ChEMBL_<n>`` row.
        if "version" in self.tables:
            for row in self.query("SELECT name FROM version"):
                if re.fullmatch(r"ChEMBL_\d+", str(row["name"] or "")):
                    return str(row["name"])
        match = re.search(r"chembl_(\d+)", self.path.name, re.IGNORECASE)
        return f"ChEMBL_{match.group(1)}" if match else "ChEMBL_unknown"

    def _columns(self, table: str) -> list[str]:
        return [r["name"] for r in self.query(f"PRAGMA table_info({table})")]

    # -- molecules --------------------------------------------------------

    _MOLECULE_SQL = """
        SELECT md.*, cs.canonical_smiles AS _canonical_smiles,
               cs.standard_inchi AS _standard_inchi,
               cs.standard_inchi_key AS _standard_inchi_key,
               COALESCE(pmd.chembl_id, md.chembl_id) AS _parent_chembl_id,
               COALESCE(amd.chembl_id, md.chembl_id) AS _active_chembl_id,
               COALESCE(mh.parent_molregno, md.molregno) AS _parent_molregno
        FROM molecule_dictionary md
        LEFT JOIN compound_structures cs ON cs.molregno = md.molregno
        LEFT JOIN molecule_hierarchy mh ON mh.molregno = md.molregno
        LEFT JOIN molecule_dictionary pmd ON pmd.molregno = mh.parent_molregno
        LEFT JOIN molecule_dictionary amd ON amd.molregno = mh.active_molregno
    """
    _SKIP = {"molregno", "chembl_id", "chebi_par_id", "indication_class"}

    def _records(self, rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
        if not rows:
            return []
        molregnos = [r["molregno"] for r in rows]
        parents = {r["molregno"] for r in rows if r["molregno"] == r["_parent_molregno"]}
        synonyms: dict[int, list[dict[str, Any]]] = {}
        for r in self.query_in(
            "SELECT molregno, synonyms, syn_type FROM molecule_synonyms "
            "WHERE molregno IN ({in}) ORDER BY molsyn_id",
            molregnos,
        ):
            synonyms.setdefault(r["molregno"], []).append(
                {"molecule_synonym": r["synonyms"], "syn_type": r["syn_type"], "synonyms": r["synonyms"]}
            )
        # The API lists synonyms in a locale collation; reproduce it so the
        # priority-stable ordering in get_synonyms is the same offline.
        for rows_ in synonyms.values():
            rows_.sort(key=_synonym_sort_key)
        atc = self._atc_codes(molregnos, parents)
        out = []
        for r in rows:
            record = {k: r[k] for k in r.keys() if not k.startswith("_") and k not in self._SKIP}
            record.update(
                molecule_chembl_id=r["chembl_id"],
                molecule_hierarchy={
                    "molecule_chembl_id": r["chembl_id"],
                    "parent_chembl_id": r["_parent_chembl_id"],
                    "active_chembl_id": r["_active_chembl_id"],
                },
                molecule_structures=(
                    {
                        "canonical_smiles": r["_canonical_smiles"],
                        "standard_inchi": r["_standard_inchi"],
                        "standard_inchi_key": r["_standard_inchi_key"],
                    }
                    if r["_standard_inchi_key"] or r["_canonical_smiles"]
                    else None
                ),
                atc_classifications=atc.get(r["molregno"], []),
                molecule_synonyms=synonyms.get(r["molregno"], []),
                # The REST record's cross_references (DailyMed, EMA EPAR,
                # Wikipedia) have no table in the SQLite release.
                cross_references=[],
            )
            out.append(record)
        return out

    def _atc_codes(self, molregnos: list[int], parents: set[int]) -> dict[int, list[str]]:
        """ATC codes per molecule; a parent also collects its salts' codes."""
        own: dict[int, list[str]] = {}
        for r in self.query_in(
            "SELECT molregno, level5 FROM molecule_atc_classification "
            "WHERE molregno IN ({in}) ORDER BY mol_atc_id",
            molregnos,
        ):
            own.setdefault(r["molregno"], []).append(r["level5"])
        if parents:
            for r in self.query_in(
                "SELECT mh.parent_molregno AS parent, mac.level5 FROM molecule_hierarchy mh "
                "JOIN molecule_atc_classification mac ON mac.molregno = mh.molregno "
                "WHERE mh.parent_molregno IN ({in}) AND mh.molregno != mh.parent_molregno "
                "ORDER BY mac.mol_atc_id",
                parents,
            ):
                codes = own.setdefault(r["parent"], [])
                if r["level5"] not in codes:
                    codes.append(r["level5"])
        return own

    def molecules_by_chembl_id(self, chembl_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """REST-shaped ``molecule`` records keyed by ChEMBL ID (absent IDs are missing)."""
        ids = [c.upper() for c in chembl_ids]
        rows = self.query_in(self._MOLECULE_SQL + " WHERE md.chembl_id IN ({in})", ids)
        return {rec["molecule_chembl_id"]: rec for rec in self._records(rows)}

    def molecules_by_inchikey(self, inchikeys: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
        """Records whose ``standard_inchi_key`` equals each key (``[]`` when none)."""
        keys = [k.upper() for k in inchikeys]
        rows = self.query_in(self._MOLECULE_SQL + " WHERE cs.standard_inchi_key IN ({in})", keys)
        out: dict[str, list[dict[str, Any]]] = {k: [] for k in keys}
        for rec in self._records(rows):
            out.setdefault(_record_inchikey(rec) or "", []).append(rec)
        return out

    def molecules_by_name(self, name: str) -> list[dict[str, Any]]:
        """Records whose ``pref_name`` equals ``name`` (ChEMBL names are upper case)."""
        rows = self.query(self._MOLECULE_SQL + " WHERE md.pref_name = ? LIMIT 5", (name.upper(),))
        return self._records(rows)

    def molecules_by_synonym(self, name: str) -> list[dict[str, Any]]:
        """Records with a synonym equal to ``name``, ignoring case."""
        regs = [
            r["molregno"]
            for r in self.query(
                "SELECT DISTINCT molregno FROM molecule_synonyms WHERE UPPER(synonyms) = ? LIMIT 5",
                (name.upper(),),
            )
        ]
        if not regs:
            return []
        return self._records(self.query_in(self._MOLECULE_SQL + " WHERE md.molregno IN ({in})", regs))

    # -- the parent family --------------------------------------------------

    def _family(self, parents: Iterable[str]) -> tuple[dict[int, str], dict[str, Any]]:
        """``{molregno: parent_chembl_id}`` for every molecule under each parent.

        A molecule belongs to parent ``P`` when its ``parent_molregno`` is
        ``P`` (or it has no hierarchy row and is ``P`` itself), which is the
        REST ``parent_molecule_chembl_id`` semantics: a salt passed as the
        "parent" has no family.  Also returns ``{parent_chembl_id: row}``.
        """
        ids = [p.upper() for p in parents]
        prow = {
            r["chembl_id"]: r
            for r in self.query_in(
                "SELECT md.molregno, md.chembl_id, md.max_phase, mh.parent_molregno "
                "FROM molecule_dictionary md LEFT JOIN molecule_hierarchy mh ON mh.molregno = md.molregno "
                "WHERE md.chembl_id IN ({in})",
                ids,
            )
        }
        by_regno = {r["molregno"]: cid for cid, r in prow.items()}
        family: dict[int, str] = {}
        for cid, r in prow.items():
            if r["parent_molregno"] is None or r["parent_molregno"] == r["molregno"]:
                family[r["molregno"]] = cid
        if by_regno:
            for r in self.query_in(
                "SELECT molregno, parent_molregno FROM molecule_hierarchy WHERE parent_molregno IN ({in})",
                list(by_regno),
            ):
                family[r["molregno"]] = by_regno[r["parent_molregno"]]
        return family, dict(prow)

    def _chembl_ids(self, molregnos: Iterable[int]) -> dict[int, str]:
        return {
            r["molregno"]: r["chembl_id"]
            for r in self.query_in(
                "SELECT molregno, chembl_id FROM molecule_dictionary WHERE molregno IN ({in})",
                [m for m in molregnos if m is not None],
            )
        }

    def _refs(self, table: str, key: str, ids: list[Any]) -> dict[Any, list[dict[str, Any]]]:
        out: dict[Any, list[dict[str, Any]]] = {}
        if table not in self.tables or not ids:
            return out
        for r in self.query_in(
            f"SELECT {key} AS k, ref_type, ref_id, ref_url FROM {table} WHERE {key} IN ({{in}})", ids
        ):
            out.setdefault(r["k"], []).append(
                {"ref_id": r["ref_id"], "ref_type": r["ref_type"], "ref_url": r["ref_url"]}
            )
        return out

    # -- drug-level resources ---------------------------------------------

    def mechanisms(self, parents: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
        """``mechanism`` rows of every form of each parent, with references."""
        parents = [p.upper() for p in parents]
        family, prow = self._family(parents)
        out: dict[str, list[dict[str, Any]]] = {p: [] for p in parents}
        if not family:
            return out
        rows = self.query_in(
            "SELECT dm.*, md.chembl_id AS molecule_chembl_id, td.chembl_id AS target_chembl_id "
            "FROM drug_mechanism dm JOIN molecule_dictionary md ON md.molregno = dm.molregno "
            "LEFT JOIN target_dictionary td ON td.tid = dm.tid "
            "WHERE dm.molregno IN ({in}) ORDER BY dm.mec_id",
            list(family),
        )
        refs = self._refs("mechanism_refs", "mec_id", [r["mec_id"] for r in rows])
        for r in rows:
            parent = family[r["molregno"]]
            out[parent].append(
                {
                    "action_type": r["action_type"],
                    "binding_site_comment": r["binding_site_comment"],
                    "direct_interaction": r["direct_interaction"],
                    "disease_efficacy": r["disease_efficacy"],
                    "max_phase": prow[parent]["max_phase"],
                    "mec_id": r["mec_id"],
                    "mechanism_comment": r["mechanism_comment"],
                    "mechanism_of_action": r["mechanism_of_action"],
                    "molecular_mechanism": r["molecular_mechanism"],
                    "molecule_chembl_id": r["molecule_chembl_id"],
                    "parent_molecule_chembl_id": parent,
                    "record_id": r["record_id"],
                    "selectivity_comment": r["selectivity_comment"],
                    "site_id": r["site_id"],
                    "target_chembl_id": r["target_chembl_id"],
                    "mechanism_refs": refs.get(r["mec_id"], []),
                }
            )
        return out

    def indications(self, parents: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
        """``drug_indication`` rows of every form of each parent, with references."""
        parents = [p.upper() for p in parents]
        family, _ = self._family(parents)
        out: dict[str, list[dict[str, Any]]] = {p: [] for p in parents}
        if not family:
            return out
        rows = self.query_in(
            "SELECT di.*, md.chembl_id AS molecule_chembl_id FROM drug_indication di "
            "JOIN molecule_dictionary md ON md.molregno = di.molregno "
            "WHERE di.molregno IN ({in}) ORDER BY di.drugind_id",
            list(family),
        )
        refs = self._refs("indication_refs", "drugind_id", [r["drugind_id"] for r in rows])
        for r in rows:
            parent = family[r["molregno"]]
            out[parent].append(
                {
                    "drugind_id": r["drugind_id"],
                    "efo_id": r["efo_id"],
                    "efo_term": r["efo_term"],
                    "indication_refs": refs.get(r["drugind_id"], []),
                    "max_phase_for_ind": r["max_phase_for_ind"],
                    "mesh_heading": r["mesh_heading"],
                    "mesh_id": r["mesh_id"],
                    "molecule_chembl_id": r["molecule_chembl_id"],
                    "parent_molecule_chembl_id": parent,
                }
            )
        return out

    def warnings(self, parents: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
        """``drug_warning`` rows of every form of each parent, with references."""
        parents = [p.upper() for p in parents]
        family, _ = self._family(parents)
        out: dict[str, list[dict[str, Any]]] = {p: [] for p in parents}
        if not family:
            return out
        rows = self.query_in(
            "SELECT dw.*, md.chembl_id AS molecule_chembl_id FROM drug_warning dw "
            "JOIN molecule_dictionary md ON md.molregno = dw.molregno "
            "WHERE dw.molregno IN ({in}) ORDER BY dw.warning_id",
            list(family),
        )
        refs = self._refs("warning_refs", "warning_id", [r["warning_id"] for r in rows])
        for r in rows:
            parent = family[r["molregno"]]
            keys = r.keys()
            out[parent].append(
                {
                    "efo_id": r["efo_id"],
                    "efo_id_for_warning_class": r["efo_id_for_warning_class"]
                    if "efo_id_for_warning_class" in keys
                    else None,
                    "efo_term": r["efo_term"],
                    "molecule_chembl_id": r["molecule_chembl_id"],
                    "parent_molecule_chembl_id": parent,
                    "warning_class": r["warning_class"],
                    "warning_country": r["warning_country"],
                    "warning_description": r["warning_description"],
                    "warning_id": r["warning_id"],
                    "warning_refs": refs.get(r["warning_id"], []),
                    "warning_type": r["warning_type"],
                    "warning_year": r["warning_year"],
                }
            )
        return out

    def metabolism(self, parents: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
        """Metabolism rows whose drug record is any form of the parent."""
        parents = [p.upper() for p in parents]
        family, _ = self._family(parents)
        out: dict[str, list[dict[str, Any]]] = {p: [] for p in parents}
        if not family:
            return out
        rows = self.query_in(
            "SELECT m.*, dr.molregno AS drug_molregno, sr.molregno AS substrate_molregno, "
            "mr.molregno AS metabolite_molregno, sr.compound_name AS substrate_record_name, "
            "mr.compound_name AS metabolite_record_name, td.chembl_id AS enzyme_chembl_id "
            "FROM metabolism m JOIN compound_records dr ON dr.record_id = m.drug_record_id "
            "LEFT JOIN compound_records sr ON sr.record_id = m.substrate_record_id "
            "LEFT JOIN compound_records mr ON mr.record_id = m.metabolite_record_id "
            "LEFT JOIN target_dictionary td ON td.tid = m.enzyme_tid "
            "WHERE dr.molregno IN ({in}) ORDER BY m.met_id",
            list(family),
        )
        regnos = {r[k] for r in rows for k in ("drug_molregno", "substrate_molregno", "metabolite_molregno")}
        names = {
            r["molregno"]: (r["chembl_id"], r["pref_name"])
            for r in self.query_in(
                "SELECT molregno, chembl_id, pref_name FROM molecule_dictionary WHERE molregno IN ({in})",
                [m for m in regnos if m is not None],
            )
        }
        refs = self._refs("metabolism_refs", "met_id", [r["met_id"] for r in rows])
        for r in rows:
            parent = family[r["drug_molregno"]]
            sub = names.get(r["substrate_molregno"], (None, None))
            met = names.get(r["metabolite_molregno"], (None, None))
            out[parent].append(
                {
                    "drug_chembl_id": names.get(r["drug_molregno"], (None,))[0],
                    "enzyme_name": r["enzyme_name"],
                    "met_comment": r["met_comment"],
                    "met_conversion": r["met_conversion"],
                    "met_id": r["met_id"],
                    "metabolism_refs": refs.get(r["met_id"], []),
                    "metabolite_chembl_id": met[0],
                    "metabolite_name": met[1] or r["metabolite_record_name"],
                    "organism": r["organism"],
                    "pathway_id": r["pathway_id"],
                    "pathway_key": r["pathway_key"],
                    "substrate_chembl_id": sub[0],
                    "substrate_name": sub[1] or r["substrate_record_name"],
                    "target_chembl_id": r["enzyme_chembl_id"],
                    "tax_id": r["tax_id"],
                }
            )
        return out

    def molecule_forms(self, parents: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
        """``molecule_form`` rows: each parent and the molecules under it."""
        parents = [p.upper() for p in parents]
        family, _ = self._family(parents)
        out: dict[str, list[dict[str, Any]]] = {p: [] for p in parents}
        ids = self._chembl_ids(family)
        for regno in sorted(family):
            parent = family[regno]
            out[parent].append(
                {
                    "is_parent": ids.get(regno) == parent,
                    "molecule_chembl_id": ids.get(regno),
                    "parent_chembl_id": parent,
                }
            )
        return out

    def drug(self, parent: str) -> dict[str, Any] | None:
        """An approximation of the REST ``drug`` resource.

        ``drug.json`` is a derived API view, not a table.  This rebuilds
        ``first_approval``, the ATC codes (with the four level descriptions
        joined as the API does) and the applicants (``products`` via
        ``formulations``) for every form of the parent.  ``drug_type``,
        ``helm_notation`` and ``biotherapeutic`` are not derived and are
        ``None``.  Returns ``None`` for molecules without a clinical phase.
        """
        family, prow = self._family([parent])
        row = prow.get(parent.upper())
        if row is None or row["max_phase"] is None:
            return None
        record = self.molecules_by_chembl_id([parent]).get(parent.upper()) or {}
        codes = record.get("atc_classifications") or []
        atc = self.atc_classes(codes)
        applicants: list[str] = []
        if {"formulations", "products"} <= self.tables and family:
            for r in self.query_in(
                "SELECT DISTINCT p.applicant_full_name AS name FROM formulations f "
                "JOIN products p ON p.product_id = f.product_id "
                "WHERE f.molregno IN ({in}) AND p.applicant_full_name IS NOT NULL ORDER BY 1",
                list(family),
            ):
                applicants.append(r["name"])
        return {
            "molecule_chembl_id": parent.upper(),
            "first_approval": record.get("first_approval"),
            "drug_type": None,
            "applicants": applicants,
            "atc_classification": [
                {
                    "code": code,
                    "description": ": ".join(
                        str(atc[code].get(f"level{i}_description") or "") for i in range(1, 5)
                    )
                    if code in atc
                    else None,
                }
                for code in codes
            ],
            "helm_notation": None,
            "biotherapeutic": None,
        }

    def atc_classes(self, codes: Iterable[str]) -> dict[str, dict[str, Any]]:
        """``atc_classification`` rows keyed by level-5 code."""
        return {
            r["level5"]: dict(r)
            for r in self.query_in("SELECT * FROM atc_classification WHERE level5 IN ({in})", codes)
        }

    # -- targets ----------------------------------------------------------

    def targets(self, target_chembl_ids: Iterable[str]) -> list[dict[str, Any]]:
        """Target rows with components, gene-symbol synonyms and class IDs."""
        rows = self.query_in(
            "SELECT tid, chembl_id, pref_name, target_type, organism, tax_id "
            "FROM target_dictionary WHERE chembl_id IN ({in})",
            target_chembl_ids,
        )
        tids = [r["tid"] for r in rows]
        comps: dict[int, list[dict[str, Any]]] = {}
        comp_ids: list[int] = []
        for r in self.query_in(
            "SELECT tc.tid, tc.component_id, cs.accession, cs.component_type, cs.description "
            "FROM target_components tc JOIN component_sequences cs ON cs.component_id = tc.component_id "
            "WHERE tc.tid IN ({in}) ORDER BY tc.component_id",
            tids,
        ):
            comp_ids.append(r["component_id"])
            comps.setdefault(r["tid"], []).append(
                {
                    "accession": r["accession"],
                    "component_id": r["component_id"],
                    "component_type": r["component_type"],
                    "component_description": r["description"],
                    "target_component_synonyms": [],
                    "protein_classifications": [],
                }
            )
        synonyms: dict[int, list[dict[str, Any]]] = {}
        for r in self.query_in(
            "SELECT component_id, component_synonym, syn_type FROM component_synonyms "
            "WHERE component_id IN ({in}) ORDER BY compsyn_id",
            comp_ids,
        ):
            synonyms.setdefault(r["component_id"], []).append(
                {"component_synonym": r["component_synonym"], "syn_type": r["syn_type"]}
            )
        classes: dict[int, list[dict[str, Any]]] = {}
        for r in self.query_in(
            "SELECT component_id, protein_class_id FROM component_class "
            "WHERE component_id IN ({in}) ORDER BY comp_class_id",
            comp_ids,
        ):
            classes.setdefault(r["component_id"], []).append(
                {"protein_classification_id": r["protein_class_id"]}
            )
        out = []
        for r in rows:
            components = comps.get(r["tid"], [])
            for comp in components:
                comp["target_component_synonyms"] = synonyms.get(comp["component_id"], [])
                comp["protein_classifications"] = classes.get(comp["component_id"], [])
            out.append(
                {
                    "target_chembl_id": r["chembl_id"],
                    "pref_name": r["pref_name"],
                    "target_type": r["target_type"],
                    "organism": r["organism"],
                    "tax_id": r["tax_id"],
                    "target_components": components,
                }
            )
        return out

    def protein_classes(self, class_ids: Iterable[int]) -> list[dict[str, Any]]:
        """``protein_classification`` rows for the given class IDs."""
        return [
            dict(r)
            for r in self.query_in(
                "SELECT * FROM protein_classification WHERE protein_class_id IN ({in})",
                [int(c) for c in class_ids],
            )
        ]

    # -- activities ---------------------------------------------------------

    def activities(
        self, parent: str, limit: int | None, only_with_pchembl: bool = True
    ) -> list[dict[str, Any]]:
        """Top-``limit`` activities, or all rows when ``limit=None``.

        Use the complete table before quality/species/assay filtering for
        target inference. A potency-ordered row cap biases target coverage.
        """
        family, _ = self._family([parent])
        if not family:
            return []
        has_action = "action_type" in self._columns("activities")
        has_le = "ligand_eff" in self.tables
        has_bao = "bioassay_ontology" in self.tables
        select = [
            "a.activity_id, a.molregno, a.standard_type, a.standard_value, a.standard_units,",
            "a.standard_relation, a.pchembl_value, a.activity_comment, a.data_validity_comment,",
            "a.potential_duplicate, s.chembl_id AS assay_chembl_id, s.assay_type,",
            "s.description AS assay_description, td.chembl_id AS target_chembl_id,",
            "td.pref_name AS target_pref_name, td.organism AS target_organism,",
            "td.tax_id AS target_tax_id, d.chembl_id AS document_chembl_id,",
            "d.year AS document_year, d.journal AS document_journal",
        ]
        joins = [
            "FROM activities a JOIN assays s ON s.assay_id = a.assay_id",
            "LEFT JOIN target_dictionary td ON td.tid = s.tid",
            "LEFT JOIN docs d ON d.doc_id = a.doc_id",
        ]
        if has_action:
            select.append(", a.action_type AS act_action_type")
            if "action_type" in self.tables:
                select.append(", act.description AS act_action_description, act.parent_type AS act_parent_type")
                joins.append("LEFT JOIN action_type act ON act.action_type = a.action_type")
        if has_bao:
            select.append(", b.label AS bao_label")
            joins.append("LEFT JOIN bioassay_ontology b ON b.bao_id = s.bao_format")
        if has_le:
            select.append(", le.bei, le.le, le.lle, le.sei")
            joins.append("LEFT JOIN ligand_eff le ON le.activity_id = a.activity_id")
        # The unary ``+`` keeps the planner on the molregno index: with
        # ``ORDER BY pchembl_value ... LIMIT`` it may otherwise walk the
        # 24M-row pchembl index looking for this compound's rows.
        where = "WHERE a.molregno IN ({in})"
        order = "ORDER BY a.activity_id"
        if only_with_pchembl:
            where += " AND +a.pchembl_value IS NOT NULL"
            order = "ORDER BY +a.pchembl_value DESC, a.activity_id"
        sql = f"SELECT {' '.join(select)} {' '.join(joins)} {where} {order}"
        regnos = list(family)
        if len(regnos) > _SQL_CHUNK:  # pragma: no cover - families are small
            regnos = regnos[:_SQL_CHUNK]
        marks = ",".join("?" * len(regnos))
        if limit is not None:
            if limit < 1:
                raise ValueError("Activity limit must be positive or None")
            sql += " LIMIT ?"
        rows = self.query(sql.replace("{in}", marks),
                          (*regnos, int(limit)) if limit is not None else tuple(regnos))
        ids = self._chembl_ids({r["molregno"] for r in rows})
        out = []
        for r in rows:
            keys = r.keys()
            action = None
            if "act_action_type" in keys and r["act_action_type"]:
                action = {
                    "action_type": r["act_action_type"],
                    "description": r["act_action_description"] if "act_action_description" in keys else None,
                    "parent_type": r["act_parent_type"] if "act_parent_type" in keys else None,
                }
            le = None
            if "bei" in keys and any(r[k] is not None for k in ("bei", "le", "lle", "sei")):
                le = {k: (None if r[k] is None else str(r[k])) for k in ("bei", "le", "lle", "sei")}
            out.append(
                {
                    "activity_id": r["activity_id"],
                    "molecule_chembl_id": ids.get(r["molregno"]),
                    "parent_molecule_chembl_id": parent.upper(),
                    "target_chembl_id": r["target_chembl_id"],
                    "target_pref_name": r["target_pref_name"],
                    "target_organism": r["target_organism"],
                    "target_tax_id": r["target_tax_id"],
                    "standard_type": r["standard_type"],
                    "standard_value": r["standard_value"],
                    "standard_units": r["standard_units"],
                    "standard_relation": r["standard_relation"],
                    "pchembl_value": r["pchembl_value"],
                    "action_type": action,
                    "assay_chembl_id": r["assay_chembl_id"],
                    "assay_type": r["assay_type"],
                    "assay_description": r["assay_description"],
                    "bao_label": r["bao_label"] if "bao_label" in keys else None,
                    "ligand_efficiency": le,
                    "document_chembl_id": r["document_chembl_id"],
                    "document_year": r["document_year"],
                    "document_journal": r["document_journal"],
                    "data_validity_comment": r["data_validity_comment"],
                    "potential_duplicate": r["potential_duplicate"],
                    "activity_comment": r["activity_comment"],
                }
            )
        return out


# =====================================================================
# Annotator
# =====================================================================


class ChEMBLAnnotator:
    """Aggregate drug and perturbation annotations from ChEMBL.

    Parameters
    ----------
    rate_limit_delay
        Seconds to wait between API calls (default 0.2).  Ignored by the
        SQLite backend.
    cache
        Cache identifier resolutions, molecule records and protein-class
        lookups on the instance.  A full annotation touches the molecule
        record from several methods, and a screen's compounds share target
        classes, so caching cuts the request count substantially.  Set to
        ``False`` to always hit the API.
    backend
        ``"rest"`` (default) queries the ChEMBL web services.  ``"sqlite"``
        answers every method from a local ChEMBL SQLite release given by
        ``db_path``, with the same return shapes and no network access.
    db_path
        ``chembl_<release>.db`` for ``backend="sqlite"``.
    flexmatch
        Allow a ``canonical_smiles__flexmatch`` structure search when the
        exact InChIKey lookup of a SMILES finds nothing.  Off by default:
        it is a slow structure search, and its hits are accepted only when
        their ``standard_inchi_key`` equals the query's.
    batch_errors
        Extra exception types that mark one batch of a ``*_many`` call as
        failed (and trigger bisection) instead of propagating.
        :class:`ChEMBLServiceError` is always included.  Anything else a
        replaced :func:`_get_json` raises propagates unchanged.

    Attributes
    ----------
    release
        ``"ChEMBL_37"`` etc. for the SQLite backend; ``None`` for REST.

    Notes
    -----
    A full :meth:`annotate` call issues roughly a dozen requests per
    compound.  For a screen, resolve identity with :meth:`resolve_many`,
    fetch parents with :meth:`get_molecule_records_many` (which also primes
    the per-compound methods), and use the ``*_many`` getters.

    Examples
    --------
    >>> ann = ChEMBLAnnotator()
    >>> status = ann.get_development_status("imatinib")
    >>> status["development_phase"]
    'approved'
    >>> [m["mechanism"] for m in ann.get_mechanisms("imatinib")][:1]
    ['Tyrosine-protein kinase ABL inhibitor']
    """

    def __init__(
        self,
        rate_limit_delay: float = 0.2,
        cache: bool = True,
        *,
        backend: Literal["rest", "sqlite"] = "rest",
        db_path: str | Path | None = None,
        flexmatch: bool = False,
        batch_errors: tuple[type[BaseException], ...] = (),
    ) -> None:
        if backend not in ("rest", "sqlite"):
            raise ValueError(f"backend must be 'rest' or 'sqlite', not {backend!r}")
        self.backend = backend
        self._sql: ChEMBLSqlite | None = None
        if backend == "sqlite":
            if db_path is None:
                raise ValueError("backend='sqlite' needs db_path")
            self._sql = ChEMBLSqlite.shared(db_path)
        self.release: str | None = self._sql.release if self._sql else None
        self.delay = 0.0 if self._sql else rate_limit_delay
        self.cache = cache
        self.flexmatch = flexmatch
        self._batch_errors: tuple[type[BaseException], ...] = (ChEMBLServiceError, *batch_errors)
        self._resolutions: dict[str, ChEMBLResolution] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self._primed: set[str] = set()
        self._protein_classes: dict[int, dict[str, Any]] = {}
        self._target_details: dict[str, dict[str, Any]] = {}
        self._component_classes: dict[int, list[int]] = {}
        self.activity_fetch_metadata: dict[str, dict[str, Any]] = {}

    def _sleep(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    # ==================================================================
    # REST plumbing
    # ==================================================================

    def _paged(
        self,
        endpoint: str,
        params: dict[str, Any],
        key: str,
        *,
        check: tuple[str | Callable[[dict[str, Any]], Any], set[str]] | None = None,
        max_rows: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch a filtered collection, following ``page_meta.next``.

        ``check=(field, allowed)`` validates every row as it arrives: a row
        whose ``field`` is not in ``allowed`` means the filter was ignored,
        and raises :class:`ChEMBLFilterIgnoredError` before a second page is
        requested.  A ``None`` response (legacy error convention of a
        replaced helper, or a 404 on a collection) raises
        :class:`ChEMBLServiceError` -- a collection query never legitimately
        returns nothing.
        """
        url: str = f"{CHEMBL}/{endpoint}.json"
        query: dict[str, Any] | None = {**params, "format": "json"}
        rows: list[dict[str, Any]] = []
        for _ in range(MAX_PAGES):
            self._sleep()
            data = _get_json(url, params=query)
            if data is None:
                raise ChEMBLServiceError(None, url, "no response for a collection query", kind="no_response")
            page = data.get(key) or []
            if check is not None:
                self._check_rows(endpoint, page, *check)
            rows.extend(page)
            if max_rows is not None and len(rows) >= max_rows:
                return rows[:max_rows]
            nxt = (data.get("page_meta") or {}).get("next")
            if not nxt:
                return rows
            url, query = urljoin(CHEMBL_HOST, nxt), None
        raise ChEMBLServiceError(None, url, f"more than {MAX_PAGES} pages", kind="too_many_pages")

    @staticmethod
    def _check_rows(
        endpoint: str,
        rows: list[dict[str, Any]],
        field_name: str | Callable[[dict[str, Any]], Any],
        allowed: set[str],
    ) -> None:
        def value(row: dict[str, Any]) -> Any:
            v = field_name(row) if callable(field_name) else row.get(field_name)
            return v.upper() if isinstance(v, str) else v

        foreign = [value(r) for r in rows if value(r) not in allowed]
        if foreign:
            name = field_name if isinstance(field_name, str) else getattr(field_name, "__name__", "key")
            raise ChEMBLFilterIgnoredError(endpoint, name, foreign)

    def _batched(
        self,
        keys: Iterable[str],
        fetch_chunk: Callable[[list[str]], dict[str, Any]],
        chunk_size: int,
    ) -> BatchResult:
        """Run ``fetch_chunk`` over chunks; bisect failed chunks down to one key."""
        result = BatchResult()
        for chunk in _chunks(list(dict.fromkeys(keys)), chunk_size):
            self._run_chunk(chunk, fetch_chunk, result)
        return result

    def _run_chunk(
        self,
        chunk: list[str],
        fetch_chunk: Callable[[list[str]], dict[str, Any]],
        result: BatchResult,
    ) -> None:
        try:
            values = fetch_chunk(chunk)
        except ChEMBLFilterIgnoredError as exc:
            for k in chunk:
                result.failed[k] = {"status": "filter_ignored", "error": str(exc)}
            return
        except self._batch_errors as exc:
            if len(chunk) > 1:
                mid = (len(chunk) + 1) // 2
                logger.info("ChEMBL batch of %d failed (%s); bisecting", len(chunk), exc)
                self._run_chunk(chunk[:mid], fetch_chunk, result)
                self._run_chunk(chunk[mid:], fetch_chunk, result)
                return
            result.failed[chunk[0]] = {"status": "batch_failed", "error": f"{type(exc).__name__}: {exc}"}
            return
        for k in chunk:
            result[k] = values.get(k)

    def _rows_many(
        self,
        keys: Iterable[str],
        endpoint: str,
        filter_field: str,
        row_field: str,
        collection: str,
        chunk_size: int,
    ) -> BatchResult:
        """``{key: [rows]}`` from ``<endpoint>.json?<filter_field>__in=...``."""

        def fetch(chunk: list[str]) -> dict[str, list[dict[str, Any]]]:
            allowed = {k.upper() for k in chunk}
            rows = self._paged(
                endpoint,
                {f"{filter_field}__in": ",".join(chunk), "limit": PAGE_LIMIT},
                collection,
                check=(row_field, allowed),
            )
            grouped: dict[str, list[dict[str, Any]]] = {k: [] for k in chunk}
            upper = {k.upper(): k for k in chunk}
            for row in rows:
                grouped[upper[str(row.get(row_field)).upper()]].append(row)
            return grouped

        return self._batched([k.upper() for k in keys if k], fetch, chunk_size)

    # ==================================================================
    # Identifier resolution
    # ==================================================================

    @staticmethod
    def _classify(identifier: str) -> str:
        """Guess whether an identifier is a ChEMBL ID, an InChIKey, a SMILES or a name."""
        s = identifier.strip()
        if _CHEMBL_ID_RE.match(s):
            return "chembl_id"
        if _INCHIKEY_RE.match(s):
            return "inchikey"
        try:
            from rdkit import Chem, rdBase

            # Speculative parse: a compound *name* failing here is the expected
            # path. BlockLogs rather than RDLogger.DisableLog because the latter
            # is global and permanent -- it silenced RDKit for the rest of the
            # caller's session as a side effect of one identifier lookup.
            with rdBase.BlockLogs():
                parsed = Chem.MolFromSmiles(s)
            return "smiles" if parsed is not None else "name"
        except ImportError:
            pass
        # Without RDKit a token like "CCO" is ambiguous -- it is both a valid
        # SMILES and a plausible code -- and no InChIKey can be computed, so
        # resolve() treats it as a name.
        return "smiles" if _SMILES_CHARS_RE.match(s) else "name"

    def prime_records(self, records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]]) -> None:
        """Seed the molecule-record cache with records fetched elsewhere.

        Per-compound methods then reuse them instead of requesting
        ``molecule/{id}.json`` once per source.  Accepts an iterable of
        records or ``{chembl_id: record}`` (``None`` values are skipped).
        """
        values = records.values() if isinstance(records, Mapping) else records
        for record in values:
            cid = (record or {}).get("molecule_chembl_id")
            if cid:
                self._records[str(cid).upper()] = dict(record)  # type: ignore[arg-type]
                self._primed.add(str(cid).upper())

    def get_molecule_record(self, chembl_id: str) -> dict[str, Any] | None:
        """Fetch the full ``molecule`` record for a ChEMBL ID (``None`` if absent)."""
        key = chembl_id.strip().upper()
        if key in self._records and (self.cache or key in self._primed):
            return self._records[key]
        if self._sql is not None:
            record = self._sql.molecules_by_chembl_id([key]).get(key)
        else:
            self._sleep()
            record = _get_json(f"{CHEMBL}/molecule/{key}.json")
        if record and self.cache:
            self._records[key] = record
        return record

    def get_molecule_records_many(
        self, chembl_ids: Iterable[str], chunk_size: int = DEFAULT_BATCH
    ) -> BatchResult:
        """Molecule records for many ChEMBL IDs in batched requests.

        Returns
        -------
        BatchResult
            ``{chembl_id: record}`` (``None`` when the ID does not exist);
            failed batches in ``.failed``.  Records are added to the cache,
            so later per-compound calls do not re-fetch them.
        """
        ids = [c.strip().upper() for c in chembl_ids if c and c.strip()]
        result = BatchResult()
        todo = []
        for cid in dict.fromkeys(ids):
            if self.cache and cid in self._records:
                result[cid] = self._records[cid]
            else:
                todo.append(cid)
        if not todo:
            return result

        def fetch(chunk: list[str]) -> dict[str, dict[str, Any] | None]:
            if self._sql is not None:
                found = self._sql.molecules_by_chembl_id(chunk)
            else:
                rows = self._paged(
                    "molecule",
                    {"molecule_chembl_id__in": ",".join(chunk), "limit": PAGE_LIMIT},
                    "molecules",
                    check=("molecule_chembl_id", set(chunk)),
                )
                found = {str(r["molecule_chembl_id"]).upper(): r for r in rows}
            return {k: found.get(k) for k in chunk}

        fetched = self._batched(todo, fetch, chunk_size)
        for cid, record in fetched.items():
            if record and self.cache:
                self._records[cid] = record
        result.update(fetched)
        result.failed.update(fetched.failed)
        return result

    def _molecules_where(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        self._sleep()
        url = f"{CHEMBL}/molecule.json"
        data = _get_json(url, params={**params, "format": "json"})
        if data is None:
            raise ChEMBLServiceError(None, url, "no response for a collection query", kind="no_response")
        return data.get("molecules", []) or []

    def _identity_chunk(self, keys: list[str]) -> dict[str, list[dict[str, Any]]]:
        """``{inchikey: [records]}`` by exact ``standard_inchi_key``."""
        if self._sql is not None:
            found = self._sql.molecules_by_inchikey(keys)
        else:
            rows = self._paged(
                "molecule",
                {"molecule_structures__standard_inchi_key__in": ",".join(keys), "limit": PAGE_LIMIT},
                "molecules",
                check=(_record_inchikey, set(keys)),
            )
            found = {}
            for row in rows:
                found.setdefault(_record_inchikey(row) or "", []).append(row)
        if self.cache:
            for recs in found.values():
                for rec in recs:
                    if rec.get("molecule_chembl_id"):
                        self._records.setdefault(str(rec["molecule_chembl_id"]).upper(), rec)
        return {k: found.get(k, []) for k in keys}

    def resolve(self, identifier: str, *, flexmatch: bool | None = None) -> ChEMBLResolution:
        """Map any identifier onto a ChEMBL molecule and its parent.

        * ChEMBL ID: the record itself.
        * SMILES: the standard InChIKey is computed with RDKit and matched
          exactly against ``molecule_structures.standard_inchi_key``.  A miss
          is final unless ``flexmatch`` is enabled, in which case a flexmatch
          structure search is tried and only hits with the identical
          InChIKey are accepted.  A SMILES is never sent to the name,
          synonym or free-text lookups.
        * InChIKey: exact ``standard_inchi_key`` match.
        * Anything else is a name: ``pref_name``, then synonyms, then the
          free-text search (REST only).

        The parent is resolved from the molecule hierarchy because that is
        what drug-level ChEMBL annotation is keyed on.  Provider failures
        raise (:class:`ChEMBLServiceError`, or whatever a replaced
        :func:`_get_json` raises); they are never reported as "not found".

        Returns
        -------
        ChEMBLResolution
            With ``found`` ``False`` and ``status`` explaining why if nothing
            matched.
        """
        s = identifier.strip()
        use_flex = self.flexmatch if flexmatch is None else flexmatch
        if self.cache and s in self._resolutions and flexmatch is None:
            return self._resolutions[s]

        resolution = self._resolve_uncached(s, use_flex)
        if self.cache and flexmatch is None:
            self._resolutions[s] = resolution
        if not resolution.found:
            logger.debug("No ChEMBL match for '%s' (%s)", s[:60], resolution.status)
        return resolution

    def _resolve_uncached(self, s: str, use_flex: bool) -> ChEMBLResolution:
        kind = self._classify(s)

        if kind == "chembl_id":
            record = self.get_molecule_record(s)
            if record:
                return self._resolution_from_record(s, record, "chembl_id")
            return ChEMBLResolution(identifier=s, status="no_match")

        if kind == "inchikey":
            key = s.upper()
            return self._resolution_from_candidates(s, key, self._identity_chunk([key])[key])

        if kind == "smiles" and _rdkit_available():
            key = smiles_to_inchikey(s)
            if not key:
                return ChEMBLResolution(identifier=s, status="invalid_structure")
            resolution = self._resolution_from_candidates(s, key, self._identity_chunk([key])[key])
            if resolution.status == "no_match" and use_flex:
                resolution = self._flexmatch(s, key)
            return resolution

        return self._resolve_name(s)

    def _flexmatch(self, s: str, key: str) -> ChEMBLResolution:
        """Opt-in structure search, accepting only identical InChIKeys."""
        if self._sql is not None:
            return ChEMBLResolution(identifier=s, status="no_match", inchikey=key)
        molecules = self._molecules_where(
            {"molecule_structures__canonical_smiles__flexmatch": s, "limit": 25}
        )
        exact = [m for m in molecules if _record_inchikey(m) == key]
        resolution = self._resolution_from_candidates(s, key, exact, matched_by="smiles_flexmatch")
        if not exact and molecules:
            resolution = replace(
                resolution,
                candidates=tuple(
                    (m.get("molecule_chembl_id"), (m.get("molecule_hierarchy") or {}).get("parent_chembl_id"))
                    for m in molecules
                ),
            )
        return resolution

    def _resolve_name(self, s: str) -> ChEMBLResolution:
        if self._sql is not None:
            for matched_by, lookup in (
                ("pref_name", self._sql.molecules_by_name),
                ("synonym", self._sql.molecules_by_synonym),
            ):
                molecules = lookup(s)
                if molecules:
                    return self._resolution_from_record(s, molecules[0], matched_by)
            return ChEMBLResolution(identifier=s, status="no_match")

        for matched_by, params in (
            ("pref_name", {"pref_name__iexact": s, "limit": 1}),
            ("synonym", {"molecule_synonyms__molecule_synonym__iexact": s, "limit": 1}),
        ):
            molecules = self._molecules_where(params)
            if molecules:
                return self._resolution_from_record(s, molecules[0], matched_by)

        # Last resort: ChEMBL's own free-text search, which tolerates
        # spelling and formatting differences the exact filters do not.
        self._sleep()
        url = f"{CHEMBL}/molecule/search.json"
        data = _get_json(url, params={"q": s, "limit": 1})
        if data is None:
            raise ChEMBLServiceError(None, url, "no response for a search query", kind="no_response")
        molecules = data.get("molecules", []) or []
        if molecules:
            return self._resolution_from_record(s, molecules[0], "search")
        return ChEMBLResolution(identifier=s, status="no_match")

    def _resolution_from_record(
        self, identifier: str, record: dict[str, Any], matched_by: str, inchikey: str | None = None
    ) -> ChEMBLResolution:
        chembl_id = record.get("molecule_chembl_id")
        if chembl_id and self.cache:
            self._records.setdefault(str(chembl_id).upper(), record)
        hierarchy = record.get("molecule_hierarchy") or {}
        parent = hierarchy.get("parent_chembl_id") or chembl_id
        return ChEMBLResolution(
            identifier=identifier,
            molecule_chembl_id=chembl_id,
            parent_chembl_id=parent,
            pref_name=record.get("pref_name"),
            matched_by=matched_by,
            status="matched",
            inchikey=inchikey,
            matched_inchikey=_record_inchikey(record),
        )

    def _resolution_from_candidates(
        self,
        identifier: str,
        key: str,
        records: list[dict[str, Any]],
        matched_by: str = "standard_inchi_key",
    ) -> ChEMBLResolution:
        if not records:
            return ChEMBLResolution(identifier=identifier, status="no_match", inchikey=key)
        pairs = []
        for rec in records:
            cid = rec.get("molecule_chembl_id")
            pairs.append((cid, (rec.get("molecule_hierarchy") or {}).get("parent_chembl_id") or cid))
        parents = {p for _, p in pairs}
        if len(parents) > 1:
            return ChEMBLResolution(
                identifier=identifier,
                status="ambiguous",
                inchikey=key,
                candidates=tuple(sorted(pairs, key=lambda p: (str(p[1]), str(p[0])))),
            )
        # One parent: prefer the record that is the parent itself, then the
        # lowest ChEMBL number, so the choice is deterministic.
        parent = next(iter(parents))

        def rank(rec: dict[str, Any]) -> tuple[int, int]:
            cid = str(rec.get("molecule_chembl_id") or "")
            digits = re.sub(r"\D", "", cid)
            return (0 if cid == parent else 1, int(digits) if digits else 0)

        best = min(records, key=rank)
        return self._resolution_from_record(identifier, best, matched_by, inchikey=key)

    def resolve_many(
        self, identifiers: Iterable[str], chunk_size: int = DEFAULT_BATCH
    ) -> dict[str, ChEMBLResolution]:
        """Resolve many identifiers, batching InChIKey and ChEMBL-ID lookups.

        SMILES are converted to standard InChIKeys with RDKit and looked up
        ``chunk_size`` at a time with
        ``molecule_structures__standard_inchi_key__in``; ChEMBL IDs with
        ``molecule_chembl_id__in``.  Every returned row is checked against
        the chunk it was asked for: a foreign row (ignored filter) marks the
        whole chunk ``filter_ignored``.  A chunk that fails with
        :class:`ChEMBLServiceError` (or a type in ``batch_errors``) is
        bisected down to single keys; keys that still fail are
        ``batch_failed``.  Names are resolved one at a time with
        :meth:`resolve` (a typed failure there is also ``batch_failed``).
        Opt-in flexmatch runs per structure after the exact batch misses.
        Exceptions other than :class:`ChEMBLServiceError` and the
        ``batch_errors`` types propagate.

        Returns
        -------
        dict
            ``{identifier: ChEMBLResolution}`` for every input, keyed by the
            identifier exactly as given.
        """
        identifiers = list(identifiers)
        out: dict[str, ChEMBLResolution] = {}
        by_key: dict[str, list[str]] = {}
        by_id: dict[str, list[str]] = {}
        names: list[str] = []
        rdkit_ok = _rdkit_available()
        for ident in dict.fromkeys(identifiers):
            s = str(ident).strip()
            if self.cache and s in self._resolutions:
                out[ident] = replace(self._resolutions[s], identifier=ident)
                continue
            kind = self._classify(s)
            if kind == "chembl_id":
                by_id.setdefault(s.upper(), []).append(ident)
            elif kind == "inchikey":
                by_key.setdefault(s.upper(), []).append(ident)
            elif kind == "smiles" and rdkit_ok:
                key = smiles_to_inchikey(s)
                if key:
                    by_key.setdefault(key, []).append(ident)
                else:
                    out[ident] = ChEMBLResolution(identifier=ident, status="invalid_structure")
            else:
                names.append(ident)

        if by_key:
            found = self._batched(list(by_key), self._identity_chunk, chunk_size)
            for key, idents in by_key.items():
                for ident in idents:
                    if key in found:
                        res = self._resolution_from_candidates(ident, key, found[key] or [])
                        if res.status == "no_match" and self.flexmatch and self._classify(ident) == "smiles":
                            res = replace(self._flexmatch(str(ident).strip(), key), identifier=ident)
                    else:
                        failure = found.failed.get(key, {"status": "batch_failed", "error": ""})
                        res = ChEMBLResolution(
                            identifier=ident, status=failure["status"], inchikey=key, error=failure["error"]
                        )
                    out[ident] = res

        if by_id:
            records = self.get_molecule_records_many(list(by_id), chunk_size=chunk_size)
            for cid, idents in by_id.items():
                for ident in idents:
                    if cid in records:
                        record = records[cid]
                        out[ident] = (
                            self._resolution_from_record(ident, record, "chembl_id")
                            if record
                            else ChEMBLResolution(identifier=ident, status="no_match")
                        )
                    else:
                        failure = records.failed.get(cid, {"status": "batch_failed", "error": ""})
                        out[ident] = ChEMBLResolution(
                            identifier=ident, status=failure["status"], error=failure["error"]
                        )

        for ident in names:
            try:
                out[ident] = replace(self.resolve(str(ident)), identifier=ident)
            except self._batch_errors as exc:
                out[ident] = ChEMBLResolution(
                    identifier=ident, status="batch_failed", error=f"{type(exc).__name__}: {exc}"
                )

        if self.cache:
            for ident, res in out.items():
                if res.status in {"matched", "no_match", "ambiguous", "invalid_structure"}:
                    self._resolutions.setdefault(str(ident).strip(), replace(res, identifier=str(ident).strip()))
        return {ident: out[ident] for ident in dict.fromkeys(identifiers)}

    def _parent_id(self, identifier: str) -> str | None:
        return self.resolve(identifier).parent_chembl_id

    def _collection(
        self, endpoint: str, params: dict[str, Any], key: str
    ) -> list[dict[str, Any]]:
        """One page of a collection; ``None`` (no response) raises."""
        self._sleep()
        url = f"{CHEMBL}/{endpoint}.json"
        data = _get_json(url, params={**params, "format": "json"})
        if data is None:
            raise ChEMBLServiceError(None, url, "no response for a collection query", kind="no_response")
        return data.get(key, []) or []

    def _single_parent_rows(
        self,
        endpoint: str,
        filter_field: str,
        parent: str,
        collection: str,
        limit: int | None,
        *,
        row_field: str | None = None,
    ) -> list[dict[str, Any]]:
        """Rows for one parent; validated, paginated when ``limit`` is ``None``."""
        params: dict[str, Any] = {filter_field: parent, "limit": PAGE_LIMIT if limit is None else limit}
        return self._paged(
            endpoint,
            params,
            collection,
            check=(row_field or filter_field, {parent.upper()}),
            max_rows=limit,
        )

    # ==================================================================
    # 1. Development status and regulatory record
    # ==================================================================

    def get_development_status(self, identifier: str) -> dict[str, Any]:
        """Clinical development and regulatory status of a compound.

        Answers the questions a perturbation screen actually asks of its
        compound list: is this an approved drug, a clinical candidate or a
        tool compound; when was it approved; can it be given orally; was it
        withdrawn; does it carry a black box warning.

        Parameters
        ----------
        identifier
            ChEMBL ID, SMILES, InChIKey or compound name.

        Returns
        -------
        dict
            Empty if the compound is not in ChEMBL.  ``max_phase`` is the
            raw ChEMBL value and ``development_phase`` its label; a
            missing clinical record has ``max_phase`` and
            ``development_phase`` ``None``. Curated flags are positive-only:
            zero is unrecorded, not evidence of absence. Original values
            remain in ``raw_flags``. ``natural_product`` records ChEMBL's
            COCONUT membership flag, not demonstrated natural provenance.
        """
        resolution = self.resolve(identifier)
        if not resolution.found:
            return {}
        # Flags describe the drug, and the drug is the parent form.
        record = self.get_molecule_record(
            resolution.parent_chembl_id or resolution.molecule_chembl_id  # type: ignore[arg-type]
        )
        if not record:
            return {}
        return self._development_from_record(record)

    @staticmethod
    def _development_from_record(record: dict[str, Any]) -> dict[str, Any]:
        max_phase = _as_float(record.get("max_phase"))
        availability = _as_int(record.get("availability_type"))
        chirality = _as_int(record.get("chirality"))
        curated_flags = ("oral", "parenteral", "topical", "therapeutic_flag",
                         "dosed_ingredient", "black_box_warning", "withdrawn_flag",
                         "first_in_class", "prodrug", "orphan", "natural_product",
                         "chemical_probe", "veterinary")
        positive = {name: True if _flag(record.get(name)) is True else None
                    for name in curated_flags}

        return {
            "molecule_chembl_id": record.get("molecule_chembl_id"),
            "pref_name": record.get("pref_name"),
            "molecule_type": record.get("molecule_type"),
            "structure_type": record.get("structure_type"),
            "max_phase": max_phase,
            "development_phase": _phase_label(record.get("max_phase")),
            "is_approved": None if max_phase is None or max_phase < 0 else max_phase == 4.0,
            "raw_flags": {name: record.get(name) for name in curated_flags},
            "flag_policy": "positive_only; zero/missing does not establish absence",
            "natural_product_evidence": "ChEMBL COCONUT membership flag; not a natural-versus-synthetic classification",
            "first_approval": _as_int(record.get("first_approval")),
            "availability_type": availability,
            "availability": AVAILABILITY_LABELS.get(availability)
            if availability is not None
            else None,
            **positive,
            "inorganic_flag": _flag(record.get("inorganic_flag")),
            "polymer_flag": _flag(record.get("polymer_flag")),
            "chirality": chirality,
            "chirality_label": CHIRALITY_LABELS.get(chirality)
            if chirality is not None
            else None,
            "usan_stem": record.get("usan_stem"),
            "usan_stem_definition": record.get("usan_stem_definition"),
            "usan_year": _as_int(record.get("usan_year")),
        }

    def get_approval_record(self, identifier: str) -> dict[str, Any]:
        """Approval detail from the ``drug`` endpoint.

        Complements :meth:`get_development_status` with the applicants that
        hold approvals and the ATC codes attached to the drug record.
        Returns ``{}`` for compounds that are not approved drugs.

        With ``backend="sqlite"`` the record is rebuilt from tables (see
        :meth:`ChEMBLSqlite.drug`): ``drug_type`` is ``None`` and the
        applicant names come from ``products``.
        """
        parent = self._parent_id(identifier)
        if not parent:
            return {}
        if self._sql is not None:
            drug = self._sql.drug(parent)
        else:
            drugs = self._single_parent_rows(
                "drug", "molecule_chembl_id", parent, "drugs", 1
            )
            drug = drugs[0] if drugs else None
        if not drug:
            return {}
        return {
            "molecule_chembl_id": drug.get("molecule_chembl_id"),
            "first_approval": _as_int(drug.get("first_approval")),
            "drug_type": _as_int(drug.get("drug_type")),
            "applicants": drug.get("applicants") or [],
            "atc_codes": [
                a.get("code") for a in (drug.get("atc_classification") or [])
            ],
            "helm_notation": drug.get("helm_notation"),
            "biotherapeutic": drug.get("biotherapeutic"),
        }

    # ==================================================================
    # 2. Indications (MeSH / EFO)
    # ==================================================================

    def get_indications(
        self, identifier: str, limit: int | None = DEFAULT_INDICATIONS
    ) -> list[dict[str, Any]]:
        """Therapeutic indications with the phase reached for each.

        ChEMBL records one row per (drug, indication, reference) triple, so
        the same disease can appear more than once with different maximum
        phases; rows are collapsed here to one entry per indication keeping
        the highest phase.

        Parameters
        ----------
        identifier
            ChEMBL ID, SMILES, InChIKey or compound name.
        limit
            Maximum indication rows to fetch (default
            :data:`DEFAULT_INDICATIONS`); ``None`` fetches all pages.

        Returns
        -------
        list of dict
            Sorted by ``max_phase_for_indication``, highest first.
        """
        parent = self._parent_id(identifier)
        if not parent:
            return []
        if self._sql is not None:
            rows = self._sql.indications([parent])[parent.upper()]
            rows = rows if limit is None else rows[:limit]
        else:
            rows = self._single_parent_rows(
                "drug_indication", "parent_molecule_chembl_id", parent, "drug_indications", limit
            )
        return self._merge_indications(rows)

    def get_indications_many(
        self, parents: Iterable[str], chunk_size: int = 20
    ) -> BatchResult:
        """All indications for many parents (``parent_molecule_chembl_id__in``).

        Every page is fetched, so unlike :meth:`get_indications` there is no
        row cap.  Values have the shape :meth:`get_indications` returns.
        """
        return self._parents_many(
            parents, "drug_indication", "parent_molecule_chembl_id", "drug_indications",
            chunk_size, self._merge_indications, "indications",
        )

    @staticmethod
    def _merge_indications(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = row.get("efo_id") or row.get("mesh_id") or row.get("mesh_heading") or ""
            phase = _as_float(row.get("max_phase_for_ind"))
            entry = merged.get(key)
            if entry is None:
                merged[key] = {
                    "indication": row.get("mesh_heading") or row.get("efo_term"),
                    "mesh_id": row.get("mesh_id"),
                    "mesh_heading": row.get("mesh_heading"),
                    "efo_id": row.get("efo_id"),
                    "efo_term": row.get("efo_term"),
                    "max_phase_for_indication": phase,
                    "n_references": len(row.get("indication_refs") or []),
                }
            else:
                if phase is not None and (
                    entry["max_phase_for_indication"] is None
                    or phase > entry["max_phase_for_indication"]
                ):
                    entry["max_phase_for_indication"] = phase
                entry["n_references"] += len(row.get("indication_refs") or [])

        return sorted(
            merged.values(),
            key=lambda e: (e["max_phase_for_indication"] or -99),
            reverse=True,
        )

    def _parents_many(
        self,
        parents: Iterable[str],
        endpoint: str,
        filter_field: str,
        collection: str,
        chunk_size: int,
        convert: Callable[[list[dict[str, Any]]], Any],
        sql_method: str,
        *,
        row_field: str | None = None,
    ) -> BatchResult:
        keys = [p.strip().upper() for p in parents if p and p.strip()]
        if self._sql is not None:
            getter = getattr(self._sql, sql_method)
            result = BatchResult()
            for chunk in _chunks(list(dict.fromkeys(keys)), 500):
                for parent, rows in getter(chunk).items():
                    result[parent] = convert(rows)
            return result
        raw = self._rows_many(
            keys, endpoint, filter_field, row_field or filter_field, collection, chunk_size
        )
        result = BatchResult({k: convert(v or []) for k, v in raw.items()})
        result.failed.update(raw.failed)
        return result

    # ==================================================================
    # 3. Safety: black box warnings and withdrawals
    # ==================================================================

    def get_warnings(
        self, identifier: str, limit: int | None = DEFAULT_WARNINGS
    ) -> list[dict[str, Any]]:
        """Black box warnings and withdrawal records.

        Parameters
        ----------
        identifier
            ChEMBL ID, SMILES, InChIKey or compound name.
        limit
            Maximum warning rows to fetch (default
            :data:`DEFAULT_WARNINGS`); ``None`` fetches all pages.

        Returns
        -------
        list of dict
            One entry per warning, with ``warning_type`` (e.g.
            ``"Black Box Warning"``, ``"Withdrawn"``), the toxicity class
            as an EFO term where curated, and the country and year.
        """
        parent = self._parent_id(identifier)
        if not parent:
            return []
        if self._sql is not None:
            rows = self._sql.warnings([parent])[parent.upper()]
            rows = rows if limit is None else rows[:limit]
        else:
            rows = self._single_parent_rows(
                "drug_warning", "parent_molecule_chembl_id", parent, "drug_warnings", limit
            )
        return self._warning_entries(rows)

    def get_warnings_many(self, parents: Iterable[str], chunk_size: int = DEFAULT_BATCH) -> BatchResult:
        """Warnings for many parents (``parent_molecule_chembl_id__in``)."""
        return self._parents_many(
            parents, "drug_warning", "parent_molecule_chembl_id", "drug_warnings",
            chunk_size, self._warning_entries, "warnings",
        )

    @staticmethod
    def _warning_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "warning_type": row.get("warning_type"),
                "warning_class": row.get("warning_class"),
                "warning_description": row.get("warning_description"),
                "warning_country": row.get("warning_country"),
                "warning_year": _as_int(row.get("warning_year")),
                "efo_id": row.get("efo_id"),
                "efo_term": row.get("efo_term"),
                "n_references": len(row.get("warning_refs") or []),
            }
            for row in rows
        ]

    # ==================================================================
    # 4. WHO ATC classification
    # ==================================================================

    def get_atc_classes(self, identifier: str) -> list[dict[str, Any]]:
        """WHO ATC classification, levels 1--5.

        A drug can hold several ATC codes -- aspirin has five, spanning
        antithrombotics and analgesics -- so this returns a list.

        Parameters
        ----------
        identifier
            ChEMBL ID, SMILES, InChIKey or compound name.

        Returns
        -------
        list of dict
            One entry per ATC code, each carrying all five levels with
            their descriptions.
        """
        resolution = self.resolve(identifier)
        if not resolution.found:
            return []
        record = self.get_molecule_record(
            resolution.parent_chembl_id or resolution.molecule_chembl_id  # type: ignore[arg-type]
        )
        codes = list((record or {}).get("atc_classifications") or [])
        if not codes:
            return []

        if self._sql is not None:
            rows = self._sql.atc_classes(codes)
            return [self._atc_entry(code, rows[code]) for code in codes if code in rows]

        classes: list[dict[str, Any]] = []
        for code in codes:
            self._sleep()
            data = _get_json(f"{CHEMBL}/atc_class/{code}.json")
            if not data:
                continue
            classes.append(self._atc_entry(code, data))
        return classes

    @staticmethod
    def _atc_entry(code: str, data: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "code": code,
            "who_name": data.get("who_name"),
            "level1": data.get("level1"),
            "level1_description": data.get("level1_description"),
            "level2": data.get("level2"),
            "level2_description": data.get("level2_description"),
            "level3": data.get("level3"),
            "level3_description": data.get("level3_description"),
            "level4": data.get("level4"),
            "level4_description": data.get("level4_description"),
            "level5": data.get("level5"),
        }

    # ==================================================================
    # 5. Synonyms: trade names, INN, USAN, research codes
    # ==================================================================

    def get_synonyms(
        self, identifier: str, limit: int = DEFAULT_SYNONYMS
    ) -> list[dict[str, str]]:
        """Names a compound is known by, with the type of each.

        Screens label compounds inconsistently -- trade name in one plate
        map, research code in the next -- so the synonym list is what makes
        two datasets joinable.

        Parameters
        ----------
        identifier
            ChEMBL ID, SMILES, InChIKey or compound name.
        limit
            Maximum synonyms to return (default :data:`DEFAULT_SYNONYMS`).

        Returns
        -------
        list of dict
            ``{"name": ..., "type": ...}`` where type is ``"INN"``,
            ``"TRADE_NAME"``, ``"USAN"``, ``"RESEARCH_CODE"``, etc.
        """
        resolution = self.resolve(identifier)
        if not resolution.found:
            return []
        record = self.get_molecule_record(
            resolution.parent_chembl_id or resolution.molecule_chembl_id  # type: ignore[arg-type]
        )
        synonyms = (record or {}).get("molecule_synonyms") or []

        # Preferred names first: an INN or USAN identifies the compound,
        # a trade name identifies a product.
        priority = {"INN": 0, "USAN": 1, "BAN": 2, "JAN": 3, "USP": 4, "FDA": 5}
        ordered = sorted(
            synonyms, key=lambda s: priority.get(s.get("syn_type", ""), 99)
        )
        seen: set[str] = set()
        out: list[dict[str, str]] = []
        for syn in ordered:
            name = syn.get("molecule_synonym") or syn.get("synonyms") or ""
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            out.append({"name": name, "type": syn.get("syn_type") or ""})
            if len(out) >= limit:
                break
        return out

    # ==================================================================
    # 6. Mechanism of action
    # ==================================================================

    def get_mechanisms(self, identifier: str) -> list[dict[str, Any]]:
        """Curated mechanism of action, keyed on the parent molecule.

        This is the query that motivated the module: filtering ``mechanism``
        on ``molecule_chembl_id`` misses every drug whose mechanism is
        registered against a salt form, imatinib included.  Every page is
        fetched (``limit=1000`` and ``page_meta.next``).

        Parameters
        ----------
        identifier
            ChEMBL ID, SMILES, InChIKey or compound name.

        Returns
        -------
        list of dict
            One entry per curated mechanism, with the action type, whether
            the interaction is direct, whether it is the mechanism
            responsible for efficacy, and the supporting references.
        """
        parent = self._parent_id(identifier)
        if not parent:
            return []
        if self._sql is not None:
            rows = self._sql.mechanisms([parent])[parent.upper()]
        else:
            rows = self._single_parent_rows(
                "mechanism", "parent_molecule_chembl_id", parent, "mechanisms", None
            )
        return self._mechanism_entries(rows)

    def get_mechanisms_many(
        self, parents: Iterable[str], chunk_size: int = DEFAULT_BATCH
    ) -> BatchResult:
        """Mechanisms for many parents (``parent_molecule_chembl_id__in``)."""
        return self._parents_many(
            parents, "mechanism", "parent_molecule_chembl_id", "mechanisms",
            chunk_size, self._mechanism_entries, "mechanisms",
        )

    @staticmethod
    def _mechanism_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "mechanism": row.get("mechanism_of_action"),
                "action_type": row.get("action_type"),
                "target_chembl_id": row.get("target_chembl_id"),
                "max_phase": _as_float(row.get("max_phase")),
                "direct_interaction": _flag(row.get("direct_interaction")),
                "molecular_mechanism": _flag(row.get("molecular_mechanism")),
                "disease_efficacy": _flag(row.get("disease_efficacy")),
                "mechanism_comment": row.get("mechanism_comment"),
                "selectivity_comment": row.get("selectivity_comment"),
                "binding_site_comment": row.get("binding_site_comment"),
                "references": [
                    {
                        "type": ref.get("ref_type"),
                        "id": ref.get("ref_id"),
                        "url": ref.get("ref_url"),
                    }
                    for ref in (row.get("mechanism_refs") or [])
                ],
            }
            for row in rows
        ]

    # ==================================================================
    # 7. Bioactivities and target profile
    # ==================================================================

    def get_activities(
        self,
        identifier: str,
        limit: int | None = DEFAULT_ACTIVITIES,
        only_with_pchembl: bool = True,
    ) -> list[dict[str, Any]]:
        """Bioactivity measurements, richest-first.

        Extends what :class:`MoleculeAnnotator` returns with the assay
        description and BAO format, the source document, ligand efficiency,
        and ChEMBL's data-validity flags -- the fields needed to decide
        whether a measurement should be trusted.

        Parameters
        ----------
        identifier
            ChEMBL ID, SMILES, InChIKey or compound name.
        limit
            Maximum activity rows to fetch (default
            :data:`DEFAULT_ACTIVITIES`). ``None`` fetches all matching rows
            before downstream filtering. ``activity_fetch_metadata[parent]``
            records returned rows, truncation and whether the total is known.
        only_with_pchembl
            Restrict to rows with a pChEMBL value, i.e. comparable
            concentration-response measurements.  Set ``False`` to include
            qualitative and non-standard readouts.

        Returns
        -------
        list of dict
            Ordered by pChEMBL, most potent first, when
            ``only_with_pchembl`` is set.  With the SQLite backend these are
            the top ``limit`` rows over every form of the parent.
        """
        parent = self._parent_id(identifier)
        if not parent:
            return []
        if limit is not None and limit < 1:
            raise ValueError("Activity limit must be positive or None")
        fetch_limit = None if limit is None else limit + 1
        if self._sql is not None:
            rows = self._sql.activities(parent, fetch_limit, only_with_pchembl)
        else:
            params: dict[str, Any] = {
                "parent_molecule_chembl_id": parent,
                "limit": PAGE_LIMIT if fetch_limit is None else min(fetch_limit, PAGE_LIMIT),
            }
            if only_with_pchembl:
                params["pchembl_value__isnull"] = "false"
                params["order_by"] = "-pchembl_value"
            rows = self._paged(
                "activity",
                params,
                "activities",
                check=("parent_molecule_chembl_id", {parent.upper()}),
                max_rows=fetch_limit,
            )
        truncated = limit is not None and len(rows) > limit
        self.activity_fetch_metadata[parent] = {
            "parent_chembl_id": parent, "limit": limit, "truncated": truncated,
            "returned_rows": min(len(rows), limit) if limit is not None else len(rows),
            "total_rows": None if truncated else len(rows),
            "total_rows_lower_bound": len(rows),
            "scope": "parent family; pchembl present" if only_with_pchembl else "parent family",
        }
        if truncated:
            rows = rows[:limit]
        return [
            {
                "activity_id": row.get("activity_id"),
                "molecule_chembl_id": row.get("molecule_chembl_id"),
                "target_chembl_id": row.get("target_chembl_id"),
                "target_pref_name": row.get("target_pref_name"),
                "target_organism": row.get("target_organism"),
                "activity_type": row.get("standard_type"),
                "activity_value": _as_float(row.get("standard_value")),
                "activity_units": row.get("standard_units"),
                "activity_relation": row.get("standard_relation"),
                "pchembl_value": _as_float(row.get("pchembl_value")),
                "action_type": row.get("action_type"),
                "assay_chembl_id": row.get("assay_chembl_id"),
                "assay_type": row.get("assay_type"),
                "assay_description": row.get("assay_description"),
                "bao_label": row.get("bao_label"),
                "ligand_efficiency": row.get("ligand_efficiency"),
                "document_chembl_id": row.get("document_chembl_id"),
                "document_year": _as_int(row.get("document_year")),
                "document_journal": row.get("document_journal"),
                "data_validity_comment": row.get("data_validity_comment"),
                "potential_duplicate": _flag(row.get("potential_duplicate")),
                "activity_comment": row.get("activity_comment"),
            }
            for row in rows
        ]

    def get_target_details(
        self,
        target_chembl_ids: list[str],
        include_protein_classes: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Resolve target identity for a list of ChEMBL target IDs.

        Turns ``CHEMBL1862`` into ``ABL1`` / ``P00519`` /
        ``"tyrosine protein kinase"``.  Targets are fetched in batched,
        validated requests and cached, and protein classes are shared across
        a screen's targets, so this stays cheap for a whole dataset.

        The REST ``target`` resource does not carry protein classifications
        on its components (ChEMBL_37), so they are read from
        ``target_component.json`` in one further batched request.

        Parameters
        ----------
        target_chembl_ids
            Target ChEMBL IDs.
        include_protein_classes
            Resolve the ChEMBL protein family hierarchy for each target.
            Costs up to two extra batched requests.

        Returns
        -------
        dict
            Keyed by target ChEMBL ID.
        """
        wanted = [t for t in dict.fromkeys(target_chembl_ids) if t]
        if not wanted:
            return {}

        missing = [
            t for t in wanted if not (self.cache and t in self._target_details)
        ]
        for chunk in _chunks(missing, DEFAULT_BATCH):
            if self._sql is not None:
                targets = self._sql.targets(chunk)
            else:
                targets = self._paged(
                    "target",
                    {"target_chembl_id__in": ",".join(chunk), "limit": len(chunk)},
                    "targets",
                    check=("target_chembl_id", {t.upper() for t in chunk}),
                )
            for target in targets:
                detail = self._target_detail(target)
                # Always stored, because the return value is assembled from
                # this dict. ``cache`` only decides whether a second call
                # re-fetches, which the ``missing`` filter above handles.
                self._target_details[detail["target_chembl_id"]] = detail

        details = {t: self._target_details[t] for t in wanted if t in self._target_details}

        if include_protein_classes:
            self._fill_component_classes(details.values())
            class_ids = {
                cid
                for detail in details.values()
                for cid in detail.get("_protein_class_ids", [])
            }
            classes = self._resolve_protein_classes(class_ids)
            for detail in details.values():
                paths = [
                    _protein_class_path(classes[cid].get("protein_class_desc"))
                    for cid in detail.get("_protein_class_ids", [])
                    if cid in classes
                ]
                detail["protein_class_path"] = paths[0] if paths else []
                # Level 1 is the useful grouping for colouring a plot:
                # "enzyme", "membrane receptor", "transporter", ...
                detail["target_class"] = paths[0][0] if paths and paths[0] else None
                detail["target_subclass"] = (
                    paths[0][-1] if paths and len(paths[0]) > 1 else None
                )

        return {t: {k: v for k, v in d.items() if not k.startswith("_")}
                for t, d in details.items()}

    def _fill_component_classes(self, details: Iterable[dict[str, Any]]) -> None:
        """Look up class IDs for components the ``target`` payload left bare."""
        pending = [d for d in details if d.get("_unclassified_components")]
        wanted = sorted(
            {c for d in pending for c in d["_unclassified_components"]}
            - set(self._component_classes)
        )
        for chunk in _chunks(wanted, DEFAULT_BATCH):
            rows = self._paged(
                "target_component",
                {"component_id__in": ",".join(str(c) for c in chunk), "limit": PAGE_LIMIT},
                "target_components",
                check=(lambda r: str(r.get("component_id")), {str(c) for c in chunk}),
            )
            for row in rows:
                self._component_classes[int(row["component_id"])] = [
                    int(pc["protein_classification_id"])
                    for pc in (row.get("protein_classifications") or [])
                    if pc.get("protein_classification_id") is not None
                ]
        for d in pending:
            ids = list(d.get("_protein_class_ids", []))
            for comp in d["_unclassified_components"]:
                for cid in self._component_classes.get(comp, []):
                    if cid not in ids:
                        ids.append(cid)
            d["_protein_class_ids"] = ids
            d["_unclassified_components"] = []

    @staticmethod
    def _target_detail(target: dict[str, Any]) -> dict[str, Any]:
        components = target.get("target_components") or []
        accessions = [c.get("accession") for c in components if c.get("accession")]
        gene_symbols = [
            syn.get("component_synonym")
            for comp in components
            for syn in (comp.get("target_component_synonyms") or [])
            if syn.get("syn_type") == "GENE_SYMBOL" and syn.get("component_synonym")
        ]
        class_ids = [
            pc.get("protein_classification_id")
            for comp in components
            for pc in (comp.get("protein_classifications") or [])
            if pc.get("protein_classification_id") is not None
        ]
        unclassified = [
            int(comp["component_id"])
            for comp in components
            if "protein_classifications" not in comp and comp.get("component_id") is not None
        ]
        return {
            "target_chembl_id": target.get("target_chembl_id"),
            "target_name": target.get("pref_name"),
            "target_type": target.get("target_type"),
            "organism": target.get("organism"),
            "tax_id": target.get("tax_id"),
            "uniprot_accessions": accessions,
            "uniprot_accession": accessions[0] if accessions else None,
            "gene_symbols": gene_symbols,
            "gene_symbol": gene_symbols[0] if len(gene_symbols) == 1 else None,
            "n_components": len(components),
            "_protein_class_ids": class_ids,
            "_unclassified_components": unclassified,
        }

    def _resolve_protein_classes(
        self, class_ids: set[int]
    ) -> dict[int, dict[str, Any]]:
        missing = sorted(
            cid
            for cid in class_ids
            if not (self.cache and cid in self._protein_classes)
        )
        for chunk in _chunks(missing, DEFAULT_BATCH):
            if self._sql is not None:
                rows = self._sql.protein_classes(chunk)
            else:
                # The filter field is `protein_class_id`; `protein_classification_id`
                # is the name in the response payload and is silently ignored as a
                # filter, which would return the whole 900-row class table.
                rows = self._paged(
                    "protein_classification",
                    {
                        "protein_class_id__in": ",".join(str(c) for c in chunk),
                        "limit": len(chunk),
                    },
                    "protein_classifications",
                    check=(lambda r: str(r.get("protein_class_id")), {str(c) for c in chunk}),
                )
            for row in rows:
                cid = row.get("protein_class_id")
                if cid is not None:
                    self._protein_classes[int(cid)] = row
        return {
            cid: self._protein_classes[cid]
            for cid in class_ids
            if cid in self._protein_classes
        }

    def get_target_profile(
        self,
        identifier: str,
        limit: int | None = None,
        max_targets: int | None = None,
        include_protein_classes: bool = True,
    ) -> list[dict[str, Any]]:
        """Per-target potency summary with target identity resolved.

        Collapses the activity table into one row per target -- how many
        measurements, which assay types, best and median pChEMBL -- and
        joins on gene symbol, UniProt accession and protein family.

        Parameters
        ----------
        identifier
            ChEMBL ID, SMILES, InChIKey or compound name.
        limit
            Activity rows to summarise; ``None`` (default) retrieves all.
        max_targets
            Resolve identity for at most this many targets, taking the most
            potent first. ``None`` (default) resolves every target.
        include_protein_classes
            Resolve the ChEMBL protein family hierarchy per target.

        Returns
        -------
        list of dict
            Sorted by ``best_pchembl``, most potent target first.

        See Also
        --------
        build_target_profile : Same summary from activities already fetched.
        """
        return self.build_target_profile(
            self.get_activities(identifier, limit=limit),
            max_targets=max_targets,
            include_protein_classes=include_protein_classes,
        )

    def build_target_profile(
        self,
        activities: list[dict[str, Any]],
        max_targets: int | None = None,
        include_protein_classes: bool = True,
    ) -> list[dict[str, Any]]:
        """Summarise already-fetched activities per target.

        Split out from :meth:`get_target_profile` so a caller that already
        holds the activity table -- :meth:`annotate` does -- can summarise
        it without a second fetch, and so the aggregation is testable
        without network access.

        Parameters
        ----------
        activities
            Rows as returned by :meth:`get_activities`.
        max_targets
            Resolve identity for at most this many targets, taking the most
            potent first. ``None`` (default) resolves every target.
        include_protein_classes
            Resolve the ChEMBL protein family hierarchy per target.

        Returns
        -------
        list of dict
            Sorted by ``best_pchembl``, most potent target first.
        """
        if not activities:
            return []

        grouped: dict[str, list[dict[str, Any]]] = {}
        for act in activities:
            tid = act.get("target_chembl_id")
            if tid:
                grouped.setdefault(tid, []).append(act)

        profile: list[dict[str, Any]] = []
        for tid, acts in grouped.items():
            pchembls = [a["pchembl_value"] for a in acts if a["pchembl_value"] is not None]
            profile.append(
                {
                    "target_chembl_id": tid,
                    "target_name": acts[0].get("target_pref_name"),
                    "organism": acts[0].get("target_organism"),
                    "n_measurements": len(acts),
                    "n_assays": len({a.get("assay_chembl_id") for a in acts if a.get("assay_chembl_id")}),
                    "activity_types": sorted(
                        {a["activity_type"] for a in acts if a["activity_type"]}
                    ),
                    "best_pchembl": max(pchembls) if pchembls else None,
                    "median_pchembl": statistics.median(pchembls) if pchembls else None,
                }
            )

        profile.sort(key=lambda e: (e["best_pchembl"] or -99), reverse=True)

        details = self.get_target_details(
            [e["target_chembl_id"] for e in profile[:max_targets]],
            include_protein_classes=include_protein_classes,
        )
        for entry in profile:
            detail = details.get(entry["target_chembl_id"])
            if detail:
                entry.update(
                    {
                        "target_type": detail.get("target_type"),
                        "uniprot_accession": detail.get("uniprot_accession"),
                        "gene_symbol": detail.get("gene_symbol"),
                        "gene_symbols": detail.get("gene_symbols", []),
                        "uniprot_accessions": detail.get("uniprot_accessions", []),
                        "target_class": detail.get("target_class"),
                        "target_subclass": detail.get("target_subclass"),
                        "protein_class_path": detail.get("protein_class_path"),
                    }
                )
        return profile

    @staticmethod
    def summarize_selectivity(
        target_profile: list[dict[str, Any]],
        min_measurements: int = 1,
        rank_by: Literal["best_pchembl", "median_pchembl"] = "best_pchembl",
    ) -> dict[str, Any]:
        """Condense a target profile into selectivity summary statistics.

        Parameters
        ----------
        target_profile
            Output of :meth:`get_target_profile` or
            :meth:`build_target_profile`.
        min_measurements
            Ignore targets with fewer than this many measurements when
            picking the primary target.  The default of 1 keeps every
            target, which is faithful to the data but noisy -- see the
            warning below.
        rank_by
            Whether to rank targets on their single best measurement or on
            the median across measurements.  ``"median_pchembl"`` is the
            more robust choice for a well-measured compound.

        Returns
        -------
        dict
            ``selectivity_window`` is the pChEMBL gap between the best and
            second-best *protein* target, i.e. how many orders of magnitude
            separate the primary target from the next one.  Cell-line and
            tissue readouts and unresolved target types are excluded from
            that comparison. Unknown identity is not protein evidence.

        Warning
        -------
        With the defaults, the primary target is whichever protein carries
        the single most potent measurement, and one optimistic assay can
        outrank a thoroughly characterised target.  For imatinib, ChEMBL's
        most potent protein reading is a lone ERBB2 value of 10.22 against 43
        ABL1 measurements with a median near 7.9, so the default reports
        ERBB2 rather than the ABL1 the drug is prescribed for.  Raising
        ``min_measurements``, or ranking by ``"median_pchembl"``, is what
        makes the answer robust; ``n_measurements`` is on every row of the
        profile so the caller can see which case they are in.
        """
        empty = {
            "n_targets": 0,
            "n_protein_targets": 0,
            "primary_target": None,
            "primary_target_chembl_id": None,
            "primary_target_gene": None,
            "primary_target_class": None,
            "primary_target_n_measurements": None,
            "best_pchembl": None,
            "median_pchembl": None,
            "selectivity_window": None,
            "ranked_by": rank_by,
            "min_measurements": min_measurements,
        }
        if not target_profile:
            return empty

        # Protein selectivity requires resolved protein identity. Unresolved
        # entries remain in the full profile, but cannot win this ranking.
        proteins = [
            e
            for e in target_profile
            if e.get("target_type") in ("SINGLE PROTEIN", "PROTEIN COMPLEX")
            and e.get(rank_by) is not None
            # A row exists because at least one measurement produced it, so an
            # absent count means 1 rather than 0 -- treating it as 0 would drop
            # every hand-built or externally supplied profile row.
            and (e.get("n_measurements") or 1) >= min_measurements
        ]
        ranked = sorted(proteins, key=lambda e: e[rank_by], reverse=True)
        if not ranked:
            return {**empty, "n_targets": len(target_profile)}

        best = ranked[0]
        window = (
            ranked[0][rank_by] - ranked[1][rank_by] if len(ranked) > 1 else None
        )
        return {
            "n_targets": len(target_profile),
            "n_protein_targets": sum(
                1 for e in target_profile if e.get("target_type") == "SINGLE PROTEIN"
            ),
            "primary_target": best.get("target_name"),
            "primary_target_chembl_id": best.get("target_chembl_id"),
            "primary_target_gene": best.get("gene_symbol"),
            "primary_target_genes": best.get("gene_symbols", []),
            "primary_target_class": best.get("target_class"),
            "primary_target_n_measurements": best.get("n_measurements"),
            "best_pchembl": best.get("best_pchembl"),
            "median_pchembl": best.get("median_pchembl"),
            "selectivity_window": window,
            "ranked_by": rank_by,
            "min_measurements": min_measurements,
        }

    # ==================================================================
    # 8. Metabolism
    # ==================================================================

    def get_metabolism(
        self, identifier: str, limit: int | None = DEFAULT_METABOLITES
    ) -> list[dict[str, Any]]:
        """Metabolic conversions, the enzymes involved, and the metabolites.

        Parameters
        ----------
        identifier
            ChEMBL ID, SMILES, InChIKey or compound name.
        limit
            Maximum metabolism rows to fetch (default
            :data:`DEFAULT_METABOLITES`); ``None`` fetches all pages.

        Returns
        -------
        list of dict
            One entry per curated conversion.  A prodrug's active form and
            a compound's toxic metabolite both show up here.  The REST
            filter is ``drug_chembl_id`` (the parent itself); the SQLite
            backend also includes conversions recorded against its salts.
        """
        parent = self._parent_id(identifier)
        if not parent:
            return []
        if self._sql is not None:
            rows = self._sql.metabolism([parent])[parent.upper()]
            rows = rows if limit is None else rows[:limit]
        else:
            rows = self._single_parent_rows(
                "metabolism", "drug_chembl_id", parent, "metabolisms", limit
            )
        return self._metabolism_entries(rows)

    def get_metabolism_many(self, parents: Iterable[str], chunk_size: int = DEFAULT_BATCH) -> BatchResult:
        """Metabolism for many parents (``drug_chembl_id__in``)."""
        return self._parents_many(
            parents, "metabolism", "drug_chembl_id", "metabolisms",
            chunk_size, self._metabolism_entries, "metabolism",
        )

    @staticmethod
    def _metabolism_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "substrate_name": row.get("substrate_name"),
                "substrate_chembl_id": row.get("substrate_chembl_id"),
                "metabolite_name": row.get("metabolite_name"),
                "metabolite_chembl_id": row.get("metabolite_chembl_id"),
                "enzyme_name": row.get("enzyme_name"),
                "enzyme_target_chembl_id": row.get("target_chembl_id"),
                "conversion": row.get("met_conversion"),
                "comment": row.get("met_comment"),
                "organism": row.get("organism"),
                "pathway_id": row.get("pathway_id"),
                "n_references": len(row.get("metabolism_refs") or []),
            }
            for row in rows
        ]

    # ==================================================================
    # 9. Molecular forms (parent / salts)
    # ==================================================================

    def get_molecule_forms(
        self, identifier: str, limit: int | None = DEFAULT_FORMS
    ) -> list[dict[str, Any]]:
        """Parent compound and its salt forms.

        Useful for reconciling a screen that names the salt against one
        that names the free base.
        """
        parent = self._parent_id(identifier)
        if not parent:
            return []
        if self._sql is not None:
            rows = self._sql.molecule_forms([parent])[parent.upper()]
            rows = rows if limit is None else rows[:limit]
        else:
            rows = self._single_parent_rows(
                "molecule_form", "parent_chembl_id", parent, "molecule_forms", limit
            )
        return self._form_entries(rows)

    def get_molecule_forms_many(self, parents: Iterable[str], chunk_size: int = DEFAULT_BATCH) -> BatchResult:
        """Forms for many parents (``parent_chembl_id__in``)."""
        return self._parents_many(
            parents, "molecule_form", "parent_chembl_id", "molecule_forms",
            chunk_size, self._form_entries, "molecule_forms",
        )

    def get_parent_family_audit(self, identifier: str) -> dict[str, Any]:
        """Check whether parent-family annotations cross chemical identities.

        ChEMBL families can include isotopologues as well as salts. This audit
        compares charge-neutralized fragment InChIKeys, preserving isotope and
        stereo information. An unresolved member makes transfer unverified;
        callers may retain the raw family evidence but must not call it exact.
        """
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize

        parent = self._parent_id(identifier)
        if not parent:
            return {"status": "no_match", "primary_annotation_eligible": False, "members": []}
        forms = self.get_molecule_forms(parent, limit=None)
        ids = list(dict.fromkeys([parent, *[f["molecule_chembl_id"] for f in forms]]))
        records = self.get_molecule_records_many(ids)

        def parent_key(record):
            smiles = ((record or {}).get("molecule_structures") or {}).get("canonical_smiles")
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            if molecule is None:
                return None
            molecule = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.FragmentParent(molecule))
            return Chem.MolToInchiKey(molecule)

        expected = parent_key(records.get(parent))
        members = [{"molecule_chembl_id": cid, "fragment_inchikey": parent_key(records.get(cid))}
                   for cid in ids]
        for row in members:
            row["identity_compatible"] = bool(expected and row["fragment_inchikey"] == expected)
        safe = bool(expected) and all(row["identity_compatible"] for row in members)
        return {"parent_chembl_id": parent, "status": "compatible" if safe else "family_identity_unresolved",
                "primary_annotation_eligible": safe, "parent_fragment_inchikey": expected,
                "policy": "neutralized fragment standard InChIKey; retain isotope and stereo", "members": members}

    @staticmethod
    def _form_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "molecule_chembl_id": row.get("molecule_chembl_id"),
                "parent_chembl_id": row.get("parent_chembl_id"),
                "is_parent": bool(row.get("is_parent")),
            }
            for row in rows
        ]

    # ==================================================================
    # 10. Structural analogs
    # ==================================================================

    def get_analogs(
        self,
        identifier: str,
        similarity: int = DEFAULT_ANALOG_SIMILARITY,
        limit: int = DEFAULT_ANALOGS,
    ) -> list[dict[str, Any]] | dict[str, str]:
        """Structurally similar ChEMBL compounds by Tanimoto similarity.

        Parameters
        ----------
        identifier
            ChEMBL ID, SMILES, InChIKey or compound name.
        similarity
            Minimum similarity percentage, 40--100 (default
            :data:`DEFAULT_ANALOG_SIMILARITY`).
        limit
            Maximum analogs to return (default :data:`DEFAULT_ANALOGS`).

        Returns
        -------
        list of dict
            Excludes the query compound itself.  Sorted by similarity,
            most similar first.  The SQLite backend has no similarity index
            and returns ``{"status": "not_available_offline"}``.
        """
        if self._sql is not None:
            return {"status": "not_available_offline"}
        resolution = self.resolve(identifier)
        if not resolution.found:
            return []
        record = self.get_molecule_record(resolution.molecule_chembl_id)  # type: ignore[arg-type]
        smiles = ((record or {}).get("molecule_structures") or {}).get(
            "canonical_smiles"
        )
        if not smiles:
            return []

        # The query structure is a path segment, not a parameter, so it has
        # to be quoted -- SMILES routinely contain '/', '\' and '#'.
        quoted = requests.utils.quote(smiles, safe="")
        molecules = self._collection(
            f"similarity/{quoted}/{int(similarity)}", {"limit": limit}, "molecules"
        )

        analogs = [
            {
                "molecule_chembl_id": m.get("molecule_chembl_id"),
                "pref_name": m.get("pref_name"),
                "similarity": _as_float(m.get("similarity")),
                "max_phase": _as_float(m.get("max_phase")),
                "development_phase": _phase_label(m.get("max_phase")),
                "canonical_smiles": (m.get("molecule_structures") or {}).get(
                    "canonical_smiles"
                ),
            }
            for m in molecules
            if m.get("molecule_chembl_id") != resolution.molecule_chembl_id
        ]
        analogs.sort(key=lambda a: (a["similarity"] or 0), reverse=True)
        return analogs

    # ==================================================================
    # 11. Cross references
    # ==================================================================

    def get_cross_references(self, identifier: str) -> list[dict[str, str]]:
        """ChEMBL's own cross references (DailyMed, PubChem, Wikipedia, ...)."""
        resolution = self.resolve(identifier)
        if not resolution.found:
            return []
        record = self.get_molecule_record(resolution.molecule_chembl_id)  # type: ignore[arg-type]
        return [
            {
                "source": ref.get("xref_src") or "",
                "id": ref.get("xref_id") or "",
                "name": ref.get("xref_name") or "",
            }
            for ref in ((record or {}).get("cross_references") or [])
        ]

    # ==================================================================
    # Convenience: one-call aggregation
    # ==================================================================

    def annotate(
        self,
        identifier: str,
        sources: ChEMBLSource | Literal["all"] | list[str] = "all",
    ) -> dict[str, Any]:
        """Aggregate ChEMBL annotations for one compound.

        Parameters
        ----------
        identifier
            ChEMBL ID, SMILES, InChIKey or compound name.
        sources
            ``"all"`` queries everything in :data:`ALL_SOURCES`.  Pass a
            list to select from ``"identity"``, ``"development"``,
            ``"indications"``, ``"safety"``, ``"atc"``, ``"synonyms"``,
            ``"mechanisms"``, ``"activities"``, ``"targets"``,
            ``"metabolism"``, ``"forms"``, ``"analogs"``, ``"xrefs"``.
            ``"analogs"`` is a full structure search and is therefore not
            part of ``"all"``.

        Returns
        -------
        dict
            Nested, one key per source.  ``found`` is ``False`` when the
            compound is not in ChEMBL, in which case only the identity
            keys are populated.
        """
        sources_list = (
            list(ALL_SOURCES)
            if sources == "all"
            else [sources]
            if isinstance(sources, str)
            else list(sources)
        )

        resolution = self.resolve(identifier)
        result: dict[str, Any] = {
            "identifier": identifier,
            "found": resolution.found,
            "chembl_id": resolution.molecule_chembl_id,
            "parent_chembl_id": resolution.parent_chembl_id,
            "pref_name": resolution.pref_name,
            "matched_by": resolution.matched_by,
            "match_status": resolution.status,
        }
        if not resolution.found:
            return result

        if "development" in sources_list:
            result["development"] = self.get_development_status(identifier)
            result["approval"] = self.get_approval_record(identifier)

        if "indications" in sources_list:
            result["indications"] = self.get_indications(identifier)

        if "safety" in sources_list:
            result["warnings"] = self.get_warnings(identifier)

        if "atc" in sources_list:
            result["atc_classes"] = self.get_atc_classes(identifier)

        if "synonyms" in sources_list:
            result["synonyms"] = self.get_synonyms(identifier)

        if "mechanisms" in sources_list:
            result["mechanisms"] = self.get_mechanisms(identifier)

        # The target profile is derived from the activity table, so fetch it
        # once and serve both sources from it.
        if "targets" in sources_list or "activities" in sources_list:
            activities = self.get_activities(identifier, limit=None if "targets" in sources_list else DEFAULT_ACTIVITIES)
            parent = resolution.parent_chembl_id or resolution.molecule_chembl_id
            result["activity_fetch"] = dict(self.activity_fetch_metadata.get(parent, {}))
            if "activities" in sources_list:
                result["activities"] = activities
            if "targets" in sources_list:
                profile = self.build_target_profile(activities)
                result["target_profile"] = profile
                result["selectivity"] = self.summarize_selectivity(profile)

        if "metabolism" in sources_list:
            result["metabolism"] = self.get_metabolism(identifier)

        if "forms" in sources_list:
            result["molecule_forms"] = self.get_molecule_forms(identifier)

        if "analogs" in sources_list:
            result["analogs"] = self.get_analogs(identifier)

        if "xrefs" in sources_list:
            result["cross_references"] = self.get_cross_references(identifier)

        return result

    def annotate_batch(
        self,
        identifiers: list[str],
        sources: str | list[str] = "all",
    ) -> dict[str, dict[str, Any]]:
        """Annotate a list of compounds.

        Identity is resolved up front with :meth:`resolve_many`, so
        structures cost one batched lookup per :data:`DEFAULT_BATCH`.

        Returns
        -------
        dict
            Mapping identifier to annotation dict.
        """
        self.resolve_many(identifiers)
        results: dict[str, dict[str, Any]] = {}
        total = len(identifiers)
        n_found = 0
        for i, ident in enumerate(identifiers):
            results[ident] = self.annotate(ident, sources=sources)
            n_found += bool(results[ident].get("found"))
            if (i + 1) % 10 == 0:
                logger.info("Annotated %d/%d compounds", i + 1, total)
        logger.info(
            "Annotated %d compounds, %d matched in ChEMBL", total, n_found
        )
        return results

    def annotate_adata(
        self,
        adata,  # anndata.AnnData
        column: str,
        sources: str | list[str] = "all",
        copy: bool = True,
    ):
        """Annotate drug perturbations in an AnnData with ChEMBL metadata.

        Reads compound identifiers from ``adata.obs[column]``, annotates
        each unique compound, and stores:

        - Summary columns as ``drug_*`` in ``adata.obs``
        - Full annotation dicts in ``adata.uns["chembl_annotations"]``
        - Fetch caps in ``adata.uns["chembl_annotation_limits"]``

        Parameters
        ----------
        adata
            AnnData with compound identifiers in ``.obs[column]``.
        column
            Column in ``adata.obs`` holding ChEMBL IDs, SMILES, or names.
        sources
            Annotation sources to query (see :meth:`annotate`).
        copy
            If ``True``, operate on a copy.

        Returns
        -------
        AnnData with ChEMBL annotations added.
        """
        if copy:
            adata = adata.copy()

        if column not in adata.obs.columns:
            raise ValueError(
                f"Column '{column}' not found in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )

        identifiers = adata.obs[column].astype(str).values
        unique_ids = list(dict.fromkeys(identifiers))
        logger.info(
            "Annotating %d unique compounds from %d observations",
            len(unique_ids), len(identifiers),
        )

        annotations = self.annotate_batch(unique_ids, sources=sources)
        adata.uns["chembl_annotations"] = annotations

        rows = [annotations.get(ident, {}) for ident in identifiers]
        for name, values in chembl_summary_columns(rows).items():
            adata.obs[name] = values

        # `drug_n_indications` and `drug_n_targets` count what was fetched,
        # and the fetch is capped, so a well-studied drug reports the cap
        # rather than the truth. Flag the saturated rows rather than let a
        # near-constant column pass for a measurement -- the same contract
        # GeneAnnotator uses for its capped counts.
        n_indications = adata.obs["drug_n_indications"]
        adata.obs["drug_n_indications_at_limit"] = [
            n >= DEFAULT_INDICATIONS for n in n_indications
        ]
        adata.uns["chembl_annotation_limits"] = {
            "drug_n_indications": DEFAULT_INDICATIONS,
            "drug_n_activities": None if any("target_profile" in r for r in rows) else DEFAULT_ACTIVITIES,
            "drug_n_targets": None,
            "drug_n_metabolites": DEFAULT_METABOLITES,
        }

        n_found = sum(1 for r in rows if r.get("found"))
        logger.info(
            "ChEMBL annotations stored in adata.obs (drug_*) and "
            "adata.uns['chembl_annotations']; %d/%d observations matched",
            n_found, len(rows),
        )
        return adata


def chembl_summary_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Flatten per-compound annotation dicts into ``obs`` columns.

    Kept separate from :meth:`ChEMBLAnnotator.annotate_adata` so the
    flattening can be tested without an AnnData.
    """
    columns: dict[str, list[Any]] = {}

    def add(name: str, fn) -> None:
        columns[f"drug_{name}"] = [fn(row) for row in rows]

    def dev(row: dict[str, Any], key: str) -> Any:
        return (row.get("development") or {}).get(key)

    add("chembl_id", lambda r: r.get("chembl_id") or "")
    add("parent_chembl_id", lambda r: r.get("parent_chembl_id") or "")
    add("pref_name", lambda r: r.get("pref_name") or "")
    add("in_chembl", lambda r: bool(r.get("found")))

    add("max_phase", lambda r: dev(r, "max_phase"))
    add("development_phase", lambda r: dev(r, "development_phase") or "")
    add("is_approved", lambda r: dev(r, "is_approved"))
    add("first_approval", lambda r: dev(r, "first_approval"))
    add("availability", lambda r: dev(r, "availability") or "")
    add("molecule_type", lambda r: dev(r, "molecule_type") or "")
    add("oral", lambda r: dev(r, "oral"))
    add("parenteral", lambda r: dev(r, "parenteral"))
    add("topical", lambda r: dev(r, "topical"))
    add("is_prodrug", lambda r: dev(r, "prodrug"))
    add("first_in_class", lambda r: dev(r, "first_in_class"))
    add("coconut_member", lambda r: dev(r, "natural_product"))
    add("is_orphan", lambda r: dev(r, "orphan"))
    add("is_withdrawn", lambda r: dev(r, "withdrawn_flag"))
    add("has_black_box_warning", lambda r: dev(r, "black_box_warning"))

    add("n_indications", lambda r: len(r.get("indications") or []))
    add("top_indication", _top_indication)

    add("n_warnings", lambda r: len(r.get("warnings") or []))
    add("withdrawn_reason", _withdrawn_reason)

    add("atc_code", _first_atc_code)
    add("atc_level1", _atc_level1)

    add("n_mechanisms", lambda r: len(r.get("mechanisms") or []))
    add("moa", _primary_moa)
    add("action_type", _primary_action_type)

    def sel(row: dict[str, Any], key: str) -> Any:
        return (row.get("selectivity") or {}).get(key)

    add("n_targets", lambda r: sel(r, "n_targets") or 0)
    add("n_protein_targets", lambda r: sel(r, "n_protein_targets") or 0)
    add("primary_target", lambda r: sel(r, "primary_target") or "")
    add("primary_target_gene", lambda r: sel(r, "primary_target_gene") or "")
    add("target_class", lambda r: sel(r, "primary_target_class") or "")
    add("best_pchembl", lambda r: sel(r, "best_pchembl"))
    add("selectivity_window", lambda r: sel(r, "selectivity_window"))

    add("n_activities", lambda r: len(r.get("activities") or []))
    add("n_metabolites", lambda r: len(r.get("metabolism") or []))
    add("n_synonyms", lambda r: len(r.get("synonyms") or []))

    return columns


def _top_indication(row: dict[str, Any]) -> str:
    indications = row.get("indications") or []
    if not indications:
        return ""
    return indications[0].get("indication") or ""


def _withdrawn_reason(row: dict[str, Any]) -> str:
    """The toxicity class behind a withdrawal, when one is curated."""
    for warning in row.get("warnings") or []:
        if warning.get("warning_type") == "Withdrawn":
            return warning.get("warning_class") or warning.get(
                "warning_description"
            ) or "withdrawn"
    return ""


def _first_atc_code(row: dict[str, Any]) -> str:
    classes = row.get("atc_classes") or []
    return classes[0].get("code") or "" if classes else ""


def _atc_level1(row: dict[str, Any]) -> str:
    classes = row.get("atc_classes") or []
    return classes[0].get("level1_description") or "" if classes else ""


def _primary_moa(row: dict[str, Any]) -> str:
    """The mechanism responsible for efficacy, else the first curated one."""
    mechanisms = row.get("mechanisms") or []
    for mech in mechanisms:
        if mech.get("disease_efficacy"):
            return mech.get("mechanism") or ""
    return mechanisms[0].get("mechanism") or "" if mechanisms else ""


def _primary_action_type(row: dict[str, Any]) -> str:
    mechanisms = row.get("mechanisms") or []
    for mech in mechanisms:
        if mech.get("disease_efficacy"):
            return mech.get("action_type") or ""
    return mechanisms[0].get("action_type") or "" if mechanisms else ""
