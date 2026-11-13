"""Small-molecule annotation from multiple public databases.

Aggregates metadata for a molecule (given as SMILES, name, InChI, InChIKey
or PubChem CID) from these categories of data sources:

1. **Structural / physicochemical** -- RDKit (local)
2. **Bioactivities & targets** -- ChEMBL REST API
3. **Ontological roles** -- ChEBI: the ``has role`` relations of the entity
   whose InChIKey matches, from a pinned ``chebi_core.obo`` release
   (:class:`~embpy.resources.molecule.chebi_obo.ChebiOboIndex`) or, online,
   from the ChEBI 2.0 REST API (``es_search`` + ``compounds``). The older
   OLS label lookup lives on as :meth:`MoleculeAnnotator.get_chebi_terms`.
4. **Pathway annotations** -- KEGG, through a KEGG index built from KEGG's
   own bulk tables (:class:`~embpy.resources.molecule.kegg_index.KeggIndex`)
   and joined on InChIKey. KEGG's ``pubchem:`` namespace holds PubChem
   *substance* IDs (SIDs), so a compound ID (CID) is never sent to it.
5. **Cross-database IDs** -- PubChem (ChEMBL optional, last and non-fatal)
6. **Assay-name keywords** -- PubChem bioassay summaries: disease keywords
   in the names of assays where the compound was *Active*. This is weak
   evidence, not a curated disease association.
7. **Drug & clinical record** -- ChEMBL, via
   :class:`~embpy.resources.molecule.chembl.ChEMBLAnnotator`: development
   phase, indications, safety warnings, ATC class, synonyms, metabolism,
   and per-target potency profiles.

Category 7 is delegated to :mod:`embpy.resources.molecule.chembl`, which
covers considerably more of ChEMBL than the ``molecule``/``activity``/
``mechanism`` endpoints used here and correctly keys drug-level queries on
the parent molecule -- see that module's docstring for why that matters.

Every HTTP call goes through the module-level helpers :func:`_get_json`,
:func:`_get_text` and :func:`_post_json`, so a caller (e.g. a batch driver
with its own cache, pacing and circuit breaker) can monkeypatch them. The
helpers keep their legacy convention -- ``None`` on HTTP 404 and on any
failure -- and additionally record the outcome of the last call in a
thread-local, so the newer status-returning methods can tell "provider says
not found" (404) from "request failed". A *patched* helper is expected to
raise on failure; ``None`` from a patched helper is read as "not found".

Does **not** reimplement pertpy functionality (drug-target gene
annotation, MoA lookup tables, cell-line metadata).
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import warnings
from collections.abc import Iterable
from typing import Any, Literal

import requests

from .chembl import (
    DEFAULT_ACTIVITIES,
    DEFAULT_INDICATIONS,
    DEFAULT_METABOLITES,
    DEFAULT_TARGETS,
    ChEMBLAnnotator,
    ChEMBLFilterIgnoredError,
    ChEMBLServiceError,
    chembl_summary_columns,
)
from .resolver import DrugResolver

logger = logging.getLogger(__name__)

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
#: ChEBI 2.0 public REST API (``es_search``, ``compounds``, ``compound``).
CHEBI_REST = "https://www.ebi.ac.uk/chebi/backend/api/public"
#: OLS4 term lookup, used only by :meth:`MoleculeAnnotator.get_chebi_terms`.
OLS_CHEBI_TERMS = "https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms"
KEGG = "https://rest.kegg.jp"
UNICHEM = "https://www.ebi.ac.uk/unichem/rest"

#: Largest ``chebi_ids`` list sent to ``/compounds/`` (40 verified live).
CHEBI_COMPOUNDS_BATCH = 40

#: Keywords searched (as whole words) in the names of *Active* assays by
#: :meth:`MoleculeAnnotator.get_disease_associations`.
ASSAY_NAME_KEYWORDS: tuple[str, ...] = (
    "cancer", "diabetes", "alzheimer", "parkinson", "hiv", "malaria",
    "inflammation", "asthma", "obesity", "hypertension", "leukemia",
    "lymphoma", "melanoma", "hepatitis",
)

_INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
_CID_RE = re.compile(r"^(?:CID:?\s*)?(\d+)$", re.IGNORECASE)

# Outcome of the last call made through an *unpatched* helper, per thread.
_HTTP_STATE = threading.local()
_UNSET = object()


def _record_http(url: str, status: int | None, error: str | None) -> None:
    _HTTP_STATE.last = {"url": url, "status": status, "error": error}


def _failure(url: str, exc: Exception) -> None:
    response = getattr(exc, "response", None)
    _record_http(url, getattr(response, "status_code", None), f"{type(exc).__name__}: {exc}")
    logger.debug("Request failed for %s: %s", url, exc)


def _get_json(url: str, params: dict | None = None, timeout: int = 30) -> dict | None:
    """GET a JSON document; ``None`` on HTTP 404 or any failure (legacy)."""
    _record_http(url, None, None)
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 404:
            _record_http(url, 404, None)
            return None
        resp.raise_for_status()
        _record_http(url, resp.status_code, None)
        return resp.json()
    except Exception as e:  # noqa: BLE001
        _failure(url, e)
        return None


def _get_text(url: str, params: dict | None = None, timeout: int = 30) -> str | None:
    """GET a text document; ``None`` on HTTP 404 or any failure (legacy)."""
    _record_http(url, None, None)
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 404:
            _record_http(url, 404, None)
            return None
        resp.raise_for_status()
        _record_http(url, resp.status_code, None)
        return resp.text
    except Exception as e:  # noqa: BLE001
        _failure(url, e)
        return None


def _post_json(
    url: str,
    json_data: Any = None,
    data: Any = None,
    timeout: int = 30,
) -> dict | None:
    """POST and parse JSON; ``None`` on HTTP 404 or any failure (legacy)."""
    _record_http(url, None, None)
    try:
        resp = requests.post(url, json=json_data, data=data, timeout=timeout)
        if resp.status_code == 404:
            _record_http(url, 404, None)
            return None
        resp.raise_for_status()
        _record_http(url, resp.status_code, None)
        return resp.json()
    except Exception as e:  # noqa: BLE001
        _failure(url, e)
        return None


def _call(helper, *args, **kwargs) -> tuple[Any, str, str | None, int | None]:
    """Call a module helper and classify its answer.

    Returns ``(data, outcome, error, http_status)`` with ``outcome`` one of
    ``"ok"``, ``"not_found"`` (HTTP 404, a PubChem ``PUGREST.NotFound``
    body, or ``None`` from a patched helper) or ``"failed"`` (an unpatched
    helper swallowed an error, or a non-NotFound ``Fault`` body came back).
    Exceptions raised by a patched helper propagate unchanged.
    """
    _HTTP_STATE.last = _UNSET
    data = helper(*args, **kwargs)
    last = getattr(_HTTP_STATE, "last", _UNSET)
    _HTTP_STATE.last = _UNSET
    if data is not None:
        fault = _pubchem_fault(data)
        if fault is None:
            return data, "ok", None, None
        if fault[0] == "PUGREST.NotFound":
            return None, "not_found", None, 404
        return None, "failed", f"{fault[0]}: {fault[1]}", None
    if last is _UNSET:
        # A patched helper raises on failure, so None means "not found".
        return None, "not_found", None, None
    status = last.get("status")
    if status == 404:
        return None, "not_found", None, 404
    error = last.get("error") or "request failed"
    return None, "failed", error, status


class AnnotationServiceError(RuntimeError):
    """A provider request failed (5xx, timeout, connection or ``Fault``).

    Raised by list-returning methods (:meth:`MoleculeAnnotator.get_disease_associations`,
    :meth:`MoleculeAnnotator.get_chebi_terms`) when an unpatched module helper
    swallowed an error, so an outage is never reported as an empty answer.
    """

    def __init__(self, service: str, error: str | None, status_code: int | None = None) -> None:
        self.service, self.error, self.status_code = service, error or "request failed", status_code
        super().__init__(f"{service}: {self.error}")


def _require(helper, service: str, *args, **kwargs) -> Any:
    """``_call`` that returns the data (``None`` for not-found) and raises on failure."""
    data, outcome, error, http_status = _call(helper, *args, **kwargs)
    if outcome == "failed":
        raise AnnotationServiceError(service, error, http_status)
    return data


def _pubchem_fault(data: Any) -> tuple[str, str] | None:
    """``(code, message)`` of a PubChem ``Fault`` body, else ``None``."""
    if isinstance(data, dict) and isinstance(data.get("Fault"), dict):
        fault = data["Fault"]
        return str(fault.get("Code") or "PUGREST.Unknown"), str(fault.get("Message") or "")
    return None


def _chunks(items: list, size: int) -> Iterable[list]:
    size = max(1, int(size))
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _positive_cids(values: Any) -> list[int]:
    """Sorted unique positive integer CIDs (PubChem uses ``0`` for none)."""
    out: set[int] = set()
    for value in values or []:
        try:
            cid = int(value)
        except (TypeError, ValueError):
            continue
        if cid > 0:
            out.add(cid)
    return sorted(out)


def _cid_status(cids: list[int]) -> dict[str, Any]:
    if not cids:
        return {"status": "no_match", "cid": None, "cids": []}
    if len(cids) == 1:
        return {"status": "matched", "cid": cids[0], "cids": cids}
    return {"status": "ambiguous", "cid": None, "cids": cids}


def _normalise_chebi_id(value: Any) -> str | None:
    text = str(value or "").strip().upper().replace("CHEBI_", "CHEBI:")
    if text.isdigit():
        text = f"CHEBI:{text}"
    if re.fullmatch(r"CHEBI:\d+", text):
        return text
    return None


def _chebi_number(chebi_id: str) -> int:
    try:
        return int(chebi_id.split(":", 1)[1])
    except (IndexError, ValueError):
        return 0


def _chebi_result(**fields: Any) -> dict[str, Any]:
    """Every key of a :meth:`MoleculeAnnotator.lookup_chebi_roles` result."""
    base: dict[str, Any] = {
        "status": None, "chebi_id": None, "name": None, "stars": None,
        "match_tier": None, "inchikey": None, "query_inchikey": None,
        "roles": [], "candidates": [], "connectivity": None, "release": None,
    }
    base.update(fields)
    return base


def _kegg_result(**fields: Any) -> dict[str, Any]:
    """Every key of a :meth:`MoleculeAnnotator.lookup_kegg` result."""
    base: dict[str, Any] = {
        "status": None, "entries": [], "compound_pathways": [],
        "drug_maps": [], "target_pathways": [], "release": None,
        "inchikey": None, "match_tier": None,
    }
    base.update(fields)
    return base


#: :class:`MoleculeAnnotator` source groups served by
#: :class:`~embpy.resources.molecule.chembl.ChEMBLAnnotator`, and the
#: ChEMBLAnnotator sources each pulls in.
#:
#: ChEMBLAnnotator's ``identity`` is always returned, and its ``xrefs`` are
#: left out because :meth:`MoleculeAnnotator.get_cross_references` already
#: covers cross-database IDs from PubChem and UniChem. ``activities`` rides
#: along with ``target_profile`` because the profile is derived from the same
#: fetch, so asking for one and getting both costs nothing extra.
_CHEMBL_SOURCE_GROUPS: dict[str, tuple[str, ...]] = {
    "drug": (
        "development",
        "indications",
        "safety",
        "atc",
        "synonyms",
        "mechanisms",
        "forms",
    ),
    "target_profile": ("targets", "activities"),
    "metabolism": ("metabolism",),
    "analogs": ("analogs",),
}


def _chembl_sources_for(sources_list: list[str]) -> list[str]:
    """Translate ``MoleculeAnnotator`` source names into ChEMBL ones."""
    selected: list[str] = []
    for group in sources_list:
        for source in _CHEMBL_SOURCE_GROUPS.get(group, ()):
            if source not in selected:
                selected.append(source)
    return selected


class MoleculeAnnotator:
    """Aggregate annotations for small molecules from public databases.

    Parameters
    ----------
    rate_limit_delay : float
        Seconds to wait between API calls (default 0.2).
    chebi_index
        Optional offline ChEBI index used by :meth:`lookup_chebi_roles` when
        no ``index=`` is passed: a
        :class:`~embpy.resources.molecule.chebi_obo.ChebiOboIndex` or a path
        to ``chebi_core.obo(.gz)``.
    kegg_index
        Optional KEGG index used by :meth:`lookup_kegg` when no ``index=`` is
        passed: a :class:`~embpy.resources.molecule.kegg_index.KeggIndex` or
        the directory written by
        :func:`~embpy.resources.molecule.kegg_index.build_kegg_reference`.
    """

    def __init__(
        self,
        rate_limit_delay: float = 0.2,
        *,
        chebi_index: Any = None,
        kegg_index: Any = None,
    ) -> None:
        self.delay = rate_limit_delay
        self._chembl: ChEMBLAnnotator | None = None
        self.chebi_index = chebi_index
        self.kegg_index = kegg_index
        self._loaded_indexes: dict[tuple[str, str], Any] = {}
        self._warned_kegg_index = False

    def _sleep(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    @property
    def chembl(self) -> ChEMBLAnnotator:
        """Deep ChEMBL annotator, sharing this instance's rate limit.

        Held as a single instance so its identifier, molecule-record and
        protein-class caches are reused across every call made through
        this annotator.
        """
        if self._chembl is None:
            self._chembl = ChEMBLAnnotator(rate_limit_delay=self.delay)
        return self._chembl

    # ==================================================================
    # Identifier resolution
    # ==================================================================

    @staticmethod
    def _parse_smiles(identifier: str):
        """RDKit molecule for a SMILES string, or ``None``.

        Raises ``ImportError`` when RDKit is not installed.
        """
        from rdkit import Chem, rdBase
        # Speculative: the identifier may be a SMILES or may be a name, and
        # the only way to tell is to try. A name failing to parse is the
        # expected path, not an error, so do not let RDKit log it -- without
        # this, annotating a panel by name prints five lines of
        # "SMILES Parse Error" per compound over the caller's output.
        with rdBase.BlockLogs():
            return Chem.MolFromSmiles(identifier)

    def _resolve_to_smiles(self, identifier: str) -> str | None:
        """Best-effort resolution of any identifier to canonical SMILES.

        The PubChem leg asks for ``IsomericSMILES`` and reads the answer
        through :meth:`DrugResolver._extract_smiles`, because PubChem
        renamed these properties: a request for ``CanonicalSMILES`` now
        comes back under the key ``ConnectivitySMILES``, and one for
        ``IsomericSMILES`` under ``SMILES``. Reading the requested name
        straight back out yields ``None`` for every compound.
        """
        s = identifier.strip()
        try:
            from rdkit import Chem

            mol = self._parse_smiles(s)
            if mol is not None:
                return Chem.MolToSmiles(mol)
        except ImportError:
            pass

        data = _get_json(
            f"{PUBCHEM}/compound/name/{requests.utils.quote(s, safe='')}/property/IsomericSMILES/JSON",
        )
        if data:
            props = data.get("PropertyTable", {}).get("Properties", [])
            return DrugResolver._extract_smiles(props)
        return None

    def _inchikey_for(
        self,
        identifier: str | None,
        *,
        inchikey: str | None = None,
        allow_name_lookup: bool = True,
    ) -> tuple[str | None, str | None]:
        """Standard InChIKey for an identifier, as ``(inchikey, error)``.

        An explicit ``inchikey`` wins. Otherwise the identifier may itself be
        an InChIKey, an InChI or a SMILES (converted locally with RDKit), or
        -- when ``allow_name_lookup`` -- a name resolved to SMILES through
        PubChem first.
        """
        if inchikey is not None:
            key = str(inchikey).strip().upper()
            if _INCHIKEY_RE.match(key):
                return key, None
            return None, f"not a standard InChIKey: {inchikey!r}"
        s = str(identifier or "").strip()
        if not s:
            return None, "empty identifier"
        if _INCHIKEY_RE.match(s.upper()):
            return s.upper(), None
        try:
            from rdkit import Chem, rdBase
        except ImportError:
            return None, "RDKit is required to derive an InChIKey; pass inchikey="
        if s.startswith("InChI="):
            with rdBase.BlockLogs():
                mol = Chem.MolFromInchi(s)
        else:
            mol = self._parse_smiles(s)
        if mol is None and allow_name_lookup and not s.startswith("InChI="):
            smiles = self._resolve_to_smiles(s)
            if smiles:
                mol = self._parse_smiles(smiles)
        if mol is None:
            return None, f"could not parse {s[:80]!r} as a structure"
        key = Chem.MolToInchiKey(mol)
        if not key or not _INCHIKEY_RE.match(key):
            return None, "RDKit produced no standard InChIKey"
        return key, None

    def resolve_pubchem_cid(self, identifier: str) -> dict[str, Any]:
        """Resolve an identifier to a PubChem CID, with an explicit status.

        Routes, by what the identifier looks like:

        * a CID (``"2244"`` / ``"CID2244"``): used as given, no request;
        * an InChIKey: :meth:`resolve_pubchem_cids_by_inchikey`;
        * a SMILES that RDKit parses (or an InChI, via its InChIKey): the
          canonical SMILES is sent as the ``smiles`` *query parameter* of
          ``GET {PUBCHEM}/compound/smiles/cids/JSON``. It is never put in the
          URL path: PubChem answers HTTP 400 for any SMILES containing ``/``
          (E/Z bonds), raw or ``%2F``-encoded. No name fallback is tried for
          a parsed SMILES;
        * anything else: a PubChem name lookup.

        Returns
        -------
        dict
            ``status`` (``matched`` | ``no_match`` | ``ambiguous`` |
            ``invalid_structure`` | ``service_error``), ``cid`` (``int`` only
            when matched), ``cids`` (every positive CID returned) and
            ``method``. PubChem's ``[0]`` / ``[]`` answers are ``no_match``.
        """
        s = str(identifier or "").strip()
        if not s:
            return {"status": "invalid_structure", "cid": None, "cids": [],
                    "method": "none", "error": "empty identifier"}

        cid_match = _CID_RE.match(s)
        if cid_match:
            cid = int(cid_match.group(1))
            if cid > 0:
                return {"status": "matched", "cid": cid, "cids": [cid], "method": "cid"}
            return {"status": "no_match", "cid": None, "cids": [], "method": "cid"}

        if _INCHIKEY_RE.match(s.upper()):
            res = self.resolve_pubchem_cids_by_inchikey([s.upper()])[s.upper()]
            out = {k: res.get(k) for k in ("status", "cid", "cids")}
            if out["status"] == "batch_failed":
                out["status"] = "service_error"
                out["error"] = res.get("error")
            return {**out, "method": "pubchem_inchikey"}

        try:
            from rdkit import Chem

            have_rdkit = True
        except ImportError:
            have_rdkit = False

        if have_rdkit:
            if s.startswith("InChI="):
                key, error = self._inchikey_for(s, allow_name_lookup=False)
                if key is None:
                    return {"status": "invalid_structure", "cid": None, "cids": [],
                            "method": "pubchem_inchikey", "error": error}
                res = self.resolve_pubchem_cid(key)
                return {**res, "method": "pubchem_inchikey"}
            mol = self._parse_smiles(s)
            if mol is not None:
                canon = Chem.MolToSmiles(mol)
                data, outcome, error, http_status = _call(
                    _get_json, f"{PUBCHEM}/compound/smiles/cids/JSON",
                    params={"smiles": canon},
                )
                if outcome == "ok":
                    cids = _positive_cids((data.get("IdentifierList") or {}).get("CID"))
                    result = {**_cid_status(cids), "method": "pubchem_smiles",
                              "query_inchikey": Chem.MolToInchiKey(mol),
                              "matched_inchikey": None, "match_tier": "unverified",
                              "primary_annotation_eligible": False}
                    if len(cids) == 1:
                        props, prop_outcome, prop_error, _ = _call(
                            _get_json, f"{PUBCHEM}/compound/cid/{cids[0]}/property/InChIKey/JSON")
                        rows = ((props or {}).get("PropertyTable") or {}).get("Properties") or []
                        rows = [r for r in rows if r.get("CID") == cids[0]]
                        if prop_outcome == "ok" and len(rows) == 1:
                            actual = str(rows[0].get("InChIKey") or "").upper()
                            expected = result["query_inchikey"]
                            if _INCHIKEY_RE.fullmatch(actual):
                                result["matched_inchikey"] = actual
                                result["match_tier"] = ("exact" if actual == expected else
                                    "connectivity_only" if actual[:14] == expected[:14] else "standardized_structure_differs")
                                result["primary_annotation_eligible"] = actual == expected
                        if prop_outcome != "ok":
                            result["identity_verification_error"] = prop_error or prop_outcome
                    return result
                if outcome == "not_found":
                    return {"status": "no_match", "cid": None, "cids": [], "method": "pubchem_smiles"}
                status = "invalid_structure" if http_status == 400 else "service_error"
                return {"status": status, "cid": None, "cids": [], "method": "pubchem_smiles",
                        "error": error}
        elif s.startswith("InChI="):
            return {"status": "invalid_structure", "cid": None, "cids": [],
                    "method": "none", "error": "RDKit is required to resolve InChI input"}

        data, outcome, error, _ = _call(
            _get_json, f"{PUBCHEM}/compound/name/{requests.utils.quote(s, safe='')}/cids/JSON",
        )
        if outcome == "ok":
            cids = _positive_cids((data.get("IdentifierList") or {}).get("CID"))
            return {**_cid_status(cids), "method": "pubchem_name"}
        if outcome == "not_found":
            return {"status": "no_match", "cid": None, "cids": [], "method": "pubchem_name"}
        return {"status": "service_error", "cid": None, "cids": [], "method": "pubchem_name",
                "error": error}

    def _cid_or_raise(self, identifier: str) -> int | None:
        """CID of ``identifier``; ``None`` for a provider no-match, raises on a failed request."""
        res = self.resolve_pubchem_cid(identifier)
        if res["status"] == "service_error":
            raise AnnotationServiceError("pubchem", res.get("error"))
        return res["cid"]

    def _resolve_pubchem_cid(self, identifier: str) -> int | None:
        """Resolve identifier to PubChem CID (``None`` unless uniquely matched)."""
        return self.resolve_pubchem_cid(identifier)["cid"]

    def resolve_pubchem_cids_by_inchikey(
        self,
        inchikeys: Iterable[str],
        chunk_size: int = 50,
    ) -> dict[str, dict[str, Any]]:
        """Map standard InChIKeys to PubChem CIDs in batched GETs.

        Sends ``GET {PUBCHEM}/compound/inchikey/<k1,k2,...>/property/InChIKey/JSON``
        per chunk of ``chunk_size`` keys. The answer lists ``(CID, InChIKey)``
        pairs, so every CID is mapped back to the key it came from (the
        POST ``inchikey/cids`` route loses that mapping).

        Returns
        -------
        dict
            ``{inchikey: {"status", "cid", "cids", "method"}}`` with status
            ``matched`` (one CID), ``ambiguous`` (several; ``cid`` is
            ``None``), ``no_match`` (absent from the answer, a
            ``PUGREST.NotFound`` for the whole chunk, or not a valid
            InChIKey) or ``batch_failed`` (the chunk's request failed;
            transient, never to be read as "not in PubChem"). Exceptions
            raised by a patched helper propagate.
        """
        result: dict[str, dict[str, Any]] = {}
        wanted: dict[str, list[str]] = {}
        for raw in inchikeys:
            original = str(raw).strip()
            if not original:
                continue
            key = original.upper()
            if not _INCHIKEY_RE.match(key):
                result[original] = {"status": "no_match", "cid": None, "cids": [],
                                    "method": "pubchem_inchikey", "reason": "invalid_inchikey"}
                continue
            wanted.setdefault(key, []).append(original)

        keys = list(wanted)
        for n, chunk in enumerate(_chunks(keys, chunk_size)):
            if n:
                self._sleep()
            url = f"{PUBCHEM}/compound/inchikey/{','.join(chunk)}/property/InChIKey/JSON"
            data, outcome, error, _ = _call(_get_json, url)
            found: dict[str, list[int]] = {k: [] for k in chunk}
            if outcome == "failed":
                for k in chunk:
                    for original in wanted[k]:
                        result[original] = {"status": "batch_failed", "cid": None, "cids": [],
                                            "method": "pubchem_inchikey", "error": error}
                continue
            if outcome == "ok":
                for row in (data.get("PropertyTable") or {}).get("Properties") or []:
                    key = str(row.get("InChIKey") or "").upper()
                    if key in found:
                        found[key].extend(_positive_cids([row.get("CID")]))
                    else:
                        logger.debug("PubChem returned an InChIKey not in the chunk: %s", key)
            for k in chunk:
                status = {**_cid_status(_positive_cids(found[k])), "method": "pubchem_inchikey"}
                for original in wanted[k]:
                    result[original] = dict(status)
        return result

    def get_pubchem_xrefs_batch(
        self,
        cids: Iterable[int | str],
        chunk_size: int = 50,
    ) -> dict[int, dict[str, Any]]:
        """PubChem RegistryID cross-references for many CIDs.

        Sends ``GET {PUBCHEM}/compound/cid/<c1,c2,...>/xrefs/RegistryID/JSON``
        per chunk (keep ``chunk_size`` at 50 or below: a 50-CID request took
        about 5 s live).

        Returns
        -------
        dict
            ``{cid: {"status", "registry_ids", "chebi_ids", "chembl_ids"}}``
            with status ``success`` (RegistryIDs returned), ``empty_result``
            (PubChem lists none for the CID), ``batch_failed`` (the chunk's
            request failed) or ``invalid_cid`` (not a positive integer). The
            RegistryIDs pool every depositor's identifiers; ``chebi_ids``
            and ``chembl_ids`` are only the ones with those prefixes.
        """
        result: dict[int, dict[str, Any]] = {}
        valid: list[int] = []
        for raw in cids:
            try:
                cid = int(raw)
            except (TypeError, ValueError):
                cid = 0
            if cid <= 0:
                result[raw] = {"status": "invalid_cid", "registry_ids": [], "chebi_ids": [],
                               "chembl_ids": []}
                continue
            if cid not in valid:
                valid.append(cid)

        for n, chunk in enumerate(_chunks(valid, chunk_size)):
            if n:
                self._sleep()
            url = f"{PUBCHEM}/compound/cid/{','.join(str(c) for c in chunk)}/xrefs/RegistryID/JSON"
            data, outcome, error, _ = _call(_get_json, url)
            if outcome == "failed":
                for cid in chunk:
                    result[cid] = {"status": "batch_failed", "registry_ids": [], "chebi_ids": [],
                                   "chembl_ids": [], "error": error}
                continue
            ids: dict[int, list[str]] = {cid: [] for cid in chunk}
            if outcome == "ok":
                for info in (data.get("InformationList") or {}).get("Information") or []:
                    try:
                        cid = int(info.get("CID"))
                    except (TypeError, ValueError):
                        continue
                    if cid in ids:
                        for rid in info.get("RegistryID") or []:
                            if str(rid) not in ids[cid]:
                                ids[cid].append(str(rid))
            for cid in chunk:
                rids = ids[cid]
                result[cid] = {
                    "status": "success" if rids else "empty_result",
                    "registry_ids": rids,
                    "chebi_ids": sorted({r for r in rids if r.startswith("CHEBI:")}, key=_chebi_number),
                    "chembl_ids": sorted({r for r in rids if re.fullmatch(r"CHEMBL\d+", r)}),
                }
        return result

    def _resolve_chembl_id(self, smiles: str) -> str | None:
        """Resolve SMILES to ChEMBL molecule ID (flexmatch; legacy).

        Kept for :meth:`get_bioactivities`. :meth:`get_cross_references`
        uses an exact standard-InChIKey query instead.
        """
        data = _get_json(
            f"{CHEMBL}/molecule.json",
            params={
                "molecule_structures__canonical_smiles__flexmatch": smiles,
                "limit": 1,
                "format": "json",
            },
        )
        if data and data.get("molecules"):
            return data["molecules"][0].get("molecule_chembl_id")
        return None

    def _chembl_id_by_inchikey(self, inchikey: str) -> tuple[str | None, str | None]:
        """``(chembl_id, error)`` for the molecule whose standard InChIKey is ``inchikey``.

        Rows are verified against the key, so an ignored filter (ChEMBL then
        returns unfiltered molecules) cannot produce a false match.
        """
        data, outcome, error, _ = _call(
            _get_json,
            f"{CHEMBL}/molecule.json",
            params={
                "molecule_structures__standard_inchi_key": inchikey,
                "limit": 5,
                "format": "json",
            },
        )
        if outcome == "failed":
            return None, error
        for mol in (data or {}).get("molecules") or []:
            structures = mol.get("molecule_structures") or {}
            if str(structures.get("standard_inchi_key") or "").upper() == inchikey:
                return mol.get("molecule_chembl_id"), None
        return None, None

    def _index(self, index: Any, kind: Literal["chebi", "kegg"]) -> Any:
        """The index object to use; loads (once) an index given as a path."""
        if index is None:
            index = self.chebi_index if kind == "chebi" else self.kegg_index
        if index is None or not isinstance(index, (str, os.PathLike)):
            return index
        path = os.fspath(index)
        cache_key = (kind, path)
        if cache_key not in self._loaded_indexes:
            if kind == "chebi":
                from .chebi_obo import ChebiOboIndex

                self._loaded_indexes[cache_key] = ChebiOboIndex.from_obo(path)
            else:
                from .kegg_index import KeggIndex

                self._loaded_indexes[cache_key] = KeggIndex.from_dir(path)
        return self._loaded_indexes[cache_key]

    # ==================================================================
    # 1. Structural & Physicochemical (RDKit -- local)
    # ==================================================================

    def get_physicochemical_properties(self, smiles: str) -> dict[str, Any]:
        """Compute physicochemical properties using RDKit.

        Returns MW, LogP, PSA, HBD, HBA, rotatable bonds, Fsp3, QED,
        Lipinski violations, synthetic accessibility, and toxicity alerts.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import QED, Descriptors, rdMolDescriptors
        except ImportError:
            logger.warning("RDKit not available; skipping physicochemical properties")
            return {}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Invalid SMILES: {smiles}"}

        props: dict[str, Any] = {
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 3),
            "tpsa": round(Descriptors.TPSA(mol), 2),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "fsp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "formula": rdMolDescriptors.CalcMolFormula(mol),
        }

        # QED drug-likeness
        try:
            props["qed"] = round(QED.qed(mol), 3)
        except Exception:  # noqa: BLE001
            props["qed"] = None

        # Lipinski Rule of Five violations
        violations = 0
        if props["molecular_weight"] > 500:
            violations += 1
        if props["logp"] > 5:
            violations += 1
        if props["hbd"] > 5:
            violations += 1
        if props["hba"] > 10:
            violations += 1
        props["lipinski_violations"] = violations

        # Synthetic accessibility
        try:
            import os
            import sys

            from rdkit.Chem import RDConfig
            sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
            if sa_path not in sys.path:
                sys.path.insert(0, sa_path)
            from sascorer import calculateScore  # type: ignore[import-not-found]
            props["sa_score"] = round(calculateScore(mol), 2)
        except Exception:  # noqa: BLE001
            props["sa_score"] = None

        # PAINS filter
        try:
            from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
            fc_params = FilterCatalogParams()
            fc_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            catalog = FilterCatalog(fc_params)
            entry = catalog.GetFirstMatch(mol)
            props["pains_alert"] = entry is not None
            if entry:
                props["pains_description"] = entry.GetDescription()
        except Exception:  # noqa: BLE001
            props["pains_alert"] = None

        # Murcko scaffold
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            props["murcko_scaffold"] = Chem.MolToSmiles(scaffold)
        except Exception:  # noqa: BLE001
            props["murcko_scaffold"] = None

        return props

    # ==================================================================
    # 2. Bioactivities & Targets (ChEMBL)
    # ==================================================================

    def get_bioactivities(
        self,
        smiles: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch bioactivity data (IC50, Ki, EC50, etc.) from ChEMBL."""
        chembl_id = self._resolve_chembl_id(smiles)
        if not chembl_id:
            logger.debug("No ChEMBL ID found for %s", smiles[:60])
            return []

        self._sleep()
        data = _get_json(
            f"{CHEMBL}/activity.json",
            params={
                "molecule_chembl_id": chembl_id,
                "limit": limit,
                "format": "json",
            },
        )
        if not data:
            return []

        activities = []
        for act in data.get("activities", []):
            activities.append({
                "target_chembl_id": act.get("target_chembl_id"),
                "target_pref_name": act.get("target_pref_name"),
                "target_organism": act.get("target_organism"),
                "activity_type": act.get("standard_type"),
                "activity_value": act.get("standard_value"),
                "activity_units": act.get("standard_units"),
                "activity_relation": act.get("standard_relation"),
                "assay_type": act.get("assay_type"),
                "pchembl_value": act.get("pchembl_value"),
            })
        return activities

    def get_target_proteins(self, smiles: str) -> list[dict[str, str]]:
        """Get unique target proteins for a molecule from ChEMBL."""
        activities = self.get_bioactivities(smiles, limit=100)
        seen = set()
        targets = []
        for act in activities:
            tid = act.get("target_chembl_id")
            if tid and tid not in seen:
                seen.add(tid)
                targets.append({
                    "target_chembl_id": tid,
                    "target_name": act.get("target_pref_name", ""),
                    "organism": act.get("target_organism", ""),
                })
        return targets

    def get_mechanism_of_action(self, smiles: str) -> list[dict[str, str]]:
        """Get mechanism of action annotations from ChEMBL.

        Delegates to :class:`~embpy.resources.molecule.chembl.ChEMBLAnnotator`,
        which keys the query on the parent molecule. Filtering ``mechanism``
        on ``molecule_chembl_id`` -- as this method used to -- returns nothing
        for any drug whose mechanism ChEMBL registered against a salt form,
        imatinib included.

        ``target_name`` is resolved through the ``target`` endpoint: the
        ``mechanism`` payload carries only ``target_chembl_id``, so reading a
        ``target_name`` straight off it always yielded an empty string.

        Returns
        -------
        list of dict
            Keys ``mechanism``, ``action_type``, ``target_name`` and
            ``target_chembl_id``. See
            :meth:`ChEMBLAnnotator.get_mechanisms` for the full record,
            including references and the direct-interaction and
            disease-efficacy flags.
        """
        mechanisms = self.chembl.get_mechanisms(smiles)
        if not mechanisms:
            return []

        target_ids = [m["target_chembl_id"] for m in mechanisms if m.get("target_chembl_id")]
        details = self.chembl.get_target_details(
            target_ids, include_protein_classes=False,
        )
        return [
            {
                "mechanism": m.get("mechanism") or "",
                "action_type": m.get("action_type") or "",
                "target_name": (
                    details.get(m.get("target_chembl_id") or "", {}).get("target_name")
                    or ""
                ),
                "target_chembl_id": m.get("target_chembl_id") or "",
            }
            for m in mechanisms
        ]

    # ==================================================================
    # 3. Ontology (ChEBI): roles, and the legacy OLS term lookup
    # ==================================================================

    def get_chebi_terms(
        self,
        identifier: str,
        *,
        cid: int | None = None,
        chebi_ids: Iterable[str] | None = None,
    ) -> list[dict[str, str]]:
        """ChEBI term label and definition for a molecule (OLS4).

        This is what ``get_chebi_roles`` used to return: the molecule's *own*
        ChEBI term (label and description), not its roles. For roles use
        :meth:`lookup_chebi_roles`.

        The ChEBI IDs come from ``chebi_ids`` when given, else from PubChem
        RegistryID xrefs of ``cid`` (resolved from ``identifier`` when not
        given). At most the first three IDs are looked up.

        ``[]`` means PubChem or OLS answered with nothing. A failed request
        (an unpatched helper returned ``None`` for a 5xx, timeout or
        connection error) raises :class:`AnnotationServiceError`; exceptions
        from patched helpers propagate unchanged.
        """
        if chebi_ids is None:
            if cid is None:
                cid = self._cid_or_raise(identifier)
            if cid is None:
                return []
            self._sleep()
            data = _require(_get_json, "pubchem", f"{PUBCHEM}/compound/cid/{cid}/xrefs/RegistryID/JSON")
            if not data:
                return []
            ids = []
            for info in data.get("InformationList", {}).get("Information", []):
                for rid in info.get("RegistryID", []):
                    if str(rid).startswith("CHEBI:"):
                        ids.append(str(rid))
        else:
            ids = [i for i in (_normalise_chebi_id(c) for c in chebi_ids) if i]

        terms = []
        for chebi_id in ids[:3]:
            self._sleep()
            chebi_num = chebi_id.replace("CHEBI:", "")
            result = _require(_get_json, "ols", OLS_CHEBI_TERMS, params={"short_form": f"CHEBI_{chebi_num}"})
            if result and result.get("_embedded", {}).get("terms"):
                term = result["_embedded"]["terms"][0]
                terms.append({
                    "chebi_id": chebi_id,
                    "name": term.get("label", ""),
                    "description": term.get("description", [""])[0] if term.get("description") else "",
                })
        return terms

    def get_chebi_roles(self, identifier: str, *, cid: int | None = None) -> list[dict[str, str]]:
        """Deprecated alias of :meth:`get_chebi_terms`.

        Despite its name this never returned roles, only the molecule's own
        ChEBI term label. Use :meth:`get_chebi_terms` for that, or
        :meth:`lookup_chebi_roles` for the ``has role`` relations.
        """
        warnings.warn(
            "MoleculeAnnotator.get_chebi_roles returns ChEBI term labels, not roles, "
            "and is deprecated: use get_chebi_terms() for the same output, or "
            "lookup_chebi_roles() for ChEBI 'has role' relations.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_chebi_terms(identifier, cid=cid)

    def get_chebi_compounds(
        self,
        chebi_ids: Iterable[str],
        chunk_size: int = CHEBI_COMPOUNDS_BATCH,
    ) -> dict[str, dict[str, Any]]:
        """ChEBI 2.0 compound records for many IDs, roles included.

        Sends ``GET {CHEBI_REST}/compounds/?chebi_ids=<id,...>`` per chunk
        (at most :data:`CHEBI_COMPOUNDS_BATCH` IDs). Each answer entry is
        unwrapped from ``data``; ``exists`` and ``primary_chebi_id`` are
        honoured, so a secondary ID reports its primary entity.

        Returns
        -------
        dict
            ``{requested_id: {"status", "chebi_id", "name", "stars",
            "inchikey", "roles"}}`` with status ``success``, ``not_found``
            (``exists`` false, or absent from the answer) or
            ``batch_failed``. ``roles`` are the outgoing ``has role``
            relations as ``{"chebi_id", "name"}``.
        """
        ids = list(dict.fromkeys(i for i in (_normalise_chebi_id(c) for c in chebi_ids) if i))
        chunk_size = max(1, min(int(chunk_size), CHEBI_COMPOUNDS_BATCH))
        result: dict[str, dict[str, Any]] = {}
        for n, chunk in enumerate(_chunks(ids, chunk_size)):
            if n:
                self._sleep()
            data, outcome, error, _ = _call(
                _get_json, f"{CHEBI_REST}/compounds/", params={"chebi_ids": ",".join(chunk)},
            )
            for chebi_id in chunk:
                empty = {"chebi_id": None, "name": None, "stars": None, "inchikey": None, "roles": []}
                if outcome == "failed":
                    result[chebi_id] = {"status": "batch_failed", **empty, "error": error}
                    continue
                entry = (data or {}).get(chebi_id) if isinstance(data, dict) else None
                if entry is None and isinstance(data, dict):
                    entry = next(
                        (v for v in data.values()
                         if isinstance(v, dict) and v.get("standardized_chebi_id") == chebi_id),
                        None,
                    )
                record = entry.get("data") if isinstance(entry, dict) else None
                if not isinstance(record, dict) or entry.get("exists") is False:
                    result[chebi_id] = {"status": "not_found", **empty}
                    continue
                relations = (record.get("ontology_relations") or {}).get("outgoing_relations") or []
                roles: list[dict[str, str]] = []
                for rel in relations:
                    if rel.get("relation_type") != "has role":
                        continue
                    role = {"chebi_id": f"CHEBI:{rel.get('final_id')}", "name": rel.get("final_name") or ""}
                    if role not in roles:
                        roles.append(role)
                structure = record.get("default_structure") or {}
                result[chebi_id] = {
                    "status": "success",
                    "chebi_id": entry.get("primary_chebi_id") or record.get("chebi_accession") or chebi_id,
                    "name": record.get("name"),
                    "stars": record.get("stars"),
                    "inchikey": structure.get("standard_inchi_key") if isinstance(structure, dict) else None,
                    "roles": roles,
                }
        return result

    def _chebi_es_search(self, term: str) -> tuple[list[dict[str, Any]] | None, str | None]:
        """ChEBI ``es_search`` hits as ``(hits, error)``; 404 gives ``[]``."""
        data, outcome, error, _ = _call(_get_json, f"{CHEBI_REST}/es_search/", params={"term": term})
        if outcome == "not_found":
            return [], None
        if outcome == "failed":
            return None, error
        if isinstance(data, dict) and "results" not in data and "No compound matches" in str(data.get("detail", "")):
            return [], None
        hits = []
        for item in (data or {}).get("results") or []:
            src = item.get("_source") or {}
            chebi_id = _normalise_chebi_id(src.get("chebi_accession"))
            if chebi_id:
                hits.append({
                    "chebi_id": chebi_id,
                    "name": src.get("name"),
                    "stars": src.get("stars"),
                    "inchikey": str(src.get("inchikey") or "").upper() or None,
                })
        return hits, None

    def _lookup_chebi_online(self, key: str) -> dict[str, Any]:
        """Online counterpart of ``ChebiOboIndex.lookup`` (same result keys)."""
        base = {"query_inchikey": key, "source": "chebi_rest"}
        hits, error = self._chebi_es_search(key)
        if hits is None:
            return _chebi_result(status="service_error", error=error, **base)
        candidates = [h for h in hits if h["inchikey"] == key]
        tier = "exact"
        if not candidates:
            block = key.split("-", 1)[0]
            self._sleep()
            hits, error = self._chebi_es_search(block)
            if hits is None:
                return _chebi_result(status="service_error", error=error, **base)
            candidates = [h for h in hits
                          if h["inchikey"] and h["inchikey"].split("-", 1)[0] == block and h["inchikey"] != key]
            tier = "connectivity"
            if not candidates:
                return _chebi_result(status="no_chebi_entity", **base)
            if len({h["inchikey"] for h in candidates}) > 1:
                return _chebi_result(
                    status="ambiguous_connectivity", match_tier="connectivity",
                    candidates=[{**h, "match_tier": tier} for h in candidates], **base,
                )
        # Entities sharing the deciding key: best = most curated (stars), then
        # lowest ChEBI number; roles are the union over all of them, as in
        # the offline index.
        ranked = sorted(candidates, key=lambda h: (-(h.get("stars") or 0), _chebi_number(h["chebi_id"])))
        self._sleep()
        records = self.get_chebi_compounds([h["chebi_id"] for h in ranked])
        failed = [r for r in records.values() if r["status"] != "success"]
        if failed:
            return _chebi_result(
                status="service_error", match_tier=tier, candidates=ranked,
                error=failed[0].get("error") or "compounds/ did not return every matched entity",
                **base,
            )
        roles: dict[str, dict[str, str]] = {}
        out_candidates = []
        for hit in ranked:
            record = records[hit["chebi_id"]]
            out_candidates.append({**hit, "match_tier": tier, "roles": record["roles"]})
            for role in record["roles"]:
                roles.setdefault(role["chebi_id"], role)
        best = records[ranked[0]["chebi_id"]]
        return _chebi_result(
            status="success" if roles else "matched_no_role",
            chebi_id=best["chebi_id"],
            name=best["name"] if best["name"] is not None else ranked[0].get("name"),
            stars=best["stars"] if best["stars"] is not None else ranked[0].get("stars"),
            match_tier=tier,
            inchikey=best.get("inchikey") or ranked[0].get("inchikey"),
            roles=list(roles.values()),
            candidates=out_candidates,
            **base,
        )

    def lookup_chebi_roles(
        self,
        identifier: str | None = None,
        *,
        inchikey: str | None = None,
        index: Any = None,
        online: bool = True,
    ) -> dict[str, Any]:
        """ChEBI ``has role`` relations of the entity matching a structure.

        The structure is matched on its standard InChIKey (``inchikey``, or
        computed with RDKit from ``identifier``). Tier ``exact`` is an
        identical InChIKey; tier ``connectivity`` matches only the first
        (connectivity) block and is reported separately -- it is used only
        when no exact entity exists, and several stereo variants give
        ``ambiguous_connectivity`` rather than a silent pick. Only direct
        ``has role`` relations are returned; nothing is inherited via
        ``is_a``.

        Parameters
        ----------
        identifier
            SMILES, InChI, InChIKey or (online only) a name.
        inchikey
            A precomputed standard InChIKey; takes precedence.
        index
            A :class:`~embpy.resources.molecule.chebi_obo.ChebiOboIndex`
            (or a path to ``chebi_core.obo(.gz)``) for an offline,
            release-pinned lookup. Defaults to the annotator's
            ``chebi_index``. With an index no request is sent.
        online
            Without an index, query the ChEBI 2.0 REST API (``es_search`` by
            InChIKey, then ``compounds/``). Also allows a PubChem name lookup
            for a non-structure identifier.

        Returns
        -------
        dict
            ``status`` (``success`` | ``matched_no_role`` |
            ``no_chebi_entity`` | ``ambiguous_connectivity`` |
            ``invalid_structure`` | ``service_error``), ``chebi_id``,
            ``name``, ``stars``, ``match_tier``, ``inchikey`` (the matched
            ChEBI entity's key), ``query_inchikey``, ``roles``
            (``[{"chebi_id", "name"}]``, union over the entities sharing the
            deciding key), ``candidates``, ``connectivity`` (offline only: the
            supplementary connectivity tier when the exact tier matched) and
            ``release`` (``None`` online, where ``source`` is
            ``"chebi_rest"``).
        """
        idx = self._index(index, "chebi")
        key, error = self._inchikey_for(identifier, inchikey=inchikey, allow_name_lookup=online)
        if key is None:
            return _chebi_result(status="invalid_structure", error=error,
                                 release=getattr(idx, "release", None))
        if idx is not None:
            out = _chebi_result(**(idx.lookup(key) or {}))
            if out.get("query_inchikey") is None:
                out["query_inchikey"] = key
            if out.get("release") is None:
                out["release"] = getattr(idx, "release", None)
            return out
        if not online:
            return _chebi_result(status="service_error", query_inchikey=key,
                                 error="no ChEBI index given and online=False")
        return self._lookup_chebi_online(key)

    # ==================================================================
    # 4. Pathway Annotations (KEGG)
    # ==================================================================

    def lookup_kegg(
        self,
        identifier: str | None = None,
        *,
        inchikey: str | None = None,
        index: Any = None,
        cid: int | None = None,
        chebi_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """KEGG COMPOUND/DRUG entries and pathways for a structure.

        Looks the structure's standard InChIKey up in a KEGG index built from
        KEGG's own bulk tables (``conv/{compound,drug}/pubchem`` gives
        PubChem *SIDs*, mapped to CIDs and InChIKeys through PubChem).
        A PubChem CID is never sent to KEGG's ``pubchem:`` namespace, which
        holds SIDs: doing so mapped CIDs to unrelated compounds.

        Parameters
        ----------
        identifier
            SMILES, InChI or InChIKey (a name is resolved through PubChem).
        inchikey
            A precomputed standard InChIKey; takes precedence.
        index
            A :class:`~embpy.resources.molecule.kegg_index.KeggIndex`, or the
            directory written by
            :func:`~embpy.resources.molecule.kegg_index.build_kegg_reference`.
            Defaults to the annotator's ``kegg_index``.
        cid, chebi_ids
            Identifiers resolved by other routes, forwarded to
            :meth:`KeggIndex.lookup` only as identity evidence: an entry they
            reach whose InChIKey differs gives ``identity_mismatch``, never a
            match.

        Returns
        -------
        dict
            The :meth:`KeggIndex.lookup` result -- ``status`` (``success`` |
            ``not_in_kegg`` | ``kegg_entry_no_pathway`` |
            ``identity_mismatch``), ``entries``, ``compound_pathways``,
            ``drug_maps``, ``target_pathways``, ``release`` -- plus
            ``inchikey``. ``status`` is ``index_required`` when no index is
            available (no online route is attempted) and
            ``invalid_structure`` when no InChIKey can be derived.
        """
        idx = self._index(index, "kegg")
        if idx is None:
            return _kegg_result(
                status="index_required", inchikey=inchikey,
                error="KEGG lookups need a KeggIndex (see kegg_index.build_kegg_reference)",
            )
        key, error = self._inchikey_for(identifier, inchikey=inchikey)
        if key is None:
            return _kegg_result(status="invalid_structure", error=error,
                                release=getattr(idx, "release", None))
        extra: dict[str, Any] = {}
        if cid is not None:
            extra["cid"] = cid
        if chebi_ids is not None:
            extra["chebi_ids"] = list(chebi_ids)
        out = _kegg_result(**(idx.lookup(key, **extra) or {}))
        if out.get("inchikey") is None:
            out["inchikey"] = key
        if out.get("release") is None:
            out["release"] = getattr(idx, "release", None)
        return out

    def get_kegg_pathways(
        self,
        identifier: str,
        *,
        inchikey: str | None = None,
        index: Any = None,
    ) -> list[dict[str, Any]]:
        """KEGG pathways for a molecule, as one flat list (backward compatible).

        The union of :meth:`lookup_kegg`'s ``compound_pathways``,
        ``drug_maps`` and ``target_pathways``, each dict carrying the legacy
        ``pathway_id`` / ``pathway_name`` keys plus ``id``, ``name`` and
        ``kind`` (``compound_pathway`` | ``drug_map`` | ``target_pathway``).
        Use :meth:`lookup_kegg` to keep the three kinds and the match tier
        apart. Without a KEGG index this logs a warning and returns ``[]``.
        """
        result = self.lookup_kegg(identifier, inchikey=inchikey, index=index)
        if result["status"] == "index_required":
            if not self._warned_kegg_index:
                logger.warning(
                    "get_kegg_pathways needs a KEGG index (MoleculeAnnotator(kegg_index=...) "
                    "or index=...); returning no pathways",
                )
                self._warned_kegg_index = True
            return []
        return self._flatten_kegg(result)

    @staticmethod
    def _flatten_kegg(result: dict[str, Any]) -> list[dict[str, Any]]:
        """The three pathway kinds of a :meth:`lookup_kegg` result as one list."""
        pathways: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for kind, field in (("compound_pathway", "compound_pathways"),
                            ("drug_map", "drug_maps"),
                            ("target_pathway", "target_pathways")):
            for item in result.get(field) or []:
                if not isinstance(item, dict):
                    continue
                pid = item.get("id") or item.get("pathway_id")
                if not pid or (kind, pid) in seen:
                    continue
                seen.add((kind, pid))
                name = item.get("name") or item.get("pathway_name") or pid
                pathways.append({**item, "id": pid, "name": name, "pathway_id": pid,
                                 "pathway_name": name, "kind": kind})
        return pathways

    # ==================================================================
    # 5. Cross-Database IDs (PubChem; ChEMBL optional)
    # ==================================================================

    def get_cross_references(
        self,
        identifier: str,
        *,
        cid: int | None = None,
        chembl_id: str | None = None,
        include_chembl: bool = True,
        registry_ids: Iterable[str] | None = None,
        inchikey: str | None = None,
    ) -> dict[str, Any]:
        """Map a molecule to IDs in multiple databases.

        PubChem first: the CID (``cid``, or resolved from ``identifier``),
        its RegistryID xrefs (``registry_ids`` when already fetched, e.g.
        by :meth:`get_pubchem_xrefs_batch`) and its InChIKey (``inchikey``
        when known). ChEMBL runs last: ``chembl_id`` when given, otherwise
        (``include_chembl``) an exact standard-InChIKey query. A ChEMBL
        failure never drops the PubChem fields; it sets ``status`` to
        ``"partial"`` and is recorded under ``errors``. That covers a failed
        request through the unpatched module helper and ChEMBL's typed
        errors; an exception raised by a patched helper (e.g. a driver's
        transport refusing the request) propagates.

        RegistryIDs pool every depositor's identifiers, so no KEGG ID is
        read from them (a ``C\\d{5}`` pattern also matches NCIt codes); use
        :meth:`lookup_kegg` for KEGG.

        Returns
        -------
        dict
            ``pubchem_cid``, ``pubchem_status``, ``chebi_id``,
            ``drugbank_id``, ``hmdb_id``, ``chembl_ids_pubchem`` (ChEMBL IDs
            PubChem depositors list), ``inchikey``, ``chembl_id``,
            ``status`` (``success`` | ``partial`` | ``no_match`` |
            ``ambiguous`` | ``invalid_structure`` | ``service_error``) and
            ``errors``.
        """
        errors: list[str] = []
        if cid is None:
            res = self.resolve_pubchem_cid(identifier)
            cid, pubchem_status = res["cid"], res["status"]
            if res.get("error"):
                errors.append(f"pubchem: {res['error']}")
        else:
            pubchem_status = "matched"
        refs: dict[str, Any] = {"pubchem_cid": cid, "pubchem_status": pubchem_status}

        if cid is None:
            refs["status"] = pubchem_status
            refs["errors"] = errors
            if chembl_id is not None:
                refs["chembl_id"] = chembl_id
            return refs

        if registry_ids is None:
            self._sleep()
            data, outcome, error, _ = _call(_get_json, f"{PUBCHEM}/compound/cid/{cid}/xrefs/RegistryID/JSON")
            rids: list[str] = []
            if outcome == "failed":
                errors.append(f"pubchem xrefs: {error}")
            for info in ((data or {}).get("InformationList") or {}).get("Information") or []:
                rids.extend(str(r) for r in info.get("RegistryID") or [])
        else:
            rids = [str(r) for r in registry_ids]
        for rid in rids:
            if rid.startswith("CHEBI:"):
                refs.setdefault("chebi_id", rid)
            elif re.fullmatch(r"DB\d{5}", rid):
                refs.setdefault("drugbank_id", rid)
            elif rid.startswith("HMDB"):
                refs.setdefault("hmdb_id", rid)
        refs["chembl_ids_pubchem"] = sorted({r for r in rids if re.fullmatch(r"CHEMBL\d+", r)})

        if inchikey is None:
            self._sleep()
            data, outcome, error, _ = _call(_get_json, f"{PUBCHEM}/compound/cid/{cid}/property/InChIKey/JSON")
            if outcome == "failed":
                errors.append(f"pubchem inchikey: {error}")
            props = ((data or {}).get("PropertyTable") or {}).get("Properties") or []
            if props:
                inchikey = props[0].get("InChIKey")
        refs["inchikey"] = inchikey

        if chembl_id is not None:
            refs["chembl_id"] = chembl_id
        elif include_chembl:
            refs["chembl_id"] = None
            if inchikey:
                self._sleep()
                try:
                    refs["chembl_id"], chembl_error = self._chembl_id_by_inchikey(str(inchikey).upper())
                except (ChEMBLServiceError, ChEMBLFilterIgnoredError) as exc:
                    # Only ChEMBL's own typed errors are absorbed; an unpatched
                    # helper's failure arrives as ``chembl_error``, and anything a
                    # patched helper raises (refusal, circuit open) propagates.
                    chembl_error = f"{type(exc).__name__}: {exc}"
                if chembl_error:
                    errors.append(f"chembl: {chembl_error}")
                    refs["chembl_status"] = "failed"
            else:
                errors.append("chembl: no InChIKey to query with")

        refs["errors"] = errors
        refs["status"] = "partial" if errors else "success"
        return refs

    # ==================================================================
    # 6. Assay-name keywords (PubChem bioassays)
    # ==================================================================

    def get_disease_associations(
        self,
        identifier: str,
        *,
        cid: int | None = None,
        keywords: Iterable[str] = ASSAY_NAME_KEYWORDS,
    ) -> list[dict[str, Any]]:
        """Disease keywords in the names of PubChem assays where the compound was *Active*.

        Reads ``{PUBCHEM}/compound/cid/{cid}/assaysummary/JSON``, locating
        the ``Assay Name``, ``Activity Outcome``, ``AID`` and ``Target
        Accession`` columns by name from ``Table.Columns``, keeping only rows
        whose outcome is ``Active`` and scanning all of them. Keywords match
        whole words, case-insensitively.

        This is an **assay-name keyword** signal, not a curated disease
        association: most matches are cell-line screens (e.g. NCI-60). The
        method name is kept for backward compatibility.

        Returns
        -------
        list of dict
            One per keyword: ``keyword`` (also under the legacy key
            ``disease``), ``assay`` (the first matching assay name),
            ``aids``, ``n_active_assays``, ``target_accessions`` and
            ``source`` (``"pubchem_assay_name_keyword"``). ``[]`` means
            PubChem has no assay summary (404 / ``PUGREST.NotFound``) or no
            Active assay name matched.

        Raises
        ------
        AnnotationServiceError
            When the CID resolution or the assaysummary request failed
            (5xx, timeout, connection error, non-NotFound ``Fault``), so an
            outage never reads as "no keywords". Exceptions from patched
            helpers propagate unchanged.
        """
        if cid is None:
            cid = self._cid_or_raise(identifier)
        if cid is None:
            return []

        self._sleep()
        data = _require(_get_json, "pubchem", f"{PUBCHEM}/compound/cid/{cid}/assaysummary/JSON")
        if not data:
            return []
        table = data.get("Table") or {}
        columns = [str(c) for c in (table.get("Columns") or {}).get("Column") or []]
        try:
            i_name = columns.index("Assay Name")
            i_outcome = columns.index("Activity Outcome")
        except ValueError:
            logger.warning("PubChem assaysummary for CID %s lacks Assay Name/Activity Outcome columns", cid)
            return []
        i_aid = columns.index("AID") if "AID" in columns else None
        i_target = columns.index("Target Accession") if "Target Accession" in columns else None

        patterns = {k: re.compile(rf"\b{re.escape(k.lower())}\b") for k in keywords}
        found: dict[str, dict[str, Any]] = {}
        for row in table.get("Row") or []:
            cells = row.get("Cell") or []
            if len(cells) <= max(i_name, i_outcome):
                continue
            if str(cells[i_outcome]).strip().casefold() != "active":
                continue
            assay_name = str(cells[i_name])
            lowered = assay_name.lower()
            for keyword, pattern in patterns.items():
                if not pattern.search(lowered):
                    continue
                hit = found.setdefault(keyword, {
                    "keyword": keyword, "disease": keyword, "assay": assay_name,
                    "aids": [], "n_active_assays": 0, "target_accessions": [],
                    "source": "pubchem_assay_name_keyword",
                })
                aid = str(cells[i_aid]) if i_aid is not None and i_aid < len(cells) else None
                if aid is None or aid not in hit["aids"]:
                    hit["n_active_assays"] += 1
                    if aid is not None:
                        hit["aids"].append(aid)
                if i_target is not None and i_target < len(cells):
                    target = str(cells[i_target]).strip()
                    if target and target not in hit["target_accessions"]:
                        hit["target_accessions"].append(target)
        return list(found.values())

    # ==================================================================
    # Convenience: one-call aggregation
    # ==================================================================

    def annotate(
        self,
        identifier: str,
        sources: Literal["all", "structural", "bioactivity", "ontology",
                         "pathways", "xrefs", "diseases", "drug",
                         "target_profile", "metabolism",
                         "analogs"] | list[str] = "all",
    ) -> dict[str, Any]:
        """Aggregate all annotations for a molecule in one call.

        Parameters
        ----------
        identifier
            SMILES, compound name, InChI, PubChem CID, or ChEMBL ID.
        sources
            Which annotation sources to query. ``"all"`` queries
            everything except ``"analogs"``. Pass a list to select
            specific ones:

            ``"structural"``
                RDKit physicochemical properties.
            ``"bioactivity"``
                ChEMBL activities, unique targets, mechanism of action.
            ``"ontology"``
                ChEBI ``has role`` relations (:meth:`lookup_chebi_roles`;
                offline with the annotator's ``chebi_index``, else ChEBI
                REST). Stored as a status dict under ``chebi_roles``.
            ``"pathways"``
                KEGG entries and pathways (:meth:`lookup_kegg`, under
                ``kegg``; flattened under ``kegg_pathways``). Needs the
                annotator's ``kegg_index``; otherwise ``index_required``.
            ``"xrefs"``
                PubChem cross-database IDs (ChEMBL last, non-fatal).
            ``"diseases"``
                Disease keywords in names of PubChem assays where the
                compound was active (weak evidence).
            ``"drug"``
                ChEMBL clinical record: development phase, indications,
                safety warnings, ATC class, synonyms, mechanisms, forms.
            ``"target_profile"``
                Per-target potency summary with gene symbols and protein
                families, plus a selectivity summary.
            ``"metabolism"``
                ChEMBL metabolite and enzyme records.
            ``"analogs"``
                Structurally similar ChEMBL compounds. Excluded from
                ``"all"`` because a similarity search scans the whole
                database and costs far more than a keyed lookup.

        Returns
        -------
        dict
            Nested, with one key per annotation category. The four
            ChEMBL groups above land together under ``result["drug"]``
            -- see :meth:`ChEMBLAnnotator.annotate` for its shape.
        """
        if sources == "all":
            sources_list = [
                "structural", "bioactivity", "ontology",
                "pathways", "xrefs", "diseases",
                "drug", "target_profile", "metabolism",
            ]
        elif isinstance(sources, str):
            sources_list = [sources]
        else:
            sources_list = list(sources)

        smiles = self._resolve_to_smiles(identifier)
        result: dict[str, Any] = {
            "identifier": identifier,
            "canonical_smiles": smiles,
        }

        # ChEMBL resolves names, ChEMBL IDs and structures on its own, so the
        # drug record is worth fetching even for a compound RDKit and PubChem
        # could not turn into a SMILES -- which is the common case for
        # biologics and for screen labels that are trade names.
        chembl_sources = _chembl_sources_for(sources_list)
        if chembl_sources:
            result["drug"] = self.chembl.annotate(
                identifier, sources=chembl_sources,
            )

        if smiles is None:
            logger.warning("Could not resolve '%s' to SMILES", identifier)
            result["error"] = "Could not resolve identifier to SMILES"
            return result

        if "structural" in sources_list:
            result["physicochemical"] = self.get_physicochemical_properties(smiles)

        if "bioactivity" in sources_list:
            result["bioactivities"] = self.get_bioactivities(smiles)
            result["targets"] = self.get_target_proteins(smiles)
            result["mechanism_of_action"] = self.get_mechanism_of_action(smiles)

        if "ontology" in sources_list:
            result["chebi_roles"] = self.lookup_chebi_roles(smiles)

        if "pathways" in sources_list:
            kegg = self.lookup_kegg(smiles)
            result["kegg"] = kegg
            result["kegg_pathways"] = self._flatten_kegg(kegg)

        if "xrefs" in sources_list or "diseases" in sources_list:
            # One CID resolution shared by both PubChem sources.
            pubchem = self.resolve_pubchem_cid(smiles)
            result["pubchem_identity"] = pubchem
            if "xrefs" in sources_list:
                result["cross_references"] = (
                    self.get_cross_references(smiles, cid=pubchem["cid"])
                    if pubchem["cid"] is not None
                    else {"pubchem_cid": None, "pubchem_status": pubchem["status"],
                          "status": pubchem["status"], "errors": []}
                )
            if "diseases" in sources_list:
                try:
                    result["disease_associations"] = (
                        self.get_disease_associations(smiles, cid=pubchem["cid"])
                        if pubchem["cid"] is not None else []
                    )
                except AnnotationServiceError as exc:
                    # Unknown, not empty: the request failed.
                    result["disease_associations"] = None
                    result["disease_associations_error"] = str(exc)

        return result

    def annotate_batch(
        self,
        identifiers: list[str],
        sources: str | list[str] = "all",
    ) -> dict[str, dict[str, Any]]:
        """Annotate a list of molecules.

        Returns
        -------
        Dict mapping identifier to annotation dict.
        """
        results: dict[str, dict[str, Any]] = {}
        total = len(identifiers)
        for i, ident in enumerate(identifiers):
            results[ident] = self.annotate(ident, sources=sources)
            if (i + 1) % 10 == 0:
                logger.info("Annotated %d/%d molecules", i + 1, total)
        logger.info("Annotated %d molecules total", total)
        return results

    def annotate_adata(
        self,
        adata,  # anndata.AnnData
        column: str,
        sources: str | list[str] = "all",
        copy: bool = True,
    ):
        """Annotate perturbations in an AnnData with molecule metadata.

        Reads molecule identifiers from ``adata.obs[column]``, fetches
        annotations for each unique identifier, and stores:

        - Scalar properties (MW, LogP, QED, etc.) as new columns in
          ``adata.obs``
        - Full annotation dicts in ``adata.uns["molecule_annotations"]``

        Parameters
        ----------
        adata
            AnnData with a molecule identifier column in ``.obs``.
        column
            Column in ``adata.obs`` containing identifiers.
        sources
            Annotation sources to query (see :meth:`annotate`).
        copy
            If ``True``, operate on a copy.

        Returns
        -------
        AnnData with molecule annotations added.
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
            "Annotating %d unique molecules from %d cells",
            len(unique_ids), len(identifiers),
        )

        annotations = self.annotate_batch(unique_ids, sources=sources)
        adata.uns["molecule_annotations"] = annotations

        scalar_keys = [
            "molecular_weight", "logp", "tpsa", "hbd", "hba",
            "rotatable_bonds", "qed", "lipinski_violations",
            "fsp3", "heavy_atoms",
        ]

        for key in scalar_keys:
            values = []
            for ident in identifiers:
                ann = annotations.get(ident, {})
                phys = ann.get("physicochemical", {})
                values.append(phys.get(key))
            adata.obs[f"mol_{key}"] = values

        n_targets = []
        for ident in identifiers:
            ann = annotations.get(ident, {})
            targets = ann.get("targets", [])
            n_targets.append(len(targets))
        adata.obs["mol_n_targets"] = n_targets

        smiles_col = []
        for ident in identifiers:
            ann = annotations.get(ident, {})
            smiles_col.append(ann.get("canonical_smiles", ""))
        adata.obs["mol_canonical_smiles"] = smiles_col

        # ChEMBL drug columns, present only when a ChEMBL source group was
        # requested. Kept under the `drug_` prefix so the clinical record
        # stays distinguishable from the `mol_` structural properties.
        drug_rows = [
            annotations.get(ident, {}).get("drug") or {} for ident in identifiers
        ]
        if any(drug_rows):
            for name, values in chembl_summary_columns(drug_rows).items():
                adata.obs[name] = values
            adata.obs["drug_n_indications_at_limit"] = [
                n >= DEFAULT_INDICATIONS for n in adata.obs["drug_n_indications"]
            ]
            adata.uns["chembl_annotation_limits"] = {
                "drug_n_indications": DEFAULT_INDICATIONS,
                "drug_n_activities": DEFAULT_ACTIVITIES,
                "drug_n_targets": DEFAULT_TARGETS,
                "drug_n_metabolites": DEFAULT_METABOLITES,
            }
            logger.info(
                "ChEMBL drug annotations stored in adata.obs (drug_*); "
                "%d/%d observations matched a ChEMBL compound",
                sum(1 for r in drug_rows if r.get("found")), len(drug_rows),
            )

        logger.info("Molecule annotations stored in adata.obs (mol_*) and adata.uns")
        return adata
