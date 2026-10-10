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

Two ChEMBL behaviours shape this module.

First, drug-level annotation hangs off the **parent** molecule, not the salt
that was dosed.  ``mechanism?molecule_chembl_id=CHEMBL941`` -- imatinib --
returns nothing at all, because the mechanism row belongs to imatinib
mesylate (``CHEMBL1642``) and names ``CHEMBL941`` only as its parent.  Every
drug-level query here resolves the molecule hierarchy first and filters on
``parent_molecule_chembl_id``, which for imatinib is the difference between
0 and 4 mechanisms, and between 52 and 134 indications.

Second, ChEMBL silently ignores filter parameters it does not recognise and
returns the unfiltered collection -- a misspelled field name yields a
plausible-looking response covering the whole database.  Field names here
follow the ChEMBL_37 schema exactly.
"""

from __future__ import annotations

import logging
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Literal

import requests

logger = logging.getLogger(__name__)

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"

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

#: ``max_phase`` in ``molecule_dictionary``.  ``None`` means a preclinical
#: compound with bioactivity data but no clinical record.
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
# Characters that can legally appear in a SMILES string. Used only to guess
# whether an identifier is a structure or a name when RDKit is unavailable.
_SMILES_CHARS_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]()=#$:/\\.%*]+$")


def _get_json(url: str, params: dict | None = None, timeout: int = 30) -> dict | None:
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:  # noqa: BLE001
        logger.debug("Request failed for %s: %s", url, e)
        return None


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
        return "preclinical"
    return MAX_PHASE_LABELS.get(f)


def _protein_class_path(class_desc: str | None) -> list[str]:
    """Split a ``protein_class_desc`` into its hierarchy levels.

    ChEMBL joins the levels with two spaces, e.g.
    ``"enzyme  kinase  protein kinase  tk  abl"``.
    """
    if not class_desc:
        return []
    return [part.strip() for part in class_desc.split("  ") if part.strip()]


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
        How the match was made: ``"chembl_id"``, ``"smiles"``,
        ``"pref_name"``, ``"synonym"`` or ``"search"``.
    """

    identifier: str
    molecule_chembl_id: str | None = None
    parent_chembl_id: str | None = None
    pref_name: str | None = None
    matched_by: str | None = None

    @property
    def found(self) -> bool:
        """Whether the identifier matched a ChEMBL molecule."""
        return self.molecule_chembl_id is not None


class ChEMBLAnnotator:
    """Aggregate drug and perturbation annotations from ChEMBL.

    Parameters
    ----------
    rate_limit_delay
        Seconds to wait between API calls (default 0.2).
    cache
        Cache identifier resolutions, molecule records and protein-class
        lookups on the instance.  A full annotation touches the molecule
        record from several methods, and a screen's compounds share target
        classes, so caching cuts the request count substantially.  Set to
        ``False`` to always hit the API.

    Notes
    -----
    A full :meth:`annotate` call issues roughly a dozen requests per
    compound.  For a large screen, narrow ``sources`` to what the analysis
    actually uses.

    Examples
    --------
    >>> ann = ChEMBLAnnotator()
    >>> status = ann.get_development_status("imatinib")
    >>> status["development_phase"]
    'approved'
    >>> [m["mechanism"] for m in ann.get_mechanisms("imatinib")][:1]
    ['Tyrosine-protein kinase ABL inhibitor']
    """

    def __init__(self, rate_limit_delay: float = 0.2, cache: bool = True) -> None:
        self.delay = rate_limit_delay
        self.cache = cache
        self._resolutions: dict[str, ChEMBLResolution] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self._protein_classes: dict[int, dict[str, Any]] = {}
        self._target_details: dict[str, dict[str, Any]] = {}

    def _sleep(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    # ==================================================================
    # Identifier resolution
    # ==================================================================

    @staticmethod
    def _classify(identifier: str) -> str:
        """Guess whether an identifier is a ChEMBL ID, a SMILES or a name."""
        s = identifier.strip()
        if _CHEMBL_ID_RE.match(s):
            return "chembl_id"
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
        # SMILES and a plausible code -- so try it as a structure and let
        # resolve() fall through to the name lookups if that misses.
        return "smiles" if _SMILES_CHARS_RE.match(s) else "name"

    def get_molecule_record(self, chembl_id: str) -> dict[str, Any] | None:
        """Fetch the full ``molecule`` record for a ChEMBL ID."""
        key = chembl_id.upper()
        if self.cache and key in self._records:
            return self._records[key]
        self._sleep()
        record = _get_json(f"{CHEMBL}/molecule/{key}.json")
        if record and self.cache:
            self._records[key] = record
        return record

    def _molecules_where(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        self._sleep()
        data = _get_json(f"{CHEMBL}/molecule.json", params={**params, "format": "json"})
        return (data or {}).get("molecules", []) or []

    def resolve(self, identifier: str) -> ChEMBLResolution:
        """Map any identifier onto a ChEMBL molecule and its parent.

        Accepts a ChEMBL ID, a SMILES string, a preferred name, a trade
        name or a research code.  The parent is resolved from the molecule
        hierarchy because that is what drug-level ChEMBL annotation is
        keyed on -- see the module docstring.

        Parameters
        ----------
        identifier
            ChEMBL ID, SMILES, or compound name.

        Returns
        -------
        ChEMBLResolution
            With ``found`` ``False`` if nothing matched.
        """
        s = identifier.strip()
        if self.cache and s in self._resolutions:
            return self._resolutions[s]

        resolution = self._resolve_uncached(s)
        if self.cache:
            self._resolutions[s] = resolution
        if not resolution.found:
            logger.debug("No ChEMBL match for '%s'", s[:60])
        return resolution

    def _resolve_uncached(self, s: str) -> ChEMBLResolution:
        kind = self._classify(s)

        if kind == "chembl_id":
            record = self.get_molecule_record(s)
            if record:
                return self._resolution_from_record(s, record, "chembl_id")
            return ChEMBLResolution(identifier=s)

        # Ordered cheapest/most-specific first. A SMILES-looking string still
        # falls through to the name lookups, because without RDKit the guess
        # is only a guess.
        attempts: list[tuple[str, dict[str, Any]]] = []
        if kind == "smiles":
            attempts.append(
                (
                    "smiles",
                    {"molecule_structures__canonical_smiles__flexmatch": s, "limit": 1},
                )
            )
        attempts += [
            ("pref_name", {"pref_name__iexact": s, "limit": 1}),
            (
                "synonym",
                {"molecule_synonyms__molecule_synonym__iexact": s, "limit": 1},
            ),
        ]

        for matched_by, params in attempts:
            molecules = self._molecules_where(params)
            if molecules:
                return self._resolution_from_record(s, molecules[0], matched_by)

        # Last resort: ChEMBL's own free-text search, which tolerates
        # spelling and formatting differences the exact filters do not.
        self._sleep()
        data = _get_json(f"{CHEMBL}/molecule/search.json", params={"q": s, "limit": 1})
        molecules = (data or {}).get("molecules", []) or []
        if molecules:
            return self._resolution_from_record(s, molecules[0], "search")

        return ChEMBLResolution(identifier=s)

    def _resolution_from_record(
        self, identifier: str, record: dict[str, Any], matched_by: str
    ) -> ChEMBLResolution:
        chembl_id = record.get("molecule_chembl_id")
        if chembl_id and self.cache:
            self._records.setdefault(chembl_id.upper(), record)
        hierarchy = record.get("molecule_hierarchy") or {}
        parent = hierarchy.get("parent_chembl_id") or chembl_id
        return ChEMBLResolution(
            identifier=identifier,
            molecule_chembl_id=chembl_id,
            parent_chembl_id=parent,
            pref_name=record.get("pref_name"),
            matched_by=matched_by,
        )

    def _parent_id(self, identifier: str) -> str | None:
        return self.resolve(identifier).parent_chembl_id

    def _collection(
        self, endpoint: str, params: dict[str, Any], key: str
    ) -> list[dict[str, Any]]:
        self._sleep()
        data = _get_json(
            f"{CHEMBL}/{endpoint}.json", params={**params, "format": "json"}
        )
        return (data or {}).get(key, []) or []

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
            ChEMBL ID, SMILES, or compound name.

        Returns
        -------
        dict
            Empty if the compound is not in ChEMBL.  ``max_phase`` is the
            raw ChEMBL value and ``development_phase`` its label; a
            preclinical compound has ``max_phase`` ``None`` and
            ``development_phase`` ``"preclinical"``.
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

        max_phase = _as_float(record.get("max_phase"))
        availability = _as_int(record.get("availability_type"))
        chirality = _as_int(record.get("chirality"))

        return {
            "molecule_chembl_id": record.get("molecule_chembl_id"),
            "pref_name": record.get("pref_name"),
            "molecule_type": record.get("molecule_type"),
            "structure_type": record.get("structure_type"),
            "max_phase": max_phase,
            "development_phase": _phase_label(record.get("max_phase")),
            "is_approved": max_phase == 4.0,
            "first_approval": _as_int(record.get("first_approval")),
            "availability_type": availability,
            "availability": AVAILABILITY_LABELS.get(availability)
            if availability is not None
            else None,
            "oral": _flag(record.get("oral")),
            "parenteral": _flag(record.get("parenteral")),
            "topical": _flag(record.get("topical")),
            "therapeutic_flag": _flag(record.get("therapeutic_flag")),
            "dosed_ingredient": _flag(record.get("dosed_ingredient")),
            "black_box_warning": _flag(record.get("black_box_warning")),
            "withdrawn_flag": _flag(record.get("withdrawn_flag")),
            "first_in_class": _flag(record.get("first_in_class")),
            "prodrug": _flag(record.get("prodrug")),
            "orphan": _flag(record.get("orphan")),
            "natural_product": _flag(record.get("natural_product")),
            "chemical_probe": _flag(record.get("chemical_probe")),
            "inorganic_flag": _flag(record.get("inorganic_flag")),
            "polymer_flag": _flag(record.get("polymer_flag")),
            "veterinary": _flag(record.get("veterinary")),
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
        """
        parent = self._parent_id(identifier)
        if not parent:
            return {}
        drugs = self._collection(
            "drug", {"molecule_chembl_id": parent, "limit": 1}, "drugs"
        )
        if not drugs:
            return {}
        drug = drugs[0]
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
        self, identifier: str, limit: int = DEFAULT_INDICATIONS
    ) -> list[dict[str, Any]]:
        """Therapeutic indications with the phase reached for each.

        ChEMBL records one row per (drug, indication, reference) triple, so
        the same disease can appear more than once with different maximum
        phases; rows are collapsed here to one entry per indication keeping
        the highest phase.

        Parameters
        ----------
        identifier
            ChEMBL ID, SMILES, or compound name.
        limit
            Maximum indication rows to fetch (default
            :data:`DEFAULT_INDICATIONS`).

        Returns
        -------
        list of dict
            Sorted by ``max_phase_for_indication``, highest first.
        """
        parent = self._parent_id(identifier)
        if not parent:
            return []
        rows = self._collection(
            "drug_indication",
            {"parent_molecule_chembl_id": parent, "limit": limit},
            "drug_indications",
        )

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

    # ==================================================================
    # 3. Safety: black box warnings and withdrawals
    # ==================================================================

    def get_warnings(
        self, identifier: str, limit: int = DEFAULT_WARNINGS
    ) -> list[dict[str, Any]]:
        """Black box warnings and withdrawal records.

        Parameters
        ----------
        identifier
            ChEMBL ID, SMILES, or compound name.
        limit
            Maximum warning rows to fetch (default
            :data:`DEFAULT_WARNINGS`).

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
        rows = self._collection(
            "drug_warning",
            {"parent_molecule_chembl_id": parent, "limit": limit},
            "drug_warnings",
        )
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
            ChEMBL ID, SMILES, or compound name.

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

        classes: list[dict[str, Any]] = []
        for code in codes:
            self._sleep()
            data = _get_json(f"{CHEMBL}/atc_class/{code}.json")
            if not data:
                continue
            classes.append(
                {
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
            )
        return classes

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
            ChEMBL ID, SMILES, or compound name.
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
        registered against a salt form, imatinib included.

        Parameters
        ----------
        identifier
            ChEMBL ID, SMILES, or compound name.

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
        rows = self._collection(
            "mechanism", {"parent_molecule_chembl_id": parent}, "mechanisms"
        )
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
        limit: int = DEFAULT_ACTIVITIES,
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
            ChEMBL ID, SMILES, or compound name.
        limit
            Maximum activity rows to fetch (default
            :data:`DEFAULT_ACTIVITIES`).
        only_with_pchembl
            Restrict to rows with a pChEMBL value, i.e. comparable
            concentration-response measurements.  Set ``False`` to include
            qualitative and non-standard readouts.

        Returns
        -------
        list of dict
            Ordered by pChEMBL, most potent first, when
            ``only_with_pchembl`` is set.
        """
        parent = self._parent_id(identifier)
        if not parent:
            return []
        params: dict[str, Any] = {
            "parent_molecule_chembl_id": parent,
            "limit": limit,
        }
        if only_with_pchembl:
            params["pchembl_value__isnull"] = "false"
            params["order_by"] = "-pchembl_value"

        rows = self._collection("activity", params, "activities")
        return [
            {
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
        ``"tyrosine protein kinase"``.  Targets are fetched in one batched
        request and cached, and protein classes are shared across a
        screen's targets, so this stays cheap for a whole dataset.

        Parameters
        ----------
        target_chembl_ids
            Target ChEMBL IDs.
        include_protein_classes
            Resolve the ChEMBL protein family hierarchy for each target.
            Costs one extra batched request.

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
        for chunk in (missing[i : i + 20] for i in range(0, len(missing), 20)):
            targets = self._collection(
                "target",
                {"target_chembl_id__in": ",".join(chunk), "limit": len(chunk)},
                "targets",
            )
            for target in targets:
                detail = self._target_detail(target)
                # Always stored, because the return value is assembled from
                # this dict. ``cache`` only decides whether a second call
                # re-fetches, which the ``missing`` filter above handles.
                self._target_details[detail["target_chembl_id"]] = detail

        details = {t: self._target_details[t] for t in wanted if t in self._target_details}

        if include_protein_classes:
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
        return {
            "target_chembl_id": target.get("target_chembl_id"),
            "target_name": target.get("pref_name"),
            "target_type": target.get("target_type"),
            "organism": target.get("organism"),
            "tax_id": target.get("tax_id"),
            "uniprot_accessions": accessions,
            "uniprot_accession": accessions[0] if accessions else None,
            "gene_symbols": gene_symbols,
            "gene_symbol": gene_symbols[0] if gene_symbols else None,
            "n_components": len(components),
            "_protein_class_ids": class_ids,
        }

    def _resolve_protein_classes(
        self, class_ids: set[int]
    ) -> dict[int, dict[str, Any]]:
        missing = [
            cid
            for cid in class_ids
            if not (self.cache and cid in self._protein_classes)
        ]
        for chunk in (missing[i : i + 20] for i in range(0, len(missing), 20)):
            # The filter field is `protein_class_id`; `protein_classification_id`
            # is the name in the response payload and is silently ignored as a
            # filter, which would return the whole 900-row class table.
            rows = self._collection(
                "protein_classification",
                {
                    "protein_class_id__in": ",".join(str(c) for c in chunk),
                    "limit": len(chunk),
                },
                "protein_classifications",
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
        limit: int = DEFAULT_ACTIVITIES,
        max_targets: int = DEFAULT_TARGETS,
        include_protein_classes: bool = True,
    ) -> list[dict[str, Any]]:
        """Per-target potency summary with target identity resolved.

        Collapses the activity table into one row per target -- how many
        measurements, which assay types, best and median pChEMBL -- and
        joins on gene symbol, UniProt accession and protein family.

        Parameters
        ----------
        identifier
            ChEMBL ID, SMILES, or compound name.
        limit
            Activity rows to summarise (default
            :data:`DEFAULT_ACTIVITIES`).
        max_targets
            Resolve identity for at most this many targets, taking the most
            potent first (default :data:`DEFAULT_TARGETS`).
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
        max_targets: int = DEFAULT_TARGETS,
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
            potent first (default :data:`DEFAULT_TARGETS`).
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
            tissue readouts are excluded from that comparison, but only when
            target types were resolved -- a target whose type is unknown is
            kept rather than silently dropped.

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

        # A target whose type could not be resolved is kept: `max_targets`
        # caps identity resolution, so an unresolved type means "not looked
        # up", not "not a protein".
        proteins = [
            e
            for e in target_profile
            if e.get("target_type") in (None, "SINGLE PROTEIN", "PROTEIN COMPLEX")
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
        self, identifier: str, limit: int = DEFAULT_METABOLITES
    ) -> list[dict[str, Any]]:
        """Metabolic conversions, the enzymes involved, and the metabolites.

        Parameters
        ----------
        identifier
            ChEMBL ID, SMILES, or compound name.
        limit
            Maximum metabolism rows to fetch (default
            :data:`DEFAULT_METABOLITES`).

        Returns
        -------
        list of dict
            One entry per curated conversion.  A prodrug's active form and
            a compound's toxic metabolite both show up here.
        """
        parent = self._parent_id(identifier)
        if not parent:
            return []
        rows = self._collection(
            "metabolism",
            {"drug_chembl_id": parent, "limit": limit},
            "metabolisms",
        )
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
        self, identifier: str, limit: int = DEFAULT_FORMS
    ) -> list[dict[str, Any]]:
        """Parent compound and its salt forms.

        Useful for reconciling a screen that names the salt against one
        that names the free base.
        """
        parent = self._parent_id(identifier)
        if not parent:
            return []
        rows = self._collection(
            "molecule_form",
            {"parent_chembl_id": parent, "limit": limit},
            "molecule_forms",
        )
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
    ) -> list[dict[str, Any]]:
        """Structurally similar ChEMBL compounds by Tanimoto similarity.

        Parameters
        ----------
        identifier
            ChEMBL ID, SMILES, or compound name.
        similarity
            Minimum similarity percentage, 40--100 (default
            :data:`DEFAULT_ANALOG_SIMILARITY`).
        limit
            Maximum analogs to return (default :data:`DEFAULT_ANALOGS`).

        Returns
        -------
        list of dict
            Excludes the query compound itself.  Sorted by similarity,
            most similar first.
        """
        resolution = self.resolve(identifier)
        if not resolution.found:
            return []
        record = self.get_molecule_record(resolution.molecule_chembl_id)  # type: ignore[arg-type]
        smiles = ((record or {}).get("molecule_structures") or {}).get(
            "canonical_smiles"
        )
        if not smiles:
            return []

        self._sleep()
        # The query structure is a path segment, not a parameter, so it has
        # to be quoted -- SMILES routinely contain '/', '\' and '#'.
        quoted = requests.utils.quote(smiles, safe="")
        data = _get_json(
            f"{CHEMBL}/similarity/{quoted}/{int(similarity)}.json",
            params={"limit": limit, "format": "json"},
        )
        molecules = (data or {}).get("molecules", []) or []

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
            ChEMBL ID, SMILES, or compound name.
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
            activities = self.get_activities(identifier)
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

        Returns
        -------
        dict
            Mapping identifier to annotation dict.
        """
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
            "drug_n_activities": DEFAULT_ACTIVITIES,
            "drug_n_targets": DEFAULT_TARGETS,
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
    add("is_approved", lambda r: bool(dev(r, "is_approved")))
    add("first_approval", lambda r: dev(r, "first_approval"))
    add("availability", lambda r: dev(r, "availability") or "")
    add("molecule_type", lambda r: dev(r, "molecule_type") or "")
    add("oral", lambda r: dev(r, "oral"))
    add("parenteral", lambda r: dev(r, "parenteral"))
    add("topical", lambda r: dev(r, "topical"))
    add("is_prodrug", lambda r: dev(r, "prodrug"))
    add("first_in_class", lambda r: dev(r, "first_in_class"))
    add("is_natural_product", lambda r: dev(r, "natural_product"))
    add("is_orphan", lambda r: dev(r, "orphan"))
    add("is_withdrawn", lambda r: bool(dev(r, "withdrawn_flag")))
    add("has_black_box_warning", lambda r: bool(dev(r, "black_box_warning")))

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
