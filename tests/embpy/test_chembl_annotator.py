"""Tests for ``embpy.resources.molecule.chembl`` -- ChEMBLAnnotator.

Payloads mirror real ChEMBL_37 responses (aspirin CHEMBL25, imatinib
CHEMBL941 / mesylate CHEMBL1642, rofecoxib CHEMBL122), trimmed to the
fields the code reads.

Several tests here assert on the *request* rather than the response,
because two of ChEMBL's behaviours fail silently:

* Drug-level rows are registered against the parent molecule, so filtering
  on ``molecule_chembl_id`` returns an empty list for any drug curated
  against a salt form -- a wrong answer that looks like "no data".
* ChEMBL ignores filter parameters it does not recognise and returns the
  unfiltered collection, so a misspelled field name yields a plausible
  response covering the whole database.

Neither shows up as an error, so the field names are pinned by test, and
every filtered response is validated row by row against the requested keys.

The regression tests at the bottom use payloads recorded from the live API
on 2026-11-11 (``fixtures/chembl/``, see ``PROVENANCE.json`` there).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import requests
from anndata import AnnData

from embpy.resources.molecule import chembl as chembl_mod
from embpy.resources.molecule.chembl import (
    ALL_SOURCES,
    AVAILABILITY_LABELS,
    DEFAULT_INDICATIONS,
    BatchResult,
    ChEMBLAnnotator,
    ChEMBLFilterIgnoredError,
    ChEMBLResolution,
    ChEMBLServiceError,
    _as_float,
    _as_int,
    _flag,
    _phase_label,
    _protein_class_path,
    chembl_summary_columns,
)

# =====================================================================
# Fake API
# =====================================================================


class FakeAPI:
    """Dispatch ``_get_json`` calls to canned payloads by URL and params.

    Ordered ``side_effect`` lists are unusable here: a single ``annotate``
    call fans out across a dozen endpoints and any change to call order
    silently reshuffles which payload lands where.
    """

    def __init__(self, routes: list[tuple[object, object]]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, params=None, timeout=30):
        self.calls.append((url, dict(params or {})))
        for matcher, payload in self.routes:
            if self._matches(matcher, url, params or {}):
                # A callable payload answers from the request params, the way
                # a filtered endpoint does.
                return payload(dict(params or {})) if callable(payload) else payload
        return None

    @staticmethod
    def _matches(matcher, url: str, params: dict) -> bool:
        if callable(matcher):
            return matcher(url, params)
        return str(matcher) in url

    def requests_to(self, fragment: str) -> list[tuple[str, dict]]:
        return [(u, p) for u, p in self.calls if fragment in u]

    def params_for(self, fragment: str) -> dict:
        hits = self.requests_to(fragment)
        assert hits, f"no request was made to {fragment!r}"
        return hits[0][1]


def route(fragment: str | None = None, param: str | None = None):
    """Match a request by URL fragment and/or presence of a param key."""

    def matcher(url: str, params: dict) -> bool:
        if fragment is not None and fragment not in url:
            return False
        if param is not None and param not in params:
            return False
        return True

    return matcher


ASPIRIN_RECORD = {
    "molecule_chembl_id": "CHEMBL25",
    "pref_name": "ASPIRIN",
    "molecule_type": "Small molecule",
    "structure_type": "MOL",
    "max_phase": "4.0",
    "first_approval": 1950,
    "availability_type": 2,
    "chirality": 2,
    "oral": True,
    "parenteral": False,
    "topical": True,
    "therapeutic_flag": 1,
    "dosed_ingredient": True,
    "black_box_warning": "0",
    "withdrawn_flag": False,
    "first_in_class": 0,
    "prodrug": 0,
    "orphan": -1,
    "natural_product": 1,
    "chemical_probe": 0,
    "inorganic_flag": 0,
    "polymer_flag": False,
    "veterinary": 0,
    "usan_stem": None,
    "usan_stem_definition": None,
    "usan_year": None,
    "molecule_hierarchy": {
        "molecule_chembl_id": "CHEMBL25",
        "parent_chembl_id": "CHEMBL25",
        "active_chembl_id": "CHEMBL25",
    },
    "molecule_structures": {"canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O"},
    "atc_classifications": ["B01AC06", "N02BA01"],
    "molecule_synonyms": [
        {"molecule_synonym": "8-hour bayer", "syn_type": "TRADE_NAME"},
        {"molecule_synonym": "Aspirin", "syn_type": "INN"},
        {"molecule_synonym": "ASPIRIN", "syn_type": "TRADE_NAME"},
        {"molecule_synonym": "Acetylsalicylic acid", "syn_type": "USAN"},
    ],
    "cross_references": [
        {"xref_src": "DailyMed", "xref_id": "aspirin", "xref_name": "aspirin"},
    ],
}

# Imatinib is the case that motivated keying on the parent: the drug-level
# rows belong to the mesylate salt and name CHEMBL941 only as parent.
IMATINIB_RECORD = {
    "molecule_chembl_id": "CHEMBL941",
    "pref_name": "IMATINIB",
    "max_phase": "4.0",
    "first_approval": 2001,
    "availability_type": 1,
    "molecule_type": "Small molecule",
    "molecule_hierarchy": {
        "molecule_chembl_id": "CHEMBL941",
        "parent_chembl_id": "CHEMBL941",
    },
    "molecule_structures": {"canonical_smiles": "Cc1ccc(N)cc1"},
    "atc_classifications": ["L01EA01"],
    "molecule_synonyms": [],
}

MESYLATE_RECORD = {
    "molecule_chembl_id": "CHEMBL1642",
    "pref_name": "IMATINIB MESYLATE",
    "max_phase": "4.0",
    "molecule_hierarchy": {
        "molecule_chembl_id": "CHEMBL1642",
        "parent_chembl_id": "CHEMBL941",
    },
    "molecule_structures": {"canonical_smiles": "Cc1ccc(N)cc1.CS(O)(=O)=O"},
}


@pytest.fixture
def annotator():
    return ChEMBLAnnotator(rate_limit_delay=0)


@pytest.fixture
def no_cache_annotator():
    return ChEMBLAnnotator(rate_limit_delay=0, cache=False)


def with_api(routes):
    """Patch the module's ``_get_json`` with a :class:`FakeAPI`."""
    api = FakeAPI(routes)
    return patch("embpy.resources.molecule.chembl._get_json", api), api


#: The API's answer for a name nobody has: empty collections, not ``None``.
#: ``None`` from a collection is a missing response and raises.
NO_MATCH_ROUTES = [
    (route("molecule.json"), {"molecules": []}),
    ("molecule/search.json", {"molecules": []}),
]


def record_batch_route(*records):
    """Answer ``molecule.json?molecule_chembl_id__in=`` with the given records."""

    def matcher(url, params):
        return "molecule.json" in url and "molecule_chembl_id__in" in params

    def payload_for(params):
        wanted = set(params["molecule_chembl_id__in"].split(","))
        return {
            "molecules": [r for r in records if r["molecule_chembl_id"] in wanted],
            "page_meta": {"next": None},
        }

    return matcher, payload_for


class RoutingAPI(FakeAPI):
    """FakeAPI whose payloads may be callables of the request params."""

    def __call__(self, url, params=None, timeout=30):
        self.calls.append((url, dict(params or {})))
        for matcher, payload in self.routes:
            if self._matches(matcher, url, params or {}):
                return payload(dict(params or {})) if callable(payload) else payload
        return None


def with_routing_api(routes):
    api = RoutingAPI(routes)
    return patch("embpy.resources.molecule.chembl._get_json", api), api


# =====================================================================
# Field decoding helpers
# =====================================================================


class TestDecodeHelpers:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("4.0", 4.0), (4, 4.0), (None, None), ("", None), ("nope", None)],
    )
    def test_as_float(self, value, expected):
        assert _as_float(value) == expected

    def test_as_int_truncates_float_strings(self):
        # ChEMBL sends years and enums as "1950" / "2.0" interchangeably.
        assert _as_int("1950") == 1950
        assert _as_int("2.0") == 2
        assert _as_int(None) is None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1, True),
            ("1", True),
            (0, False),
            ("0", False),
            (True, True),
            (False, False),
            (None, None),
        ],
    )
    def test_flag(self, value, expected):
        assert _flag(value) is expected

    def test_negative_flag_is_not_false(self):
        # -1 means "preclinical compound, question does not apply". Folding
        # it to False would assert that a tool compound is *not* an orphan
        # drug, which is a different claim from "not applicable".
        assert _flag(-1) is None
        assert _flag("-1") is None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("4.0", "approved"),
            (3, "phase 3"),
            ("0.5", "early phase 1"),
            (-1, "clinical phase unknown"),
            (None, None),
        ],
    )
    def test_phase_label(self, value, expected):
        assert _phase_label(value) == expected

    def test_availability_labels_cover_the_schema(self):
        assert AVAILABILITY_LABELS[-2] == "withdrawn"
        assert AVAILABILITY_LABELS[2] == "over the counter"

    def test_protein_class_path_splits_on_double_space(self):
        # A class name can itself contain a single space ("protein kinase"),
        # so the levels are separated by two.
        assert _protein_class_path("enzyme  kinase  protein kinase  tk  abl") == [
            "enzyme",
            "kinase",
            "protein kinase",
            "tk",
            "abl",
        ]

    def test_protein_class_path_empty(self):
        assert _protein_class_path(None) == []
        assert _protein_class_path("") == []


# =====================================================================
# Identifier resolution
# =====================================================================


class TestClassify:
    def test_chembl_id(self, annotator):
        assert annotator._classify("CHEMBL25") == "chembl_id"
        assert annotator._classify("chembl941") == "chembl_id"

    def test_smiles(self, annotator):
        assert annotator._classify("CC(=O)Oc1ccccc1C(=O)O") == "smiles"

    def test_name(self, annotator):
        assert annotator._classify("aspirin") == "name"
        assert annotator._classify("imatinib mesylate") == "name"


class TestResolve:
    def test_resolves_chembl_id_directly(self, annotator):
        patcher, api = with_api([("molecule/CHEMBL25", ASPIRIN_RECORD)])
        with patcher:
            res = annotator.resolve("CHEMBL25")
        assert res.found
        assert res.molecule_chembl_id == "CHEMBL25"
        assert res.parent_chembl_id == "CHEMBL25"
        assert res.matched_by == "chembl_id"

    def test_resolves_by_preferred_name(self, annotator):
        patcher, api = with_api(
            [(route("molecule.json", "pref_name__iexact"), {"molecules": [ASPIRIN_RECORD]})]
        )
        with patcher:
            res = annotator.resolve("aspirin")
        assert res.molecule_chembl_id == "CHEMBL25"
        assert res.matched_by == "pref_name"

    def test_falls_back_to_synonym(self, annotator):
        patcher, api = with_api(
            [
                (route("molecule.json", "pref_name__iexact"), {"molecules": []}),
                (
                    route("molecule.json", "molecule_synonyms__molecule_synonym__iexact"),
                    {"molecules": [ASPIRIN_RECORD]},
                ),
            ]
        )
        with patcher:
            res = annotator.resolve("8-hour bayer")
        assert res.matched_by == "synonym"

    def test_falls_back_to_free_text_search(self, annotator):
        patcher, api = with_api(
            [
                (route("molecule.json"), {"molecules": []}),
                ("molecule/search.json", {"molecules": [ASPIRIN_RECORD]}),
            ]
        )
        with patcher:
            res = annotator.resolve("asprin")
        assert res.matched_by == "search"
        assert res.molecule_chembl_id == "CHEMBL25"

    def test_smiles_resolves_by_exact_inchikey(self, annotator):
        pytest.importorskip("rdkit")
        record = {
            **ASPIRIN_RECORD,
            "molecule_structures": {
                "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "standard_inchi_key": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
            },
        }
        patcher, api = with_api(
            [
                (
                    route("molecule.json", "molecule_structures__standard_inchi_key__in"),
                    {"molecules": [record], "page_meta": {"next": None}},
                )
            ]
        )
        with patcher:
            res = annotator.resolve("CC(=O)Oc1ccccc1C(=O)O")
        assert res.status == "matched"
        assert res.matched_by == "standard_inchi_key"
        assert res.inchikey == res.matched_inchikey == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
        params = api.params_for("molecule.json")
        assert params["molecule_structures__standard_inchi_key__in"] == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
        assert len(api.calls) == 1

    def test_smiles_miss_never_falls_back_to_names_or_search(self, annotator):
        """A SMILES cannot match a name; those three requests were pure cost."""
        pytest.importorskip("rdkit")
        patcher, api = with_api(NO_MATCH_ROUTES)
        with patcher:
            res = annotator.resolve("CC(=O)Oc1ccccc1C(=O)O")
        assert res.status == "no_match"
        assert not res.found
        assert len(api.calls) == 1
        params = api.calls[0][1]
        assert "molecule_structures__standard_inchi_key__in" in params
        assert not api.requests_to("search.json")
        assert not any("flexmatch" in k for _, p in api.calls for k in p)

    def test_inchikey_identifier_is_exact(self, annotator):
        patcher, api = with_api(NO_MATCH_ROUTES)
        with patcher:
            res = annotator.resolve("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
        assert res.status == "no_match"
        assert api.params_for("molecule.json") == {
            "molecule_structures__standard_inchi_key__in": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
            "limit": 1000,
            "format": "json",
        }

    def test_salt_resolves_to_its_parent(self, annotator):
        patcher, api = with_api([("molecule/CHEMBL1642", MESYLATE_RECORD)])
        with patcher:
            res = annotator.resolve("CHEMBL1642")
        assert res.molecule_chembl_id == "CHEMBL1642"
        assert res.parent_chembl_id == "CHEMBL941"

    def test_not_found(self, annotator):
        patcher, api = with_api(NO_MATCH_ROUTES)
        with patcher:
            res = annotator.resolve("not-a-real-compound")
        assert not res.found
        assert res.status == "no_match"
        assert res.molecule_chembl_id is None

    def test_missing_response_is_not_a_miss(self, annotator):
        """``None`` from a collection query is an outage, never "not in ChEMBL"."""
        patcher, api = with_api([])
        with patcher, pytest.raises(ChEMBLServiceError):
            annotator.resolve("not-a-real-compound")

    def test_resolution_is_cached(self, annotator):
        patcher, api = with_api(
            [(route("molecule.json", "pref_name__iexact"), {"molecules": [ASPIRIN_RECORD]})]
        )
        with patcher:
            annotator.resolve("aspirin")
            annotator.resolve("aspirin")
        assert len(api.requests_to("molecule.json")) == 1

    def test_cache_can_be_disabled(self, no_cache_annotator):
        patcher, api = with_api(
            [(route("molecule.json", "pref_name__iexact"), {"molecules": [ASPIRIN_RECORD]})]
        )
        with patcher:
            no_cache_annotator.resolve("aspirin")
            no_cache_annotator.resolve("aspirin")
        assert len(api.requests_to("molecule.json")) == 2

    def test_found_property(self):
        assert not ChEMBLResolution(identifier="x").found
        assert ChEMBLResolution(identifier="x", molecule_chembl_id="CHEMBL1").found


# =====================================================================
# Development status
# =====================================================================


class TestDevelopmentStatus:
    def _status(self, annotator):
        patcher, api = with_api([("molecule/CHEMBL25", ASPIRIN_RECORD)])
        with patcher:
            return annotator.get_development_status("CHEMBL25")

    def test_decodes_phase_and_approval(self, annotator):
        status = self._status(annotator)
        assert status["max_phase"] == 4.0
        assert status["development_phase"] == "approved"
        assert status["is_approved"] is True
        assert status["first_approval"] == 1950

    def test_decodes_enums_to_labels(self, annotator):
        status = self._status(annotator)
        assert status["availability"] == "over the counter"
        assert status["chirality_label"] == "achiral"

    def test_decodes_routes_and_flags(self, annotator):
        status = self._status(annotator)
        assert status["oral"] is True
        assert status["parenteral"] is None
        assert status["natural_product"] is True
        assert status["withdrawn_flag"] is None
        assert status["black_box_warning"] is None
        assert status["raw_flags"]["black_box_warning"] == ASPIRIN_RECORD["black_box_warning"]

    def test_not_applicable_flag_is_none(self, annotator):
        # aspirin's `orphan` is -1 in ChEMBL
        assert self._status(annotator)["orphan"] is None

    def test_unknown_compound_returns_empty(self, annotator):
        patcher, api = with_api(NO_MATCH_ROUTES)
        with patcher:
            assert annotator.get_development_status("nope") == {}

    def test_status_uses_the_parent_record(self, annotator):
        """A salt's flags come from the drug, i.e. the parent."""
        patcher, api = with_api(
            [
                ("molecule/CHEMBL1642", MESYLATE_RECORD),
                ("molecule/CHEMBL941", IMATINIB_RECORD),
            ]
        )
        with patcher:
            status = annotator.get_development_status("CHEMBL1642")
        assert status["molecule_chembl_id"] == "CHEMBL941"
        assert status["first_approval"] == 2001


class TestApprovalRecord:
    def test_reads_applicants_and_atc(self, annotator):
        patcher, api = with_api(
            [
                ("molecule/CHEMBL25", ASPIRIN_RECORD),
                (
                    "drug.json",
                    {
                        "drugs": [
                            {
                                "molecule_chembl_id": "CHEMBL25",
                                "first_approval": 1950,
                                "drug_type": 1,
                                "applicants": ["Lannett Co Inc", "Barr Laboratories Inc"],
                                "atc_classification": [
                                    {"code": "B01AC06", "description": "..."},
                                ],
                                "helm_notation": None,
                                "biotherapeutic": None,
                            }
                        ]
                    },
                ),
            ]
        )
        with patcher:
            record = annotator.get_approval_record("CHEMBL25")
        assert record["applicants"] == ["Lannett Co Inc", "Barr Laboratories Inc"]
        assert record["atc_codes"] == ["B01AC06"]

    def test_non_drug_returns_empty(self, annotator):
        patcher, api = with_api(
            [("molecule/CHEMBL25", ASPIRIN_RECORD), ("drug.json", {"drugs": []})]
        )
        with patcher:
            assert annotator.get_approval_record("CHEMBL25") == {}


# =====================================================================
# Indications
# =====================================================================


INDICATION_ROWS = {
    "drug_indications": [
        {
            "efo_id": "EFO:0000275",
            "efo_term": "atrial fibrillation",
            "mesh_id": "D001281",
            "mesh_heading": "Atrial Fibrillation",
            "max_phase_for_ind": "3.0",
            "indication_refs": [{"ref_type": "ClinicalTrials"}],
            "parent_molecule_chembl_id": "CHEMBL25",
        },
        # Same indication, higher phase, from another reference.
        {
            "efo_id": "EFO:0000275",
            "efo_term": "atrial fibrillation",
            "mesh_id": "D001281",
            "mesh_heading": "Atrial Fibrillation",
            "max_phase_for_ind": "4.0",
            "indication_refs": [{"ref_type": "FDA"}, {"ref_type": "ATC"}],
            "parent_molecule_chembl_id": "CHEMBL25",
        },
        {
            "efo_id": "EFO:0003843",
            "efo_term": "pain",
            "mesh_id": "D010146",
            "mesh_heading": "Pain",
            "max_phase_for_ind": "2.0",
            "indication_refs": [],
            "parent_molecule_chembl_id": "CHEMBL25",
        },
    ]
}


class TestIndications:
    def _indications(self, annotator):
        patcher, api = with_api(
            [("molecule/CHEMBL25", ASPIRIN_RECORD), ("drug_indication.json", INDICATION_ROWS)]
        )
        with patcher:
            return annotator.get_indications("CHEMBL25")

    def test_keyed_on_parent_molecule(self, annotator):
        patcher, api = with_api(
            [("molecule/CHEMBL25", ASPIRIN_RECORD), ("drug_indication.json", INDICATION_ROWS)]
        )
        with patcher:
            annotator.get_indications("CHEMBL25")
        params = api.params_for("drug_indication.json")
        assert "parent_molecule_chembl_id" in params
        assert "molecule_chembl_id" not in params

    def test_duplicate_indications_are_merged(self, annotator):
        indications = self._indications(annotator)
        assert len(indications) == 2

    def test_merge_keeps_the_highest_phase(self, annotator):
        af = next(i for i in self._indications(annotator) if i["efo_id"] == "EFO:0000275")
        assert af["max_phase_for_indication"] == 4.0

    def test_merge_sums_references(self, annotator):
        af = next(i for i in self._indications(annotator) if i["efo_id"] == "EFO:0000275")
        assert af["n_references"] == 3

    def test_sorted_by_phase_descending(self, annotator):
        indications = self._indications(annotator)
        phases = [i["max_phase_for_indication"] for i in indications]
        assert phases == sorted(phases, reverse=True)

    def test_indication_falls_back_to_efo_term(self, annotator):
        patcher, api = with_api(
            [
                ("molecule/CHEMBL25", ASPIRIN_RECORD),
                (
                    "drug_indication.json",
                    {
                        "drug_indications": [
                            {
                                "efo_id": "EFO:1",
                                "efo_term": "some disease",
                                "mesh_heading": None,
                                "max_phase_for_ind": "4.0",
                                "parent_molecule_chembl_id": "CHEMBL25",
                            }
                        ]
                    },
                ),
            ]
        )
        with patcher:
            indications = annotator.get_indications("CHEMBL25")
        assert indications[0]["indication"] == "some disease"

    def test_unknown_compound_returns_empty(self, annotator):
        patcher, api = with_api(NO_MATCH_ROUTES)
        with patcher:
            assert annotator.get_indications("nope") == []

    def test_foreign_rows_are_rejected(self, annotator):
        """An ignored filter returns other drugs' rows; none may be kept."""
        rows = {"drug_indications": [{**INDICATION_ROWS["drug_indications"][0],
                                      "parent_molecule_chembl_id": "CHEMBL1"}]}
        patcher, api = with_api(
            [("molecule/CHEMBL25", ASPIRIN_RECORD), ("drug_indication.json", rows)]
        )
        with patcher, pytest.raises(ChEMBLFilterIgnoredError):
            annotator.get_indications("CHEMBL25")


# =====================================================================
# Safety warnings
# =====================================================================


WARNING_ROWS = {
    "drug_warnings": [
        {
            "warning_type": "Black Box Warning",
            "warning_class": None,
            "warning_description": None,
            "warning_country": "United States",
            "warning_year": None,
            "efo_id": None,
            "efo_term": None,
            "warning_refs": [],
        },
        {
            "warning_type": "Withdrawn",
            "warning_class": "cardiotoxicity",
            "warning_description": "Risk for heart attack and stroke",
            "warning_country": "Worldwide",
            "warning_year": 2004,
            "efo_id": "EFO:0000712",
            "efo_term": "stroke",
            "warning_refs": [{"ref_type": "WHO"}, {"ref_type": "PubMed"}],
        },
    ]
}


ROFECOXIB_RECORD = {
    "molecule_chembl_id": "CHEMBL122",
    "pref_name": "ROFECOXIB",
    "max_phase": "4.0",
    "withdrawn_flag": True,
    "molecule_hierarchy": {"molecule_chembl_id": "CHEMBL122", "parent_chembl_id": "CHEMBL122"},
}
for _row in WARNING_ROWS["drug_warnings"]:
    _row["parent_molecule_chembl_id"] = "CHEMBL122"


class TestWarnings:
    def test_keyed_on_parent_molecule(self, annotator):
        patcher, api = with_api(
            [("molecule/CHEMBL122", ROFECOXIB_RECORD), ("drug_warning.json", WARNING_ROWS)]
        )
        with patcher:
            annotator.get_warnings("CHEMBL122")
        assert "parent_molecule_chembl_id" in api.params_for("drug_warning.json")

    def test_parses_withdrawal(self, annotator):
        patcher, api = with_api(
            [("molecule/CHEMBL122", ROFECOXIB_RECORD), ("drug_warning.json", WARNING_ROWS)]
        )
        with patcher:
            warnings = annotator.get_warnings("CHEMBL122")
        withdrawn = next(w for w in warnings if w["warning_type"] == "Withdrawn")
        assert withdrawn["warning_class"] == "cardiotoxicity"
        assert withdrawn["warning_year"] == 2004
        assert withdrawn["n_references"] == 2


# =====================================================================
# ATC classification
# =====================================================================


class TestATCClasses:
    def test_resolves_every_code_on_the_record(self, annotator):
        patcher, api = with_api(
            [
                ("molecule/CHEMBL25", ASPIRIN_RECORD),
                (
                    "atc_class/B01AC06",
                    {
                        "level1": "B",
                        "level1_description": "BLOOD AND BLOOD FORMING ORGANS",
                        "level2": "B01",
                        "level5": "B01AC06",
                        "who_name": "acetylsalicylic acid",
                    },
                ),
                (
                    "atc_class/N02BA01",
                    {
                        "level1": "N",
                        "level1_description": "NERVOUS SYSTEM",
                        "level5": "N02BA01",
                        "who_name": "acetylsalicylic acid",
                    },
                ),
            ]
        )
        with patcher:
            classes = annotator.get_atc_classes("CHEMBL25")
        assert [c["code"] for c in classes] == ["B01AC06", "N02BA01"]
        assert classes[0]["level1_description"] == "BLOOD AND BLOOD FORMING ORGANS"

    def test_no_atc_codes(self, annotator):
        record = {**ASPIRIN_RECORD, "atc_classifications": []}
        patcher, api = with_api([("molecule/CHEMBL25", record)])
        with patcher:
            assert annotator.get_atc_classes("CHEMBL25") == []


# =====================================================================
# Synonyms
# =====================================================================


class TestSynonyms:
    def test_inn_ranked_before_trade_names(self, annotator):
        patcher, api = with_api([("molecule/CHEMBL25", ASPIRIN_RECORD)])
        with patcher:
            synonyms = annotator.get_synonyms("CHEMBL25")
        assert synonyms[0]["type"] == "INN"

    def test_case_insensitive_dedup(self, annotator):
        # "Aspirin" (INN) and "ASPIRIN" (trade name) are one name.
        patcher, api = with_api([("molecule/CHEMBL25", ASPIRIN_RECORD)])
        with patcher:
            synonyms = annotator.get_synonyms("CHEMBL25")
        lowered = [s["name"].lower() for s in synonyms]
        assert len(lowered) == len(set(lowered))

    def test_limit_is_respected(self, annotator):
        patcher, api = with_api([("molecule/CHEMBL25", ASPIRIN_RECORD)])
        with patcher:
            assert len(annotator.get_synonyms("CHEMBL25", limit=2)) == 2


# =====================================================================
# Mechanism of action
# =====================================================================


MECHANISM_ROWS = {
    "mechanisms": [
        {
            "mechanism_of_action": "Tyrosine-protein kinase ABL inhibitor",
            "action_type": "INHIBITOR",
            "target_chembl_id": "CHEMBL1862",
            "max_phase": 4,
            "direct_interaction": 1,
            "molecular_mechanism": 1,
            "disease_efficacy": 1,
            "mechanism_comment": None,
            "selectivity_comment": None,
            "binding_site_comment": None,
            "molecule_chembl_id": "CHEMBL1642",
            "parent_molecule_chembl_id": "CHEMBL941",
            "mechanism_refs": [
                {"ref_type": "DailyMed", "ref_id": "x", "ref_url": "http://x"},
            ],
        }
    ]
}


class TestMechanisms:
    def test_keyed_on_parent_not_molecule(self, annotator):
        """The regression this module exists for.

        ``mechanism?molecule_chembl_id=CHEMBL941`` returns zero rows for
        imatinib because the mechanism belongs to the mesylate. Querying by
        ``molecule_chembl_id`` reports "no mechanism of action" for a drug
        that has four.
        """
        patcher, api = with_api(
            [("molecule/CHEMBL941", IMATINIB_RECORD), ("mechanism.json", MECHANISM_ROWS)]
        )
        with patcher:
            mechanisms = annotator.get_mechanisms("CHEMBL941")
        params = api.params_for("mechanism.json")
        assert params.get("parent_molecule_chembl_id") == "CHEMBL941"
        assert "molecule_chembl_id" not in params
        assert len(mechanisms) == 1

    def test_exposes_curation_flags(self, annotator):
        patcher, api = with_api(
            [("molecule/CHEMBL941", IMATINIB_RECORD), ("mechanism.json", MECHANISM_ROWS)]
        )
        with patcher:
            mech = annotator.get_mechanisms("CHEMBL941")[0]
        assert mech["direct_interaction"] is True
        assert mech["disease_efficacy"] is True
        assert mech["action_type"] == "INHIBITOR"
        assert mech["references"][0]["type"] == "DailyMed"

    def test_a_salt_resolves_to_the_parents_mechanisms(self, annotator):
        patcher, api = with_api(
            [
                ("molecule/CHEMBL1642", MESYLATE_RECORD),
                ("molecule/CHEMBL941", IMATINIB_RECORD),
                ("mechanism.json", MECHANISM_ROWS),
            ]
        )
        with patcher:
            mechanisms = annotator.get_mechanisms("CHEMBL1642")
        assert api.params_for("mechanism.json")["parent_molecule_chembl_id"] == "CHEMBL941"
        assert len(mechanisms) == 1


# =====================================================================
# Activities and target profile
# =====================================================================


ACTIVITY_ROWS = {
    "activities": [
        {
            "parent_molecule_chembl_id": "CHEMBL941",
            "target_chembl_id": "CHEMBL1862",
            "target_pref_name": "Tyrosine-protein kinase ABL1",
            "target_organism": "Homo sapiens",
            "standard_type": "IC50",
            "standard_value": "10",
            "standard_units": "nM",
            "standard_relation": "=",
            "pchembl_value": "8.0",
            "assay_chembl_id": "CHEMBL111",
            "assay_type": "B",
            "assay_description": "Inhibition of ABL1",
            "bao_label": "single protein format",
            "document_chembl_id": "CHEMBL_DOC1",
            "document_year": 2005,
            "document_journal": "J Med Chem",
            "data_validity_comment": None,
            "potential_duplicate": 0,
            "activity_comment": None,
            "action_type": None,
            "ligand_efficiency": {"bei": "20"},
        },
        {
            "parent_molecule_chembl_id": "CHEMBL941",
            "target_chembl_id": "CHEMBL1862",
            "target_pref_name": "Tyrosine-protein kinase ABL1",
            "target_organism": "Homo sapiens",
            "standard_type": "Ki",
            "standard_value": "30",
            "standard_units": "nM",
            "pchembl_value": "7.0",
            "assay_type": "B",
        },
        {
            "parent_molecule_chembl_id": "CHEMBL941",
            "target_chembl_id": "CHEMBL2331053",
            "target_pref_name": "K562",
            "target_organism": "Homo sapiens",
            "standard_type": "IC50",
            "standard_value": "5",
            "standard_units": "nM",
            "pchembl_value": "9.0",
            "assay_type": "F",
        },
        {
            "parent_molecule_chembl_id": "CHEMBL941",
            "target_chembl_id": "CHEMBL1913",
            "target_pref_name": "PDGFR beta",
            "target_organism": "Homo sapiens",
            "standard_type": "IC50",
            "pchembl_value": "6.0",
            "assay_type": "B",
        },
    ]
}

TARGET_ROWS = {
    "targets": [
        {
            "target_chembl_id": "CHEMBL1862",
            "pref_name": "Tyrosine-protein kinase ABL1",
            "target_type": "SINGLE PROTEIN",
            "organism": "Homo sapiens",
            "tax_id": 9606,
            "target_components": [
                {
                    "accession": "P00519",
                    "component_id": 173,
                    "target_component_synonyms": [
                        {"component_synonym": "ABL1", "syn_type": "GENE_SYMBOL"},
                        {"component_synonym": "c-ABL", "syn_type": "UNIPROT"},
                    ],
                    "protein_classifications": [{"protein_classification_id": 130}],
                }
            ],
        },
        {
            "target_chembl_id": "CHEMBL2331053",
            "pref_name": "K562",
            "target_type": "CELL-LINE",
            "organism": "Homo sapiens",
            "target_components": [],
        },
        {
            "target_chembl_id": "CHEMBL1913",
            "pref_name": "PDGFR beta",
            "target_type": "SINGLE PROTEIN",
            "organism": "Homo sapiens",
            "target_components": [
                {
                    "accession": "P09619",
                    "component_id": 200,
                    "target_component_synonyms": [
                        {"component_synonym": "PDGFRB", "syn_type": "GENE_SYMBOL"},
                    ],
                    "protein_classifications": [{"protein_classification_id": 130}],
                }
            ],
        },
    ]
}

PROTEIN_CLASS_ROWS = {
    "protein_classifications": [
        {
            "protein_class_id": 130,
            "protein_class_desc": "enzyme  kinase  protein kinase  tk  abl",
            "class_level": 5,
            "pref_name": "Tyrosine protein kinase Abl family",
        }
    ]
}


def _targets_for(params):
    """``target.json?target_chembl_id__in=`` answered like the API: filtered."""
    wanted = set(params["target_chembl_id__in"].split(","))
    return {"targets": [t for t in TARGET_ROWS["targets"] if t["target_chembl_id"] in wanted]}


DRUG_NONE = ("drug.json", {"drugs": []})


def _activity_routes():
    return [
        ("molecule/CHEMBL941", IMATINIB_RECORD),
        ("activity.json", ACTIVITY_ROWS),
        ("target.json", _targets_for),
        ("protein_classification.json", PROTEIN_CLASS_ROWS),
    ]


class TestActivities:
    def test_keyed_on_parent_and_ordered_by_potency(self, annotator):
        patcher, api = with_api(_activity_routes())
        with patcher:
            annotator.get_activities("CHEMBL941")
        params = api.params_for("activity.json")
        assert params.get("parent_molecule_chembl_id") == "CHEMBL941"
        assert params.get("pchembl_value__isnull") == "false"
        assert params.get("order_by") == "-pchembl_value"

    def test_only_with_pchembl_can_be_disabled(self, annotator):
        patcher, api = with_api(_activity_routes())
        with patcher:
            annotator.get_activities("CHEMBL941", only_with_pchembl=False)
        params = api.params_for("activity.json")
        assert "pchembl_value__isnull" not in params

    def test_numeric_fields_are_coerced(self, annotator):
        patcher, api = with_api(_activity_routes())
        with patcher:
            acts = annotator.get_activities("CHEMBL941")
        assert acts[0]["activity_value"] == 10.0
        assert acts[0]["pchembl_value"] == 8.0
        assert acts[0]["document_year"] == 2005

    def test_exposes_provenance_fields(self, annotator):
        patcher, api = with_api(_activity_routes())
        with patcher:
            act = annotator.get_activities("CHEMBL941")[0]
        assert act["assay_description"] == "Inhibition of ABL1"
        assert act["bao_label"] == "single protein format"
        assert act["document_journal"] == "J Med Chem"
        assert act["potential_duplicate"] is False


class TestTargetDetails:
    def test_batches_targets_into_one_request(self, annotator):
        patcher, api = with_api(_activity_routes())
        with patcher:
            annotator.get_target_details(
                ["CHEMBL1862", "CHEMBL2331053", "CHEMBL1913"],
                include_protein_classes=False,
            )
        assert len(api.requests_to("target.json")) == 1
        assert api.params_for("target.json")["target_chembl_id__in"] == (
            "CHEMBL1862,CHEMBL2331053,CHEMBL1913"
        )

    def test_extracts_gene_symbol_and_accession(self, annotator):
        patcher, api = with_api(_activity_routes())
        with patcher:
            details = annotator.get_target_details(["CHEMBL1862"])
        abl = details["CHEMBL1862"]
        assert abl["gene_symbol"] == "ABL1"
        assert abl["uniprot_accession"] == "P00519"
        assert abl["target_type"] == "SINGLE PROTEIN"

    def test_resolves_protein_class_hierarchy(self, annotator):
        patcher, api = with_api(_activity_routes())
        with patcher:
            details = annotator.get_target_details(["CHEMBL1862"])
        abl = details["CHEMBL1862"]
        assert abl["target_class"] == "enzyme"
        assert abl["target_subclass"] == "abl"
        assert abl["protein_class_path"][2] == "protein kinase"

    def test_protein_class_filter_uses_the_schema_field_name(self, annotator):
        """``protein_class_id`` filters; ``protein_classification_id`` does not.

        The response payload names the column ``protein_class_id`` while the
        nested reference on a target component is
        ``protein_classification_id``. Passing the latter as a filter is
        ignored by ChEMBL, which then returns the entire ~900-row class
        table and quietly mislabels every target.
        """
        patcher, api = with_api(_activity_routes())
        with patcher:
            annotator.get_target_details(["CHEMBL1862"])
        params = api.params_for("protein_classification.json")
        assert "protein_class_id__in" in params
        assert "protein_classification_id__in" not in params

    def test_internal_keys_are_not_leaked(self, annotator):
        patcher, api = with_api(_activity_routes())
        with patcher:
            details = annotator.get_target_details(["CHEMBL1862"])
        assert not [k for k in details["CHEMBL1862"] if k.startswith("_")]

    def test_empty_input(self, annotator):
        patcher, api = with_api([])
        with patcher:
            assert annotator.get_target_details([]) == {}

    def test_target_details_are_cached_across_calls(self, annotator):
        patcher, api = with_api(_activity_routes())
        with patcher:
            annotator.get_target_details(["CHEMBL1862"], include_protein_classes=False)
            annotator.get_target_details(["CHEMBL1862"], include_protein_classes=False)
        assert len(api.requests_to("target.json")) == 1


class TestTargetProfile:
    def _profile(self, annotator):
        patcher, api = with_api(_activity_routes())
        with patcher:
            return annotator.get_target_profile("CHEMBL941")

    def test_one_row_per_target(self, annotator):
        assert len(self._profile(annotator)) == 3

    def test_aggregates_measurements(self, annotator):
        abl = next(
            e for e in self._profile(annotator) if e["target_chembl_id"] == "CHEMBL1862"
        )
        assert abl["n_measurements"] == 2
        assert abl["best_pchembl"] == 8.0
        assert abl["median_pchembl"] == 7.5
        assert abl["activity_types"] == ["IC50", "Ki"]

    def test_sorted_by_potency(self, annotator):
        best = [e["best_pchembl"] for e in self._profile(annotator)]
        assert best == sorted(best, reverse=True)

    def test_joins_target_identity(self, annotator):
        abl = next(
            e for e in self._profile(annotator) if e["target_chembl_id"] == "CHEMBL1862"
        )
        assert abl["gene_symbol"] == "ABL1"
        assert abl["target_class"] == "enzyme"

    def test_build_profile_needs_no_network(self, annotator):
        patcher, api = with_api([])
        with patcher:
            profile = annotator.build_target_profile(
                [
                    {
                        "target_chembl_id": "T1",
                        "target_pref_name": "t",
                        "target_organism": "Homo sapiens",
                        "activity_type": "IC50",
                        "pchembl_value": 7.0,
                    }
                ],
                max_targets=0,
                include_protein_classes=False,
            )
        assert profile[0]["best_pchembl"] == 7.0
        assert not api.calls

    def test_empty_activities(self, annotator):
        patcher, api = with_api(
            [("molecule/CHEMBL941", IMATINIB_RECORD), ("activity.json", {"activities": []})]
        )
        with patcher:
            assert annotator.get_target_profile("CHEMBL941") == []

    def test_max_targets_caps_identity_resolution(self, annotator):
        patcher, api = with_api(_activity_routes())
        with patcher:
            annotator.get_target_profile("CHEMBL941", max_targets=1)
        assert api.params_for("target.json")["target_chembl_id__in"] == "CHEMBL2331053"


class TestSelectivity:
    def _summary(self, annotator):
        patcher, api = with_api(_activity_routes())
        with patcher:
            return annotator.summarize_selectivity(
                annotator.get_target_profile("CHEMBL941")
            )

    def test_primary_target_is_a_protein_not_a_cell_line(self, annotator):
        """K562 is the most potent row but it is a cell line, not a target."""
        summary = self._summary(annotator)
        assert summary["primary_target_gene"] == "ABL1"
        assert summary["best_pchembl"] == 8.0

    def test_selectivity_window_between_protein_targets(self, annotator):
        # ABL1 8.0 vs PDGFRB 6.0
        assert self._summary(annotator)["selectivity_window"] == 2.0

    def test_counts(self, annotator):
        summary = self._summary(annotator)
        assert summary["n_targets"] == 3
        assert summary["n_protein_targets"] == 2

    def test_empty_profile(self, annotator):
        summary = annotator.summarize_selectivity([])
        assert summary["n_targets"] == 0
        assert summary["primary_target"] is None
        assert summary["selectivity_window"] is None

    def test_single_target_has_no_window(self, annotator):
        summary = annotator.summarize_selectivity(
            [
                {
                    "target_chembl_id": "T1",
                    "target_name": "t",
                    "target_type": "SINGLE PROTEIN",
                    "best_pchembl": 8.0,
                }
            ]
        )
        assert summary["selectivity_window"] is None
        assert summary["n_protein_targets"] == 1


# =====================================================================
# Metabolism, forms, analogs, xrefs
# =====================================================================


class TestMetabolism:
    def test_parses_conversion(self, annotator):
        patcher, api = with_api(
            [
                ("molecule/CHEMBL25", ASPIRIN_RECORD),
                (
                    "metabolism.json",
                    {
                        "metabolisms": [
                            {
                                "drug_chembl_id": "CHEMBL25",
                                "substrate_name": "ASPIRIN",
                                "substrate_chembl_id": "CHEMBL25",
                                "metabolite_name": "SALICYLIC ACID",
                                "metabolite_chembl_id": "CHEMBL424",
                                "enzyme_name": "Esterases",
                                "met_conversion": "Ester hydrolysis",
                                "met_comment": "Active metabolite",
                                "organism": None,
                                "pathway_id": 1,
                                "target_chembl_id": None,
                                "metabolism_refs": [{"ref_type": "DOI"}],
                            }
                        ]
                    },
                ),
            ]
        )
        with patcher:
            metabolism = annotator.get_metabolism("CHEMBL25")
        assert metabolism[0]["metabolite_name"] == "SALICYLIC ACID"
        assert metabolism[0]["enzyme_name"] == "Esterases"
        assert metabolism[0]["n_references"] == 1

    def test_uses_drug_chembl_id_filter(self, annotator):
        patcher, api = with_api(
            [("molecule/CHEMBL25", ASPIRIN_RECORD), ("metabolism.json", {"metabolisms": []})]
        )
        with patcher:
            annotator.get_metabolism("CHEMBL25")
        # The metabolism table keys on `drug_chembl_id`, not `molecule_*`.
        assert "drug_chembl_id" in api.params_for("metabolism.json")


class TestMoleculeForms:
    def test_lists_parent_and_salts(self, annotator):
        patcher, api = with_api(
            [
                ("molecule/CHEMBL25", ASPIRIN_RECORD),
                (
                    "molecule_form.json",
                    {
                        "molecule_forms": [
                            {
                                "molecule_chembl_id": "CHEMBL25",
                                "parent_chembl_id": "CHEMBL25",
                                "is_parent": True,
                            },
                            {
                                "molecule_chembl_id": "CHEMBL1697753",
                                "parent_chembl_id": "CHEMBL25",
                                "is_parent": False,
                            },
                        ]
                    },
                ),
            ]
        )
        with patcher:
            forms = annotator.get_molecule_forms("CHEMBL25")
        assert len(forms) == 2
        assert forms[0]["is_parent"] is True


class TestAnalogs:
    def _routes(self):
        return [
            ("molecule/CHEMBL25", ASPIRIN_RECORD),
            (
                "similarity/",
                {
                    "molecules": [
                        {
                            "molecule_chembl_id": "CHEMBL25",
                            "pref_name": "ASPIRIN",
                            "similarity": "100",
                            "max_phase": "4.0",
                            "molecule_structures": {"canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O"},
                        },
                        {
                            "molecule_chembl_id": "CHEMBL1697753",
                            "pref_name": "ASPIRIN DL-LYSINE",
                            "similarity": "85",
                            "max_phase": None,
                            "molecule_structures": {"canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O.N"},
                        },
                    ]
                },
            ),
        ]

    def test_excludes_the_query_compound(self, annotator):
        patcher, api = with_api(self._routes())
        with patcher:
            analogs = annotator.get_analogs("CHEMBL25")
        assert [a["molecule_chembl_id"] for a in analogs] == ["CHEMBL1697753"]

    def test_labels_phase_of_analogs(self, annotator):
        patcher, api = with_api(self._routes())
        with patcher:
            analogs = annotator.get_analogs("CHEMBL25")
        assert analogs[0]["development_phase"] is None

    def test_smiles_is_url_quoted(self, annotator):
        """A raw SMILES in a path segment breaks on ``/``, ``\\`` and ``#``.

        The structure is a path segment rather than a query parameter, so an
        unescaped ``/`` would split the path and an unescaped ``#`` would
        truncate the URL at a fragment.
        """
        record = {
            **ASPIRIN_RECORD,
            "molecule_structures": {"canonical_smiles": r"C/C=C\C#N"},
        }
        patcher, api = with_api(
            [("molecule/CHEMBL25", record), ("similarity/", {"molecules": []})]
        )
        with patcher:
            annotator.get_analogs("CHEMBL25")
        url = api.requests_to("similarity/")[0][0]
        # ".../similarity/<quoted smiles>/<threshold>.json" -- the trailing
        # "/70" is a real path segment, so only the structure part is checked.
        smiles_segment = url.split("similarity/")[1].rsplit("/", 1)[0]
        assert smiles_segment == "C%2FC%3DC%5CC%23N"
        assert url.endswith("/70.json")

    def test_no_structure_returns_empty(self, annotator):
        record = {**ASPIRIN_RECORD, "molecule_structures": {}}
        patcher, api = with_api([("molecule/CHEMBL25", record)])
        with patcher:
            assert annotator.get_analogs("CHEMBL25") == []


class TestCrossReferences:
    def test_reads_chembl_xrefs(self, annotator):
        patcher, api = with_api([("molecule/CHEMBL25", ASPIRIN_RECORD)])
        with patcher:
            refs = annotator.get_cross_references("CHEMBL25")
        assert refs == [{"source": "DailyMed", "id": "aspirin", "name": "aspirin"}]


# =====================================================================
# Aggregation
# =====================================================================


class TestAnnotate:
    def test_all_sources_excludes_analogs(self):
        assert "analogs" not in ALL_SOURCES
        assert "development" in ALL_SOURCES

    def test_unknown_compound_short_circuits(self, annotator):
        patcher, api = with_api(NO_MATCH_ROUTES)
        with patcher:
            result = annotator.annotate("not-a-compound")
        assert result["found"] is False
        assert "development" not in result

    def test_selected_sources_only(self, annotator):
        patcher, api = with_api([("molecule/CHEMBL25", ASPIRIN_RECORD), DRUG_NONE])
        with patcher:
            result = annotator.annotate("CHEMBL25", sources=["development"])
        assert "development" in result
        assert "indications" not in result
        assert not api.requests_to("drug_indication.json")

    def test_targets_and_activities_share_one_fetch(self, annotator):
        """The profile is derived from the activity table, not a second call."""
        patcher, api = with_api(_activity_routes())
        with patcher:
            result = annotator.annotate(
                "CHEMBL941", sources=["targets", "activities"]
            )
        assert len(api.requests_to("activity.json")) == 1
        assert "activities" in result
        assert "target_profile" in result
        assert "selectivity" in result

    def test_identity_always_present(self, annotator):
        patcher, api = with_api([("molecule/CHEMBL25", ASPIRIN_RECORD), DRUG_NONE])
        with patcher:
            result = annotator.annotate("CHEMBL25", sources=["development"])
        assert result["chembl_id"] == "CHEMBL25"
        assert result["parent_chembl_id"] == "CHEMBL25"
        assert result["pref_name"] == "ASPIRIN"


def _screen_routes():
    return [record_batch_route(ASPIRIN_RECORD), ("molecule/CHEMBL25", ASPIRIN_RECORD), DRUG_NONE, *NO_MATCH_ROUTES]


class TestAnnotateBatch:
    def test_maps_identifier_to_annotation(self, annotator):
        patcher, api = with_routing_api(_screen_routes())
        with patcher:
            results = annotator.annotate_batch(
                ["CHEMBL25", "nope"], sources=["development"]
            )
        assert results["CHEMBL25"]["found"] is True
        assert results["nope"]["found"] is False

    def test_identity_is_batched_up_front(self, annotator):
        patcher, api = with_routing_api(_screen_routes())
        with patcher:
            annotator.annotate_batch(["CHEMBL25"], sources=["development"])
        # One batched record request, reused by the per-source methods.
        molecule_calls = [p for u, p in api.calls if "molecule.json" in u or "/molecule/" in u]
        assert [p.get("molecule_chembl_id__in") for p in molecule_calls] == ["CHEMBL25"]


# =====================================================================
# obs column flattening
# =====================================================================


class TestSummaryColumns:
    def test_empty_row_yields_defaults_not_crashes(self):
        cols = chembl_summary_columns([{}])
        assert cols["drug_chembl_id"] == [""]
        assert cols["drug_in_chembl"] == [False]
        assert cols["drug_n_indications"] == [0]
        assert cols["drug_max_phase"] == [None]

    def test_flattens_nested_development(self):
        cols = chembl_summary_columns(
            [{"found": True, "development": {"development_phase": "approved", "is_approved": True}}]
        )
        assert cols["drug_development_phase"] == ["approved"]
        assert cols["drug_is_approved"] == [True]

    def test_moa_prefers_the_efficacy_mechanism(self):
        cols = chembl_summary_columns(
            [
                {
                    "mechanisms": [
                        {"mechanism": "off-target binder", "action_type": "BINDING AGENT",
                         "disease_efficacy": False},
                        {"mechanism": "COX inhibitor", "action_type": "INHIBITOR",
                         "disease_efficacy": True},
                    ]
                }
            ]
        )
        assert cols["drug_moa"] == ["COX inhibitor"]
        assert cols["drug_action_type"] == ["INHIBITOR"]

    def test_moa_falls_back_to_first_mechanism(self):
        cols = chembl_summary_columns(
            [{"mechanisms": [{"mechanism": "unclear", "action_type": "OTHER"}]}]
        )
        assert cols["drug_moa"] == ["unclear"]

    def test_withdrawn_reason_uses_the_withdrawal_row(self):
        cols = chembl_summary_columns(
            [
                {
                    "warnings": [
                        {"warning_type": "Black Box Warning", "warning_class": "hepatotoxicity"},
                        {"warning_type": "Withdrawn", "warning_class": "cardiotoxicity"},
                    ]
                }
            ]
        )
        assert cols["drug_withdrawn_reason"] == ["cardiotoxicity"]

    def test_no_withdrawal_leaves_reason_blank(self):
        cols = chembl_summary_columns(
            [{"warnings": [{"warning_type": "Black Box Warning", "warning_class": "x"}]}]
        )
        assert cols["drug_withdrawn_reason"] == [""]

    def test_every_column_has_one_value_per_row(self):
        rows = [{}, {"found": True}, {"indications": [{"indication": "pain"}]}]
        for name, values in chembl_summary_columns(rows).items():
            assert len(values) == len(rows), name


# =====================================================================
# annotate_adata
# =====================================================================


class TestAnnotateAdata:
    def _adata(self):
        adata = AnnData(obs=pd.DataFrame({"compound": ["CHEMBL25", "CHEMBL25", "nope"]}))
        adata.obs.index = [f"cell_{i}" for i in range(3)]
        return adata

    def test_adds_drug_columns_and_uns(self, annotator):
        patcher, api = with_routing_api(_screen_routes())
        with patcher:
            out = annotator.annotate_adata(
                self._adata(), column="compound", sources=["development"],
            )
        assert out.obs["drug_pref_name"].tolist() == ["ASPIRIN", "ASPIRIN", ""]
        assert out.obs["drug_development_phase"].tolist()[0] == "approved"
        assert "chembl_annotations" in out.uns
        assert "chembl_annotation_limits" in out.uns

    def test_unique_compounds_are_fetched_once(self, annotator):
        patcher, api = with_routing_api(_screen_routes())
        with patcher:
            annotator.annotate_adata(
                self._adata(), column="compound", sources=["development"],
            )
        record_requests = api.requests_to("molecule/CHEMBL25") + [
            c for c in api.calls if "CHEMBL25" in c[1].get("molecule_chembl_id__in", "")
        ]
        assert len(record_requests) == 1

    def test_flags_saturated_indication_counts(self, annotator):
        rows = {
            "drug_indications": [
                {"efo_id": f"EFO:{i}", "mesh_heading": f"d{i}", "max_phase_for_ind": "4.0",
                 "parent_molecule_chembl_id": "CHEMBL25"}
                for i in range(DEFAULT_INDICATIONS)
            ]
        }
        patcher, api = with_routing_api(
            [*_screen_routes()[:3], ("drug_indication.json", rows), *NO_MATCH_ROUTES]
        )
        with patcher:
            out = annotator.annotate_adata(
                self._adata(), column="compound", sources=["indications"],
            )
        assert out.obs["drug_n_indications_at_limit"].tolist() == [True, True, False]

    def test_copy_is_respected(self, annotator):
        adata = self._adata()
        patcher, api = with_routing_api(_screen_routes())
        with patcher:
            out = annotator.annotate_adata(
                adata, column="compound", sources=["development"], copy=True,
            )
        assert "drug_pref_name" not in adata.obs.columns
        assert "drug_pref_name" in out.obs.columns

    def test_missing_column_raises(self, annotator):
        with pytest.raises(ValueError, match="not found"):
            annotator.annotate_adata(self._adata(), column="nonexistent")


# =====================================================================
# MoleculeAnnotator integration
# =====================================================================


class TestMoleculeAnnotatorIntegration:
    def test_source_groups_map_onto_chembl_sources(self):
        from embpy.resources.molecule.annotator import _chembl_sources_for

        assert _chembl_sources_for(["structural"]) == []
        drug = _chembl_sources_for(["drug"])
        assert "development" in drug and "indications" in drug and "safety" in drug
        assert _chembl_sources_for(["target_profile"]) == ["targets", "activities"]

    def test_source_groups_are_deduplicated(self):
        from embpy.resources.molecule.annotator import _chembl_sources_for

        selected = _chembl_sources_for(["drug", "drug", "target_profile"])
        assert len(selected) == len(set(selected))

    def test_all_includes_the_chembl_groups(self):
        from embpy.resources.molecule.annotator import (
            _CHEMBL_SOURCE_GROUPS,
            MoleculeAnnotator,
        )

        annotator = MoleculeAnnotator(rate_limit_delay=0)
        with patch.object(
            MoleculeAnnotator, "_resolve_to_smiles", return_value=None
        ), patch.object(ChEMBLAnnotator, "annotate", return_value={"found": False}) as ann:
            annotator.annotate("aspirin", sources="all")
        requested = ann.call_args.kwargs["sources"]
        for group in _CHEMBL_SOURCE_GROUPS["drug"]:
            assert group in requested

    def test_mechanism_of_action_keeps_legacy_keys(self):
        from embpy.resources.molecule.annotator import MoleculeAnnotator

        annotator = MoleculeAnnotator(rate_limit_delay=0)
        patcher, api = with_api(
            [("molecule/CHEMBL941", IMATINIB_RECORD), ("mechanism.json", MECHANISM_ROWS)]
            + _activity_routes()
        )
        with patcher:
            moa = annotator.get_mechanism_of_action("CHEMBL941")
        assert set(moa[0]) == {
            "mechanism",
            "action_type",
            "target_name",
            "target_chembl_id",
        }

    def test_mechanism_of_action_populates_target_name(self):
        """The ``mechanism`` payload has no ``target_name`` to read.

        Reading one off it -- as this method used to -- yielded ``""`` for
        every mechanism ever returned. The name has to come from the
        ``target`` endpoint.
        """
        from embpy.resources.molecule.annotator import MoleculeAnnotator

        annotator = MoleculeAnnotator(rate_limit_delay=0)
        patcher, api = with_api(
            [
                ("molecule/CHEMBL941", IMATINIB_RECORD),
                ("mechanism.json", MECHANISM_ROWS),
                ("target.json", _targets_for),
            ]
        )
        with patcher:
            moa = annotator.get_mechanism_of_action("CHEMBL941")
        assert moa[0]["target_name"] == "Tyrosine-protein kinase ABL1"
        assert moa[0]["target_chembl_id"] == "CHEMBL1862"

    def test_pubchem_smiles_key_rename_is_handled(self):
        """PubChem answers a ``CanonicalSMILES`` request under another key.

        A request for ``CanonicalSMILES`` returns ``ConnectivitySMILES``, and
        one for ``IsomericSMILES`` returns ``SMILES``. Reading back the name
        that was asked for resolves every compound to ``None``.
        """
        from embpy.resources.molecule import annotator as annotator_mod

        annotator = annotator_mod.MoleculeAnnotator(rate_limit_delay=0)
        payload = {
            "PropertyTable": {
                "Properties": [{"CID": 2244, "SMILES": "CC(=O)Oc1ccccc1C(=O)O"}]
            }
        }
        with patch.object(annotator_mod, "_get_json", return_value=payload):
            assert annotator._resolve_to_smiles("aspirin") == "CC(=O)Oc1ccccc1C(=O)O"

    def test_connectivity_smiles_key_is_also_accepted(self):
        from embpy.resources.molecule import annotator as annotator_mod

        annotator = annotator_mod.MoleculeAnnotator(rate_limit_delay=0)
        payload = {
            "PropertyTable": {
                "Properties": [{"ConnectivitySMILES": "CC(=O)Oc1ccccc1C(=O)O"}]
            }
        }
        with patch.object(annotator_mod, "_get_json", return_value=payload):
            assert annotator._resolve_to_smiles("aspirin") == "CC(=O)Oc1ccccc1C(=O)O"

    def test_chembl_property_is_reused(self):
        from embpy.resources.molecule.annotator import MoleculeAnnotator

        annotator = MoleculeAnnotator(rate_limit_delay=0)
        assert annotator.chembl is annotator.chembl


# =====================================================================
# Typed errors from the HTTP helper
# =====================================================================


class _Response:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class TestGetJson:
    @pytest.mark.parametrize(("code", "kind"), [(500, "http_5xx"), (503, "http_5xx"), (429, "rate_limited")])
    def test_outages_raise_typed_errors(self, code, kind):
        with patch.object(chembl_mod.requests, "get", return_value=_Response(code)):
            with pytest.raises(ChEMBLServiceError) as err:
                chembl_mod._get_json(f"{chembl_mod.CHEMBL}/status.json")
        assert err.value.status_code == code
        assert err.value.kind == kind

    def test_timeout_raises(self):
        with patch.object(chembl_mod.requests, "get", side_effect=requests.Timeout("slow")):
            with pytest.raises(ChEMBLServiceError) as err:
                chembl_mod._get_json(f"{chembl_mod.CHEMBL}/molecule/CHEMBL25.json")
        assert err.value.kind == "timeout"
        assert err.value.status_code is None

    def test_connection_error_raises(self):
        with patch.object(chembl_mod.requests, "get", side_effect=requests.ConnectionError("down")):
            with pytest.raises(ChEMBLServiceError) as err:
                chembl_mod._get_json(f"{chembl_mod.CHEMBL}/molecule/CHEMBL25.json")
        assert err.value.kind == "connection"

    def test_404_is_absent_not_an_error(self):
        with patch.object(chembl_mod.requests, "get", return_value=_Response(404)):
            assert chembl_mod._get_json(f"{chembl_mod.CHEMBL}/molecule/CHEMBL0.json") is None

    def test_html_error_page_with_200_raises(self):
        with patch.object(chembl_mod.requests, "get", return_value=_Response(200, None, "<html>")):
            with pytest.raises(ChEMBLServiceError) as err:
                chembl_mod._get_json(f"{chembl_mod.CHEMBL}/status.json")
        assert err.value.kind == "invalid_json"

    def test_success_returns_payload(self):
        with patch.object(chembl_mod.requests, "get", return_value=_Response(200, {"a": 1})):
            assert chembl_mod._get_json(f"{chembl_mod.CHEMBL}/status.json") == {"a": 1}

    def test_outage_propagates_out_of_getters(self, annotator):
        """An outage must not read as "no mechanisms"."""

        def boom(url, params=None, timeout=30):
            raise ChEMBLServiceError(500, url)

        with patch.object(chembl_mod, "_get_json", boom), pytest.raises(ChEMBLServiceError):
            annotator.get_mechanisms("CHEMBL941")


# =====================================================================
# Recorded payloads (fixtures/chembl, 2026-11-11, ChEMBL_37)
# =====================================================================

FIXTURES = Path(__file__).parent / "fixtures" / "chembl"


def fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


ASPIRIN_KEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
IMATINIB_KEY = "KTUFNOKKBVMGRW-UHFFFAOYSA-N"
MESYLATE_KEY = "YLMAHDNUQAMNNX-UHFFFAOYSA-N"
ABSENT_KEY = "AAAAAAAAAAAAAA-UHFFFAOYSA-N"
IMATINIB_SMILES = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"


def _keys_in(params):
    return params["molecule_structures__standard_inchi_key__in"].split(",")


def identity_api(fail=None):
    """Serve the recorded InChIKey batch, filtered to each request's keys.

    ``fail(keys)`` may raise to simulate a failed batch.
    """
    recorded = fixture("molecule_inchikey_in")["molecules"]

    def payload(params):
        keys = _keys_in(params)
        if fail is not None:
            fail(keys)
        return {
            "molecules": [m for m in recorded if m["molecule_structures"]["standard_inchi_key"] in keys],
            "page_meta": {"next": None, "total_count": None},
        }

    return with_routing_api([(route("molecule.json", "molecule_structures__standard_inchi_key__in"), payload)])


class TestResolveMany:
    def test_one_batched_request_maps_every_key(self, annotator):
        patcher, api = identity_api()
        with patcher:
            res = annotator.resolve_many([ASPIRIN_KEY, IMATINIB_KEY, MESYLATE_KEY, ABSENT_KEY])
        assert len(api.calls) == 1
        assert sorted(_keys_in(api.calls[0][1])) == sorted([ASPIRIN_KEY, IMATINIB_KEY, MESYLATE_KEY, ABSENT_KEY])
        assert res[ASPIRIN_KEY].molecule_chembl_id == "CHEMBL25"
        assert res[IMATINIB_KEY].parent_chembl_id == "CHEMBL941"
        # The mesylate is matched as itself, and its parent is imatinib.
        assert res[MESYLATE_KEY].molecule_chembl_id == "CHEMBL1642"
        assert res[MESYLATE_KEY].parent_chembl_id == "CHEMBL941"
        assert res[ABSENT_KEY].status == "no_match"
        assert all(r.matched_inchikey == k for k, r in res.items() if r.found)

    def test_smiles_inputs_are_converted_to_inchikeys(self, annotator):
        pytest.importorskip("rdkit")
        patcher, api = identity_api()
        with patcher:
            res = annotator.resolve_many([IMATINIB_SMILES, "CC(=O)Oc1ccccc1C(=O)O"])
        assert len(api.calls) == 1
        assert res[IMATINIB_SMILES].molecule_chembl_id == "CHEMBL941"
        assert res[IMATINIB_SMILES].inchikey == IMATINIB_KEY
        assert res["CC(=O)Oc1ccccc1C(=O)O"].molecule_chembl_id == "CHEMBL25"

    def test_invalid_structure(self, annotator):
        pytest.importorskip("rdkit")
        patcher, api = identity_api()
        with patcher, patch.object(chembl_mod, "smiles_to_inchikey", return_value=None):
            res = annotator.resolve_many(["CCO"])
        assert res["CCO"].status == "invalid_structure"
        assert not api.calls

    def test_foreign_row_marks_the_chunk_filter_ignored(self, annotator):
        """ChEMBL ignores an unsupported filter and returns the whole table.

        The recorded payload is the live answer to a misspelled InChIKey
        filter: 2,921,148 molecules, none of them the one asked for.
        """
        ignored = fixture("molecule_ignored_toplevel_filter")
        patcher, api = with_api([(route("molecule.json"), ignored)])
        with patcher:
            res = annotator.resolve_many([ASPIRIN_KEY, IMATINIB_KEY])
        assert {r.status for r in res.values()} == {"filter_ignored"}
        assert not any(r.found for r in res.values())
        # Rejected on the first page: the next page of the full table is never requested.
        assert len(api.calls) == 1

    def test_failed_batch_of_50_bisects_into_25_and_25(self, annotator):
        keys = [f"{'A' * 13}{chr(65 + i // 26)}-{chr(65 + i % 26) * 10}-N" for i in range(50)]

        def fail(chunk):
            if len(chunk) > 25:
                raise ChEMBLServiceError(500, "molecule.json")

        patcher, api = identity_api(fail)
        with patcher:
            res = annotator.resolve_many(keys, chunk_size=50)
        assert [len(_keys_in(p)) for _, p in api.calls] == [50, 25, 25]
        assert {r.status for r in res.values()} == {"no_match"}

    def test_persistent_failure_isolates_one_key(self, annotator):
        def fail(chunk):
            if IMATINIB_KEY in chunk:
                raise ChEMBLServiceError(500, "molecule.json")

        patcher, api = identity_api(fail)
        with patcher:
            res = annotator.resolve_many([ASPIRIN_KEY, IMATINIB_KEY, MESYLATE_KEY, ABSENT_KEY])
        assert res[IMATINIB_KEY].status == "batch_failed"
        assert "500" in res[IMATINIB_KEY].error
        assert res[ASPIRIN_KEY].status == "matched"
        assert res[MESYLATE_KEY].status == "matched"
        assert res[ABSENT_KEY].status == "no_match"

    def test_failed_keys_are_not_cached(self, annotator):
        state = {"fail": True}

        def fail(chunk):
            if state["fail"]:
                raise ChEMBLServiceError(500, "molecule.json")

        patcher, api = identity_api(fail)
        with patcher:
            assert annotator.resolve_many([IMATINIB_KEY])[IMATINIB_KEY].status == "batch_failed"
            state["fail"] = False
            assert annotator.resolve_many([IMATINIB_KEY])[IMATINIB_KEY].status == "matched"

    def test_foreign_exceptions_propagate(self, annotator):
        """A replaced helper's own errors (e.g. a circuit breaker) are not swallowed."""

        class FetchError(RuntimeError):
            pass

        def fail(chunk):
            raise FetchError("circuit_open")

        patcher, api = identity_api(fail)
        with patcher, pytest.raises(FetchError):
            annotator.resolve_many([ASPIRIN_KEY, IMATINIB_KEY])
        assert len(api.calls) == 1

    def test_batch_errors_opt_in_bisects_foreign_exceptions(self):
        class FetchError(RuntimeError):
            pass

        def fail(chunk):
            if IMATINIB_KEY in chunk:
                raise FetchError("service_error")

        annotator = ChEMBLAnnotator(rate_limit_delay=0, batch_errors=(FetchError,))
        patcher, api = identity_api(fail)
        with patcher:
            res = annotator.resolve_many([ASPIRIN_KEY, IMATINIB_KEY])
        assert res[IMATINIB_KEY].status == "batch_failed"
        assert res[ASPIRIN_KEY].status == "matched"

    def test_missing_response_is_batch_failed_not_no_match(self, annotator):
        patcher, api = with_api([])
        with patcher:
            res = annotator.resolve_many([ASPIRIN_KEY])
        assert res[ASPIRIN_KEY].status == "batch_failed"

    def test_ambiguous_when_one_key_has_two_parents(self, annotator):
        rows = [
            {"molecule_chembl_id": "CHEMBL1", "molecule_hierarchy": {"parent_chembl_id": "CHEMBL1"},
             "molecule_structures": {"standard_inchi_key": ASPIRIN_KEY}},
            {"molecule_chembl_id": "CHEMBL2", "molecule_hierarchy": {"parent_chembl_id": "CHEMBL2"},
             "molecule_structures": {"standard_inchi_key": ASPIRIN_KEY}},
        ]
        patcher, api = with_api([(route("molecule.json"), {"molecules": rows})])
        with patcher:
            res = annotator.resolve(ASPIRIN_KEY)
        assert res.status == "ambiguous"
        assert not res.found
        assert {c[0] for c in res.candidates} == {"CHEMBL1", "CHEMBL2"}


class TestFlexmatch:
    def test_off_by_default(self, annotator):
        assert annotator.flexmatch is False

    def test_opt_in_accepts_only_identical_inchikeys(self):
        """The recorded imatinib flexmatch returns 3 molecules (total_count 3).

        Only CHEMBL941 has the query's InChIKey; the mesylate and a
        deuterated/stereo variant must not be accepted.
        """
        pytest.importorskip("rdkit")
        flex = fixture("molecule_flexmatch_imatinib")
        annotator = ChEMBLAnnotator(rate_limit_delay=0, flexmatch=True)
        patcher, api = with_api(
            [
                (route("molecule.json", "molecule_structures__standard_inchi_key__in"), {"molecules": []}),
                (route("molecule.json", "molecule_structures__canonical_smiles__flexmatch"), flex),
            ]
        )
        with patcher:
            res = annotator.resolve(IMATINIB_SMILES)
        assert res.molecule_chembl_id == "CHEMBL941"
        assert res.matched_by == "smiles_flexmatch"
        assert res.matched_inchikey == IMATINIB_KEY
        assert int(api.params_for("molecule.json")["limit"]) == 1000  # exact lookup went first

    def test_opt_in_rejects_non_identical_hits(self):
        pytest.importorskip("rdkit")
        flex = fixture("molecule_flexmatch_imatinib")
        flex = {**flex, "molecules": [m for m in flex["molecules"] if m["molecule_chembl_id"] != "CHEMBL941"]}
        annotator = ChEMBLAnnotator(rate_limit_delay=0, flexmatch=True)
        patcher, api = with_api(
            [
                (route("molecule.json", "molecule_structures__standard_inchi_key__in"), {"molecules": []}),
                (route("molecule.json", "molecule_structures__canonical_smiles__flexmatch"), flex),
            ]
        )
        with patcher:
            res = annotator.resolve(IMATINIB_SMILES)
        assert res.status == "no_match"
        assert {c[0] for c in res.candidates} == {"CHEMBL1642", "CHEMBL2386595"}


class TestBatchGetters:
    def _records_api(self):
        recorded = fixture("molecule_chembl_id_in")["molecules"]
        return with_routing_api([record_batch_route(*recorded)])

    def test_records_many_primes_per_source_methods(self, annotator):
        """Parent records are fetched once; sources never re-fetch molecule/{P}.json."""
        patcher, api = self._records_api()
        with patcher:
            records = annotator.get_molecule_records_many(["CHEMBL941", "CHEMBL25", "CHEMBL1642", "CHEMBL0"])
            assert records.status("CHEMBL0") == "not_found"
            assert records["CHEMBL941"]["pref_name"] == "IMATINIB"
            dev = annotator.get_development_status("CHEMBL941")
            syn = annotator.get_synonyms("CHEMBL25")
            salt = annotator.resolve("CHEMBL1642")
        assert len(api.calls) == 1
        assert not api.requests_to("molecule/")
        assert dev["first_approval"] == 2001
        assert syn[0]["type"] == "FDA"  # aspirin has no INN row in ChEMBL_37
        assert salt.parent_chembl_id == "CHEMBL941"

    def test_prime_records(self, annotator):
        recorded = fixture("molecule_chembl_id_in")["molecules"]
        annotator.prime_records(recorded)
        patcher, api = with_api([])
        with patcher:
            assert annotator.get_development_status("CHEMBL25")["pref_name"] == "ASPIRIN"
        assert not api.calls

    def test_records_many_validates_rows(self, annotator):
        recorded = fixture("molecule_chembl_id_in")["molecules"]
        patcher, api = with_api([(route("molecule.json"), {"molecules": recorded})])
        with patcher:
            records = annotator.get_molecule_records_many(["CHEMBL25"])
        assert records.status("CHEMBL25") == "filter_ignored"
        assert "CHEMBL25" not in records

    def test_mechanisms_many(self, annotator):
        patcher, api = with_api([("mechanism.json", fixture("mechanism_parent_in"))])
        with patcher:
            mechs = annotator.get_mechanisms_many(["CHEMBL941", "CHEMBL25", "CHEMBL3"])
        assert api.params_for("mechanism.json")["parent_molecule_chembl_id__in"] == "CHEMBL941,CHEMBL25,CHEMBL3"
        assert len(mechs["CHEMBL941"]) == 4
        assert {m["target_chembl_id"] for m in mechs["CHEMBL941"]} == {
            "CHEMBL1862", "CHEMBL1913", "CHEMBL1936", "CHEMBL2096618"
        }
        assert [m["mechanism"] for m in mechs["CHEMBL25"]] == ["Cyclooxygenase inhibitor"]
        assert mechs["CHEMBL3"] == []
        assert mechs.status("CHEMBL3") == "success"

    def test_mechanisms_many_rejects_an_ignored_filter(self, annotator):
        """Recorded: ``parent_chembl_id__in`` is not a mechanism filter (7,561 rows)."""
        patcher, api = with_api([("mechanism.json", fixture("mechanism_ignored_filter"))])
        with patcher:
            mechs = annotator.get_mechanisms_many(["CHEMBL941"])
        assert mechs.status("CHEMBL941") == "filter_ignored"
        assert len(api.calls) == 1

    def test_mechanisms_many_bisects_failures(self, annotator):
        def payload(params):
            parents = params["parent_molecule_chembl_id__in"].split(",")
            if len(parents) > 1:
                raise ChEMBLServiceError(500, "mechanism.json")
            rows = fixture("mechanism_parent_in")["mechanisms"]
            return {"mechanisms": [r for r in rows if r["parent_molecule_chembl_id"] in parents]}

        patcher, api = with_routing_api([("mechanism.json", payload)])
        with patcher:
            mechs = annotator.get_mechanisms_many(["CHEMBL941", "CHEMBL25"])
        assert len(mechs["CHEMBL941"]) == 4 and len(mechs["CHEMBL25"]) == 1
        assert len(api.calls) == 3

    def test_get_mechanisms_follows_pagination(self, annotator):
        page1, page2 = fixture("mechanism_page1"), fixture("mechanism_page2")
        patcher, api = with_api(
            [
                ("molecule/CHEMBL941", IMATINIB_RECORD),
                (lambda url, params: "offset=2" in url, page2),
                ("mechanism.json", page1),
            ]
        )
        with patcher:
            mechs = annotator.get_mechanisms("CHEMBL941")
        assert len(mechs) == 4
        assert len(api.requests_to("mechanism.json")) == 2
        assert api.params_for("mechanism.json")["limit"] == 1000

    def test_indications_many(self, annotator):
        patcher, api = with_api([("drug_indication.json", fixture("drug_indication_parent_in_trimmed"))])
        with patcher:
            ind = annotator.get_indications_many(["CHEMBL941", "CHEMBL25"])
        assert ind["CHEMBL941"] and ind["CHEMBL25"]
        assert all("max_phase_for_indication" in e for e in ind["CHEMBL941"])

    def test_warnings_many(self, annotator):
        patcher, api = with_api([("drug_warning.json", fixture("drug_warning_parent_in"))])
        with patcher:
            warn = annotator.get_warnings_many(["CHEMBL941", "CHEMBL25", "CHEMBL122"])
        assert warn["CHEMBL941"] == [] and warn["CHEMBL25"] == []
        assert {w["warning_type"] for w in warn["CHEMBL122"]} == {"Black Box Warning", "Withdrawn"}

    def test_metabolism_and_forms_many(self, annotator):
        patcher, api = with_api(
            [
                ("metabolism.json", fixture("metabolism_drug_in")),
                ("molecule_form.json", fixture("molecule_form_parent_in")),
            ]
        )
        with patcher:
            met = annotator.get_metabolism_many(["CHEMBL941", "CHEMBL25"])
            forms = annotator.get_molecule_forms_many(["CHEMBL941", "CHEMBL25"])
        assert "drug_chembl_id__in" in api.params_for("metabolism.json")
        assert "parent_chembl_id__in" in api.params_for("molecule_form.json")
        assert met["CHEMBL941"] == []
        assert {m["metabolite_name"] for m in met["CHEMBL25"]} >= {"SALICYLIC ACID"}
        assert {f["molecule_chembl_id"] for f in forms["CHEMBL941"]} == {"CHEMBL941", "CHEMBL1642", "CHEMBL2386595"}

    def test_batch_result_status(self):
        result = BatchResult({"A": [], "B": None})
        result.failed["C"] = {"status": "batch_failed", "error": "x"}
        assert [result.status(k) for k in "ABCD"] == ["success", "not_found", "batch_failed", "not_requested"]


class TestRecordedTargets:
    def test_protein_classes_come_from_target_component(self, annotator):
        """ChEMBL_37 ``target`` components carry no protein classification.

        Reading it off the target payload left ``target_class`` ``None`` for
        every target; the classes are on ``target_component``.
        """
        def filtered(name, collection, param, field):
            rows = fixture(name)[collection]

            def payload(params):
                wanted = set(params[param].split(","))
                return {collection: [r for r in rows if str(r[field]) in wanted]}

            return payload

        patcher, api = with_routing_api(
            [
                ("target_component.json",
                 filtered("target_component_in", "target_components", "component_id__in", "component_id")),
                ("protein_classification.json",
                 filtered("protein_classification_in", "protein_classifications",
                          "protein_class_id__in", "protein_class_id")),
                ("target.json", filtered("target_in", "targets", "target_chembl_id__in", "target_chembl_id")),
            ]
        )
        with patcher:
            details = annotator.get_target_details(["CHEMBL1862", "CHEMBL2094253", "CHEMBL2096618"])
        abl = details["CHEMBL1862"]
        assert abl["gene_symbol"] == "ABL1"
        assert abl["target_class"] == "enzyme"
        assert abl["protein_class_path"] == ["enzyme", "kinase", "protein kinase", "tk", "abl"]
        assert details["CHEMBL2094253"]["target_type"] == "PROTEIN FAMILY"
        assert details["CHEMBL2094253"]["gene_symbols"] == ["PTGS2", "PTGS1"]
        assert len(api.requests_to("target_component.json")) == 1
        assert len(api.requests_to("protein_classification.json")) == 1

    def test_activities_are_validated(self, annotator):
        patcher, api = with_api(
            [("molecule/CHEMBL941", IMATINIB_RECORD), ("activity.json", fixture("activity_imatinib_top5"))]
        )
        with patcher:
            acts = annotator.get_activities("CHEMBL941", limit=5)
        assert [a["pchembl_value"] for a in acts] == [10.22, 9.96, 9.82, 9.7, 9.7]
        assert len(api.calls) == 3  # one extra row establishes truncation
        assert annotator.activity_fetch_metadata["CHEMBL941"]["truncated"] is True
