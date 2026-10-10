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

Neither shows up as an error, so the field names are pinned by test.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from anndata import AnnData

from embpy.resources.molecule.chembl import (
    ALL_SOURCES,
    AVAILABILITY_LABELS,
    DEFAULT_INDICATIONS,
    ChEMBLAnnotator,
    ChEMBLResolution,
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
                return payload
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
            (None, "preclinical"),
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

    def test_smiles_identifier_tries_flexmatch_first(self, annotator):
        patcher, api = with_api(
            [
                (
                    route("molecule.json", "molecule_structures__canonical_smiles__flexmatch"),
                    {"molecules": [ASPIRIN_RECORD]},
                )
            ]
        )
        with patcher:
            res = annotator.resolve("CC(=O)Oc1ccccc1C(=O)O")
        assert res.matched_by == "smiles"
        assert api.requests_to("molecule.json")

    def test_salt_resolves_to_its_parent(self, annotator):
        patcher, api = with_api([("molecule/CHEMBL1642", MESYLATE_RECORD)])
        with patcher:
            res = annotator.resolve("CHEMBL1642")
        assert res.molecule_chembl_id == "CHEMBL1642"
        assert res.parent_chembl_id == "CHEMBL941"

    def test_not_found(self, annotator):
        patcher, api = with_api([])
        with patcher:
            res = annotator.resolve("not-a-real-compound")
        assert not res.found
        assert res.molecule_chembl_id is None

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
        assert status["parenteral"] is False
        assert status["natural_product"] is True
        assert status["withdrawn_flag"] is False
        assert status["black_box_warning"] is False

    def test_not_applicable_flag_is_none(self, annotator):
        # aspirin's `orphan` is -1 in ChEMBL
        assert self._status(annotator)["orphan"] is None

    def test_unknown_compound_returns_empty(self, annotator):
        patcher, api = with_api([])
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
        },
        # Same indication, higher phase, from another reference.
        {
            "efo_id": "EFO:0000275",
            "efo_term": "atrial fibrillation",
            "mesh_id": "D001281",
            "mesh_heading": "Atrial Fibrillation",
            "max_phase_for_ind": "4.0",
            "indication_refs": [{"ref_type": "FDA"}, {"ref_type": "ATC"}],
        },
        {
            "efo_id": "EFO:0003843",
            "efo_term": "pain",
            "mesh_id": "D010146",
            "mesh_heading": "Pain",
            "max_phase_for_ind": "2.0",
            "indication_refs": [],
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
        patcher, api = with_api([])
        with patcher:
            assert annotator.get_indications("nope") == []


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


class TestWarnings:
    def test_keyed_on_parent_molecule(self, annotator):
        patcher, api = with_api(
            [("molecule/CHEMBL122", ASPIRIN_RECORD), ("drug_warning.json", WARNING_ROWS)]
        )
        with patcher:
            annotator.get_warnings("CHEMBL122")
        assert "parent_molecule_chembl_id" in api.params_for("drug_warning.json")

    def test_parses_withdrawal(self, annotator):
        patcher, api = with_api(
            [("molecule/CHEMBL122", ASPIRIN_RECORD), ("drug_warning.json", WARNING_ROWS)]
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


def _activity_routes():
    return [
        ("molecule/CHEMBL941", IMATINIB_RECORD),
        ("activity.json", ACTIVITY_ROWS),
        ("target.json", TARGET_ROWS),
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
                include_protein_classes=False,
            )
        assert profile[0]["best_pchembl"] == 7.0

    def test_empty_activities(self, annotator):
        patcher, api = with_api([("molecule/CHEMBL941", IMATINIB_RECORD)])
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
        assert analogs[0]["development_phase"] == "preclinical"

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
        patcher, api = with_api([])
        with patcher:
            result = annotator.annotate("not-a-compound")
        assert result["found"] is False
        assert "development" not in result

    def test_selected_sources_only(self, annotator):
        patcher, api = with_api([("molecule/CHEMBL25", ASPIRIN_RECORD)])
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
        patcher, api = with_api([("molecule/CHEMBL25", ASPIRIN_RECORD)])
        with patcher:
            result = annotator.annotate("CHEMBL25", sources=["development"])
        assert result["chembl_id"] == "CHEMBL25"
        assert result["parent_chembl_id"] == "CHEMBL25"
        assert result["pref_name"] == "ASPIRIN"


class TestAnnotateBatch:
    def test_maps_identifier_to_annotation(self, annotator):
        patcher, api = with_api([("molecule/CHEMBL25", ASPIRIN_RECORD)])
        with patcher:
            results = annotator.annotate_batch(
                ["CHEMBL25", "nope"], sources=["development"]
            )
        assert results["CHEMBL25"]["found"] is True
        assert results["nope"]["found"] is False


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
        patcher, api = with_api([("molecule/CHEMBL25", ASPIRIN_RECORD)])
        with patcher:
            out = annotator.annotate_adata(
                self._adata(), column="compound", sources=["development"],
            )
        assert out.obs["drug_pref_name"].tolist() == ["ASPIRIN", "ASPIRIN", ""]
        assert out.obs["drug_development_phase"].tolist()[0] == "approved"
        assert "chembl_annotations" in out.uns
        assert "chembl_annotation_limits" in out.uns

    def test_unique_compounds_are_fetched_once(self, annotator):
        patcher, api = with_api([("molecule/CHEMBL25", ASPIRIN_RECORD)])
        with patcher:
            annotator.annotate_adata(
                self._adata(), column="compound", sources=["development"],
            )
        assert len(api.requests_to("molecule/CHEMBL25")) == 1

    def test_flags_saturated_indication_counts(self, annotator):
        rows = {
            "drug_indications": [
                {"efo_id": f"EFO:{i}", "mesh_heading": f"d{i}", "max_phase_for_ind": "4.0"}
                for i in range(DEFAULT_INDICATIONS)
            ]
        }
        patcher, api = with_api(
            [("molecule/CHEMBL25", ASPIRIN_RECORD), ("drug_indication.json", rows)]
        )
        with patcher:
            out = annotator.annotate_adata(
                self._adata(), column="compound", sources=["indications"],
            )
        assert out.obs["drug_n_indications_at_limit"].tolist() == [True, True, False]

    def test_copy_is_respected(self, annotator):
        adata = self._adata()
        patcher, api = with_api([("molecule/CHEMBL25", ASPIRIN_RECORD)])
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
                ("target.json", TARGET_ROWS),
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
