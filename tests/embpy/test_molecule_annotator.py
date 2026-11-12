"""Tests for embpy.resources.molecule_annotator -- MoleculeAnnotator.

Provider payloads come from ``tests/embpy/fixtures/molecule`` (recorded live
on 2026-11-11, see ``SOURCES.txt`` there). No test sends a request: HTTP is
replaced either at the module helpers (``_get_json`` / ``_get_text``, the
way a batch driver patches them) or at ``requests.get`` underneath the
unpatched helpers, to exercise their 404-versus-failure bookkeeping.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest
import requests

from embpy.resources.molecule import annotator as mod
from embpy.resources.molecule_annotator import MoleculeAnnotator

FIXTURES = Path(__file__).parent / "fixtures" / "molecule"
PUBCHEM = mod.PUBCHEM
CHEBI_REST = mod.CHEBI_REST

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_IK = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
IMATINIB = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
IMATINIB_IK = "KTUFNOKKBVMGRW-UHFFFAOYSA-N"
# 4-hydroxytamoxifen as stored in the LINCS cohort: E/Z bonds written with
# '/' and '\'. Every PubChem request that put this in the URL path got 400.
HYDROXYTAMOXIFEN = r"CC/C(=C(\c1ccc(O)cc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"
BACKSLASH_ONLY = r"F\C=C\F"  # RDKit canonicalises this to F/C=C/F


def load(name: str):
    path = FIXTURES / name
    if path.suffix == ".txt":
        return path.read_text().strip()
    return json.loads(path.read_text())


@pytest.fixture
def annotator():
    return MoleculeAnnotator(rate_limit_delay=0)


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Fail any real HTTP request (tests patch requests.get themselves when needed)."""
    if os.environ.get("EMBPY_NETWORK_TESTS") == "1":
        return

    def refuse(*args, **kwargs):
        raise requests.ConnectionError(f"network disabled in unit tests: {args[:1]}")

    monkeypatch.setattr(requests, "get", refuse)
    monkeypatch.setattr(requests, "post", refuse)


class FakeHelpers:
    """Stand-in for the module helpers, dispatching on the URL.

    ``routes`` maps a predicate ``(url, params) -> bool`` to a response
    (an object, ``None`` or an exception instance to raise); the first
    match wins. Every call is recorded; an unrouted call fails the test.
    """

    def __init__(self, routes=None):
        self.routes = list(routes or [])
        self.calls: list[tuple[str, dict | None]] = []

    def add(self, predicate, response):
        self.routes.append((predicate, response))
        return self

    def __call__(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params) if params else None))
        for predicate, response in self.routes:
            if predicate(url, params or {}):
                if isinstance(response, BaseException):
                    raise response
                return response(url, params or {}) if callable(response) else response
        raise AssertionError(f"unexpected request: {url} {params}")

    def urls(self):
        return [u for u, _ in self.calls]


def no_network(*args, **kwargs):
    raise AssertionError(f"no request expected, got {args} {kwargs}")


class FakeResponse:
    """Minimal ``requests.Response`` for the unpatched helpers."""

    def __init__(self, status_code, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)


# =====================================================================
# Physicochemical properties (RDKit -- no API needed)
# =====================================================================


class TestPhysicochemicalProperties:
    def test_ethanol(self, annotator):
        props = annotator.get_physicochemical_properties("CCO")
        assert props["molecular_weight"] > 40
        assert "logp" in props
        assert "tpsa" in props
        assert "hbd" in props
        assert "hba" in props
        assert "lipinski_violations" in props
        assert props["lipinski_violations"] == 0

    def test_invalid_smiles(self, annotator):
        props = annotator.get_physicochemical_properties("INVALID_SMILES_XYZ")
        assert "error" in props

    def test_caffeine_properties(self, annotator):
        props = annotator.get_physicochemical_properties("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        assert props["molecular_weight"] > 190
        assert props["aromatic_rings"] >= 1
        assert "qed" in props

    def test_formula(self, annotator):
        props = annotator.get_physicochemical_properties("CCO")
        assert props["formula"] == "C2H6O"


# =====================================================================
# Bioactivities (mocked ChEMBL)
# =====================================================================


class TestBioactivities:
    @patch("embpy.resources.molecule.annotator._get_json")
    def test_get_bioactivities(self, mock_get, annotator):
        mock_get.side_effect = [
            {"molecules": [{"molecule_chembl_id": "CHEMBL25"}]},
            {"activities": [
                {
                    "target_chembl_id": "CHEMBL220",
                    "target_pref_name": "Acetylcholinesterase",
                    "target_organism": "Homo sapiens",
                    "standard_type": "IC50",
                    "standard_value": "100",
                    "standard_units": "nM",
                    "standard_relation": "=",
                    "assay_type": "B",
                    "pchembl_value": "7.0",
                },
            ]},
        ]
        activities = annotator.get_bioactivities("CCO")
        assert len(activities) == 1
        assert activities[0]["activity_type"] == "IC50"

    @patch("embpy.resources.molecule.annotator._get_json")
    def test_no_chembl_id(self, mock_get, annotator):
        mock_get.return_value = None
        activities = annotator.get_bioactivities("INVALID")
        assert activities == []


class TestTargetProteins:
    @patch("embpy.resources.molecule.annotator._get_json")
    def test_get_targets(self, mock_get, annotator):
        mock_get.side_effect = [
            {"molecules": [{"molecule_chembl_id": "CHEMBL25"}]},
            {"activities": [
                {"target_chembl_id": "CHEMBL220", "target_pref_name": "AChE", "target_organism": "Human"},
                {"target_chembl_id": "CHEMBL220", "target_pref_name": "AChE", "target_organism": "Human"},
                {"target_chembl_id": "CHEMBL333", "target_pref_name": "COX-2", "target_organism": "Human"},
            ]},
        ]
        targets = annotator.get_target_proteins("CCO")
        assert len(targets) == 2


# =====================================================================
# PubChem identity (T11)
# =====================================================================


def _smiles_route(url, params):
    return url == f"{PUBCHEM}/compound/smiles/cids/JSON"


class TestPubChemSmilesResolution:
    @pytest.mark.parametrize(
        ("smiles", "fixture", "cid"),
        [
            (HYDROXYTAMOXIFEN, "pubchem/smiles_cids_449459.json", 449459),
            (BACKSLASH_ONLY, "pubchem/smiles_cids_backslash.json", 2733526),
        ],
    )
    def test_smiles_goes_in_query_params_never_in_path(self, annotator, smiles, fixture, cid):
        from rdkit import Chem

        fake = FakeHelpers().add(_smiles_route, load(fixture))
        with patch.object(mod, "_get_json", fake):
            res = annotator.resolve_pubchem_cid(smiles)
        assert res == {"status": "matched", "cid": cid, "cids": [cid], "method": "pubchem_smiles"}
        assert len(fake.calls) == 1
        url, params = fake.calls[0]
        canon = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        assert "/" in canon  # RDKit rewrites '\' bonds with '/', the 400 trigger
        assert params == {"smiles": canon}
        assert url == f"{PUBCHEM}/compound/smiles/cids/JSON"
        path = urlsplit(url).path
        assert "\\" not in url and "=" not in path and "(" not in path

    @pytest.mark.parametrize("payload", [load("pubchem/smiles_cids_zero.json"), {"IdentifierList": {"CID": []}}])
    def test_cid_zero_or_empty_is_no_match_without_name_fallback(self, annotator, payload):
        fake = FakeHelpers().add(_smiles_route, payload)
        with patch.object(mod, "_get_json", fake):
            res = annotator.resolve_pubchem_cid(ASPIRIN)
            # No /compound/name/<SMILES> fallback for an input that parsed as SMILES.
            assert len(fake.calls) == 1
            assert annotator._resolve_pubchem_cid(ASPIRIN) is None
        assert res["status"] == "no_match"
        assert res["cid"] is None and res["cids"] == []

    def test_legacy_int_wrapper(self, annotator):
        fake = FakeHelpers().add(_smiles_route, load("pubchem/smiles_cids_449459.json"))
        with patch.object(mod, "_get_json", fake):
            assert annotator._resolve_pubchem_cid(HYDROXYTAMOXIFEN) == 449459

    def test_patched_helper_errors_propagate(self, annotator):
        class FetchError(RuntimeError):
            pass

        fake = FakeHelpers().add(_smiles_route, FetchError("transient_error"))
        with patch.object(mod, "_get_json", fake), pytest.raises(FetchError):
            annotator.resolve_pubchem_cid(ASPIRIN)

    def test_unpatched_400_is_invalid_structure_not_no_match(self, annotator):
        body = load("pubchem/badrequest_400_standardize.json")
        with patch.object(mod.requests, "get", return_value=FakeResponse(400, body)):
            res = annotator.resolve_pubchem_cid(ASPIRIN)
        assert res["status"] == "invalid_structure"
        assert "400" in res["error"]

    def test_unpatched_connection_error_is_service_error(self, annotator):
        with patch.object(mod.requests, "get", side_effect=requests.ConnectionError("down")):
            res = annotator.resolve_pubchem_cid(ASPIRIN)
        assert res["status"] == "service_error"
        assert res["cid"] is None

    def test_cid_string_needs_no_request(self, annotator):
        with patch.object(mod, "_get_json", no_network):
            assert annotator.resolve_pubchem_cid("2244")["cid"] == 2244
            assert annotator.resolve_pubchem_cid("CID 2244")["status"] == "matched"

    def test_name_is_quoted_completely(self, annotator):
        fake = FakeHelpers().add(lambda u, p: "/compound/name/" in u, load("pubchem/smiles_cids_449459.json"))
        with patch.object(mod, "_get_json", fake):
            res = annotator.resolve_pubchem_cid("afimoxifene (Z/E)")
        assert res["method"] == "pubchem_name" and res["status"] == "matched"
        assert "%2F" in fake.urls()[0] and "(Z/E)" not in fake.urls()[0]


class TestPubChemInChIKeyBatch:
    @staticmethod
    def _recorded_answer(rows):
        """Serve the recorded rows whose key is in the requested chunk."""

        def respond(url, params):
            keys = set(url.split("/compound/inchikey/")[1].split("/")[0].split(","))
            hits = [r for r in rows if r["InChIKey"] in keys]
            if not hits:
                return None  # what a patched helper hands back for PUGREST.NotFound
            return {"PropertyTable": {"Properties": hits}}

        return respond

    def test_maps_each_cid_back_to_its_key(self, annotator):
        keys = load("pubchem/inchikey_property_50_request.txt").split(",")
        rows = load("pubchem/inchikey_property_50.json")["PropertyTable"]["Properties"]
        fake = FakeHelpers().add(lambda u, p: "/compound/inchikey/" in u, self._recorded_answer(rows))
        with patch.object(mod, "_get_json", fake):
            res = annotator.resolve_pubchem_cids_by_inchikey(keys, chunk_size=20)

        assert len(fake.calls) == 3  # 20 + 20 + 10 keys
        for url, params in fake.calls:
            assert url.endswith("/property/InChIKey/JSON") and params is None
            assert len(url.split("/compound/inchikey/")[1].split("/")[0].split(",")) <= 20
        assert set(res) == set(keys)
        statuses = [r["status"] for r in res.values()]
        assert statuses.count("matched") == 47
        assert statuses.count("ambiguous") == 2
        assert statuses.count("no_match") == 1
        assert res["FLVRDMUHUXVRET-UHFFFAOYSA-N"]["cid"] == 54682505
        amb = res["GBXSMTUPTTWBMN-UAGQMJEPSA-N"]
        assert amb["cid"] is None and amb["cids"] == [5362032, 25322997]
        assert res["CFBUZOUXXHZCFB-LDTOLXSISA-N"]["status"] == "no_match"
        expected = {r["InChIKey"]: r["CID"] for r in rows}
        for key, r in res.items():
            if r["status"] == "matched":
                assert r["cid"] == expected[key]

    def test_unpatched_404_chunk_is_no_match(self, annotator):
        body = load("pubchem/inchikey_notfound_404.json")
        with patch.object(mod.requests, "get", return_value=FakeResponse(404, body)):
            res = annotator.resolve_pubchem_cids_by_inchikey(["ZZZZZZZZZZZZZZ-UHFFFAOYSA-N"])
        assert res["ZZZZZZZZZZZZZZ-UHFFFAOYSA-N"]["status"] == "no_match"

    def test_fault_body_returned_as_data_is_no_match(self, annotator):
        fake = FakeHelpers().add(lambda u, p: True, load("pubchem/inchikey_notfound_404.json"))
        with patch.object(mod, "_get_json", fake):
            res = annotator.resolve_pubchem_cids_by_inchikey(["ZZZZZZZZZZZZZZ-UHFFFAOYSA-N"])
        assert res["ZZZZZZZZZZZZZZ-UHFFFAOYSA-N"]["status"] == "no_match"

    def test_key_absent_from_answer_is_no_match(self, annotator):
        fake = FakeHelpers().add(lambda u, p: True, load("pubchem/inchikey_property_mixed.json"))
        with patch.object(mod, "_get_json", fake):
            res = annotator.resolve_pubchem_cids_by_inchikey([ASPIRIN_IK, "ZZZZZZZZZZZZZZ-UHFFFAOYSA-N"])
        assert res[ASPIRIN_IK] == {"status": "matched", "cid": 2244, "cids": [2244], "method": "pubchem_inchikey"}
        assert res["ZZZZZZZZZZZZZZ-UHFFFAOYSA-N"]["status"] == "no_match"

    def test_failed_batch_marks_only_its_keys_batch_failed(self, annotator):
        keys = load("pubchem/inchikey_property_50_request.txt").split(",")
        rows = load("pubchem/inchikey_property_50.json")["PropertyTable"]["Properties"]
        second_chunk = set(keys[25:])
        answer = self._recorded_answer(rows)

        def fake_get(url, params=None, timeout=None):
            requested = set(url.split("/compound/inchikey/")[1].split("/")[0].split(","))
            if requested & second_chunk:
                raise requests.ConnectionError("connection reset")
            payload = answer(url, params)
            return FakeResponse(200, payload) if payload else FakeResponse(404, {})

        with patch.object(mod.requests, "get", side_effect=fake_get):
            res = annotator.resolve_pubchem_cids_by_inchikey(keys, chunk_size=25)
        for key in keys[25:]:
            assert res[key]["status"] == "batch_failed"
            assert "ConnectionError" in res[key]["error"]
        assert {res[k]["status"] for k in keys[:25]} <= {"matched", "ambiguous", "no_match"}
        assert all(res[k]["status"] != "batch_failed" for k in keys[:25])

    def test_patched_failure_propagates(self, annotator):
        fake = FakeHelpers().add(lambda u, p: True, RuntimeError("transient_error"))
        with patch.object(mod, "_get_json", fake), pytest.raises(RuntimeError):
            annotator.resolve_pubchem_cids_by_inchikey([ASPIRIN_IK])

    def test_invalid_key_not_sent(self, annotator):
        with patch.object(mod, "_get_json", no_network):
            res = annotator.resolve_pubchem_cids_by_inchikey(["not-a-key"])
        assert res["not-a-key"]["status"] == "no_match"
        assert res["not-a-key"]["reason"] == "invalid_inchikey"

    def test_single_inchikey_identifier_uses_batch_route(self, annotator):
        rows = load("pubchem/inchikey_property_50.json")["PropertyTable"]["Properties"]
        fake = FakeHelpers().add(lambda u, p: "/compound/inchikey/" in u, self._recorded_answer(rows))
        with patch.object(mod, "_get_json", fake):
            res = annotator.resolve_pubchem_cid("GBXSMTUPTTWBMN-UAGQMJEPSA-N")
        assert res["status"] == "ambiguous" and res["method"] == "pubchem_inchikey"


class TestPubChemXrefsBatch:
    def test_batch_xrefs(self, annotator):
        fake = FakeHelpers().add(lambda u, p: "/xrefs/RegistryID/" in u, load("pubchem/xrefs_registry_3cids.json"))
        with patch.object(mod, "_get_json", fake):
            res = annotator.get_pubchem_xrefs_batch([2244, "449459", 170000000, 99999999999])
        assert fake.urls() == [f"{PUBCHEM}/compound/cid/2244,449459,170000000,99999999999/xrefs/RegistryID/JSON"]
        assert res[2244]["status"] == "success"
        assert res[2244]["chebi_ids"] == ["CHEBI:15365"]
        assert res[2244]["chembl_ids"] == ["CHEMBL25"]
        assert res[449459]["chebi_ids"] == ["CHEBI:44616"]
        assert res[170000000]["status"] == "success" and res[170000000]["chebi_ids"] == []
        assert res[99999999999]["status"] == "empty_result"

    def test_chunking_and_failure(self, annotator):
        def fake_get(url, params=None, timeout=None):
            if "/cid/3,4/" in url:
                return FakeResponse(503, {"Fault": {"Code": "PUGREST.ServerBusy", "Message": "busy"}})
            return FakeResponse(404, load("pubchem/assay_notfound_404.json"))

        with patch.object(mod.requests, "get", side_effect=fake_get):
            res = annotator.get_pubchem_xrefs_batch([1, 2, 3, 4, 0], chunk_size=2)
        assert res[1]["status"] == res[2]["status"] == "empty_result"
        assert res[3]["status"] == res[4]["status"] == "batch_failed"
        assert res[0]["status"] == "invalid_cid"


# =====================================================================
# Cross-references (PubChem first, ChEMBL last and non-fatal)
# =====================================================================


def _xref_routes(chembl_response):
    xrefs = load("pubchem/xrefs_registry_3cids.json")
    aspirin_info = {"InformationList": {"Information": [xrefs["InformationList"]["Information"][0]]}}
    return FakeHelpers([
        (_smiles_route, {"IdentifierList": {"CID": [2244]}}),
        (lambda u, p: u.endswith("/cid/2244/xrefs/RegistryID/JSON"), aspirin_info),
        (lambda u, p: u.endswith("/cid/2244/property/InChIKey/JSON"),
         {"PropertyTable": {"Properties": [{"CID": 2244, "InChIKey": ASPIRIN_IK}]}}),
        (lambda u, p: "chembl/api/data/molecule.json" in u, chembl_response),
    ])


class TestCrossReferences:
    def test_get_cross_references(self, annotator):
        fake = _xref_routes(load("chembl/molecule_by_inchikey_aspirin.json"))
        with patch.object(mod, "_get_json", fake):
            refs = annotator.get_cross_references(ASPIRIN)
        assert refs["status"] == "success"
        assert refs["pubchem_cid"] == 2244
        assert refs["chebi_id"] == "CHEBI:15365"
        assert refs["drugbank_id"] == "DB00945"
        assert refs["hmdb_id"] == "HMDB0001879"
        assert refs["chembl_ids_pubchem"] == ["CHEMBL25"]
        assert refs["inchikey"] == ASPIRIN_IK
        assert refs["chembl_id"] == "CHEMBL25"
        assert "kegg_compound_id" not in refs  # never guessed from RegistryIDs
        # ChEMBL ran last, by exact standard InChIKey, never flexmatch.
        url, params = fake.calls[-1]
        assert "molecule.json" in url
        assert params["molecule_structures__standard_inchi_key"] == ASPIRIN_IK
        assert not any("flexmatch" in k for k in params)

    def test_chembl_failure_is_partial_not_fatal(self, annotator):
        # The unpatched helper's legacy convention: it records the failure and returns None.
        def chembl_503(url, params):
            mod._record_http(url, 503, "HTTPError: 503 Server Error")

        fake = _xref_routes(chembl_503)
        with patch.object(mod, "_get_json", fake):
            refs = annotator.get_cross_references(ASPIRIN)
        assert refs["status"] == "partial"
        assert refs["chembl_id"] is None and refs["chembl_status"] == "failed"
        assert refs["pubchem_cid"] == 2244 and refs["chebi_id"] == "CHEBI:15365"
        assert any("503" in e for e in refs["errors"])

    def test_chembl_typed_error_is_partial(self, annotator):
        fake = _xref_routes(mod.ChEMBLServiceError(503))
        with patch.object(mod, "_get_json", fake):
            refs = annotator.get_cross_references(ASPIRIN)
        assert refs["status"] == "partial" and refs["chembl_id"] is None
        assert any("ChEMBLServiceError" in e for e in refs["errors"])

    def test_patched_helper_exception_propagates(self, annotator):
        class NotAttempted(RuntimeError):
            pass

        fake = _xref_routes(NotAttempted("circuit_open"))
        with patch.object(mod, "_get_json", fake), pytest.raises(NotAttempted):
            annotator.get_cross_references(ASPIRIN)

    def test_chembl_row_with_other_key_is_rejected(self, annotator):
        other = load("chembl/molecule_by_inchikey_aspirin.json")
        other["molecules"][0]["molecule_structures"]["standard_inchi_key"] = IMATINIB_IK
        fake = _xref_routes(other)
        with patch.object(mod, "_get_json", fake):
            refs = annotator.get_cross_references(ASPIRIN)
        assert refs["chembl_id"] is None and refs["status"] == "success"

    def test_given_ids_skip_requests(self, annotator):
        rids = load("pubchem/xrefs_registry_3cids.json")["InformationList"]["Information"][0]["RegistryID"]
        with patch.object(mod, "_get_json", no_network):
            refs = annotator.get_cross_references(
                ASPIRIN, cid=2244, registry_ids=rids, inchikey=ASPIRIN_IK, chembl_id="CHEMBL25",
            )
        assert refs["pubchem_cid"] == 2244 and refs["chembl_id"] == "CHEMBL25"
        assert refs["chebi_id"] == "CHEBI:15365" and refs["status"] == "success"

    def test_include_chembl_false(self, annotator):
        fake = _xref_routes(AssertionError("ChEMBL must not be called"))
        with patch.object(mod, "_get_json", fake):
            refs = annotator.get_cross_references(ASPIRIN, include_chembl=False)
        assert "chembl_id" not in refs and refs["status"] == "success"
        assert not any("chembl" in u for u in fake.urls())

    def test_unresolved_cid_reports_status(self, annotator):
        fake = FakeHelpers().add(_smiles_route, load("pubchem/smiles_cids_zero.json"))
        with patch.object(mod, "_get_json", fake):
            refs = annotator.get_cross_references(ASPIRIN)
        assert refs["pubchem_cid"] is None and refs["status"] == "no_match"


# =====================================================================
# Assay-name disease keywords (T18)
# =====================================================================


def _assay_route(payload):
    return FakeHelpers().add(lambda u, p: u.endswith("/assaysummary/JSON"), payload)


class TestDiseaseAssociations:
    def test_inactive_rows_never_count(self, annotator):
        table = load("pubchem/assaysummary_74603.json")
        cols = table["Table"]["Columns"]["Column"]
        names = [r["Cell"][cols.index("Assay Name")].lower() for r in table["Table"]["Row"]
                 if r["Cell"][cols.index("Activity Outcome")] != "Active"]
        assert any("cancer" in n or "leukemia" in n for n in names)  # the trap is real
        fake = _assay_route(table)
        with patch.object(mod, "_get_json", fake):
            assert annotator.get_disease_associations("ignored", cid=74603) == []
        assert fake.urls() == [f"{PUBCHEM}/compound/cid/74603/assaysummary/JSON"]

    def test_active_rows_scanned_and_columns_found_by_name(self, annotator):
        table = load("pubchem/assaysummary_2244_trimmed.json")
        fake = _assay_route(table)
        with patch.object(mod, "_get_json", fake):
            hits = annotator.get_disease_associations(ASPIRIN, cid=2244)
        by_kw = {h["keyword"]: h for h in hits}
        assert set(by_kw) == {"cancer"}
        assert by_kw["cancer"]["disease"] == "cancer"
        assert by_kw["cancer"]["n_active_assays"] == 3
        assert by_kw["cancer"]["source"] == "pubchem_assay_name_keyword"
        assert "pediatric cancer" in by_kw["cancer"]["assay"]

        # Reordering the columns must not change the answer.
        cols = table["Table"]["Columns"]["Column"]
        order = list(reversed(range(len(cols))))
        table["Table"]["Columns"]["Column"] = [cols[i] for i in order]
        for row in table["Table"]["Row"]:
            row["Cell"] = [row["Cell"][i] for i in order]
        with patch.object(mod, "_get_json", _assay_route(table)):
            assert annotator.get_disease_associations(ASPIRIN, cid=2244) == hits

    def test_not_found(self, annotator):
        with patch.object(mod.requests, "get",
                          return_value=FakeResponse(404, load("pubchem/assay_notfound_404.json"))):
            assert annotator.get_disease_associations("x", cid=1) == []

    @pytest.mark.parametrize("failure", [
        FakeResponse(503, {"Fault": {"Code": "PUGREST.ServerBusy", "Message": "Too many requests"}}),
        requests.ConnectionError("connection reset"),
    ])
    def test_failed_request_is_not_an_empty_answer(self, annotator, failure):
        kwargs = {"side_effect": failure} if isinstance(failure, BaseException) else {"return_value": failure}
        with patch.object(mod.requests, "get", **kwargs), pytest.raises(mod.AnnotationServiceError) as err:
            annotator.get_disease_associations("x", cid=1)
        assert err.value.service == "pubchem"

    def test_patched_helper_exception_propagates(self, annotator):
        class NotAttempted(RuntimeError):
            pass

        with patch.object(mod, "_get_json", _assay_route(NotAttempted("deadline"))), pytest.raises(NotAttempted):
            annotator.get_disease_associations("x", cid=1)

    def test_annotate_reports_unknown_not_empty(self, annotator):
        with patch.object(annotator, "resolve_pubchem_cid",
                          return_value={"status": "matched", "cid": 1, "cids": [1], "method": "smiles"}), \
                patch.object(mod.requests, "get", side_effect=requests.ConnectionError("reset")):
            result = annotator.annotate(ASPIRIN, sources=["diseases"])
        assert result["disease_associations"] is None
        assert "pubchem" in result["disease_associations_error"]


# =====================================================================
# ChEBI (T13 package level; offline index faked, online from recordings)
# =====================================================================


class FakeChebiIndex:
    release = "chebi/257"

    def __init__(self, table):
        self.table = table
        self.queries: list[str] = []

    def lookup(self, inchikey):
        self.queries.append(inchikey)
        return self.table.get(inchikey, {"status": "no_chebi_entity", "inchikey": inchikey})


def _chebi_online_routes(es: dict[str, object], compounds_fixture: str = "chebi/compounds_mixed.json"):
    recorded = {**load("chebi/compounds_aspirin.json"), **load(compounds_fixture)}

    def es_search(url, params):
        return es[params["term"]]

    def compounds(url, params):
        ids = params["chebi_ids"].split(",")
        return {i: recorded[i] for i in ids if i in recorded}

    return FakeHelpers([
        (lambda u, p: u == f"{CHEBI_REST}/es_search/", es_search),
        (lambda u, p: u == f"{CHEBI_REST}/compounds/", compounds),
    ])


class TestChebiRolesOffline:
    def test_index_lookup_uses_rdkit_inchikey_and_no_network(self, annotator):
        idx = FakeChebiIndex({ASPIRIN_IK: {
            "status": "success", "chebi_id": "CHEBI:15365", "name": "acetylsalicylic acid", "stars": 3,
            "match_tier": "exact", "inchikey": ASPIRIN_IK,
            "roles": [{"chebi_id": "CHEBI:35475", "name": "non-steroidal anti-inflammatory drug"}],
            "candidates": [], "release": "chebi/257",
        }})
        with patch.object(mod, "_get_json", no_network):
            res = annotator.lookup_chebi_roles(ASPIRIN, index=idx)
        assert idx.queries == [ASPIRIN_IK]
        assert res["status"] == "success" and res["match_tier"] == "exact"
        assert res["roles"][0]["chebi_id"] == "CHEBI:35475"

    def test_defaults_filled_and_release_from_index(self, annotator):
        idx = FakeChebiIndex({})
        res = annotator.lookup_chebi_roles(inchikey=ASPIRIN_IK.lower(), index=idx)
        assert idx.queries == [ASPIRIN_IK]
        assert res["status"] == "no_chebi_entity"
        assert res["release"] == "chebi/257"
        assert set(res) >= {"status", "chebi_id", "name", "stars", "match_tier", "inchikey",
                            "roles", "candidates", "release"}

    def test_invalid_structure(self, annotator):
        idx = FakeChebiIndex({})
        with patch.object(mod, "_get_json", no_network):
            res = annotator.lookup_chebi_roles("C1CC(", index=idx, online=False)
        assert res["status"] == "invalid_structure"
        assert idx.queries == []

    def test_constructor_index_given_as_path_loads_lazily(self, monkeypatch, tmp_path):
        loaded = []

        class FakeObo(FakeChebiIndex):
            @classmethod
            def from_obo(cls, path):
                loaded.append(path)
                return cls({})

        fake_module = types.ModuleType("embpy.resources.molecule.chebi_obo")
        fake_module.ChebiOboIndex = FakeObo
        monkeypatch.setitem(sys.modules, "embpy.resources.molecule.chebi_obo", fake_module)
        path = tmp_path / "chebi_core.obo.gz"
        ann = MoleculeAnnotator(rate_limit_delay=0, chebi_index=path)
        ann.lookup_chebi_roles(inchikey=ASPIRIN_IK)
        ann.lookup_chebi_roles(inchikey=ASPIRIN_IK)
        assert loaded == [os.fspath(path)]  # loaded once, then reused

    def test_annotate_ontology_uses_index(self):
        idx = FakeChebiIndex({ASPIRIN_IK: {"status": "matched_no_role", "match_tier": "exact"}})
        ann = MoleculeAnnotator(rate_limit_delay=0, chebi_index=idx)
        with patch.object(mod, "_get_json", no_network):
            result = ann.annotate(ASPIRIN, sources=["ontology"])
        assert result["chebi_roles"]["status"] == "matched_no_role"


class TestChebiRolesOnline:
    def test_exact_with_roles(self, annotator):
        fake = _chebi_online_routes({ASPIRIN_IK: load("chebi/es_aspirin.json")})
        with patch.object(mod, "_get_json", fake):
            res = annotator.lookup_chebi_roles(ASPIRIN)
        assert res["status"] == "success"
        assert res["chebi_id"] == "CHEBI:15365" and res["stars"] == 3
        assert res["match_tier"] == "exact" and res["inchikey"] == ASPIRIN_IK
        assert {"chebi_id": "CHEBI:35475", "name": "non-steroidal anti-inflammatory drug"} in res["roles"]
        assert len(res["roles"]) == 35
        assert fake.calls[-1] == (f"{CHEBI_REST}/compounds/", {"chebi_ids": "CHEBI:15365"})
        assert not any("pubchem" in u or "ols4" in u for u in fake.urls())

    def test_exact_two_star_without_role(self, annotator):
        key = "NKANXQFJJICGDU-UHFFFAOYSA-N"  # stereo-unspecified tamoxifen
        fake = _chebi_online_routes({key: load("chebi/es_tamoxifen_exact.json")})
        with patch.object(mod, "_get_json", fake):
            res = annotator.lookup_chebi_roles(inchikey=key)
        assert res["status"] == "matched_no_role"
        assert res["chebi_id"] == "CHEBI:92766" and res["stars"] == 2
        assert res["match_tier"] == "exact" and res["roles"] == []

    def test_connectivity_single_candidate_is_flagged(self, annotator):
        key = "NFDFQCUYFHCNBW-ZZZZZZZZSA-N"  # same connectivity as dienestrol
        fake = _chebi_online_routes({
            key: load("chebi/es_404_no_match.json"),
            "NFDFQCUYFHCNBW": load("chebi/es_dienestrol_block.json"),
        })
        with patch.object(mod, "_get_json", fake):
            res = annotator.lookup_chebi_roles(inchikey=key)
        assert res["status"] == "success"
        assert res["match_tier"] == "connectivity"
        assert res["chebi_id"] == "CHEBI:4518"
        assert res["query_inchikey"] == key
        assert res["inchikey"] == "NFDFQCUYFHCNBW-UHFFFAOYSA-N"  # the matched entity's key
        assert res["roles"] == [{"chebi_id": "CHEBI:76988", "name": "xenoestrogen"}]

    def test_connectivity_with_three_stereo_variants_is_ambiguous(self, annotator):
        key = "GBABOYUKABKIAF-JKVZWNDSSA-N"
        fake = _chebi_online_routes({
            key: load("chebi/es_404_no_match.json"),
            "GBABOYUKABKIAF": load("chebi/es_vinorelbine_block.json"),
        })
        with patch.object(mod, "_get_json", fake):
            res = annotator.lookup_chebi_roles(inchikey=key)
        assert res["status"] == "ambiguous_connectivity"
        assert res["chebi_id"] is None and res["roles"] == []
        assert len(res["candidates"]) == 3
        assert not any(u.endswith("/compounds/") for u in fake.urls())

    def test_unpatched_404_means_no_entity(self, annotator):
        body = load("chebi/es_404_no_match.json")
        with patch.object(mod.requests, "get", return_value=FakeResponse(404, body)):
            res = annotator.lookup_chebi_roles(inchikey="GBABOYUKABKIAF-JKVZWNDSSA-N")
        assert res["status"] == "no_chebi_entity"

    def test_unpatched_failure_is_service_error(self, annotator):
        with patch.object(mod.requests, "get", side_effect=requests.Timeout("read timeout")):
            res = annotator.lookup_chebi_roles(inchikey=ASPIRIN_IK)
        assert res["status"] == "service_error"
        assert "Timeout" in res["error"]

    def test_offline_without_index(self, annotator):
        with patch.object(mod, "_get_json", no_network):
            res = annotator.lookup_chebi_roles(inchikey=ASPIRIN_IK, online=False)
        assert res["status"] == "service_error"

    def test_compounds_secondary_and_missing_ids(self, annotator):
        fake = _chebi_online_routes({})
        with patch.object(mod, "_get_json", fake):
            res = annotator.get_chebi_compounds(["CHEBI:2890", "999999999", "CHEBI:4518"])
        assert fake.calls == [(f"{CHEBI_REST}/compounds/", {"chebi_ids": "CHEBI:2890,CHEBI:999999999,CHEBI:4518"})]
        assert res["CHEBI:2890"]["status"] == "success"
        assert res["CHEBI:2890"]["chebi_id"] == "CHEBI:15365"  # secondary -> primary
        assert res["CHEBI:999999999"]["status"] == "not_found"
        assert res["CHEBI:4518"]["inchikey"] == "NFDFQCUYFHCNBW-UHFFFAOYSA-N"

    def test_compounds_batches_of_at_most_40(self, annotator):
        fake = FakeHelpers().add(lambda u, p: True, {})
        with patch.object(mod, "_get_json", fake):
            res = annotator.get_chebi_compounds([f"CHEBI:{i}" for i in range(1, 90)], chunk_size=100)
        assert [len(p["chebi_ids"].split(",")) for _, p in fake.calls] == [40, 40, 9]
        assert {r["status"] for r in res.values()} == {"not_found"}


class TestChebiRolesWithRealOboIndex:
    """The package path against the real ChebiOboIndex on its mini OBO fixture.

    Also checks that the online path (recorded ChEBI REST payloads) reaches
    the same status and entity for the same structures.
    """

    MINI_OBO = Path(__file__).parent / "fixtures" / "chebi" / "chebi_core_257_mini.obo"

    @pytest.fixture
    def obo_index(self):
        chebi_obo = pytest.importorskip("embpy.resources.molecule.chebi_obo")
        if not self.MINI_OBO.exists():
            pytest.skip("mini OBO fixture not present")
        return chebi_obo.ChebiOboIndex.from_obo(self.MINI_OBO)

    @pytest.mark.parametrize(
        ("query", "es", "status", "chebi_id", "tier"),
        [
            (ASPIRIN, {ASPIRIN_IK: "chebi/es_aspirin.json"}, "success", "CHEBI:15365", "exact"),
            ("NKANXQFJJICGDU-UHFFFAOYSA-N", {"NKANXQFJJICGDU-UHFFFAOYSA-N": "chebi/es_tamoxifen_exact.json"},
             "matched_no_role", "CHEBI:92766", "exact"),
            ("GBABOYUKABKIAF-JKVZWNDSSA-N", {"GBABOYUKABKIAF-JKVZWNDSSA-N": "chebi/es_404_no_match.json",
                                             "GBABOYUKABKIAF": "chebi/es_vinorelbine_block.json"},
             "ambiguous_connectivity", None, "connectivity"),
        ],
    )
    def test_offline_and_online_agree(self, obo_index, query, es, status, chebi_id, tier):
        ann = MoleculeAnnotator(rate_limit_delay=0, chebi_index=obo_index)
        with patch.object(mod, "_get_json", no_network):
            offline = ann.lookup_chebi_roles(query)
        assert (offline["status"], offline["chebi_id"], offline["match_tier"]) == (status, chebi_id, tier)
        assert offline["release"]

        fake = _chebi_online_routes({k: load(v) for k, v in es.items()})
        with patch.object(mod, "_get_json", fake):
            online = MoleculeAnnotator(rate_limit_delay=0).lookup_chebi_roles(query)
        assert (online["status"], online["chebi_id"], online["match_tier"]) == (status, chebi_id, tier)
        if status == "success":
            offline_roles = {r["chebi_id"] for r in offline["roles"]}
            assert offline_roles and offline_roles <= {r["chebi_id"] for r in online["roles"]}


class TestChebiTerms:
    def test_terms_from_given_ids(self, annotator):
        fake = FakeHelpers().add(lambda u, p: u == mod.OLS_CHEBI_TERMS, load("chebi/ols_terms_aspirin.json"))
        with patch.object(mod, "_get_json", fake):
            terms = annotator.get_chebi_terms(ASPIRIN, chebi_ids=["CHEBI:15365"])
        assert terms[0]["chebi_id"] == "CHEBI:15365"
        assert terms[0]["name"] == "acetylsalicylic acid"
        assert fake.calls == [(mod.OLS_CHEBI_TERMS, {"short_form": "CHEBI_15365"})]

    def test_terms_with_cid_skip_resolution(self, annotator):
        xrefs = load("pubchem/xrefs_registry_3cids.json")
        aspirin_info = {"InformationList": {"Information": [xrefs["InformationList"]["Information"][0]]}}
        fake = FakeHelpers([
            (lambda u, p: u.endswith("/cid/2244/xrefs/RegistryID/JSON"), aspirin_info),
            (lambda u, p: u == mod.OLS_CHEBI_TERMS, load("chebi/ols_terms_aspirin.json")),
        ])
        with patch.object(mod, "_get_json", fake):
            terms = annotator.get_chebi_terms("ignored", cid=2244)
        assert [t["chebi_id"] for t in terms] == ["CHEBI:15365"]
        assert not any("/compound/smiles/" in u for u in fake.urls())

    def test_failed_xref_request_raises(self, annotator):
        with patch.object(mod.requests, "get", side_effect=requests.ConnectionError("reset")), \
                pytest.raises(mod.AnnotationServiceError):
            annotator.get_chebi_terms("ignored", cid=2244)

    def test_failed_ols_request_raises(self, annotator):
        def ols_500(url, params):
            mod._record_http(url, 500, "HTTPError: 500")

        fake = FakeHelpers().add(lambda u, p: u == mod.OLS_CHEBI_TERMS, ols_500)
        with patch.object(mod, "_get_json", fake), pytest.raises(mod.AnnotationServiceError) as err:
            annotator.get_chebi_terms(ASPIRIN, chebi_ids=["CHEBI:15365"])
        assert err.value.service == "ols" and err.value.status_code == 500

    def test_get_chebi_roles_is_a_deprecated_alias(self, annotator):
        xrefs = load("pubchem/xrefs_registry_3cids.json")
        aspirin_info = {"InformationList": {"Information": [xrefs["InformationList"]["Information"][0]]}}
        fake = FakeHelpers([
            (lambda u, p: u.endswith("/cid/2244/xrefs/RegistryID/JSON"), aspirin_info),
            (lambda u, p: u == mod.OLS_CHEBI_TERMS, load("chebi/ols_terms_aspirin.json")),
        ])
        with patch.object(mod, "_get_json", fake), pytest.deprecated_call():
            legacy = annotator.get_chebi_roles(ASPIRIN, cid=2244)
        with patch.object(mod, "_get_json", fake):
            assert legacy == annotator.get_chebi_terms(ASPIRIN, cid=2244)
        assert legacy[0]["name"] == "acetylsalicylic acid"

    def test_no_soap_endpoint_left(self):
        assert not hasattr(mod, "CHEBI_API")
        assert "webservices/chebi" not in Path(mod.__file__).read_text()


# =====================================================================
# KEGG (T12 package level; KeggIndex faked from recorded KEGG/PubChem rows)
# =====================================================================


class FakeKeggIndex:
    """Mimics a KeggIndex built from the recorded rows below.

    KEGG ``conv/compound/pubchem`` and ``conv/drug/pubchem`` give *SIDs*
    (``pubchem:4594 -> cpd:C01405``, ``pubchem:7847177 -> dr:D00109``);
    PubChem ``substance/sid/4594/cids`` gives CID 2244, whose InChIKey is
    aspirin's. The index is therefore keyed by InChIKey.
    """

    release = "kegg 117.0+/11-11"
    CONV_ROWS = ["pubchem:4594\tcpd:C01405", "pubchem:7847177\tdr:D00109"]
    SID_TO_CID = {4594: 2244, 7847177: 2244}
    CID_TO_INCHIKEY = {2244: ASPIRIN_IK}

    def __init__(self):
        self.queries: list[str] = []
        by_key: dict[str, list[dict]] = {}
        for row in self.CONV_ROWS:
            sid, kegg = row.split("\t")
            cid = self.SID_TO_CID[int(sid.split(":")[1])]
            ns, kegg_id = kegg.split(":")
            by_key.setdefault(self.CID_TO_INCHIKEY[cid], []).append(
                {"kegg_id": kegg_id, "namespace": ns, "match_tier": "exact"})
        self.entries = by_key

    def lookup(self, inchikey):
        self.queries.append(inchikey)
        if inchikey == IMATINIB_IK:
            return {"status": "identity_mismatch", "entries": [], "release": self.release}
        entries = self.entries.get(inchikey)
        if not entries:
            return {"status": "not_in_kegg", "entries": [], "compound_pathways": [], "drug_maps": [],
                    "target_pathways": [], "release": self.release}
        return {
            "status": "success", "entries": entries,
            "compound_pathways": [{"id": "map00627", "name": "Aminobenzoate degradation"}],
            "drug_maps": [{"id": "map07112", "name": "Antiinflammatory drugs"}],
            "target_pathways": [{"id": "hsa00590", "name": "Arachidonic acid metabolism"},
                                {"id": "map00627", "name": "duplicate id, other kind"}],
            "release": self.release,
        }


class TestKegg:
    def test_lookup_by_inchikey_never_by_cid(self, annotator):
        idx = FakeKeggIndex()
        recorder = FakeHelpers()  # any request at all fails the test
        with patch.object(mod, "_get_json", recorder), patch.object(mod, "_get_text", recorder):
            res = annotator.lookup_kegg(ASPIRIN, index=idx)
        assert recorder.calls == []
        assert idx.queries == [ASPIRIN_IK]
        assert res["status"] == "success" and res["inchikey"] == ASPIRIN_IK
        assert {(e["kegg_id"], e["namespace"], e["match_tier"]) for e in res["entries"]} == {
            ("C01405", "cpd", "exact"), ("D00109", "dr", "exact")}
        # compound, drug-map and target pathways stay separate fields
        assert [p["id"] for p in res["compound_pathways"]] == ["map00627"]
        assert [p["id"] for p in res["drug_maps"]] == ["map07112"]
        assert [p["id"] for p in res["target_pathways"]] == ["hsa00590", "map00627"]

    def test_cid_5291_never_reaches_kegg_pubchem_namespace(self, annotator):
        # The old code sent CID 5291 (imatinib) as pubchem:5291 and got C02226.
        idx = FakeKeggIndex()
        old_answer = FakeHelpers().add(lambda u, p: "conv/compound/pubchem:" in u, "pubchem:5291\tcpd:C02226")
        with patch.object(mod, "_get_text", old_answer), patch.object(mod, "_get_json", old_answer):
            res = annotator.lookup_kegg(IMATINIB, index=idx)
            flat = annotator.get_kegg_pathways(IMATINIB, index=idx)
        assert old_answer.calls == []
        assert "C02226" not in json.dumps(res)
        assert res["status"] == "identity_mismatch" and flat == []

    def test_missing_index_is_index_required_not_empty(self, annotator, caplog):
        with patch.object(mod, "_get_json", no_network), patch.object(mod, "_get_text", no_network):
            res = annotator.lookup_kegg(ASPIRIN)
            assert annotator.get_kegg_pathways(ASPIRIN) == []
        assert res["status"] == "index_required"
        assert "KEGG index" in caplog.text

    def test_get_kegg_pathways_flattens_with_kind(self, annotator):
        flat = annotator.get_kegg_pathways(ASPIRIN, index=FakeKeggIndex())
        assert [(p["pathway_id"], p["kind"]) for p in flat] == [
            ("map00627", "compound_pathway"), ("map07112", "drug_map"),
            ("hsa00590", "target_pathway"), ("map00627", "target_pathway")]
        assert flat[0]["pathway_name"] == "Aminobenzoate degradation"

    def test_not_in_kegg_and_invalid_structure(self, annotator):
        idx = FakeKeggIndex()
        assert annotator.lookup_kegg(inchikey="ZZZZZZZZZZZZZZ-UHFFFAOYSA-N", index=idx)["status"] == "not_in_kegg"
        res = annotator.lookup_kegg(inchikey="garbage", index=idx)
        assert res["status"] == "invalid_structure"

    def test_index_dir_loaded_lazily(self, monkeypatch, tmp_path):
        loaded = []

        class FakeKeggDir(FakeKeggIndex):
            @classmethod
            def from_dir(cls, path):
                loaded.append(path)
                return cls()

        fake_module = types.ModuleType("embpy.resources.molecule.kegg_index")
        fake_module.KeggIndex = FakeKeggDir
        monkeypatch.setitem(sys.modules, "embpy.resources.molecule.kegg_index", fake_module)
        ann = MoleculeAnnotator(rate_limit_delay=0, kegg_index=tmp_path)
        assert ann.lookup_kegg(inchikey=ASPIRIN_IK)["status"] == "success"
        assert ann.annotate(ASPIRIN, sources=["pathways"])["kegg"]["status"] == "success"
        assert loaded == [os.fspath(tmp_path)]


# =====================================================================
# Annotate (aggregation)
# =====================================================================


class TestAnnotate:
    def test_structural_only(self, annotator):
        result = annotator.annotate("CCO", sources="structural")
        assert "physicochemical" in result
        assert result["canonical_smiles"] is not None

    def test_unknown_molecule(self, annotator):
        result = annotator.annotate("TOTALLY_FAKE_MOLECULE_12345", sources="structural")
        assert "physicochemical" in result or "error" in result

    def test_pathways_without_index(self, annotator):
        with patch.object(mod, "_get_json", no_network), patch.object(mod, "_get_text", no_network):
            result = annotator.annotate(ASPIRIN, sources=["pathways"])
        assert result["kegg"]["status"] == "index_required"
        assert result["kegg_pathways"] == []

    def test_xrefs_and_diseases_share_one_cid_lookup(self, annotator):
        fake = _xref_routes(load("chembl/molecule_by_inchikey_aspirin.json"))
        fake.add(lambda u, p: u.endswith("/assaysummary/JSON"), load("pubchem/assaysummary_2244_trimmed.json"))
        with patch.object(mod, "_get_json", fake):
            result = annotator.annotate(ASPIRIN, sources=["xrefs", "diseases"])
        assert sum(1 for u in fake.urls() if "/compound/smiles/" in u) == 1
        assert result["cross_references"]["pubchem_cid"] == 2244
        assert result["disease_associations"][0]["keyword"] == "cancer"


# =====================================================================
# annotate_adata
# =====================================================================


class TestAnnotateAdata:
    def test_annotate_adata(self, annotator):
        import pandas as pd
        from anndata import AnnData

        adata = AnnData(
            obs=pd.DataFrame({"drug": ["CCO", "c1ccccc1", "CCO"]}),
        )
        adata.obs.index = [f"cell_{i}" for i in range(3)]

        result = annotator.annotate_adata(
            adata, column="drug", sources="structural",
        )
        assert "mol_molecular_weight" in result.obs.columns
        assert "mol_logp" in result.obs.columns
        assert "molecule_annotations" in result.uns

    def test_missing_column_raises(self, annotator):
        from anndata import AnnData

        adata = AnnData()
        with pytest.raises(ValueError, match="not found"):
            annotator.annotate_adata(adata, column="nonexistent")


# =====================================================================
# Opt-in live check of provider top-level keys
# =====================================================================


@pytest.mark.skipif(os.environ.get("EMBPY_NETWORK_TESTS") != "1",
                    reason="live provider check; set EMBPY_NETWORK_TESTS=1")
class TestLiveProviderShapes:
    def test_pubchem_and_chebi_top_level_keys(self):
        ann = MoleculeAnnotator(rate_limit_delay=0.5)
        res = ann.resolve_pubchem_cids_by_inchikey([ASPIRIN_IK])
        assert res[ASPIRIN_IK]["cid"] == 2244
        assert ann.resolve_pubchem_cid(HYDROXYTAMOXIFEN)["cid"] == 449459
        xrefs = ann.get_pubchem_xrefs_batch([2244])[2244]
        assert "CHEBI:15365" in xrefs["chebi_ids"]
        roles = ann.lookup_chebi_roles(inchikey=ASPIRIN_IK)
        assert roles["status"] == "success" and roles["chebi_id"] == "CHEBI:15365"
