"""Tests for embpy.resources.molecule.kegg_index (KEGG entries keyed by InChIKey).

KEGG tables are never committed (KEGG data are for academic use only), so the
``fixtures/kegg/*.tsv`` and ``info.txt`` files are synthetic stand-ins: they
keep the recorded row format of the KEGG REST bodies (``conv``/``link``/
``list``/``info``, recorded 2026-11-11) but every KEGG entry and pathway id is
renamed into an unused range (C9xxxx, D9xxxx, map9xxxx, map077xx-map079xx,
hsa9xxxx) and every pathway name is a placeholder. The PubChem SIDs, CIDs,
InChIKeys and ChEBI ids are real, and the two PubChem JSON files are complete
recorded responses (see ``fixtures/kegg/provenance.json``). No test touches
the network except the opt-in smoke test at the end (``EMBPY_NETWORK_TESTS=1``),
which checks the real KEGG/PubChem shapes.

The regression at stake: KEGG's ``pubchem:`` namespace holds PubChem SIDs, so
the old ``conv/compound/pubchem:{CID}`` lookup mapped CID 5291 (imatinib) to
the KEGG compound whose SID is 5291 (citraconic acid; C92226 here) and CID
3413 to PQQ (SID 3413; C90113 here). Here KEGG SIDs go through PubChem's
standardized SID->CID mapping and the join is on InChIKey.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from embpy.resources.molecule import kegg_index
from embpy.resources.molecule.kegg_index import (
    INCHIKEY_COLUMNS,
    INCHIKEY_FILE,
    KEGG,
    KEGG_TABLE_FILES,
    KEGG_TABLES,
    MANIFEST_FILE,
    PUBCHEM_CID_INCHIKEY_URL,
    PUBCHEM_SID_CIDS_URL,
    KeggIndex,
    KeggReferenceError,
    build_kegg_reference,
    parse_kegg_info,
)

FIX = Path(__file__).parent / "fixtures" / "kegg"
PATH_TO_NAME = {path: name for name, path in KEGG_TABLES.items()}

# Standard InChIKeys (real, from the PubChem fixture) and the synthetic KEGG ids standing in for them.
ASPIRIN = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"  # CID 2244: C91405 (SID 4594), D90109 (SID 7847177)
IMATINIB = "KTUFNOKKBVMGRW-UHFFFAOYSA-N"  # CID 5291: D98066 (SID 96024758)
IMATINIB_MESYLATE = "YLMAHDNUQAMNNX-UHFFFAOYSA-N"  # CID 123596: D91441
DEXAMETHASONE = "UREBDLICKHMUKA-CXSFZGCWSA-N"  # CID 5743: C95643, D90292
BETAMETHASONE = "UREBDLICKHMUKA-DVTGEIKXSA-N"  # CID 9782: D90244 (same connectivity block)
STEROID_NO_STEREO = "UREBDLICKHMUKA-UHFFFAOYSA-N"  # the same skeleton with no stereo layer
CITRACONATE = "HNEGQIOMVPPMNR-IHWYPQMZSA-N"  # CID 643798: C92226 (SID 5291)
PQQ = "MMXZSJMASHPLLR-UHFFFAOYSA-N"  # CID 1024: C90113 (SID 3413), no pathway links
WATER = "XLYOFNOQVPJJNP-UHFFFAOYSA-N"  # CID 962: C90001 (SID 3303)
NOT_IN_FIXTURE = "YGSDEFSMJLZEOE-UHFFFAOYSA-N"  # salicylic acid; absent from the trimmed tables

ALL_KEYS = [
    ASPIRIN, IMATINIB, IMATINIB_MESYLATE, DEXAMETHASONE, BETAMETHASONE,
    STEROID_NO_STEREO, CITRACONATE, PQQ, WATER, NOT_IN_FIXTURE,
]


class FakeClock:
    """Monotonic fake clock; ``sleep`` advances it."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.now += seconds
        self.slept += seconds


class RecordedProviders:
    """Serves the recorded KEGG and PubChem payloads and logs every call.

    ``overrides`` maps a KEGG REST path or a PubChem URL to a replacement
    return value, or to a callable that is called instead (it may raise).
    Each call takes 50 ms of fake time.
    """

    def __init__(self, clock: FakeClock, overrides: dict | None = None) -> None:
        self.clock = clock
        self.overrides = overrides or {}
        self.calls: list[dict] = []

    def _override(self, key, *args):
        value = self.overrides[key]
        return value(*args) if callable(value) else value

    def get_text(self, url, params=None, timeout=30):
        self.calls.append({"method": "GET", "url": url, "t": self.clock()})
        self.clock.now += 0.05
        assert url.startswith(KEGG + "/"), url
        path = url[len(KEGG) + 1:]
        if path in self.overrides:
            return self._override(path, url)
        return (FIX / KEGG_TABLE_FILES[PATH_TO_NAME[path]]).read_text()

    def post_json(self, url, json_data=None, data=None, timeout=30):
        self.calls.append({"method": "POST", "url": url, "data": data, "t": self.clock()})
        self.clock.now += 0.05
        if url in self.overrides:
            return self._override(url, url, data)
        if url == PUBCHEM_SID_CIDS_URL:
            wanted = {int(x) for x in data["sid"].split(",")}
            info = json.loads((FIX / "pubchem_sid_cids_standardized.json").read_text())
            return {"InformationList": {"Information": [
                i for i in info["InformationList"]["Information"] if i["SID"] in wanted
            ]}}
        if url == PUBCHEM_CID_INCHIKEY_URL:
            wanted = {int(x) for x in data["cid"].split(",")}
            table = json.loads((FIX / "pubchem_cid_inchikey.json").read_text())
            return {"PropertyTable": {"Properties": [
                p for p in table["PropertyTable"]["Properties"] if p["CID"] in wanted
            ]}}
        raise AssertionError(f"unexpected POST {url}")

    def gets(self):
        return [c for c in self.calls if c["method"] == "GET"]

    def posts(self, url=None):
        return [c for c in self.calls if c["method"] == "POST" and (url is None or c["url"] == url)]


def _build(out_dir, *, overrides=None, **kwargs):
    clock = FakeClock()
    providers = RecordedProviders(clock, overrides)
    manifest = build_kegg_reference(
        out_dir,
        get_text=providers.get_text,
        post_json=providers.post_json,
        clock=clock,
        sleep=clock.sleep,
        **kwargs,
    )
    return manifest, providers, clock


def _map_rows(out_dir):
    lines = (Path(out_dir) / INCHIKEY_FILE).read_text().splitlines()
    return lines[0].split("\t"), [tuple(line.split("\t")) for line in lines[1:]]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("kegg_ref")
    manifest, providers, clock = _build(out)
    return out, manifest, providers, clock


@pytest.fixture(scope="module")
def index(built):
    return KeggIndex.from_dir(built[0])


def _ids(items):
    return [item["id"] for item in items]


def _entries(result):
    return {(e["kegg_id"], e["namespace"], e["match_tier"]) for e in result["entries"]}


# =====================================================================
# build_kegg_reference
# =====================================================================


class TestBuild:
    def test_requests_only_the_bulk_tables(self, built):
        _, _, providers, _ = built
        urls = [c["url"] for c in providers.gets()]
        assert sorted(urls) == sorted(f"{KEGG}/{p}" for p in KEGG_TABLES.values())
        assert len(urls) == 11
        # No per-entity KEGG call of any kind, in particular never conv/compound/pubchem:{CID}.
        assert all(":" not in url[len(KEGG) + 1:] for url in urls)

    def test_sids_go_to_the_sid_endpoint_and_cids_to_the_property_endpoint(self, built):
        _, _, providers, _ = built
        (sid_post,) = providers.posts(PUBCHEM_SID_CIDS_URL)
        (cid_post,) = providers.posts(PUBCHEM_CID_INCHIKEY_URL)
        assert "cids_type=standardized" in sid_post["url"]
        assert providers.calls.index(sid_post) < providers.calls.index(cid_post)
        sids = sorted(int(x) for x in sid_post["data"]["sid"].split(","))
        assert sids == [3303, 3413, 4594, 5291, 7847124, 7847177, 7847310, 7847358, 7848504,
                        17396633, 96024758]
        cids = [int(x) for x in cid_post["data"]["cid"].split(",")]
        # The standardized CIDs, not the SIDs: SID 4594 (aspirin) -> CID 2244, SID 5291 -> CID 643798.
        assert cids == [962, 1024, 2244, 5291, 5743, 9782, 123596, 643798]

    def test_inchikey_map(self, built):
        out, _, _, _ = built
        header, rows = _map_rows(out)
        assert tuple(header) == INCHIKEY_COLUMNS
        assert len(rows) == 11
        assert ("C90001", "cpd", "3303", "962", WATER) in rows  # SID 3303 is water, CID 962
        assert ("C91405", "cpd", "4594", "2244", ASPIRIN) in rows
        assert ("D90109", "dr", "7847177", "2244", ASPIRIN) in rows
        assert ("D98066", "dr", "96024758", "5291", IMATINIB) in rows
        assert ("C92226", "cpd", "5291", "643798", CITRACONATE) in rows
        assert ("C90113", "cpd", "3413", "1024", PQQ) in rows
        assert ("D91441", "dr", "7848504", "123596", IMATINIB_MESYLATE) in rows
        # A KEGG DRUG entry whose SID has no standardized compound keeps an empty structure.
        assert ("D90056", "dr", "7847124", "", "") in rows
        # cpd rows come before dr rows, each sorted by KEGG id.
        assert rows == sorted(rows, key=lambda r: (r[1] != "cpd", r[0]))
        assert [r[0] for r in rows][:5] == ["C90001", "C90113", "C91405", "C92226", "C95643"]

    def test_tables_written_as_received(self, built):
        out, _, _, _ = built
        for name, filename in KEGG_TABLE_FILES.items():
            assert (out / filename).read_bytes() == (FIX / filename).read_bytes(), name

    def test_manifest(self, built):
        out, manifest, _, _ = built
        assert manifest == json.loads((out / MANIFEST_FILE).read_text())
        assert manifest["source"] == "kegg"
        assert manifest["release"] == "KEGG pathway 2026/11/09, compound 2026/11/10, drug 2026/11/10, genes 2026/11/10"
        assert manifest["kegg_release"]["pathway"] == "2026/11/09"
        assert set(manifest["files"]) == set(KEGG_TABLE_FILES.values()) | {INCHIKEY_FILE}
        for filename, meta in manifest["files"].items():
            data = (out / filename).read_bytes()
            assert meta["sha256"] == hashlib.sha256(data).hexdigest(), filename
            assert meta["bytes"] == len(data)
            assert meta["release"] == manifest["release"]
            assert "academic" in meta["licence"]
            assert meta["fetched_at"]
        assert manifest["files"]["conv_compound_pubchem.tsv"]["url"] == f"{KEGG}/conv/compound/pubchem"
        assert manifest["files"]["link_pathway_hsa.tsv"]["rows"] == 14
        assert manifest["files"][INCHIKEY_FILE]["url"] is None
        assert manifest["counts"] == {
            "cpd_entries": 5,
            "dr_entries": 6,
            "sids": 11,
            "sids_without_cid": 1,
            "cids": 8,
            "cids_without_inchikey": 0,
            "invalid_inchikeys": 0,
            "entries_with_cid": 10,
            "entries_with_inchikey": 10,
            "distinct_inchikeys": 8,
        }
        assert [r["url"] for r in manifest["pubchem_requests"]] == [PUBCHEM_SID_CIDS_URL, PUBCHEM_CID_INCHIKEY_URL]

    def test_kegg_paced_at_one_request_per_second(self, built):
        _, _, providers, clock = built
        starts = [c["t"] for c in providers.gets()]
        gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
        assert min(gaps) >= 1.0 - 1e-9
        assert starts[-1] - starts[0] >= 10.0 - 1e-9

    @pytest.mark.parametrize("rate", [2.0, 3.0])
    def test_faster_kegg_rate_within_the_documented_cap(self, tmp_path, rate):
        _, providers, _ = _build(tmp_path, kegg_rate=rate)
        starts = [c["t"] for c in providers.gets()]
        assert min(b - a for a, b in zip(starts, starts[1:], strict=False)) >= 1.0 / rate - 1e-9

    def test_pubchem_chunked_and_paced(self, tmp_path, built):
        _, providers, _ = _build(tmp_path, sid_chunk_size=3, cid_chunk_size=2)
        assert len(providers.posts(PUBCHEM_SID_CIDS_URL)) == 4  # 11 SIDs / 3
        assert len(providers.posts(PUBCHEM_CID_INCHIKEY_URL)) == 4  # 8 CIDs / 2
        starts = [c["t"] for c in providers.posts()]
        assert min(b - a for a, b in zip(starts, starts[1:], strict=False)) >= 0.25 - 1e-9
        assert (tmp_path / INCHIKEY_FILE).read_bytes() == (built[0] / INCHIKEY_FILE).read_bytes()

    @pytest.mark.parametrize(
        "kwargs",
        [{"kegg_rate": 3.5}, {"kegg_rate": 0}, {"pubchem_rate": 6}, {"sid_chunk_size": 0}],
    )
    def test_bad_settings_rejected(self, tmp_path, kwargs):
        with pytest.raises(ValueError):
            _build(tmp_path, **kwargs)

    def test_module_helpers_are_the_default(self, tmp_path, monkeypatch):
        clock = FakeClock()
        providers = RecordedProviders(clock)
        monkeypatch.setattr(kegg_index, "_get_text", providers.get_text)
        monkeypatch.setattr(kegg_index, "_post_json", providers.post_json)
        build_kegg_reference(tmp_path, clock=clock, sleep=clock.sleep)
        assert len(providers.gets()) == 11
        assert len(providers.posts()) == 2
        assert (tmp_path / INCHIKEY_FILE).is_file()


class TestBuildFailures:
    def test_helper_none_is_a_failure_and_nothing_is_written(self, tmp_path):
        out = tmp_path / "kegg"
        with pytest.raises(KeggReferenceError, match="link_pathway_hsa"):
            _build(out, overrides={"link/pathway/hsa": None})
        assert not out.exists() or not any(out.iterdir())

    def test_patched_helper_exception_propagates(self, tmp_path):
        class CircuitOpen(Exception):
            pass

        def refuse(url):
            raise CircuitOpen(url)

        with pytest.raises(CircuitOpen):
            _build(tmp_path / "kegg", overrides={"list/pathway": refuse})
        assert not (tmp_path / "kegg").exists()

    def test_pubchem_none_is_a_failure(self, tmp_path):
        with pytest.raises(KeggReferenceError, match="SID->CID"):
            _build(tmp_path / "kegg", overrides={PUBCHEM_SID_CIDS_URL: None})
        with pytest.raises(KeggReferenceError, match="CID->InChIKey"):
            _build(tmp_path / "kegg2", overrides={PUBCHEM_CID_INCHIKEY_URL: None})

    def test_empty_table_rejected(self, tmp_path):
        # KEGG answers "no result" with a bare newline; a bulk table must never be empty.
        with pytest.raises(KeggReferenceError, match="empty"):
            _build(tmp_path, overrides={"conv/compound/pubchem": "\n"})

    def test_malformed_table_rejected(self, tmp_path):
        with pytest.raises(KeggReferenceError, match="unexpected row"):
            _build(tmp_path, overrides={"conv/drug/pubchem": "<html><body>Forbidden</body></html>\n"})
        # Columns swapped (the per-entity conv/pubchem/<entry> orientation).
        swapped = "cpd:C91405\tpubchem:4594\n"
        with pytest.raises(KeggReferenceError, match="unexpected row"):
            _build(tmp_path, overrides={"conv/compound/pubchem": swapped})

    def test_unrequested_sid_in_response_rejected(self, tmp_path):
        payload = {"InformationList": {"Information": [{"SID": 1, "CID": [2244]}]}}
        with pytest.raises(KeggReferenceError, match="not requested"):
            _build(tmp_path, overrides={PUBCHEM_SID_CIDS_URL: payload})

    def test_failed_rebuild_leaves_existing_reference_untouched(self, tmp_path, built):
        out = tmp_path / "kegg"
        shutil.copytree(built[0], out)
        before = {p.name: p.read_bytes() for p in out.iterdir()}
        with pytest.raises(KeggReferenceError):
            _build(out, overrides={PUBCHEM_CID_INCHIKEY_URL: None})
        assert {p.name: p.read_bytes() for p in out.iterdir()} == before


# =====================================================================
# KeggIndex
# =====================================================================


class TestIndexLoading:
    def test_release_and_size(self, index, built):
        assert index.release == built[1]["release"]
        assert len(index) == 11

    def test_missing_folder(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            KeggIndex.from_dir(tmp_path / "missing")

    def test_tables_without_inchikey_map_are_not_an_index(self):
        # The fixture folder holds the KEGG tables but no kegg_inchikey.tsv.
        with pytest.raises(FileNotFoundError, match=INCHIKEY_FILE):
            KeggIndex.from_dir(FIX)

    def test_modified_file_detected(self, tmp_path, built):
        out = tmp_path / "kegg"
        shutil.copytree(built[0], out)
        with (out / "link_pathway_drug.tsv").open("a") as fh:
            fh.write("dr:D90109\tpath:map90010\n")
        with pytest.raises(KeggReferenceError, match="sha256"):
            KeggIndex.from_dir(out)
        unverified = KeggIndex.from_dir(out, verify=False)
        assert "map90010" in _ids(unverified.lookup(ASPIRIN)["compound_pathways"])

    def test_malformed_map_row(self, tmp_path, built):
        out = tmp_path / "kegg"
        shutil.copytree(built[0], out)
        with (out / INCHIKEY_FILE).open("a") as fh:
            fh.write("C93316\tcpd\t1\t2\tnot-a-key\n")
        with pytest.raises(KeggReferenceError, match="malformed"):
            KeggIndex.from_dir(out, verify=False)


class TestLookup:
    def test_aspirin_maps_to_compound_and_drug_exactly(self, index):
        r = index.lookup(ASPIRIN)
        assert r["status"] == "success"
        assert r["match_tier"] == "exact"
        assert _entries(r) == {("C91405", "cpd", "exact"), ("D90109", "dr", "exact")}
        assert [e["kegg_id"] for e in r["entries"]] == ["C91405", "D90109"]
        c01405 = r["entries"][0]
        assert c01405["sids"] == [4594] and c01405["cids"] == [2244]
        assert c01405["inchikey"] == ASPIRIN and c01405["chebi_ids"] == ["CHEBI:15365"]
        assert r["compound_pathways"] == [{
            "id": "map94976", "name": "Synthetic reference map 94976", "match_tier": "exact",
            "kegg_ids": ["C91405"], "namespaces": ["cpd"],
        }]
        assert _ids(r["drug_maps"]) == ["map07748", "map07749", "map07810", "map07919"]
        assert r["drug_maps"][3]["name"] == "Synthetic drug map - class 919"
        # D90109 targets PTGS1 (hsa:5742) and PTGS2 (hsa:5743).
        assert [t["gene"] for t in r["targets"]] == ["hsa:5742", "hsa:5743"]
        assert _ids(r["target_pathways"]) == ["hsa90590", "hsa91100", "hsa95200"]
        hsa90590 = r["target_pathways"][0]
        assert hsa90590["name"] == "Synthetic human pathway 90590"
        assert set(hsa90590["genes"]) == {"hsa:5742", "hsa:5743"}
        assert hsa90590["kegg_ids"] == ["D90109"]
        assert r["release"] == index.release
        assert r["mismatched"] == [] and r["unverified"] == []

    def test_imatinib_target_pathways(self, index):
        r = index.lookup(IMATINIB)
        assert r["status"] == "success"
        # The mesylate salt (D91441) has another InChIKey and connectivity block.
        assert _entries(r) == {("D98066", "dr", "exact")}
        assert _ids(r["drug_maps"]) == ["map07745"]
        assert r["drug_maps"][0]["name"] == "Synthetic drug map - class 745"
        assert {t["entrez_id"] for t in r["targets"]} == {"25", "3815", "5156"}
        assert _ids(r["target_pathways"]) == ["hsa94010", "hsa94110", "hsa95200", "hsa95220"]
        cml = r["target_pathways"][3]
        assert cml["name"] == "Synthetic human pathway 95220"
        assert cml["genes"] == ["hsa:25"]
        assert set(r["target_pathways"][2]["genes"]) == {"hsa:25", "hsa:3815", "hsa:5156"}
        # D98066's own non-map07 link stays a compound-pathway item, marked as coming from KEGG DRUG.
        assert r["compound_pathways"] == [{
            "id": "map94976", "name": "Synthetic reference map 94976", "match_tier": "exact",
            "kegg_ids": ["D98066"], "namespaces": ["dr"],
        }]

    def test_cid_used_as_sid_never_maps(self, index):
        """CID 5291 is imatinib; SID 5291 is C92226. They must never meet."""
        for key in ALL_KEYS:
            for cid in (None, 5291, 3413):
                r = index.lookup(key, cid=cid)
                seen = {e["kegg_id"] for e in r["entries"] + r["mismatched"] + r["unverified"]}
                if key != CITRACONATE:
                    assert "C92226" not in seen, (key, cid)
                if key != PQQ:
                    assert "C90113" not in seen, (key, cid)  # SID 3413 is PQQ; CID 3413 is not
        r = index.lookup(IMATINIB, cid=5291)
        assert [e["kegg_id"] for e in r["entries"]] == ["D98066"]
        assert r["mismatched"] == []

    def test_identity_mismatch_through_cid(self, index):
        # The caller's CID (2244, aspirin) disagrees with the structure's own InChIKey.
        r = index.lookup(NOT_IN_FIXTURE, cid=2244)
        assert r["status"] == "identity_mismatch"
        assert r["entries"] == [] and r["match_tier"] is None
        assert r["compound_pathways"] == r["drug_maps"] == r["target_pathways"] == r["targets"] == []
        assert [(m["kegg_id"], m["namespace"], m["via"], m["inchikeys"]) for m in r["mismatched"]] == [
            ("C91405", "cpd", ["cid"], [ASPIRIN]),
            ("D90109", "dr", ["cid"], [ASPIRIN]),
        ]

    def test_identity_mismatch_through_chebi(self, index):
        r = index.lookup(NOT_IN_FIXTURE, chebi_ids=["CHEBI:15365"])
        assert r["status"] == "identity_mismatch"
        assert {m["kegg_id"] for m in r["mismatched"]} == {"C91405", "D90109"}
        assert all(m["via"] == ["chebi"] for m in r["mismatched"])
        # Agreeing evidence adds nothing and is not reported as a mismatch.
        agreed = index.lookup(ASPIRIN, cid=2244, chebi_ids=["15365"])
        assert agreed["status"] == "success" and agreed["mismatched"] == []

    def test_entry_without_structure_is_never_accepted(self, index):
        # D90056's SID has no standardized CID, so its identity cannot be checked.
        r = index.lookup(NOT_IN_FIXTURE, chebi_ids=["chebi:9765"])
        assert r["status"] == "not_in_kegg"
        assert r["entries"] == [] and r["mismatched"] == []
        assert [(u["kegg_id"], u["via"]) for u in r["unverified"]] == [("D90056", ["chebi"])]

    def test_exact_match_with_connectivity_neighbour(self, index):
        r = index.lookup(DEXAMETHASONE)
        assert r["status"] == "success" and r["match_tier"] == "exact"
        assert [(e["kegg_id"], e["match_tier"]) for e in r["entries"]] == [
            ("C95643", "exact"), ("D90292", "exact"), ("D90244", "connectivity"),
        ]
        assert r["entries"][2]["inchikey"] == BETAMETHASONE
        # Every map is already supported exactly by D90292, so betamethasone adds no support.
        assert _ids(r["drug_maps"]) == ["map07743", "map07746", "map07925"]
        assert all(m["match_tier"] == "exact" and m["kegg_ids"] == ["D90292"] for m in r["drug_maps"])
        assert all(p["match_tier"] == "exact" for p in r["target_pathways"])

    def test_connectivity_only_support_is_flagged(self, index):
        r = index.lookup(BETAMETHASONE)
        tiers = {m["id"]: (m["match_tier"], m["kegg_ids"]) for m in r["drug_maps"]}
        assert tiers == {
            "map07925": ("exact", ["D90244"]),
            "map07743": ("connectivity", ["D90292"]),
            "map07746": ("connectivity", ["D90292"]),
        }
        strict = index.lookup(BETAMETHASONE, connectivity=False)
        assert _entries(strict) == {("D90244", "dr", "exact")}
        assert _ids(strict["drug_maps"]) == ["map07925"]

    def test_connectivity_tier_only(self, index):
        r = index.lookup(STEROID_NO_STEREO)
        assert r["status"] == "success"
        assert r["match_tier"] == "connectivity"
        assert {e["match_tier"] for e in r["entries"]} == {"connectivity"}
        items = r["compound_pathways"] + r["drug_maps"] + r["target_pathways"] + r["targets"]
        assert items and {i["match_tier"] for i in items} == {"connectivity"}
        assert index.lookup(STEROID_NO_STEREO, connectivity=False)["status"] == "not_in_kegg"

    def test_entry_without_pathway(self, index):
        r = index.lookup(PQQ)
        assert r["status"] == "kegg_entry_no_pathway"
        assert _entries(r) == {("C90113", "cpd", "exact")}

    def test_not_in_kegg(self, index):
        r = index.lookup(NOT_IN_FIXTURE)
        assert r["status"] == "not_in_kegg"
        assert r["entries"] == [] and r["match_tier"] is None
        assert r["release"] == index.release

    def test_families_stay_separate(self, index):
        for key in ALL_KEYS:
            r = index.lookup(key)
            assert all(i.startswith("map") and not i.startswith("map07") for i in _ids(r["compound_pathways"]))
            assert all(i.startswith("map07") for i in _ids(r["drug_maps"]))
            assert all(i.startswith("hsa") for i in _ids(r["target_pathways"]))

    def test_water_is_sid_3303(self, index):
        r = index.lookup(WATER)
        assert [(e["kegg_id"], e["sids"], e["cids"]) for e in r["entries"]] == [("C90001", [3303], [962])]
        assert _ids(r["compound_pathways"]) == ["map90190", "map91100", "map94976"]

    def test_input_validation(self, index):
        assert index.lookup(ASPIRIN.lower())["status"] == "success"
        for bad in ("", "aspirin", "BSYNRYMUTXBXSQ-UHFFFAOYSA", "C91405"):
            with pytest.raises(ValueError):
                index.lookup(bad)
        with pytest.raises(ValueError):
            index.lookup(ASPIRIN, chebi_ids=["CHEBI:abc"])

    def test_lookup_many(self, index):
        out = index.lookup_many([ASPIRIN, PQQ, NOT_IN_FIXTURE])
        assert {k: v["status"] for k, v in out.items()} == {
            ASPIRIN: "success", PQQ: "kegg_entry_no_pathway", NOT_IN_FIXTURE: "not_in_kegg",
        }


def test_parse_kegg_info():
    info = parse_kegg_info((FIX / "info.txt").read_text())
    assert info["pathway"] == "2026/11/09"
    assert info["drug"] == "2026/11/10"
    assert info["compound"] == "2026/11/10"


# =====================================================================
# Opt-in live check of the payload shapes (EMBPY_NETWORK_TESTS=1)
# =====================================================================


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("EMBPY_NETWORK_TESTS") != "1", reason="set EMBPY_NETWORK_TESTS=1 for live checks")
def test_live_kegg_and_pubchem_shapes():
    import time

    info = kegg_index._get_text(f"{KEGG}/info/kegg", timeout=60)
    assert "pathway" in parse_kegg_info(info)
    time.sleep(1.1)
    # KEGG's pubchem: namespace is SIDs: aspirin's KEGG COMPOUND entry is SID 4594 ...
    assert kegg_index._get_text(f"{KEGG}/conv/pubchem/cpd:C01405", timeout=60).strip() == "cpd:C01405\tpubchem:4594"
    # ... which PubChem standardizes to CID 2244.
    payload = kegg_index._post_json(PUBCHEM_SID_CIDS_URL, data={"sid": "4594"}, timeout=60)
    assert payload["InformationList"]["Information"] == [{"SID": 4594, "CID": [2244]}]
    time.sleep(0.3)
    props = kegg_index._post_json(PUBCHEM_CID_INCHIKEY_URL, data={"cid": "2244"}, timeout=60)
    assert props["PropertyTable"]["Properties"] == [{"CID": 2244, "InChIKey": ASPIRIN}]
