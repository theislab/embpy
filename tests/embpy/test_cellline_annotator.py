"""Tests for embpy.resources.cellline.annotator.CellLineAnnotator.

Every provider payload used here was recorded from the live APIs on
2026-11-11 and trimmed (see fixtures/cellline/SOURCES.txt). No test touches
the network except the opt-in ``network`` checks at the end
(``EMBPY_NETWORK_TESTS=1``).
"""

from __future__ import annotations

import copy
import gzip
import inspect
import json
import os
from pathlib import Path

import pandas as pd
import pytest
from anndata import AnnData

import embpy.resources.cellline.annotator as mod
from embpy.resources.cellline.annotator import CellLineAnnotator

FIXTURES = Path(__file__).parent / "fixtures" / "cellline"
DEPMAP_CSV = FIXTURES / "depmap_Model_24Q4_rows.csv"
PASSPORTS_CSV = FIXTURES / "passports_model_list_20240110_rows.csv"


def fx(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeHTTP:
    """Stand-in for the module-level ``_get_json``: replays responses in order."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, params=None, timeout=30):
        self.calls.append((url, dict(params or {})))
        if not self.responses:
            raise AssertionError(f"unexpected request: {url} {params}")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return copy.deepcopy(response)


@pytest.fixture
def http(monkeypatch):
    def install(*responses):
        fake = FakeHTTP(*responses)
        monkeypatch.setattr(mod, "_get_json", fake)
        return fake

    return install


@pytest.fixture
def annotator():
    return CellLineAnnotator(rate_limit_delay=0)


def by_acc(payload, accession):
    for entry in payload["Cellosaurus"]["cell-line-list"]:
        if any(a["value"] == accession for a in entry["accession-list"]):
            return entry
    raise KeyError(accession)


COMMON_KEYS = {"status", "accession", "rrid", "name", "synonyms", "species", "disease", "site", "tissue", "xrefs"}


# ----------------------------------------------------------------------
# Identifier helpers
# ----------------------------------------------------------------------


class TestIdentifierHelpers:
    def test_cellosaurus_id(self):
        assert CellLineAnnotator._is_cellosaurus_id("CVCL_0023") is True
        assert CellLineAnnotator._is_cellosaurus_id("A549") is False

    def test_depmap_id(self):
        assert CellLineAnnotator._is_depmap_id("ACH-000681") is True
        assert CellLineAnnotator._is_depmap_id("A549") is False
        assert CellLineAnnotator._is_depmap_id("ACH-12345") is False

    def test_passports_id(self):
        assert CellLineAnnotator._is_passports_id("SIDM00148") is True
        assert CellLineAnnotator._is_passports_id("MCF7") is False

    def test_normalize(self):
        assert mod.normalize_cell_name("MCF-7") == mod.normalize_cell_name("MCF7") == "mcf7"
        assert mod.normalize_cell_name("PC.3") == "pc3"

    def test_backward_compatible_import(self):
        from embpy.resources.cellline_annotator import CellLineAnnotator as Legacy

        assert Legacy is CellLineAnnotator

    def test_endpoints(self):
        assert mod.CMP_API == "https://api.cellmodelpassports.sanger.ac.uk"
        assert "/v1" not in mod.CMP_API
        assert "depmap.org/portal" not in inspect.getsource(mod)


# ----------------------------------------------------------------------
# Cellosaurus v2 parsing
# ----------------------------------------------------------------------


class TestCellosaurusParsing:
    def test_search_entry_parses_species_site_disease(self):
        batch = fx("cellosaurus_search_ac_batch4.json")
        mcf7 = mod.parse_cellosaurus_entry(by_acc(batch, "CVCL_0031"))
        assert mcf7["accession"] == mcf7["rrid"] == "CVCL_0031"
        assert mcf7["name"] == "MCF-7"
        assert "MCF7" in mcf7["synonyms"]
        assert mcf7["species"] == "Homo sapiens"
        assert mcf7["species_label"] == "Homo sapiens (Human)"
        assert mcf7["species_taxids"] == ["9606"]
        assert mcf7["site"] == mcf7["tissue"] == "Pleural effusion"
        assert mcf7["site_type"] == "Metastatic"
        assert mcf7["sites"][0]["uberon"] == ["UBERON_0000175"]
        assert mcf7["disease"] == "Invasive breast carcinoma of no special type"
        assert mcf7["diseases"][0] == {"database": "NCIt", "accession": "C4194",
                                       "label": "Invasive breast carcinoma of no special type"}
        assert mcf7["category"] == "Cancer cell line"
        assert mcf7["sex"] == "Female"
        assert mcf7["depmap_ids"] == ["ACH-000019"]
        assert mcf7["passports_ids"] == ["SIDM00148"]
        assert set(mcf7["lincs_ids"]) == {"LCL-1460", "50029"}
        assert mcf7["has_lincs_xref"] is True
        assert {"database": "DepMap", "accession": "ACH-000019"} in mcf7["xrefs"]
        assert mcf7["cross_references"]["DepMap"] == "ACH-000019"
        assert mcf7["is_problematic"] is False
        assert COMMON_KEYS <= set(mcf7)

        pc3 = mod.parse_cellosaurus_entry(by_acc(batch, "CVCL_0035"))
        assert (pc3["name"], pc3["site"], pc3["disease"], pc3["sex"]) == ("PC-3", "Bone", "Prostate carcinoma", "Male")
        ht29 = mod.parse_cellosaurus_entry(by_acc(batch, "CVCL_0320"))
        assert (ht29["site"], ht29["disease"]) == ("Colon", "Colon adenocarcinoma")

    def test_asc_record_has_no_lincs_xref(self):
        """LINCS maps ASC to CVCL_U602, which is ASC52telo: flag for review."""
        asc = mod.parse_cellosaurus_entry(by_acc(fx("cellosaurus_search_ac_batch4.json"), "CVCL_U602"))
        assert asc["name"] == "ASC52telo"
        assert asc["category"] == "Telomerase immortalized cell line"
        assert asc["cell_type"] == "Mesenchymal stem cell of adipose tissue"
        assert asc["disease"] == ""
        assert asc["has_lincs_xref"] is False
        assert asc["lincs_ids"] == asc["depmap_ids"] == asc["passports_ids"] == []

    def test_detail_record(self):
        detail = fx("cellosaurus_detail_CVCL_0031.json")
        assert set(detail["Cellosaurus"]) == {"cell-line-list", "publication-list"}
        rec = mod.parse_cellosaurus_entry(detail["Cellosaurus"]["cell-line-list"][0])
        assert rec["accession"] == "CVCL_0031"
        assert rec["species"] == "Homo sapiens"
        assert rec["site"] == "Pleural effusion"
        assert rec["disease"] == "Invasive breast carcinoma of no special type"
        assert rec["msi_status"] == "Stable (MSS)"
        assert rec["age"] == "69Y"
        assert rec["depmap_ids"] == ["ACH-000019"]

        pc3 = mod.parse_cellosaurus_entry(fx("cellosaurus_detail_CVCL_0035.json")["Cellosaurus"]["cell-line-list"][0])
        assert (pc3["accession"], pc3["site"], pc3["disease"]) == ("CVCL_0035", "Bone", "Prostate carcinoma")

    def test_problematic_and_parent(self):
        kb = mod.parse_cellosaurus_entry(fx("cellosaurus_search_ac_CVCL_0372_problematic.json")
                                         ["Cellosaurus"]["cell-line-list"][0])
        assert kb["is_problematic"] is True
        assert "HeLa derivative" in kb["problems"][0]
        assert kb["parent_accession"] == "CVCL_0030"
        assert kb["parent_cell_line"] == "HeLa"

    def test_payload_shapes(self):
        assert mod.cellosaurus_entries(fx("cellosaurus_search_name_NKDBA_empty.json")) == []
        assert mod.cellosaurus_entries(fx("cellosaurus_detail_404.json")) is None
        assert mod.cellosaurus_entries({"resultList": {"result": [{"accession": "CVCL_0023"}]}}) is None


# ----------------------------------------------------------------------
# Cellosaurus resolution
# ----------------------------------------------------------------------


class TestCellosaurusResolution:
    def test_accession_is_resolved_with_one_search(self, annotator, http):
        fake = http(fx("cellosaurus_search_ac_batch4.json"))
        info = annotator.get_cellosaurus_info("CVCL_0031")
        assert info["status"] == "success"
        assert info["method"] == "accession"
        assert info["accession"] == "CVCL_0031"
        assert info["species"] == "Homo sapiens"
        # The unrequested rows of the recorded batch payload are ignored.
        assert len(fake.calls) == 1
        url, params = fake.calls[0]
        assert url == "https://api.cellosaurus.org/search/cell-line"
        assert params["q"] == "ac:CVCL_0031"
        assert params["fields"] == mod.CELLOSAURUS_FIELDS
        assert params["format"] == "json"

    def test_batch_one_request_relevance_order(self, annotator, http):
        fake = http(fx("cellosaurus_search_ac_batch4.json"))
        wanted = ["CVCL_0031", "CVCL_0035", "CVCL_0320", "CVCL_U602"]
        out = annotator.get_cellosaurus_batch(wanted)
        assert list(out) == wanted
        assert {k: v["accession"] for k, v in out.items()} == dict(zip(wanted, wanted, strict=True))
        assert all(v["status"] == "success" and v["method"] == "accession" for v in out.values())
        assert len(fake.calls) == 1
        params = fake.calls[0][1]
        assert params["q"] == "ac:CVCL_0031 OR ac:CVCL_0035 OR ac:CVCL_0320 OR ac:CVCL_U602"
        assert params["rows"] >= len(wanted)

    def test_batch_missing_accession_is_no_match(self, annotator, http):
        http(fx("cellosaurus_search_ac_batch4.json"))
        out = annotator.get_cellosaurus_batch(["CVCL_0031", "CVCL_VU89"])
        assert out["CVCL_0031"]["status"] == "success"
        assert out["CVCL_VU89"]["status"] == "no_match"
        assert out["CVCL_VU89"]["reason"] == "accession_not_found"

    def test_batch_chunks(self, annotator, http):
        payload = fx("cellosaurus_search_ac_batch4.json")
        fake = http(payload, payload)
        out = annotator.get_cellosaurus_batch(["CVCL_0031", "CVCL_0035", "CVCL_0320"], chunk_size=2)
        assert [c[1]["q"] for c in fake.calls] == ["ac:CVCL_0031 OR ac:CVCL_0035", "ac:CVCL_0320"]
        assert all(v["status"] == "success" for v in out.values())

    def test_batch_failure_is_service_error_not_empty(self, annotator, http):
        http(None)
        out = annotator.get_cellosaurus_batch(["CVCL_0031", "CVCL_0035"])
        assert {v["status"] for v in out.values()} == {"service_error"}
        assert {v["reason"] for v in out.values()} == {"request_failed"}

    def test_unparsed_response(self, annotator, http):
        """A 2xx with a non-empty provider list that the parser cannot read is not 'empty'."""
        drifted = fx("cellosaurus_search_ac_batch4.json")
        for entry in drifted["Cellosaurus"]["cell-line-list"]:
            entry["accessionList"] = entry.pop("accession-list")
        http(drifted)
        out = annotator.get_cellosaurus_batch(["CVCL_0031"])
        assert out["CVCL_0031"]["status"] == "service_error"
        assert out["CVCL_0031"]["reason"] == "unparsed_response"

        http(fx("cellosaurus_detail_404.json"))
        info = annotator.get_cellosaurus_info("CVCL_0031")
        assert (info["status"], info["reason"]) == ("service_error", "unparsed_response")

    def test_patched_helper_exceptions_propagate(self, annotator, http):
        http(RuntimeError("transport refused"))
        with pytest.raises(RuntimeError, match="transport refused"):
            annotator.get_cellosaurus_batch(["CVCL_0031"])

    def test_depmap_id_via_exact_xref(self, annotator, http):
        fake = http(fx("cellosaurus_search_dr_ACH-000019.json"))
        info = annotator.get_cellosaurus_info("ACH-000019")
        assert info["status"] == "success"
        assert info["accession"] == "CVCL_0031"
        assert info["method"] == "xref:DepMap"
        assert fake.calls[0][1]["q"] == 'dr:"DepMap; ACH-000019"'

    def test_caller_mapping_comes_before_name_search(self, http):
        ann = CellLineAnnotator(rate_limit_delay=0, cellosaurus_ids={"MCF7": "CVCL_0031", "PC3": "CVCL_0035"})
        fake = http(fx("cellosaurus_search_ac_batch4.json"))
        info = ann.get_cellosaurus_info("MCF7")
        assert (info["status"], info["method"], info["accession"]) == ("success", "mapping", "CVCL_0031")
        assert fake.calls[0][1]["q"] == "ac:CVCL_0031"

        fake = http(fx("cellosaurus_search_ac_batch4.json"))
        out = ann.get_cellosaurus_batch(["MCF7", "PC3", "CVCL_0320"])
        assert len(fake.calls) == 1
        assert {k: (v["accession"], v["method"]) for k, v in out.items()} == {
            "MCF7": ("CVCL_0031", "mapping"), "PC3": ("CVCL_0035", "mapping"), "CVCL_0320": ("CVCL_0320", "accession"),
        }

    def test_pc3_name_search_is_ambiguous(self, annotator, http):
        fake = http(fx("cellosaurus_search_name_PC3.json"))
        info = annotator.get_cellosaurus_info("PC3")
        assert info["status"] == "ambiguous"
        assert info["method"] == "name_search"
        accs = {c["accession"] for c in info["candidates"]}
        assert accs == {"CVCL_C8XA", "CVCL_4055", "CVCL_0035", "CVCL_S982", "CVCL_4011"}
        # The mouse hybridoma and the puma line are not human candidates.
        assert {c["accession"] for c in info["other_species_matches"]} == {"CVCL_E2RM", "CVCL_B6Z1"}
        assert info["accession"] == ""
        assert len(fake.calls) == 1
        params = fake.calls[0][1]
        assert params["q"] == 'id:"PC3" OR sy:"PC3"'
        assert params["rows"] == mod.CELLOSAURUS_ROWS

    def test_mcf7_name_search_skips_derived_lines(self, annotator, http):
        """The parent is row 606 of 632; derived lines like 'MCF7 AREc32' are not exact matches."""
        fake = http(fx("cellosaurus_search_name_MCF7_trimmed.json"), fx("cellosaurus_search_ac_batch4.json"))
        info = annotator.get_cellosaurus_info("MCF7")
        assert info["status"] == "success"
        assert info["accession"] == "CVCL_0031"
        assert info["method"] == "name_search"
        assert info["name_match"] == "identifier"  # "MCF-7" normalises to "mcf7"
        assert info["disease"] == "Invasive breast carcinoma of no special type"
        assert [c[1]["q"] for c in fake.calls] == ['id:"MCF7" OR sy:"MCF7"', "ac:CVCL_0031"]

    def test_name_search_single_human_hit(self, annotator, http):
        payload = fx("cellosaurus_search_name_HA1E.json")
        http(payload, payload)
        info = annotator.get_cellosaurus_info("HA1E")
        assert (info["status"], info["accession"], info["name_match"]) == ("success", "CVCL_VU89", "identifier")
        assert info["parent_accession"] == "CVCL_C9AK"

    def test_name_search_no_hit(self, annotator, http):
        http(fx("cellosaurus_search_name_NKDBA_empty.json"))
        info = annotator.get_cellosaurus_info("NKDBA")
        assert info["status"] == "no_match"
        assert info["reason"] == "no_exact_name_match"
        assert COMMON_KEYS <= set(info)

    def test_name_search_failure_is_service_error(self, annotator, http):
        http(None)
        info = annotator.get_cellosaurus_info("MCF7")
        assert info["status"] == "service_error"

    def test_name_search_can_be_disabled(self, annotator, http):
        fake = http()
        info = annotator.get_cellosaurus_info("MCF7", name_search=False)
        assert info["status"] == "no_match"
        assert fake.calls == []

    def test_release(self, annotator, http):
        http(fx("cellosaurus_release_info.json"))
        assert annotator.get_cellosaurus_release() == {
            "status": "success", "version": "57.0", "updated": "2026-10-22", "nb_cell_lines": "169026"}


# ----------------------------------------------------------------------
# DepMap (pinned Model.csv)
# ----------------------------------------------------------------------


class TestDepMap:
    def test_without_file_source_unavailable_and_no_network(self, annotator, http):
        fake = http()
        info = annotator.get_depmap_info("MCF7", rrid="CVCL_0031")
        assert info["status"] == "source_unavailable"
        assert COMMON_KEYS <= set(info)
        assert fake.calls == []

    def test_join_by_rrid(self, http):
        fake = http()
        ann = CellLineAnnotator(rate_limit_delay=0, depmap_model_csv=DEPMAP_CSV, depmap_release="24Q4")
        info = ann.get_depmap_info("PC3", rrid="CVCL_0035")
        assert info["status"] == "success"
        assert info["method"] == "rrid"
        assert info["depmap_id"] == info["accession"] == "ACH-000090"
        assert info["rrid"] == "CVCL_0035"
        assert info["name"] == "PC-3"
        assert info["lineage"] == "Prostate"
        assert info["primary_disease"] == info["disease"] == "Prostate Adenocarcinoma"
        assert info["disease_subtype"] == "Prostate Adenocarcinoma"
        assert info["growth_pattern"] == "Adherent"
        assert info["sex"] == "Male"
        assert info["site"] == "bone"
        assert info["release"] == "24Q4"
        assert {"database": "Cell_Model_Passport", "accession": "SIDM00088"} in info["xrefs"]
        assert len(info["file_sha256"]) == 64
        assert fake.calls == []

    def test_name_model_id_and_cvcl_queries(self):
        ann = CellLineAnnotator(rate_limit_delay=0, depmap_model_csv=str(DEPMAP_CSV))
        assert ann.get_depmap_info("MCF7")["depmap_id"] == "ACH-000019"
        assert ann.get_depmap_info("HT29")["rrid"] == "CVCL_0320"
        assert ann.get_depmap_info("ACH-000552")["name"] == "HT-29"
        assert ann.get_depmap_info("CVCL_0031")["method"] == "rrid"
        assert ann.get_depmap_info("NOTALINE")["status"] == "no_match"
        # The quoted multi-line PatientSubtypeFeatures field of HL-60 parses.
        assert ann.get_depmap_info("HL60")["depmap_id"] == "ACH-000002"

    def test_shared_rrid_and_shared_names(self):
        ann = CellLineAnnotator(rate_limit_delay=0, depmap_model_csv=DEPMAP_CSV)
        shared = ann.get_depmap_info("CVCL_0041")
        assert shared["status"] == "ambiguous"
        assert {c["depmap_id"] for c in shared["candidates"]} == {"ACH-000833", "ACH-001189"}
        narrowed = ann.get_depmap_info("SJRH30", rrid="CVCL_0041")
        assert (narrowed["status"], narrowed["depmap_id"], narrowed["method"]) == ("success", "ACH-001189", "rrid+name")
        assert ann.get_depmap_info("MS1")["status"] == "ambiguous"

    def test_name_match_with_conflicting_rrid_is_rejected(self):
        ann = CellLineAnnotator(rate_limit_delay=0, depmap_model_csv=DEPMAP_CSV)
        info = ann.get_depmap_info("MCF7", rrid="CVCL_9999")
        assert info["status"] == "no_match"

    def test_gzip_and_dataframe_inputs(self, tmp_path):
        gz = tmp_path / "Model.csv.gz"
        gz.write_bytes(gzip.compress(DEPMAP_CSV.read_bytes()))
        assert CellLineAnnotator(depmap_model_csv=gz).get_depmap_info("PC3")["depmap_id"] == "ACH-000090"
        frame = pd.read_csv(DEPMAP_CSV, dtype=str)
        info = CellLineAnnotator(depmap_model_csv=frame).get_depmap_info("PC3", rrid="CVCL_0035")
        assert info["depmap_id"] == "ACH-000090"


# ----------------------------------------------------------------------
# Cell Model Passports
# ----------------------------------------------------------------------


class TestPassportsAPI:
    def test_names_filter_with_included_sample(self, annotator, http):
        fake = http(fx("passports_models_names_MCF7.json"))
        info = annotator.get_passports_info("MCF7")
        assert info["status"] == "success"
        assert info["method"] == "name"
        assert info["model_id"] == info["accession"] == "SIDM00148"
        assert info["name"] == info["model_name"] == "MCF7"
        assert "IBMF-7" in info["synonyms"]
        assert info["tissue"] == "Breast"
        assert info["cancer_type"] == info["disease"] == "Breast Carcinoma"
        assert info["site"] == "Pleural Effusion"
        assert info["msi_status"] == "MSS"
        assert info["rrid"] == "CVCL_0031"
        assert info["depmap_id"] == "ACH-000019"
        assert {"database": "Cellosaurus (RRID)", "accession": "CVCL_0031"} in info["xrefs"]
        assert info["model_type"] == "Cell Line"
        assert info["growth_properties"] == "Adherent"
        assert info["mutations_per_mb"] == info["mutational_burden"] == 42.61
        assert info["ploidy_wes"] == pytest.approx(3.27465591)
        assert info["meta_count"] == 1
        url, params = fake.calls[0]
        assert url == "https://api.cellmodelpassports.sanger.ac.uk/models"
        assert params["filter"] == '[{"name":"names","op":"any","val":"MCF7"}]'
        assert set(params["include"].split(",")) >= {"sample.tissue", "sample.cancer_type", "model_msi_status"}

    def test_ignored_filter_is_rejected(self, annotator, http):
        """Unknown filters are ignored and all 2,292 models come back (PK-59 first)."""
        http(fx("passports_models_ignored_filter.json"))
        info = annotator.get_passports_info("MCF7")
        assert info["status"] == "service_error"
        assert info["reason"] == "filter_ignored"
        assert info["meta_count"] == 2292
        assert "model_id" not in info
        assert info["name"] == ""

    def test_name_without_punctuation_and_rrid_route(self, annotator, http):
        fake = http(fx("passports_models_names_PC3_empty.json"))
        assert annotator.get_passports_info("PC3")["status"] == "no_match"

        fake = http(fx("passports_models_rrid_CVCL_0035.json"))
        info = annotator.get_passports_info("PC3", rrid="CVCL_0035")
        assert (info["status"], info["method"], info["model_id"], info["names"]) == (
            "success", "rrid", "SIDM00088", ["PC-3"])
        assert (info["tissue"], info["cancer_type"]) == ("Prostate", "Prostate Carcinoma")
        assert info["msi_status"] == ""  # provider says "No Data"
        assert info["msi_statuses"] == [{"msi_status": "No Data", "current": True}]
        assert len(fake.calls) == 1
        assert json.loads(fake.calls[0][1]["filter"]) == [
            {"name": "identifiers", "op": "any", "val": {"name": "identifier", "op": "eq", "val": "CVCL_0035"}}]

    def test_rrid_miss_falls_back_to_name(self, annotator, http):
        fake = http(fx("passports_models_names_PC3_empty.json"), fx("passports_models_names_MCF7.json"))
        info = annotator.get_passports_info("MCF7", rrid="CVCL_0031")
        assert (info["status"], info["method"]) == ("success", "name")
        assert len(fake.calls) == 2

    def test_rrid_answer_must_carry_the_rrid(self, annotator, http):
        http(fx("passports_models_rrid_CVCL_0035.json"))
        info = annotator.get_passports_info("MCF7", rrid="CVCL_0031")
        assert (info["status"], info["reason"]) == ("service_error", "filter_ignored")

    def test_current_msi_item_is_used(self, annotator, http):
        doc = fx("passports_models_names_MCF7.json")
        current = next(i for i in doc["included"] if i["type"] == "model_msi_status")
        old = copy.deepcopy(current)
        old["id"] = "1"
        old["attributes"].update(current=False, msi_status="MSI")
        doc["included"].insert(0, old)
        doc["data"][0]["relationships"]["model_msi_status"]["data"].insert(0, {"id": "1", "type": "model_msi_status"})
        http(doc)
        info = annotator.get_passports_info("MCF7")
        assert info["msi_status"] == "MSS"
        assert {"msi_status": "MSI", "current": False} in info["msi_statuses"]

    def test_single_model_without_the_name_is_rejected(self, annotator, http):
        http(fx("passports_models_names_MCF7.json"))
        info = annotator.get_passports_info("T47D")
        assert info["status"] == "service_error"

    def test_failures(self, annotator, http):
        http(None)
        assert annotator.get_passports_info("MCF7")["status"] == "service_error"
        http(fx("passports_v1_models_404.json"))
        assert annotator.get_passports_info("MCF7")["reason"] == "unparsed_response"


class TestPassportsSnapshot:
    def test_jsonapi_snapshot_offline(self, tmp_path, http):
        snap = tmp_path / "passports_models.json"
        pages = [fx("passports_models_names_MCF7.json"), fx("passports_models_rrid_CVCL_0035.json")]
        snap.write_text(json.dumps(pages))
        fake = http()
        ann = CellLineAnnotator(rate_limit_delay=0, passports_models=snap, passports_release="live 2026-11-11")
        by_rrid = ann.get_passports_info("PC3", rrid="CVCL_0035")
        assert (by_rrid["status"], by_rrid["model_id"], by_rrid["method"]) == ("success", "SIDM00088", "rrid")
        assert by_rrid["release"] == "live 2026-11-11"
        normalized = ann.get_passports_info("PC3")
        assert (normalized["model_id"], normalized["method"]) == ("SIDM00088", "normalized_name")
        exact = ann.get_passports_info("MCF7")
        assert (exact["model_id"], exact["method"], exact["tissue"], exact["msi_status"]) == (
            "SIDM00148", "name", "Breast", "MSS")
        assert ann.get_passports_info("HT29", rrid="CVCL_0320")["status"] == "no_match"
        # RRID wins over the name; a name hit whose RRID differs is rejected.
        assert ann.get_passports_info("PC3", rrid="CVCL_0031")["model_id"] == "SIDM00148"
        assert ann.get_passports_info("PC3", rrid="CVCL_9999")["status"] == "no_match"
        assert fake.calls == []
        assert ann.provenance["passports"]["records"] == 2

    def test_model_list_csv(self, http):
        fake = http()
        ann = CellLineAnnotator(rate_limit_delay=0, passports_models=PASSPORTS_CSV)
        mcf7 = ann.get_passports_info("MCF7", rrid="CVCL_0031")
        assert (mcf7["status"], mcf7["model_id"], mcf7["tissue"], mcf7["cancer_type"], mcf7["msi_status"]) == (
            "success", "SIDM00148", "Breast", "Breast Carcinoma", "MSS")
        assert mcf7["synonyms"] == ["ssMCF-7", "MCF7/WT", "IBMF-7", "MCF7-CTRL"]
        assert mcf7["species"] == "Homo Sapiens"
        assert mcf7["depmap_id"] == "ACH-000019"
        assert ann.get_passports_info("HT29")["model_id"] == "SIDM00136"
        assert fake.calls == []


# ----------------------------------------------------------------------
# Combined annotation
# ----------------------------------------------------------------------


@pytest.fixture
def offline_annotator():
    return CellLineAnnotator(rate_limit_delay=0, depmap_model_csv=DEPMAP_CSV, passports_models=PASSPORTS_CSV)


class TestAnnotateCombined:
    def test_annotate_passes_rrid_downstream(self, offline_annotator, http):
        fake = http(fx("cellosaurus_search_ac_batch4.json"))
        ann = offline_annotator.annotate("CVCL_0035")
        assert ann["cellosaurus_id"] == "CVCL_0035"
        assert ann["depmap_id"] == "ACH-000090"
        assert ann["passports_id"] == "SIDM00088"
        assert ann["species"] == "Homo sapiens"
        assert ann["tissue"] == "Bone"
        assert ann["disease"] == "Prostate carcinoma"
        assert ann["lineage"] == "Prostate"
        assert ann["model_type"] == "Cell Line"
        assert ann["sources"]["depmap"]["method"] == "rrid"
        assert ann["sources"]["passports"]["method"] == "rrid"
        assert len(fake.calls) == 1

    def test_failed_sources_do_not_fill_summary(self, annotator, http):
        http(None, None)
        ann = annotator.annotate("MCF7")
        assert ann["sources"]["cellosaurus"]["status"] == "service_error"
        assert ann["sources"]["depmap"]["status"] == "source_unavailable"
        assert ann["sources"]["passports"]["status"] == "service_error"
        assert ann["species"] == ann["tissue"] == ann["disease"] == ""

    def test_annotate_single_source(self, annotator, http):
        http(fx("cellosaurus_search_ac_batch4.json"))
        ann = annotator.annotate("CVCL_0031", sources=["cellosaurus"])
        assert "cellosaurus" in ann["sources"]
        assert "depmap" not in ann["sources"]

    def test_annotate_adata_batches_cellosaurus(self, offline_annotator, http):
        fake = http(fx("cellosaurus_search_ac_batch4.json"))
        obs = pd.DataFrame({"cell_line": ["CVCL_0031", "CVCL_0031", "CVCL_0035"]},
                           index=pd.Index(["c1", "c2", "c3"]))
        result = offline_annotator.annotate_adata(AnnData(obs=obs), column="cell_line")
        assert list(result.obs["cellline_tissue"]) == ["Pleural effusion", "Pleural effusion", "Bone"]
        assert list(result.obs["cellline_lineage"]) == ["Breast", "Breast", "Prostate"]
        assert list(result.obs["cellline_msi_status"]) == ["MSS", "MSS", ""]
        assert set(result.uns["cellline_annotations"]) == {"CVCL_0031", "CVCL_0035"}
        assert len(fake.calls) == 1


class TestWikipediaSource:
    def test_get_wikipedia_info(self, annotator, http):
        http({"extract": "HeLa is an immortal cell line derived from cervical cancer cells.", "type": "standard"})
        text = annotator.get_wikipedia_info("HeLa")
        assert "immortal cell line" in text

    def test_wikipedia_fallback_cell_line_suffix(self, annotator, http):
        http({"extract": "", "type": "disambiguation"},
             {"extract": "K562 is a chronic myeloid leukemia cell line."})
        text = annotator.get_wikipedia_info("K562")
        assert "myeloid" in text.lower()

    def test_wikipedia_not_found(self, annotator, http):
        http(None, None, None)
        text = annotator.get_wikipedia_info("NONEXISTENT_CELL_XYZ")
        assert text == ""


class TestTextDescription:
    def test_get_text_description(self, offline_annotator, http):
        http(fx("cellosaurus_search_ac_batch4.json"),
             {"extract": "PC-3 cells are a human prostate cancer cell line.", "type": "standard"})
        text = offline_annotator.get_text_description("CVCL_0035")
        assert "CVCL_0035 is a cell line." in text
        assert "Species: Homo sapiens." in text
        assert "Derived from Bone tissue." in text
        assert "Disease: Prostate carcinoma." in text
        assert "Lineage: Prostate (Prostate Adenocarcinoma)." in text
        assert "prostate cancer cell line" in text


class TestTextResolverCellLine:
    def test_detect_depmap_id(self):
        from embpy.resources.text_resolver import TextResolver

        assert TextResolver._detect_entity_type("ACH-000681") == "cellline"

    def test_detect_cellosaurus_id(self):
        from embpy.resources.text_resolver import TextResolver

        assert TextResolver._detect_entity_type("CVCL_0023") == "cellline"

    def test_gene_not_cellline(self):
        from embpy.resources.text_resolver import TextResolver

        assert TextResolver._detect_entity_type("TP53") == "gene"


# ----------------------------------------------------------------------
# Opt-in live checks: provider top-level keys
# ----------------------------------------------------------------------

LIVE = pytest.mark.skipif(os.environ.get("EMBPY_NETWORK_TESTS") != "1",
                          reason="set EMBPY_NETWORK_TESTS=1 to run live provider checks")


@pytest.mark.network
@LIVE
def test_live_cellosaurus_shape():
    data = mod._get_json(f"{mod.CELLOSAURUS_API}/search/cell-line",
                         params={"q": "ac:CVCL_0031", "fields": mod.CELLOSAURUS_FIELDS, "format": "json", "rows": 10})
    assert data is not None
    entries = mod.cellosaurus_entries(data)
    assert entries and {"accession-list", "name-list", "species-list", "xref-list"} <= set(entries[0])
    assert CellLineAnnotator(rate_limit_delay=0).get_cellosaurus_info("CVCL_0031")["status"] == "success"


@pytest.mark.network
@LIVE
def test_live_passports_shape():
    data = mod._get_json(f"{mod.CMP_API}/models",
                         params={"filter": '[{"name":"names","op":"any","val":"MCF7"}]',
                                 "include": mod.PASSPORTS_INCLUDE, "page[size]": 5})
    assert data is not None
    assert {"data", "included", "meta"} <= set(data)
    assert data["meta"]["count"] == 1
    assert {"names", "model_type"} <= set(data["data"][0]["attributes"])
    assert {"sample", "model_msi_status", "identifiers"} <= set(data["data"][0]["relationships"])
