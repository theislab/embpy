"""Tests for embpy.resources.gene_annotator -- GeneAnnotator.

MyGene payloads under ``fixtures/gene/`` were recorded on 2026-11-11 (see
``fixtures/gene/provenance.json``); the fake POST below only returns the
recorded rows of the queries a chunk asks for, unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from embpy.resources.gene import annotator as gene_annotator_module
from embpy.resources.gene.annotator import (
    MYGENE,
    MYGENE_IDENTITY_FIELDS,
    MYGENE_PATHWAY_FIELDS,
    _post_json,
)
from embpy.resources.gene.reactome import ReactomeIndex
from embpy.resources.gene_annotator import GeneAnnotator

FIXTURES = Path(__file__).parent / "fixtures" / "gene"
GENES_20 = [
    "EGFR", "TP53", "CYP2D6", "ATP5B", "ATP5A1", "ATP5C1", "GNRHR2", "WARS", "GSG2", "MKL1",
    "GJA10", "TCF4", "SLC23A1", "ATF7", "AURKC", "NOTAREALGENE1", "HTR2A", "MTOR", "ESR1", "ERBB4",
]


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


class RecordedMyGene:
    """Stands in for ``_post_json``: answers ``POST /v3/query`` from recorded rows."""

    def __init__(self, *, pathway_rows=None, identity_rows=None, ensembl_rows=None, other=None):
        self.pathway_rows = pathway_rows or []
        self.identity_rows = identity_rows or []
        self.ensembl_rows = ensembl_rows or []
        self.other = other
        self.calls: list[dict] = []

    def __call__(self, url, json_data=None, data=None, timeout=30):
        self.calls.append({"url": url, "json_data": json_data, "data": data, "timeout": timeout})
        if url != f"{MYGENE}/query":
            return self.other
        assert data is not None and json_data is None, "MyGene batch must be sent as form data"
        if data["scopes"] == "ensembl.gene":
            rows = self.ensembl_rows
        elif "pathway." in data["fields"]:
            rows = self.pathway_rows
        else:
            rows = self.identity_rows
        wanted = data["q"].split(",")
        return [row for q in wanted for row in rows if row["query"] == q]

    @property
    def mygene_calls(self):
        return [c for c in self.calls if c["url"] == f"{MYGENE}/query"]


@pytest.fixture
def annotator():
    return GeneAnnotator(organism="human", rate_limit_delay=0)


@pytest.fixture
def mygene20():
    return RecordedMyGene(
        pathway_rows=_load("mygene_query_post_20.json"),
        identity_rows=_load("mygene_query_post_identity.json"),
        ensembl_rows=_load("mygene_query_post_ensembl.json"),
    )


@pytest.fixture
def reactome_index():
    return ReactomeIndex.from_dir(FIXTURES / "reactome98", release="98")


def _patch_post(fake):
    return patch("embpy.resources.gene.annotator._post_json", side_effect=fake)


# =====================================================================
# MyGene batch (T14)
# =====================================================================


class TestPathwaysBatch:
    def test_one_form_post_per_chunk_with_contract_fields(self, annotator, mygene20):
        with _patch_post(mygene20), patch("embpy.resources.gene.annotator._get_json") as get:
            result = annotator.get_pathways_batch(GENES_20)
        assert get.call_count == 0
        assert len(mygene20.mygene_calls) == 1
        data = mygene20.mygene_calls[0]["data"]
        assert data["q"] == ",".join(GENES_20)
        assert data["scopes"] == "symbol"
        assert data["species"] == "human"
        assert data["fields"] == "symbol,alias,entrezgene,pathway.kegg,pathway.wikipathways"
        assert data["fields"] == MYGENE_PATHWAY_FIELDS
        assert "pathway.reactome" not in data["fields"]
        assert list(result) == GENES_20

    def test_exact_symbol_hit_with_pathways(self, annotator, mygene20):
        with _patch_post(mygene20):
            egfr = annotator.get_pathways_batch(["EGFR"])["EGFR"]
        assert egfr["status"] == "found" and egfr["found"] is True
        assert egfr["symbol"] == "EGFR"
        assert egfr["entrezgene"] == "1956"
        assert egfr["renamed_from"] is None
        assert len(egfr["kegg"]) == 44
        assert len(egfr["wikipathways"]) == 40
        assert egfr["kegg"][0] == {"id": "hsa04010", "name": "MAPK signaling pathway - Homo sapiens (human)"}
        assert "reactome" not in egfr

    def test_cyp2d6_picks_entrez_1565_over_promoter_locus(self, annotator):
        # Wider scopes return LOC110740340 ("CYP2D6 promoter"), LOC110740341 and
        # LOC107987479 (alias "CYP2D6") besides the real gene.
        fake = RecordedMyGene(pathway_rows=_load("mygene_query_post_cyp2d6_symbol_alias_name.json"))
        with _patch_post(fake):
            cyp = annotator.get_pathways_batch(["CYP2D6"])["CYP2D6"]
        assert cyp["status"] == "found"
        assert cyp["entrezgene"] == "1565"
        assert cyp["symbol"] == "CYP2D6"
        assert cyp["renamed_from"] is None

    def test_cyp2d6_in_symbol_scope_batch(self, annotator, mygene20):
        with _patch_post(mygene20):
            cyp = annotator.get_pathways_batch(GENES_20)["CYP2D6"]
        assert (cyp["status"], cyp["entrezgene"]) == ("found", "1565")
        assert len(cyp["kegg"]) == 3

    def test_only_promoter_hits_is_identity_mismatch(self, annotator):
        fake = RecordedMyGene(pathway_rows=_load("mygene_query_post_cyp2d6_name_scope.json"))
        with _patch_post(fake):
            cyp = annotator.get_pathways_batch(["CYP2D6"])["CYP2D6"]
        assert cyp["status"] == "identity_mismatch"
        assert cyp["found"] is False
        assert cyp["kegg"] == [] and cyp["wikipathways"] == []
        assert {c["symbol"] for c in cyp["candidates"]} == {"LOC110740340", "LOC110740341"}

    @pytest.mark.parametrize(
        ("query", "symbol", "entrez"),
        [
            ("ATP5B", "ATP5F1B", "506"),
            ("ATP5A1", "ATP5F1A", "498"),
            ("ATP5C1", "ATP5F1C", "509"),
            ("WARS", "WARS1", "7453"),
            ("MKL1", "MRTFA", "57591"),
            ("GSG2", "HASPIN", "83903"),
        ],
    )
    def test_hgnc_renames_are_accepted_with_renamed_from(self, annotator, mygene20, query, symbol, entrez):
        with _patch_post(mygene20):
            rec = annotator.get_pathways_batch(GENES_20)[query]
        assert rec["status"] == "found"
        assert rec["symbol"] == symbol
        assert rec["entrezgene"] == entrez
        assert rec["renamed_from"] == query
        assert rec["match"] == "alias"

    def test_alias_as_string_and_as_list(self, annotator, mygene20):
        rows = {r["query"]: r for r in _load("mygene_query_post_20.json")}
        assert isinstance(rows["GSG2"]["alias"], str)
        assert isinstance(rows["ATP5B"]["alias"], list)
        with _patch_post(mygene20):
            result = annotator.get_pathways_batch(["GSG2", "ATP5B"])
        assert result["GSG2"]["renamed_from"] == "GSG2"
        assert result["ATP5B"]["renamed_from"] == "ATP5B"

    def test_gnrhr2_prefers_the_numeric_entrez_hit(self, annotator, mygene20):
        rows = [r for r in _load("mygene_query_post_20.json") if r["query"] == "GNRHR2"]
        assert {r["_id"] for r in rows} == {"114814", "ENSG00000211451"}
        with _patch_post(mygene20):
            rec = annotator.get_pathways_batch(["GNRHR2"])["GNRHR2"]
        assert rec["status"] == "found"
        assert rec["mygene_id"] == "114814"
        assert rec["entrezgene"] == "114814"

    @pytest.mark.parametrize(
        ("query", "entrez"), [("GJA10", "84694"), ("TCF4", "6925"), ("SLC23A1", "9963")],
    )
    def test_multi_hit_query_takes_the_exact_symbol(self, annotator, mygene20, query, entrez):
        with _patch_post(mygene20):
            rec = annotator.get_pathways_batch(GENES_20)[query]
        assert rec["status"] == "found"
        assert rec["symbol"] == query
        assert rec["entrezgene"] == entrez

    def test_notfound_row_gives_not_found(self, annotator, mygene20):
        with _patch_post(mygene20):
            rec = annotator.get_pathways_batch(GENES_20)["NOTAREALGENE1"]
        assert rec["status"] == "not_found"
        assert rec["found"] is False
        assert rec["entrezgene"] is None
        assert rec["kegg"] == [] and rec["wikipathways"] == []

    def test_found_gene_without_pathways_stays_found(self, annotator, mygene20):
        with _patch_post(mygene20):
            result = annotator.get_pathways_batch(["ATF7", "AURKC"])
        for gene in ("ATF7", "AURKC"):
            assert result[gene]["status"] == "found"
            assert result[gene]["found"] is True
            assert result[gene]["kegg"] == [] and result[gene]["wikipathways"] == []

    def test_single_dict_pathway_is_wrapped(self, annotator, mygene20):
        with _patch_post(mygene20):
            result = annotator.get_pathways_batch(["ATP5C1", "WARS"])
        assert len(result["ATP5C1"]["wikipathways"]) == 1
        assert len(result["WARS"]["kegg"]) == 1
        assert set(result["WARS"]["kegg"][0]) == {"id", "name"}

    def test_ensembl_ids_use_the_ensembl_scope(self, annotator, mygene20):
        ids = ["ENSG00000141510", "ENSG00000211451.3", "ENSG00000000000"]
        with _patch_post(mygene20):
            result = annotator.get_pathways_batch(ids)
        (call,) = mygene20.mygene_calls
        assert call["data"]["scopes"] == "ensembl.gene"
        assert call["data"]["q"] == "ENSG00000141510,ENSG00000211451,ENSG00000000000"
        assert "ensembl.gene" in call["data"]["fields"].split(",")
        assert (result["ENSG00000141510"]["status"], result["ENSG00000141510"]["entrezgene"]) == ("found", "7157")
        assert result["ENSG00000141510"]["symbol"] == "TP53"
        # Ensembl-only document: found, but no Entrez ID to join on.
        assert result["ENSG00000211451.3"]["status"] == "found"
        assert result["ENSG00000211451.3"]["entrezgene"] is None
        assert result["ENSG00000000000"]["status"] == "not_found"

    def test_mixed_inputs_send_one_post_per_scope(self, annotator, mygene20):
        with _patch_post(mygene20):
            result = annotator.get_pathways_batch(["TP53", "ENSG00000141510"])
        assert [c["data"]["scopes"] for c in mygene20.mygene_calls] == ["symbol", "ensembl.gene"]
        assert result["TP53"]["entrezgene"] == result["ENSG00000141510"]["entrezgene"] == "7157"

    def test_invalid_inputs_are_not_sent(self, annotator, mygene20):
        with _patch_post(mygene20):
            result = annotator.get_pathways_batch(["", "EGFR,TP53", "EGFR"])
        assert mygene20.mygene_calls[0]["data"]["q"] == "EGFR"
        assert result[""]["status"] == "not_found" and result[""]["reason"] == "invalid_query"
        assert result["EGFR,TP53"]["status"] == "not_found"
        assert result["EGFR"]["status"] == "found"

    def test_chunk_size_validation(self, annotator):
        with pytest.raises(ValueError):
            annotator.get_pathways_batch(["EGFR"], chunk_size=0)


class TestBatchFailures:
    def test_500_on_one_chunk_fails_exactly_that_chunk(self, annotator):
        """Real ``_post_json`` with a 500 on the second POST (requests.post mocked)."""
        rows = _load("mygene_query_post_20.json")

        class FakeResponse:
            def __init__(self, status, payload=None):
                self.status_code = status
                self._payload = payload

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise requests.HTTPError(f"{self.status_code} Server Error")

            def json(self):
                return self._payload

        calls = []

        def fake_post(url, data=None, json=None, timeout=None):
            calls.append(data)
            if len(calls) == 2:
                return FakeResponse(500)
            wanted = data["q"].split(",")
            return FakeResponse(200, [r for q in wanted for r in rows if r["query"] == q])

        with patch.object(gene_annotator_module.requests, "post", side_effect=fake_post):
            result = annotator.get_pathways_batch(GENES_20, chunk_size=5)
        assert len(calls) == 4
        failed = {g for g, r in result.items() if r["status"] == "batch_failed"}
        assert failed == set(GENES_20[5:10])
        for gene in GENES_20[5:10]:
            assert result[gene]["found"] is False
            assert result[gene]["reason"] == "request_failed"
        assert result["EGFR"]["status"] == "found"
        assert result["GJA10"]["status"] == "found"
        assert result["NOTAREALGENE1"]["status"] == "not_found"

    def test_unpatched_helper_none_is_batch_failed_not_empty(self, annotator):
        with patch("embpy.resources.gene.annotator._post_json", return_value=None):
            result = annotator.get_pathways_batch(["EGFR", "TP53"])
        assert {r["status"] for r in result.values()} == {"batch_failed"}
        assert all(r["found"] is False for r in result.values())

    def test_non_list_payload_is_batch_failed(self, annotator):
        with patch("embpy.resources.gene.annotator._post_json", return_value={"success": False}):
            rec = annotator.get_pathways_batch(["EGFR"])["EGFR"]
        assert (rec["status"], rec["reason"]) == ("batch_failed", "unexpected_payload")

    def test_query_missing_from_reply_is_batch_failed(self, annotator):
        rows = [r for r in _load("mygene_query_post_20.json") if r["query"] == "EGFR"]
        with patch("embpy.resources.gene.annotator._post_json", return_value=rows):
            result = annotator.get_pathways_batch(["EGFR", "TP53"])
        assert result["EGFR"]["status"] == "found"
        assert (result["TP53"]["status"], result["TP53"]["reason"]) == ("batch_failed", "missing_from_response")

    def test_exception_from_patched_helper_propagates(self, annotator):
        class TransportError(RuntimeError):
            pass

        with patch("embpy.resources.gene.annotator._post_json", side_effect=TransportError("budget")):
            with pytest.raises(TransportError):
                annotator.get_pathways_batch(["EGFR"])


class TestPostJsonHelper:
    def test_form_data_is_sent_as_data(self):
        with patch.object(gene_annotator_module.requests, "post") as post:
            post.return_value.json.return_value = []
            post.return_value.raise_for_status.return_value = None
            assert _post_json("https://mygene.info/v3/query", data={"q": "EGFR"}) == []
        assert post.call_args.kwargs["data"] == {"q": "EGFR"}
        assert "json" not in post.call_args.kwargs

    def test_json_body_is_still_supported(self):
        with patch.object(gene_annotator_module.requests, "post") as post:
            post.return_value.json.return_value = {"data": {}}
            post.return_value.raise_for_status.return_value = None
            assert _post_json("https://example.org/graphql", json_data={"query": "q"}) == {"data": {}}
        assert post.call_args.kwargs["json"] == {"query": "q"}

    def test_failure_returns_none(self):
        with patch.object(gene_annotator_module.requests, "post", side_effect=requests.ConnectionError("x")):
            assert _post_json("https://mygene.info/v3/query", data={"q": "EGFR"}) is None


# =====================================================================
# get_pathways (single gene, legacy shape) and the Reactome join
# =====================================================================


class TestPathways:
    def test_get_pathways_goes_through_the_batch_post(self, annotator, mygene20):
        with _patch_post(mygene20), patch("embpy.resources.gene.annotator._get_json") as get:
            result = annotator.get_pathways("ATP5B")
        assert get.call_count == 0
        assert len(mygene20.mygene_calls) == 1
        assert set(result) == {"reactome", "kegg", "wikipathways"}
        assert len(result["kegg"]) == 9
        assert len(result["wikipathways"]) == 2
        assert result["reactome"] == []  # no ReactomeIndex given; MyGene is not a Reactome source

    def test_get_pathways_takes_reactome_from_the_index(self, annotator, mygene20, reactome_index):
        with _patch_post(mygene20):
            result = annotator.get_pathways("ATP5B", reactome_index=reactome_index)
        assert [p["id"] for p in result["reactome"]] == [
            "R-HSA-1268020", "R-HSA-163210", "R-HSA-2151201", "R-HSA-8949613", "R-HSA-9837999",
        ]

    def test_get_pathways_failed_request_gives_empty_lists(self, annotator):
        with patch("embpy.resources.gene.annotator._post_json", return_value=None):
            result = annotator.get_pathways("EGFR")
        assert result == {"reactome": [], "kegg": [], "wikipathways": []}

    def test_reactome_batch_joins_on_entrez(self, annotator, mygene20, reactome_index):
        genes = ["TP53", "ATP5B", "GNRHR2", "NOTAREALGENE1", "ENSG00000211451"]
        with _patch_post(mygene20):
            result = annotator.get_reactome_pathways_batch(genes, reactome_index=reactome_index)
        assert result["TP53"]["status"] == "success"
        assert result["TP53"]["entrezgene"] == "7157"
        assert len(result["TP53"]["pathways"]) == 46
        assert result["TP53"]["release"] == "98"
        assert {t["name"] for t in result["ATP5B"]["top_level"]} == {
            "Metabolism", "Metabolism of proteins", "Organelle biogenesis and maintenance",
            "Protein localization",
        }
        assert result["GNRHR2"]["status"] == "not_in_reactome"  # Entrez 114814 has no Reactome rows
        assert result["NOTAREALGENE1"]["status"] == "not_found"
        # Ensembl-only hit: identity found, but never joined.
        assert result["ENSG00000211451"]["status"] == "no_entrez_id"
        assert result["ENSG00000211451"]["pathways"] == []

    def test_reactome_batch_reuses_gene_records(self, annotator, mygene20, reactome_index):
        with _patch_post(mygene20):
            records = annotator.get_pathways_batch(["CYP2D6"])
        with patch("embpy.resources.gene.annotator._post_json") as post:
            result = annotator.get_reactome_pathways_batch(
                ["CYP2D6"], reactome_index=reactome_index, gene_records=records,
            )
        assert post.call_count == 0
        assert {t["id"] for t in result["CYP2D6"]["top_level"]} == {"R-HSA-1430728", "R-HSA-9748784"}


# =====================================================================
# Tissue expression (mocked GTEx)
# =====================================================================


class TestTissueExpression:
    @patch("embpy.resources.gene.annotator._get_json")
    def test_get_tissue_expression(self, mock_get, annotator):
        mock_get.side_effect = [
            {"data": [{"gencodeId": "ENSG00000141510.18"}]},
            {"data": [
                {"tissueSiteDetailId": "Brain", "tissueSiteDetail": "Brain - Cortex", "median": 15.2, "numSamples": 100},
                {"tissueSiteDetailId": "Liver", "tissueSiteDetail": "Liver", "median": 5.1, "numSamples": 80},
            ]},
        ]
        tissues = annotator.get_tissue_expression("TP53")
        assert len(tissues) == 2
        assert tissues[0]["median_tpm"] >= tissues[1]["median_tpm"]

    @patch("embpy.resources.gene.annotator._get_json")
    def test_unresolvable_gene(self, mock_get, annotator):
        mock_get.return_value = {"hits": []}
        tissues = annotator.get_tissue_expression("FAKEGENE")
        assert tissues == []


# =====================================================================
# PPI partners (mocked STRING)
# =====================================================================


class TestProteinInteractions:
    @patch("embpy.resources.gene.annotator._get_text")
    @patch("embpy.resources.gene.annotator._get_json")
    def test_get_ppi(self, mock_json, mock_text, annotator):
        mock_json.return_value = {"hits": [{"symbol": "TP53"}]}
        mock_text.return_value = (
            "stringId_A\tstringId_B\tpreferredName_A\tpreferredName_B\tscore\tnscore\tfscore\tpscore\tescore\tdscore\ttscore\n"
            "9606.ENSP1\t9606.ENSP2\tTP53\tMDM2\t999\t0\t0\t0\t900\t800\t950\n"
            "9606.ENSP1\t9606.ENSP3\tTP53\tBRCA1\t850\t0\t0\t0\t700\t600\t800\n"
        )
        partners = annotator.get_protein_interactions("TP53", n_partners=5)
        assert len(partners) == 2
        assert partners[0]["partner"] == "MDM2"
        assert partners[0]["combined_score"] == 999

    @patch("embpy.resources.gene.annotator._get_text")
    @patch("embpy.resources.gene.annotator._get_json")
    def test_get_ppi_accepts_decimal_string_scores(self, mock_json, mock_text, annotator):
        mock_json.return_value = {"hits": [{"symbol": "TP53"}]}
        mock_text.return_value = (
            "stringId_A\tstringId_B\tpreferredName_A\tpreferredName_B\tscore\tnscore\tfscore\tpscore\tescore\tdscore\ttscore\n"
            "9606.ENSP1\t9606.ENSP2\tTP53\tMDM2\t0.999\t0\t0\t0\t0.900\t0.800\t0.950\n"
            "9606.ENSP1\t9606.ENSP3\tTP53\tBRCA1\t0.850\t0\t0\t0\t0.700\t0.600\t0.800\n"
        )
        partners = annotator.get_protein_interactions("TP53", n_partners=5)
        assert len(partners) == 2
        assert partners[0]["partner"] == "MDM2"
        assert partners[0]["combined_score"] == 999
        assert partners[1]["combined_score"] == 850


# =====================================================================
# Disease associations (mocked Open Targets)
# =====================================================================


OPEN_TARGETS_ROWS = {
    "data": {
        "target": {
            "associatedDiseases": {
                "rows": [
                    {"disease": {"id": "EFO_001", "name": "Breast cancer"}, "score": 0.95},
                    {"disease": {"id": "EFO_002", "name": "Lung cancer"}, "score": 0.88},
                ],
            },
        },
    },
}


class TestDiseaseAssociations:
    def test_get_diseases(self, annotator, mygene20):
        mygene20.other = OPEN_TARGETS_ROWS
        with _patch_post(mygene20):
            diseases = annotator.get_disease_associations("TP53")
        assert len(diseases) == 2
        assert diseases[0]["disease_name"] == "Breast cancer"
        identity, open_targets = mygene20.calls
        assert identity["data"]["fields"] == MYGENE_IDENTITY_FIELDS
        assert open_targets["json_data"]["variables"]["ensemblId"] == "ENSG00000141510"


# =====================================================================
# Identifier resolution and annotate (aggregation)
# =====================================================================


class TestIdentifierResolution:
    def test_symbol_to_ensembl_through_the_identity_check(self, annotator, mygene20):
        with _patch_post(mygene20):
            assert annotator._resolve_ensembl_id("ATP5B") == "ENSG00000110955"
            assert annotator._resolve_ensembl_id("NOTAREALGENE1") is None
            # cached: no second POST for the same gene
            assert annotator._resolve_ensembl_id("ATP5B") == "ENSG00000110955"
        assert len(mygene20.mygene_calls) == 2

    def test_ensembl_to_symbol(self, annotator, mygene20):
        with _patch_post(mygene20):
            assert annotator._resolve_symbol("ENSG00000141510") == "TP53"
        assert mygene20.mygene_calls[0]["data"]["scopes"] == "ensembl.gene"

    def test_failed_resolution_is_not_cached(self, annotator, mygene20):
        with patch("embpy.resources.gene.annotator._post_json", return_value=None):
            assert annotator._resolve_ensembl_id("TP53") is None
        with _patch_post(mygene20):
            assert annotator._resolve_ensembl_id("TP53") == "ENSG00000141510"


class TestAnnotate:
    def test_pathways_only(self, annotator, mygene20):
        with _patch_post(mygene20):
            result = annotator.annotate("TP53", sources="pathways")
        assert result["ensembl_id"] == "ENSG00000141510"
        assert len(result["pathways"]["kegg"]) == 49
        assert result["pathways"]["reactome"] == []

    def test_annotate_batch_fetches_pathways_in_one_post(self, annotator, mygene20, reactome_index):
        genes = ["TP53", "EGFR", "ATP5B", "NOTAREALGENE1"]
        with _patch_post(mygene20):
            result = annotator.annotate_batch(genes, sources="pathways", reactome_index=reactome_index)
        fields = [c["data"]["fields"] for c in mygene20.mygene_calls]
        assert fields.count(MYGENE_PATHWAY_FIELDS) == 1
        assert fields.count(MYGENE_IDENTITY_FIELDS) == 1
        assert len(result["TP53"]["pathways"]["reactome"]) == 46
        assert result["ATP5B"]["ensembl_id"] == "ENSG00000110955"
        assert result["NOTAREALGENE1"]["pathways"] == {"reactome": [], "kegg": [], "wikipathways": []}


# =====================================================================
# annotate_adata
# =====================================================================


class TestAnnotateAdata:
    @patch("embpy.resources.gene.annotator._get_json")
    @patch("embpy.resources.gene.annotator._get_text")
    def test_annotate_adata(self, mock_text, mock_json, annotator, mygene20):
        import pandas as pd
        from anndata import AnnData

        mock_json.return_value = None
        mock_text.return_value = ""

        adata = AnnData(
            obs=pd.DataFrame({"gene": ["TP53", "EGFR"]}),
        )
        adata.obs.index = ["c0", "c1"]

        with _patch_post(mygene20):
            result = annotator.annotate_adata(
                adata, column="gene", sources="pathways",
            )
        assert "gene_n_pathways" in result.obs.columns
        assert "gene_annotations" in result.uns
        assert list(result.obs["gene_n_pathways"]) == [49 + 59, 44 + 40]


# =====================================================================
# Opt-in live check of the recorded shapes
# =====================================================================


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("EMBPY_NETWORK_TESTS"), reason="set EMBPY_NETWORK_TESTS=1 to run")
def test_live_mygene_batch_keeps_recorded_keys():
    rows = _post_json(
        f"{MYGENE}/query",
        data={"q": "ATP5B,NOTAREALGENE1", "scopes": "symbol", "species": "human", "fields": MYGENE_PATHWAY_FIELDS},
    )
    assert isinstance(rows, list)
    by_query = {r["query"]: r for r in rows}
    assert {"query", "_id", "symbol", "entrezgene", "alias"} <= set(by_query["ATP5B"])
    assert by_query["NOTAREALGENE1"].get("notfound") is True
