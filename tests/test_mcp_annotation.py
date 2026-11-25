"""Tests for the MCP annotate_metadata tool: the records it stores and what its replies say.

No test touches the network. Gene pathways come from recorded MyGene rows and the
Reactome release 98 fixture; UniProt entries are mocked (trimmed to the fields the
tool reads).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import anndata
import pandas as pd
import pytest
import requests
from anndata import AnnData

from embpy.mcp import annotate_metadata
from embpy.resources._service_errors import report_service_error

_FIXTURES = Path(__file__).parent / "embpy" / "fixtures"
_GENE = _FIXTURES / "gene"


# ---------------------------------------------------------------------------
# Mocked and recorded services
# ---------------------------------------------------------------------------


def _go_xref(go_id: str, aspect: str, term: str) -> dict:
    return {"database": "GO", "id": go_id, "properties": [
        {"key": "GoTerm", "value": f"{aspect}:{term}"}, {"key": "GoEvidenceType", "value": "IDA:UniProtKB"}]}


def _entry(accession: str, gene: str, species: str, common: str, taxon: int, go: list) -> dict:
    """A UniProtKB entry (JSON) with only the fields annotate_metadata reads."""
    return {
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
        "primaryAccession": accession,
        "organism": {"scientificName": species, "commonName": common, "taxonId": taxon},
        "genes": [{"geneName": {"value": gene}}],
        "uniProtKBCrossReferences": [_go_xref(*g) for g in go],
    }


_P53_GO = [("GO:0005634", "C", "nucleus"), ("GO:0006915", "P", "apoptotic process"),
           ("GO:0003700", "F", "DNA-binding transcription factor activity")]
_EGFR_GO = [("GO:0005886", "C", "plasma membrane"),
            ("GO:0004714", "F", "transmembrane receptor protein tyrosine kinase activity")]


@pytest.fixture
def uniprot(monkeypatch):
    """UniProt entries and MyGene symbol lookups from mocked payloads; records every request."""
    from embpy.resources.protein import annotator as protein_annotator
    from embpy.resources.protein import resolver as protein_resolver

    state = {
        "entries": {
            "P04637": _entry("P04637", "TP53", "Homo sapiens", "Human", 9606, _P53_GO),
            "P00533": _entry("P00533", "EGFR", "Homo sapiens", "Human", 9606, _EGFR_GO),
            "P02340": _entry("P02340", "Tp53", "Mus musculus", "Mouse", 10090, _P53_GO),
        },
        # (MyGene species, upper-case symbol) -> Swiss-Prot accession
        "swissprot": {(9606, "TP53"): "P04637", (9606, "EGFR"): "P00533", (10090, "TRP53"): "P02340"},
        "down": set(),  # accessions whose entry request fails
        "requests": [],
    }

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    def http_get(url, *, params=None, timeout=30):
        state["requests"].append((url, dict(params or {})))
        if url == f"{protein_resolver.MYGENE_REST}/query":
            accession = state["swissprot"].get((params["species"], params["q"].upper()))
            return Response({"hits": [{"uniprot": {"Swiss-Prot": accession}}] if accession else []})
        if url == f"{protein_resolver.UNIPROT_REST}/uniprotkb/search":
            return Response({"results": []})
        raise AssertionError(f"unexpected request {url}")

    def get_json(url, params=None, timeout=30):
        state["requests"].append((url, dict(params or {})))
        accession = url.rsplit("/", 1)[-1].removesuffix(".json")
        if accession in state["down"]:
            report_service_error(url, requests.ConnectionError("connection reset"))
            return None
        return copy.deepcopy(state["entries"].get(accession))

    monkeypatch.setattr(protein_resolver, "_http_get", http_get)
    monkeypatch.setattr(protein_annotator, "_get_json", get_json)
    monkeypatch.setattr(protein_annotator.ProteinAnnotator, "_sleep", lambda self: None)
    return state


@pytest.fixture
def recorded_pathways(monkeypatch):
    """MyGene answers from recorded rows; Reactome is the release 98 fixture index."""
    from embpy.resources.gene import annotator as gene_annotator
    from embpy.resources.gene import reactome

    rows = json.loads((_GENE / "mygene_query_post_20.json").read_text(encoding="utf-8"))
    index = reactome.ReactomeIndex.from_dir(_GENE / "reactome98", release="98")

    def post_json(url, json_data=None, data=None, timeout=30):
        wanted = data["q"].split(",")
        return [row for q in wanted for row in rows if row["query"] == q]

    def load_reactome_index(directory=None, *, species="Homo sapiens", download=True):
        return index

    monkeypatch.setattr(gene_annotator, "_post_json", post_json)
    monkeypatch.setattr(reactome, "load_reactome_index", load_reactome_index)


def _write_obs(path: Path, **columns) -> str:
    n = len(next(iter(columns.values())))
    AnnData(obs=pd.DataFrame(columns, index=[f"c{i}" for i in range(n)])).write_h5ad(path)
    return str(path)


def _stored(path: str):
    saved = anndata.read_h5ad(path)
    record = saved.uns["embpy_annotations"]
    return saved, json.loads(record["results_json"]), record


def _go_ids(go_terms: dict) -> set[str]:
    return {term["id"] for terms in go_terms.values() for term in terms}


# ---------------------------------------------------------------------------
# 1. Records of earlier calls are kept
# ---------------------------------------------------------------------------


def test_adding_pathways_keeps_the_go_records(uniprot, recorded_pathways, tmp_path):
    """GO terms saved for TP53, then Reactome pathways added to that file: both are in the file."""
    genes = _write_obs(tmp_path / "genes.h5ad", gene=["TP53", "TP53"])
    go = annotate_metadata(path=genes, entity_type="gene", column="gene", annotation_types="go",
                           output_path=str(tmp_path / "go.h5ad"))
    assert go["status"] == "success", go
    both = annotate_metadata(path=go["output_path"], entity_type="gene", column="gene",
                             annotation_types="pathways", sources="reactome",
                             output_path=str(tmp_path / "go_pathways.h5ad"))
    assert both["status"] == "success", both
    n_pathways = both["records_preview"]["TP53"]["pathways"]["reactome"]["n"]

    saved, records, record = _stored(both["output_path"])
    assert _go_ids(records["TP53"]["go"]["uniprot"]) == {g[0] for g in _P53_GO}
    assert len(records["TP53"]["pathways"]["reactome"]) == n_pathways > 40
    assert [str(t) for t in record["annotation_types"]] == ["go", "pathways"]
    assert list(saved.obs["gene_go_n"]) == [3, 3]
    assert list(saved.obs["gene_pathways_n"]) == [n_pathways, n_pathways]
    provenance = json.loads(record["provenance_json"])
    assert provenance["gene"]["go"]["uniprot"] == {"organism": "human", "column": "gene"}
    assert provenance["gene"]["pathways"]["reactome"] == {"organism": "human", "column": "gene"}


def test_sources_of_one_type_are_kept_side_by_side(recorded_pathways, tmp_path):
    genes = _write_obs(tmp_path / "genes.h5ad", gene=["TP53"])
    reactome = annotate_metadata(path=genes, entity_type="gene", column="gene", annotation_types="pathways",
                                 sources="reactome", output_path=str(tmp_path / "reactome.h5ad"))
    kegg = annotate_metadata(path=reactome["output_path"], entity_type="gene", column="gene",
                             annotation_types="pathways", sources="kegg", output_path=str(tmp_path / "kegg.h5ad"))
    assert reactome["status"] == kegg["status"] == "success"

    saved, records, _ = _stored(kegg["output_path"])
    pathways = records["TP53"]["pathways"]
    assert set(pathways) == {"reactome", "kegg"}
    assert len(pathways["reactome"]) == reactome["records_preview"]["TP53"]["pathways"]["reactome"]["n"]
    assert len(pathways["kegg"]) == kegg["records_preview"]["TP53"]["pathways"]["kegg"]["n"]
    assert list(saved.obs["gene_pathways_n"]) == [len(pathways["reactome"]) + len(pathways["kegg"])]


def test_a_rerun_replaces_records_it_found_and_keeps_those_it_could_not_fetch(uniprot, tmp_path):
    genes = _write_obs(tmp_path / "genes.h5ad", gene=["TP53", "EGFR", "TP53"])
    first = annotate_metadata(path=genes, entity_type="gene", column="gene", annotation_types="go",
                              output_path=str(tmp_path / "first.h5ad"))
    assert first["status"] == "success", first

    # TP53's entry has changed since; EGFR's entry cannot be fetched this time.
    uniprot["entries"]["P04637"]["uniProtKBCrossReferences"] = [_go_xref("GO:0005634", "C", "nucleus")]
    uniprot["down"].add("P00533")
    again = annotate_metadata(path=first["output_path"], entity_type="gene", column="gene", annotation_types="go",
                              output_path=str(tmp_path / "again.h5ad"))
    assert again["status"] == "partial_success", again
    assert again["per_type"]["go"] == {"annotated": 1, "no_match": 0, "service_error": 1}

    saved, records, _ = _stored(again["output_path"])
    assert _go_ids(records["TP53"]["go"]["uniprot"]) == {"GO:0005634"}  # replaced, not added to
    assert _go_ids(records["EGFR"]["go"]["uniprot"]) == {g[0] for g in _EGFR_GO}  # the failed lookup kept them
    assert list(saved.obs["gene_go_n"]) == [1, 2, 1]
