"""Tests for the MCP annotate_metadata tool: the records it stores and what its replies say.

No test touches the network. Gene pathways come from recorded MyGene rows and the
Reactome release 98 fixture; UniProt entries are mocked (trimmed to the fields the
tool reads); the KEGG reference is built from the synthetic KEGG fixture tables and
the recorded PubChem payloads in ``embpy/fixtures/kegg``.
"""

from __future__ import annotations

import copy
import json
import os
import sys
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
_KEGG = _FIXTURES / "kegg"

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
ETHANOL = "CCO"
ETHANOL_INCHIKEY = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


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


# ---------------------------------------------------------------------------
# 2. KEGG pathways of molecules need a KEGG reference
# ---------------------------------------------------------------------------


@pytest.fixture
def offline_molecules(monkeypatch):
    """Molecule lookups must stay local: every web request is recorded and answered with nothing."""
    from embpy.resources.molecule import annotator as molecule_annotator

    requests_made: list[str] = []

    def refuse(url, *args, **kwargs):
        requests_made.append(url)
        return None

    for helper in ("_get_json", "_get_text", "_post_json"):
        monkeypatch.setattr(molecule_annotator, helper, refuse)
    return requests_made


@pytest.fixture(scope="module")
def kegg_reference(tmp_path_factory):
    """A KEGG reference folder (``<cache>/kegg``) built from the fixture tables, offline."""
    from embpy.resources.molecule import kegg_index

    folder = tmp_path_factory.mktemp("cache") / "kegg"
    table_of = {path: name for name, path in kegg_index.KEGG_TABLES.items()}

    def get_text(url, params=None, timeout=30):
        name = table_of[url[len(kegg_index.KEGG) + 1:]]
        return (_KEGG / kegg_index.KEGG_TABLE_FILES[name]).read_text(encoding="utf-8")

    def post_json(url, json_data=None, data=None, timeout=30):
        if url == kegg_index.PUBCHEM_SID_CIDS_URL:
            wanted = {int(x) for x in data["sid"].split(",")}
            info = json.loads((_KEGG / "pubchem_sid_cids_standardized.json").read_text(encoding="utf-8"))
            return {"InformationList": {"Information": [
                i for i in info["InformationList"]["Information"] if i["SID"] in wanted]}}
        wanted = {int(x) for x in data["cid"].split(",")}
        table = json.loads((_KEGG / "pubchem_cid_inchikey.json").read_text(encoding="utf-8"))
        return {"PropertyTable": {"Properties": [
            p for p in table["PropertyTable"]["Properties"] if p["CID"] in wanted]}}

    kegg_index.build_kegg_reference(folder, get_text=get_text, post_json=post_json,
                                    clock=lambda: 0.0, sleep=lambda seconds: None)
    return str(folder)


@pytest.mark.parametrize("configured_by", ["EMBPY_KEGG_REFERENCE", "EMBPY_CACHE"])
def test_kegg_pathways_of_a_molecule_in_the_reference(kegg_reference, configured_by, monkeypatch, tmp_path,
                                                      offline_molecules):
    if configured_by == "EMBPY_KEGG_REFERENCE":
        monkeypatch.setenv("EMBPY_KEGG_REFERENCE", kegg_reference)
        monkeypatch.setenv("EMBPY_CACHE", str(tmp_path / "empty_cache"))
    else:  # the default location, $EMBPY_CACHE/kegg
        monkeypatch.delenv("EMBPY_KEGG_REFERENCE", raising=False)
        monkeypatch.setenv("EMBPY_CACHE", str(Path(kegg_reference).parent))
    res = annotate_metadata(identifiers=[ASPIRIN], entity_type="molecule", annotation_types="pathways")
    assert res["status"] == "success", res
    kegg = res["records_preview"][ASPIRIN]["pathways"]["kegg"]["kegg_pathways"]
    # One compound pathway, four drug maps and three target pathways of the fixture's aspirin entries.
    assert kegg["n"] == 8
    assert {"id": "map94976", "name": "Synthetic reference map 94976"} in kegg["items"]
    assert os.path.normcase(res["kegg_reference"]["path"]) == os.path.normcase(os.path.abspath(kegg_reference))
    assert res["kegg_reference"]["release"].startswith("KEGG pathway 2026/11/09")
    assert offline_molecules == []


def test_no_kegg_pathways_after_a_completed_search(kegg_reference, monkeypatch, offline_molecules):
    from embpy.resources.molecule.kegg_index import KeggIndex

    looked_up = []
    real_lookup = KeggIndex.lookup

    def lookup(self, inchikey, **kwargs):
        looked_up.append(inchikey)
        return real_lookup(self, inchikey, **kwargs)

    monkeypatch.setattr(KeggIndex, "lookup", lookup)
    monkeypatch.setenv("EMBPY_KEGG_REFERENCE", kegg_reference)
    res = annotate_metadata(identifiers=[ETHANOL], entity_type="molecule", annotation_types="pathways")
    assert looked_up == [ETHANOL_INCHIKEY]  # the search ran
    assert res["status"] == "error"
    assert res["error_type"] == "NoAnnotationsFound"
    assert res["error"].startswith("No pathways found")


def test_without_a_kegg_reference_the_search_could_not_run(monkeypatch, tmp_path, offline_molecules):
    monkeypatch.delenv("EMBPY_KEGG_REFERENCE", raising=False)
    monkeypatch.setenv("EMBPY_CACHE", str(tmp_path / "empty_cache"))
    res = annotate_metadata(identifiers=[ETHANOL], entity_type="molecule", annotation_types="pathways")
    assert res["status"] == "error"
    assert res["error_type"] == "KeggReferenceUnavailable", res
    assert res["error"].startswith("The KEGG search could not run")
    assert "No pathways found" not in res["error"] + res.get("summary", "")
    assert "build_kegg_reference" in res["suggestion"]
    assert "EMBPY_KEGG_REFERENCE" in res["suggestion"]


def test_a_kegg_reference_that_cannot_be_loaded_is_reported(monkeypatch, tmp_path, offline_molecules):
    monkeypatch.setenv("EMBPY_KEGG_REFERENCE", str(tmp_path / "never_built"))
    res = annotate_metadata(identifiers=[ETHANOL], entity_type="molecule", annotation_types="pathways")
    assert res["error_type"] == "KeggReferenceUnavailable", res
    assert res["error"].startswith("The KEGG search could not run")
    assert "never_built" in res["error"]


# ---------------------------------------------------------------------------
# 3. UniProt entries must belong to the requested species
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entity_type", ["protein", "gene"])
def test_a_human_entry_asked_for_as_mouse_is_a_species_mismatch(uniprot, entity_type):
    res = annotate_metadata(identifiers=["P04637"], entity_type=entity_type, annotation_types="go",
                            organism="mouse")
    assert res["status"] == "error", res
    assert res["error_type"] == "SpeciesMismatch"
    assert "Homo sapiens" in res["error"] and "Mus musculus" in res["error"]
    assert res["per_type"]["go"] == {"annotated": 0, "no_match": 0, "service_error": 0, "species_mismatch": 1}
    assert res["species_mismatches"] == [{
        "entity": "P04637", "uniprot_accession": "P04637",
        "uniprot_species": "Homo sapiens", "uniprot_taxonomy_id": 9606,
        "requested_species": "Mus musculus", "requested_taxonomy_id": 10090,
    }]
    assert res["records_preview"] == {"P04637": {}}


def test_entries_of_the_requested_species_are_still_annotated(uniprot):
    res = annotate_metadata(identifiers=["P02340", "P04637"], entity_type="protein", annotation_types="go",
                            organism="mouse")
    assert res["status"] == "partial_success", res
    assert res["per_type"]["go"] == {"annotated": 1, "no_match": 0, "service_error": 0, "species_mismatch": 1}
    assert res["successful_items"] == ["P02340"]
    assert "P04637" in res["warning"] and "Homo sapiens" in res["warning"]

    human = annotate_metadata(identifiers=["P04637"], entity_type="protein", annotation_types="go")
    assert human["status"] == "success", human
    assert "species_mismatches" not in human
    assert human["per_type"]["go"] == {"annotated": 1, "no_match": 0, "service_error": 0}


@pytest.mark.parametrize("entity_type", ["protein", "gene"])
def test_gene_symbols_are_looked_up_in_the_requested_species(uniprot, entity_type):
    mouse = annotate_metadata(identifiers=["Trp53"], entity_type=entity_type, annotation_types="go",
                              organism="mouse")
    assert mouse["status"] == "success", mouse
    human_symbol = annotate_metadata(identifiers=["TP53"], entity_type=entity_type, annotation_types="go",
                                     organism="mouse")
    assert human_symbol["error_type"] == "NoAnnotationsFound", human_symbol
    assert {p["species"] for url, p in uniprot["requests"] if url.endswith("/query")} == {10090}
    assert not any(url.endswith("/P04637.json") for url, _ in uniprot["requests"])


# ---------------------------------------------------------------------------
# 4. Install hints name the missing package
# ---------------------------------------------------------------------------


class _NoRDKit:
    """Import hook that makes RDKit look uninstalled ("No module named 'rdkit'")."""

    def find_spec(self, name, path=None, target=None):
        if name == "rdkit" or name.startswith("rdkit."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


@pytest.fixture
def rdkit_missing(monkeypatch):
    for name in [m for m in sys.modules if m == "rdkit" or m.startswith("rdkit.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [_NoRDKit(), *sys.meta_path])


def test_missing_rdkit_names_rdkit(rdkit_missing, offline_molecules):
    res = annotate_metadata(identifiers=[ETHANOL], entity_type="molecule", annotation_types="properties")
    assert res["status"] == "error"
    assert res["error_type"] == "ModuleNotFoundError", res
    assert "pip install rdkit" in res["suggestion"]
    assert "pertpy" not in res["suggestion"]
    assert offline_molecules == []  # nothing was looked up online instead


def test_cell_line_annotation_still_names_pertpy(monkeypatch):
    import embpy.tl as tl

    def annotate_cell_lines(adata, query_id):
        raise ModuleNotFoundError("No module named 'pertpy'", name="pertpy")

    monkeypatch.setattr(tl, "annotate_cell_lines", annotate_cell_lines)
    res = annotate_metadata(identifiers=["A549"], entity_type="cell_line")
    assert res["error_type"] == "ModuleNotFoundError"
    assert "embpy[pertpy]" in res["suggestion"]


def test_a_package_missing_during_a_lookup_is_not_a_service_outage(monkeypatch):
    from embpy.resources.gene.annotator import GeneAnnotator

    def get_pathways(self, gene, *, reactome_index=None):
        raise ModuleNotFoundError("No module named 'pyensembl'", name="pyensembl")

    monkeypatch.setattr(GeneAnnotator, "get_pathways", get_pathways)
    res = annotate_metadata(identifiers=["TP53"], entity_type="gene", annotation_types="pathways", sources="kegg")
    assert res["error_type"] == "ModuleNotFoundError", res
    assert "Couldn't reach" not in res["error"]
    assert 'pip install "embpy[genome]"' in res["suggestion"]


# ---------------------------------------------------------------------------
# 5. Identifiers without output_path write nothing, and the reply says so
# ---------------------------------------------------------------------------


def test_identifiers_without_output_path_report_no_new_columns(recorded_pathways):
    res = annotate_metadata(identifiers=["TP53"], entity_type="gene", annotation_types="pathways",
                            sources="reactome")
    assert res["status"] == "success", res
    assert res["output_path"] is None
    assert res["new_columns_added"] == [] and res["added_columns"] == []


def test_identifiers_with_output_path_report_the_columns_written(recorded_pathways, tmp_path):
    out = str(tmp_path / "tp53.h5ad")
    res = annotate_metadata(identifiers=["TP53"], entity_type="gene", annotation_types="pathways",
                            sources="reactome", output_path=out)
    assert res["new_columns_added"] == ["gene_pathways_n"]
    assert "gene_pathways_n" in anndata.read_h5ad(out).obs


# ---------------------------------------------------------------------------
# 6. Each identifier column keeps its own record counts
# ---------------------------------------------------------------------------


def test_a_second_identifier_column_gets_its_own_count_column(uniprot, tmp_path):
    """Perturbed gene and target gene in one file: annotating the second keeps the first's counts."""
    pairs = _write_obs(tmp_path / "pairs.h5ad", gene=["TP53", "TP53", "EGFR"], target=["EGFR", "EGFR", "EGFR"])
    first = annotate_metadata(path=pairs, entity_type="gene", column="gene", annotation_types="go",
                              output_path=str(tmp_path / "gene.h5ad"))
    assert first["new_columns_added"] == ["gene_go_n"]
    second = annotate_metadata(path=first["output_path"], entity_type="gene", column="target",
                               annotation_types="go", output_path=str(tmp_path / "target.h5ad"))
    assert second["status"] == "success", second
    assert second["new_columns_added"] == ["gene_go_n_target"]

    saved, _, record = _stored(second["output_path"])
    assert list(saved.obs["gene_go_n"]) == [3, 3, 2]  # still the counts of column 'gene'
    assert list(saved.obs["gene_go_n_target"]) == [2, 2, 2]
    assert json.loads(record["provenance_json"])["count_columns"] == {
        "gene_go_n": {"column": "gene", "entity_type": "gene", "annotation_type": "go"},
        "gene_go_n_target": {"column": "target", "entity_type": "gene", "annotation_type": "go"},
    }

    # Annotating either column again updates its own counts, under the same names.
    uniprot["entries"]["P00533"]["uniProtKBCrossReferences"] = [_go_xref("GO:0005886", "C", "plasma membrane")]
    gene_again = annotate_metadata(path=second["output_path"], entity_type="gene", column="gene",
                                   annotation_types="go", output_path=str(tmp_path / "gene_again.h5ad"))
    assert gene_again["new_columns_added"] == ["gene_go_n"]
    target_again = annotate_metadata(path=gene_again["output_path"], entity_type="gene", column="target",
                                     annotation_types="go", output_path=str(tmp_path / "target_again.h5ad"))
    assert target_again["new_columns_added"] == ["gene_go_n_target"]
    saved = anndata.read_h5ad(target_again["output_path"])
    assert sorted(c for c in saved.obs.columns if c.startswith("gene_go_n")) == ["gene_go_n", "gene_go_n_target"]
    assert list(saved.obs["gene_go_n"]) == [3, 3, 1]
    assert list(saved.obs["gene_go_n_target"]) == [1, 1, 1]


def test_a_second_smiles_column_gets_its_own_property_columns(tmp_path, offline_molecules):
    """Drug and vehicle SMILES in one file: the vehicle's properties do not replace the drug's."""
    wells = _write_obs(tmp_path / "wells.h5ad", drug=[ETHANOL, ASPIRIN], vehicle=["O", "O"])
    drug = annotate_metadata(path=wells, entity_type="molecule", column="drug", annotation_types="properties",
                             output_path=str(tmp_path / "drug.h5ad"))
    assert drug["status"] == "success", drug
    assert "mol_molecular_weight" in drug["new_columns_added"]
    drug_weights = list(anndata.read_h5ad(drug["output_path"]).obs["mol_molecular_weight"])

    vehicle = annotate_metadata(path=drug["output_path"], entity_type="molecule", column="vehicle",
                                annotation_types="properties", output_path=str(tmp_path / "vehicle.h5ad"))
    assert vehicle["status"] == "success", vehicle
    assert {"mol_properties_n_vehicle", "mol_molecular_weight_vehicle"} <= set(vehicle["new_columns_added"])
    assert "mol_molecular_weight" not in vehicle["new_columns_added"]

    saved, _, record = _stored(vehicle["output_path"])
    assert list(saved.obs["mol_molecular_weight"]) == drug_weights  # still the drug's
    water = list(saved.obs["mol_molecular_weight_vehicle"])
    assert water[0] == water[1] and water[0] < min(drug_weights)
    property_columns = json.loads(record["provenance_json"])["property_columns"]
    assert property_columns["mol_molecular_weight"] == {"column": "drug", "property": "molecular_weight"}
    assert property_columns["mol_molecular_weight_vehicle"] == {"column": "vehicle", "property": "molecular_weight"}

    # Annotating the drug column again writes the drug's columns, under the same names.
    again = annotate_metadata(path=vehicle["output_path"], entity_type="molecule", column="drug",
                              annotation_types="properties", output_path=str(tmp_path / "again.h5ad"))
    assert "mol_molecular_weight" in again["new_columns_added"]
    assert not any(c.endswith("_drug") for c in again["new_columns_added"])
    saved = anndata.read_h5ad(again["output_path"])
    assert list(saved.obs["mol_molecular_weight"]) == drug_weights
    assert list(saved.obs["mol_molecular_weight_vehicle"]) == water
    assert offline_molecules == []


def test_every_count_column_follows_the_stored_records(uniprot, tmp_path):
    """Both columns hold EGFR: looking it up again through one column updates the other's count too."""
    pairs = _write_obs(tmp_path / "pairs.h5ad", gene=["TP53", "EGFR"], target=["EGFR", "EGFR"])
    gene = annotate_metadata(path=pairs, entity_type="gene", column="gene", annotation_types="go",
                             output_path=str(tmp_path / "gene.h5ad"))
    target = annotate_metadata(path=gene["output_path"], entity_type="gene", column="target",
                               annotation_types="go", output_path=str(tmp_path / "target.h5ad"))
    assert gene["status"] == target["status"] == "success"

    uniprot["entries"]["P00533"]["uniProtKBCrossReferences"] = [_go_xref("GO:0005886", "C", "plasma membrane")]
    again = annotate_metadata(path=target["output_path"], entity_type="gene", column="target",
                              annotation_types="go", output_path=str(tmp_path / "again.h5ad"))
    assert again["new_columns_added"] == ["gene_go_n_target"]

    saved, records, _ = _stored(again["output_path"])
    assert _go_ids(records["EGFR"]["go"]["uniprot"]) == {"GO:0005886"}
    assert list(saved.obs["gene_go_n_target"]) == [1, 1]
    assert list(saved.obs["gene_go_n"]) == [3, 1]  # EGFR's row in column 'gene' agrees with the records
