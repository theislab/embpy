"""Tests for embpy.resources.protein_annotator -- ProteinAnnotator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from embpy.resources.protein_annotator import ProteinAnnotator


MOCK_UNIPROT_ENTRY = {
    "primaryAccession": "P04637",
    "entryType": "UniProtKB reviewed (Swiss-Prot)",
    "proteinDescription": {
        "recommendedName": {"fullName": {"value": "Cellular tumor antigen p53"}},
    },
    "genes": [{"geneName": {"value": "TP53"}, "synonyms": [{"value": "P53"}]}],
    "organism": {"scientificName": "Homo sapiens", "taxonId": 9606},
    "sequence": {"length": 393},
    "entryAudit": {"lastAnnotationUpdateDate": "2025-01-01"},
    "comments": [
        {"commentType": "FUNCTION", "texts": [{"value": "Acts as a tumor suppressor"}]},
        {"commentType": "CATALYTIC ACTIVITY", "reaction": {"name": "DNA binding", "ecNumber": ""}},
        {"commentType": "PATHWAY", "texts": [{"value": "Cell cycle regulation"}]},
        {"commentType": "SUBCELLULAR LOCATION", "subcellularLocations": [
            {"location": {"value": "Nucleus"}, "topology": {}, "orientation": {}},
            {"location": {"value": "Cytoplasm"}, "topology": {}, "orientation": {}},
        ]},
        {"commentType": "DISEASE", "disease": {
            "diseaseId": "Li-Fraumeni syndrome",
            "description": "Cancer predisposition",
            "acronym": "LFS",
            "diseaseCrossReference": {"id": "151623"},
        }},
        {"commentType": "INTERACTION", "interactions": [
            {"interactantTwo": {"uniProtKBAccession": "Q00987", "geneName": "MDM2"}, "numberOfExperiments": 42},
        ]},
        {"commentType": "ALTERNATIVE PRODUCTS", "isoforms": [
            {"isoformIds": ["P04637-1"], "name": {"value": "Alpha"}, "isoformSequenceStatus": "Displayed"},
            {"isoformIds": ["P04637-2"], "name": {"value": "Beta"}, "isoformSequenceStatus": "Described"},
        ]},
    ],
    "features": [
        {"type": "Active site", "location": {"start": {"value": 120}, "end": {"value": 120}}, "description": "Proton donor"},
        {"type": "Binding site", "location": {"start": {"value": 248}, "end": {"value": 248}}, "description": "DNA", "ligand": {"name": "DNA"}},
        {"type": "Domain", "location": {"start": {"value": 94}, "end": {"value": 292}}, "description": "DNA-binding"},
        {"type": "Modified residue", "location": {"start": {"value": 15}, "end": {"value": 15}}, "description": "Phosphoserine"},
        {"type": "Natural variant", "location": {"start": {"value": 175}}, "description": "R->H", "alternativeSequence": {"originalSequence": "R", "alternativeSequences": ["H"]}},
        {"type": "Motif", "location": {"start": {"value": 339}, "end": {"value": 346}}, "description": "Nuclear localization signal"},
        {"type": "Region", "location": {"start": {"value": 1}, "end": {"value": 40}}, "description": "Transactivation domain"},
    ],
    "keywords": [
        {"id": "KW-0001", "name": "Tumor suppressor", "category": "Biological process"},
    ],
    "uniProtKBCrossReferences": [
        {"database": "GO", "id": "GO:0005634", "properties": [{"key": "GoTerm", "value": "C:nucleus"}, {"key": "GoEvidenceType", "value": "IDA"}]},
        {"database": "GO", "id": "GO:0003700", "properties": [{"key": "GoTerm", "value": "F:DNA-binding transcription factor activity"}, {"key": "GoEvidenceType", "value": "IDA"}]},
        {"database": "GO", "id": "GO:0006915", "properties": [{"key": "GoTerm", "value": "P:apoptotic process"}, {"key": "GoEvidenceType", "value": "IMP"}]},
        {"database": "IntAct", "id": "P04637", "properties": []},
        {"database": "STRING", "id": "9606.ENSP00000269305", "properties": []},
    ],
}


@pytest.fixture
def annotator():
    return ProteinAnnotator(organism="human", rate_limit_delay=0)


@pytest.fixture
def entry():
    return MOCK_UNIPROT_ENTRY


class TestEntryMetadata:
    def test_metadata(self, annotator, entry):
        meta = annotator.get_entry_metadata(entry)
        assert meta["accession"] == "P04637"
        assert meta["reviewed"] is True
        assert meta["protein_name"] == "Cellular tumor antigen p53"
        assert meta["organism"] == "Homo sapiens"
        assert meta["sequence_length"] == 393
        assert meta["gene_names"][0]["name"] == "TP53"


class TestFunction:
    def test_function(self, annotator, entry):
        func = annotator.get_function(entry)
        assert "Acts as a tumor suppressor" in func["function"]
        assert len(func["catalytic_activity"]) >= 1
        assert len(func["pathway"]) >= 1
        assert len(func["keywords"]) >= 1


class TestSubcellularLocation:
    def test_locations(self, annotator, entry):
        locs = annotator.get_subcellular_location(entry)
        assert len(locs) == 2
        assert locs[0]["location"] == "Nucleus"


class TestFunctionalSites:
    def test_sites(self, annotator, entry):
        sites = annotator.get_functional_sites(entry)
        assert len(sites["active_sites"]) == 1
        assert len(sites["binding_sites"]) == 1
        assert sites["binding_sites"][0]["ligand"] == "DNA"
        assert len(sites["motifs"]) == 1
        assert len(sites["regions"]) == 1


class TestDomains:
    def test_uniprot_domains(self, annotator, entry):
        domains = annotator.get_domains(entry)
        assert len(domains) == 1
        assert domains[0]["name"] == "DNA-binding"


class TestPTMs:
    def test_ptms(self, annotator, entry):
        ptms = annotator.get_ptms(entry)
        assert len(ptms) == 1
        assert ptms[0]["type"] == "Modified residue"
        assert ptms[0]["description"] == "Phosphoserine"


class TestDiseaseAssociations:
    def test_diseases(self, annotator, entry):
        diseases = annotator.get_disease_associations(entry)
        assert len(diseases) == 1
        assert diseases[0]["name"] == "Li-Fraumeni syndrome"

    def test_variants(self, annotator, entry):
        variants = annotator.get_variants(entry)
        assert len(variants) == 1
        assert variants[0]["original"] == "R"


class TestGOTerms:
    def test_go(self, annotator, entry):
        go = annotator.get_go_terms(entry)
        assert len(go["cellular_component"]) == 1
        assert len(go["molecular_function"]) == 1
        assert len(go["biological_process"]) == 1
        assert go["cellular_component"][0]["id"] == "GO:0005634"


class TestInteractions:
    def test_xrefs(self, annotator, entry):
        interactions = annotator.get_interaction_xrefs(entry)
        assert len(interactions) >= 3
        db_names = [i["database"] for i in interactions]
        assert "IntAct" in db_names
        assert "STRING" in db_names
        assert "UniProt" in db_names


class TestIsoformAnnotations:
    def test_isoforms(self, annotator, entry):
        isoforms = annotator.get_isoform_annotations(entry)
        assert len(isoforms) == 2
        assert isoforms[0]["name"] == "Alpha"
        assert isoforms[1]["name"] == "Beta"


class TestAnnotate:
    @patch("embpy.resources.protein_annotator._get_json")
    def test_annotate_all(self, mock_get, annotator):
        mock_get.return_value = MOCK_UNIPROT_ENTRY
        annotator._resolve_uniprot_accession = MagicMock(return_value="P04637")

        result = annotator.annotate("TP53", sources="all")
        assert "metadata" in result
        assert "function" in result
        assert "subcellular_location" in result
        assert "go_terms" in result
        assert result["metadata"]["reviewed"] is True

    @patch("embpy.resources.protein_annotator._get_json")
    def test_annotate_subset(self, mock_get, annotator):
        mock_get.return_value = MOCK_UNIPROT_ENTRY
        annotator._resolve_uniprot_accession = MagicMock(return_value="P04637")

        result = annotator.annotate("TP53", sources=["function", "diseases"])
        assert "function" in result
        assert "disease_associations" in result
        assert "go_terms" not in result


class TestAnnotateAdata:
    @patch("embpy.resources.protein_annotator._get_json")
    def test_annotate_adata(self, mock_get, annotator):
        import pandas as pd
        from anndata import AnnData

        mock_get.return_value = MOCK_UNIPROT_ENTRY
        annotator._resolve_uniprot_accession = MagicMock(return_value="P04637")

        adata = AnnData(obs=pd.DataFrame({"gene": ["TP53", "TP53"]}))
        adata.obs.index = ["c0", "c1"]

        result = annotator.annotate_adata(adata, column="gene", sources="all")
        assert "prot_reviewed" in result.obs.columns
        assert "prot_location" in result.obs.columns
        assert "prot_n_domains" in result.obs.columns
        assert "protein_annotations" in result.uns
