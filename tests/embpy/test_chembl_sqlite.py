"""Tests for the offline ChEMBL backend (``ChEMBLAnnotator(backend="sqlite")``).

A tiny synthetic database with the ChEMBL_37 column layout (checked against
the real ``chembl_37.db``) covers the joins: identity by standard InChIKey,
the molecule hierarchy, parent aggregation of salt-form rows, targets,
protein classes, activities and the derived ``drug`` record.  The shape of
every result is compared with what the REST path returns for the payloads
recorded in ``fixtures/chembl``.

When the real ChEMBL_37 SQLite file is present (the cluster reference
folder), ``TestAgainstRealRelease`` compares imatinib (CHEMBL941) and
aspirin (CHEMBL25) with the recorded REST payloads.  No test touches the
network.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from embpy.resources.molecule import chembl as chembl_mod
from embpy.resources.molecule.chembl import ChEMBLAnnotator

FIXTURES = Path(__file__).parent / "fixtures" / "chembl"
# Real-release comparison tests run only when EMBPY_CHEMBL_DB points at a
# local chembl_37.db (the SQLite file from the ChEMBL FTP release).
REAL_DB = Path(os.environ.get("EMBPY_CHEMBL_DB", "/nonexistent/chembl_37.db"))

IMATINIB_KEY = "KTUFNOKKBVMGRW-UHFFFAOYSA-N"
MESYLATE_KEY = "YLMAHDNUQAMNNX-UHFFFAOYSA-N"
ASPIRIN_KEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
IMATINIB_SMILES = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"

# Column lists as in chembl_37.db (PRAGMA table_info, 2026-11-11).
SCHEMA = {
    "molecule_dictionary": [
        "molregno", "pref_name", "chembl_id", "max_phase", "therapeutic_flag", "dosed_ingredient",
        "structure_type", "molecule_type", "first_approval", "oral", "parenteral", "topical",
        "black_box_warning", "natural_product", "first_in_class", "chirality", "prodrug",
        "inorganic_flag", "usan_year", "availability_type", "usan_stem", "polymer_flag",
        "usan_substem", "usan_stem_definition", "withdrawn_flag", "chemical_probe", "orphan",
        "veterinary",
    ],
    "compound_structures": ["molregno", "molfile", "standard_inchi", "standard_inchi_key", "canonical_smiles"],
    "molecule_hierarchy": ["molregno", "parent_molregno", "active_molregno"],
    "drug_mechanism": [
        "mec_id", "record_id", "molregno", "mechanism_of_action", "tid", "site_id", "action_type",
        "direct_interaction", "molecular_mechanism", "disease_efficacy", "mechanism_comment",
        "selectivity_comment", "binding_site_comment", "variant_id",
    ],
    "mechanism_refs": ["mecref_id", "mec_id", "ref_type", "ref_id", "ref_url"],
    "target_dictionary": ["tid", "target_type", "pref_name", "tax_id", "organism", "chembl_id", "species_group_flag"],
    "target_components": ["tid", "component_id", "targcomp_id", "homologue"],
    "component_sequences": [
        "component_id", "component_type", "accession", "sequence", "sequence_md5sum", "description",
        "tax_id", "organism", "db_source", "db_version",
    ],
    "component_synonyms": ["compsyn_id", "component_id", "component_synonym", "syn_type"],
    "component_class": ["component_id", "protein_class_id", "comp_class_id"],
    "protein_classification": [
        "protein_class_id", "parent_id", "pref_name", "short_name", "protein_class_desc", "definition", "class_level",
    ],
    "drug_indication": ["drugind_id", "record_id", "molregno", "max_phase_for_ind", "mesh_id", "mesh_heading", "efo_id", "efo_term"],
    "indication_refs": ["indref_id", "drugind_id", "ref_type", "ref_id", "ref_url"],
    "drug_warning": [
        "warning_id", "record_id", "molregno", "warning_type", "warning_class", "warning_description",
        "warning_country", "warning_year", "efo_term", "efo_id", "efo_id_for_warning_class",
    ],
    "warning_refs": ["warnref_id", "warning_id", "ref_type", "ref_id", "ref_url"],
    "metabolism": [
        "met_id", "drug_record_id", "substrate_record_id", "metabolite_record_id", "pathway_id", "pathway_key",
        "enzyme_name", "enzyme_tid", "met_conversion", "organism", "tax_id", "met_comment",
    ],
    "metabolism_refs": ["metref_id", "met_id", "ref_type", "ref_id", "ref_url"],
    "compound_records": ["record_id", "molregno", "doc_id", "compound_key", "compound_name", "src_id", "src_compound_id", "cidx"],
    "molecule_atc_classification": ["mol_atc_id", "level5", "molregno"],
    "atc_classification": [
        "who_name", "level1", "level2", "level3", "level4", "level5", "level1_description",
        "level2_description", "level3_description", "level4_description",
    ],
    "molecule_synonyms": ["molregno", "syn_type", "molsyn_id", "synonyms"],
    "activities": [
        "activity_id", "assay_id", "doc_id", "record_id", "molregno", "standard_relation", "standard_value",
        "standard_units", "standard_flag", "standard_type", "activity_comment", "data_validity_comment",
        "potential_duplicate", "pchembl_value", "bao_endpoint", "uo_units", "qudt_units", "toid", "upper_value",
        "standard_upper_value", "src_id", "type", "relation", "value", "units", "text_value",
        "standard_text_value", "action_type", "modality",
    ],
    "assays": [
        "assay_id", "doc_id", "description", "assay_type", "assay_test_type", "assay_category", "assay_organism",
        "assay_tax_id", "assay_strain", "assay_tissue", "assay_cell_type", "assay_subcellular_fraction", "tid",
        "relationship_type", "confidence_score", "src_id", "src_assay_id", "chembl_id", "cell_id", "bao_format",
        "tissue_id", "variant_id", "aidx", "assay_group",
    ],
    "docs": ["doc_id", "journal", "year", "volume", "issue", "first_page", "last_page", "pubmed_id", "doi", "chembl_id", "title"],
    "bioassay_ontology": ["bao_id", "label"],
    "ligand_eff": ["activity_id", "bei", "sei", "le", "lle"],
    "action_type": ["action_type", "description", "parent_type"],
    "products": [
        "dosage_form", "route", "trade_name", "approval_date", "ad_type", "oral", "topical", "parenteral",
        "black_box_warning", "applicant_full_name", "innovator_company", "product_id", "nda_type",
    ],
    "formulations": ["product_id", "ingredient", "strength", "record_id", "molregno", "formulation_id"],
    "version": ["name", "creation_date", "comments"],
}


def _molecule(molregno, chembl_id, pref_name, **kw):
    row = dict.fromkeys(SCHEMA["molecule_dictionary"])
    row.update(molregno=molregno, chembl_id=chembl_id, pref_name=pref_name, molecule_type="Small molecule",
               structure_type="MOL", **kw)
    return row


ROWS = {
    "molecule_dictionary": [
        _molecule(1, "CHEMBL941", "IMATINIB", max_phase=4.0, first_approval=2001, oral=1, parenteral=0, topical=0,
                  black_box_warning=0, natural_product=0, first_in_class=0, prodrug=0, orphan=1, chirality=2,
                  availability_type=1, chemical_probe=1, withdrawn_flag=0, therapeutic_flag=1, dosed_ingredient=0,
                  usan_stem="-tinib", usan_year=2001),
        _molecule(2, "CHEMBL1642", "IMATINIB MESYLATE", max_phase=4.0, first_approval=2001, oral=1),
        _molecule(3, "CHEMBL25", "ASPIRIN", max_phase=4.0, first_approval=1950, oral=1, topical=1,
                  natural_product=1, orphan=-1, availability_type=2, chirality=2),
        _molecule(4, "CHEMBL424", "SALICYLIC ACID", max_phase=4.0),
        _molecule(5, "CHEMBL2386595", None),
        _molecule(6, "CHEMBL122", "ROFECOXIB", max_phase=4.0, withdrawn_flag=1),
        _molecule(7, "CHEMBL9999", None),  # a tool compound with no clinical phase
    ],
    "compound_structures": [
        {"molregno": 1, "standard_inchi_key": IMATINIB_KEY, "canonical_smiles": IMATINIB_SMILES, "standard_inchi": "InChI=1S/imatinib"},
        {"molregno": 2, "standard_inchi_key": MESYLATE_KEY, "canonical_smiles": IMATINIB_SMILES + ".CS(=O)(=O)O"},
        {"molregno": 3, "standard_inchi_key": ASPIRIN_KEY, "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O"},
        {"molregno": 4, "standard_inchi_key": "YGSDEFSMJLZEOE-UHFFFAOYSA-N", "canonical_smiles": "O=C(O)c1ccccc1O"},
        {"molregno": 5, "standard_inchi_key": "KTUFNOKKBVMGRW-BMSJAHLVSA-N", "canonical_smiles": IMATINIB_SMILES},
        {"molregno": 7, "standard_inchi_key": "ZZZZZZZZZZZZZZ-UHFFFAOYSA-N", "canonical_smiles": "C"},
    ],
    "molecule_hierarchy": [
        {"molregno": 1, "parent_molregno": 1, "active_molregno": 1},
        {"molregno": 2, "parent_molregno": 1, "active_molregno": 1},
        {"molregno": 3, "parent_molregno": 3, "active_molregno": 3},
        {"molregno": 4, "parent_molregno": 4, "active_molregno": 4},
        {"molregno": 5, "parent_molregno": 1, "active_molregno": 1},
        # CHEMBL122 and CHEMBL9999 have no hierarchy row: their own parents.
    ],
    "drug_mechanism": [
        # Registered against the salt, as in ChEMBL.
        {"mec_id": 304, "record_id": 900, "molregno": 2, "mechanism_of_action": "Tyrosine-protein kinase ABL inhibitor",
         "tid": 10, "action_type": "INHIBITOR", "direct_interaction": 1, "molecular_mechanism": 1, "disease_efficacy": 1},
        {"mec_id": 1187, "record_id": 901, "molregno": 3, "mechanism_of_action": "Cyclooxygenase inhibitor",
         "tid": 11, "action_type": "INHIBITOR", "direct_interaction": 1, "molecular_mechanism": 1, "disease_efficacy": 1},
    ],
    "mechanism_refs": [
        {"mecref_id": 1, "mec_id": 304, "ref_type": "DailyMed", "ref_id": "setid=x", "ref_url": "http://dailymed/x"},
        {"mecref_id": 2, "mec_id": 304, "ref_type": "Wikipedia", "ref_id": "Imatinib", "ref_url": "http://wiki/x"},
    ],
    "target_dictionary": [
        {"tid": 10, "target_type": "SINGLE PROTEIN", "pref_name": "Tyrosine-protein kinase ABL1", "tax_id": 9606,
         "organism": "Homo sapiens", "chembl_id": "CHEMBL1862"},
        {"tid": 11, "target_type": "PROTEIN FAMILY", "pref_name": "Cyclooxygenase", "tax_id": 9606,
         "organism": "Homo sapiens", "chembl_id": "CHEMBL2094253"},
        {"tid": 12, "target_type": "SINGLE PROTEIN", "pref_name": "Receptor tyrosine-protein kinase erbB-2",
         "tax_id": 9606, "organism": "Homo sapiens", "chembl_id": "CHEMBL1824"},
        {"tid": 13, "target_type": "CELL-LINE", "pref_name": "K562", "tax_id": 9606, "organism": "Homo sapiens",
         "chembl_id": "CHEMBL614489"},
    ],
    "target_components": [
        {"tid": 10, "component_id": 173, "targcomp_id": 1},
        {"tid": 11, "component_id": 148, "targcomp_id": 2},
        {"tid": 11, "component_id": 162, "targcomp_id": 3},
        {"tid": 12, "component_id": 500, "targcomp_id": 4},
    ],
    "component_sequences": [
        {"component_id": 173, "component_type": "PROTEIN", "accession": "P00519", "description": "Tyrosine-protein kinase ABL1"},
        {"component_id": 148, "component_type": "PROTEIN", "accession": "P35354", "description": "Prostaglandin G/H synthase 2"},
        {"component_id": 162, "component_type": "PROTEIN", "accession": "P23219", "description": "Prostaglandin G/H synthase 1"},
        {"component_id": 500, "component_type": "PROTEIN", "accession": "P04626", "description": "erbB-2"},
    ],
    "component_synonyms": [
        {"compsyn_id": 1, "component_id": 173, "component_synonym": "ABL1", "syn_type": "GENE_SYMBOL"},
        {"compsyn_id": 2, "component_id": 173, "component_synonym": "c-ABL", "syn_type": "UNIPROT"},
        {"compsyn_id": 3, "component_id": 148, "component_synonym": "PTGS2", "syn_type": "GENE_SYMBOL"},
        {"compsyn_id": 4, "component_id": 162, "component_synonym": "PTGS1", "syn_type": "GENE_SYMBOL"},
        {"compsyn_id": 5, "component_id": 500, "component_synonym": "ERBB2", "syn_type": "GENE_SYMBOL"},
    ],
    "component_class": [
        {"component_id": 173, "protein_class_id": 130, "comp_class_id": 1},
        {"component_id": 148, "protein_class_id": 10, "comp_class_id": 2},
        {"component_id": 162, "protein_class_id": 10, "comp_class_id": 3},
        {"component_id": 500, "protein_class_id": 130, "comp_class_id": 4},
    ],
    "protein_classification": [
        {"protein_class_id": 10, "parent_id": 1, "pref_name": "Oxidoreductase", "protein_class_desc": "enzyme  reductase", "class_level": 2},
        {"protein_class_id": 130, "parent_id": 1285, "pref_name": "Tyrosine protein kinase Abl family",
         "protein_class_desc": "enzyme  kinase  protein kinase  tk  abl", "class_level": 5},
    ],
    "drug_indication": [
        {"drugind_id": 1, "record_id": 900, "molregno": 1, "max_phase_for_ind": 4.0, "mesh_id": "D015464",
         "mesh_heading": "Leukemia, Myelogenous, Chronic, BCR-ABL Positive", "efo_id": "EFO:0000339", "efo_term": "chronic myelogenous leukemia"},
        {"drugind_id": 2, "record_id": 900, "molregno": 2, "max_phase_for_ind": 3.0, "mesh_id": "D015464",
         "mesh_heading": "Leukemia, Myelogenous, Chronic, BCR-ABL Positive", "efo_id": "EFO:0000339", "efo_term": "chronic myelogenous leukemia"},
        {"drugind_id": 3, "record_id": 900, "molregno": 2, "max_phase_for_ind": 4.0, "mesh_id": "D046152",
         "mesh_heading": "Gastrointestinal Stromal Tumors", "efo_id": "EFO:0000505", "efo_term": "gastrointestinal stromal tumor"},
        {"drugind_id": 4, "record_id": 901, "molregno": 3, "max_phase_for_ind": 4.0, "mesh_id": "D010146",
         "mesh_heading": "Pain", "efo_id": "EFO:0003843", "efo_term": "pain"},
    ],
    "indication_refs": [
        {"indref_id": 1, "drugind_id": 1, "ref_type": "FDA", "ref_id": "a", "ref_url": "u"},
        {"indref_id": 2, "drugind_id": 2, "ref_type": "ClinicalTrials", "ref_id": "b", "ref_url": "u"},
        {"indref_id": 3, "drugind_id": 2, "ref_type": "ClinicalTrials", "ref_id": "c", "ref_url": "u"},
    ],
    "drug_warning": [
        {"warning_id": 1662, "record_id": 902, "molregno": 6, "warning_type": "Withdrawn", "warning_class": "cardiotoxicity",
         "warning_description": "Risk for heart attack", "warning_country": "Worldwide", "warning_year": 2004,
         "efo_term": "stroke", "efo_id": "EFO:0000712"},
    ],
    "warning_refs": [{"warnref_id": 1, "warning_id": 1662, "ref_type": "WHO", "ref_id": "x", "ref_url": "u"}],
    "metabolism": [
        {"met_id": 623, "drug_record_id": 100, "substrate_record_id": 100, "metabolite_record_id": 101,
         "pathway_id": 1, "enzyme_name": "Esterases", "met_conversion": "Ester hydrolysis", "organism": None},
    ],
    "metabolism_refs": [{"metref_id": 1, "met_id": 623, "ref_type": "DOI", "ref_id": "10.1/x", "ref_url": "u"}],
    "compound_records": [
        {"record_id": 100, "molregno": 3, "compound_name": "aspirin"},
        {"record_id": 101, "molregno": 4, "compound_name": "salicylic acid"},
        {"record_id": 900, "molregno": 2, "compound_name": "imatinib mesylate"},
    ],
    "molecule_atc_classification": [
        {"mol_atc_id": 1, "level5": "L01EA01", "molregno": 1},
        {"mol_atc_id": 2, "level5": "B01AC06", "molregno": 3},
        {"mol_atc_id": 3, "level5": "N02BA01", "molregno": 3},
        {"mol_atc_id": 4, "level5": "L01XX99", "molregno": 2},  # on the salt only
    ],
    "atc_classification": [
        {"who_name": "imatinib", "level1": "L", "level2": "L01", "level3": "L01E", "level4": "L01EA", "level5": "L01EA01",
         "level1_description": "ANTINEOPLASTIC AND IMMUNOMODULATING AGENTS", "level2_description": "ANTINEOPLASTIC AGENTS",
         "level3_description": "PROTEIN KINASE INHIBITORS", "level4_description": "BCR-ABL tyrosine kinase inhibitors"},
        {"who_name": "acetylsalicylic acid", "level1": "B", "level2": "B01", "level3": "B01A", "level4": "B01AC", "level5": "B01AC06",
         "level1_description": "BLOOD AND BLOOD FORMING ORGANS", "level2_description": "ANTITHROMBOTIC AGENTS",
         "level3_description": "ANTITHROMBOTIC AGENTS", "level4_description": "Platelet aggregation inhibitors excl. heparin"},
        {"who_name": "acetylsalicylic acid", "level1": "N", "level2": "N02", "level3": "N02B", "level4": "N02BA", "level5": "N02BA01",
         "level1_description": "NERVOUS SYSTEM", "level2_description": "ANALGESICS",
         "level3_description": "OTHER ANALGESICS AND ANTIPYRETICS", "level4_description": "Salicylic acid and derivatives"},
    ],
    "molecule_synonyms": [
        {"molregno": 1, "syn_type": "RESEARCH_CODE", "molsyn_id": 1, "synonyms": "STI-571"},
        {"molregno": 1, "syn_type": "INN", "molsyn_id": 2, "synonyms": "Imatinib"},
        {"molregno": 2, "syn_type": "TRADE_NAME", "molsyn_id": 3, "synonyms": "Gleevec"},
        {"molregno": 3, "syn_type": "FDA", "molsyn_id": 4, "synonyms": "Aspirin"},
    ],
    "activities": [
        {"activity_id": 24962908, "assay_id": 1, "doc_id": 1, "molregno": 1, "standard_relation": "=", "standard_value": 0.06,
         "standard_units": "nM", "standard_type": "IC50", "pchembl_value": 10.22, "potential_duplicate": 0, "action_type": "INHIBITOR"},
        {"activity_id": 2, "assay_id": 2, "doc_id": 1, "molregno": 2, "standard_relation": "=", "standard_value": 10.0,
         "standard_units": "nM", "standard_type": "IC50", "pchembl_value": 8.0, "potential_duplicate": 0},
        {"activity_id": 3, "assay_id": 2, "doc_id": 1, "molregno": 1, "standard_relation": "=", "standard_value": 100.0,
         "standard_units": "nM", "standard_type": "Ki", "pchembl_value": 7.0, "potential_duplicate": 1},
        {"activity_id": 4, "assay_id": 3, "doc_id": 1, "molregno": 1, "standard_relation": "=", "standard_value": 5.0,
         "standard_units": "nM", "standard_type": "IC50", "pchembl_value": 9.0},
        {"activity_id": 5, "assay_id": 2, "doc_id": 1, "molregno": 1, "standard_type": "Inhibition", "pchembl_value": None},
    ],
    "assays": [
        {"assay_id": 1, "doc_id": 1, "description": "Inhibition of HER2", "assay_type": "B", "tid": 12,
         "chembl_id": "CHEMBL5216307", "bao_format": "BAO_0000357"},
        {"assay_id": 2, "doc_id": 1, "description": "Inhibition of ABL1", "assay_type": "B", "tid": 10,
         "chembl_id": "CHEMBL111", "bao_format": "BAO_0000357"},
        {"assay_id": 3, "doc_id": 1, "description": "K562 growth", "assay_type": "F", "tid": 13,
         "chembl_id": "CHEMBL112", "bao_format": "BAO_0000219"},
    ],
    "docs": [{"doc_id": 1, "journal": "Eur J Med Chem", "year": 2022, "chembl_id": "CHEMBL5214926"}],
    "bioassay_ontology": [{"bao_id": "BAO_0000357", "label": "single protein format"},
                          {"bao_id": "BAO_0000219", "label": "cell-based format"}],
    "ligand_eff": [{"activity_id": 24962908, "bei": 20.71, "sei": 11.85, "le": 0.38, "lle": 5.63}],
    "action_type": [{"action_type": "INHIBITOR", "description": "Negatively effects the protein", "parent_type": "NEGATIVE MODULATOR"}],
    "products": [{"product_id": "N021588", "applicant_full_name": "Novartis Pharmaceuticals Corp", "trade_name": "GLEEVEC"}],
    "formulations": [{"product_id": "N021588", "ingredient": "IMATINIB MESYLATE", "record_id": 900, "molregno": 2, "formulation_id": 1}],
    "version": [
        {"name": "Bioassay Ontology 2.0", "creation_date": None, "comments": "BAO"},
        {"name": "ChEMBL_37", "creation_date": "2026-05-01 00:00:00.000000", "comments": "ChEMBL Release 37"},
        {"name": "ChEMBL_Structure_Pipeline 1.2.0", "creation_date": None, "comments": "pipeline"},
    ],
}


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("chembl") / "chembl_37.db"
    conn = sqlite3.connect(path)
    for table, columns in SCHEMA.items():
        conn.execute(f"CREATE TABLE {table} ({', '.join(columns)})")
        for row in ROWS.get(table, []):
            values = [row.get(c) for c in columns]
            conn.execute(f"INSERT INTO {table} VALUES ({','.join('?' * len(columns))})", values)
    conn.commit()
    conn.close()
    return path


@pytest.fixture()
def offline(db_path):
    """An offline annotator; any network call fails the test."""

    def no_network(*args, **kwargs):
        raise AssertionError(f"network call from the sqlite backend: {args}")

    with patch.object(chembl_mod, "_get_json", no_network), patch.object(chembl_mod.requests, "get", no_network):
        yield ChEMBLAnnotator(backend="sqlite", db_path=db_path)


def fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


# =====================================================================
# Opening the database
# =====================================================================


class TestOpening:
    def test_release_is_the_chembl_row_of_version(self, offline):
        # `version` lists BAO, EFO, RDKit, ... first; the release is ChEMBL_37.
        assert offline.release == "ChEMBL_37"

    def test_read_only_immutable_uri(self, offline):
        backend = offline._sql
        assert backend.uri.startswith("file:")
        assert backend.uri.endswith("?mode=ro&immutable=1")
        with pytest.raises(sqlite3.OperationalError):
            backend._conn().execute("CREATE TABLE x (a)")

    def test_backend_is_shared_between_instances(self, db_path):
        a = ChEMBLAnnotator(backend="sqlite", db_path=db_path)
        b = ChEMBLAnnotator(backend="sqlite", db_path=str(db_path))
        assert a._sql is b._sql

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ChEMBLAnnotator(backend="sqlite", db_path=tmp_path / "nope.db")

    def test_needs_a_path(self):
        with pytest.raises(ValueError):
            ChEMBLAnnotator(backend="sqlite")
        with pytest.raises(ValueError):
            ChEMBLAnnotator(backend="graphql")

    def test_threads_get_their_own_connection(self, offline):
        import threading

        conns = []

        def work():
            conns.append(offline._sql._conn())
            offline._sql.molecules_by_chembl_id(["CHEMBL25"])

        threads = [threading.Thread(target=work) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len({id(c) for c in conns}) == 3


# =====================================================================
# Identity
# =====================================================================


class TestIdentity:
    def test_chembl_id(self, offline):
        res = offline.resolve("CHEMBL1642")
        assert res.status == "matched"
        assert (res.molecule_chembl_id, res.parent_chembl_id) == ("CHEMBL1642", "CHEMBL941")

    def test_inchikey(self, offline):
        res = offline.resolve(MESYLATE_KEY)
        assert res.molecule_chembl_id == "CHEMBL1642"
        assert res.parent_chembl_id == "CHEMBL941"
        assert res.matched_inchikey == MESYLATE_KEY

    def test_smiles_goes_through_the_inchikey(self, offline):
        pytest.importorskip("rdkit")
        res = offline.resolve(IMATINIB_SMILES)
        assert res.matched_by == "standard_inchi_key"
        assert res.molecule_chembl_id == "CHEMBL941"
        assert res.inchikey == IMATINIB_KEY

    def test_names(self, offline):
        assert offline.resolve("imatinib").molecule_chembl_id == "CHEMBL941"
        gleevec = offline.resolve("gleevec")
        assert gleevec.matched_by == "synonym"
        assert gleevec.parent_chembl_id == "CHEMBL941"
        assert offline.resolve("no such drug").status == "no_match"

    def test_resolve_many(self, offline):
        res = offline.resolve_many([IMATINIB_KEY, ASPIRIN_KEY, "AAAAAAAAAAAAAA-UHFFFAOYSA-N", "CHEMBL1642", "CHEMBL0"])
        assert res[IMATINIB_KEY].parent_chembl_id == "CHEMBL941"
        assert res[ASPIRIN_KEY].molecule_chembl_id == "CHEMBL25"
        assert res["AAAAAAAAAAAAAA-UHFFFAOYSA-N"].status == "no_match"
        assert res["CHEMBL1642"].parent_chembl_id == "CHEMBL941"
        assert res["CHEMBL0"].status == "no_match"

    def test_record_shape_matches_rest(self, offline):
        rest = {m["molecule_chembl_id"]: m for m in fixture("molecule_chembl_id_in")["molecules"]}["CHEMBL941"]
        local = offline.get_molecule_record("CHEMBL941")
        missing = set(rest) - set(local)
        # molecule_properties (compound_properties), biotherapeutic and
        # helm_notation are not rebuilt; everything the annotator reads is.
        assert missing <= {"molecule_properties", "biotherapeutic", "helm_notation"}
        assert local["molecule_hierarchy"] == {
            "molecule_chembl_id": "CHEMBL941", "parent_chembl_id": "CHEMBL941", "active_chembl_id": "CHEMBL941",
        }
        assert set(local["molecule_structures"]) == {"canonical_smiles", "standard_inchi", "standard_inchi_key"}

    def test_records_many(self, offline):
        records = offline.get_molecule_records_many(["CHEMBL941", "CHEMBL25", "CHEMBL0"])
        assert records.status("CHEMBL941") == "success"
        assert records.status("CHEMBL0") == "not_found"


# =====================================================================
# Parent aggregation (T16)
# =====================================================================


class TestParentAggregation:
    def test_salt_mechanism_is_reported_on_the_parent(self, offline):
        mechs = offline.get_mechanisms("CHEMBL941")
        assert [m["mechanism"] for m in mechs] == ["Tyrosine-protein kinase ABL inhibitor"]
        assert mechs[0]["target_chembl_id"] == "CHEMBL1862"
        assert [r["type"] for r in mechs[0]["references"]] == ["DailyMed", "Wikipedia"]
        raw = offline._sql.mechanisms(["CHEMBL941"])["CHEMBL941"][0]
        assert raw["molecule_chembl_id"] == "CHEMBL1642"
        assert raw["parent_molecule_chembl_id"] == "CHEMBL941"

    def test_a_salt_identifier_resolves_to_the_parent_rows(self, offline):
        assert offline.get_mechanisms("CHEMBL1642") == offline.get_mechanisms("CHEMBL941")

    def test_a_salt_as_parent_has_no_family(self, offline):
        """REST semantics: parent_molecule_chembl_id=<salt> matches nothing."""
        assert offline._sql.mechanisms(["CHEMBL1642"]) == {"CHEMBL1642": []}

    def test_molecule_without_hierarchy_is_its_own_parent(self, offline):
        warnings = offline.get_warnings("CHEMBL122")
        assert warnings[0]["warning_type"] == "Withdrawn"
        assert warnings[0]["n_references"] == 1

    def test_indications_merge_across_forms(self, offline):
        ind = {i["efo_id"]: i for i in offline.get_indications("CHEMBL941", limit=None)}
        assert set(ind) == {"EFO:0000339", "EFO:0000505"}
        cml = ind["EFO:0000339"]
        assert cml["max_phase_for_indication"] == 4.0
        assert cml["n_references"] == 3

    def test_atc_codes_of_salts_join_the_parent(self, offline):
        record = offline.get_molecule_record("CHEMBL941")
        assert record["atc_classifications"] == ["L01EA01", "L01XX99"]
        # ... but a salt keeps only its own.
        assert offline.get_molecule_record("CHEMBL1642")["atc_classifications"] == ["L01XX99"]
        classes = offline.get_atc_classes("CHEMBL941")
        assert [c["code"] for c in classes] == ["L01EA01"]  # L01XX99 is not in atc_classification
        assert classes[0]["level4_description"] == "BCR-ABL tyrosine kinase inhibitors"

    def test_metabolism_goes_through_compound_records(self, offline):
        met = offline.get_metabolism("CHEMBL25")
        assert met[0]["metabolite_name"] == "SALICYLIC ACID"
        assert met[0]["metabolite_chembl_id"] == "CHEMBL424"
        assert met[0]["substrate_chembl_id"] == "CHEMBL25"
        assert met[0]["enzyme_name"] == "Esterases"
        assert met[0]["n_references"] == 1

    def test_forms(self, offline):
        forms = offline.get_molecule_forms("CHEMBL1642")
        assert {(f["molecule_chembl_id"], f["is_parent"]) for f in forms} == {
            ("CHEMBL941", True), ("CHEMBL1642", False), ("CHEMBL2386595", False),
        }

    def test_development_status(self, offline):
        dev = offline.get_development_status("CHEMBL1642")  # a salt reports its drug
        assert dev["molecule_chembl_id"] == "CHEMBL941"
        assert dev["development_phase"] == "approved"
        assert dev["oral"] is True
        assert dev["chemical_probe"] is True
        assert offline.get_development_status("CHEMBL25")["orphan"] is None

    def test_synonyms(self, offline):
        syn = offline.get_synonyms("CHEMBL941")
        assert syn[0] == {"name": "Imatinib", "type": "INN"}

    def test_approval_record(self, offline):
        rec = offline.get_approval_record("CHEMBL941")
        assert rec["applicants"] == ["Novartis Pharmaceuticals Corp"]
        assert rec["atc_codes"] == ["L01EA01", "L01XX99"]
        assert rec["first_approval"] == 2001
        assert rec["drug_type"] is None
        assert offline.get_approval_record("CHEMBL9999") == {}

    def test_analogs_are_not_available_offline(self, offline):
        assert offline.get_analogs("CHEMBL941") == {"status": "not_available_offline"}


# =====================================================================
# Activities and targets
# =====================================================================


class TestActivitiesAndTargets:
    def test_complete_retrieval_and_explicit_truncation(self, offline):
        limited = offline.get_activities("CHEMBL941", limit=1)
        assert len(limited) == 1
        assert offline.activity_fetch_metadata["CHEMBL941"]["truncated"] is True
        assert offline.activity_fetch_metadata["CHEMBL941"]["total_rows"] is None
        complete = offline.get_activities("CHEMBL941", limit=None)
        assert len(complete) > len(limited)
        assert complete[-1]["pchembl_value"] < limited[0]["pchembl_value"]
        assert offline.activity_fetch_metadata["CHEMBL941"]["total_rows"] == len(complete)
        assert offline.activity_fetch_metadata["CHEMBL941"]["truncated"] is False

    def test_top_n_by_pchembl_over_all_forms(self, offline):
        acts = offline.get_activities("CHEMBL941", limit=3)
        assert [a["pchembl_value"] for a in acts] == [10.22, 9.0, 8.0]
        top = acts[0]
        assert top["target_chembl_id"] == "CHEMBL1824"
        assert top["bao_label"] == "single protein format"
        assert top["document_year"] == 2022
        assert top["action_type"] == {
            "action_type": "INHIBITOR", "description": "Negatively effects the protein", "parent_type": "NEGATIVE MODULATOR",
        }
        assert top["ligand_efficiency"] == {"bei": "20.71", "le": "0.38", "lle": "5.63", "sei": "11.85"}

    def test_without_pchembl_filter(self, offline):
        assert len(offline.get_activities("CHEMBL941", limit=10, only_with_pchembl=False)) == 5

    def test_target_details(self, offline):
        details = offline.get_target_details(["CHEMBL1862", "CHEMBL2094253"])
        abl = details["CHEMBL1862"]
        assert abl["gene_symbol"] == "ABL1"
        assert abl["uniprot_accession"] == "P00519"
        assert abl["protein_class_path"] == ["enzyme", "kinase", "protein kinase", "tk", "abl"]
        assert abl["target_class"] == "enzyme"
        assert details["CHEMBL2094253"]["gene_symbols"] == ["PTGS2", "PTGS1"]

    def test_profile_and_selectivity(self, offline):
        profile = offline.get_target_profile("CHEMBL941")
        assert profile[0]["gene_symbol"] == "ERBB2"
        summary = offline.summarize_selectivity(profile)
        assert summary["primary_target_gene"] == "ERBB2"
        assert summary["n_protein_targets"] == 2


# =====================================================================
# Batch getters and shape parity with REST
# =====================================================================


class TestShapes:
    def test_many_getters(self, offline):
        mechs = offline.get_mechanisms_many(["CHEMBL941", "CHEMBL25", "CHEMBL1642"])
        assert len(mechs["CHEMBL941"]) == 1 and len(mechs["CHEMBL25"]) == 1
        assert mechs["CHEMBL1642"] == []
        assert offline.get_indications_many(["CHEMBL941"])["CHEMBL941"][0]["efo_id"]
        assert offline.get_warnings_many(["CHEMBL122"])["CHEMBL122"]
        assert offline.get_metabolism_many(["CHEMBL25"])["CHEMBL25"]
        assert len(offline.get_molecule_forms_many(["CHEMBL941"])["CHEMBL941"]) == 3

    @staticmethod
    def _rest_outputs():
        """The REST path run over the recorded payloads."""
        recorded = {m["molecule_chembl_id"]: m for m in fixture("molecule_chembl_id_in")["molecules"]}
        mech = fixture("mechanism_parent_in")
        ind = fixture("drug_indication_parent_in_trimmed")
        met = fixture("metabolism_drug_in")
        forms = fixture("molecule_form_parent_in")
        acts = fixture("activity_imatinib_top5")
        warn = fixture("drug_warning_parent_in")
        drug = fixture("drug_imatinib")

        def get_json(url, params=None, timeout=30):
            params = params or {}
            if "/molecule/" in url:
                return recorded.get(url.rsplit("/", 1)[1].removesuffix(".json"))
            for fragment, payload in (
                ("mechanism.json", mech), ("drug_indication.json", ind), ("metabolism.json", met),
                ("molecule_form.json", forms), ("activity.json", acts), ("drug_warning.json", warn),
                ("drug.json", drug),
            ):
                if fragment in url:
                    key = next(k for k in payload if k != "page_meta")
                    field, value = next((k, v) for k, v in params.items() if k not in ("limit", "format", "order_by", "pchembl_value__isnull"))
                    field = field.removesuffix("__in")
                    wanted = set(str(value).split(","))
                    return {key: [r for r in payload[key] if str(r.get(field)) in wanted], "page_meta": {"next": None}}
            return None

        rest = ChEMBLAnnotator(rate_limit_delay=0)
        with patch.object(chembl_mod, "_get_json", get_json):
            return {
                "development": rest.get_development_status("CHEMBL941"),
                "mechanisms": rest.get_mechanisms("CHEMBL941"),
                "indications": rest.get_indications("CHEMBL941"),
                # No recorded CHEMBL122 molecule record: parse the rows directly.
                "warnings": rest._warning_entries(warn["drug_warnings"]),
                "metabolism": rest.get_metabolism("CHEMBL25"),
                "forms": rest.get_molecule_forms("CHEMBL941"),
                "activities": rest.get_activities("CHEMBL941", limit=5),
                "approval": rest.get_approval_record("CHEMBL941"),
            }

    def test_every_getter_returns_the_rest_shape(self, offline):
        rest = self._rest_outputs()
        local = {
            "development": offline.get_development_status("CHEMBL941"),
            "mechanisms": offline.get_mechanisms("CHEMBL941"),
            "indications": offline.get_indications("CHEMBL941"),
            "warnings": offline.get_warnings("CHEMBL122"),
            "metabolism": offline.get_metabolism("CHEMBL25"),
            "forms": offline.get_molecule_forms("CHEMBL941"),
            "activities": offline.get_activities("CHEMBL941", limit=5),
            "approval": offline.get_approval_record("CHEMBL941"),
        }
        for name, value in rest.items():
            assert type(local[name]) is type(value), name
            if isinstance(value, list):
                assert value and local[name], name
                assert set(local[name][0]) == set(value[0]), name
            else:
                assert set(local[name]) == set(value), name


# =====================================================================
# Comparison with the real ChEMBL_37 release (cluster only)
# =====================================================================


@pytest.fixture(scope="module")
def real():
    return ChEMBLAnnotator(backend="sqlite", db_path=REAL_DB)


@pytest.fixture(scope="module")
def rest_records():
    return {m["molecule_chembl_id"]: m for m in fixture("molecule_chembl_id_in")["molecules"]}


@pytest.mark.skipif(not REAL_DB.is_file(), reason="ChEMBL_37 SQLite release not present")
class TestAgainstRealRelease:
    """Imatinib and aspirin from chembl_37.db versus the recorded REST payloads."""

    def test_release(self, real):
        assert real.release == "ChEMBL_37"

    def test_identity(self, real):
        res = real.resolve_many([IMATINIB_KEY, MESYLATE_KEY, ASPIRIN_KEY])
        assert res[IMATINIB_KEY].molecule_chembl_id == "CHEMBL941"
        assert (res[MESYLATE_KEY].molecule_chembl_id, res[MESYLATE_KEY].parent_chembl_id) == ("CHEMBL1642", "CHEMBL941")
        assert res[ASPIRIN_KEY].molecule_chembl_id == "CHEMBL25"

    @pytest.mark.parametrize("chembl_id", ["CHEMBL941", "CHEMBL25", "CHEMBL1642"])
    def test_development_status_equals_rest(self, real, rest_records, chembl_id):
        expected = ChEMBLAnnotator._development_from_record(rest_records[chembl_id])
        assert ChEMBLAnnotator._development_from_record(real.get_molecule_record(chembl_id)) == expected

    @pytest.mark.parametrize("chembl_id", ["CHEMBL941", "CHEMBL25", "CHEMBL1642"])
    def test_record_fields_equal_rest(self, real, rest_records, chembl_id):
        rest, local = rest_records[chembl_id], real.get_molecule_record(chembl_id)
        assert local["molecule_hierarchy"] == rest["molecule_hierarchy"]
        for key in ("canonical_smiles", "standard_inchi_key", "standard_inchi"):
            assert local["molecule_structures"][key] == rest["molecule_structures"][key]
        # Parents may collect salt ATC codes; REST's own codes must all be there.
        assert set(rest["atc_classifications"]) <= set(local["atc_classifications"])
        assert {(s["molecule_synonym"], s["syn_type"]) for s in local["molecule_synonyms"]} == {
            (s["molecule_synonym"], s["syn_type"]) for s in rest["molecule_synonyms"]
        }

    def test_synonym_order_equals_rest(self, real, rest_records):
        rest = ChEMBLAnnotator(rate_limit_delay=0)
        rest.prime_records(rest_records.values())
        for cid in ("CHEMBL941", "CHEMBL25"):
            assert real.get_synonyms(cid) == rest.get_synonyms(cid)

    def test_mechanisms_equal_rest(self, real):
        rest_rows = fixture("mechanism_parent_in")["mechanisms"]
        expected = {
            p: sorted(
                (m["mechanism"], m["action_type"], m["target_chembl_id"], m["max_phase"], m["direct_interaction"])
                for m in ChEMBLAnnotator._mechanism_entries([r for r in rest_rows if r["parent_molecule_chembl_id"] == p])
            )
            for p in ("CHEMBL941", "CHEMBL25")
        }
        local = real.get_mechanisms_many(["CHEMBL941", "CHEMBL25"])
        for p, rows in expected.items():
            assert sorted(
                (m["mechanism"], m["action_type"], m["target_chembl_id"], m["max_phase"], m["direct_interaction"])
                for m in local[p]
            ) == rows
        raw = {r["mec_id"]: r for r in real._sql.mechanisms(["CHEMBL941"])["CHEMBL941"]}
        for row in rest_rows:
            if row["parent_molecule_chembl_id"] == "CHEMBL941":
                assert raw[row["mec_id"]]["molecule_chembl_id"] == row["molecule_chembl_id"]
                assert len(raw[row["mec_id"]]["mechanism_refs"]) == len(row["mechanism_refs"])

    def test_indications_equal_rest(self, real):
        # Untrimmed REST totals on 2026-11-11: imatinib 52 + 82 rows, aspirin 167 + 1.
        rows = real._sql.indications(["CHEMBL941", "CHEMBL25"])
        assert len(rows["CHEMBL941"]) == 134
        assert len(rows["CHEMBL25"]) == 168
        by_id = {r["drugind_id"]: r for p in rows.values() for r in p}
        for rest in fixture("drug_indication_parent_in_trimmed")["drug_indications"]:
            local = by_id[rest["drugind_id"]]
            for key in ("efo_id", "mesh_id", "molecule_chembl_id", "parent_molecule_chembl_id"):
                assert local[key] == rest[key]
            assert float(local["max_phase_for_ind"]) == float(rest["max_phase_for_ind"])
            assert len(local["indication_refs"]) == len(rest["indication_refs"])

    def test_warnings_equal_rest(self, real):
        rest = fixture("drug_warning_parent_in")["drug_warnings"]
        rows = real._sql.warnings(["CHEMBL941", "CHEMBL25", "CHEMBL122"])
        assert rows["CHEMBL941"] == [] and rows["CHEMBL25"] == []
        assert sorted(r["warning_id"] for r in rows["CHEMBL122"]) == sorted(r["warning_id"] for r in rest)

    def test_metabolism_contains_rest(self, real):
        rest = {r["met_id"]: r for r in fixture("metabolism_drug_in")["metabolisms"]}
        local = {r["met_id"]: r for r in real._sql.metabolism(["CHEMBL25"])["CHEMBL25"]}
        assert set(rest) <= set(local)
        for met_id, row in rest.items():
            for key in ("enzyme_name", "metabolite_chembl_id", "substrate_chembl_id", "metabolite_name", "substrate_name"):
                assert local[met_id][key] == row[key], (met_id, key)

    def test_forms_equal_rest(self, real):
        rest = fixture("molecule_form_parent_in")["molecule_forms"]
        local = real._sql.molecule_forms(["CHEMBL941", "CHEMBL25"])
        assert {(f["molecule_chembl_id"], f["parent_chembl_id"], f["is_parent"]) for p in local.values() for f in p} == {
            (f["molecule_chembl_id"], f["parent_chembl_id"], f["is_parent"]) for f in rest
        }

    def test_activities_top5_equal_rest(self, real):
        rest = fixture("activity_imatinib_top5")["activities"]
        local = real.get_activities("CHEMBL941", limit=5)
        assert [a["pchembl_value"] for a in local] == [float(r["pchembl_value"]) for r in rest]
        assert local[0]["target_chembl_id"] == rest[0]["target_chembl_id"]
        assert local[0]["assay_chembl_id"] == rest[0]["assay_chembl_id"]
        assert local[0]["document_chembl_id"] == rest[0]["document_chembl_id"]
        assert local[0]["bao_label"] == rest[0]["bao_label"]
        assert local[0]["action_type"] == rest[0]["action_type"]

    def test_target_details_equal_rest(self, real):
        targets = ["CHEMBL1862", "CHEMBL1913", "CHEMBL1936", "CHEMBL2094253", "CHEMBL2096618"]
        comps = {c["component_id"]: c for c in fixture("target_component_in")["target_components"]}
        classes = fixture("protein_classification_in")["protein_classifications"]

        def get_json(url, params=None, timeout=30):
            if "target_component.json" in url:
                wanted = set(params["component_id__in"].split(","))
                return {"target_components": [c for k, c in comps.items() if str(k) in wanted]}
            if "protein_classification.json" in url:
                wanted = set(params["protein_class_id__in"].split(","))
                return {"protein_classifications": [c for c in classes if str(c["protein_class_id"]) in wanted]}
            if "target.json" in url:
                return fixture("target_in")
            return None

        rest = ChEMBLAnnotator(rate_limit_delay=0)
        with patch.object(chembl_mod, "_get_json", get_json):
            expected = rest.get_target_details(targets)
        assert real.get_target_details(targets) == expected

    def test_approval_equals_rest_where_derivable(self, real):
        rest = fixture("drug_imatinib")["drugs"][0]
        local = real.get_approval_record("CHEMBL941")
        assert local["first_approval"] == rest["first_approval"]
        assert local["atc_codes"] == [a["code"] for a in rest["atc_classification"]]
        assert local["applicants"]
