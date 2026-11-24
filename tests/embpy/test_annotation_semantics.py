"""Regression cases uncovered by the LINCS annotation verification."""
import pytest

from embpy.resources.molecule.chembl import ChEMBLAnnotator


@pytest.mark.parametrize("phase", [None, -1, 0.5, 1, 2, 3, 4])
def test_unrecorded_drug_properties_are_not_negative_evidence(phase):
    flags = {name: 0 for name in ("oral", "parenteral", "topical", "black_box_warning",
                                  "withdrawn_flag", "chemical_probe", "natural_product")}
    result = ChEMBLAnnotator._development_from_record({"max_phase": phase, **flags})
    assert all(result[name] is None for name in flags)
    assert all(result["raw_flags"][name] == 0 for name in flags)
    if phase is None or phase < 0:
        assert result["is_approved"] is None
    if phase is None:
        assert result["development_phase"] is None


def test_anndata_summary_preserves_unknown_flags():
    from embpy.resources.molecule.chembl import chembl_summary_columns
    unknown = ChEMBLAnnotator._development_from_record({"max_phase": -1, "black_box_warning": 0})
    positive = ChEMBLAnnotator._development_from_record({"max_phase": 4, "black_box_warning": 1, "withdrawn_flag": 1})
    columns = chembl_summary_columns([{}, {"development": unknown}, {"development": positive}])
    for name in ("drug_is_approved", "drug_is_withdrawn", "drug_has_black_box_warning"):
        assert columns[name] == [None, None, True]


def test_research_chemical_probe_is_retained_without_clinical_phase():
    result = ChEMBLAnnotator._development_from_record({"chemical_probe": 1})
    assert result["chemical_probe"] is True
    assert result["max_phase"] is None


def test_unknown_target_type_cannot_win_protein_selectivity():
    profile = [{"target_type": None, "target_name": "unknown", "best_pchembl": 12},
               {"target_type": "SINGLE PROTEIN", "target_name": "confirmed", "best_pchembl": 7}]
    result = ChEMBLAnnotator.summarize_selectivity(profile)
    assert result["primary_target"] == "confirmed"
    assert result["selectivity_window"] is None


def test_cellosaurus_site_does_not_claim_tissue_of_origin():
    import json
    from pathlib import Path
    from embpy.resources.cellline.annotator import cellosaurus_entries, parse_cellosaurus_entry
    path = Path(__file__).parent / "fixtures/cellline/cellosaurus_detail_CVCL_0035.json"
    record = parse_cellosaurus_entry(cellosaurus_entries(json.loads(path.read_text()))[0])
    assert record["sampling_site"] == record["site"]
    assert record["sampling_site_type"] == record["site_type"]
    assert record["tissue"] == ""


@pytest.mark.parametrize("child,safe", [("CC[NH3+].[Cl-]", True), ("[13CH3]CN", False)])
def test_parent_family_distinguishes_salts_from_isotope_variants(monkeypatch, child, safe):
    api = ChEMBLAnnotator(rate_limit_delay=0)
    monkeypatch.setattr(api, "_parent_id", lambda _: "CHEMBL1")
    monkeypatch.setattr(api, "get_molecule_forms", lambda *a, **k: [{"molecule_chembl_id": "CHEMBL2"}])
    records = {cid: {"molecule_structures": {"canonical_smiles": smiles}}
               for cid, smiles in [("CHEMBL1", "CCN"), ("CHEMBL2", child)]}
    monkeypatch.setattr(api, "get_molecule_records_many", lambda _: records)
    result = api.get_parent_family_audit("CHEMBL1")
    assert result["primary_annotation_eligible"] is safe


def test_pubchem_standardization_is_not_an_exact_match(monkeypatch):
    from embpy.resources.molecule import annotator as module
    from rdkit import Chem
    smiles = "N[C@H](C)C(=O)O"
    other = Chem.MolToInchiKey(Chem.MolFromSmiles("N[C@@H](C)C(=O)O"))
    def response(url, **kwargs):
        if "/smiles/cids/" in url:
            return {"IdentifierList": {"CID": [123]}}
        return {"PropertyTable": {"Properties": [{"CID": 123, "InChIKey": other}]}}
    monkeypatch.setattr(module, "_get_json", response)
    result = module.MoleculeAnnotator(rate_limit_delay=0).resolve_pubchem_cid(smiles)
    assert result["cid"] == 123
    assert result["match_tier"] == "connectivity_only"
    assert result["primary_annotation_eligible"] is False
