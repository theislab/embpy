"""Tests for embpy.resources.molecule_annotator -- MoleculeAnnotator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embpy.resources.molecule_annotator import MoleculeAnnotator


@pytest.fixture
def annotator():
    return MoleculeAnnotator(rate_limit_delay=0)


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
    @patch("embpy.resources.molecule_annotator._get_json")
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

    @patch("embpy.resources.molecule_annotator._get_json")
    def test_no_chembl_id(self, mock_get, annotator):
        mock_get.return_value = None
        activities = annotator.get_bioactivities("INVALID")
        assert activities == []


class TestTargetProteins:
    @patch("embpy.resources.molecule_annotator._get_json")
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
# Cross-references (mocked PubChem)
# =====================================================================


class TestCrossReferences:
    @patch("embpy.resources.molecule_annotator._get_json")
    def test_get_cross_references(self, mock_get, annotator):
        mock_get.side_effect = [
            {"IdentifierList": {"CID": [702]}},
            {"molecules": [{"molecule_chembl_id": "CHEMBL545"}]},
            {"InformationList": {"Information": [
                {"RegistryID": ["CHEBI:16236", "DB00898", "C00469"]},
            ]}},
            {"PropertyTable": {"Properties": [{"InChIKey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"}]}},
        ]
        refs = annotator.get_cross_references("ethanol")
        assert refs["pubchem_cid"] == 702
        assert "chebi_id" in refs


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
