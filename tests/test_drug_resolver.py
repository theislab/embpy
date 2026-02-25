"""Tests for DrugResolver with mocked HTTP calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from embpy.resources.drug_resolver import DrugResolver


@pytest.fixture
def resolver():
    """DrugResolver with RDKit disabled to avoid dependency issues in tests."""
    return DrugResolver(use_rdkit=False, sleep_sec=0.0)


@pytest.fixture
def resolver_with_rdkit():
    """DrugResolver with RDKit enabled."""
    return DrugResolver(use_rdkit=True, sleep_sec=0.0)


class TestNameToSmiles:
    def test_direct_pubchem_hit(self, resolver):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "PropertyTable": {
                "Properties": [{"CanonicalSMILES": "CCO"}]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch("embpy.resources.drug_resolver.requests.get", return_value=mock_response):
            result = resolver.name_to_smiles("ethanol")
            assert result == "CCO"

    def test_fallback_to_cid(self, resolver):
        call_count = [0]

        def side_effect(url, **kwargs):
            call_count[0] += 1
            mock = MagicMock()
            mock.raise_for_status = MagicMock()

            if call_count[0] == 1:
                mock.raise_for_status.side_effect = Exception("Not found")
                return mock
            elif call_count[0] == 2:
                mock.json.return_value = {
                    "IdentifierList": {"CID": [702]}
                }
                return mock
            else:
                mock.json.return_value = {
                    "PropertyTable": {
                        "Properties": [{"CanonicalSMILES": "CCO"}]
                    }
                }
                return mock

        with patch("embpy.resources.drug_resolver.requests.get", side_effect=side_effect):
            result = resolver.name_to_smiles("ethanol")
            assert result == "CCO"

    def test_returns_none_on_failure(self, resolver):
        mock = MagicMock()
        mock.raise_for_status.side_effect = Exception("API error")

        with patch("embpy.resources.drug_resolver.requests.get", return_value=mock):
            result = resolver.name_to_smiles("completely_fake_drug_xyz")
            assert result is None


class TestSmilesToNames:
    def test_smiles_to_names_success(self, resolver):
        call_count = [0]

        def side_effect(url, **kwargs):
            call_count[0] += 1
            mock = MagicMock()
            mock.raise_for_status = MagicMock()

            if "cids" in url:
                mock.json.return_value = {
                    "IdentifierList": {"CID": [702]}
                }
            elif "property/Title" in url:
                mock.json.return_value = {
                    "PropertyTable": {
                        "Properties": [{"Title": "Ethanol"}]
                    }
                }
            elif "synonyms" in url:
                mock.json.return_value = {
                    "InformationList": {
                        "Information": [
                            {"Synonym": ["Ethanol", "Ethyl alcohol", "EtOH"]}
                        ]
                    }
                }
            return mock

        with patch("embpy.resources.drug_resolver.requests.get", side_effect=side_effect):
            names = resolver.smiles_to_names("CCO", top_k=3)
            assert len(names) <= 3
            assert "Ethanol" in names

    def test_smiles_to_names_no_cid(self, resolver):
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"IdentifierList": {"CID": []}}

        with patch("embpy.resources.drug_resolver.requests.get", return_value=mock):
            names = resolver.smiles_to_names("INVALID_SMILES")
            assert names == []


class TestSmilesToPrimaryName:
    def test_returns_first_name(self, resolver):
        resolver.smiles_to_names = MagicMock(return_value=["Ethanol"])
        result = resolver.smiles_to_primary_name("CCO")
        assert result == "Ethanol"

    def test_returns_none_when_empty(self, resolver):
        resolver.smiles_to_names = MagicMock(return_value=[])
        result = resolver.smiles_to_primary_name("FAKE")
        assert result is None


class TestCidToNames:
    def test_cid_to_names(self, resolver):
        call_count = [0]

        def side_effect(url, **kwargs):
            call_count[0] += 1
            mock = MagicMock()
            mock.raise_for_status = MagicMock()

            if "property/Title" in url:
                mock.json.return_value = {
                    "PropertyTable": {
                        "Properties": [{"Title": "Aspirin"}]
                    }
                }
            else:
                mock.json.return_value = {
                    "InformationList": {
                        "Information": [
                            {"Synonym": ["Aspirin", "Acetylsalicylic acid"]}
                        ]
                    }
                }
            return mock

        with patch("embpy.resources.drug_resolver.requests.get", side_effect=side_effect):
            names = resolver.cid_to_names(2244)
            assert "Aspirin" in names


class TestCleanAndCanonicaliseSMILES:
    """Tests for DrugResolver._clean_and_canonicalise_smiles.

    These tests verify that structurally equivalent but differently-written SMILES
    are standardised to the same canonical form. Tests that require RDKit are
    skipped automatically when it is not installed.
    """

    def test_valid_smiles_returns_canonical(self, resolver_with_rdkit):
        if not resolver_with_rdkit._rdkit_available:
            pytest.skip("RDKit not available")
        result = resolver_with_rdkit._clean_and_canonicalise_smiles("CCO")
        assert result == "CCO"

    def test_invalid_smiles_returns_none(self, resolver_with_rdkit):
        if not resolver_with_rdkit._rdkit_available:
            pytest.skip("RDKit not available")
        result = resolver_with_rdkit._clean_and_canonicalise_smiles("INVALID_SMILES_XXX")
        assert result is None

    def test_returns_smiles_without_rdkit(self, resolver):
        result = resolver._clean_and_canonicalise_smiles("CCO")
        assert result == "CCO"

    def test_kekulized_and_aromatic_benzene_are_identical(self, resolver_with_rdkit):
        """Kekulé and aromatic notations for benzene must yield the same output."""
        if not resolver_with_rdkit._rdkit_available:
            pytest.skip("RDKit not available")
        r = resolver_with_rdkit._clean_and_canonicalise_smiles
        assert r("C1=CC=CC=C1") == r("c1ccccc1")

    def test_aspirin_different_atom_orderings_are_identical(self, resolver_with_rdkit):
        """Aspirin written with different atom traversal orders must canonicalise identically."""
        if not resolver_with_rdkit._rdkit_available:
            pytest.skip("RDKit not available")
        r = resolver_with_rdkit._clean_and_canonicalise_smiles
        assert r("CC(=O)Oc1ccccc1C(=O)O") == r("OC(=O)c1ccccc1OC(C)=O")

    def test_ethanol_different_orderings_are_identical(self, resolver_with_rdkit):
        """Simple reordering of atoms in ethanol must produce the same canonical SMILES."""
        if not resolver_with_rdkit._rdkit_available:
            pytest.skip("RDKit not available")
        r = resolver_with_rdkit._clean_and_canonicalise_smiles
        assert r("CCO") == r("OCC")

    def test_salt_stripping_removes_counterions(self, resolver_with_rdkit):
        """Sodium acetate fragments should reduce to acetic acid after salt stripping."""
        if not resolver_with_rdkit._rdkit_available:
            pytest.skip("RDKit not available")
        r = resolver_with_rdkit._clean_and_canonicalise_smiles
        assert r("CC(=O)[O-].[Na+]") == r("CC(=O)O")

    def test_charge_neutralization(self, resolver_with_rdkit):
        """Acetate anion should neutralise to acetic acid."""
        if not resolver_with_rdkit._rdkit_available:
            pytest.skip("RDKit not available")
        r = resolver_with_rdkit._clean_and_canonicalise_smiles
        assert r("CC(=O)[O-]") == r("CC(=O)O")

    def test_isotope_labels_are_stripped(self, resolver_with_rdkit):
        """13C-labelled and unlabelled ethanol must clean to the same molecule."""
        if not resolver_with_rdkit._rdkit_available:
            pytest.skip("RDKit not available")
        r = resolver_with_rdkit._clean_and_canonicalise_smiles
        assert r("[13CH3]CO") == r("CCO")

    def test_stereoisomers_are_kept_distinct(self, resolver_with_rdkit):
        """Cleaning must not destroy stereochemistry: L- and D-alanine must remain different."""
        if not resolver_with_rdkit._rdkit_available:
            pytest.skip("RDKit not available")
        r = resolver_with_rdkit._clean_and_canonicalise_smiles
        assert r("N[C@@H](C)C(=O)O") != r("N[C@H](C)C(=O)O")

    def test_largest_fragment_kept_in_mixture(self, resolver_with_rdkit):
        """In a multi-component SMILES the largest molecule must be returned."""
        if not resolver_with_rdkit._rdkit_available:
            pytest.skip("RDKit not available")
        r = resolver_with_rdkit._clean_and_canonicalise_smiles
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        assert r(f"{aspirin}.COCCO") == r(aspirin)

    def test_empty_string_returns_none(self, resolver_with_rdkit):
        """An empty string must return None."""
        if not resolver_with_rdkit._rdkit_available:
            pytest.skip("RDKit not available")
        assert resolver_with_rdkit._clean_and_canonicalise_smiles("") is None
