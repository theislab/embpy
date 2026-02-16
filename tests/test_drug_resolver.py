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


class TestRDKitCanonical:
    def test_rdkit_canonical_with_rdkit(self, resolver_with_rdkit):
        if not resolver_with_rdkit._rdkit_available:
            pytest.skip("RDKit not available")
        result = resolver_with_rdkit._rdkit_canonical("CCO")
        assert result == "CCO"

    def test_rdkit_canonical_invalid_smiles(self, resolver_with_rdkit):
        if not resolver_with_rdkit._rdkit_available:
            pytest.skip("RDKit not available")
        result = resolver_with_rdkit._rdkit_canonical("INVALID_SMILES_XXX")
        assert result is None

    def test_rdkit_canonical_without_rdkit(self, resolver):
        result = resolver._rdkit_canonical("CCO")
        assert result == "CCO"
