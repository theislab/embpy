"""Small-molecule annotation from multiple public databases.

Aggregates metadata for a molecule (given as SMILES, name, or InChI)
from six categories of data sources:

1. **Structural / physicochemical** -- RDKit (local)
2. **Bioactivities & targets** -- ChEMBL REST API
3. **Ontological roles** -- ChEBI REST API
4. **Pathway annotations** -- KEGG REST API
5. **Cross-database IDs** -- PubChem + UniChem REST APIs
6. **Disease associations** -- PubChem REST API

Does **not** reimplement pertpy functionality (drug-target gene
annotation, MoA lookup tables, cell-line metadata).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

import requests

logger = logging.getLogger(__name__)

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
CHEBI_API = "https://www.ebi.ac.uk/webservices/chebi/2.0/test"
KEGG = "https://rest.kegg.jp"
UNICHEM = "https://www.ebi.ac.uk/unichem/rest"


def _get_json(url: str, params: dict | None = None, timeout: int = 30) -> dict | None:
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:  # noqa: BLE001
        logger.debug("Request failed for %s: %s", url, e)
        return None


def _get_text(url: str, timeout: int = 30) -> str | None:
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.text
    except Exception as e:  # noqa: BLE001
        logger.debug("Request failed for %s: %s", url, e)
        return None


class MoleculeAnnotator:
    """Aggregate annotations for small molecules from public databases.

    Parameters
    ----------
    rate_limit_delay : float
        Seconds to wait between API calls (default 0.2).
    """

    def __init__(self, rate_limit_delay: float = 0.2) -> None:
        self.delay = rate_limit_delay

    def _sleep(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    # ==================================================================
    # Identifier resolution
    # ==================================================================

    def _resolve_to_smiles(self, identifier: str) -> str | None:
        """Best-effort resolution of any identifier to canonical SMILES."""
        s = identifier.strip()
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                return Chem.MolToSmiles(mol)
        except ImportError:
            pass

        data = _get_json(
            f"{PUBCHEM}/compound/name/{requests.utils.quote(s)}/property/CanonicalSMILES/JSON",
        )
        if data:
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("CanonicalSMILES")
        return None

    def _resolve_pubchem_cid(self, identifier: str) -> int | None:
        """Resolve identifier to PubChem CID."""
        s = identifier.strip()
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                canon = Chem.MolToSmiles(mol)
                data = _get_json(
                    f"{PUBCHEM}/compound/smiles/{requests.utils.quote(canon)}/cids/JSON",
                )
                if data:
                    cids = data.get("IdentifierList", {}).get("CID", [])
                    if cids:
                        return cids[0]
        except ImportError:
            pass

        data = _get_json(
            f"{PUBCHEM}/compound/name/{requests.utils.quote(s)}/cids/JSON",
        )
        if data:
            cids = data.get("IdentifierList", {}).get("CID", [])
            if cids:
                return cids[0]
        return None

    def _resolve_chembl_id(self, smiles: str) -> str | None:
        """Resolve SMILES to ChEMBL molecule ID."""
        data = _get_json(
            f"{CHEMBL}/molecule.json",
            params={
                "molecule_structures__canonical_smiles__flexmatch": smiles,
                "limit": 1,
                "format": "json",
            },
        )
        if data and data.get("molecules"):
            return data["molecules"][0].get("molecule_chembl_id")
        return None

    # ==================================================================
    # 1. Structural & Physicochemical (RDKit -- local)
    # ==================================================================

    def get_physicochemical_properties(self, smiles: str) -> dict[str, Any]:
        """Compute physicochemical properties using RDKit.

        Returns MW, LogP, PSA, HBD, HBA, rotatable bonds, Fsp3, QED,
        Lipinski violations, synthetic accessibility, and toxicity alerts.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, QED, rdMolDescriptors
        except ImportError:
            logger.warning("RDKit not available; skipping physicochemical properties")
            return {}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Invalid SMILES: {smiles}"}

        props: dict[str, Any] = {
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 3),
            "tpsa": round(Descriptors.TPSA(mol), 2),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "fsp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "formula": rdMolDescriptors.CalcMolFormula(mol),
        }

        # QED drug-likeness
        try:
            props["qed"] = round(QED.qed(mol), 3)
        except Exception:  # noqa: BLE001
            props["qed"] = None

        # Lipinski Rule of Five violations
        violations = 0
        if props["molecular_weight"] > 500:
            violations += 1
        if props["logp"] > 5:
            violations += 1
        if props["hbd"] > 5:
            violations += 1
        if props["hba"] > 10:
            violations += 1
        props["lipinski_violations"] = violations

        # Synthetic accessibility
        try:
            from rdkit.Chem import RDConfig
            import os, sys
            sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
            if sa_path not in sys.path:
                sys.path.insert(0, sa_path)
            from sascorer import calculateScore  # type: ignore[import-not-found]
            props["sa_score"] = round(calculateScore(mol), 2)
        except Exception:  # noqa: BLE001
            props["sa_score"] = None

        # PAINS filter
        try:
            from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
            fc_params = FilterCatalogParams()
            fc_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            catalog = FilterCatalog(fc_params)
            entry = catalog.GetFirstMatch(mol)
            props["pains_alert"] = entry is not None
            if entry:
                props["pains_description"] = entry.GetDescription()
        except Exception:  # noqa: BLE001
            props["pains_alert"] = None

        # Murcko scaffold
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            props["murcko_scaffold"] = Chem.MolToSmiles(scaffold)
        except Exception:  # noqa: BLE001
            props["murcko_scaffold"] = None

        return props

    # ==================================================================
    # 2. Bioactivities & Targets (ChEMBL)
    # ==================================================================

    def get_bioactivities(
        self,
        smiles: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch bioactivity data (IC50, Ki, EC50, etc.) from ChEMBL."""
        chembl_id = self._resolve_chembl_id(smiles)
        if not chembl_id:
            logger.debug("No ChEMBL ID found for %s", smiles[:60])
            return []

        self._sleep()
        data = _get_json(
            f"{CHEMBL}/activity.json",
            params={
                "molecule_chembl_id": chembl_id,
                "limit": limit,
                "format": "json",
            },
        )
        if not data:
            return []

        activities = []
        for act in data.get("activities", []):
            activities.append({
                "target_chembl_id": act.get("target_chembl_id"),
                "target_pref_name": act.get("target_pref_name"),
                "target_organism": act.get("target_organism"),
                "activity_type": act.get("standard_type"),
                "activity_value": act.get("standard_value"),
                "activity_units": act.get("standard_units"),
                "activity_relation": act.get("standard_relation"),
                "assay_type": act.get("assay_type"),
                "pchembl_value": act.get("pchembl_value"),
            })
        return activities

    def get_target_proteins(self, smiles: str) -> list[dict[str, str]]:
        """Get unique target proteins for a molecule from ChEMBL."""
        activities = self.get_bioactivities(smiles, limit=100)
        seen = set()
        targets = []
        for act in activities:
            tid = act.get("target_chembl_id")
            if tid and tid not in seen:
                seen.add(tid)
                targets.append({
                    "target_chembl_id": tid,
                    "target_name": act.get("target_pref_name", ""),
                    "organism": act.get("target_organism", ""),
                })
        return targets

    def get_mechanism_of_action(self, smiles: str) -> list[dict[str, str]]:
        """Get mechanism of action annotations from ChEMBL."""
        chembl_id = self._resolve_chembl_id(smiles)
        if not chembl_id:
            return []

        self._sleep()
        data = _get_json(
            f"{CHEMBL}/mechanism.json",
            params={
                "molecule_chembl_id": chembl_id,
                "format": "json",
            },
        )
        if not data:
            return []

        return [
            {
                "mechanism": m.get("mechanism_of_action", ""),
                "action_type": m.get("action_type", ""),
                "target_name": m.get("target_name", ""),
                "target_chembl_id": m.get("target_chembl_id", ""),
            }
            for m in data.get("mechanisms", [])
        ]

    # ==================================================================
    # 3. Ontological Roles (ChEBI)
    # ==================================================================

    def get_chebi_roles(self, identifier: str) -> list[dict[str, str]]:
        """Get ChEBI ontological roles for a molecule.

        Uses PubChem to map to ChEBI ID, then queries ChEBI for roles.
        """
        cid = self._resolve_pubchem_cid(identifier)
        if cid is None:
            return []

        self._sleep()
        data = _get_json(
            f"{PUBCHEM}/compound/cid/{cid}/xrefs/RegistryID/JSON",
        )
        if not data:
            return []

        chebi_ids = []
        xrefs = data.get("InformationList", {}).get("Information", [])
        for info in xrefs:
            for rid in info.get("RegistryID", []):
                if str(rid).startswith("CHEBI:"):
                    chebi_ids.append(str(rid))

        if not chebi_ids:
            return []

        roles = []
        for chebi_id in chebi_ids[:3]:
            self._sleep()
            chebi_num = chebi_id.replace("CHEBI:", "")
            result = _get_json(
                f"https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms",
                params={"short_form": f"CHEBI_{chebi_num}"},
            )
            if result and result.get("_embedded", {}).get("terms"):
                term = result["_embedded"]["terms"][0]
                roles.append({
                    "chebi_id": chebi_id,
                    "name": term.get("label", ""),
                    "description": term.get("description", [""])[0] if term.get("description") else "",
                })
        return roles

    # ==================================================================
    # 4. Pathway Annotations (KEGG)
    # ==================================================================

    def get_kegg_pathways(self, identifier: str) -> list[dict[str, str]]:
        """Get KEGG pathway annotations for a molecule."""
        cid = self._resolve_pubchem_cid(identifier)
        if cid is None:
            return []

        self._sleep()
        text = _get_text(f"{KEGG}/conv/compound/pubchem:{cid}")
        if not text or not text.strip():
            return []

        kegg_id = None
        for line in text.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                kegg_id = parts[1].strip()
                break

        if not kegg_id:
            return []

        self._sleep()
        text = _get_text(f"{KEGG}/link/pathway/{kegg_id}")
        if not text or not text.strip():
            return []

        pathways = []
        for line in text.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                pw_id = parts[1].strip()
                self._sleep()
                pw_text = _get_text(f"{KEGG}/get/{pw_id}")
                pw_name = pw_id
                if pw_text:
                    for pw_line in pw_text.split("\n"):
                        if pw_line.startswith("NAME"):
                            pw_name = pw_line.replace("NAME", "").strip()
                            break
                pathways.append({"pathway_id": pw_id, "pathway_name": pw_name})
        return pathways

    # ==================================================================
    # 5. Cross-Database IDs (PubChem + UniChem)
    # ==================================================================

    def get_cross_references(self, identifier: str) -> dict[str, Any]:
        """Map a molecule to IDs in multiple databases."""
        cid = self._resolve_pubchem_cid(identifier)
        refs: dict[str, Any] = {"pubchem_cid": cid}

        if cid is None:
            return refs

        smiles = self._resolve_to_smiles(identifier)
        if smiles:
            chembl_id = self._resolve_chembl_id(smiles)
            refs["chembl_id"] = chembl_id

        self._sleep()
        data = _get_json(
            f"{PUBCHEM}/compound/cid/{cid}/xrefs/RegistryID/JSON",
        )
        if data:
            xrefs = data.get("InformationList", {}).get("Information", [])
            for info in xrefs:
                for rid in info.get("RegistryID", []):
                    rid_str = str(rid)
                    if rid_str.startswith("CHEBI:"):
                        refs.setdefault("chebi_id", rid_str)
                    elif rid_str.startswith("DB") and len(rid_str) <= 10:
                        refs.setdefault("drugbank_id", rid_str)
                    elif rid_str.startswith("C") and rid_str[1:].isdigit():
                        refs.setdefault("kegg_compound_id", rid_str)
                    elif rid_str.startswith("HMDB"):
                        refs.setdefault("hmdb_id", rid_str)

        self._sleep()
        data = _get_json(
            f"{PUBCHEM}/compound/cid/{cid}/property/InChIKey/JSON",
        )
        if data:
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                refs["inchikey"] = props[0].get("InChIKey")

        return refs

    # ==================================================================
    # 6. Disease Associations (PubChem)
    # ==================================================================

    def get_disease_associations(self, identifier: str) -> list[dict[str, str]]:
        """Get disease associations from PubChem bioassays."""
        cid = self._resolve_pubchem_cid(identifier)
        if cid is None:
            return []

        self._sleep()
        data = _get_json(
            f"{PUBCHEM}/compound/cid/{cid}/assaysummary/JSON",
        )
        if not data:
            return []

        diseases: dict[str, str] = {}
        for table in data.get("Table", {}).get("Row", [])[:200]:
            cells = table.get("Cell", [])
            if len(cells) > 5:
                assay_name = str(cells[1]) if len(cells) > 1 else ""
                target_name = str(cells[4]) if len(cells) > 4 else ""
                for keyword in ["cancer", "diabetes", "alzheimer", "parkinson",
                                "hiv", "malaria", "inflammation", "asthma",
                                "obesity", "hypertension", "leukemia",
                                "lymphoma", "melanoma", "hepatitis"]:
                    if keyword in assay_name.lower() or keyword in target_name.lower():
                        diseases[keyword] = assay_name
        return [{"disease": k, "assay": v} for k, v in diseases.items()]

    # ==================================================================
    # Convenience: one-call aggregation
    # ==================================================================

    def annotate(
        self,
        identifier: str,
        sources: Literal["all", "structural", "bioactivity", "ontology",
                         "pathways", "xrefs", "diseases"] | list[str] = "all",
    ) -> dict[str, Any]:
        """Aggregate all annotations for a molecule in one call.

        Parameters
        ----------
        identifier
            SMILES, compound name, InChI, or PubChem CID.
        sources
            Which annotation sources to query. ``"all"`` queries
            everything. Pass a list to select specific ones:
            ``["structural", "bioactivity", "ontology", "pathways",
            "xrefs", "diseases"]``.

        Returns
        -------
        Nested dict with keys per annotation category.
        """
        if sources == "all":
            sources_list = [
                "structural", "bioactivity", "ontology",
                "pathways", "xrefs", "diseases",
            ]
        elif isinstance(sources, str):
            sources_list = [sources]
        else:
            sources_list = list(sources)

        smiles = self._resolve_to_smiles(identifier)
        result: dict[str, Any] = {
            "identifier": identifier,
            "canonical_smiles": smiles,
        }

        if smiles is None:
            logger.warning("Could not resolve '%s' to SMILES", identifier)
            result["error"] = "Could not resolve identifier to SMILES"
            return result

        if "structural" in sources_list:
            result["physicochemical"] = self.get_physicochemical_properties(smiles)

        if "bioactivity" in sources_list:
            result["bioactivities"] = self.get_bioactivities(smiles)
            result["targets"] = self.get_target_proteins(smiles)
            result["mechanism_of_action"] = self.get_mechanism_of_action(smiles)

        if "ontology" in sources_list:
            result["chebi_roles"] = self.get_chebi_roles(identifier)

        if "pathways" in sources_list:
            result["kegg_pathways"] = self.get_kegg_pathways(identifier)

        if "xrefs" in sources_list:
            result["cross_references"] = self.get_cross_references(identifier)

        if "diseases" in sources_list:
            result["disease_associations"] = self.get_disease_associations(identifier)

        return result

    def annotate_batch(
        self,
        identifiers: list[str],
        sources: str | list[str] = "all",
    ) -> dict[str, dict[str, Any]]:
        """Annotate a list of molecules.

        Returns
        -------
        Dict mapping identifier to annotation dict.
        """
        results: dict[str, dict[str, Any]] = {}
        total = len(identifiers)
        for i, ident in enumerate(identifiers):
            results[ident] = self.annotate(ident, sources=sources)
            if (i + 1) % 10 == 0:
                logger.info("Annotated %d/%d molecules", i + 1, total)
        logger.info("Annotated %d molecules total", total)
        return results

    def annotate_adata(
        self,
        adata,  # anndata.AnnData
        column: str,
        sources: str | list[str] = "all",
        copy: bool = True,
    ):
        """Annotate perturbations in an AnnData with molecule metadata.

        Reads molecule identifiers from ``adata.obs[column]``, fetches
        annotations for each unique identifier, and stores:

        - Scalar properties (MW, LogP, QED, etc.) as new columns in
          ``adata.obs``
        - Full annotation dicts in ``adata.uns["molecule_annotations"]``

        Parameters
        ----------
        adata
            AnnData with a molecule identifier column in ``.obs``.
        column
            Column in ``adata.obs`` containing identifiers.
        sources
            Annotation sources to query (see :meth:`annotate`).
        copy
            If ``True``, operate on a copy.

        Returns
        -------
        AnnData with molecule annotations added.
        """
        import numpy as np

        if copy:
            adata = adata.copy()

        if column not in adata.obs.columns:
            raise ValueError(
                f"Column '{column}' not found in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )

        identifiers = adata.obs[column].astype(str).values
        unique_ids = list(dict.fromkeys(identifiers))
        logger.info(
            "Annotating %d unique molecules from %d cells",
            len(unique_ids), len(identifiers),
        )

        annotations = self.annotate_batch(unique_ids, sources=sources)
        adata.uns["molecule_annotations"] = annotations

        scalar_keys = [
            "molecular_weight", "logp", "tpsa", "hbd", "hba",
            "rotatable_bonds", "qed", "lipinski_violations",
            "fsp3", "heavy_atoms",
        ]

        for key in scalar_keys:
            values = []
            for ident in identifiers:
                ann = annotations.get(ident, {})
                phys = ann.get("physicochemical", {})
                values.append(phys.get(key))
            adata.obs[f"mol_{key}"] = values

        n_targets = []
        for ident in identifiers:
            ann = annotations.get(ident, {})
            targets = ann.get("targets", [])
            n_targets.append(len(targets))
        adata.obs["mol_n_targets"] = n_targets

        smiles_col = []
        for ident in identifiers:
            ann = annotations.get(ident, {})
            smiles_col.append(ann.get("canonical_smiles", ""))
        adata.obs["mol_canonical_smiles"] = smiles_col

        logger.info("Molecule annotations stored in adata.obs (mol_*) and adata.uns")
        return adata
