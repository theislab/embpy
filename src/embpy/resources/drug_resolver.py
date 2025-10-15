# drug_resolver.py
from __future__ import annotations

import logging
import time

import requests

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
HEADERS = {"Accept": "application/json", "User-Agent": "DrugResolver/0.1 (contact: you@example.com)"}


def _get_json(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    try:
        r.raise_for_status()
        return r.json()
    except Exception:
        logging.error("HTTP error for %s\nStatus: %s\nBody: %s", url, r.status_code, r.text[:300])
        raise


def _get_text(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=30)
    try:
        r.raise_for_status()
        return r.text
    except Exception:
        logging.error("HTTP error for %s\nStatus: %s\nBody: %s", url, r.status_code, r.text[:300])
        raise


class DrugResolver:
    """
    Resolve small molecules between names and SMILES using PubChem.

      - name -> canonical SMILES
      - SMILES -> common name(s) (via CID + synonyms)
    Optionally standardizes SMILES if RDKit is available.
    """

    def __init__(self, use_rdkit: bool = True, sleep_sec: float = 0.0):
        self.sleep_sec = sleep_sec
        try:
            if use_rdkit:
                from rdkit import Chem  # noqa: F401

                self._rdkit_available = True
            else:
                self._rdkit_available = False
        except Exception:
            self._rdkit_available = False
            logging.info("RDKit not available; proceeding without SMILES canonicalization.")

    # ---------- Helpers ----------
    def _sleep(self):
        if self.sleep_sec:
            time.sleep(self.sleep_sec)

    def _rdkit_canonical(self, smiles: str) -> str | None:
        if not self._rdkit_available:
            return smiles
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    # ---------- Public API ----------
    def name_to_smiles(self, name: str) -> str | None:
        """
        Return PubChem *canonical* SMILES for a common/brand/IUPAC name.

        Tries: (1) direct property, (2) CID->property, (3) NIH Cactus fallback.
        """
        q = requests.utils.quote(name)

        # 1) Direct: name -> CanonicalSMILES
        try:
            url = f"{PUBCHEM}/compound/name/{q}/property/CanonicalSMILES/JSON"
            js = _get_json(url)
            props = js.get("PropertyTable", {}).get("Properties", [])
            if props and "CanonicalSMILES" in props[0]:
                smi = props[0]["CanonicalSMILES"]
                return self._rdkit_canonical(smi) or smi
        except Exception:
            pass

        # 2) Fallback: name -> CID -> CanonicalSMILES
        try:
            url = f"{PUBCHEM}/compound/name/{q}/cids/JSON"
            js = _get_json(url)
            cids = js.get("IdentifierList", {}).get("CID", [])
            if cids:
                cid = cids[0]
                self._sleep()
                url = f"{PUBCHEM}/compound/cid/{cid}/property/CanonicalSMILES/JSON"
                js = _get_json(url)
                props = js.get("PropertyTable", {}).get("Properties", [])
                if props and "CanonicalSMILES" in props[0]:
                    smi = props[0]["CanonicalSMILES"]
                    return self._rdkit_canonical(smi) or smi
            else:
                logging.warning("No PubChem CID for name=%r", name)
        except Exception:
            pass

        # 3) Last resort: NIH Cactus
        try:
            url = f"https://cactus.nci.nih.gov/chemical/structure/{q}/smiles"
            smi = _get_text(url).strip()
            if smi and "Error" not in smi:
                return self._rdkit_canonical(smi) or smi
        except Exception:
            pass

        logging.error("Could not resolve SMILES for %r", name)
        return None

    def smiles_to_names(self, smiles: str, top_k: int = 5) -> list[str]:
        """
        Return up to top_k names (Preferred/IUPAC/common) for a SMILES.

        Order: Preferred title (if available), then synonyms.
        """
        smi = self._rdkit_canonical(smiles) or smiles

        # 1) SMILES -> CID
        try:
            url = f"{PUBCHEM}/compound/smiles/{requests.utils.quote(smi)}/cids/JSON"
            js = _get_json(url)
            cids = js.get("IdentifierList", {}).get("CID", [])
            if not cids:
                logging.warning("No PubChem CID for SMILES=%r", smi)
                return []
            cid = cids[0]
            self._sleep()
        except Exception:
            return []

        # 2) CID -> Title
        names: list[str] = []
        try:
            url = f"{PUBCHEM}/compound/cid/{cid}/property/Title/JSON"
            js = _get_json(url)
            props = js.get("PropertyTable", {}).get("Properties", [])
            title = props[0]["Title"] if props and props[0].get("Title") else None
            if title:
                names.append(title)
        except Exception:
            pass

        # 3) CID -> Synonyms
        try:
            url = f"{PUBCHEM}/compound/cid/{cid}/synonyms/JSON"
            js = _get_json(url)
            inf = js.get("InformationList", {}).get("Information", [])
            syns = inf[0].get("Synonym", []) if inf else []
            for s in syns:
                if s not in names:
                    names.append(s)
        except Exception:
            pass

        return names[:top_k]

    def cid_to_names(self, cid: int | str) -> list[str]:
        """Get preferred name + synonyms for a PubChem CID."""
        names: list[str] = []
        # Title
        try:
            url = f"{PUBCHEM}/compound/cid/{cid}/property/Title/JSON"
            js = _get_json(url)
            props = js.get("PropertyTable", {}).get("Properties", [])
            title = props[0]["Title"] if props and props[0].get("Title") else None
            if title:
                names.append(title)
        except Exception:
            pass
        # Synonyms
        try:
            url = f"{PUBCHEM}/compound/cid/{cid}/synonyms/JSON"
            js = _get_json(url)
            inf = js.get("InformationList", {}).get("Information", [])
            syns = inf[0].get("Synonym", []) if inf else []
            for s in syns:
                if s not in names:
                    names.append(s)
        except Exception:
            pass
        return names

    def smiles_to_primary_name(self, smiles: str) -> str | None:
        """Return a single best name for a SMILES (preferred title if present)."""
        names = self.smiles_to_names(smiles, top_k=1)
        return names[0] if names else None
