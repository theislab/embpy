# drug_resolver.py
from __future__ import annotations

import logging
import re
import time
from typing import Literal
from urllib.parse import quote as _url_quote

import requests

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
HEADERS = {"Accept": "application/json", "User-Agent": "DrugResolver/0.1 (contact: you@example.com)"}

_TARGET_RE = re.compile(
    r"^[A-Z][A-Z0-9]*\.(inhibitor|activator|agonist|antagonist|modulator|blocker)$",
    re.IGNORECASE,
)
_CONTROL_TERMS = frozenset(
    {
        "control",
        "vehicle",
        "dmso",
        "dmso_tf",
        "untreated",
        "mock",
        "nan",
        "",
    }
)

_STEREO_PREFIX_RE = re.compile(r"^\([+\-RS±]+\)-")
_SALT_SUFFIX_RE = re.compile(
    r"\s+(?:di|mono|tri|hemi|bis?)?"
    r"(?:hydrochloride|mesylate|tosylate|sodium|sulfate|citrate|maleate|"
    r"phosphate|acetate|fumarate|hemifumarate|tartrate|succinate|besylate|"
    r"besilate|nitrate|bromide|chloride|iodide|potassium|calcium)$",
    re.IGNORECASE,
)
_GREEK_MAP = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta"}


_MAX_RETRIES = 3
_RETRY_CODES = frozenset({429, 500, 502, 503, 504})


def _request_with_backoff(
    url: str,
    *,
    params: dict | None = None,
    accept_json: bool = True,
) -> requests.Response:
    """HTTP GET with automatic retry + exponential back-off on server errors."""
    hdrs = HEADERS if accept_json else {"User-Agent": HEADERS["User-Agent"]}
    for attempt in range(_MAX_RETRIES):
        r = requests.get(url, params=params, headers=hdrs, timeout=30)
        if r.status_code not in _RETRY_CODES:
            return r
        wait = 2**attempt  # 1 s, 2 s, 4 s
        logging.warning("Rate-limited (%s) on %s — retrying in %ss", r.status_code, url, wait)
        time.sleep(wait)
    return r  # return last response even if still failing


def _get_json(url: str, params: dict | None = None) -> dict:
    r = _request_with_backoff(url, params=params, accept_json=True)
    try:
        r.raise_for_status()
        return r.json()
    except (requests.HTTPError, ValueError):
        logging.error("HTTP error for %s\nStatus: %s\nBody: %s", url, r.status_code, r.text[:300])
        raise


def _get_text(url: str) -> str:
    r = _request_with_backoff(url, accept_json=False)
    try:
        r.raise_for_status()
        return r.text
    except requests.HTTPError:
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
        except ImportError:
            self._rdkit_available = False
            logging.info("RDKit not available; proceeding without SMILES canonicalization.")

        try:
            import cirpy  # noqa: F401  # type: ignore[import-not-found]

            self._cirpy_available = True
        except ImportError:
            self._cirpy_available = False
            logging.info("CIRpy not available; CIR fallback disabled.")

    # ---------- Helpers ----------
    def _sleep(self):
        if self.sleep_sec:
            time.sleep(self.sleep_sec)

    def _clean_and_canonicalise_smiles(self, smiles: str) -> str | None:
        """Standardise a SMILES string by removing isotope labels, cleaning, and stripping salts.

        Parameters
        ----------
        smiles:
            Input SMILES string to clean.

        Returns
        -------
        str | None
            Canonical SMILES after standardisation, the original ``smiles`` if RDKit is
            unavailable, or ``None`` if the input is invalid or empty.
        """
        if not smiles:
            return None
        if not self._rdkit_available:
            logging.warning("RDKit not available; returning SMILES unchanged.")
            return smiles
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        rdMolStandardize.IsotopeParentInPlace(mol)  # removes isotope labels
        rdMolStandardize.CleanupInPlace(mol)  # normalize, sanitize, remove explicit Hs
        rdMolStandardize.RemoveFragmentsInPlace(mol)  # strip known solvents/salts
        rdMolStandardize.FragmentParentInPlace(mol, skipStandardize=True)  # keep largest remaining fragment
        rdMolStandardize.Uncharger().unchargeInPlace(mol)  # neutralize charges
        return Chem.MolToSmiles(mol)

    # ---------- Name cleaning helpers ----------
    @staticmethod
    def clean_name(name: str) -> str:
        """Strip trailing salt / formulation info from a drug name.

        Removes parenthesised suffixes that typically describe the salt form,
        e.g. ``"Almonertinib (hydrochloride)"`` becomes ``"Almonertinib"``.
        Leading parenthetical groups like ``"(R)-Verapamil"`` are kept.

        Parameters
        ----------
        name
            Raw drug name string.

        Returns
        -------
        str
            Cleaned name (unchanged when no salt suffix is detected).
        """
        idx = name.find(" (")
        return name[:idx].strip() if idx > 0 else name

    @staticmethod
    def classify_name(
        name: str,
    ) -> Literal["drug_name", "control", "target_description", "ambiguous"]:
        """Classify a drug-column entry into a semantic category.

        Categories:

        - ``"control"`` — vehicle / DMSO / untreated / mock.
        - ``"target_description"`` — gene-target pattern such as
          ``"ACLY.inhibitor"`` or ``"ACSS2.modulator"``.
        - ``"ambiguous"`` — very short strings (≤ 2 characters) that could be
          gene names, abbreviations, or noise.
        - ``"drug_name"`` — everything else (assumed to be a resolvable
          small-molecule name).

        Parameters
        ----------
        name
            Raw string from a drug column.

        Returns
        -------
        One of ``"drug_name"``, ``"control"``, ``"target_description"``,
        ``"ambiguous"``.
        """
        stripped = name.strip()
        if stripped.lower() in _CONTROL_TERMS:
            return "control"
        if _TARGET_RE.match(stripped):
            return "target_description"
        if len(stripped) <= 2:
            return "ambiguous"
        return "drug_name"

    @staticmethod
    def _name_variants(name: str) -> list[str]:
        """Generate candidate name variants for resolution, ordered most → least specific.

        Transformations applied (each only if it produces a new string):

        1. Original name
        2. Parenthesised salt stripped — ``"Almonertinib (hydrochloride)"`` → ``"Almonertinib"``
        3. Non-parenthesised salt stripped — ``"Elimusertib hydrochloride"`` → ``"Elimusertib"``
        4. Greek letters replaced — ``"18β-Glycyrrhetinic acid"`` → ``"18beta-Glycyrrhetinic acid"``
        5. Hyphens removed from compound codes — ``"AZD-8055"`` → ``"AZD8055"``
        6. Text after ``?`` removed — ``"Glesatinib?(MGCD265)"`` → ``"Glesatinib"``

        Stereochemistry prefixes like ``(R)-``, ``(S)-``, ``(+)-`` are
        **never** stripped — they encode distinct molecules.
        """
        seen: dict[str, None] = dict.fromkeys([name])

        paren_clean = DrugResolver.clean_name(name)
        if paren_clean != name:
            seen[paren_clean] = None

        no_salt = _SALT_SUFFIX_RE.sub("", name).strip()
        if no_salt != name:
            seen[no_salt] = None

        greek = name
        for g, latin in _GREEK_MAP.items():
            greek = greek.replace(g, latin)
        if greek != name:
            seen[greek] = None

        if re.match(r"^[A-Z]{2,}", name) and "-" in name:
            seen[name.replace("-", "")] = None

        if "?" in name:
            before_q = name.split("?")[0].strip()
            if before_q:
                seen[before_q] = None

        return list(seen)

    # ---------- Private resolution ----------
    @staticmethod
    def _extract_smiles(props: list[dict]) -> str | None:
        """Extract a SMILES string from a PubChem Properties response.

        Prefers ``IsomericSMILES`` (preserves stereochemistry) over
        ``CanonicalSMILES`` / ``ConnectivitySMILES`` (which drop it).
        """
        if not props:
            return None
        for key in ("IsomericSMILES", "CanonicalSMILES", "ConnectivitySMILES"):
            if key in props[0]:
                return props[0][key]
        return None

    def _try_resolve(self, name: str) -> str | None:
        """Try to resolve *name* to canonical SMILES via PubChem, NIH Cactus, and CIRpy."""
        q = _url_quote(name, safe="")

        # 1) Direct: name -> SMILES
        try:
            url = f"{PUBCHEM}/compound/name/{q}/property/IsomericSMILES/JSON"
            js = _get_json(url)
            props = js.get("PropertyTable", {}).get("Properties", [])
            smi = self._extract_smiles(props)
            if smi:
                return self._clean_and_canonicalise_smiles(smi) or smi
        except (requests.RequestException, ValueError):
            pass

        self._sleep()

        # 2) Fallback: name -> CID -> SMILES
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
                smi = self._extract_smiles(props)
                if smi:
                    return self._clean_and_canonicalise_smiles(smi) or smi
            else:
                logging.warning("No PubChem CID for name=%r", name)
        except (requests.RequestException, ValueError):
            pass

        self._sleep()

        # 3) NIH Cactus (raw URL)
        try:
            url = f"https://cactus.nci.nih.gov/chemical/structure/{q}/smiles"
            smi = _get_text(url).strip()
            if smi and "Error" not in smi:
                return self._clean_and_canonicalise_smiles(smi) or smi
        except (requests.RequestException, ValueError):
            pass

        # 4) CIRpy — tries multiple CIR resolvers (name_by_opsin, etc.)
        if self._cirpy_available:
            self._sleep()
            try:
                import cirpy  # type: ignore[import-not-found]

                smi = cirpy.resolve(name, "smiles")
                if smi:
                    return self._clean_and_canonicalise_smiles(smi) or smi
            except (OSError, ValueError):
                pass

        return None

    # ---------- Public API ----------
    def name_to_smiles(self, name: str) -> str | None:
        """Return PubChem canonical SMILES for a common / brand / IUPAC name.

        The method performs the following steps:

        1. **Classify** the input via :meth:`classify_name`.  Controls and
           target descriptions (e.g. ``"ACLY.inhibitor"``) are skipped
           immediately (returns ``None``).
        2. **Generate name variants** via :meth:`_name_variants` (salt
           stripping, stereochemistry prefix removal, Greek letter
           replacement, hyphen removal, etc.).
        3. **Try resolving** each variant in order through PubChem (direct
           lookup, CID fallback), NIH Cactus, and CIRpy until one succeeds.

        Parameters
        ----------
        name
            Drug name string (common name, brand name, or IUPAC name).

        Returns
        -------
        str | None
            Canonical SMILES, or ``None`` if the name cannot be resolved or
            is classified as a non-drug identifier.
        """
        classification = self.classify_name(name)
        if classification in ("control", "target_description"):
            logging.info(
                "Skipping non-drug identifier: %r (classified as %s)",
                name,
                classification,
            )
            return None

        for variant in self._name_variants(name):
            result = self._try_resolve(variant)
            if result is not None:
                return result
            self._sleep()

        logging.error("Could not resolve SMILES for %r", name)
        return None

    def smiles_to_names(self, smiles: str, top_k: int = 5) -> list[str]:
        """
        Return up to top_k names (Preferred/IUPAC/common) for a SMILES.

        Order: Preferred title (if available), then synonyms.
        """
        smi = self._clean_and_canonicalise_smiles(smiles) or smiles

        # 1) SMILES -> CID
        try:
            url = f"{PUBCHEM}/compound/smiles/{_url_quote(smi, safe='')}/cids/JSON"
            js = _get_json(url)
            cids = js.get("IdentifierList", {}).get("CID", [])
            if not cids:
                logging.warning("No PubChem CID for SMILES=%r", smi)
                return []
            cid = cids[0]
            self._sleep()
        except (requests.RequestException, ValueError):
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
        except (requests.RequestException, ValueError):
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
        except (requests.RequestException, ValueError):
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
        except (requests.RequestException, ValueError):
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
        except (requests.RequestException, ValueError):
            pass
        return names

    def smiles_to_primary_name(self, smiles: str) -> str | None:
        """Return a single best name for a SMILES (preferred title if present)."""
        names = self.smiles_to_names(smiles, top_k=1)
        return names[0] if names else None
