"""Fetch rich text descriptions for biological entities from public knowledge sources.

Supports genes, proteins, and small molecules from six databases:

1. **MyGene.info** -- NCBI gene summaries, names, types, pathways
2. **NCBI Gene** -- curated gene descriptions via Entrez E-utilities
3. **Ensembl** -- gene biotype and description from Ensembl REST
4. **UniProt** -- protein function, subcellular location, disease involvement
5. **Wikipedia** -- plain-text article extracts via the REST API
6. **PubChem** -- compound descriptions and pharmacology text
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Literal

import requests

logger = logging.getLogger(__name__)

MYGENE = "https://mygene.info/v3"
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ENSEMBL_REST = "https://rest.ensembl.org"
UNIPROT_REST = "https://rest.uniprot.org"
WIKIPEDIA_REST = "https://en.wikipedia.org/api/rest_v1"
PUBCHEM_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

ALL_GENE_SOURCES = ("mygene", "ncbi", "ensembl", "wikipedia")
ALL_PROTEIN_SOURCES = ("uniprot", "wikipedia")
ALL_MOLECULE_SOURCES = ("pubchem", "wikipedia")
ALL_SOURCES = ("mygene", "ncbi", "ensembl", "uniprot", "wikipedia", "pubchem")


def _get_json(url: str, params: dict | None = None,
              headers: dict | None = None, timeout: int = 30) -> dict | None:
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:  # noqa: BLE001
        logger.debug("Request failed for %s: %s", url, e)
        return None


def _get_text(url: str, params: dict | None = None,
              headers: dict | None = None, timeout: int = 30) -> str | None:
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.text
    except Exception as e:  # noqa: BLE001
        logger.debug("Request failed for %s: %s", url, e)
        return None


def _is_ensembl_id(s: str) -> bool:
    return bool(re.match(r"^ENS[A-Z]*[GTRPE]\d{11}", s, re.IGNORECASE))


class TextResolver:
    """Fetch text descriptions for genes, proteins, and molecules.

    Parameters
    ----------
    organism
        Default organism for gene/protein lookups (e.g. ``"human"``,
        ``"mouse"``).
    rate_limit_delay
        Seconds to wait between API calls.
    """

    def __init__(
        self,
        organism: str = "human",
        rate_limit_delay: float = 0.3,
    ) -> None:
        self.organism = organism
        self.delay = rate_limit_delay

    def _sleep(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    # ==================================================================
    # Source: MyGene.info
    # ==================================================================

    def _fetch_mygene(self, gene: str) -> str:
        """Fetch gene summary from MyGene.info."""
        scope = "ensembl.gene" if _is_ensembl_id(gene) else "symbol"
        data = _get_json(
            f"{MYGENE}/query",
            params={
                "q": gene,
                "scopes": scope,
                "fields": "symbol,name,summary,type_of_gene,pathway",
                "species": self.organism,
                "size": 1,
            },
        )
        if not data or "hits" not in data or not data["hits"]:
            return ""

        hit = data["hits"][0]
        parts = []
        symbol = hit.get("symbol", gene)
        name = hit.get("name", "")
        if name:
            parts.append(f"{symbol} ({name}).")
        summary = hit.get("summary", "")
        if summary:
            parts.append(summary)
        gene_type = hit.get("type_of_gene", "")
        if gene_type:
            parts.append(f"Gene type: {gene_type}.")

        pathways = hit.get("pathway", {})
        if isinstance(pathways, dict):
            pw_names = []
            for db_pws in pathways.values():
                if isinstance(db_pws, list):
                    pw_names.extend(p.get("name", "") for p in db_pws if isinstance(p, dict))
                elif isinstance(db_pws, dict):
                    pw_names.append(db_pws.get("name", ""))
            if pw_names:
                parts.append(f"Pathways: {', '.join(pw_names[:10])}.")

        return " ".join(parts).strip()

    # ==================================================================
    # Source: NCBI Gene (Entrez)
    # ==================================================================

    def _fetch_ncbi(self, gene: str) -> str:
        """Fetch gene description from NCBI Gene via E-utilities."""
        search = _get_json(
            f"{NCBI_EUTILS}/esearch.fcgi",
            params={
                "db": "gene",
                "term": f"{gene}[Gene Name] AND {self.organism}[Organism]",
                "retmode": "json",
                "retmax": 1,
            },
        )
        if not search:
            return ""
        ids = search.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return ""

        self._sleep()
        summary = _get_json(
            f"{NCBI_EUTILS}/esummary.fcgi",
            params={"db": "gene", "id": ids[0], "retmode": "json"},
        )
        if not summary:
            return ""

        result = summary.get("result", {}).get(ids[0], {})
        parts = []
        name = result.get("name", "")
        desc = result.get("description", "")
        nom_name = result.get("nomenclaturename", "")
        nom_symbol = result.get("nomenclaturesymbol", "")
        gene_summary = result.get("summary", "")

        if nom_symbol and nom_name:
            parts.append(f"{nom_symbol} ({nom_name}).")
        elif name:
            parts.append(f"{gene} ({name}).")
        if desc:
            parts.append(desc)
        if gene_summary:
            parts.append(gene_summary)

        return " ".join(parts).strip()

    # ==================================================================
    # Source: Ensembl REST
    # ==================================================================

    def _fetch_ensembl(self, gene: str) -> str:
        """Fetch gene description from Ensembl REST API."""
        if _is_ensembl_id(gene):
            url = f"{ENSEMBL_REST}/lookup/id/{gene}"
        else:
            url = f"{ENSEMBL_REST}/lookup/symbol/{self.organism}/{gene}"

        data = _get_json(url, headers={"Content-Type": "application/json"})
        if not data:
            return ""

        parts = []
        display = data.get("display_name", "")
        desc = data.get("description", "")
        biotype = data.get("biotype", "")

        if display:
            parts.append(f"{display}.")
        if desc:
            desc_clean = re.sub(r"\s*\[Source:.*?\]", "", desc)
            parts.append(desc_clean.strip())
        if biotype:
            parts.append(f"Biotype: {biotype}.")

        return " ".join(parts).strip()

    # ==================================================================
    # Source: UniProt
    # ==================================================================

    def _fetch_uniprot(self, identifier: str) -> str:
        """Fetch protein function description from UniProt."""
        from .protein_resolver import ProteinResolver, ORGANISM_TAXON

        if re.match(r"^[A-Z][0-9][A-Z0-9]{3}[0-9](-\d+)?$", identifier):
            accession = identifier
        else:
            pr = ProteinResolver(organism=self.organism)
            id_type = "ensembl_id" if _is_ensembl_id(identifier) else "symbol"
            accession = pr.resolve_uniprot_id(identifier, id_type=id_type)
            if not accession:
                return ""

        data = _get_json(
            f"{UNIPROT_REST}/uniprotkb/{accession}.json",
            headers={"Accept": "application/json"},
        )
        if not data:
            return ""

        parts = []
        protein_name = ""
        prot_desc = data.get("proteinDescription", {})
        rec_name = prot_desc.get("recommendedName", {})
        if rec_name:
            full = rec_name.get("fullName", {}).get("value", "")
            if full:
                protein_name = full
                parts.append(f"{full}.")

        comments = data.get("comments", [])
        for comment in comments:
            ct = comment.get("commentType", "")
            texts = comment.get("texts", [])
            if ct == "FUNCTION" and texts:
                for t in texts:
                    val = t.get("value", "")
                    if val:
                        parts.append(val)
            elif ct == "SUBCELLULAR LOCATION":
                locs = comment.get("subcellularLocations", [])
                loc_names = []
                for loc in locs:
                    loc_val = loc.get("location", {}).get("value", "")
                    if loc_val:
                        loc_names.append(loc_val)
                if loc_names:
                    parts.append(f"Subcellular location: {', '.join(loc_names)}.")
            elif ct == "DISEASE" and texts:
                for t in texts:
                    val = t.get("value", "")
                    if val:
                        parts.append(f"Disease: {val}")
                        break

        return " ".join(parts).strip()

    # ==================================================================
    # Source: Wikipedia
    # ==================================================================

    def _fetch_wikipedia(self, query: str) -> str:
        """Fetch plain-text extract from Wikipedia."""
        title = query.replace(" ", "_")
        data = _get_json(f"{WIKIPEDIA_REST}/page/summary/{title}")
        if not data:
            return ""

        extract = data.get("extract", "")
        if not extract or data.get("type") == "disambiguation":
            gene_title = f"{title} (gene)"
            data = _get_json(f"{WIKIPEDIA_REST}/page/summary/{gene_title}")
            if data:
                extract = data.get("extract", "")

        return extract.strip() if extract else ""

    # ==================================================================
    # Source: PubChem
    # ==================================================================

    def _fetch_pubchem(self, identifier: str) -> str:
        """Fetch compound description from PubChem."""
        data = _get_json(
            f"{PUBCHEM_REST}/compound/name/{identifier}/description/JSON",
        )
        if not data:
            data = _get_json(
                f"{PUBCHEM_REST}/compound/smiles/{identifier}/description/JSON",
            )
        if not data:
            return ""

        descriptions = data.get("InformationList", {}).get("Information", [])
        parts = []
        for info in descriptions[:3]:
            desc = info.get("Description", "")
            if desc and len(desc) > 20:
                parts.append(desc)
                break
        return " ".join(parts).strip()

    # ==================================================================
    # Public API
    # ==================================================================

    def get_gene_description(
        self,
        gene: str,
        sources: list[str] | str = "all",
    ) -> dict[str, str]:
        """Fetch text descriptions for a gene from multiple sources.

        Parameters
        ----------
        gene
            Gene symbol or Ensembl ID.
        sources
            Source names to query. ``"all"`` queries MyGene, NCBI,
            Ensembl, and Wikipedia. Pass a list to select specific ones.

        Returns
        -------
        Dict mapping source name to description text. Empty string for
        sources that returned no results.
        """
        if sources == "all":
            src_list = list(ALL_GENE_SOURCES)
        else:
            src_list = list(sources)

        result: dict[str, str] = {}
        for src in src_list:
            self._sleep()
            if src == "mygene":
                result["mygene"] = self._fetch_mygene(gene)
            elif src == "ncbi":
                result["ncbi"] = self._fetch_ncbi(gene)
            elif src == "ensembl":
                result["ensembl"] = self._fetch_ensembl(gene)
            elif src == "wikipedia":
                result["wikipedia"] = self._fetch_wikipedia(gene)
            elif src == "uniprot":
                result["uniprot"] = self._fetch_uniprot(gene)
            else:
                logger.warning("Unknown gene source '%s'", src)
        return result

    def get_protein_description(
        self,
        protein: str,
        sources: list[str] | str = "all",
    ) -> dict[str, str]:
        """Fetch text descriptions for a protein.

        Parameters
        ----------
        protein
            Gene symbol, Ensembl ID, or UniProt accession.
        sources
            ``"all"`` queries UniProt and Wikipedia.

        Returns
        -------
        Dict mapping source name to description text.
        """
        if sources == "all":
            src_list = list(ALL_PROTEIN_SOURCES)
        else:
            src_list = list(sources)

        result: dict[str, str] = {}
        for src in src_list:
            self._sleep()
            if src == "uniprot":
                result["uniprot"] = self._fetch_uniprot(protein)
            elif src == "wikipedia":
                result["wikipedia"] = self._fetch_wikipedia(protein)
            elif src == "mygene":
                result["mygene"] = self._fetch_mygene(protein)
            else:
                logger.warning("Unknown protein source '%s'", src)
        return result

    def get_molecule_description(
        self,
        identifier: str,
        sources: list[str] | str = "all",
    ) -> dict[str, str]:
        """Fetch text descriptions for a small molecule.

        Parameters
        ----------
        identifier
            Drug name, SMILES, or PubChem CID.
        sources
            ``"all"`` queries PubChem and Wikipedia.

        Returns
        -------
        Dict mapping source name to description text.
        """
        if sources == "all":
            src_list = list(ALL_MOLECULE_SOURCES)
        else:
            src_list = list(sources)

        result: dict[str, str] = {}
        for src in src_list:
            self._sleep()
            if src == "pubchem":
                result["pubchem"] = self._fetch_pubchem(identifier)
            elif src == "wikipedia":
                result["wikipedia"] = self._fetch_wikipedia(identifier)
            else:
                logger.warning("Unknown molecule source '%s'", src)
        return result

    def get_description(
        self,
        identifier: str,
        entity_type: Literal["gene", "protein", "molecule", "cellline", "auto"] = "auto",
        sources: list[str] | str = "all",
    ) -> dict[str, str]:
        """Fetch text descriptions, auto-detecting entity type if needed.

        Parameters
        ----------
        identifier
            Gene symbol, Ensembl ID, UniProt accession, drug name, or SMILES.
        entity_type
            ``"auto"`` detects the type heuristically.
        sources
            Sources to query (``"all"`` for the entity type's defaults).

        Returns
        -------
        Dict mapping source name to description text.
        """
        if entity_type == "auto":
            entity_type = self._detect_entity_type(identifier)

        if entity_type == "gene":
            return self.get_gene_description(identifier, sources=sources)
        elif entity_type == "protein":
            return self.get_protein_description(identifier, sources=sources)
        elif entity_type == "molecule":
            return self.get_molecule_description(identifier, sources=sources)
        elif entity_type == "cellline":
            return self.get_cellline_description(identifier)
        else:
            raise ValueError(f"Unknown entity_type '{entity_type}'")

    def get_combined_description(
        self,
        identifier: str,
        entity_type: Literal["gene", "protein", "molecule", "cellline", "auto"] = "auto",
        sources: list[str] | str = "all",
        template: str | None = None,
    ) -> str:
        """Fetch from all sources and combine into a single text string.

        Parameters
        ----------
        identifier
            Biological entity identifier.
        entity_type
            ``"auto"`` to auto-detect.
        sources
            Sources to query.
        template
            Custom template string. If ``None``, uses section-header format.
            The template receives ``{identifier}``, ``{entity_type}``, and
            each source name as a keyword (e.g. ``{mygene}``, ``{uniprot}``).

        Returns
        -------
        Combined description string suitable for text embedding.
        """
        if entity_type == "auto":
            entity_type = self._detect_entity_type(identifier)

        descs = self.get_description(identifier, entity_type=entity_type, sources=sources)

        if template is not None:
            return template.format(
                identifier=identifier,
                entity_type=entity_type,
                **descs,
            )

        parts = []
        for source, text in descs.items():
            if text:
                parts.append(f"[{source.upper()}] {text}")

        if not parts:
            return f"{identifier} ({entity_type}). No description available."

        return "\n".join(parts)

    # ==================================================================
    # Cell line descriptions
    # ==================================================================

    def get_cellline_description(self, name: str) -> dict[str, str]:
        """Fetch text description for a cell line.

        Uses ``CellLineAnnotator`` to gather metadata from Cellosaurus,
        DepMap, and Cell Model Passports, then returns a dict with a
        single ``"cellline"`` key containing the combined text.

        Parameters
        ----------
        name
            Cell line name or identifier.

        Returns
        -------
        Dict with key ``"cellline"`` -> description text.
        """
        from .cellline_annotator import CellLineAnnotator

        ann = CellLineAnnotator(rate_limit_delay=self.delay)
        text = ann.get_text_description(name)
        return {"cellline": text}

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _detect_entity_type(
        identifier: str,
    ) -> Literal["gene", "protein", "molecule", "cellline"]:
        """Heuristic detection of entity type."""
        s = identifier.strip()

        if re.match(r"^ACH-\d{6}$", s):
            return "cellline"
        if re.match(r"^CVCL_[A-Za-z0-9]+$", s):
            return "cellline"

        smiles_chars = set("=()[]#@+\\/-.")
        if len(s) > 5 and (smiles_chars & set(s)):
            return "molecule"

        if _is_ensembl_id(s):
            return "gene"

        if re.match(r"^[A-Z][0-9][A-Z0-9]{3}[0-9](-\d+)?$", s):
            return "protein"

        if s.isupper() and len(s) <= 15:
            return "gene"

        return "gene"
