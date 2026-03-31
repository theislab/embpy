"""Cell line annotation from multiple public databases.

Aggregates metadata for a cell line (given by name or ID) from
three sources:

1. **Cellosaurus** -- species, tissue, disease, cross-references,
   STR profiles, contamination flags (150k+ cell lines)
2. **DepMap / CCLE** -- lineage, primary disease, growth properties,
   driver mutations (~2000 cancer cell lines)
3. **Cell Model Passports** (Sanger) -- cancer type, MSI status,
   ploidy, mutational burden, model type (~1700 models)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

CELLOSAURUS_API = "https://api.cellosaurus.org"
DEPMAP_API = "https://depmap.org/portal/api"
CMP_API = "https://api.cellmodelpassports.sanger.ac.uk/v1"

ALL_SOURCES = ("cellosaurus", "depmap", "passports")


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


class CellLineAnnotator:
    """Aggregate annotations for cell lines from public databases.

    Parameters
    ----------
    rate_limit_delay
        Seconds to wait between API calls (default 0.3).
    """

    def __init__(self, rate_limit_delay: float = 0.3) -> None:
        self.delay = rate_limit_delay

    def _sleep(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    # ==================================================================
    # Identifier helpers
    # ==================================================================

    @staticmethod
    def _is_cellosaurus_id(s: str) -> bool:
        return bool(re.match(r"^CVCL_[A-Za-z0-9]+$", s))

    @staticmethod
    def _is_depmap_id(s: str) -> bool:
        return bool(re.match(r"^ACH-\d{6}$", s))

    # ==================================================================
    # Source: Cellosaurus
    # ==================================================================

    def _resolve_cellosaurus_id(self, name: str) -> str | None:
        """Resolve a cell line name to a Cellosaurus accession."""
        if self._is_cellosaurus_id(name):
            return name

        data = _get_json(
            f"{CELLOSAURUS_API}/search/cell-line",
            params={"q": f'id:"{name}"', "format": "json", "rows": 1},
        )
        if not data:
            data = _get_json(
                f"{CELLOSAURUS_API}/search/cell-line",
                params={"q": name, "format": "json", "rows": 1},
            )
        if not data:
            return None

        results = data.get("resultList", {}).get("result", [])
        if not results:
            return None
        return results[0].get("accession", None)

    def get_cellosaurus_info(self, name: str) -> dict[str, Any]:
        """Query Cellosaurus API for cell line metadata.

        Parameters
        ----------
        name
            Cell line name or Cellosaurus accession (e.g. ``"CVCL_0023"``).

        Returns
        -------
        Dict with keys: ``name``, ``accession``, ``category``,
        ``species``, ``tissue``, ``disease``, ``cross_references``,
        ``is_problematic``, ``parent_cell_line``.
        """
        acc = self._resolve_cellosaurus_id(name)
        if not acc:
            return {}

        self._sleep()
        data = _get_json(f"{CELLOSAURUS_API}/cell-line/{acc}", params={"format": "json"})
        if not data:
            return {}

        result: dict[str, Any] = {
            "name": "",
            "accession": acc,
            "category": "",
            "species": "",
            "tissue": "",
            "disease": "",
            "cross_references": {},
            "is_problematic": False,
            "parent_cell_line": "",
        }

        cl = data.get("cellLine", data) if "cellLine" in data else data

        name_list = cl.get("nameList", [])
        if name_list:
            first_name = name_list[0] if isinstance(name_list[0], str) else name_list[0].get("value", "")
            result["name"] = first_name

        result["category"] = cl.get("category", "")

        species_list = cl.get("speciesList", [])
        if species_list:
            sp = species_list[0]
            result["species"] = sp.get("value", "") if isinstance(sp, dict) else str(sp)

        derived_from = cl.get("derivedFromSite", [])
        if derived_from:
            site = derived_from[0]
            result["tissue"] = site.get("value", "") if isinstance(site, dict) else str(site)

        disease_list = cl.get("diseaseList", [])
        if disease_list:
            d = disease_list[0]
            result["disease"] = d.get("value", "") if isinstance(d, dict) else str(d)

        xref_list = cl.get("xrefList", [])
        xrefs = {}
        for xref in xref_list:
            if isinstance(xref, dict):
                db = xref.get("database", "")
                xid = xref.get("accession", "")
                if db and xid:
                    xrefs[db] = xid
        result["cross_references"] = xrefs

        problem_list = cl.get("registrationProblemList", [])
        result["is_problematic"] = len(problem_list) > 0

        parent = cl.get("parentCellLine", {})
        if isinstance(parent, dict):
            result["parent_cell_line"] = parent.get("value", "")

        return result

    # ==================================================================
    # Source: DepMap / CCLE
    # ==================================================================

    def get_depmap_info(self, name: str) -> dict[str, Any]:
        """Query DepMap portal for cell line info.

        Parameters
        ----------
        name
            Cell line name (e.g. ``"A549"``) or DepMap ID (``"ACH-000681"``).

        Returns
        -------
        Dict with keys: ``depmap_id``, ``cell_line_name``, ``lineage``,
        ``lineage_subtype``, ``primary_disease``, ``disease_subtype``,
        ``growth_pattern``.
        """
        self._sleep()

        data = _get_json(
            f"{DEPMAP_API}/cell_line",
            params={"name": name},
        )
        if not data and self._is_depmap_id(name):
            data = _get_json(
                f"{DEPMAP_API}/cell_line/{name}",
            )
        if not data:
            data = _get_json(
                f"{DEPMAP_API}/cell_line",
                params={"q": name},
            )

        if not data:
            return {}

        if isinstance(data, list):
            data = data[0] if data else {}

        return {
            "depmap_id": data.get("depmap_id", ""),
            "cell_line_name": data.get("cell_line_name", data.get("stripped_cell_line_name", "")),
            "lineage": data.get("lineage", ""),
            "lineage_subtype": data.get("lineage_subtype", ""),
            "primary_disease": data.get("primary_disease", ""),
            "disease_subtype": data.get("disease_subtype", ""),
            "growth_pattern": data.get("growth_pattern", ""),
            "culture_medium": data.get("culture_medium", ""),
            "sex": data.get("sex", ""),
            "source": data.get("source", ""),
        }

    # ==================================================================
    # Source: Cell Model Passports (Sanger)
    # ==================================================================

    def get_passports_info(self, name: str) -> dict[str, Any]:
        """Query Cell Model Passports API.

        Parameters
        ----------
        name
            Cell line / model name (e.g. ``"A549"``).

        Returns
        -------
        Dict with keys: ``model_id``, ``model_name``, ``tissue``,
        ``cancer_type``, ``cancer_type_detail``, ``model_type``,
        ``msi_status``, ``ploidy``, ``mutational_burden``.
        """
        self._sleep()
        data = _get_json(
            f"{CMP_API}/models",
            params={"filter[model_name]": name, "page[size]": 1},
        )
        if not data:
            return {}

        records = data.get("data", [])
        if not records:
            data = _get_json(
                f"{CMP_API}/models",
                params={"filter[names]": name, "page[size]": 1},
            )
            if data:
                records = data.get("data", [])

        if not records:
            return {}

        attrs = records[0].get("attributes", {})
        return {
            "model_id": records[0].get("id", ""),
            "model_name": attrs.get("model_name", ""),
            "tissue": attrs.get("tissue", ""),
            "cancer_type": attrs.get("cancer_type", ""),
            "cancer_type_detail": attrs.get("cancer_type_detail", ""),
            "model_type": attrs.get("model_type", ""),
            "msi_status": attrs.get("msi_status", ""),
            "ploidy": attrs.get("ploidy", ""),
            "mutational_burden": attrs.get("mutational_burden", ""),
            "growth_properties": attrs.get("growth_properties", ""),
        }

    # ==================================================================
    # Combined annotation
    # ==================================================================

    def annotate(
        self,
        name: str,
        sources: str | list[str] = "all",
    ) -> dict[str, Any]:
        """Aggregate metadata from all sources for a cell line.

        Parameters
        ----------
        name
            Cell line name or identifier.
        sources
            Which sources to query: ``"all"`` or a list like
            ``["cellosaurus", "depmap"]``.

        Returns
        -------
        Combined annotation dict with top-level summary fields and
        per-source details under ``"sources"``.
        """
        if sources == "all":
            src_list = list(ALL_SOURCES)
        else:
            src_list = list(sources)

        result: dict[str, Any] = {
            "name": name,
            "species": "",
            "tissue": "",
            "disease": "",
            "lineage": "",
            "lineage_subtype": "",
            "growth_pattern": "",
            "model_type": "",
            "msi_status": "",
            "is_problematic": False,
            "cellosaurus_id": "",
            "depmap_id": "",
            "cross_references": {},
            "sources": {},
        }

        if "cellosaurus" in src_list:
            cello = self.get_cellosaurus_info(name)
            result["sources"]["cellosaurus"] = cello
            if cello:
                result["cellosaurus_id"] = cello.get("accession", "")
                result["species"] = result["species"] or cello.get("species", "")
                result["tissue"] = result["tissue"] or cello.get("tissue", "")
                result["disease"] = result["disease"] or cello.get("disease", "")
                result["is_problematic"] = cello.get("is_problematic", False)
                result["cross_references"].update(cello.get("cross_references", {}))

        if "depmap" in src_list:
            depmap = self.get_depmap_info(name)
            result["sources"]["depmap"] = depmap
            if depmap:
                result["depmap_id"] = depmap.get("depmap_id", "")
                result["lineage"] = depmap.get("lineage", "")
                result["lineage_subtype"] = depmap.get("lineage_subtype", "")
                result["disease"] = result["disease"] or depmap.get("primary_disease", "")
                result["growth_pattern"] = depmap.get("growth_pattern", "")

        if "passports" in src_list:
            passports = self.get_passports_info(name)
            result["sources"]["passports"] = passports
            if passports:
                result["tissue"] = result["tissue"] or passports.get("tissue", "")
                result["disease"] = result["disease"] or passports.get("cancer_type", "")
                result["model_type"] = passports.get("model_type", "")
                result["msi_status"] = passports.get("msi_status", "")

        return result

    def annotate_batch(
        self,
        names: list[str],
        sources: str | list[str] = "all",
    ) -> list[dict[str, Any]]:
        """Annotate multiple cell lines.

        Parameters
        ----------
        names
            List of cell line names or identifiers.
        sources
            Sources to query.

        Returns
        -------
        List of annotation dicts, one per cell line.
        """
        results = []
        for name in names:
            try:
                results.append(self.annotate(name, sources=sources))
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to annotate cell line '%s': %s", name, e)
                results.append({"name": name, "error": str(e)})
        return results

    def annotate_adata(
        self,
        adata,
        column: str,
        sources: str | list[str] = "all",
    ):
        """Annotate cell lines in an AnnData object.

        Parameters
        ----------
        adata
            AnnData with cell line identifiers in ``obs[column]``.
        column
            Column name containing cell line names.
        sources
            Sources to query.

        Returns
        -------
        The AnnData with annotation columns added to ``.obs`` and
        full annotation dicts in ``.uns["cellline_annotations"]``.
        """
        if column not in adata.obs.columns:
            raise KeyError(f"Column '{column}' not in adata.obs")

        unique_names = adata.obs[column].dropna().unique().tolist()
        logger.info("Annotating %d unique cell lines from column '%s'", len(unique_names), column)

        annotations = {}
        for name in unique_names:
            annotations[name] = self.annotate(str(name), sources=sources)

        for field in ("species", "tissue", "disease", "lineage", "model_type", "msi_status"):
            vals = []
            for cl in adata.obs[column]:
                ann = annotations.get(str(cl), {})
                vals.append(ann.get(field, ""))
            adata.obs[f"cellline_{field}"] = vals

        adata.uns["cellline_annotations"] = annotations
        return adata

    # ==================================================================
    # Text description
    # ==================================================================

    # ==================================================================
    # Wikipedia
    # ==================================================================

    def get_wikipedia_info(self, name: str) -> str:
        """Fetch the Wikipedia article extract for a cell line.

        Parameters
        ----------
        name
            Cell line name (e.g. ``"HeLa"``, ``"A549"``).

        Returns
        -------
        Plain-text extract from the Wikipedia article, or ``""``
        if no article is found.
        """
        self._sleep()
        title = name.replace(" ", "_")
        data = _get_json(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
        )
        if data and data.get("extract"):
            if data.get("type") != "disambiguation":
                return data["extract"].strip()

        cell_title = f"{title}_(cell_line)"
        data = _get_json(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{cell_title}",
        )
        if data and data.get("extract"):
            return data["extract"].strip()

        cells_title = f"{title}_cells"
        data = _get_json(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{cells_title}",
        )
        if data and data.get("extract"):
            return data["extract"].strip()

        return ""

    # ==================================================================
    # Text description
    # ==================================================================

    def get_text_description(self, name: str) -> str:
        """Build a text description of a cell line for embedding.

        Combines structured metadata from Cellosaurus/DepMap/Passports
        with the Wikipedia article extract into a single text suitable
        for text embedding models.

        Parameters
        ----------
        name
            Cell line name or identifier.

        Returns
        -------
        Text description string.
        """
        ann = self.annotate(name)

        parts = []
        cl_name = ann.get("name", name)
        parts.append(f"{cl_name} is a cell line.")

        species = ann.get("species", "")
        if species:
            parts.append(f"Species: {species}.")

        tissue = ann.get("tissue", "")
        if tissue:
            parts.append(f"Derived from {tissue} tissue.")

        disease = ann.get("disease", "")
        if disease:
            parts.append(f"Disease: {disease}.")

        lineage = ann.get("lineage", "")
        if lineage:
            subtype = ann.get("lineage_subtype", "")
            if subtype:
                parts.append(f"Lineage: {lineage} ({subtype}).")
            else:
                parts.append(f"Lineage: {lineage}.")

        growth = ann.get("growth_pattern", "")
        if growth:
            parts.append(f"Growth pattern: {growth}.")

        model_type = ann.get("model_type", "")
        if model_type:
            parts.append(f"Model type: {model_type}.")

        msi = ann.get("msi_status", "")
        if msi:
            parts.append(f"Microsatellite instability status: {msi}.")

        if ann.get("is_problematic"):
            parts.append("Warning: this cell line has known problems (contamination or misidentification).")

        wiki = self.get_wikipedia_info(cl_name or name)
        if wiki:
            parts.append(wiki)

        return " ".join(parts)
