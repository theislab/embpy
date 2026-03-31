"""Protein annotation from UniProt and complementary databases.

Aggregates functional metadata for a protein (given as UniProt
accession, gene symbol, or Ensembl ID) from multiple sources:

1. **Functional annotation** -- UniProt (function, catalytic activity,
   pathway, keywords)
2. **Subcellular location** -- UniProt comments
3. **Active sites, binding sites & motifs** -- UniProt features
4. **Domains & families** -- UniProt features + InterPro REST API
5. **Post-translational modifications** -- UniProt features
6. **Disease & variant associations** -- UniProt disease comments
   and variant features
7. **GO terms** -- UniProt cross-references (molecular function,
   biological process, cellular component)
8. **Protein-protein interactions** -- UniProt cross-references to
   IntAct / STRING
9. **Isoform-specific annotations** -- UniProt alternative products
10. **Review status** -- Swiss-Prot (reviewed) vs TrEMBL (unreviewed)

Does **not** reimplement pertpy functionality.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

import requests

logger = logging.getLogger(__name__)

UNIPROT_REST = "https://rest.uniprot.org"
INTERPRO_API = "https://www.ebi.ac.uk/interpro/api"


def _get_json(url: str, params: dict | None = None, timeout: int = 30) -> dict | list | None:
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:  # noqa: BLE001
        logger.debug("Request failed for %s: %s", url, e)
        return None


class ProteinAnnotator:
    """Aggregate functional annotations for proteins from UniProt.

    Parameters
    ----------
    organism : str
        Default organism (default ``"human"``).
    rate_limit_delay : float
        Seconds between API calls (default 0.2).
    """

    def __init__(
        self,
        organism: str = "human",
        rate_limit_delay: float = 0.2,
    ) -> None:
        self.organism = organism
        self.delay = rate_limit_delay

    def _sleep(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    # ==================================================================
    # UniProt entry fetching
    # ==================================================================

    def _resolve_uniprot_accession(self, identifier: str, id_type: str = "auto") -> str | None:
        """Resolve any identifier to a UniProt accession."""
        s = identifier.strip()

        if id_type == "uniprot_id" or (id_type == "auto" and len(s) <= 10 and s[0].isalpha()):
            if "-" in s:
                return s.split("-")[0]
            return s

        from .protein_resolver import ProteinResolver
        pr = ProteinResolver(organism=self.organism)
        import re
        if re.match(r"^ENS[A-Z]*G\d{11}", s, re.IGNORECASE):
            return pr.resolve_uniprot_id(s, id_type="ensembl_id")
        return pr.resolve_uniprot_id(s, id_type="symbol")

    def _fetch_uniprot_entry(self, accession: str) -> dict | None:
        """Fetch the full UniProt JSON entry for an accession."""
        self._sleep()
        return _get_json(f"{UNIPROT_REST}/uniprotkb/{accession}.json")

    # ==================================================================
    # 1. Functional annotation
    # ==================================================================

    def get_function(self, entry: dict) -> dict[str, Any]:
        """Extract function description, catalytic activity, and pathway."""
        result: dict[str, Any] = {
            "function": [],
            "catalytic_activity": [],
            "pathway": [],
            "keywords": [],
        }

        for comment in entry.get("comments", []):
            ctype = comment.get("commentType", "")
            if ctype == "FUNCTION":
                for text in comment.get("texts", []):
                    result["function"].append(text.get("value", ""))
            elif ctype == "CATALYTIC ACTIVITY":
                reaction = comment.get("reaction", {})
                result["catalytic_activity"].append({
                    "name": reaction.get("name", ""),
                    "ec": reaction.get("ecNumber", ""),
                })
            elif ctype == "PATHWAY":
                for text in comment.get("texts", []):
                    result["pathway"].append(text.get("value", ""))

        for kw in entry.get("keywords", []):
            result["keywords"].append({
                "id": kw.get("id", ""),
                "name": kw.get("name", ""),
                "category": kw.get("category", ""),
            })

        return result

    # ==================================================================
    # 2. Subcellular location
    # ==================================================================

    def get_subcellular_location(self, entry: dict) -> list[dict[str, str]]:
        """Extract subcellular location annotations."""
        locations = []
        for comment in entry.get("comments", []):
            if comment.get("commentType") == "SUBCELLULAR LOCATION":
                for subloc in comment.get("subcellularLocations", []):
                    loc = subloc.get("location", {})
                    locations.append({
                        "location": loc.get("value", ""),
                        "topology": subloc.get("topology", {}).get("value", ""),
                        "orientation": subloc.get("orientation", {}).get("value", ""),
                    })
        return locations

    # ==================================================================
    # 3. Active sites, binding sites & motifs
    # ==================================================================

    def get_functional_sites(self, entry: dict) -> dict[str, list[dict[str, Any]]]:
        """Extract active sites, binding sites, and motifs from features."""
        result: dict[str, list[dict[str, Any]]] = {
            "active_sites": [],
            "binding_sites": [],
            "motifs": [],
            "regions": [],
        }

        for feat in entry.get("features", []):
            ftype = feat.get("type", "")
            loc = feat.get("location", {})
            start = loc.get("start", {}).get("value")
            end_val = loc.get("end", {}).get("value")
            desc = feat.get("description", "")

            info = {"start": start, "end": end_val, "description": desc}

            if ftype == "Active site":
                result["active_sites"].append(info)
            elif ftype == "Binding site":
                ligand = feat.get("ligand", {})
                info["ligand"] = ligand.get("name", "")
                result["binding_sites"].append(info)
            elif ftype == "Motif":
                result["motifs"].append(info)
            elif ftype == "Region":
                result["regions"].append(info)

        return result

    # ==================================================================
    # 4. Domains & families
    # ==================================================================

    def get_domains(self, entry: dict) -> list[dict[str, Any]]:
        """Extract domain annotations from UniProt features."""
        domains = []
        for feat in entry.get("features", []):
            if feat.get("type") == "Domain":
                loc = feat.get("location", {})
                domains.append({
                    "name": feat.get("description", ""),
                    "start": loc.get("start", {}).get("value"),
                    "end": loc.get("end", {}).get("value"),
                })
        return domains

    def get_interpro_domains(self, accession: str) -> list[dict[str, str]]:
        """Fetch domain/family annotations from InterPro."""
        self._sleep()
        data = _get_json(
            f"{INTERPRO_API}/protein/UniProt/{accession}",
        )
        if not data or not isinstance(data, dict):
            return []

        domains = []
        metadata = data.get("metadata", {})
        if metadata:
            domains.append({
                "source": "InterPro",
                "accession": metadata.get("accession", ""),
                "name": metadata.get("name", ""),
                "type": metadata.get("type", ""),
            })

        # Also fetch entry matches
        self._sleep()
        match_data = _get_json(
            f"{INTERPRO_API}/entry/interpro/protein/UniProt/{accession}",
        )
        if match_data and isinstance(match_data, dict):
            for result in match_data.get("results", []):
                meta = result.get("metadata", {})
                domains.append({
                    "source": "InterPro",
                    "accession": meta.get("accession", ""),
                    "name": meta.get("name", ""),
                    "type": meta.get("type", ""),
                })

        return domains

    # ==================================================================
    # 5. Post-translational modifications
    # ==================================================================

    def get_ptms(self, entry: dict) -> list[dict[str, Any]]:
        """Extract post-translational modification features."""
        ptm_types = {
            "Modified residue", "Glycosylation", "Disulfide bond",
            "Cross-link", "Lipidation",
        }
        ptms = []
        for feat in entry.get("features", []):
            if feat.get("type") in ptm_types:
                loc = feat.get("location", {})
                ptms.append({
                    "type": feat.get("type", ""),
                    "description": feat.get("description", ""),
                    "start": loc.get("start", {}).get("value"),
                    "end": loc.get("end", {}).get("value"),
                })
        return ptms

    # ==================================================================
    # 6. Disease & variant associations
    # ==================================================================

    def get_disease_associations(self, entry: dict) -> list[dict[str, str]]:
        """Extract disease involvement from UniProt comments."""
        diseases = []
        for comment in entry.get("comments", []):
            if comment.get("commentType") == "DISEASE":
                disease = comment.get("disease", {})
                diseases.append({
                    "name": disease.get("diseaseId", ""),
                    "description": disease.get("description", ""),
                    "acronym": disease.get("acronym", ""),
                    "mim_id": str(disease.get("diseaseCrossReference", {}).get("id", "")),
                })
        return diseases

    def get_variants(self, entry: dict) -> list[dict[str, Any]]:
        """Extract sequence variant features (mutations, polymorphisms)."""
        variants = []
        for feat in entry.get("features", []):
            if feat.get("type") in ("Natural variant", "Mutagenesis"):
                loc = feat.get("location", {})
                variants.append({
                    "type": feat.get("type", ""),
                    "position": loc.get("start", {}).get("value"),
                    "original": feat.get("alternativeSequence", {}).get("originalSequence", ""),
                    "variation": ", ".join(
                        feat.get("alternativeSequence", {}).get("alternativeSequences", [])
                    ),
                    "description": feat.get("description", ""),
                })
        return variants

    # ==================================================================
    # 7. GO terms
    # ==================================================================

    def get_go_terms(self, entry: dict) -> dict[str, list[dict[str, str]]]:
        """Extract Gene Ontology terms from cross-references."""
        go: dict[str, list[dict[str, str]]] = {
            "molecular_function": [],
            "biological_process": [],
            "cellular_component": [],
        }
        aspect_map = {"F": "molecular_function", "P": "biological_process", "C": "cellular_component"}

        for xref in entry.get("uniProtKBCrossReferences", []):
            if xref.get("database") == "GO":
                go_id = xref.get("id", "")
                props = {p["key"]: p["value"] for p in xref.get("properties", [])}
                term = props.get("GoTerm", "")
                aspect_code = term[0] if term else ""
                aspect = aspect_map.get(aspect_code, "")

                if aspect:
                    go[aspect].append({
                        "id": go_id,
                        "term": term[2:] if len(term) > 2 else term,
                        "evidence": props.get("GoEvidenceType", ""),
                    })
        return go

    # ==================================================================
    # 8. Protein-protein interactions (cross-refs)
    # ==================================================================

    def get_interaction_xrefs(self, entry: dict) -> list[dict[str, str]]:
        """Extract PPI database cross-references (IntAct, STRING)."""
        interactions = []
        for xref in entry.get("uniProtKBCrossReferences", []):
            db = xref.get("database", "")
            if db in ("IntAct", "STRING", "BioGRID", "MINT"):
                interactions.append({
                    "database": db,
                    "id": xref.get("id", ""),
                })

        for comment in entry.get("comments", []):
            if comment.get("commentType") == "INTERACTION":
                for interaction in comment.get("interactions", []):
                    partner = interaction.get("interactantTwo", {})
                    interactions.append({
                        "database": "UniProt",
                        "id": partner.get("uniProtKBAccession", ""),
                        "gene": partner.get("geneName", ""),
                        "n_experiments": interaction.get("numberOfExperiments", 0),
                    })
        return interactions

    # ==================================================================
    # 9. Isoform-specific annotations
    # ==================================================================

    def get_isoform_annotations(self, entry: dict) -> list[dict[str, Any]]:
        """Extract alternative products / isoform annotations."""
        isoforms = []
        for comment in entry.get("comments", []):
            if comment.get("commentType") == "ALTERNATIVE PRODUCTS":
                for iso in comment.get("isoforms", []):
                    isoforms.append({
                        "id": iso.get("isoformIds", [""])[0] if iso.get("isoformIds") else "",
                        "name": iso.get("name", {}).get("value", ""),
                        "sequence_status": iso.get("isoformSequenceStatus", ""),
                        "note": iso.get("note", {}).get("texts", [{}])[0].get("value", "")
                        if iso.get("note") else "",
                    })
        return isoforms

    # ==================================================================
    # 10. Review status & metadata
    # ==================================================================

    def get_entry_metadata(self, entry: dict) -> dict[str, Any]:
        """Extract review status, organism, gene names, protein names."""
        protein_desc = entry.get("proteinDescription", {})
        rec_name = protein_desc.get("recommendedName", {})
        full_name = rec_name.get("fullName", {}).get("value", "")

        gene_names = []
        for gene in entry.get("genes", []):
            gene_names.append({
                "name": gene.get("geneName", {}).get("value", ""),
                "synonyms": [s.get("value", "") for s in gene.get("synonyms", [])],
            })

        organism = entry.get("organism", {})

        return {
            "accession": entry.get("primaryAccession", ""),
            "entry_type": entry.get("entryType", ""),
            "reviewed": entry.get("entryType", "") == "UniProtKB reviewed (Swiss-Prot)",
            "protein_name": full_name,
            "gene_names": gene_names,
            "organism": organism.get("scientificName", ""),
            "taxonomy_id": organism.get("taxonId"),
            "sequence_length": entry.get("sequence", {}).get("length"),
            "last_modified": entry.get("entryAudit", {}).get("lastAnnotationUpdateDate", ""),
        }

    # ==================================================================
    # Convenience: one-call aggregation
    # ==================================================================

    def annotate(
        self,
        identifier: str,
        id_type: str = "auto",
        sources: str | list[str] = "all",
    ) -> dict[str, Any]:
        """Aggregate all protein annotations in one call.

        Parameters
        ----------
        identifier
            UniProt accession, gene symbol, or Ensembl ID.
        id_type
            ``"uniprot_id"``, ``"symbol"``, ``"ensembl_id"``, or
            ``"auto"`` (detect).
        sources
            Which annotation categories to include. ``"all"`` includes
            everything. Pass a list to select: ``["function",
            "location", "sites", "domains", "ptms", "diseases",
            "go", "interactions", "isoforms", "metadata"]``.

        Returns
        -------
        Nested dict with keys per annotation category.
        """
        if sources == "all":
            sources_list = [
                "function", "location", "sites", "domains", "ptms",
                "diseases", "go", "interactions", "isoforms", "metadata",
            ]
        elif isinstance(sources, str):
            sources_list = [sources]
        else:
            sources_list = list(sources)

        accession = self._resolve_uniprot_accession(identifier, id_type)
        result: dict[str, Any] = {
            "identifier": identifier,
            "uniprot_accession": accession,
        }

        if accession is None:
            result["error"] = "Could not resolve to UniProt accession"
            return result

        entry = self._fetch_uniprot_entry(accession)
        if entry is None:
            result["error"] = f"Could not fetch UniProt entry for {accession}"
            return result

        if "metadata" in sources_list:
            result["metadata"] = self.get_entry_metadata(entry)

        if "function" in sources_list:
            result["function"] = self.get_function(entry)

        if "location" in sources_list:
            result["subcellular_location"] = self.get_subcellular_location(entry)

        if "sites" in sources_list:
            result["functional_sites"] = self.get_functional_sites(entry)

        if "domains" in sources_list:
            result["uniprot_domains"] = self.get_domains(entry)
            result["interpro_domains"] = self.get_interpro_domains(accession)

        if "ptms" in sources_list:
            result["ptms"] = self.get_ptms(entry)

        if "diseases" in sources_list:
            result["disease_associations"] = self.get_disease_associations(entry)
            result["variants"] = self.get_variants(entry)

        if "go" in sources_list:
            result["go_terms"] = self.get_go_terms(entry)

        if "interactions" in sources_list:
            result["interactions"] = self.get_interaction_xrefs(entry)

        if "isoforms" in sources_list:
            result["isoform_annotations"] = self.get_isoform_annotations(entry)

        return result

    def annotate_batch(
        self,
        identifiers: list[str],
        id_type: str = "auto",
        sources: str | list[str] = "all",
    ) -> dict[str, dict[str, Any]]:
        """Annotate a list of proteins."""
        results: dict[str, dict[str, Any]] = {}
        total = len(identifiers)
        for i, ident in enumerate(identifiers):
            results[ident] = self.annotate(ident, id_type=id_type, sources=sources)
            if (i + 1) % 10 == 0:
                logger.info("Annotated %d/%d proteins", i + 1, total)
        logger.info("Annotated %d proteins total", total)
        return results

    def annotate_adata(
        self,
        adata,  # anndata.AnnData
        column: str,
        id_type: str = "auto",
        sources: str | list[str] = "all",
        copy: bool = True,
    ):
        """Annotate protein perturbations in an AnnData.

        Reads identifiers from ``adata.obs[column]``, fetches
        annotations for each unique protein, and stores:

        - Summary columns as ``prot_*`` in ``adata.obs``
        - Full annotation dicts in ``adata.uns["protein_annotations"]``

        Parameters
        ----------
        adata
            AnnData with protein/gene identifiers in ``.obs[column]``.
        column
            Column in ``adata.obs``.
        id_type
            Identifier type (``"auto"``, ``"symbol"``, etc.).
        sources
            Annotation sources (see :meth:`annotate`).
        copy
            If ``True``, operate on a copy.

        Returns
        -------
        AnnData with protein annotations added.
        """
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
            "Annotating %d unique proteins from %d cells",
            len(unique_ids), len(identifiers),
        )

        annotations = self.annotate_batch(unique_ids, id_type=id_type, sources=sources)
        adata.uns["protein_annotations"] = annotations

        # Store summary columns
        reviewed_col = []
        n_domains_col = []
        n_ptms_col = []
        n_diseases_col = []
        n_interactions_col = []
        n_isoforms_col = []
        location_col = []
        protein_name_col = []

        for ident in identifiers:
            ann = annotations.get(ident, {})

            meta = ann.get("metadata", {})
            reviewed_col.append(meta.get("reviewed", False))
            protein_name_col.append(meta.get("protein_name", ""))

            domains = ann.get("uniprot_domains", [])
            n_domains_col.append(len(domains))

            ptms = ann.get("ptms", [])
            n_ptms_col.append(len(ptms))

            diseases = ann.get("disease_associations", [])
            n_diseases_col.append(len(diseases))

            interactions = ann.get("interactions", [])
            n_interactions_col.append(len(interactions))

            isoforms = ann.get("isoform_annotations", [])
            n_isoforms_col.append(len(isoforms))

            locs = ann.get("subcellular_location", [])
            location_col.append(
                locs[0]["location"] if locs else ""
            )

        adata.obs["prot_reviewed"] = reviewed_col
        adata.obs["prot_name"] = protein_name_col
        adata.obs["prot_n_domains"] = n_domains_col
        adata.obs["prot_n_ptms"] = n_ptms_col
        adata.obs["prot_n_diseases"] = n_diseases_col
        adata.obs["prot_n_interactions"] = n_interactions_col
        adata.obs["prot_n_isoforms"] = n_isoforms_col
        adata.obs["prot_location"] = location_col

        logger.info(
            "Protein annotations stored in adata.obs (prot_*) and adata.uns"
        )
        return adata
