"""Gene annotation from multiple public databases.

Aggregates metadata for a gene (given as symbol or Ensembl ID)
from four categories:

1. **Pathway annotations** -- Reactome, KEGG, WikiPathways (via MyGene.info)
2. **Tissue & expression context** -- GTEx expression, HPA subcellular
   localization
3. **Interaction networks** -- STRING-DB PPI partners, DoRothEA regulons
4. **Disease & clinical relevance** -- Open Targets, GWAS Catalog

Does **not** reimplement pertpy functionality (GO term lookup,
basic gene annotation summaries).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

import requests

logger = logging.getLogger(__name__)

MYGENE = "https://mygene.info/v3"
STRING_API = "https://string-db.org/api"
OPEN_TARGETS = "https://api.platform.opentargets.org/api/v4/graphql"
GWAS_CATALOG = "https://www.ebi.ac.uk/gwas/rest/api"
GTEX_API = "https://gtexportal.org/api/v2"
HPA_API = "https://www.proteinatlas.org"


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


def _post_json(url: str, json_data: dict, timeout: int = 30) -> dict | None:
    try:
        resp = requests.post(url, json=json_data, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:  # noqa: BLE001
        logger.debug("POST failed for %s: %s", url, e)
        return None


def _get_text(url: str, params: dict | None = None, timeout: int = 30) -> str | None:
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.text
    except Exception as e:  # noqa: BLE001
        logger.debug("Request failed for %s: %s", url, e)
        return None


class GeneAnnotator:
    """Aggregate annotations for genes from public databases.

    Parameters
    ----------
    organism : str
        Default organism (default ``"human"``).
    rate_limit_delay : float
        Seconds to wait between API calls (default 0.2).
    """

    def __init__(
        self,
        organism: str = "human",
        rate_limit_delay: float = 0.2,
    ) -> None:
        self.organism = organism
        self.delay = rate_limit_delay
        self._string_species = 9606 if organism.lower() in ("human", "homo_sapiens") else 9606

    def _sleep(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    # ==================================================================
    # Identifier resolution
    # ==================================================================

    def _resolve_ensembl_id(self, gene: str) -> str | None:
        """Resolve a gene symbol to Ensembl gene ID via MyGene.info."""
        if gene.startswith("ENSG"):
            return gene.split(".")[0]
        data = _get_json(
            f"{MYGENE}/query",
            params={
                "q": gene,
                "scopes": "symbol,alias",
                "species": self.organism,
                "fields": "ensembl.gene",
                "size": 1,
            },
        )
        if data:
            hits = data.get("hits", [])
            if hits:
                ens = hits[0].get("ensembl", {})
                if isinstance(ens, dict):
                    return ens.get("gene")
                if isinstance(ens, list) and ens:
                    return ens[0].get("gene")
        return None

    def _resolve_symbol(self, gene: str) -> str | None:
        """Resolve an Ensembl ID to gene symbol via MyGene.info."""
        if not gene.startswith("ENSG"):
            return gene
        data = _get_json(
            f"{MYGENE}/query",
            params={
                "q": gene,
                "scopes": "ensembl.gene",
                "species": self.organism,
                "fields": "symbol",
                "size": 1,
            },
        )
        if data:
            hits = data.get("hits", [])
            if hits:
                return hits[0].get("symbol")
        return None

    # ==================================================================
    # 1. Pathway Annotations (MyGene.info)
    # ==================================================================

    def get_pathways(self, gene: str) -> dict[str, list[dict[str, str]]]:
        """Get pathway annotations from Reactome, KEGG, and WikiPathways.

        Uses MyGene.info which aggregates all three sources.

        Parameters
        ----------
        gene
            Gene symbol or Ensembl ID.

        Returns
        -------
        Dict with keys ``"reactome"``, ``"kegg"``, ``"wikipathways"``,
        each containing a list of ``{id, name}`` dicts.
        """
        scope = "ensembl.gene" if gene.startswith("ENSG") else "symbol"
        data = _get_json(
            f"{MYGENE}/query",
            params={
                "q": gene,
                "scopes": scope,
                "species": self.organism,
                "fields": "pathway.reactome,pathway.kegg,pathway.wikipathways",
                "size": 1,
            },
        )
        result: dict[str, list[dict[str, str]]] = {
            "reactome": [],
            "kegg": [],
            "wikipathways": [],
        }
        if not data or not data.get("hits"):
            return result

        pathways = data["hits"][0].get("pathway", {})

        for source in ("reactome", "kegg", "wikipathways"):
            entries = pathways.get(source, [])
            if isinstance(entries, dict):
                entries = [entries]
            for entry in entries:
                result[source].append({
                    "id": str(entry.get("id", "")),
                    "name": str(entry.get("name", "")),
                })
        return result

    # ==================================================================
    # 2. Tissue & Expression Context
    # ==================================================================

    def get_tissue_expression(self, gene: str) -> list[dict[str, Any]]:
        """Get tissue expression profile from GTEx.

        Parameters
        ----------
        gene
            Gene symbol or Ensembl ID.

        Returns
        -------
        List of dicts with ``tissue``, ``median_tpm``, ``n_samples``.
        """
        ensembl_id = self._resolve_ensembl_id(gene)
        if not ensembl_id:
            logger.debug("Could not resolve %s to Ensembl ID for GTEx", gene)
            return []

        self._sleep()
        data = _get_json(
            f"{GTEX_API}/expression/medianGeneExpression",
            params={
                "gencodeId": ensembl_id,
                "datasetId": "gtex_v8",
            },
        )
        if not data or "data" not in data:
            return []

        tissues = []
        for entry in data["data"]:
            tissues.append({
                "tissue": entry.get("tissueSiteDetailId", ""),
                "tissue_name": entry.get("tissueSiteDetail", ""),
                "median_tpm": entry.get("median", 0),
                "n_samples": entry.get("numSamples", 0),
            })
        tissues.sort(key=lambda x: x["median_tpm"], reverse=True)
        return tissues

    def get_subcellular_localization(self, gene: str) -> dict[str, Any]:
        """Get subcellular localization from Human Protein Atlas.

        Parameters
        ----------
        gene
            Gene symbol or Ensembl ID.

        Returns
        -------
        Dict with ``locations``, ``reliability``, and ``cell_line``.
        """
        ensembl_id = self._resolve_ensembl_id(gene)
        if not ensembl_id:
            return {}

        self._sleep()
        data = _get_json(
            f"{HPA_API}/{ensembl_id}.json",
        )
        if not data:
            return {}

        if isinstance(data, list) and data:
            data = data[0]

        subcell = data.get("Subcellular location", [])
        if not subcell:
            return {"locations": [], "source": "HPA"}

        locations = []
        for entry in subcell if isinstance(subcell, list) else [subcell]:
            loc = entry if isinstance(entry, dict) else {}
            locations.append({
                "location": loc.get("location", ""),
                "reliability": loc.get("reliability", ""),
                "enhanced": loc.get("enhanced", False),
                "supported": loc.get("supported", False),
            })
        return {"locations": locations, "source": "HPA"}

    # ==================================================================
    # 3. Interaction Networks
    # ==================================================================

    def get_protein_interactions(
        self,
        gene: str,
        n_partners: int = 10,
        score_threshold: int = 400,
    ) -> list[dict[str, Any]]:
        """Get top protein-protein interaction partners from STRING-DB.

        Parameters
        ----------
        gene
            Gene symbol or Ensembl ID.
        n_partners
            Maximum number of interaction partners to return.
        score_threshold
            Minimum combined score (0-1000).

        Returns
        -------
        List of dicts with ``partner``, ``combined_score``,
        ``experimental_score``.
        """
        symbol = self._resolve_symbol(gene)
        if not symbol:
            return []

        self._sleep()
        text = _get_text(
            f"{STRING_API}/tsv/interaction_partners",
            params={
                "identifiers": symbol,
                "species": self._string_species,
                "limit": n_partners,
                "required_score": score_threshold,
                "caller_identity": "embpy",
            },
        )
        if not text:
            return []

        lines = text.strip().split("\n")
        if len(lines) < 2:
            return []

        header = lines[0].split("\t")
        partners = []
        for line in lines[1:]:
            cols = line.split("\t")
            if len(cols) < len(header):
                continue
            row = dict(zip(header, cols))
            partners.append({
                "partner": row.get("preferredName_B", row.get("stringId_B", "")),
                "combined_score": int(row.get("score", 0)),
                "nscore": float(row.get("nscore", 0)),
                "fscore": float(row.get("fscore", 0)),
                "pscore": float(row.get("pscore", 0)),
                "escore": float(row.get("escore", 0)),
                "dscore": float(row.get("dscore", 0)),
                "tscore": float(row.get("tscore", 0)),
            })
        partners.sort(key=lambda x: x["combined_score"], reverse=True)
        return partners

    def get_transcription_factors(self, gene: str) -> list[dict[str, Any]]:
        """Get transcription factors regulating this gene via DoRothEA.

        Uses the ``decoupler`` package (lazy import). Falls back to
        an empty list if not installed.

        Parameters
        ----------
        gene
            Gene symbol.

        Returns
        -------
        List of dicts with ``tf``, ``target``, ``weight``,
        ``confidence``.
        """
        symbol = self._resolve_symbol(gene)
        if not symbol:
            return []

        try:
            import decoupler as dc  # type: ignore[import-untyped]
            dorothea = dc.get_dorothea(organism=self.organism)
            regulons = dorothea[dorothea["target"] == symbol]
            return [
                {
                    "tf": row["source"],
                    "target": row["target"],
                    "weight": float(row.get("weight", 0)),
                    "confidence": str(row.get("confidence", "")),
                }
                for _, row in regulons.iterrows()
            ]
        except ImportError:
            logger.debug(
                "decoupler not installed; skipping DoRothEA TF lookup. "
                "Install with: pip install decoupler"
            )
            return []
        except Exception as e:  # noqa: BLE001
            logger.debug("DoRothEA lookup failed for %s: %s", symbol, e)
            return []

    # ==================================================================
    # 4. Disease & Clinical Relevance
    # ==================================================================

    def get_disease_associations(
        self,
        gene: str,
        top_n: int = 20,
    ) -> list[dict[str, Any]]:
        """Get disease associations from Open Targets Platform.

        Parameters
        ----------
        gene
            Gene symbol or Ensembl ID.
        top_n
            Maximum number of associations to return.

        Returns
        -------
        List of dicts with ``disease_id``, ``disease_name``, ``score``,
        ``evidence_count``.
        """
        ensembl_id = self._resolve_ensembl_id(gene)
        if not ensembl_id:
            return []

        self._sleep()
        query = """
        query TargetDiseases($ensemblId: String!, $size: Int!) {
          target(ensemblId: $ensemblId) {
            associatedDiseases(page: {size: $size, index: 0}) {
              rows {
                disease {
                  id
                  name
                }
                score
                datatypeScores {
                  componentId: id
                  score
                }
              }
            }
          }
        }
        """
        result = _post_json(
            OPEN_TARGETS,
            json_data={
                "query": query,
                "variables": {"ensemblId": ensembl_id, "size": top_n},
            },
        )
        if not result:
            return []

        target = result.get("data", {}).get("target")
        if not target:
            return []

        rows = target.get("associatedDiseases", {}).get("rows", [])
        diseases = []
        for row in rows:
            disease = row.get("disease", {})
            diseases.append({
                "disease_id": disease.get("id", ""),
                "disease_name": disease.get("name", ""),
                "score": round(row.get("score", 0), 4),
            })
        return diseases

    def get_gwas_associations(
        self,
        gene: str,
        top_n: int = 20,
    ) -> list[dict[str, Any]]:
        """Get GWAS Catalog associations for a gene.

        Parameters
        ----------
        gene
            Gene symbol.
        top_n
            Maximum number of results.

        Returns
        -------
        List of dicts with ``trait``, ``p_value``, ``study``, ``snp``.
        """
        symbol = self._resolve_symbol(gene)
        if not symbol:
            return []

        self._sleep()
        data = _get_json(
            f"{GWAS_CATALOG}/associations/search/findByGeneName",
            params={"geneName": symbol, "size": top_n},
        )
        if not data:
            return []

        associations_data = (
            data.get("_embedded", {}).get("associations", [])
        )
        results = []
        for assoc in associations_data:
            trait_names = []
            for t in assoc.get("efoTraits", []):
                trait_names.append(t.get("trait", ""))

            snps = []
            for locus in assoc.get("loci", []):
                for sr in locus.get("strongestRiskAlleles", []):
                    snps.append(sr.get("riskAlleleName", ""))

            results.append({
                "traits": trait_names,
                "p_value": assoc.get("pvalue"),
                "snps": snps,
                "study": assoc.get("study", {}).get("publicationInfo", {}).get("title", "")
                if isinstance(assoc.get("study"), dict) else "",
            })
        return results

    # ==================================================================
    # Convenience: one-call aggregation
    # ==================================================================

    def annotate(
        self,
        gene: str,
        id_type: Literal["symbol", "ensembl_id", "auto"] = "auto",
        sources: str | list[str] = "all",
    ) -> dict[str, Any]:
        """Aggregate all annotations for a gene in one call.

        Parameters
        ----------
        gene
            Gene symbol or Ensembl ID.
        id_type
            ``"symbol"``, ``"ensembl_id"``, or ``"auto"`` (detect).
        sources
            Which annotation sources to query. ``"all"`` queries
            everything. Pass a list to select: ``["pathways",
            "expression", "interactions", "diseases"]``.

        Returns
        -------
        Nested dict with keys per annotation category.
        """
        if sources == "all":
            sources_list = ["pathways", "expression", "interactions", "diseases"]
        elif isinstance(sources, str):
            sources_list = [sources]
        else:
            sources_list = list(sources)

        result: dict[str, Any] = {
            "gene": gene,
            "ensembl_id": self._resolve_ensembl_id(gene),
            "symbol": self._resolve_symbol(gene),
        }

        if "pathways" in sources_list:
            result["pathways"] = self.get_pathways(gene)

        if "expression" in sources_list:
            result["tissue_expression"] = self.get_tissue_expression(gene)
            result["subcellular_localization"] = self.get_subcellular_localization(gene)

        if "interactions" in sources_list:
            result["ppi_partners"] = self.get_protein_interactions(gene)
            result["transcription_factors"] = self.get_transcription_factors(gene)

        if "diseases" in sources_list:
            result["disease_associations"] = self.get_disease_associations(gene)
            result["gwas_associations"] = self.get_gwas_associations(gene)

        return result

    def annotate_batch(
        self,
        genes: list[str],
        sources: str | list[str] = "all",
    ) -> dict[str, dict[str, Any]]:
        """Annotate a list of genes.

        Returns
        -------
        Dict mapping gene identifier to annotation dict.
        """
        results: dict[str, dict[str, Any]] = {}
        total = len(genes)
        for i, gene in enumerate(genes):
            results[gene] = self.annotate(gene, sources=sources)
            if (i + 1) % 10 == 0:
                logger.info("Annotated %d/%d genes", i + 1, total)
        logger.info("Annotated %d genes total", total)
        return results

    def annotate_adata(
        self,
        adata,  # anndata.AnnData
        column: str,
        sources: str | list[str] = "all",
        copy: bool = True,
    ):
        """Annotate gene perturbations in an AnnData.

        Reads gene identifiers from ``adata.obs[column]``, fetches
        annotations for each unique gene, and stores:

        - Summary counts as ``gene_*`` columns in ``adata.obs``
        - Full annotation dicts in ``adata.uns["gene_annotations"]``

        Parameters
        ----------
        adata
            AnnData with gene identifiers in ``.obs[column]``.
        column
            Column in ``adata.obs`` containing gene symbols or
            Ensembl IDs.
        sources
            Annotation sources to query (see :meth:`annotate`).
        copy
            If ``True``, operate on a copy.

        Returns
        -------
        AnnData with gene annotations added.
        """
        if copy:
            adata = adata.copy()

        if column not in adata.obs.columns:
            raise ValueError(
                f"Column '{column}' not found in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )

        identifiers = adata.obs[column].astype(str).values
        unique_genes = list(dict.fromkeys(identifiers))
        logger.info(
            "Annotating %d unique genes from %d cells",
            len(unique_genes), len(identifiers),
        )

        annotations = self.annotate_batch(unique_genes, sources=sources)
        adata.uns["gene_annotations"] = annotations

        # Store summary columns in .obs
        n_pathways_col = []
        n_ppi_col = []
        n_diseases_col = []
        n_tfs_col = []
        top_tissue_col = []

        for ident in identifiers:
            ann = annotations.get(ident, {})

            pw = ann.get("pathways", {})
            total_pw = sum(len(v) for v in pw.values()) if isinstance(pw, dict) else 0
            n_pathways_col.append(total_pw)

            ppi = ann.get("ppi_partners", [])
            n_ppi_col.append(len(ppi))

            diseases = ann.get("disease_associations", [])
            n_diseases_col.append(len(diseases))

            tfs = ann.get("transcription_factors", [])
            n_tfs_col.append(len(tfs))

            tissues = ann.get("tissue_expression", [])
            top_tissue_col.append(
                tissues[0]["tissue_name"] if tissues else ""
            )

        adata.obs["gene_n_pathways"] = n_pathways_col
        adata.obs["gene_n_ppi_partners"] = n_ppi_col
        adata.obs["gene_n_disease_assoc"] = n_diseases_col
        adata.obs["gene_n_transcription_factors"] = n_tfs_col
        adata.obs["gene_top_tissue"] = top_tissue_col

        logger.info(
            "Gene annotations stored in adata.obs (gene_*) and adata.uns"
        )
        return adata
