"""Gene annotation from multiple public databases.

Aggregates metadata for a gene (given as symbol or Ensembl ID)
from four categories:

1. **Pathway annotations** -- KEGG and WikiPathways via MyGene.info;
   Reactome from the Reactome release files (:mod:`embpy.resources.gene.reactome`)
2. **Tissue & expression context** -- GTEx expression, HPA subcellular
   localization
3. **Interaction networks** -- STRING-DB PPI partners, DoRothEA regulons
4. **Disease & clinical relevance** -- Open Targets, GWAS Catalog

Does **not** reimplement pertpy functionality (GO term lookup,
basic gene annotation summaries).

MyGene.info notes (measured 2026-11-11, build 20261030):

- ``GET /v3/query`` ignores ``scopes``: ``q=CYP2D6&scopes=symbol`` is a full-text
  query whose top hit is the ``LOC110740340`` "CYP2D6 promoter" locus, and
  ``q=symbol:ATP5B`` returns nothing for a renamed gene. Gene lookups therefore
  use ``POST /v3/query`` with ``scopes=symbol`` (or ``ensembl.gene``), which
  honours the scope, resolves previous HGNC symbols (ATP5B -> ATP5F1B) and
  echoes each input in a ``query`` field. Each returned hit is then checked
  against the input (see :meth:`GeneAnnotator.query_genes_batch`).
- ``pathway.reactome`` in MyGene is **not** a Reactome source: only 2,753 human
  genes carry it (EGFR and TP53 have none). Reactome comes from Reactome's own
  release files through :class:`~embpy.resources.gene.reactome.ReactomeIndex`.
- ``pathway.kegg`` and ``pathway.wikipathways`` come from ConsensusPathDB
  (version 35 in this build), an older snapshot, not from KEGG or WikiPathways
  directly.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

import requests

if TYPE_CHECKING:
    from .reactome import ReactomeIndex

logger = logging.getLogger(__name__)

MYGENE = "https://mygene.info/v3"
#: Largest ``q`` list MyGene accepts in one ``POST /v3/query``.
MYGENE_MAX_BATCH = 1000
#: Default batch size: one failed POST only affects this many genes.
MYGENE_DEFAULT_CHUNK = 250
#: Fields requested by :meth:`GeneAnnotator.get_pathways_batch`.
MYGENE_PATHWAY_FIELDS = "symbol,alias,entrezgene,pathway.kegg,pathway.wikipathways"
#: Fields requested when only the identity (and Ensembl ID) is needed.
MYGENE_IDENTITY_FIELDS = "symbol,alias,entrezgene,ensembl.gene"
#: MyGene pathway providers returned by :meth:`GeneAnnotator.get_pathways_batch`.
MYGENE_PATHWAY_SOURCES = ("kegg", "wikipathways")
#: MyGene accepts these common names for ``species``; anything else is sent as a taxid.
_MYGENE_SPECIES_NAMES = {"human", "mouse", "rat", "fruitfly", "nematode", "zebrafish", "thale-cress", "frog", "pig"}

# Request sizes for the two paginated sources. They are *caps*, not measurements:
# a gene with 300 STRING partners still reports DEFAULT_PPI_PARTNERS. The summary
# columns written by annotate_adata therefore carry companion `*_at_limit` flags
# so a saturated value is distinguishable from a real count.
DEFAULT_PPI_PARTNERS = 10
DEFAULT_DISEASE_ASSOCIATIONS = 20

STRING_API = "https://string-db.org/api"
OPEN_TARGETS = "https://api.platform.opentargets.org/api/v4/graphql"
GWAS_CATALOG = "https://www.ebi.ac.uk/gwas/rest/api"
GTEX_API = "https://gtexportal.org/api/v2"
HPA_API = "https://www.proteinatlas.org"

STRING_TAXON = {
    "human": 9606, "homo_sapiens": 9606,
    "mouse": 10090, "mus_musculus": 10090,
    "rat": 10116, "rattus_norvegicus": 10116,
    "zebrafish": 7955, "danio_rerio": 7955,
    "drosophila": 7227, "drosophila_melanogaster": 7227,
    "worm": 6239, "caenorhabditis_elegans": 6239,
    "yeast": 559292, "saccharomyces_cerevisiae": 559292,
    "chicken": 9031, "gallus_gallus": 9031,
    "pig": 9823, "sus_scrofa": 9823,
    "dog": 9615, "canis_lupus_familiaris": 9615,
}

_HUMAN_ONLY_SOURCES = {"gtex", "hpa", "gwas_catalog", "open_targets"}


def _is_ensembl_gene_id(s: str) -> bool:
    """Check if a string is an Ensembl gene ID (any species)."""
    import re
    return bool(re.match(r"^ENS[A-Z]*G\d{11}(\.\d+)?$", s, re.IGNORECASE))


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


def _post_json(
    url: str,
    json_data: Any = None,
    data: Any = None,
    timeout: int = 30,
) -> Any:
    """POST a JSON body (``json_data``) or form fields (``data``) and parse the reply.

    Returns the decoded JSON (a dict or, for MyGene batch queries, a list), or
    ``None`` on any failure. ``None`` means the request failed, never that the
    answer was empty; callers report it as a failed batch.
    """
    try:
        if data is not None:
            resp = requests.post(url, data=data, timeout=timeout)
        else:
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


def _as_list(value: Any) -> list:
    """MyGene returns one item as a scalar/dict and several as a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _hit_entrez(hit: dict) -> str | None:
    """Numeric NCBI Gene ID of a MyGene hit, or ``None`` for Ensembl-only hits."""
    for value in (hit.get("entrezgene"), hit.get("_id")):
        if isinstance(value, bool) or value is None:
            continue
        text = str(value).strip()
        if text.isdigit():
            return text
    return None


def _hit_ensembl_genes(hit: dict) -> list[str]:
    genes = []
    for entry in _as_list(hit.get("ensembl")):
        if isinstance(entry, dict) and entry.get("gene"):
            genes.append(str(entry["gene"]))
    return genes


def _hit_summary(hit: dict) -> dict[str, Any]:
    return {
        "mygene_id": None if hit.get("_id") is None else str(hit.get("_id")),
        "symbol": hit.get("symbol"),
        "entrezgene": _hit_entrez(hit),
        "alias": [str(a) for a in _as_list(hit.get("alias"))],
    }


def _pathway_entries(value: Any) -> list[dict[str, str]]:
    """Normalise a MyGene ``pathway.<source>`` value (dict or list) to ``[{id, name}]``."""
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in _as_list(value):
        if not isinstance(entry, dict):
            continue
        item = (str(entry.get("id", "")), str(entry.get("name", "")))
        if item in seen:
            continue
        seen.add(item)
        entries.append({"id": item[0], "name": item[1]})
    return entries


def _unique_hits(hits: list[dict]) -> list[dict]:
    out, seen = [], set()
    for hit in hits:
        key = hit.get("_id")
        if key is not None and key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out


def _prefer_entrez(hits: list[dict]) -> dict | None:
    """One hit, or the only one with a numeric Entrez ID; ``None`` if still ambiguous."""
    hits = _unique_hits(hits)
    if len(hits) == 1:
        return hits[0]
    numeric = [h for h in hits if _hit_entrez(h) is not None]
    if len(numeric) == 1:
        return numeric[0]
    return None


def _string_score_1000(value: Any) -> int:
    """Parse STRING scores returned as either 0-1000 ints or 0-1 floats."""
    if value is None or value == "":
        return 0
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0
    if 0 <= score <= 1:
        score *= 1000
    return int(round(score))


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
        self._string_species = STRING_TAXON.get(organism.lower(), 9606)
        self._is_human = organism.lower() in ("human", "homo_sapiens")
        # Identity results of successful lookups, keyed by the stripped input.
        self._identity_cache: dict[str, dict[str, Any]] = {}

    def _sleep(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    @property
    def _mygene_species(self) -> str:
        org = self.organism.lower()
        if self._is_human:
            return "human"
        if org in _MYGENE_SPECIES_NAMES:
            return org
        if org in STRING_TAXON:
            return str(STRING_TAXON[org])
        return self.organism

    # ==================================================================
    # Identifier resolution (MyGene.info batch POST)
    # ==================================================================

    def query_genes_batch(
        self,
        genes: list[str],
        *,
        fields: str = MYGENE_IDENTITY_FIELDS,
        chunk_size: int = MYGENE_DEFAULT_CHUNK,
    ) -> dict[str, dict[str, Any]]:
        """Look genes up in MyGene.info and check that each hit is the gene asked for.

        Symbols are sent with ``scopes=symbol`` and Ensembl gene IDs (version
        stripped) with ``scopes=ensembl.gene``, as one ``POST /v3/query`` per
        chunk of at most ``chunk_size`` inputs (``q`` is a comma-separated
        list). The rows of the reply are grouped by their ``query`` echo and,
        per input:

        1. Hits whose ``symbol`` equals the query (case-insensitive) are the
           match. If there are several, the only one with a numeric Entrez ID
           wins (GNRHR2 has an Entrez gene and an Ensembl-only document);
           otherwise the input is ``ambiguous``.
        2. Otherwise hits whose ``alias`` (a string or a list) contains the
           query are a rename (ATP5B -> ATP5F1B); one such hit, or the only one
           with a numeric Entrez ID, is accepted with ``renamed_from`` set.
        3. Otherwise the hits are some other gene: ``identity_mismatch``.

        For Ensembl inputs the match is a hit whose ``ensembl.gene`` equals the
        ID. Rows flagged ``notfound`` give ``not_found``. If the POST fails or
        the reply is not a list, every input of that chunk is ``batch_failed``;
        an input missing from an otherwise valid reply is ``batch_failed`` too.

        Exceptions raised by ``_post_json`` (for example by a caller's
        transport that replaces it) propagate unchanged.

        Returns
        -------
        ``{gene: {"status", "query", "scope", "hit", "match", "renamed_from",
        "candidates"}}`` keyed by the inputs as given (duplicates collapse).
        ``status`` is ``found``, ``not_found``, ``ambiguous``,
        ``identity_mismatch`` or ``batch_failed``; ``hit`` is the raw MyGene
        row of the accepted hit (``None`` unless found); ``match`` is
        ``"symbol"``, ``"alias"`` or ``"ensembl"``; ``candidates`` summarises
        the hits of an ambiguous or mismatched input. A ``reason`` key explains
        ``batch_failed`` and invalid inputs.
        """
        genes = list(genes)
        chunk_size = int(chunk_size)
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        chunk_size = min(chunk_size, MYGENE_MAX_BATCH)

        results: dict[str, dict[str, Any]] = {}
        term_of: dict[str, tuple[str, str]] = {}
        terms: dict[str, list[str]] = {"symbol": [], "ensembl.gene": []}
        queued: set[tuple[str, str]] = set()
        for gene in genes:
            if gene in results or gene in term_of:
                continue
            text = "" if gene is None else str(gene).strip()
            if not text or "," in text:
                # MyGene splits q on commas, so such an input cannot be asked for.
                results[gene] = self._identity_record(text, None, "not_found", reason="invalid_query")
                continue
            if _is_ensembl_gene_id(text):
                scope, term = "ensembl.gene", text.split(".")[0]
            else:
                scope, term = "symbol", text
            term_of[gene] = (scope, term)
            if (scope, term) not in queued:
                queued.add((scope, term))
                terms[scope].append(term)

        resolved: dict[tuple[str, str], dict[str, Any]] = {}
        first = True
        for scope, scope_terms in terms.items():
            scope_fields = fields
            if scope == "ensembl.gene" and "ensembl.gene" not in fields.split(","):
                scope_fields = f"{fields},ensembl.gene"
            for start in range(0, len(scope_terms), chunk_size):
                chunk = scope_terms[start:start + chunk_size]
                if not first:
                    self._sleep()
                first = False
                payload = _post_json(
                    f"{MYGENE}/query",
                    data={
                        "q": ",".join(chunk),
                        "scopes": scope,
                        "species": self._mygene_species,
                        "fields": scope_fields,
                    },
                )
                if not isinstance(payload, list):
                    reason = "request_failed" if payload is None else "unexpected_payload"
                    logger.warning(
                        "MyGene batch of %d %s queries failed (%s)", len(chunk), scope, reason,
                    )
                    for term in chunk:
                        resolved[(scope, term)] = self._identity_record(term, scope, "batch_failed", reason=reason)
                    continue
                rows_by_query: dict[str, list[dict]] = defaultdict(list)
                rows_by_folded: dict[str, list[dict]] = defaultdict(list)
                for row in payload:
                    if isinstance(row, dict) and row.get("query") is not None:
                        rows_by_query[str(row["query"])].append(row)
                        rows_by_folded[str(row["query"]).casefold()].append(row)
                for term in chunk:
                    rows = rows_by_query.get(term) or rows_by_folded.get(term.casefold())
                    resolved[(scope, term)] = self._select_hit(term, scope, rows)

        for gene, key in term_of.items():
            results[gene] = resolved[key]
        return {gene: results[gene] for gene in dict.fromkeys(genes) if gene in results}

    @staticmethod
    def _identity_record(
        term: str,
        scope: str | None,
        status: str,
        *,
        hit: dict | None = None,
        match: str | None = None,
        renamed_from: str | None = None,
        candidates: list[dict] | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "status": status,
            "query": term,
            "scope": scope,
            "hit": hit,
            "match": match,
            "renamed_from": renamed_from,
            "candidates": [_hit_summary(h) for h in _unique_hits(candidates or [])],
        }
        if reason is not None:
            record["reason"] = reason
        return record

    def _select_hit(self, term: str, scope: str, rows: list[dict] | None) -> dict[str, Any]:
        """Apply the identity rules of :meth:`query_genes_batch` to one query's rows."""
        if not rows:
            return self._identity_record(term, scope, "batch_failed", reason="missing_from_response")
        hits = [r for r in rows if not r.get("notfound")]
        if not hits:
            return self._identity_record(term, scope, "not_found")
        folded = term.casefold()

        if scope == "ensembl.gene":
            exact = [h for h in hits if folded in {g.casefold() for g in _hit_ensembl_genes(h)}]
            if exact:
                pick = _prefer_entrez(exact)
                if pick is not None:
                    return self._identity_record(term, scope, "found", hit=pick, match="ensembl")
                return self._identity_record(term, scope, "ambiguous", candidates=hits)
            return self._identity_record(term, scope, "identity_mismatch", candidates=hits)

        exact = [h for h in hits if str(h.get("symbol") or "").casefold() == folded]
        if exact:
            pick = _prefer_entrez(exact)
            if pick is not None:
                return self._identity_record(term, scope, "found", hit=pick, match="symbol")
            return self._identity_record(term, scope, "ambiguous", candidates=hits)

        renamed = [
            h for h in hits
            if folded in {str(a).casefold() for a in _as_list(h.get("alias"))}
        ]
        if renamed:
            pick = _prefer_entrez(renamed)
            if pick is not None:
                return self._identity_record(
                    term, scope, "found", hit=pick, match="alias", renamed_from=term,
                )
            return self._identity_record(term, scope, "ambiguous", candidates=hits)

        return self._identity_record(term, scope, "identity_mismatch", candidates=hits)

    def _resolve_identity(self, gene: str) -> dict[str, Any]:
        """Identity record for one gene, cached per instance unless the request failed."""
        key = str(gene).strip()
        cached = self._identity_cache.get(key)
        if cached is not None:
            return cached
        record = self.query_genes_batch([gene], fields=MYGENE_IDENTITY_FIELDS)[gene]
        if record["status"] != "batch_failed":
            self._identity_cache[key] = record
        return record

    def _prime_identity_cache(self, genes: list[str]) -> None:
        """Resolve many genes in batches so later per-gene lookups hit the cache."""
        todo = [
            g for g in dict.fromkeys(genes)
            if g is not None and str(g).strip() not in self._identity_cache
        ]
        if not todo:
            return
        for gene, record in self.query_genes_batch(todo, fields=MYGENE_IDENTITY_FIELDS).items():
            if record["status"] != "batch_failed":
                self._identity_cache[str(gene).strip()] = record

    def _resolve_ensembl_id(self, gene: str) -> str | None:
        """Resolve a gene symbol to Ensembl gene ID via MyGene.info."""
        if _is_ensembl_gene_id(gene):
            return gene.split(".")[0]
        record = self._resolve_identity(gene)
        if record["status"] != "found":
            return None
        ensembl = _hit_ensembl_genes(record["hit"])
        return ensembl[0] if ensembl else None

    def _resolve_symbol(self, gene: str) -> str | None:
        """Resolve an Ensembl ID to gene symbol via MyGene.info."""
        if not _is_ensembl_gene_id(gene):
            return gene
        record = self._resolve_identity(gene)
        if record["status"] != "found":
            return None
        return record["hit"].get("symbol")

    # ==================================================================
    # 1. Pathway Annotations
    # ==================================================================

    def get_pathways_batch(
        self,
        genes: list[str],
        chunk_size: int = MYGENE_DEFAULT_CHUNK,
    ) -> dict[str, dict[str, Any]]:
        """KEGG and WikiPathways memberships for many genes from MyGene.info.

        One ``POST /v3/query`` per chunk (``scopes=symbol``, ``species``, and
        ``fields=symbol,alias,entrezgene,pathway.kegg,pathway.wikipathways``),
        with the identity rules of :meth:`query_genes_batch`. MyGene's
        ``pathway.reactome`` is deliberately not requested: it covers only a
        few thousand human genes. Use :meth:`get_reactome_pathways_batch` with a
        :class:`~embpy.resources.gene.reactome.ReactomeIndex` for Reactome.

        Parameters
        ----------
        genes
            Gene symbols and/or Ensembl gene IDs.
        chunk_size
            Inputs per POST (default 250, at most 1000). A failed POST marks
            exactly the genes of its chunk ``batch_failed``.

        Returns
        -------
        ``{gene: {"status", "found", "query", "symbol", "entrezgene",
        "mygene_id", "renamed_from", "match", "kegg", "wikipathways",
        "candidates"}}``. ``status`` is ``found``, ``not_found``,
        ``ambiguous``, ``identity_mismatch`` or ``batch_failed``; ``found`` is
        ``True`` only for ``found``, so a found gene with no pathways (empty
        lists) is distinguishable from a gene that was not found or not
        fetched. ``kegg`` and ``wikipathways`` are ``[{id, name}]`` lists
        (empty unless found). ``entrezgene`` is the NCBI Gene ID as a string,
        or ``None`` for an Ensembl-only hit.
        """
        identities = self.query_genes_batch(genes, fields=MYGENE_PATHWAY_FIELDS, chunk_size=chunk_size)
        out: dict[str, dict[str, Any]] = {}
        for gene, record in identities.items():
            hit = record["hit"] if record["status"] == "found" else None
            pathway = (hit or {}).get("pathway")
            if not isinstance(pathway, dict):
                pathway = {}
            entry: dict[str, Any] = {
                "status": record["status"],
                "found": hit is not None,
                "query": record["query"],
                "symbol": hit.get("symbol") if hit else None,
                "entrezgene": _hit_entrez(hit) if hit else None,
                "mygene_id": None if not hit or hit.get("_id") is None else str(hit["_id"]),
                "renamed_from": record["renamed_from"],
                "match": record["match"],
                "candidates": record["candidates"],
            }
            for source in MYGENE_PATHWAY_SOURCES:
                entry[source] = _pathway_entries(pathway.get(source)) if hit else []
            if "reason" in record:
                entry["reason"] = record["reason"]
            out[gene] = entry
        return out

    def get_reactome_pathways_batch(
        self,
        genes: list[str],
        *,
        reactome_index: ReactomeIndex,
        gene_records: dict[str, dict[str, Any]] | None = None,
        chunk_size: int = MYGENE_DEFAULT_CHUNK,
    ) -> dict[str, dict[str, Any]]:
        """Reactome pathways for many genes, joined on the NCBI Gene ID.

        Genes are resolved with :meth:`get_pathways_batch` (or taken from
        ``gene_records``, a previous result of it) and their ``entrezgene`` is
        looked up in ``reactome_index``. Only identity-checked hits with a
        numeric Entrez ID are joined.

        Returns
        -------
        ``{gene: {"status", "symbol", "entrezgene", "pathways", "top_level",
        "release"}}``. ``status`` is the index's ``success`` or
        ``not_in_reactome`` for a joined gene, ``no_entrez_id`` for a found gene
        without a numeric Entrez ID, or the MyGene status (``not_found``,
        ``ambiguous``, ``identity_mismatch``, ``batch_failed``) otherwise.
        """
        if gene_records is None:
            gene_records = self.get_pathways_batch(genes, chunk_size=chunk_size)
        out: dict[str, dict[str, Any]] = {}
        for gene in dict.fromkeys(genes):
            record = gene_records.get(gene)
            base = {
                "symbol": None,
                "entrezgene": None,
                "pathways": [],
                "top_level": [],
                "release": reactome_index.release,
            }
            if record is None:
                out[gene] = {"status": "batch_failed", "reason": "missing_gene_record", **base}
                continue
            base["symbol"] = record.get("symbol")
            base["entrezgene"] = record.get("entrezgene")
            if record.get("status") != "found":
                out[gene] = {"status": record.get("status", "batch_failed"), **base}
            elif not record.get("entrezgene"):
                out[gene] = {"status": "no_entrez_id", **base}
            else:
                hit = reactome_index.lookup(record["entrezgene"])
                out[gene] = {
                    **base,
                    "status": hit["status"],
                    "pathways": hit["pathways"],
                    "top_level": hit["top_level"],
                }
        return out

    @staticmethod
    def _pathways_from_record(
        gene: str,
        record: dict[str, Any],
        reactome_index: ReactomeIndex | None,
    ) -> dict[str, list[dict[str, str]]]:
        """The legacy ``{"reactome", "kegg", "wikipathways"}`` view of a batch record."""
        if record["status"] == "batch_failed":
            logger.warning(
                "MyGene lookup for %s failed (%s); returning empty pathways. "
                "Use get_pathways_batch() to see the status.",
                gene, record.get("reason", "batch_failed"),
            )
        elif record["status"] != "found":
            logger.debug("MyGene lookup for %s: %s", gene, record["status"])
        reactome: list[dict[str, str]] = []
        if reactome_index is not None and record["status"] == "found" and record.get("entrezgene"):
            reactome = reactome_index.lookup(record["entrezgene"])["pathways"]
        return {
            "reactome": reactome,
            "kegg": list(record.get("kegg", [])),
            "wikipathways": list(record.get("wikipathways", [])),
        }

    def get_pathways(
        self,
        gene: str,
        *,
        reactome_index: ReactomeIndex | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        """Get Reactome, KEGG and WikiPathways annotations for one gene.

        KEGG and WikiPathways come from MyGene.info through
        :meth:`get_pathways_batch` (a one-gene batch, so renamed symbols
        resolve and the hit is identity-checked). Reactome comes only from
        ``reactome_index`` (a :class:`~embpy.resources.gene.reactome.ReactomeIndex`);
        without one the ``"reactome"`` list is empty. MyGene's own
        ``pathway.reactome`` is not used.

        Parameters
        ----------
        gene
            Gene symbol or Ensembl ID.
        reactome_index
            Reactome release index used for the ``"reactome"`` list.

        Returns
        -------
        Dict with keys ``"reactome"``, ``"kegg"``, ``"wikipathways"``,
        each containing a list of ``{id, name}`` dicts. A gene that was not
        found, was ambiguous, or whose request failed gives empty lists; call
        :meth:`get_pathways_batch` to tell those cases apart.
        """
        record = self.get_pathways_batch([gene])[gene]
        return self._pathways_from_record(gene, record, reactome_index)


    # ==================================================================
    # 2. Tissue & Expression Context
    # ==================================================================

    def _resolve_gtex_gencode_id(self, gene: str) -> str | None:
        """Resolve a gene to its versioned GENCODE ID used by GTEx.

        The GTEx expression API requires a versioned GENCODE ID
        (e.g. ``ENSG00000165704.14``) from GENCODE v26.  We query the
        GTEx reference endpoint which maps symbols / Ensembl IDs to the
        correct versioned ID.
        """
        symbol = self._resolve_symbol(gene)
        query = symbol or gene
        data = _get_json(
            f"{GTEX_API}/reference/gene",
            params={"geneId": query},
        )
        if data and data.get("data"):
            return data["data"][0].get("gencodeId")
        return None

    def get_tissue_expression(self, gene: str) -> list[dict[str, Any]]:
        """Get tissue expression profile from GTEx (human only).

        Parameters
        ----------
        gene
            Gene symbol or Ensembl ID.

        Returns
        -------
        List of dicts with ``tissue``, ``median_tpm``, ``n_samples``.
        """
        if not self._is_human:
            logger.debug("GTEx is human-only; skipping for organism '%s'", self.organism)
            return []

        gencode_id = self._resolve_gtex_gencode_id(gene)
        if not gencode_id:
            logger.debug("Could not resolve %s to GTEx GENCODE ID", gene)
            return []

        self._sleep()
        data = _get_json(
            f"{GTEX_API}/expression/medianGeneExpression",
            params={
                "gencodeId": gencode_id,
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
        """Get subcellular localization from Human Protein Atlas (human only).

        Parameters
        ----------
        gene
            Gene symbol or Ensembl ID.

        Returns
        -------
        Dict with ``locations`` (list of ``{location, is_main}`` dicts),
        ``reliability``, and ``source``.
        """
        if not self._is_human:
            logger.debug("HPA is human-only; skipping for organism '%s'", self.organism)
            return {}

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

        main_locs = set(data.get("Subcellular main location", []))
        all_locs = data.get("Subcellular location", [])
        reliability = data.get("Reliability (IF)", "")

        if not all_locs:
            return {"locations": [], "reliability": reliability, "source": "HPA"}

        locations = []
        for entry in all_locs if isinstance(all_locs, list) else [all_locs]:
            if isinstance(entry, str):
                locations.append({
                    "location": entry,
                    "is_main": entry in main_locs,
                })
            elif isinstance(entry, dict):
                locations.append({
                    "location": entry.get("location", str(entry)),
                    "is_main": entry.get("location", "") in main_locs,
                })

        return {"locations": locations, "reliability": reliability, "source": "HPA"}

    # ==================================================================
    # 3. Interaction Networks
    # ==================================================================

    def get_protein_interactions(
        self,
        gene: str,
        n_partners: int = DEFAULT_PPI_PARTNERS,
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
                "combined_score": _string_score_1000(row.get("score")),
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
        top_n: int = DEFAULT_DISEASE_ASSOCIATIONS,
    ) -> list[dict[str, Any]]:
        """Get disease associations from Open Targets Platform (human only).

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
        if not self._is_human:
            logger.debug("Open Targets is human-only; skipping for organism '%s'", self.organism)
            return []

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
        if not self._is_human:
            logger.debug("GWAS Catalog is human-only; skipping for organism '%s'", self.organism)
            return []

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

    @staticmethod
    def _sources_list(sources: str | list[str]) -> list[str]:
        if sources == "all":
            return ["pathways", "expression", "interactions", "diseases"]
        if isinstance(sources, str):
            return [sources]
        return list(sources)

    def annotate(
        self,
        gene: str,
        id_type: Literal["symbol", "ensembl_id", "auto"] = "auto",
        sources: str | list[str] = "all",
        *,
        reactome_index: ReactomeIndex | None = None,
        _pathway_record: dict[str, Any] | None = None,
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
        reactome_index
            Reactome release index for the ``pathways["reactome"]`` list
            (see :meth:`get_pathways`).

        Returns
        -------
        Nested dict with keys per annotation category.
        """
        sources_list = self._sources_list(sources)

        result: dict[str, Any] = {
            "gene": gene,
            "ensembl_id": self._resolve_ensembl_id(gene),
            "symbol": self._resolve_symbol(gene),
        }

        if "pathways" in sources_list:
            if _pathway_record is not None:
                result["pathways"] = self._pathways_from_record(gene, _pathway_record, reactome_index)
            else:
                result["pathways"] = self.get_pathways(gene, reactome_index=reactome_index)

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
        *,
        reactome_index: ReactomeIndex | None = None,
        chunk_size: int = MYGENE_DEFAULT_CHUNK,
    ) -> dict[str, dict[str, Any]]:
        """Annotate a list of genes.

        Gene identities and pathways are fetched from MyGene.info in batches
        (one POST per ``chunk_size`` genes) instead of once per gene.

        Returns
        -------
        Dict mapping gene identifier to annotation dict.
        """
        genes = list(genes)
        sources_list = self._sources_list(sources)
        self._prime_identity_cache(genes)
        pathway_records = (
            self.get_pathways_batch(genes, chunk_size=chunk_size)
            if "pathways" in sources_list else {}
        )
        results: dict[str, dict[str, Any]] = {}
        total = len(genes)
        for i, gene in enumerate(genes):
            results[gene] = self.annotate(
                gene,
                sources=sources_list,
                reactome_index=reactome_index,
                _pathway_record=pathway_records.get(gene),
            )
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
            if tissues:
                top = tissues[0]
                top_tissue_col.append(
                    top.get("tissue_name") or top.get("tissue", "")
                )
            else:
                top_tissue_col.append("")

        adata.obs["gene_n_pathways"] = n_pathways_col
        adata.obs["gene_n_ppi_partners"] = n_ppi_col
        adata.obs["gene_n_disease_assoc"] = n_diseases_col
        adata.obs["gene_n_transcription_factors"] = n_tfs_col
        adata.obs["gene_top_tissue"] = top_tissue_col

        # `gene_n_ppi_partners` and `gene_n_disease_assoc` count what was *fetched*,
        # and the fetch is capped, so for a well-studied gene they report the cap
        # rather than the truth. Flag the saturated rows instead of letting a
        # constant column look like a measurement -- regressing on one of these
        # without the flag is a silent mistake.
        adata.obs["gene_n_ppi_partners_at_limit"] = [
            n >= DEFAULT_PPI_PARTNERS for n in n_ppi_col
        ]
        adata.obs["gene_n_disease_assoc_at_limit"] = [
            n >= DEFAULT_DISEASE_ASSOCIATIONS for n in n_diseases_col
        ]
        adata.uns["gene_annotation_limits"] = {
            "gene_n_ppi_partners": DEFAULT_PPI_PARTNERS,
            "gene_n_disease_assoc": DEFAULT_DISEASE_ASSOCIATIONS,
        }

        logger.info(
            "Gene annotations stored in adata.obs (gene_*) and adata.uns"
        )
        return adata
