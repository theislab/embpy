"""Protein sequence resolver with UniProt canonical and isoform support.

Resolves gene identifiers (symbol, Ensembl ID, UniProt accession) to
protein sequences via the UniProt REST API.  Supports fetching canonical
sequences, individual isoforms, or all isoforms for a gene.

Example
-------
>>> resolver = ProteinResolver()
>>> seq = resolver.get_canonical_sequence("TP53", id_type="symbol")
>>> len(seq)
393
>>> isoforms = resolver.get_isoforms("TP53", id_type="symbol")
>>> list(isoforms.keys())
['P04637-1', 'P04637-2', ...]
"""

from __future__ import annotations

import logging
import time
from typing import Literal

import requests

logger = logging.getLogger(__name__)

UNIPROT_REST = "https://rest.uniprot.org"
MYGENE_REST = "https://mygene.info/v3"

ORGANISM_TAXON = {
    "human": 9606,
    "homo_sapiens": 9606,
    "mouse": 10090,
    "mus_musculus": 10090,
    "rat": 10116,
    "rattus_norvegicus": 10116,
    "zebrafish": 7955,
    "danio_rerio": 7955,
    "drosophila": 7227,
    "drosophila_melanogaster": 7227,
    "worm": 6239,
    "caenorhabditis_elegans": 6239,
    "yeast": 559292,
    "saccharomyces_cerevisiae": 559292,
    "chicken": 9031,
    "gallus_gallus": 9031,
    "pig": 9823,
    "sus_scrofa": 9823,
    "dog": 9615,
    "canis_lupus_familiaris": 9615,
    "macaque": 9544,
    "macaca_mulatta": 9544,
}


class ProteinResolver:
    """Resolve gene identifiers to protein sequences via UniProt.

    Supports canonical sequences, isoform enumeration, and bulk
    fetching for all protein-coding genes.

    Parameters
    ----------
    organism : str
        Default organism for lookups (e.g. ``"human"``).
    request_timeout : int
        HTTP timeout in seconds for API calls.
    rate_limit_delay : float
        Delay in seconds between consecutive API calls to respect
        rate limits.
    """

    def __init__(
        self,
        organism: str = "human",
        request_timeout: int = 30,
        rate_limit_delay: float = 0.1,
    ) -> None:
        self.organism = organism
        self.timeout = request_timeout
        self.rate_limit_delay = rate_limit_delay
        self._uniprot_cache: dict[str, str | None] = {}

    # ------------------------------------------------------------------
    # Identifier resolution
    # ------------------------------------------------------------------

    def resolve_uniprot_id(
        self,
        identifier: str,
        id_type: Literal["symbol", "ensembl_id", "uniprot_id"] = "symbol",
        organism: str | None = None,
    ) -> str | None:
        """Map a gene identifier to a reviewed UniProt (Swiss-Prot) accession.

        Parameters
        ----------
        identifier
            Gene symbol, Ensembl gene ID, or UniProt accession.
        id_type
            Type of identifier.
        organism
            Organism name.  Falls back to ``self.organism``.

        Returns
        -------
        UniProt accession string, or ``None`` if not found.
        """
        if id_type == "uniprot_id":
            return identifier.strip().split("-")[0]

        org = organism or self.organism
        cache_key = f"{id_type}:{identifier}:{org}"
        if cache_key in self._uniprot_cache:
            return self._uniprot_cache[cache_key]

        accession = self._resolve_via_mygene(identifier, id_type, org)
        if accession is None:
            accession = self._resolve_via_uniprot_search(identifier, id_type, org)

        self._uniprot_cache[cache_key] = accession
        return accession

    def _resolve_via_mygene(
        self,
        identifier: str,
        id_type: str,
        organism: str,
    ) -> str | None:
        scopes = {"symbol": "symbol", "ensembl_id": "ensembl.gene"}
        scope = scopes.get(id_type)
        if scope is None:
            return None
        try:
            resp = requests.get(
                f"{MYGENE_REST}/query",
                params={
                    "q": identifier,
                    "scopes": scope,
                    "species": organism,
                    "fields": "uniprot.Swiss-Prot",
                    "size": 1,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
            if not hits:
                return None
            sp = hits[0].get("uniprot", {}).get("Swiss-Prot")
            if isinstance(sp, str):
                return sp
            if isinstance(sp, list) and sp:
                return sp[0]
        except Exception as e:  # noqa: BLE001
            logger.debug("MyGene resolution failed for %s: %s", identifier, e)
        return None

    def _resolve_via_uniprot_search(
        self,
        identifier: str,
        id_type: str,
        organism: str,
    ) -> str | None:
        taxon = ORGANISM_TAXON.get(organism.lower())
        if taxon is None:
            logger.warning("Unknown organism '%s' for UniProt search", organism)
            return None

        if id_type == "symbol":
            query = f"gene_exact:{identifier} AND organism_id:{taxon} AND reviewed:true"
        elif id_type == "ensembl_id":
            query = f"xref:{identifier} AND organism_id:{taxon} AND reviewed:true"
        else:
            return None

        try:
            resp = requests.get(
                f"{UNIPROT_REST}/uniprotkb/search",
                params={
                    "query": query,
                    "format": "json",
                    "size": 1,
                    "fields": "accession",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                return results[0]["primaryAccession"]
        except Exception as e:  # noqa: BLE001
            logger.debug("UniProt search failed for %s: %s", identifier, e)
        return None

    # ------------------------------------------------------------------
    # Canonical sequence
    # ------------------------------------------------------------------

    def get_canonical_sequence(
        self,
        identifier: str,
        id_type: Literal["symbol", "ensembl_id", "uniprot_id"] = "symbol",
        organism: str | None = None,
    ) -> str | None:
        """Fetch the canonical protein sequence from UniProt.

        Parameters
        ----------
        identifier
            Gene symbol, Ensembl gene ID, or UniProt accession.
        id_type
            Type of identifier.
        organism
            Organism name.

        Returns
        -------
        Amino acid sequence string, or ``None`` on failure.
        """
        accession = self.resolve_uniprot_id(identifier, id_type, organism)
        if accession is None:
            logger.warning("Could not resolve UniProt ID for %s '%s'", id_type, identifier)
            return None
        return self._fetch_fasta_sequence(accession)

    def _fetch_fasta_sequence(self, accession: str) -> str | None:
        """Fetch the FASTA sequence for a UniProt accession."""
        try:
            resp = requests.get(
                f"{UNIPROT_REST}/uniprotkb/{accession}.fasta",
                timeout=self.timeout,
            )
            resp.raise_for_status()
            lines = resp.text.strip().split("\n")
            return "".join(lines[1:])
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to fetch FASTA for %s: %s", accession, e)
            return None

    # ------------------------------------------------------------------
    # Isoforms
    # ------------------------------------------------------------------

    def get_isoforms(
        self,
        identifier: str,
        id_type: Literal["symbol", "ensembl_id", "uniprot_id"] = "symbol",
        organism: str | None = None,
        include_canonical: bool = True,
    ) -> dict[str, str]:
        """Fetch all protein isoforms for a gene from UniProt.

        Parameters
        ----------
        identifier
            Gene symbol, Ensembl gene ID, or UniProt accession.
        id_type
            Type of identifier.
        organism
            Organism name.
        include_canonical
            If ``True`` (default), the canonical sequence is included
            in the returned dict under the base accession.

        Returns
        -------
        Dict mapping isoform accession (e.g. ``"P04637-1"``) to
        amino acid sequence.  Empty dict on failure.
        """
        accession = self.resolve_uniprot_id(identifier, id_type, organism)
        if accession is None:
            logger.warning("Could not resolve UniProt ID for %s '%s'", id_type, identifier)
            return {}

        try:
            resp = requests.get(
                f"{UNIPROT_REST}/uniprotkb/{accession}.fasta",
                params={"includeIsoform": "true"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return self._parse_multi_fasta(resp.text, include_canonical)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to fetch isoforms for %s: %s", accession, e)
            return {}

    @staticmethod
    def _parse_multi_fasta(
        fasta_text: str,
        include_canonical: bool = True,
    ) -> dict[str, str]:
        """Parse a multi-entry FASTA string into {accession: sequence}."""
        entries: dict[str, str] = {}
        current_id: str | None = None
        lines: list[str] = []

        for line in fasta_text.strip().split("\n"):
            if line.startswith(">"):
                if current_id is not None:
                    entries[current_id] = "".join(lines)
                parts = line[1:].split("|")
                if len(parts) >= 2:
                    current_id = parts[1]
                else:
                    current_id = line[1:].split()[0]
                lines = []
            else:
                lines.append(line.strip())

        if current_id is not None:
            entries[current_id] = "".join(lines)

        if not include_canonical:
            base = min(entries.keys(), key=len) if entries else None
            if base and "-" not in base:
                entries.pop(base, None)

        return entries

    # ------------------------------------------------------------------
    # Bulk / batch methods
    # ------------------------------------------------------------------

    def get_canonical_sequences_batch(
        self,
        identifiers: list[str],
        id_type: Literal["symbol", "ensembl_id", "uniprot_id"] = "symbol",
        organism: str | None = None,
    ) -> dict[str, str]:
        """Fetch canonical sequences for a list of identifiers.

        Parameters
        ----------
        identifiers
            List of gene identifiers.
        id_type
            Type of identifiers.
        organism
            Organism name.

        Returns
        -------
        Dict mapping identifier to amino acid sequence.
        """
        results: dict[str, str] = {}
        total = len(identifiers)
        for i, ident in enumerate(identifiers):
            seq = self.get_canonical_sequence(ident, id_type, organism)
            if seq:
                results[ident] = seq
            if (i + 1) % 100 == 0:
                logger.info("Fetched canonical protein %d/%d ...", i + 1, total)
            time.sleep(self.rate_limit_delay)
        logger.info("Fetched %d/%d canonical protein sequences.", len(results), total)
        return results

    def get_isoforms_batch(
        self,
        identifiers: list[str],
        id_type: Literal["symbol", "ensembl_id", "uniprot_id"] = "symbol",
        organism: str | None = None,
        include_canonical: bool = True,
    ) -> dict[str, dict[str, str]]:
        """Fetch isoforms for a list of identifiers.

        Parameters
        ----------
        identifiers
            List of gene identifiers.
        id_type
            Type of identifiers.
        organism
            Organism name.
        include_canonical
            Whether to include the canonical sequence.

        Returns
        -------
        Nested dict: ``{identifier: {isoform_accession: sequence}}``.
        """
        results: dict[str, dict[str, str]] = {}
        total = len(identifiers)
        for i, ident in enumerate(identifiers):
            isoforms = self.get_isoforms(ident, id_type, organism, include_canonical)
            if isoforms:
                results[ident] = isoforms
            if (i + 1) % 100 == 0:
                logger.info("Fetched isoforms for %d/%d genes ...", i + 1, total)
            time.sleep(self.rate_limit_delay)
        logger.info("Fetched isoforms for %d/%d genes.", len(results), total)
        return results

    def get_all_canonical_sequences(
        self,
        organism: str | None = None,
        biotype: str = "protein_coding",
    ) -> dict[str, str]:
        """Fetch canonical protein sequences for all genes of a given biotype.

        Uses pyensembl to enumerate genes, then fetches each canonical
        protein from UniProt.

        Parameters
        ----------
        organism
            Organism name (default: ``self.organism``).
        biotype
            Gene biotype filter (default: ``"protein_coding"``).

        Returns
        -------
        Dict mapping Ensembl gene ID to canonical amino acid sequence.
        """
        org = organism or self.organism
        gene_ids = self._get_all_gene_ids(biotype)
        if not gene_ids:
            logger.error("No genes found for biotype '%s'", biotype)
            return {}

        logger.info(
            "Fetching canonical protein sequences for %d %s genes ...",
            len(gene_ids), biotype,
        )
        return self.get_canonical_sequences_batch(
            gene_ids, id_type="ensembl_id", organism=org,
        )

    def get_all_isoform_sequences(
        self,
        organism: str | None = None,
        biotype: str = "protein_coding",
        include_canonical: bool = True,
    ) -> dict[str, dict[str, str]]:
        """Fetch all isoform sequences for all genes of a given biotype.

        Parameters
        ----------
        organism
            Organism name.
        biotype
            Gene biotype filter.
        include_canonical
            Whether to include the canonical sequence in each gene's
            isoform dict.

        Returns
        -------
        Nested dict: ``{ensembl_gene_id: {isoform_accession: sequence}}``.
        """
        org = organism or self.organism
        gene_ids = self._get_all_gene_ids(biotype)
        if not gene_ids:
            logger.error("No genes found for biotype '%s'", biotype)
            return {}

        logger.info(
            "Fetching isoform sequences for %d %s genes ...",
            len(gene_ids), biotype,
        )
        return self.get_isoforms_batch(
            gene_ids,
            id_type="ensembl_id",
            organism=org,
            include_canonical=include_canonical,
        )

    def _get_all_gene_ids(self, biotype: str = "protein_coding") -> list[str]:
        """Get all gene IDs for a biotype using pyensembl."""
        try:
            import pyensembl
            ens = pyensembl.EnsemblRelease(release=109, species=self.organism)
            ens.download()
            ens.index()
            all_genes = ens.genes()
            if biotype.lower() != "all":
                all_genes = [g for g in all_genes if g.biotype == biotype]
            gene_ids = [g.gene_id for g in all_genes]
            logger.info("Found %d %s genes with biotype '%s'", len(gene_ids), self.organism, biotype)
            return gene_ids
        except ImportError:
            logger.error("pyensembl is required for bulk gene enumeration")
            return []
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to enumerate genes for %s: %s", self.organism, e)
            return []
