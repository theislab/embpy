"""Cross-species ortholog resolution via Ensembl Compara REST API.

Provides automated mapping of gene symbols across species using
one-to-one and one-to-many ortholog relationships curated by Ensembl
Compara.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

import requests

logger = logging.getLogger(__name__)

# Ensembl REST API base
_ENSEMBL_REST = "https://rest.ensembl.org"

# Map common names to Ensembl species identifiers
SPECIES_MAP: dict[str, str] = {
    "human": "homo_sapiens",
    "homo_sapiens": "homo_sapiens",
    "mouse": "mus_musculus",
    "mus_musculus": "mus_musculus",
    "rat": "rattus_norvegicus",
    "rattus_norvegicus": "rattus_norvegicus",
    "zebrafish": "danio_rerio",
    "danio_rerio": "danio_rerio",
    "drosophila": "drosophila_melanogaster",
    "drosophila_melanogaster": "drosophila_melanogaster",
    "worm": "caenorhabditis_elegans",
    "caenorhabditis_elegans": "caenorhabditis_elegans",
    "yeast": "saccharomyces_cerevisiae",
    "saccharomyces_cerevisiae": "saccharomyces_cerevisiae",
    "chicken": "gallus_gallus",
    "gallus_gallus": "gallus_gallus",
    "pig": "sus_scrofa",
    "sus_scrofa": "sus_scrofa",
    "dog": "canis_lupus_familiaris",
    "canis_lupus_familiaris": "canis_lupus_familiaris",
    "macaque": "macaca_mulatta",
    "macaca_mulatta": "macaca_mulatta",
    "chimpanzee": "pan_troglodytes",
    "pan_troglodytes": "pan_troglodytes",
    "gorilla": "gorilla_gorilla",
    "gorilla_gorilla": "gorilla_gorilla",
    "cow": "bos_taurus",
    "bos_taurus": "bos_taurus",
    "frog": "xenopus_tropicalis",
    "xenopus_tropicalis": "xenopus_tropicalis",
}


class OrthologResult:
    """Container for a single ortholog hit."""

    __slots__ = (
        "source_symbol", "source_species",
        "target_symbol", "target_species", "target_ensembl_id",
        "orthology_type", "perc_id", "perc_pos",
        "target_perc_id", "target_perc_pos",
    )

    def __init__(
        self,
        source_symbol: str,
        source_species: str,
        target_symbol: str,
        target_species: str,
        target_ensembl_id: str,
        orthology_type: str,
        perc_id: float,
        perc_pos: float,
        target_perc_id: float,
        target_perc_pos: float,
    ) -> None:
        self.source_symbol = source_symbol
        self.source_species = source_species
        self.target_symbol = target_symbol
        self.target_species = target_species
        self.target_ensembl_id = target_ensembl_id
        self.orthology_type = orthology_type
        self.perc_id = perc_id
        self.perc_pos = perc_pos
        self.target_perc_id = target_perc_id
        self.target_perc_pos = target_perc_pos

    def __repr__(self) -> str:
        return (
            f"OrthologResult({self.source_symbol} [{self.source_species}] "
            f"-> {self.target_symbol} [{self.target_species}], "
            f"type={self.orthology_type}, identity={self.perc_id:.1f}%)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_symbol": self.source_symbol,
            "source_species": self.source_species,
            "target_symbol": self.target_symbol,
            "target_species": self.target_species,
            "target_ensembl_id": self.target_ensembl_id,
            "orthology_type": self.orthology_type,
            "perc_id": self.perc_id,
            "perc_pos": self.perc_pos,
            "target_perc_id": self.target_perc_id,
            "target_perc_pos": self.target_perc_pos,
        }


class OrthologResolver:
    """Resolve orthologs across species using Ensembl Compara.

    Parameters
    ----------
    source_species : str
        Default source species (e.g. ``"human"``, ``"mouse"``).
    rate_limit_delay : float
        Seconds between consecutive API calls.
    request_timeout : int
        HTTP timeout in seconds.
    """

    def __init__(
        self,
        source_species: str = "human",
        rate_limit_delay: float = 0.15,
        request_timeout: int = 30,
    ) -> None:
        self.source_species = self._normalize(source_species)
        self.rate_limit_delay = rate_limit_delay
        self.request_timeout = request_timeout
        self._cache: dict[tuple[str, str], list[OrthologResult]] = {}
        self._symbol_cache: dict[str, str] = {}

    @staticmethod
    def _normalize(species: str) -> str:
        key = species.lower().strip()
        if key in SPECIES_MAP:
            return SPECIES_MAP[key]
        return key

    def get_orthologs(
        self,
        symbol: str,
        target_species: str | list[str] | None = None,
        source_species: str | None = None,
        orthology_type: Literal[
            "ortholog_one2one", "ortholog_one2many",
            "ortholog_many2many", "all",
        ] = "all",
    ) -> list[OrthologResult]:
        """Find orthologs for a gene symbol.

        Parameters
        ----------
        symbol
            Gene symbol in the source species (e.g. ``"TP53"``).
        target_species
            One or more target species. ``None`` returns orthologs in
            all available species.
        source_species
            Override the default source species.
        orthology_type
            Filter by orthology type. ``"all"`` keeps everything.

        Returns
        -------
        List of :class:`OrthologResult` objects.
        """
        src = self._normalize(source_species or self.source_species)
        cache_key = (symbol, src)

        if cache_key not in self._cache:
            self._cache[cache_key] = self._fetch_orthologs(symbol, src)

        results = self._cache[cache_key]

        if target_species is not None:
            if isinstance(target_species, str):
                target_species = [target_species]
            targets = {self._normalize(t) for t in target_species}
            results = [r for r in results if r.target_species in targets]

        if orthology_type != "all":
            results = [r for r in results if r.orthology_type == orthology_type]

        self._resolve_symbols(results)
        return results

    def get_orthologs_batch(
        self,
        symbols: list[str],
        target_species: str | list[str] | None = None,
        source_species: str | None = None,
        orthology_type: str = "all",
    ) -> dict[str, list[OrthologResult]]:
        """Find orthologs for multiple gene symbols.

        Returns
        -------
        Dict mapping each source symbol to its ortholog results.
        """
        result = {}
        for symbol in symbols:
            result[symbol] = self.get_orthologs(
                symbol,
                target_species=target_species,
                source_species=source_species,
                orthology_type=orthology_type,
            )
        return result

    def get_ortholog_table(
        self,
        symbols: list[str],
        target_species: list[str],
        source_species: str | None = None,
        orthology_type: str = "ortholog_one2one",
    ) -> "pandas.DataFrame":
        """Build a tabular mapping of genes across species.

        Returns a DataFrame with columns: source_symbol, plus one column
        per target species containing the ortholog symbol (or NaN).
        Additional columns: sequence identity per species pair.

        Parameters
        ----------
        symbols
            Gene symbols in the source species.
        target_species
            Target species to include as columns.
        source_species
            Source species (default: instance default).
        orthology_type
            Filter type (default one-to-one for clean mapping).
        """
        import pandas as pd

        src = self._normalize(source_species or self.source_species)
        targets_norm = [self._normalize(t) for t in target_species]

        rows = []
        for symbol in symbols:
            orthologs = self.get_orthologs(
                symbol, target_species=targets_norm,
                source_species=src, orthology_type=orthology_type,
            )
            row: dict[str, Any] = {"source_symbol": symbol, "source_species": src}
            for tgt in targets_norm:
                hits = [o for o in orthologs if o.target_species == tgt]
                if hits:
                    best = max(hits, key=lambda o: o.perc_id)
                    short = tgt.split("_")[0]
                    row[f"{short}_symbol"] = best.target_symbol
                    row[f"{short}_identity"] = best.perc_id
                    row[f"{short}_ensembl_id"] = best.target_ensembl_id
                else:
                    short = tgt.split("_")[0]
                    row[f"{short}_symbol"] = None
                    row[f"{short}_identity"] = None
                    row[f"{short}_ensembl_id"] = None
            rows.append(row)

        return pd.DataFrame(rows)

    def _fetch_orthologs(
        self, symbol: str, source_species: str,
    ) -> list[OrthologResult]:
        """Query Ensembl Compara REST API for orthologs."""
        url = (
            f"{_ENSEMBL_REST}/homology/symbol/"
            f"{source_species}/{symbol}"
        )
        params = {
            "type": "orthologues",
            "sequence": "none",
        }
        headers = {"Content-Type": "application/json"}

        time.sleep(self.rate_limit_delay)

        try:
            resp = requests.get(
                url, params=params, headers=headers,
                timeout=self.request_timeout,
            )
            if resp.status_code == 429:
                retry = float(resp.headers.get("Retry-After", "2"))
                logger.info("Rate limited by Ensembl, sleeping %.1fs", retry)
                time.sleep(retry)
                resp = requests.get(
                    url, params=params, headers=headers,
                    timeout=self.request_timeout,
                )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(
                "Ensembl ortholog lookup failed for %s (%s): %s",
                symbol, source_species, exc,
            )
            return []

        data = resp.json()
        homologies = (
            data.get("data", [{}])[0]
            .get("homologies", [])
        )

        results = []
        for hom in homologies:
            target = hom.get("target", {})
            tgt_species = target.get("species", "").lower().replace(" ", "_")
            tgt_id = target.get("id", "")

            results.append(OrthologResult(
                source_symbol=symbol,
                source_species=source_species,
                target_symbol=tgt_id,
                target_species=tgt_species,
                target_ensembl_id=tgt_id,
                orthology_type=hom.get("type", ""),
                perc_id=float(hom.get("source", {}).get("perc_id", 0)),
                perc_pos=float(hom.get("source", {}).get("perc_pos", 0)),
                target_perc_id=float(target.get("perc_id", 0)),
                target_perc_pos=float(target.get("perc_pos", 0)),
            ))

        logger.info(
            "Found %d orthologs for %s (%s)",
            len(results), symbol, source_species,
        )
        return results

    def _resolve_symbols(self, results: list[OrthologResult]) -> None:
        """Resolve Ensembl gene IDs to gene symbols for the given results.

        Only queries the API for IDs not already in the symbol cache.
        Mutates target_symbol in place.
        """
        unresolved = [
            r.target_ensembl_id
            for r in results
            if r.target_ensembl_id not in self._symbol_cache
            and r.target_ensembl_id
        ]
        unresolved = list(set(unresolved))

        if unresolved:
            batch_size = 50
            for i in range(0, len(unresolved), batch_size):
                batch = unresolved[i : i + batch_size]
                time.sleep(self.rate_limit_delay)
                try:
                    resp = requests.post(
                        f"{_ENSEMBL_REST}/lookup/id",
                        json={"ids": batch},
                        headers={"Content-Type": "application/json"},
                        timeout=self.request_timeout,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for eid, info in data.items():
                        self._symbol_cache[eid] = (
                            info.get("display_name") or eid
                        )
                except requests.RequestException as exc:
                    logger.warning("Ensembl ID lookup failed: %s", exc)
                    for eid in batch:
                        self._symbol_cache.setdefault(eid, eid)

        for r in results:
            if r.target_ensembl_id in self._symbol_cache:
                r.target_symbol = self._symbol_cache[r.target_ensembl_id]
