"""Reactome pathway membership from the Reactome release files.

Reactome is read from the files Reactome itself publishes for each release, not
from MyGene.info. MyGene's ``pathway.reactome`` field is not a usable Reactome
source: in MyGene build 20261030 only 2,753 human genes carry it (EGFR and TP53
have none), against 11,492 human NCBI Gene IDs in Reactome release 98's
``NCBI2Reactome.txt``.

Three files are joined:

``NCBI2Reactome.txt``
    NCBI Gene ID, pathway stId, browser URL, pathway name, evidence code,
    species. It maps genes to their *lowest-level* pathways and covers every
    species, so rows are filtered on the species column (``"Homo sapiens"`` by
    default). The same ID can appear for several species (for example 1253480
    has both Homo sapiens and Gallus gallus rows), so the filter is on the row,
    not on the ID.
``ReactomePathwaysRelation.txt``
    ``parent<TAB>child`` stId pairs for every species. Walking a pathway's
    parents up until no parent is left gives its top-level pathway(s); the
    hierarchy is a DAG, so one pathway can reach more than one top-level
    pathway. For release 98 the human roots are exactly the 29 pathways that
    ``ContentService/data/pathways/top/9606`` returns, Disease included.
``ReactomePathways.txt``
    stId, name, species. Names carry stray outer whitespace in both this file
    and ``NCBI2Reactome.txt`` (204 and 2,369 rows in release 98), which is
    stripped.

Keys are NCBI Gene IDs (MyGene's ``entrezgene``). A MyGene hit with no numeric
Entrez ID (an Ensembl-only document) has no key here and is never joined.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Release the URLs below are pinned to. ``download.reactome.org/<release>/``
#: keeps old releases; ``reactome.org/download/current/`` moves with each release.
REACTOME_RELEASE = "98"
_PINNED_BASE = f"https://download.reactome.org/{REACTOME_RELEASE}"
REACTOME_CURRENT_BASE = "https://reactome.org/download/current"

#: Files :meth:`ReactomeIndex.from_dir` reads, keyed by the file name they are
#: stored under. Verified on 2026-11-11 with HEAD requests: HTTP 200,
#: Content-Length 35,250,351 / 635,604 / 1,596,703 bytes, Last-Modified
#: 2026-10-22 (the ``current`` copies are byte-identical in size and were
#: modified on 2026-10-18).
REACTOME_FILES: dict[str, str] = {
    "NCBI2Reactome.txt": f"{_PINNED_BASE}/NCBI2Reactome.txt",
    "ReactomePathwaysRelation.txt": f"{_PINNED_BASE}/ReactomePathwaysRelation.txt",
    "ReactomePathways.txt": f"{_PINNED_BASE}/ReactomePathways.txt",
}

#: Not read by :class:`ReactomeIndex` (it rebuilds the hierarchy itself); listed
#: for completeness. 97,248,132 bytes for release 98.
REACTOME_OPTIONAL_FILES: dict[str, str] = {
    "NCBI2Reactome_All_Levels.txt": f"{_PINNED_BASE}/NCBI2Reactome_All_Levels.txt",
}

#: Returns the bare release number as text (``98`` on 2026-11-11).
REACTOME_VERSION_URL = "https://reactome.org/ContentService/data/database/version"
#: The 29 human top-level pathways, for cross-checking the computed roots.
REACTOME_TOP_LEVEL_URL = "https://reactome.org/ContentService/data/pathways/top/9606"
#: Read from https://reactome.org/license on 2026-11-11.
REACTOME_LICENCE = (
    "CC0 1.0 for Reactome data and files derived from it; "
    "CC BY 4.0 for illustrations (https://reactome.org/license)"
)

_RELEASE_KEYS = ("release", "version", "database_version", "reactome_release")


def _open_text(path: Path) -> io.TextIOBase:
    """Open a plain or gzip-compressed text file."""
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, encoding="utf-8")


def _rows(path: Path, n_cols: int) -> Iterator[list[str]]:
    """Yield tab-separated rows with at least ``n_cols`` columns."""
    with _open_text(path) as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < n_cols:
                logger.debug("%s:%d: expected %d columns, got %d", path.name, line_no, n_cols, len(cols))
                continue
            yield cols


def _find(directory: Path, name: str, *, required: bool = True) -> Path | None:
    for candidate in (directory / name, directory / f"{name}.gz"):
        if candidate.is_file():
            return candidate
    if required:
        raise FileNotFoundError(f"{name} (or {name}.gz) not found in {directory}")
    return None


def _read_release(path: Path | None) -> str | None:
    """Read the release from ``version.json``.

    Accepts a bare number or string (``98``, ``"98"``) or an object with one of
    the keys ``release``, ``version``, ``database_version`` or
    ``reactome_release``.
    """
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("Could not read Reactome release from %s: %s", path, e)
        return None
    if isinstance(data, (int, str)) and str(data).strip():
        return str(data).strip()
    if isinstance(data, dict):
        for key in _RELEASE_KEYS:
            value = data.get(key)
            if isinstance(value, (int, str)) and str(value).strip():
                return str(value).strip()
    logger.warning("No release key (%s) in %s", ", ".join(_RELEASE_KEYS), path)
    return None


def _normalise_gene_id(entrez_id: Any) -> str | None:
    if entrez_id is None or isinstance(entrez_id, bool):
        return None
    if isinstance(entrez_id, int):
        return str(entrez_id)
    text = str(entrez_id).strip()
    return text or None


class ReactomeIndex:
    """Gene -> Reactome pathway index built from one Reactome release.

    Parameters
    ----------
    gene_pathways
        NCBI Gene ID -> set of lowest-level pathway stIds (species-filtered).
    names
        stId -> pathway name (already stripped).
    parents
        child stId -> parent stIds.
    roots
        stIds of the species' top-level pathways.
    release
        Reactome release, e.g. ``"98"``.
    species
        Species the rows were filtered on.
    """

    def __init__(
        self,
        gene_pathways: dict[str, set[str]],
        names: dict[str, str],
        parents: dict[str, set[str]],
        roots: Iterable[str],
        *,
        release: str | None = None,
        species: str = "Homo sapiens",
        files: dict[str, str] | None = None,
    ) -> None:
        self._gene_pathways = gene_pathways
        self._names = names
        self._parents = parents
        self._roots = frozenset(roots)
        self._top_cache: dict[str, frozenset[str]] = {}
        self.release = release
        self.species = species
        self.files = dict(files or {})

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dir(
        cls,
        path: str | os.PathLike[str],
        *,
        species: str = "Homo sapiens",
        release: str | None = None,
    ) -> ReactomeIndex:
        """Build the index from a folder holding the :data:`REACTOME_FILES`.

        Each file may also be gzip-compressed (``<name>.gz``). The release is
        taken from ``release`` if given, else from an optional ``version.json``
        in the same folder.
        """
        directory = Path(path)
        if release is None:
            release = _read_release(_find(directory, "version.json", required=False))
        return cls.from_files(
            ncbi2reactome=_find(directory, "NCBI2Reactome.txt"),
            relation=_find(directory, "ReactomePathwaysRelation.txt"),
            pathways=_find(directory, "ReactomePathways.txt"),
            species=species,
            release=release,
        )

    @classmethod
    def from_files(
        cls,
        ncbi2reactome: str | os.PathLike[str],
        relation: str | os.PathLike[str],
        pathways: str | os.PathLike[str],
        *,
        species: str = "Homo sapiens",
        release: str | None = None,
    ) -> ReactomeIndex:
        """Build the index from the three release files given one by one."""
        ncbi_path, relation_path, pathways_path = Path(ncbi2reactome), Path(relation), Path(pathways)

        names: dict[str, str] = {}
        species_pathways: set[str] = set()
        for st_id, name, row_species in (c[:3] for c in _rows(pathways_path, 3)):
            st_id = st_id.strip()
            names[st_id] = name.strip()
            if row_species.strip() == species:
                species_pathways.add(st_id)

        parents: dict[str, set[str]] = defaultdict(set)
        for parent, child in (c[:2] for c in _rows(relation_path, 2)):
            parents[child.strip()].add(parent.strip())

        gene_pathways: dict[str, set[str]] = defaultdict(set)
        n_rows = n_kept = 0
        for gene_id, st_id, _url, name, _evidence, row_species in (c[:6] for c in _rows(ncbi_path, 6)):
            n_rows += 1
            if row_species.strip() != species:
                continue
            n_kept += 1
            st_id = st_id.strip()
            gene_pathways[gene_id.strip()].add(st_id)
            # ReactomePathways.txt is the name of record; NCBI2Reactome only fills gaps.
            names.setdefault(st_id, name.strip())

        roots = sorted(p for p in species_pathways if not parents.get(p))
        logger.info(
            "Reactome %s (%s): %d of %d NCBI2Reactome rows kept, %d genes, %d top-level pathways",
            release or "release unknown", species, n_kept, n_rows, len(gene_pathways), len(roots),
        )
        return cls(
            dict(gene_pathways),
            names,
            dict(parents),
            roots,
            release=release,
            species=species,
            files={
                "NCBI2Reactome.txt": str(ncbi_path),
                "ReactomePathwaysRelation.txt": str(relation_path),
                "ReactomePathways.txt": str(pathways_path),
            },
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._gene_pathways)

    def __contains__(self, entrez_id: object) -> bool:
        key = _normalise_gene_id(entrez_id)
        return key is not None and key in self._gene_pathways

    @property
    def top_level_pathways(self) -> list[dict[str, str]]:
        """The species' top-level pathways (29 for human in release 98)."""
        return [self._entry(s) for s in sorted(self._roots)]

    def pathway_name(self, st_id: str) -> str:
        return self._names.get(st_id, "")

    def top_level_of(self, st_id: str) -> list[dict[str, str]]:
        """Top-level pathway(s) reached by walking ``st_id``'s parents."""
        return [self._entry(s) for s in sorted(self._top_level_ids(st_id))]

    def lookup(self, entrez_id: Any) -> dict[str, Any]:
        """Reactome pathways for one NCBI Gene ID.

        Returns
        -------
        ``{"status": "success" | "not_in_reactome", "entrez_id", "pathways",
        "top_level", "release"}``. ``pathways`` are the lowest-level pathways
        (``{"id", "name"}``, sorted by stId); ``top_level`` are their top-level
        ancestors. A missing, Ensembl-only or unknown ID gives
        ``not_in_reactome`` with empty lists.
        """
        key = _normalise_gene_id(entrez_id)
        st_ids = self._gene_pathways.get(key, set()) if key is not None else set()
        if not st_ids:
            return {
                "status": "not_in_reactome",
                "entrez_id": key,
                "pathways": [],
                "top_level": [],
                "release": self.release,
            }
        top: set[str] = set()
        for st_id in st_ids:
            top |= self._top_level_ids(st_id)
        return {
            "status": "success",
            "entrez_id": key,
            "pathways": [self._entry(s) for s in sorted(st_ids)],
            "top_level": [self._entry(s) for s in sorted(top)],
            "release": self.release,
        }

    def lookup_many(self, entrez_ids: Iterable[Any]) -> dict[Any, dict[str, Any]]:
        """:meth:`lookup` for several IDs, keyed by the IDs as given."""
        return {gene_id: self.lookup(gene_id) for gene_id in entrez_ids}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _entry(self, st_id: str) -> dict[str, str]:
        return {"id": st_id, "name": self._names.get(st_id, "")}

    def _top_level_ids(self, st_id: str) -> frozenset[str]:
        cached = self._top_cache.get(st_id)
        if cached is not None:
            return cached
        found: set[str] = set()
        seen: set[str] = set()
        stack = [st_id]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            node_parents = self._parents.get(node)
            if node_parents:
                stack.extend(node_parents)
            elif node in self._roots:
                found.add(node)
            else:
                # No parent and not a known top-level pathway of this species:
                # the pathway is missing from ReactomePathways.txt. Do not
                # promote it to a top-level category.
                logger.debug("Reactome pathway %s has no parent and is not a top-level pathway", node)
        result = frozenset(found)
        self._top_cache[st_id] = result
        return result
