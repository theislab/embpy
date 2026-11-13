r"""Offline ChEBI role lookup from a pinned ``chebi_core.obo`` release.

:class:`ChebiOboIndex` stream-parses ChEBI's ``chebi_core.obo`` (plain or
gzipped) once and answers "which ChEBI entity is this structure, and which
roles does ChEBI assert for it?" by InChIKey, without any network call.

The parser was written against release 257 (``data-version: chebi/257``,
2026-11-05, https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi_core.obo.gz,
CC BY 4.0). The tags it reads, as they appear in that file:

``id: CHEBI:15365``
    Term identifier.
``name: acetylsalicylic acid``
    OBO escapes such as ``\{`` and ``\"`` are unescaped.
``subset: 3:STAR``
    ChEBI star rating (1 = preliminary, 2 = annotated by a third party,
    3 = manually annotated by the ChEBI team).
``alt_id: CHEBI:22188``
    Secondary ID that now points at this term.
``is_a: CHEBI:26596 ! salicylates``
    Structural class. Kept for :meth:`ChebiOboIndex.ancestors`, but roles
    are **never** inherited through it.
``relationship: RO:0000087 CHEBI:35475 ! has role non-steroidal anti-inflammatory drug``
    The direct ``has role`` assertion. Only this relation produces roles.
``property_value: chemrof:inchi_key_string "BSYNRYMUTXBXSQ-UHFFFAOYSA-N" xsd:string``
    Standard InChIKey of the entity.
``is_obsolete: true``
    Obsolete stanza (release 257 has one, ``CHEBI:27189``, with no other tags).

Matching
--------
:meth:`ChebiOboIndex.lookup` matches in two tiers:

1. ``exact``: the full standard InChIKey. Several ChEBI entities can share
   one key (1,473 keys in release 257, e.g. valpromide CHEBI:74562 and its
   1-star duplicate CHEBI:46316). They describe the same structure, so the
   highest-star entity (then the lowest ID) is reported as the match and the
   roles are the union of the direct ``has role`` targets of all of them.
2. ``connectivity``: the first InChIKey block (the skeleton, without
   stereo, isotope or protonation layers). Used only when the exact key is
   absent. Exactly one distinct ChEBI InChIKey in the block gives a match
   flagged ``match_tier="connectivity"``; more than one gives
   ``ambiguous_connectivity`` with every candidate listed and no roles, so no
   stereo variant is silently preferred.

When the exact tier matches, the result also carries a ``"connectivity"``
entry: the same connectivity analysis over the *other* keys of the block.
That lets a caller emit a separately flagged connectivity label family (for
example tamoxifen: the stereo-free key is the role-less 2-star CHEBI:92766,
while the curated 3-star CHEBI:41774 is only reachable by connectivity)
without counting a structure twice. When the top-level match is itself a
connectivity match, ``"connectivity"`` is ``None``.

Statuses: ``success`` (matched, at least one direct role),
``matched_no_role`` (matched, ChEBI asserts no role; common for 2-star LINCS
imports), ``ambiguous_connectivity``, ``no_chebi_entity`` and
``invalid_structure`` (not a valid InChIKey, or a SMILES RDKit cannot parse).

Memory and speed: IDs are stored as ints and the InChIKey block index is a
sorted list searched with :mod:`bisect`, so the ~219,000 terms of release 257
load in seconds and hold only names, stars, ``is_a`` parents, ``has role``
edges and InChIKeys (definitions and other properties are skipped).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import logging
import re
from bisect import bisect_left
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Relation ID of ``has role`` in ChEBI OBO files.
HAS_ROLE = "RO:0000087"
#: Default download location of the CORE flavour (structures and relations, no synonyms or xrefs).
CHEBI_OBO_URL = "https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi_core.obo.gz"
#: Licence of ChEBI data, recorded in provenance.
CHEBI_LICENCE = "CC BY 4.0"
#: File names tried, in order, when :meth:`ChebiOboIndex.from_obo` is given a directory.
CHEBI_OBO_FILENAMES = ("chebi_core.obo.gz", "chebi_core.obo", "chebi.obo.gz", "chebi.obo")

#: Every status :meth:`ChebiOboIndex.lookup` can return (``service_error`` is
#: reserved for online callers sharing this result shape; the index never emits it).
LOOKUP_STATUSES = (
    "success",
    "matched_no_role",
    "no_chebi_entity",
    "ambiguous_connectivity",
    "invalid_structure",
    "service_error",
)

#: ``property_value`` prefixes that carry the InChIKey. ``chemrof:inchi_key_string``
#: is what release 257 uses (verified). The purl form is the property name used by
#: ChEBI OBO files before the ChEBI 2.0 migration; it is accepted for older pinned
#: releases but was not re-verified against such a file in this change.
_INCHIKEY_PROPERTIES = (
    "chemrof:inchi_key_string ",
    "http://purl.obolibrary.org/obo/chebi/inchikey ",
)
_INCHIKEY_RE = re.compile(r"[A-Z]{14}-[A-Z]{10}-[A-Z]")
_OBO_ESCAPES = {"n": "\n", "t": "\t", "W": " "}
_HAS_ROLE_COMMENT = "has role "
_HEADER_TAGS = frozenset({"format-version", "data-version", "date", "saved-by", "ontology", "default-namespace"})


def _chebi_int(value: str) -> int | None:
    """Return the integer of the leading ``CHEBI:<n>`` token of an OBO value, or None."""
    end = value.find(" ")
    token = value if end < 0 else value[:end]
    if token[:6] != "CHEBI:":
        return None
    try:
        return int(token[6:])
    except ValueError:
        return None


def _unescape(value: str) -> str:
    r"""Undo OBO backslash escapes (``\{`` -> ``{``, ``\"`` -> ``"``, ``\n`` -> newline)."""
    if "\\" not in value:
        return value
    out: list[str] = []
    chars = iter(value)
    for char in chars:
        if char == "\\":
            nxt = next(chars, "")
            out.append(_OBO_ESCAPES.get(nxt, nxt))
        else:
            out.append(char)
    return "".join(out)


def _format_id(number: int) -> str:
    return f"CHEBI:{number}"


def _parse_id_arg(chebi_id: str | int) -> int | None:
    """Accept ``"CHEBI:15365"``, ``"CHEBI_15365"``, ``"15365"`` or ``15365``."""
    if isinstance(chebi_id, int):
        return chebi_id
    text = str(chebi_id).strip().upper()
    for prefix in ("CHEBI:", "CHEBI_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return int(text) if text.isdigit() else None


def normalise_inchikey(value: Any) -> str | None:
    """Return a stripped, upper-case InChIKey, or None when ``value`` is not one.

    Accepts an optional ``InChIKey=`` prefix. Both standard (``...SA-N``) and
    non-standard keys are syntactically valid; only standard keys occur in ChEBI.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text[:9].lower() == "inchikey=":
        text = text[9:]
    text = text.upper()
    return text if _INCHIKEY_RE.fullmatch(text) else None


def inchikey_from_structure(identifier: str) -> str | None:
    """Compute a standard InChIKey from a SMILES or ``InChI=`` string with RDKit.

    Returns None when RDKit cannot parse the structure. Raises ImportError when
    RDKit is not installed (a missing dependency is not an invalid structure).
    """
    try:
        from rdkit import Chem, rdBase
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "inchikey_from_structure needs RDKit (pip install rdkit); pass an InChIKey to ChebiOboIndex.lookup instead"
        ) from exc
    text = identifier.strip()
    if not text:
        return None
    block = rdBase.BlockLogs() if hasattr(rdBase, "BlockLogs") else None
    try:
        mol = Chem.MolFromInchi(text) if text.startswith("InChI=") else Chem.MolFromSmiles(text)
        if mol is None:
            return None
        key = Chem.MolToInchiKey(mol)
    finally:
        del block
    return key or None


def _open_text(path: Path) -> io.TextIOBase:
    with open(path, "rb") as handle:
        magic = handle.read(2)
    if magic == b"\x1f\x8b":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ParsedObo:
    """Plain containers filled by :func:`_parse_obo_lines` (IDs are ints)."""

    __slots__ = (
        "header",
        "names",
        "stars",
        "is_a",
        "roles",
        "role_comment_names",
        "inchikey_of",
        "alt_ids",
        "obsolete",
        "replaced_by",
        "relation_names",
        "n_terms",
    )

    def __init__(self) -> None:
        self.header: dict[str, str] = {}
        self.names: dict[int, str] = {}
        self.stars: dict[int, int] = {}
        self.is_a: dict[int, tuple[int, ...]] = {}
        self.roles: dict[int, tuple[int, ...]] = {}
        self.role_comment_names: dict[int, str] = {}
        self.inchikey_of: dict[int, str] = {}
        self.alt_ids: dict[int, int] = {}
        self.obsolete: set[int] = set()
        self.replaced_by: dict[int, int] = {}
        self.relation_names: dict[str, str] = {}
        self.n_terms = 0


def _parse_obo_lines(lines: Iterable[str], *, keep_is_a: bool = True) -> _ParsedObo:
    """Single streaming pass over OBO lines; keeps only what role lookup needs."""
    parsed = _ParsedObo()
    header = parsed.header
    names = parsed.names
    stars = parsed.stars
    is_a = parsed.is_a
    roles = parsed.roles
    role_comment_names = parsed.role_comment_names
    inchikey_of = parsed.inchikey_of
    alt_ids = parsed.alt_ids
    obsolete = parsed.obsolete
    replaced_by = parsed.replaced_by
    relation_names = parsed.relation_names
    inchikey_props = _INCHIKEY_PROPERTIES

    kind: str | None = None  # None while still in the header
    cur: int | None = None
    typedef_id: str | None = None
    name: str | None = None
    star = 0
    parents: list[int] = []
    role_ids: list[int] = []
    key: str | None = None
    alts: list[int] = []
    is_obsolete = False
    replacement: int | None = None

    def store() -> None:
        if kind == "Term" and cur is not None:
            parsed.n_terms += 1
            if name is not None:
                names[cur] = _unescape(name)
            if star:
                stars[cur] = star
            if parents and keep_is_a:
                is_a[cur] = tuple(parents)
            if role_ids:
                roles[cur] = tuple(dict.fromkeys(role_ids))
            if is_obsolete:
                obsolete.add(cur)
                if replacement is not None:
                    replaced_by[cur] = replacement
            elif key is not None:
                inchikey_of[cur] = key
            for alt in alts:
                alt_ids[alt] = cur
        elif kind == "Typedef" and typedef_id is not None and name is not None:
            relation_names[typedef_id] = _unescape(name)

    for line in lines:
        first = line[:1]
        if first == "[":
            store()
            kind = line.strip()[1:-1]
            cur = typedef_id = name = key = replacement = None
            star = 0
            parents = []
            role_ids = []
            alts = []
            is_obsolete = False
            continue
        if first in ("\n", "\r", "!", ""):
            continue
        tag, sep, value = line.partition(": ")
        if not sep:
            continue
        value = value.rstrip("\r\n")
        if kind == "Term":
            if tag == "property_value":
                if value.startswith(inchikey_props):
                    start = value.find('"') + 1
                    stop = value.find('"', start)
                    if start and stop > start:
                        key = value[start:stop].strip().upper()
            elif tag == "subset":
                # "3:STAR" (release 257); "3_STAR" in pre-2.0 files.
                if value[1:6] in (":STAR", "_STAR") and value[:1].isdigit():
                    star = max(star, int(value[:1]))
            elif tag == "is_a":
                parent = _chebi_int(value)
                if parent is not None:
                    parents.append(parent)
            elif tag == "relationship":
                relation, _, rest = value.partition(" ")
                if relation == HAS_ROLE:
                    target = _chebi_int(rest)
                    if target is not None:
                        role_ids.append(target)
                        if target not in role_comment_names:
                            _, bang, comment = rest.partition(" ! ")
                            if bang:
                                comment = comment.strip()
                                if comment.startswith(_HAS_ROLE_COMMENT):
                                    comment = comment[len(_HAS_ROLE_COMMENT) :]
                                role_comment_names[target] = comment
            elif tag == "id":
                cur = _chebi_int(value)
            elif tag == "name":
                name = value
            elif tag == "alt_id":
                alt = _chebi_int(value)
                if alt is not None:
                    alts.append(alt)
            elif tag == "is_obsolete":
                is_obsolete = value.strip() == "true"
            elif tag == "replaced_by":
                replacement = _chebi_int(value)
        elif kind is None:
            if tag == "property_value":
                if value.startswith("owl:versionInfo "):
                    start = value.find('"') + 1
                    stop = value.find('"', start)
                    if start and stop > start:
                        header["version_info"] = value[start:stop]
                elif value.startswith("terms:license "):
                    header["license"] = value.split(" ", 2)[1]
            elif tag in _HEADER_TAGS and tag not in header:
                header[tag] = value.strip()
        elif kind == "Typedef":
            if tag == "id":
                typedef_id = value.split(" ", 1)[0]
            elif tag == "name":
                name = value
    store()
    return parsed


def _release_from_header(header: Mapping[str, str]) -> str:
    version = header.get("data-version", "")
    number = version.rsplit("/", 1)[-1] if version else ""
    if not number:
        number = header.get("version_info", "")
    return f"ChEBI_{number}" if number else "ChEBI_unknown"


class ChebiOboIndex:
    """In-memory ChEBI index for offline role lookup by InChIKey.

    Build it with :meth:`from_obo`. The index is read-only after construction
    and safe to share between threads.

    Attributes
    ----------
    release
        ``"ChEBI_<n>"`` from the file's ``data-version: chebi/<n>`` header
        (``"ChEBI_257"`` for the pinned release).
    data_version
        The raw ``data-version`` header value, e.g. ``"chebi/257"``.
    date
        The raw ``date`` header value, e.g. ``"05:11:2026 18:04"``.
    source_path
        The file the index was built from (None when built from lines).
    sha256
        sha256 of that file as stored on disk (the ``.gz`` itself when gzipped).
    """

    def __init__(self, parsed: _ParsedObo, *, source_path: str | None = None, sha256: str | None = None) -> None:
        self._names = parsed.names
        self._stars = parsed.stars
        self._is_a = parsed.is_a
        self._roles = parsed.roles
        self._role_comment_names = parsed.role_comment_names
        self._alt_ids = parsed.alt_ids
        self._obsolete = frozenset(parsed.obsolete)
        self._replaced_by = parsed.replaced_by
        self._relation_names = parsed.relation_names
        self._inchikey_of = parsed.inchikey_of
        self._n_terms = parsed.n_terms
        ids_by_key: dict[str, list[int]] = {}
        for number, key in parsed.inchikey_of.items():
            ids_by_key.setdefault(key, []).append(number)
        self._ids_by_key: dict[str, tuple[int, ...]] = {key: tuple(sorted(ids)) for key, ids in ids_by_key.items()}
        self._sorted_keys: list[str] = sorted(self._ids_by_key)
        self.header: dict[str, str] = dict(parsed.header)
        self.data_version: str | None = parsed.header.get("data-version")
        self.date: str | None = parsed.header.get("date")
        self.release: str = _release_from_header(parsed.header)
        self.source_path = source_path
        self.sha256 = sha256

    # ------------------------------------------------------------------ build
    @classmethod
    def from_obo(cls, path: str | Path, *, keep_is_a: bool = True, compute_sha256: bool = True) -> ChebiOboIndex:
        """Parse ``chebi_core.obo`` or ``chebi_core.obo.gz`` (gzip is detected from the file's magic bytes).

        Parameters
        ----------
        path
            The OBO file, or a directory containing one of
            :data:`CHEBI_OBO_FILENAMES`.
        keep_is_a
            Keep ``is_a`` parents (needed only for :meth:`ancestors` and
            :meth:`term`); set False to save memory.
        compute_sha256
            Hash the file for provenance.

        Raises
        ------
        FileNotFoundError
            When the file (or a known file name in the directory) is missing.
        ValueError
            When a file with many terms yields no InChIKey at all, which means
            the OBO dialect is not the one this parser understands.
        """
        source = Path(path)
        if source.is_dir():
            for filename in CHEBI_OBO_FILENAMES:
                if (source / filename).is_file():
                    source = source / filename
                    break
            else:
                raise FileNotFoundError(f"no ChEBI OBO file ({', '.join(CHEBI_OBO_FILENAMES)}) in {source}")
        if not source.is_file():
            raise FileNotFoundError(f"ChEBI OBO file not found: {source}")
        with _open_text(source) as handle:
            parsed = _parse_obo_lines(handle, keep_is_a=keep_is_a)
        if parsed.n_terms >= 1000 and not parsed.inchikey_of:
            raise ValueError(
                f"{source}: parsed {parsed.n_terms} terms but no InChIKey property "
                f"(expected one of {[p.strip() for p in _INCHIKEY_PROPERTIES]}); unsupported ChEBI OBO dialect"
            )
        index = cls(parsed, source_path=str(source), sha256=_sha256(source) if compute_sha256 else None)
        logger.info(
            "Loaded ChEBI %s from %s: %d terms, %d InChIKeys, %d with has_role",
            index.release,
            source,
            index._n_terms,
            len(index._ids_by_key),
            len(index._roles),
        )
        return index

    @classmethod
    def from_lines(cls, lines: Iterable[str], *, keep_is_a: bool = True) -> ChebiOboIndex:
        """Build an index from OBO text lines (mainly for tests and in-memory data)."""
        return cls(_parse_obo_lines(lines, keep_is_a=keep_is_a))

    # --------------------------------------------------------------- metadata
    def __len__(self) -> int:
        return self._n_terms

    def __repr__(self) -> str:
        return f"ChebiOboIndex(release={self.release!r}, terms={self._n_terms}, inchikeys={len(self._ids_by_key)})"

    @property
    def stats(self) -> dict[str, Any]:
        """Counts for provenance and sanity checks."""
        keyed = len(self._inchikey_of)
        with_role = sum(1 for number in self._inchikey_of if number in self._roles)
        return {
            "release": self.release,
            "terms": self._n_terms,
            "obsolete_terms": len(self._obsolete),
            "alt_ids": len(self._alt_ids),
            "entities_with_inchikey": keyed,
            "distinct_inchikeys": len(self._ids_by_key),
            "entities_with_inchikey_and_role": with_role,
            "terms_with_role": len(self._roles),
            "has_role_edges": sum(len(targets) for targets in self._roles.values()),
        }

    def provenance(self) -> dict[str, Any]:
        """Release, file and licence record for manifests."""
        return {
            "source": "ChEBI OBO",
            "release": self.release,
            "data_version": self.data_version,
            "date": self.date,
            "path": self.source_path,
            "sha256": self.sha256,
            "url": CHEBI_OBO_URL,
            "licence": CHEBI_LICENCE,
        }

    @property
    def relation_names(self) -> dict[str, str]:
        """``{relation_id: name}`` from the file's ``[Typedef]`` stanzas."""
        return dict(self._relation_names)

    # ------------------------------------------------------------- term access
    def _primary(self, number: int) -> int:
        return self._alt_ids.get(number, number)

    def _name(self, number: int) -> str | None:
        name = self._names.get(number)
        if name is None:
            name = self._role_comment_names.get(number)
        return name

    def resolve_id(self, chebi_id: str | int) -> str | None:
        """Return the primary ``CHEBI:<n>`` for a primary or secondary (``alt_id``) ID, or None if unknown."""
        number = _parse_id_arg(chebi_id)
        if number is None:
            return None
        number = self._primary(number)
        if number in self._names or number in self._obsolete or number in self._inchikey_of:
            return _format_id(number)
        return None

    def role_name(self, chebi_id: str | int) -> str | None:
        """Name of a term (falls back to the ``! has role <name>`` comment for targets absent from the file)."""
        number = _parse_id_arg(chebi_id)
        return None if number is None else self._name(self._primary(number))

    def _roles_of(self, number: int) -> list[dict[str, str | None]]:
        out: list[dict[str, str | None]] = []
        for target in self._roles.get(number, ()):
            primary = self._primary(target)
            out.append({"chebi_id": _format_id(primary), "name": self._name(primary)})
        return out

    def term(self, chebi_id: str | int) -> dict[str, Any] | None:
        """Everything the index holds about one term, or None when the ID is unknown.

        Secondary IDs resolve to their primary term (``requested_id`` keeps what
        was asked). ``roles`` are the direct ``has role`` targets only.
        """
        number = _parse_id_arg(chebi_id)
        if number is None:
            return None
        primary = self._primary(number)
        known = primary in self._names or primary in self._obsolete or primary in self._inchikey_of
        if not known:
            return None
        replacement = self._replaced_by.get(primary)
        return {
            "chebi_id": _format_id(primary),
            "requested_id": _format_id(number),
            "name": self._names.get(primary),
            "stars": self._stars.get(primary),
            "inchikey": self._inchikey_of.get(primary),
            "roles": self._roles_of(primary),
            "is_a": [_format_id(self._primary(parent)) for parent in self._is_a.get(primary, ())],
            "obsolete": primary in self._obsolete,
            "replaced_by": _format_id(replacement) if replacement is not None else None,
        }

    def ancestors(self, chebi_id: str | int) -> list[str]:
        """All ``is_a`` ancestors of a term, nearest first (breadth-first). Not used for role lookup."""
        number = _parse_id_arg(chebi_id)
        if number is None:
            return []
        seen: dict[int, None] = {}
        frontier = [self._primary(number)]
        while frontier:
            nxt: list[int] = []
            for node in frontier:
                for parent in self._is_a.get(node, ()):
                    parent = self._primary(parent)
                    if parent not in seen:
                        seen[parent] = None
                        nxt.append(parent)
            frontier = nxt
        return [_format_id(n) for n in seen]

    # ------------------------------------------------------------------ lookup
    def _block_keys(self, block: str) -> list[str]:
        keys = self._sorted_keys
        start = bisect_left(keys, block)
        out: list[str] = []
        for position in range(start, len(keys)):
            key = keys[position]
            if not key.startswith(block):
                break
            out.append(key)
        return out

    def _candidate(self, number: int, key: str, tier: str) -> dict[str, Any]:
        return {
            "chebi_id": _format_id(number),
            "name": self._names.get(number),
            "stars": self._stars.get(number),
            "inchikey": key,
            "match_tier": tier,
            "roles": self._roles_of(number),
        }

    def _empty(self, status: str, query: str | None) -> dict[str, Any]:
        return {
            "status": status,
            "chebi_id": None,
            "name": None,
            "stars": None,
            "match_tier": None,
            "inchikey": None,
            "query_inchikey": query,
            "roles": [],
            "candidates": [],
            "connectivity": None,
            "release": self.release,
        }

    def _tier(self, keys: list[str], tier: str, query: str) -> dict[str, Any] | None:
        """Result for the entities carrying ``keys`` (one key = a match, several = ambiguous)."""
        if not keys:
            return None
        candidates = [self._candidate(number, key, tier) for key in keys for number in self._ids_by_key[key]]
        result = self._empty("no_chebi_entity", query)
        result["match_tier"] = tier
        result["primary_annotation_eligible"] = tier == "exact"
        result["identity_evidence"] = "exact_standard_inchikey" if tier == "exact" else "connectivity_only_stereochemistry_unverified"
        result["role_interpretation"] = "Database-asserted has_role, including investigational and repurposing uses; not necessarily an established mechanism or approved indication"
        result["candidates"] = candidates
        if len(keys) > 1:
            result["status"] = "ambiguous_connectivity"
            return result
        ranked = sorted(candidates, key=lambda c: (-(c["stars"] or 0), int(c["chebi_id"][6:])))
        best = ranked[0]
        roles: dict[str, dict[str, Any]] = {}
        for candidate in ranked:
            for role in candidate["roles"]:
                roles.setdefault(role["chebi_id"], role)
        result.update(
            status="success" if roles else "matched_no_role",
            chebi_id=best["chebi_id"],
            name=best["name"],
            stars=best["stars"],
            inchikey=keys[0],
            roles=list(roles.values()),
            candidates=ranked,
        )
        return result

    def lookup(self, inchikey: str | None) -> dict[str, Any]:
        """Look a structure up by standard InChIKey.

        Returns
        -------
        dict
            ``status`` (see :data:`LOOKUP_STATUSES`), ``chebi_id``, ``name``,
            ``stars``, ``match_tier`` (``"exact"``, ``"connectivity"`` or None),
            ``inchikey`` (the matched ChEBI key), ``query_inchikey``, ``roles``
            (``[{"chebi_id", "name"}]``, direct ``has role`` only), ``candidates``
            (every entity considered at the deciding tier, best first, each with
            its own roles), ``connectivity`` (the supplementary connectivity
            tier over the block's other keys when the exact tier matched, else
            None) and ``release``.
        """
        query = normalise_inchikey(inchikey)
        if query is None:
            result = self._empty("invalid_structure", None)
            result["error"] = f"not a valid InChIKey: {inchikey!r}"
            return result
        block = query[:14]
        exact = self._tier([query], "exact", query) if query in self._ids_by_key else None
        if exact is not None:
            others = [key for key in self._block_keys(block) if key != query]
            exact["connectivity"] = self._tier(others, "connectivity", query)
            return exact
        connectivity = self._tier(self._block_keys(block), "connectivity", query)
        return connectivity if connectivity is not None else self._empty("no_chebi_entity", query)

    def lookup_many(self, inchikeys: Iterable[str | None]) -> dict[str | None, dict[str, Any]]:
        """``{inchikey: lookup(inchikey)}`` for several keys (duplicates are looked up once)."""
        return {key: self.lookup(key) for key in dict.fromkeys(inchikeys)}

    def lookup_structure(self, identifier: str | None) -> dict[str, Any]:
        """Like :meth:`lookup`, but also accepts a SMILES or ``InChI=`` string (converted with RDKit).

        A string that is neither an InChIKey nor a structure RDKit can parse
        gives ``invalid_structure``. RDKit is only imported for non-InChIKey input.
        """
        if normalise_inchikey(identifier) is not None:
            return self.lookup(identifier)
        if not isinstance(identifier, str) or not identifier.strip():
            result = self._empty("invalid_structure", None)
            result["error"] = f"empty or non-string structure: {identifier!r}"
            return result
        key = inchikey_from_structure(identifier)
        if key is None:
            result = self._empty("invalid_structure", None)
            result["error"] = f"RDKit could not parse the structure: {identifier!r}"
            return result
        return self.lookup(key)


__all__ = [
    "CHEBI_LICENCE",
    "CHEBI_OBO_FILENAMES",
    "CHEBI_OBO_URL",
    "HAS_ROLE",
    "LOOKUP_STATUSES",
    "ChebiOboIndex",
    "inchikey_from_structure",
    "normalise_inchikey",
]
