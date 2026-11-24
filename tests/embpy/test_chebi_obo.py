"""Tests for the offline ChEBI OBO role index (plan test T13).

The fixture ``fixtures/chebi/chebi_core_257_mini.obo`` holds real, unedited
stanzas cut on 2026-11-11 from ChEBI release 257
(https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi_core.obo.gz,
sha256 68bccfe3865a260cd94bbc2ba6f8fabe7175191bee7a1ce7676a8a6e02c03ea3,
CC BY 4.0): the file header, aspirin CHEBI:15365 and its conjugate base
CHEBI:13719, the two tamoxifen entries CHEBI:92766 (2-star, no role) and
CHEBI:41774 (3-star), the three vinorelbine-block stereo variants
CHEBI:480999 / 93786 / 94502, the valpromide exact-key duplicates CHEBI:74562
(3-star) and CHEBI:46316 (1-star), two role terms (CHEBI:35475, CHEBI:35610),
an escaped-name term (CHEBI:102167), the one obsolete stanza CHEBI:27189 and
three [Typedef] stanzas.

Query keys marked "hypothetical" below are lookup inputs (stereo or protonation
variants of a real block that ChEBI does not hold), not payload content.

``test_real_release`` runs against the full file only when ``EMBPY_CHEBI_OBO``
points at it.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import time
from pathlib import Path

import pytest

from embpy.resources.molecule.chebi_obo import (
    LOOKUP_STATUSES,
    ChebiOboIndex,
    normalise_inchikey,
)

FIXTURE = Path(__file__).parent / "fixtures" / "chebi" / "chebi_core_257_mini.obo"

ASPIRIN = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
TAMOXIFEN_STEREO_FREE = "NKANXQFJJICGDU-UHFFFAOYSA-N"
TAMOXIFEN_Z = "NKANXQFJJICGDU-QPLCGJKRSA-N"
VALPROMIDE = "OMOMUFTZPTXCHP-UHFFFAOYSA-N"
VINORELBINE_BLOCK_UNLISTED = "GBABOYUKABKIAF-ZZZZZZZZSA-N"  # hypothetical stereo variant
IMATINIB = "KTUFNOKKBVMGRW-UHFFFAOYSA-N"  # real key, not in the fixture
CONTRACT_KEYS = {"status", "chebi_id", "name", "stars", "match_tier", "inchikey", "roles", "candidates", "release"}


@pytest.fixture(scope="module")
def index() -> ChebiOboIndex:
    return ChebiOboIndex.from_obo(FIXTURE)


def role_ids(result):
    return [role["chebi_id"] for role in result["roles"]]


def test_header_release_and_counts(index):
    assert index.release == "ChEBI_257"
    assert index.data_version == "chebi/257"
    assert index.date == "05:11:2026 18:04"
    assert index.header["license"] == "https://creativecommons.org/licenses/by/4.0/"
    assert len(index) == 13
    stats = index.stats
    assert stats["obsolete_terms"] == 1
    assert stats["entities_with_inchikey"] == 10
    assert stats["distinct_inchikeys"] == 9
    assert stats["alt_ids"] == 7
    assert stats["has_role_edges"] == 35 + 15 + 2 + 4
    assert index.relation_names["RO:0000087"] == "has role"
    provenance = index.provenance()
    assert provenance["licence"] == "CC BY 4.0"
    assert provenance["sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()


def test_exact_match_with_roles_is_success(index):
    result = index.lookup(ASPIRIN)
    assert CONTRACT_KEYS <= set(result)
    assert result["status"] == "success"
    assert result["chebi_id"] == "CHEBI:15365"
    assert result["name"] == "acetylsalicylic acid"
    assert result["stars"] == 3
    assert result["match_tier"] == "exact"
    assert result["inchikey"] == ASPIRIN
    assert result["release"] == "ChEBI_257"
    assert len(result["roles"]) == 35
    assert {"chebi_id": "CHEBI:35475", "name": "non-steroidal anti-inflammatory drug"} in result["roles"]
    # Role term absent from the file: the name falls back to the "! has role ..." comment.
    assert {"chebi_id": "CHEBI:88188", "name": "drug allergen"} in result["roles"]
    # Only RO:0000087 targets become roles; conjugate-acid and functional-parent edges do not.
    assert "CHEBI:13719" not in role_ids(result)
    assert "CHEBI:16914" not in role_ids(result)
    # The block's other key (the conjugate base, -M) is the supplementary connectivity tier.
    conn = result["connectivity"]
    assert conn["status"] == "matched_no_role"
    assert conn["match_tier"] == "connectivity"
    assert conn["chebi_id"] == "CHEBI:13719"
    assert conn["inchikey"] == "BSYNRYMUTXBXSQ-UHFFFAOYSA-M"


def test_two_star_entry_without_role_is_matched_no_role(index):
    result = index.lookup(TAMOXIFEN_STEREO_FREE)
    assert result["status"] == "matched_no_role"
    assert result["chebi_id"] == "CHEBI:92766"
    assert result["stars"] == 2
    assert result["match_tier"] == "exact"
    assert result["roles"] == []
    # Roles are not inherited through is_a (CHEBI:92766 is_a stilbenoid).
    assert index.term("CHEBI:92766")["is_a"] == ["CHEBI:26776"]
    # The curated stereo form is reachable only through the flagged connectivity tier.
    conn = result["connectivity"]
    assert conn["status"] == "success"
    assert conn["match_tier"] == "connectivity"
    assert conn["chebi_id"] == "CHEBI:41774"
    assert conn["name"] == "tamoxifen"
    assert conn["stars"] == 3
    assert conn["inchikey"] == TAMOXIFEN_Z
    assert {"chebi_id": "CHEBI:50792", "name": "estrogen receptor antagonist"} in conn["roles"]
    assert len(conn["roles"]) == 15


def test_connectivity_block_with_three_stereo_variants_is_ambiguous(index):
    result = index.lookup(VINORELBINE_BLOCK_UNLISTED)
    assert result["status"] == "ambiguous_connectivity"
    assert result["match_tier"] == "connectivity"
    assert result["chebi_id"] is None
    assert result["roles"] == []
    assert result["connectivity"] is None
    assert sorted(c["chebi_id"] for c in result["candidates"]) == ["CHEBI:480999", "CHEBI:93786", "CHEBI:94502"]
    assert len({c["inchikey"] for c in result["candidates"]}) == 3
    vinorelbine = next(c for c in result["candidates"] if c["chebi_id"] == "CHEBI:480999")
    assert vinorelbine["stars"] == 3
    assert {"chebi_id": "CHEBI:35610", "name": "antineoplastic agent"} in vinorelbine["roles"]


def test_exact_hit_keeps_exact_tier_and_flags_ambiguous_connectivity(index):
    result = index.lookup("GBABOYUKABKIAF-UHFFFAOYSA-N")
    assert result["status"] == "matched_no_role"
    assert result["match_tier"] == "exact"
    assert result["chebi_id"] == "CHEBI:93786"
    assert result["connectivity"]["status"] == "ambiguous_connectivity"
    assert {c["chebi_id"] for c in result["connectivity"]["candidates"]} == {"CHEBI:480999", "CHEBI:94502"}


def test_exact_key_shared_by_two_entities_prefers_stars_and_unions_roles(index):
    result = index.lookup(VALPROMIDE)
    assert result["status"] == "success"
    assert result["match_tier"] == "exact"
    assert result["chebi_id"] == "CHEBI:74562"  # 3-star beats the lower-numbered 1-star CHEBI:46316
    assert result["name"] == "valpromide"
    assert [c["chebi_id"] for c in result["candidates"]] == ["CHEBI:74562", "CHEBI:46316"]
    assert role_ids(result) == ["CHEBI:176497", "CHEBI:25212", "CHEBI:35623", "CHEBI:50905"]
    assert result["connectivity"] is None


def test_single_connectivity_key_is_a_flagged_match(index):
    result = index.lookup("OMOMUFTZPTXCHP-UHFFFAOYSA-M")  # hypothetical protonation variant
    assert result["status"] == "success"
    assert result["match_tier"] == "connectivity"
    assert result["chebi_id"] == "CHEBI:74562"
    assert result["inchikey"] == VALPROMIDE
    assert result["query_inchikey"] == "OMOMUFTZPTXCHP-UHFFFAOYSA-M"
    assert result["connectivity"] is None


def test_absent_key_is_no_chebi_entity(index):
    result = index.lookup(IMATINIB)
    assert result["status"] == "no_chebi_entity"
    assert result["chebi_id"] is None
    assert result["match_tier"] is None
    assert result["candidates"] == []
    assert result["roles"] == []


@pytest.mark.parametrize("bad", [None, "", "aspirin", "BSYNRYMUTXBXSQ-UHFFFAOYSA", "BSYNRYMUTXBXSQ_UHFFFAOYSA-N", 42])
def test_malformed_inchikey_is_invalid_structure(index, bad):
    result = index.lookup(bad)
    assert result["status"] == "invalid_structure"
    assert result["chebi_id"] is None
    assert "error" in result


def test_inchikey_normalisation(index):
    assert normalise_inchikey(" inchikey=bsynrymutxbxsq-uhfffaoysa-n ") == ASPIRIN
    assert index.lookup("InChIKey=" + ASPIRIN.lower())["chebi_id"] == "CHEBI:15365"
    queries = (ASPIRIN, TAMOXIFEN_STEREO_FREE, VINORELBINE_BLOCK_UNLISTED, IMATINIB, "x")
    assert {index.lookup(key)["status"] for key in queries} <= set(LOOKUP_STATUSES)


def test_alt_ids_obsolete_names_and_is_a(index):
    assert index.resolve_id("CHEBI:22188") == "CHEBI:15365"
    assert index.resolve_id("CHEBI_9396") == "CHEBI:41774"
    assert index.resolve_id(76925) is None  # alt_id of a term that is not in the mini file
    assert index.resolve_id("CHEBI:999999999") is None
    assert index.resolve_id("not an id") is None
    term = index.term("CHEBI:2890")
    assert term["chebi_id"] == "CHEBI:15365"
    assert term["requested_id"] == "CHEBI:2890"
    assert term["inchikey"] == ASPIRIN
    assert term["is_a"] == ["CHEBI:140310", "CHEBI:22723", "CHEBI:26596"]
    obsolete = index.term("CHEBI:27189")
    assert obsolete["obsolete"] is True
    assert obsolete["name"] is None
    assert index.term("CHEBI:102167")["name"] == (
        "N-{4-[2-ethyl-1-(1,2,4-triazol-1-yl)butyl]phenyl}-1,3-benzothiazol-2-amine"
    )
    assert index.role_name("CHEBI:35475") == "non-steroidal anti-inflammatory drug"
    assert index.ancestors("CHEBI:74562") == ["CHEBI:29348"]


def test_gzip_and_directory_inputs_match_plain_file(index, tmp_path):
    gz_path = tmp_path / "chebi_core.obo.gz"
    with open(FIXTURE, "rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    from_gz = ChebiOboIndex.from_obo(gz_path)
    from_dir = ChebiOboIndex.from_obo(tmp_path)
    assert from_gz.stats == index.stats == from_dir.stats
    assert from_gz.sha256 == hashlib.sha256(gz_path.read_bytes()).hexdigest()
    assert from_gz.lookup(TAMOXIFEN_STEREO_FREE) == index.lookup(TAMOXIFEN_STEREO_FREE)
    with pytest.raises(FileNotFoundError):
        ChebiOboIndex.from_obo(tmp_path / "missing.obo.gz")
    with pytest.raises(FileNotFoundError):
        ChebiOboIndex.from_obo(tmp_path / "empty_dir_that_does_not_exist")


def test_unsupported_dialect_without_inchikeys_raises(tmp_path):
    lines = ["format-version: 1.2\n", "data-version: chebi/1\n", "\n"]
    for number in range(1, 1201):
        lines += ["[Term]\n", f"id: CHEBI:{number}\n", f"name: term {number}\n", "subset: 2:STAR\n", "\n"]
    path = tmp_path / "no_keys.obo"
    path.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match="no InChIKey"):
        ChebiOboIndex.from_obo(path)


def test_keep_is_a_false_drops_only_the_hierarchy():
    slim = ChebiOboIndex.from_obo(FIXTURE, keep_is_a=False, compute_sha256=False)
    assert slim.sha256 is None
    assert slim.term("CHEBI:15365")["is_a"] == []
    assert slim.lookup(ASPIRIN)["status"] == "success"


def test_structures_via_rdkit(index):
    pytest.importorskip("rdkit")
    assert index.lookup_structure(ASPIRIN_SMILES)["chebi_id"] == "CHEBI:15365"
    assert index.lookup_structure(ASPIRIN)["chebi_id"] == "CHEBI:15365"
    invalid = index.lookup_structure("C1CC(")
    assert invalid["status"] == "invalid_structure"
    assert index.lookup_structure("")["status"] == "invalid_structure"
    assert index.lookup_structure(None)["status"] == "invalid_structure"


def test_annotator_lookup_chebi_roles_uses_the_index(index):
    """Contract: MoleculeAnnotator.lookup_chebi_roles(index=...) is the offline path."""
    pytest.importorskip("rdkit")
    from embpy.resources.molecule.annotator import MoleculeAnnotator

    if not hasattr(MoleculeAnnotator, "lookup_chebi_roles"):
        pytest.skip("MoleculeAnnotator.lookup_chebi_roles not implemented in this tree yet")
    annotator = MoleculeAnnotator(rate_limit_delay=0)
    assert annotator.lookup_chebi_roles(ASPIRIN_SMILES, index=index)["chebi_id"] == "CHEBI:15365"
    assert annotator.lookup_chebi_roles(inchikey=TAMOXIFEN_STEREO_FREE, index=index)["status"] == "matched_no_role"
    assert annotator.lookup_chebi_roles("C1CC(", index=index)["status"] == "invalid_structure"


def test_real_release():
    """Opt-in: EMBPY_CHEBI_OBO=/path/to/chebi_core.obo.gz (release 257)."""
    path = os.environ.get("EMBPY_CHEBI_OBO")
    if not path or not Path(path).exists():
        pytest.skip("set EMBPY_CHEBI_OBO to a chebi_core.obo(.gz) file to run")
    started = time.perf_counter()
    real = ChebiOboIndex.from_obo(path)
    elapsed = time.perf_counter() - started
    print(f"loaded {real!r} in {elapsed:.1f} s; stats {real.stats}")
    assert elapsed < 60
    assert real.stats["terms"] > 200_000
    aspirin = real.lookup(ASPIRIN)
    assert aspirin["status"] == "success" and aspirin["chebi_id"] == "CHEBI:15365"
    tamoxifen = real.lookup(TAMOXIFEN_STEREO_FREE)
    assert tamoxifen["status"] == "matched_no_role" and tamoxifen["chebi_id"] == "CHEBI:92766"
    assert tamoxifen["connectivity"]["chebi_id"] == "CHEBI:41774"
    if real.release == "ChEBI_257":
        assert real.stats["terms"] == 219_121
        assert real.stats["entities_with_inchikey"] == 190_183
        assert real.stats["has_role_edges"] == 59_168
