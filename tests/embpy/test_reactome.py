"""Tests for embpy.resources.gene.reactome -- ReactomeIndex (plan test T15).

``fixtures/gene/reactome98/`` holds unmodified lines of the Reactome release 98
files (see ``fixtures/gene/provenance.json``): every Homo sapiens
NCBI2Reactome row of six genes, Gallus gallus and Mus musculus rows, the
ancestor edges of their pathways, and the 29 human top-level pathways that
``ContentService/data/pathways/top/9606`` returned.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import urllib.request
from pathlib import Path

import pytest

from embpy.resources.gene.reactome import (
    REACTOME_FILES,
    REACTOME_RELEASE,
    REACTOME_VERSION_URL,
    ReactomeIndex,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gene" / "reactome98"
TOP_LEVEL_IDS = {p["stId"] for p in json.loads((FIXTURE_DIR / "top_level_9606.json").read_text())}


@pytest.fixture(scope="module")
def index() -> ReactomeIndex:
    return ReactomeIndex.from_dir(FIXTURE_DIR, release="98")


def _copy_release(tmp_path: Path) -> Path:
    out = tmp_path / "reactome"
    out.mkdir()
    for name in REACTOME_FILES:
        shutil.copy(FIXTURE_DIR / name, out / name)
    return out


class TestParsing:
    def test_fixture_really_contains_non_human_rows(self):
        species = {line.rstrip("\n").split("\t")[5] for line in (FIXTURE_DIR / "NCBI2Reactome.txt").open()}
        assert {"Homo sapiens", "Gallus gallus", "Mus musculus"} <= species

    def test_non_human_rows_are_dropped(self, index):
        # 1253480 has 2 Homo sapiens rows (one pathway) and 4 Gallus gallus rows.
        hit = index.lookup("1253480")
        assert hit["status"] == "success"
        assert [p["id"] for p in hit["pathways"]] == ["R-HSA-844623"]
        # 100017 only has Mus musculus rows.
        assert index.lookup("100017")["status"] == "not_in_reactome"

    def test_evidence_duplicates_collapse(self, index):
        # 506 (ATP5F1B) has 6 rows: R-HSA-1268020 once as IEA and once as TAS.
        assert len(index.lookup("506")["pathways"]) == 5
        assert len(index.lookup("7157")["pathways"]) == 46
        assert len(index.lookup("1956")["pathways"]) == 37

    def test_names_are_stripped(self, index):
        raw = {
            line.split("\t")[0]: line.rstrip("\n").split("\t")[1]
            for line in (FIXTURE_DIR / "ReactomePathways.txt").open()
        }
        assert raw["R-HSA-69560"] == "Transcriptional activation of p53 responsive genes  "
        assert index.pathway_name("R-HSA-69560") == "Transcriptional activation of p53 responsive genes"
        names = {p["id"]: p["name"] for p in index.lookup("7157")["pathways"]}
        # Outer whitespace goes, the text itself (double spaces included) is Reactome's.
        assert names["R-HSA-69895"] == "Transcriptional  activation of  cell cycle inhibitor p21"
        assert all(n == n.strip() for n in names.values())

    def test_lookup_shape(self, index):
        hit = index.lookup(7453)  # int is accepted
        assert hit == {
            "status": "success",
            "entrez_id": "7453",
            "pathways": [{"id": "R-HSA-379716", "name": "Cytosolic tRNA aminoacylation"}],
            "top_level": [{"id": "R-HSA-392499", "name": "Metabolism of proteins"}],
            "release": "98",
        }


class TestHierarchy:
    def test_roots_are_the_29_human_top_level_pathways(self, index):
        roots = {p["id"] for p in index.top_level_pathways}
        assert len(TOP_LEVEL_IDS) == 29
        assert roots == TOP_LEVEL_IDS
        assert "R-HSA-1643685" in roots  # Disease is a top-level category too

    def test_walk_up_through_relation(self, index):
        assert {p["id"] for p in index.lookup("506")["top_level"]} == {
            "R-HSA-9609507", "R-HSA-1430728", "R-HSA-1852241", "R-HSA-392499",
        }
        assert {p["name"] for p in index.lookup("1565")["top_level"]} == {"Metabolism", "Drug ADME"}
        for gene in ("7157", "1956", "1565", "506", "7453", "1253480"):
            top = {p["id"] for p in index.lookup(gene)["top_level"]}
            assert top and top <= TOP_LEVEL_IDS

    def test_top_level_of_a_mid_level_pathway(self, index):
        # Formation of ATP by chemiosmotic coupling -> ... -> Metabolism
        assert index.top_level_of("R-HSA-163210") == [{"id": "R-HSA-1430728", "name": "Metabolism"}]

    def test_pathway_missing_from_pathways_file_is_not_promoted(self, tmp_path):
        out = _copy_release(tmp_path)
        pathways = (out / "ReactomePathways.txt").read_text().splitlines(keepends=True)
        (out / "ReactomePathways.txt").write_text(
            "".join(line for line in pathways if not line.startswith("R-HSA-379716\t"))
        )
        relation = (out / "ReactomePathwaysRelation.txt").read_text().splitlines(keepends=True)
        (out / "ReactomePathwaysRelation.txt").write_text(
            "".join(line for line in relation if not line.rstrip("\n").endswith("\tR-HSA-379716"))
        )
        hit = ReactomeIndex.from_dir(out).lookup("7453")
        assert hit["pathways"] == [{"id": "R-HSA-379716", "name": "Cytosolic tRNA aminoacylation"}]
        assert hit["top_level"] == []


class TestJoinKeys:
    def test_ensembl_only_or_missing_ids_are_not_joined(self, index):
        for key in ("ENSG00000211451", None, "", "114814"):
            hit = index.lookup(key)
            assert hit["status"] == "not_in_reactome"
            assert hit["pathways"] == [] and hit["top_level"] == []
        assert "7157" in index and 7157 in index and "ENSG00000141510" not in index

    def test_lookup_many(self, index):
        out = index.lookup_many(["7157", "114814"])
        assert out["7157"]["status"] == "success"
        assert out["114814"]["status"] == "not_in_reactome"


class TestFromDir:
    def test_release_from_version_json(self, tmp_path):
        out = _copy_release(tmp_path)
        (out / "version.json").write_text(json.dumps({"release": 98, "fetched_at": "2026-11-11"}))
        assert ReactomeIndex.from_dir(out).release == "98"
        (out / "version.json").write_text("98")
        assert ReactomeIndex.from_dir(out).release == "98"

    def test_release_is_none_without_version_json(self, tmp_path):
        out = _copy_release(tmp_path)
        assert ReactomeIndex.from_dir(out).release is None

    def test_gzip_files_are_read(self, tmp_path, index):
        out = tmp_path / "gz"
        out.mkdir()
        for name in REACTOME_FILES:
            with (FIXTURE_DIR / name).open("rb") as src, gzip.open(out / f"{name}.gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
        gz = ReactomeIndex.from_dir(out, release="98")
        assert gz.lookup("7157") == index.lookup("7157")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ReactomeIndex.from_dir(tmp_path)

    def test_files_and_urls_are_pinned_to_the_release(self):
        assert set(REACTOME_FILES) == {"NCBI2Reactome.txt", "ReactomePathwaysRelation.txt", "ReactomePathways.txt"}
        for name, url in REACTOME_FILES.items():
            assert url == f"https://download.reactome.org/{REACTOME_RELEASE}/{name}"


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("EMBPY_NETWORK_TESTS"), reason="set EMBPY_NETWORK_TESTS=1 to run")
def test_live_release_files_exist():
    for url in REACTOME_FILES.values():
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https URL
            assert resp.status == 200
    with urllib.request.urlopen(REACTOME_VERSION_URL, timeout=30) as resp:  # noqa: S310
        assert resp.read().decode().strip().isdigit()


class _Download:
    """Stands in for ``requests.get``: serves the fixture files, optionally broken."""

    def __init__(self, *, short=False, fail=False):
        self.short, self.fail = short, fail
        self.urls: list[str] = []

    def __call__(self, url, stream=False, timeout=None):
        import requests

        self.urls.append(url)
        if self.fail:
            raise requests.ConnectionError("connection refused")
        body = (FIXTURE_DIR / url.rsplit("/", 1)[1]).read_bytes()
        return _Response(body, length=len(body) + (100 if self.short else 0))


class _Response:
    def __init__(self, body: bytes, length: int):
        self.body = body
        self.status_code = 200
        self.headers = {"Content-Length": str(length)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1):
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start:start + chunk_size]


class TestDownload:
    @pytest.fixture(autouse=True)
    def _fresh_index_cache(self, monkeypatch):
        from embpy.resources.gene import reactome

        monkeypatch.setattr(reactome, "_INDEXES", {})

    def test_cache_dir_follows_embpy_cache(self, monkeypatch, tmp_path):
        from embpy.resources.gene.reactome import reactome_cache_dir

        monkeypatch.setenv("EMBPY_CACHE", str(tmp_path))
        assert reactome_cache_dir() == tmp_path / "reactome" / REACTOME_RELEASE

    def test_first_use_downloads_then_reuses_the_files(self, monkeypatch, tmp_path):
        import requests

        from embpy.resources.gene import reactome

        fake = _Download()
        monkeypatch.setattr(requests, "get", fake)
        folder = tmp_path / "cache"
        index = reactome.load_reactome_index(folder)
        assert sorted(fake.urls) == sorted(REACTOME_FILES.values())
        assert {p.name for p in folder.iterdir()} == {*REACTOME_FILES, "version.json"}
        assert index.release == REACTOME_RELEASE
        hit = index.lookup(7157)
        assert {"id": "R-HSA-111448", "name": "Activation of NOXA and translocation to mitochondria"} in hit["pathways"]

        assert reactome.load_reactome_index(folder) is index  # parsed once per process
        monkeypatch.setattr(reactome, "_INDEXES", {})
        reactome.load_reactome_index(folder)  # a new process: read from disk, no download
        assert len(fake.urls) == len(REACTOME_FILES)

    def test_short_download_is_not_kept(self, monkeypatch, tmp_path):
        import requests

        from embpy.resources.gene.reactome import ReactomeUnavailableError, download_reactome_files

        monkeypatch.setattr(requests, "get", _Download(short=True))
        with pytest.raises(ReactomeUnavailableError, match="incomplete download"):
            download_reactome_files(tmp_path)
        assert not list(tmp_path.iterdir())

    def test_no_network_raises_reactome_unavailable(self, monkeypatch, tmp_path):
        import requests

        from embpy.resources.gene.reactome import ReactomeUnavailableError, load_reactome_index

        monkeypatch.setattr(requests, "get", _Download(fail=True))
        with pytest.raises(ReactomeUnavailableError, match="^Couldn't reach Reactome"):
            load_reactome_index(tmp_path)
        assert not list(tmp_path.iterdir())
