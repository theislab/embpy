"""The scPerturb registry has to name files that actually exist.

Every one of the thirteen original entries carried a fabricated filename -- a
``_rna.h5ad`` suffix and an invented GSE accession, e.g.
``DatlingerBock2017_GSE92872_rna.h5ad`` where the Zenodo record holds
``DatlingerBock2017.h5ad``. Nothing matched, so ``load_scperturb`` raised
``HTTPError: 404`` for every dataset in the registry and had never once
succeeded.

The 404 is the whole problem: it names a URL, so it reads as Zenodo being down
rather than as embpy asking for a file that has never existed. These tests pin
the shape of the registry offline, and one opt-in test checks the names against
the live record.
"""

from __future__ import annotations

import json
import os
import urllib.request
from unittest.mock import patch

import pytest

from embpy.pp import list_scperturb_datasets, scperturb_info
from embpy.pp.scperturb_handler import (
    _ZENODO_RNASEQ_RECORD,
    _download_file,
    _zenodo_download_url,
)


class TestRegistryShape:
    def test_no_fabricated_rna_suffix(self):
        """The tell of the original bug: a `_rna.h5ad` suffix on every entry.

        No file in the scPerturb record uses it -- the RNA modality is either
        implicit or spelled `_RNA`.
        """
        for name in list_scperturb_datasets():
            filename = scperturb_info(name).filename
            assert not filename.endswith("_rna.h5ad"), (name, filename)

    def test_every_entry_names_an_h5ad(self):
        for name in list_scperturb_datasets():
            assert scperturb_info(name).filename.endswith(".h5ad")

    def test_filenames_are_unique(self):
        files = [scperturb_info(n).filename for n in list_scperturb_datasets()]
        assert len(files) == len(set(files))

    def test_the_phantom_entries_are_gone(self):
        """Two entries named datasets absent from the record entirely.

        `PapaleandrouSchraivogel2019` duplicated `SchraivogelSteinmetz2020`
        under a garbled author name, and `UrsuHein2022` is not in this record.
        Neither had a file to point at, so both could only ever 404.
        """
        names = set(list_scperturb_datasets())
        assert "PapaleandrouSchraivogel2019" not in names
        assert "UrsuHein2022" not in names
        # TianLuo2019 is not a scPerturb dataset either; its reference was the
        # Tian *Kampmann* 2019 Neuron paper, so the key was garbled.
        assert "TianLuo2019" not in names
        assert "TianKampmann2019" in names

    def test_a_chemical_screen_is_reachable(self):
        """The registry was entirely genetic; sci-Plex is the compound screen."""
        chemical = [
            n for n in list_scperturb_datasets()
            if scperturb_info(n).perturbation_type == "chemical"
        ]
        assert chemical
        assert "SrivatsanTrapnell2020_sciplex3" in chemical


class TestHelpful404:
    def test_a_missing_file_says_so_instead_of_raising_httperror(self, tmp_path):
        """A stale registry entry must not look like a network outage."""
        import urllib.error

        err = urllib.error.HTTPError(
            "http://zenodo.example/x", 404, "NOT FOUND", {}, None,
        )
        record = {"files": [{"key": "DatlingerBock2017.h5ad"}]}

        with patch("urllib.request.urlretrieve", side_effect=err), patch(
            "urllib.request.urlopen"
        ) as urlopen:
            urlopen.return_value.__enter__.return_value = _FakeResponse(record)
            with pytest.raises(FileNotFoundError) as excinfo:
                _download_file(
                    "http://zenodo.example/x",
                    tmp_path / "DatlingerBock2017_GSE92872_rna.h5ad",
                    record_id="7041849",
                )

        message = str(excinfo.value)
        assert "no file named" in message
        # It should point at the real name rather than only saying "not found".
        assert "DatlingerBock2017.h5ad" in message
        assert "not a network problem" in message

    def test_a_non_404_is_re_raised_untouched(self, tmp_path):
        """A real outage must stay a real outage."""
        import urllib.error

        err = urllib.error.HTTPError(
            "http://zenodo.example/x", 503, "SERVICE UNAVAILABLE", {}, None,
        )
        with patch("urllib.request.urlretrieve", side_effect=err):
            with pytest.raises(urllib.error.HTTPError):
                _download_file(
                    "http://zenodo.example/x", tmp_path / "x.h5ad",
                    record_id="7041849",
                )

    def test_partial_file_is_cleaned_up(self, tmp_path):
        import urllib.error

        dest = tmp_path / "x.h5ad"
        part = dest.with_suffix(".h5ad.part")

        def _fail(url, path):
            open(path, "wb").write(b"partial")
            raise urllib.error.HTTPError(url, 503, "boom", {}, None)

        with patch("urllib.request.urlretrieve", side_effect=_fail):
            with pytest.raises(urllib.error.HTTPError):
                _download_file("http://x/y", dest)
        assert not part.exists()
        assert not dest.exists()


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()


def test_url_shape():
    url = _zenodo_download_url("7041849", "DatlingerBock2017.h5ad")
    assert url == (
        "https://zenodo.org/records/7041849/files/"
        "DatlingerBock2017.h5ad?download=1"
    )


@pytest.mark.skipif(
    not os.environ.get("EMBPY_TEST_NETWORK"),
    reason="hits zenodo.org; set EMBPY_TEST_NETWORK=1 to run",
)
def test_every_filename_exists_in_the_live_record():
    """The check that would have caught the original bug.

    Opt-in because it needs network, but it is the only test that verifies the
    registry against reality rather than against itself.
    """
    with urllib.request.urlopen(
        f"https://zenodo.org/api/records/{_ZENODO_RNASEQ_RECORD}", timeout=60
    ) as resp:
        available = {f["key"] for f in json.load(resp)["files"]}

    missing = {
        name: scperturb_info(name).filename
        for name in list_scperturb_datasets()
        if scperturb_info(name).filename not in available
    }
    assert not missing, f"registry names files absent from Zenodo: {missing}"
