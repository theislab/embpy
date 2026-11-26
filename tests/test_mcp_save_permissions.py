"""Permissions of the .h5ad files the MCP tools save (POSIX; ACL checks where setfacl works).

A saved file must get exactly what any program creating a file in that folder gets
(mode, and the folder's default ACL), and overwriting a file must keep the access
that was granted on that file.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest
from anndata import AnnData

from embpy.mcp.tools import _save_anndata_safely

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX permissions and ACLs")

COLLABORATOR = "65534"  # a uid other than ours, standing in for a collaborator


def _adata(n: int = 3) -> AnnData:
    return AnnData(obs={"a": list(range(n))})


def _require_acls(folder) -> None:
    """Skip unless setfacl/getfacl exist and the file system accepts ACLs."""
    if not (shutil.which("setfacl") and shutil.which("getfacl")):
        pytest.skip("setfacl/getfacl not installed")
    probe = folder / "acl_probe"
    probe.write_text("x")
    done = subprocess.run(["setfacl", "-m", f"u:{COLLABORATOR}:rw", str(probe)], capture_output=True)
    probe.unlink()
    if done.returncode:
        pytest.skip(f"ACLs not supported here: {done.stderr.decode(errors='replace').strip()}")


def _acl(path) -> list[str]:
    """getfacl entries, numeric, including '#effective' notes (the mask's effect)."""
    out = subprocess.run(
        ["getfacl", "--omit-header", "--numeric", str(path)], capture_output=True, text=True, check=True
    ).stdout
    return sorted(" ".join(line.split()) for line in out.splitlines() if line.strip())


def test_new_file_gets_the_mode_of_an_ordinary_new_file(tmp_path):
    reference = tmp_path / "ordinary.h5ad"
    _adata().write_h5ad(reference)
    saved = tmp_path / "saved.h5ad"
    _save_anndata_safely(_adata(), str(saved))
    assert saved.stat().st_mode & 0o7777 == reference.stat().st_mode & 0o7777


def test_overwrite_keeps_the_file_mode(tmp_path):
    out = tmp_path / "shared.h5ad"
    _adata().write_h5ad(out)
    os.chmod(out, 0o640)
    _save_anndata_safely(_adata(4), str(out))
    assert out.stat().st_mode & 0o777 == 0o640


def test_new_file_follows_the_folder_default_acl(tmp_path):
    """A folder shared through a default ACL: the collaborator keeps read/write, as for any new file."""
    _require_acls(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    subprocess.run(["setfacl", "-d", "-m", f"u:{COLLABORATOR}:rw", str(shared)], check=True)
    reference = shared / "ordinary.h5ad"
    _adata().write_h5ad(reference)
    saved = shared / "saved.h5ad"
    _save_anndata_safely(_adata(), str(saved))
    assert _acl(saved) == _acl(reference)
    assert f"user:{COLLABORATOR}:rw-" in _acl(saved)


def test_overwrite_keeps_access_granted_on_the_file(tmp_path):
    _require_acls(tmp_path)  # first: macOS has no setfacl
    out = tmp_path / "shared.h5ad"
    _adata().write_h5ad(out)
    subprocess.run(["setfacl", "-m", f"u:{COLLABORATOR}:rw", str(out)], check=True)
    before = _acl(out)
    assert f"user:{COLLABORATOR}:rw-" in before
    _save_anndata_safely(_adata(4), str(out))
    assert _acl(out) == before
