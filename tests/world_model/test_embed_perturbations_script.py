"""Black-box test for the embed_perturbations script.

We don't actually call BioEmbedder. We:
  1. Build a tiny synthetic AnnData with a known mix of control / gene /
     unknown labels.
  2. Patch BioEmbedderProvider._get_embedder to return a deterministic
     stub.
  3. Drive embed_perturbations.main and assert the sidecar JSON +
     status distribution match the input mix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")

from world_model.data.embeddings import (  # noqa: E402
    BioEmbedderProvider,
    EmbeddingStatus,
)
from world_model.scripts import embed_perturbations as ep  # noqa: E402


class _StubEmbedder:
    def __init__(self, resolved: dict[str, np.ndarray]) -> None:
        self._resolved = resolved

    def embed_genes_batch(self, *, model, identifiers, **_):
        del model
        return [self._resolved.get(i) for i in identifiers]


def _write_anndata(path: Path, labels: list[str]) -> None:
    import pandas as pd

    n_obs = len(labels)
    X = np.zeros((n_obs, 4), dtype=np.float32)
    obs = pd.DataFrame({"perturbation": labels})
    var = pd.DataFrame(index=[f"g{i}" for i in range(4)])
    ad = anndata.AnnData(X=X, obs=obs, var=var)
    ad.write_h5ad(path)


def test_script_classifies_and_filters(monkeypatch, tmp_path: Path) -> None:
    labels = [
        "non-targeting",
        "non-targeting_1",
        "NTC",
        "AAVS1",
        "control",
        "TP53",
        "TP53+MYC",
        "NT5C2",  # real gene that starts with NT; must NOT be control
        "unknown_gene_zzz",  # unresolved
    ]
    h5ad = tmp_path / "synth.h5ad"
    _write_anndata(h5ad, labels)

    resolved = {
        "TP53": np.linspace(0.1, 0.4, 4).astype(np.float32),
        "MYC": np.linspace(0.5, 0.2, 4).astype(np.float32),
        "NT5C2": None,  # explicitly unresolved by the embedder
    }

    # Patch the lazy embedder loader on the provider class.
    def _stub_get_embedder(self):
        if self._embedder is None:
            self._embedder = _StubEmbedder({k: v for k, v in resolved.items() if v is not None})
        return self._embedder

    monkeypatch.setattr(
        BioEmbedderProvider,
        "_get_embedder",
        _stub_get_embedder,
    )

    output_npz = tmp_path / "out.npz"
    output_h5ad = tmp_path / "out.h5ad"
    rc = ep.main(
        [
            "--dataset",
            "replogle",
            "--h5ad",
            str(h5ad),
            "--model",
            "stub_v0",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output",
            str(output_npz),
            "--output-h5ad",
            str(output_h5ad),
            "--obsm-key",
            "X_pert_stub",
        ]
    )
    assert rc == 0
    sidecar = output_npz.with_suffix(output_npz.suffix + ".status.json")
    assert sidecar.exists(), f"status sidecar missing at {sidecar}"
    meta = json.loads(sidecar.read_text())

    # Gene-side labels passed into the embedder (uniques, in input order):
    # TP53, TP53+MYC, NT5C2, unknown_gene_zzz. No control variant.
    assert meta["counts"][EmbeddingStatus.CONTROL.value] == 0, (
        "embed_perturbations must filter every control variant before "
        "reaching the embedder; the sidecar should report 0 control rows."
    )
    assert meta["counts"][EmbeddingStatus.RESOLVED.value] >= 2, meta["counts"]
    assert meta["counts"][EmbeddingStatus.UNRESOLVED.value] >= 1, meta["counts"]
    assert any("NT5C2" in s for s in meta["unresolved_symbols_first_50"])
    assert any("unknown_gene_zzz" in s for s in meta["unresolved_symbols_first_50"])

    # NPZ on disk carries (symbols, embeddings, statuses).
    arch = np.load(output_npz, allow_pickle=True)
    syms = list(arch["symbols"])
    assert "non-targeting" not in syms
    assert "AAVS1" not in syms
    assert "NTC" not in syms
    assert "control" not in syms
    assert "TP53" in syms
    assert "NT5C2" in syms

    attached = anndata.read_h5ad(output_h5ad)
    assert "X_pert_stub" in attached.obsm
    assert "X_pert_stub" in attached.uns["perturbations"]
    assert attached.uns["perturbations"]["X_pert_stub"]["entity_type"] == "perturbation"


def test_fail_on_unresolved_returns_nonzero(monkeypatch, tmp_path: Path) -> None:
    labels = ["TP53", "unknown_gene_zzz"]
    h5ad = tmp_path / "synth.h5ad"
    _write_anndata(h5ad, labels)
    resolved = {"TP53": np.zeros(3, dtype=np.float32) + 0.5}

    def _stub_get_embedder(self):
        if self._embedder is None:
            self._embedder = _StubEmbedder(resolved)
        return self._embedder

    monkeypatch.setattr(
        BioEmbedderProvider,
        "_get_embedder",
        _stub_get_embedder,
    )
    rc = ep.main(
        [
            "--dataset",
            "replogle",
            "--h5ad",
            str(h5ad),
            "--model",
            "stub_v0",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output",
            str(tmp_path / "out.npz"),
            "--fail-on-unresolved",
        ]
    )
    assert rc == 2


def test_script_uses_public_embed_for_perturbation_morphology(monkeypatch, tmp_path: Path) -> None:
    labels = ["non-targeting", "TP53", "MISSING"]
    h5ad = tmp_path / "synth_morph.h5ad"
    _write_anndata(h5ad, labels)

    class _FakeBioEmbedder:
        calls: list[dict] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def embed(self, identifiers, **kwargs):
            self.calls.append({"identifiers": list(identifiers), "kwargs": dict(kwargs)})
            assert kwargs["entity_type"] == "perturbation"
            assert kwargs["model"] == "subcell_mae_rybg"
            return {
                "schema_version": "embpy.uns_embedding.v1",
                "entity_type": "perturbation",
                "id_scheme": "perturbation_label",
                "entity_ids": ["TP53"],
                "matrix": np.ones((1, 5), dtype=np.float32),
                "aliases": {"TP53": {"perturbation_label": "TP53"}},
                "model": {"name": "subcell_mae_rybg"},
            }

    import embpy.embedder as embedder_mod

    monkeypatch.setattr(embedder_mod, "BioEmbedder", _FakeBioEmbedder)
    out = tmp_path / "morph.h5ad"
    rc = ep.main(
        [
            "--dataset",
            "replogle",
            "--h5ad",
            str(h5ad),
            "--model",
            "subcell_mae_rybg",
            "--entity-type",
            "perturbation",
            "--output-h5ad",
            str(out),
            "--obsm-key",
            "X_pert_subcell_mae_rybg",
            "--morphology-dataset",
            "hpa",
            "--max-images",
            "1",
        ]
    )

    assert rc == 0
    attached = anndata.read_h5ad(out)
    assert "X_pert_subcell_mae_rybg" in attached.obsm
    assert attached.obsm["X_pert_subcell_mae_rybg"].shape == (3, 5)
    assert "X_pert_subcell_mae_rybg" in attached.uns["perturbations"]
    statuses = attached.obs["X_pert_subcell_mae_rybg_status"].astype(str).tolist()
    assert statuses == ["CONTROL", "RESOLVED", "UNRESOLVED"]
