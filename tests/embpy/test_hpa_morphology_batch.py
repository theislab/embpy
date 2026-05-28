"""Tests for HPA morphology batch resolution."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_hpa_batch_builds_shared_catalog_once(monkeypatch, tmp_path):
    import embpy.resources.hpa_images as hpa_images
    from embpy.embedder import BioEmbedder

    calls = []

    def fake_catalog(*, xml_source=None, cache_path=None, genes=None, cell_line=None):
        calls.append(
            {
                "xml_source": xml_source,
                "cache_path": cache_path,
                "genes": tuple(genes or ()),
                "cell_line": cell_line,
            }
        )
        return pd.DataFrame(
            [
                {
                    "gene": "TP53",
                    "antibody": "HPA000001",
                    "plate": "1",
                    "position": "A1",
                    "sample": "1",
                    "image_url_prefix": "https://images.example/1/A1_1",
                }
            ]
        )

    monkeypatch.setattr(hpa_images, "build_hpa_subcellular_catalog", fake_catalog)
    monkeypatch.setattr(
        hpa_images,
        "fetch_hpa_if_image_by_prefix",
        lambda _prefix: np.zeros((4, 8, 8), dtype=np.uint8),
    )

    embedder = BioEmbedder(device="cpu")
    monkeypatch.setattr(
        embedder,
        "embed_morphological_batch",
        lambda images, **_kwargs: [np.ones(3, dtype=np.float32) for _ in images],
    )

    matrix, labels = embedder.embed_perturbation_morphology_batch(
        ["TP53", "MISSING"],
        dataset="hpa",
        source="subcell",
        local_dir=str(tmp_path / "morphology_cache" / "nadig"),
        max_images=1,
        verbose=False,
    )

    assert len(calls) == 1
    assert set(calls[0]["genes"]) == {"TP53", "MISSING"}
    assert str(calls[0]["xml_source"]).endswith("morphology_cache/_hpa/proteinatlas.xml.gz")
    assert str(calls[0]["cache_path"]).endswith(".csv")
    assert labels == ["TP53"]
    assert matrix.shape == (1, 3)
