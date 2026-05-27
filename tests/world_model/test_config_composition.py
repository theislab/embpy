from __future__ import annotations

from pathlib import Path

import pytest

from world_model.configs import (
    WorldModelConfig,
    load_yaml_config,
    validate_world_model_data_sources,
)


def test_extends_recursively_merges_and_child_wins(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "run_name: base\n"
        "data:\n"
        "  dataset: replogle\n"
        "  batch_size: 64\n"
        "action_embedding:\n"
        "  source: anndata_obsm\n"
        "  obsm_key: X_pert_base\n"
        "  model_name: base\n"
    )
    child.write_text(
        "extends: base.yaml\n"
        "run_name: child\n"
        "data:\n"
        "  batch_size: 8\n"
        "action_embedding:\n"
        "  model_name: child\n"
    )

    cfg = load_yaml_config(child)

    assert cfg.run_name == "child"
    assert cfg.data.dataset == "replogle"
    assert cfg.data.batch_size == 8
    assert cfg.action_embedding.obsm_key == "X_pert_base"
    assert cfg.action_embedding.model_name == "child"


def test_extends_preserves_unknown_key_validation(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_a_world_model_field: true\n")

    with pytest.raises(KeyError, match="Unknown config keys"):
        load_yaml_config(bad)


def test_missing_parent_config_raises_clear_error(tmp_path: Path) -> None:
    child = tmp_path / "child.yaml"
    child.write_text("extends: missing.yaml\n")

    with pytest.raises(FileNotFoundError, match="Config parent"):
        load_yaml_config(child)


def test_dataset_configs_load() -> None:
    root = Path("src/world_model/world_model/configs")

    nadig = load_yaml_config(root / "datasets/nadig.yaml")
    replogle = load_yaml_config(root / "datasets/replogle.yaml")
    smoke = load_yaml_config(root / "experiments/smoke.yaml")

    assert nadig.data.control_label == "control"
    assert "data/crispr_datasets/nadig" in nadig.data.h5ad_path
    assert replogle.data.control_label == "control"
    assert smoke.mode == "single"
    assert smoke.action_embedding.obsm_key == "X_pert_tiny"


def test_control_label_validation_for_local_h5ad(tmp_path: Path) -> None:
    ad = pytest.importorskip("anndata")
    np = pytest.importorskip("numpy")

    path = tmp_path / "tiny.h5ad"
    adata = ad.AnnData(X=np.ones((3, 2)), obs={"perturbation": ["control", "GENE1", "GENE2"]})
    adata.obsm["X_state"] = np.ones((3, 4), dtype=np.float32)
    adata.obsm["X_pert"] = np.ones((3, 5), dtype=np.float32)
    adata.write_h5ad(path)

    cfg = WorldModelConfig()
    cfg.data.h5ad_path = str(path)
    cfg.data.control_label = "control"
    validate_world_model_data_sources(cfg)

    cfg.data.control_label = "non-targeting"
    with pytest.raises(ValueError, match="control label"):
        validate_world_model_data_sources(cfg)
