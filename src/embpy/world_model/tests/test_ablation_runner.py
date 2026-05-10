"""Tests for the ablation runner.

The real :func:`embpy.world_model.scripts.train.main` is replaced with a
no-op ``fake_train_main`` that writes only the per-run artifacts the
aggregator needs. This keeps the tests fast and dependency-free
(no real model downloads, no GPU).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml


def _write_fake_train_outputs(run_dir: Path, model_name: str) -> None:
    """Mimic the per-run artifact set ``scripts/train.py`` produces."""
    run_dir.mkdir(parents=True, exist_ok=True)
    # comparison.csv: world_model + a baseline.
    (run_dir / "comparison.csv").write_text(
        "name,r2,mse\nworld_model,0.5,0.1\nmean_baseline,0.2,0.3\n"
    )
    (run_dir / "world_model_metrics.csv").write_text("name,r2,mse\nworld_model,0.5,0.1\n")
    (run_dir / "baselines.csv").write_text("baseline,metric,value\nmean_baseline,r2,0.2\n")
    (run_dir / "action_embedding_meta.json").write_text(json.dumps({
        "source": "bio_embedder",
        "embedding_dim": 16,
        "n_symbols": 10,
        "n_unresolved": 0,
        "model_name": model_name,
    }))
    (run_dir / "train_log.csv").write_text(
        "epoch,train_loss,val_loss\n1,1.0,1.2\n2,0.5,0.7\n"
    )
    (run_dir / "config.yaml").write_text("placeholder: true\n")


def _make_fake_train_main(success_keys: set[str], failed_keys: set[str]):
    def fake(argv: list[str]) -> None:
        cfg_path = Path(argv[argv.index("--config") + 1])
        # Reading the dumped YAML lets us key behaviour off model_name without
        # needing to import the WorldModelConfig module.
        cfg = yaml.safe_load(cfg_path.read_text())
        run_dir = Path(cfg["output_dir"])
        model_name = cfg["action_embedding"]["model_name"]
        spec_key = run_dir.name
        if spec_key in failed_keys:
            raise RuntimeError(f"Synthetic failure for spec {spec_key}")
        if spec_key in success_keys:
            _write_fake_train_outputs(run_dir, model_name)
            return
        # Default: succeed.
        _write_fake_train_outputs(run_dir, model_name)

    return fake


def _write_base_config(path: Path, *, output_dir: Path) -> Path:
    cfg = {
        "run_name": "ablate_test",
        "output_dir": str(output_dir / "single"),
        "seed": 0,
        "mode": "single",
        "data": {
            "dataset": "replogle",
            "h5ad_path": str(output_dir / "tiny.h5ad"),
            "gene_embedding_path": "",
            "perturbation_key": "perturbation",
            "control_label": "non-targeting",
            "cell_type_key": None,
            "n_top_genes": 10,
            "log_normalize": True,
            "stack_size": 2,
            "sequence_length": 4,
            "n_pert": 2,
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "n_sequences_per_epoch": 8,
        },
        "action_embedding": {"source": "precomputed", "path": ""},
        "split": {"split_by": "perturbation", "train_fraction": 0.6, "seed": 0},
        "encoder": {"kind": "transformer", "d_model": 8, "n_layers": 1,
                    "n_heads": 2, "dropout": 0.0},
        "dynamics": {"kind": "gpt", "d_model": 8, "n_layers": 1, "n_heads": 2,
                     "dropout": 0.0, "max_sequence_length": 8, "use_action_token": True},
        "loss": {"latent_mse": 1.0, "decoder_mse": 0.0, "info_nce": 0.0,
                 "info_nce_temperature": 0.1},
        "optim": {"lr": 1.0e-3, "weight_decay": 0.0, "betas": [0.9, 0.95],
                  "grad_clip": 1.0, "scheduler": "constant", "warmup_steps": 0},
        "train": {"n_epochs": 1, "log_every_n_steps": 1, "eval_every_n_epochs": 1,
                  "save_every_n_epochs": 1, "device": "cpu", "amp": False,
                  "enable_tensorboard": False, "enable_csv_log": True},
        "eval": {"n_control_samples_per_pert": 2, "use_cell_eval": False,
                 "deg_top_k": 4, "save_predictions": False},
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


@pytest.fixture
def base_setup(tmp_path: Path):
    base_path = _write_base_config(tmp_path / "base.yaml", output_dir=tmp_path)
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(yaml.safe_dump({"grid": [
        {"key": "alpha", "model_name": "m_alpha"},
        {"key": "beta", "model_name": "m_beta"},
    ]}))
    return base_path, grid_path


def test_dry_run_prints_planned_configs(capsys, base_setup):
    base_path, grid_path = base_setup
    from embpy.world_model.scripts import ablate_action_encoder as runner

    rc = runner.main([
        "--base-config", str(base_path),
        "--grid", str(grid_path),
        "--output-root", str(base_path.parent / "ablation"),
        "--dry-run",
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "Specs (2):" in captured
    assert "key=alpha" in captured
    assert "key=beta" in captured
    assert "action_embedding.source     = bio_embedder" in captured
    assert "action_embedding.model_name = m_alpha" in captured


def test_runner_writes_per_spec_outputs_and_summary(monkeypatch, base_setup, tmp_path):
    base_path, grid_path = base_setup
    output_root = tmp_path / "ablation"

    from embpy.world_model.scripts import ablate_action_encoder as runner

    fake = _make_fake_train_main(success_keys={"alpha", "beta"}, failed_keys=set())
    # Skip the real shared-split pre-compute; just write a placeholder NPZ
    # and pass it through.
    def fake_precompute(base_cfg, root):
        path = root / "_shared_split" / "replogle.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, dummy=np.zeros(1, dtype=np.float32))
        return path, "deadbeef" * 8

    monkeypatch.setattr(runner, "precompute_shared_split", fake_precompute)
    monkeypatch.setattr(runner.train_script, "main", fake)

    rc = runner.main([
        "--base-config", str(base_path),
        "--grid", str(grid_path),
        "--output-root", str(output_root),
    ])
    assert rc == 0

    for key in ("alpha", "beta"):
        run_dir = output_root / key
        assert (run_dir / "world_model_metrics.csv").exists()
        assert (run_dir / "_ablation_run.json").exists()
        meta = json.loads((run_dir / "_ablation_run.json").read_text())
        assert meta["status"] == "ok"
        assert meta["wall_clock_s"] >= 0.0

    long_path = output_root / "summary_long.csv"
    wide_path = output_root / "summary_wide.csv"
    assert long_path.exists()
    assert wide_path.exists()

    import pandas as pd
    wide = pd.read_csv(wide_path)
    assert set(wide["grid_key"]) == {"alpha", "beta"}
    assert "r2" in wide.columns and "mse" in wide.columns


def test_runner_continues_on_spec_failure(monkeypatch, base_setup, tmp_path):
    base_path, grid_path = base_setup
    output_root = tmp_path / "ablation_fail"

    from embpy.world_model.scripts import ablate_action_encoder as runner

    fake = _make_fake_train_main(success_keys={"alpha"}, failed_keys={"beta"})

    def fake_precompute(base_cfg, root):
        path = root / "_shared_split" / "replogle.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, dummy=np.zeros(1, dtype=np.float32))
        return path, "deadbeef" * 8

    monkeypatch.setattr(runner, "precompute_shared_split", fake_precompute)
    monkeypatch.setattr(runner.train_script, "main", fake)

    rc = runner.main([
        "--base-config", str(base_path),
        "--grid", str(grid_path),
        "--output-root", str(output_root),
    ])
    assert rc == 1

    alpha_meta = json.loads((output_root / "alpha" / "_ablation_run.json").read_text())
    beta_meta = json.loads((output_root / "beta" / "_ablation_run.json").read_text())
    assert alpha_meta["status"] == "ok"
    assert beta_meta["status"] == "failed"
    assert "Synthetic failure" in beta_meta["error"]
    assert "traceback" in beta_meta

    assert (output_root / "summary_long.csv").exists()
    assert (output_root / "summary_wide.csv").exists()


def test_only_filter_keeps_a_subset(monkeypatch, base_setup, tmp_path):
    base_path, grid_path = base_setup
    output_root = tmp_path / "ablation_only"

    from embpy.world_model.scripts import ablate_action_encoder as runner

    fake = _make_fake_train_main(success_keys={"alpha"}, failed_keys=set())

    def fake_precompute(base_cfg, root):
        path = root / "_shared_split" / "replogle.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, dummy=np.zeros(1, dtype=np.float32))
        return path, "deadbeef" * 8

    monkeypatch.setattr(runner, "precompute_shared_split", fake_precompute)
    monkeypatch.setattr(runner.train_script, "main", fake)

    rc = runner.main([
        "--base-config", str(base_path),
        "--grid", str(grid_path),
        "--output-root", str(output_root),
        "--only", "alpha",
    ])
    assert rc == 0
    assert (output_root / "alpha").exists()
    assert not (output_root / "beta").exists()
