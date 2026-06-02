from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from anndata import AnnData

from world_model.evaluation import cell_eval_runner


def _toy_anndatas() -> tuple[AnnData, AnnData]:
    obs = pd.DataFrame(
        {"perturbation": ["control", "control", "geneA", "geneA", "geneB", "geneB"]},
        index=[f"cell{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    real_x = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [1.1, 0.4, 0.0],
            [0.0, 1.0, 0.2],
            [0.0, 0.9, 0.3],
        ],
        dtype=np.float32,
    )
    pred_x = real_x + 0.05
    real = AnnData(X=real_x, obs=obs.copy(), var=var.copy())
    pred = AnnData(X=pred_x, obs=obs.copy(), var=var.copy())
    return real, pred


def test_missing_cell_eval_can_fail_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    real, pred = _toy_anndatas()
    monkeypatch.setattr(cell_eval_runner, "_try_import_cell_eval", lambda: None)

    with pytest.raises(RuntimeError, match="cell-eval is required"):
        cell_eval_runner.run_cell_eval(
            real,
            pred,
            perturbation_key="perturbation",
            control_label="control",
            use_cell_eval=True,
            require_cell_eval=True,
        )


def test_missing_cell_eval_falls_back_when_not_required(monkeypatch: pytest.MonkeyPatch) -> None:
    real, pred = _toy_anndatas()
    monkeypatch.setattr(cell_eval_runner, "_try_import_cell_eval", lambda: None)

    per_pert, agg = cell_eval_runner.run_cell_eval(
        real,
        pred,
        perturbation_key="perturbation",
        control_label="control",
        use_cell_eval=True,
        require_cell_eval=False,
    )

    assert set(per_pert["perturbation"]) == {"geneA", "geneB"}
    assert {"mse", "mae", "r2", "pearson", "spearman", "delta_cosine", "deg_overlap@50"}.issubset(
        per_pert.columns
    )
    assert int(agg["n_perturbations"].iloc[0]) == 2


def test_current_cell_eval_api_is_called_and_converted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real, pred = _toy_anndatas()
    captured: dict[str, object] = {}

    class FakePolarsFrame:
        def __init__(self, df: pd.DataFrame) -> None:
            self._df = df

        def to_pandas(self) -> pd.DataFrame:
            return self._df

    class FakeMetricsEvaluator:
        def __init__(self, **kwargs: object) -> None:
            captured["init"] = kwargs

        def compute(self, *, profile: str, write_csv: bool) -> tuple[FakePolarsFrame, FakePolarsFrame]:
            captured["compute"] = {"profile": profile, "write_csv": write_csv}
            return (
                FakePolarsFrame(pd.DataFrame({"perturbation": ["geneA"], "mse": [0.1]})),
                FakePolarsFrame(pd.DataFrame({"mse": [0.1], "n_perturbations": [1]})),
            )

    class FakeCellEvalModule:
        MetricsEvaluator = FakeMetricsEvaluator

    monkeypatch.setattr(cell_eval_runner, "_try_import_cell_eval", lambda: FakeCellEvalModule)

    per_pert, agg = cell_eval_runner.run_cell_eval(
        real,
        pred,
        perturbation_key="perturbation",
        control_label="control",
        use_cell_eval=True,
        require_cell_eval=True,
        profile="full",
        num_threads=8,
        outdir=str(tmp_path / "cell_eval"),
    )

    assert captured["init"] == {
        "adata_pred": pred,
        "adata_real": real,
        "control_pert": "control",
        "pert_col": "perturbation",
        "num_threads": 8,
        "outdir": str(tmp_path / "cell_eval"),
    }
    assert captured["compute"] == {"profile": "full", "write_csv": False}
    assert per_pert.to_dict("records") == [{"perturbation": "geneA", "mse": 0.1}]
    assert agg.to_dict("records") == [{"mse": 0.1, "n_perturbations": 1}]


def test_required_cell_eval_raises_when_external_call_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    real, pred = _toy_anndatas()

    class BrokenMetricsEvaluator:
        def __init__(self, **_: object) -> None:
            pass

        def compute(self, **_: object) -> tuple[pd.DataFrame, pd.DataFrame]:
            raise ValueError("bad AnnData")

    class FakeCellEvalModule:
        MetricsEvaluator = BrokenMetricsEvaluator

    monkeypatch.setattr(cell_eval_runner, "_try_import_cell_eval", lambda: FakeCellEvalModule)

    with pytest.raises(RuntimeError, match="installed but failed to run"):
        cell_eval_runner.run_cell_eval(
            real,
            pred,
            perturbation_key="perturbation",
            control_label="control",
            use_cell_eval=True,
            require_cell_eval=True,
        )
