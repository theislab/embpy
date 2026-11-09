"""Tiny fixture checks for the cluster benchmark's data boundaries."""

import pandas as pd
import pytest

from embpy.scripts.benchmark_fusion_lincs import _compound_groups, _csv_view


def test_metadata_is_never_a_feature(tmp_path):
    path = tmp_path / "vectors.csv"
    pd.DataFrame(
        {"row_index": [100, 101, 102], "smiles": ["C", "CC", "CCC"], "e_1": [2.0, 3.0, 4.0], "e_0": [7.0, 8.0, 9.0]}
    ).to_csv(path, index=False)
    matrix, report = _csv_view(path, {"C", "CCC"})
    assert list(matrix.columns) == ["e_0", "e_1"]
    assert list(matrix.index) == ["C", "CCC"]
    assert report["excluded_metadata_columns"] == ["row_index"]
    assert len(report["sha256"]) == 64


def test_conflicting_duplicate_vectors_fail(tmp_path):
    path = tmp_path / "vectors.csv"
    pd.DataFrame({"smiles": ["C", "C"], "e_0": [1.0, 2.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Conflicting"):
        _csv_view(path, {"C"})


def test_explicit_conflict_exclusion_is_reported(tmp_path):
    path = tmp_path / "vectors.csv"
    pd.DataFrame({"smiles": ["C", "C", "CC"], "e_0": [1.0, 2.0, 3.0]}).to_csv(path, index=False)
    matrix, report = _csv_view(path, {"C", "CC"}, conflicting="drop")
    assert list(matrix.index) == ["CC"]
    assert report["conflicting_duplicate_max_abs_differences"] == {"C": 1.0}


def test_compound_alias_groups_are_transitive():
    obs = pd.DataFrame({"pert_iname": ["A", "A", "B", "B", "C"], "canonical_smiles": ["s1", "s2", "s2", "s3", "s4"]})
    groups = _compound_groups(obs)
    assert groups["s1"] == groups["s2"] == groups["s3"]
    assert groups["s4"] != groups["s1"]
