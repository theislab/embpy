"""Tests for embpy.resources.gene.control.ControlPolicy.

Pin down the curated regex set so a future refactor cannot quietly
re-introduce the bug Part A.1 fixed (NTRK1 / NT5C2 / CTRL2 falsely
classified as control).
"""

from __future__ import annotations

import pytest

from embpy.resources.gene.control import (
    DEFAULT_CONTROL_PATTERNS,
    ControlClassification,
    ControlPolicy,
)


@pytest.mark.parametrize(
    "label",
    [
        "non-targeting",
        "non_targeting",
        "nontargeting",
        "non-targeting_1",
        "non-targeting_g1",
        "Non-Targeting",
        "NON-TARGETING_42",
        "NTC",
        "NTC-1",
        "NTC_a",
        "NT1",
        "NT12",
        "control",
        "Control",
        "CONTROL_GROUP_1",
        "ctrl",
        "CTRL_a",
        "safe-harbor",
        "safe_harbor",
        "SAFE-HARBOR_x",
        "AAVS1",
        "AAVS1_2",
        "scrambled",
        "scramble",
        "scramble_3",
        "empty-vector",
        "empty_vector",
    ],
)
def test_default_patterns_match_known_controls(label: str) -> None:
    policy = ControlPolicy.default()
    assert policy.is_control(label), f"{label!r} should match a control pattern"


@pytest.mark.parametrize(
    "label",
    [
        "NTRK1",        # real gene starting with NT
        "NT5C2",        # real gene starting with NT
        "NT5C3A",       # real gene starting with NT
        "CTRL2",        # real gene literally named CTRL2 in some libraries
        "CONTROL_GENE_X",  # not in the matched set; only "control" prefix + non-alnum tail matches
        "TP53",
        "MYC",
        "BRCA1",
        "AARS1",
        "NTCRELATEDGENE",  # No separator; not "NTC[-_]..." form
    ],
)
def test_default_patterns_do_not_match_real_genes(label: str) -> None:
    policy = ControlPolicy.default()
    assert not policy.is_control(label), (
        f"{label!r} should NOT match any control pattern but did"
    )


def test_empty_and_none_are_control() -> None:
    policy = ControlPolicy.default()
    assert policy.is_control(None)
    assert policy.is_control("")
    assert policy.is_control("   ")


def test_extra_labels_allowlist() -> None:
    policy = ControlPolicy.from_iterable(["GFP_only", "luciferase_guide"])
    assert policy.is_control("GFP_only")
    assert policy.is_control("gfp_only")           # case-insensitive
    assert policy.is_control("luciferase_guide")
    assert not policy.is_control("GFP")            # not in extras
    # The curated regex defaults still apply alongside extras.
    assert policy.is_control("non-targeting_42")


def test_extra_labels_do_not_match_real_genes() -> None:
    policy = ControlPolicy.from_iterable(["GFP_only"])
    assert not policy.is_control("TP53")
    assert not policy.is_control("MYC")


def test_split_combo_handles_all_separators() -> None:
    policy = ControlPolicy.default()
    assert policy.split_combo("TP53") == ["TP53"]
    assert policy.split_combo("TP53+MYC") == ["TP53", "MYC"]
    assert policy.split_combo("TP53,MYC") == ["TP53", "MYC"]
    assert policy.split_combo("TP53|MYC") == ["TP53", "MYC"]
    assert policy.split_combo(" TP53 , MYC ") == ["TP53", "MYC"]
    assert policy.split_combo(None) == []
    assert policy.split_combo("") == []
    assert policy.split_combo("   ") == []


def test_classify_pure_control() -> None:
    policy = ControlPolicy.default()
    r = policy.classify("non-targeting_1")
    assert isinstance(r, ControlClassification)
    assert r.kind == "control"
    assert r.gene_components == ()
    assert r.control_components == ("non-targeting_1",)


def test_classify_pure_gene() -> None:
    policy = ControlPolicy.default()
    r = policy.classify("TP53+MYC")
    assert r.kind == "gene"
    assert r.gene_components == ("TP53", "MYC")
    assert r.control_components == ()


def test_classify_mixed_label() -> None:
    policy = ControlPolicy.default()
    r = policy.classify("TP53+non-targeting")
    assert r.kind == "mixed"
    assert r.gene_components == ("TP53",)
    assert r.control_components == ("non-targeting",)


def test_classify_strict_raises_on_mixed() -> None:
    policy = ControlPolicy.from_iterable([], strict=True)
    with pytest.raises(ValueError, match="strict"):
        policy.classify("TP53+non-targeting")


def test_default_patterns_constant_is_tuple_for_immutability() -> None:
    # Regression guard: external callers should not be able to mutate
    # the curated defaults by appending to the module-level constant.
    assert isinstance(DEFAULT_CONTROL_PATTERNS, tuple)


def test_classify_returns_components_in_input_order() -> None:
    policy = ControlPolicy.default()
    r = policy.classify("MYC+TP53,BRCA1|EGFR")
    assert r.components == ("MYC", "TP53", "BRCA1", "EGFR")
