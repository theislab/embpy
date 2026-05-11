"""Pytest configuration for the world_model test suite.

Adds the ``--run-smoke-regression`` flag used by the post-split
byte-equivalence regression test (test_post_split_smoke.py). The flag
is off by default so the unit-test suite stays fast on every push; CI
opts in once per release.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-smoke-regression",
        action="store_true",
        default=False,
        help=(
            "Run the slow byte-equivalent smoke-pipeline regression "
            "(tests/world_model/test_post_split_smoke.py). Compares "
            "post-split outputs against the snapshot under "
            "tests/_snapshots/pre_split/."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "smoke_regression: end-to-end regression that requires --run-smoke-regression to run.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-smoke-regression"):
        return
    skip = pytest.mark.skip(reason="needs --run-smoke-regression")
    for item in items:
        if "smoke_regression" in item.keywords:
            item.add_marker(skip)
