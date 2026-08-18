"""A missing optional backend must produce an install line that works.

Before this, `get_model("boltz2")` on a machine without boltz advised
``pip install unknown``: ``Boltz2Wrapper.load`` catches ``ModuleNotFoundError``
and re-raises a friendly ``ImportError`` whose own text names no module, so
every pattern in ``_guess_missing_package`` missed it.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from embpy.embedder import (
    _guess_missing_package,
    _missing_package_from_exception,
)
from embpy.embedder_registry.extras import WRAPPER_EXTRAS
from embpy.errors import DependencyError


def _boltz_style_error() -> ImportError:
    """Rebuild the exact chain Boltz2Wrapper.load raises (structure_models.py:85)."""
    try:
        try:
            raise ModuleNotFoundError("No module named 'boltz'")
        except ImportError as cause:
            raise ImportError(
                "Boltz-2 is not installed. Install with:\n"
                "  pip install boltz[cuda]\n"
                "See: https://github.com/jwohlwend/boltz"
            ) from cause
    except ImportError as exc:
        return exc


class TestChainWalking:
    def test_the_regex_guesser_alone_still_misses_it(self):
        """Pin the reason the fallback is needed, so it is not 'fixed' away."""
        assert _guess_missing_package(str(_boltz_style_error())) is None

    def test_chain_walk_recovers_the_module_name(self):
        assert _missing_package_from_exception(_boltz_style_error()) == "boltz"

    def test_plain_exception_without_a_cause(self):
        assert _missing_package_from_exception(
            ModuleNotFoundError("No module named 'mamba_ssm'")
        ) == "mamba_ssm"

    def test_returns_none_when_no_link_names_a_module(self):
        assert _missing_package_from_exception(ImportError("something went wrong")) is None

    def test_terminates_on_a_self_referential_chain(self):
        """__context__ cycles must not hang the error path."""
        first = ImportError("no module here")
        second = ImportError("nor here")
        first.__context__ = second
        second.__context__ = first
        assert _missing_package_from_exception(first) is None


@pytest.fixture(scope="module")
def declared_extras() -> set[str]:
    root = Path(__file__).resolve().parents[2]
    with open(root / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    return set(data["project"]["optional-dependencies"])


class TestExtrasTable:
    """The table is hand-maintained, so guard it against drift on both sides."""

    def test_every_extra_is_declared_in_pyproject(self, declared_extras):
        unknown = {e for e in WRAPPER_EXTRAS.values() if e not in declared_extras}
        assert not unknown, f"WRAPPER_EXTRAS names extras that do not exist: {unknown}"

    def test_every_wrapper_name_is_real(self):
        """Catches a typo'd key, which would silently disable the hint.

        Read from the registry *source* rather than the imported dict: a wrapper
        whose backend is not installed resolves to ``None`` at import time, and
        those are exactly the entries this table exists for.
        """
        stale = set(WRAPPER_EXTRAS) - _all_wrapper_names_in_source()
        assert not stale, f"WRAPPER_EXTRAS names unknown wrappers: {stale}"

    def test_the_source_scan_finds_the_registry(self):
        """Guard the guard: an empty scan would make the test above vacuous."""
        names = _all_wrapper_names_in_source()
        assert "Boltz2Wrapper" in names and len(names) > 10

    def test_boltz_is_mapped(self):
        """The case that motivated the table."""
        assert WRAPPER_EXTRAS["Boltz2Wrapper"] == "boltz"


def _all_wrapper_names_in_source() -> set[str]:
    """Wrapper class names referenced by the registry modules, import-independent."""
    registry_dir = Path(__file__).resolve().parents[2] / "src" / "embpy" / "embedder_registry"
    names: set[str] = set()
    for path in registry_dir.glob("*.py"):
        names.update(re.findall(r"\(([A-Za-z0-9_]+Wrapper)\b", path.read_text()))
    return names


class TestDependencyErrorMessage:
    def test_extra_produces_an_installable_line(self):
        err = DependencyError(package="embpy[boltz]", feature="model 'boltz2'")
        assert "pip install embpy[boltz]" in str(err)
        assert "unknown" not in str(err)

    def test_the_old_placeholder_is_what_we_are_avoiding(self):
        """Documents the regression: this is what users used to be told."""
        assert "pip install unknown" in str(DependencyError(package="unknown"))
