"""Package-boundary enforcement: ``embpy`` must not import ``world_model``.

Since the Part C package split, ``world_model`` is a sibling top-level
package built on top of ``embpy``. The arrow is one-way: world_model
may freely depend on embpy, but embpy must not pull world_model in.
Eager imports across that boundary would resurrect the original
"infrastructure + modelling are one package" coupling we just spent a
PR breaking.

This test imports ``embpy`` end-to-end and asserts the world_model
graph is *not* present in ``sys.modules`` afterwards. It is the
canonical guardrail: if any future PR re-introduces
``from world_model.X import Y`` in ``embpy/``, this test fails.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest


@pytest.fixture()
def fresh_import_state():
    """Pop every embpy / world_model entry from sys.modules and restore later."""
    keys = [k for k in sys.modules if k == "embpy" or k.startswith("embpy.") or k == "world_model" or k.startswith("world_model.")]
    saved = {k: sys.modules.pop(k) for k in keys}
    try:
        yield
    finally:
        for k in list(sys.modules):
            if k == "embpy" or k.startswith("embpy.") or k == "world_model" or k.startswith("world_model."):
                sys.modules.pop(k, None)
        for k, v in saved.items():
            sys.modules[k] = v


def test_embpy_import_does_not_pull_world_model(fresh_import_state):
    """`import embpy` must leave `world_model` and `embpy.world_model` unloaded."""
    importlib.import_module("embpy")

    leaked = {k for k in sys.modules if k == "world_model" or k.startswith("world_model.")}
    assert not leaked, (
        f"`import embpy` leaked `world_model` modules into sys.modules: {sorted(leaked)}. "
        "embpy must not import world_model -- the boundary is one-way."
    )

    assert "embpy.world_model" not in sys.modules, (
        "`import embpy` triggered the embpy.world_model deprecation shim. "
        "The shim must only load when the user explicitly opts in via "
        "`import embpy.world_model` or attribute access."
    )


def test_shim_lazy_attr_access_warns_and_redirects(fresh_import_state):
    """`embpy.world_model` accessed as an attribute fires the shim's warning."""
    embpy = importlib.import_module("embpy")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        wm = embpy.world_model
    msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("embpy.world_model" in m and "world_model" in m for m in msgs), (
        f"Expected a DeprecationWarning mentioning the move, got: {msgs}"
    )

    real_wm = importlib.import_module("world_model")
    assert wm is real_wm, (
        "embpy.world_model should redirect to the top-level world_model package "
        "(sys.modules swap in the shim's __init__.py)."
    )


def test_dotted_shim_imports_resolve_to_world_model(fresh_import_state):
    """`from embpy.world_model.training import WorldModelTrainer` reaches the real package."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        shim_mod = importlib.import_module("embpy.world_model.training")
        real_mod = importlib.import_module("world_model.training")
    assert shim_mod is real_mod, (
        "Dotted imports through `embpy.world_model.X` must resolve to "
        "`world_model.X` (not load a parallel copy)."
    )
