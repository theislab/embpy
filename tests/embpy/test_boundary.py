"""Package-boundary enforcement: ``embpy`` must not import ``world_model``.

Since the Part C package split, ``world_model`` is a sibling top-level
package built on top of ``embpy``. The arrow is one-way: world_model
may freely depend on embpy, but embpy must not pull world_model in.
Eager imports across that boundary would resurrect the original
"infrastructure + modelling are one package" coupling we just spent a
PR breaking.

After the deprecation shim was removed (the codemod rewrote every
internal caller and embpy is not externally distributed), the
``embpy.world_model`` import path no longer exists. The tests below
assert both halves of that contract:

* ``import embpy`` leaves ``world_model.*`` out of ``sys.modules``;
* attribute access ``embpy.world_model`` raises ``AttributeError``;
* dotted ``import embpy.world_model`` raises ``ModuleNotFoundError``.

If any future PR re-introduces a shim or a real
``from world_model.X import Y`` inside ``src/embpy/``, these tests
fail.
"""

from __future__ import annotations

import importlib
import sys

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
        "`import embpy` produced an `embpy.world_model` entry in "
        "sys.modules. The deprecation shim was removed; any reappearance "
        "of `src/embpy/world_model/` re-creates the boundary leak."
    )


def test_embpy_world_model_attribute_no_longer_exists(fresh_import_state):
    """`embpy.world_model` must raise AttributeError (the shim is gone)."""
    embpy = importlib.import_module("embpy")
    with pytest.raises(AttributeError, match="world_model"):
        _ = embpy.world_model


def test_dotted_embpy_world_model_import_fails(fresh_import_state):
    """`import embpy.world_model` and `from embpy.world_model.X import Y` must fail."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("embpy.world_model")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("embpy.world_model.training")


# --- Audit step 1: lazy import surface ---------------------------------------


_HEAVY_SUBPACKAGES = ("dt", "models", "pl", "pp", "resources", "tl")


def test_import_embpy_does_not_eagerly_load_subpackages(fresh_import_state):
    """`import embpy` must not eagerly walk the heavy subpackage tree.

    Pre-step-1, `embpy/__init__.py` did
    `from . import dt, models, pl, pp, resources, tl`, which transitively
    pulled in anndata, scanpy, transformers, torch, pyarrow, ... and
    cost minutes of cold-import on lustre. After step 1, those imports
    are lazy. This test fails if any future PR puts them back.
    """
    importlib.import_module("embpy")
    eager = [f"embpy.{sub}" for sub in _HEAVY_SUBPACKAGES if f"embpy.{sub}" in sys.modules]
    assert not eager, (
        f"`import embpy` eagerly loaded {eager}. embpy/__init__.py "
        "must rely on the PEP-562 __getattr__ hook for these."
    )


def test_import_embpy_does_not_eagerly_load_embedder(fresh_import_state):
    """`import embpy` must not pull `embpy.embedder` (3,600-line module)."""
    importlib.import_module("embpy")
    assert "embpy.embedder" not in sys.modules, (
        "`import embpy` triggered `embpy.embedder`. The embedder is "
        "loaded lazily on first `embpy.BioEmbedder` access; eager "
        "imports here re-introduce the cold-import storm."
    )


def test_lazy_bioembedder_attr_access_still_works(fresh_import_state):
    """`embpy.BioEmbedder` (and `from embpy import BioEmbedder`) still resolve."""
    embpy = importlib.import_module("embpy")
    cls_via_attr = embpy.BioEmbedder
    from embpy import BioEmbedder as cls_via_from

    from embpy.embedder import BioEmbedder as cls_direct

    assert cls_via_attr is cls_direct
    assert cls_via_from is cls_direct
    # Touching it must have loaded embedder.py lazily exactly once.
    assert "embpy.embedder" in sys.modules
