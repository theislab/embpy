"""Model registry package for embpy (audit step 3).

After step 3 the registry is split across seven per-modality
submodules (``dna``, ``protein``, ``molecule``, ``text``,
``morphology``, ``singlecell``, ``api``). ``flat`` merges them into a
single public ``MODEL_REGISTRY`` so the existing user-facing import
path ``from embpy.embedder import MODEL_REGISTRY`` returns a
byte-equivalent dict object.

The DNA species sets (``HUMAN_ONLY_MODELS``, ``MOUSE_ONLY_MODELS``,
``MULTI_SPECIES_DNA``) live in ``dna.py`` per the audit
recommendation (section 1: "Currently consumed only inside
``embedder.py``; should live next to the DNA registry"). They are
re-exported here and through ``flat``.

The four re-exports resolve lazily. ``flat`` maps model keys to wrapper
*classes*, so importing it imports every wrapper module and with them torch and
transformers. Doing that in this ``__init__`` meant even
``embpy.embedder_registry.extras`` -- a bare ``dict[str, str]`` whose docstring
promises that "importing this module must never import a backend" -- could not
be imported on the lightweight core install. Now ``import
embpy.embedder_registry`` costs nothing and ``MODEL_REGISTRY`` pays for itself
on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # The `# pyright: ignore` is a targeted suppression for a false
    # positive specific to the Cursor `cursorpyright` integration: the IDE
    # can lose its automatic source-root inference and report
    # `from .flat import ...` here as unresolved even though runtime
    # resolution is correct. The `[tool.pyright]` block in `pyproject.toml`
    # declares `extraPaths = ["src"]` so a future IDE reload should clear
    # this without the ignore; it is left in place as a safety net.
    from .flat import (  # pyright: ignore[reportMissingImports]
        HUMAN_ONLY_MODELS,
        MODEL_REGISTRY,
        MOUSE_ONLY_MODELS,
        MULTI_SPECIES_DNA,
    )

_LAZY = frozenset(
    {
        "HUMAN_ONLY_MODELS",
        "MODEL_REGISTRY",
        "MOUSE_ONLY_MODELS",
        "MULTI_SPECIES_DNA",
    }
)


def __getattr__(name: str) -> Any:
    """PEP 562 lazy re-export of the four public symbols from ``flat``."""
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import flat

    value = getattr(flat, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(_LAZY)


__all__ = [
    "HUMAN_ONLY_MODELS",
    "MODEL_REGISTRY",
    "MOUSE_ONLY_MODELS",
    "MULTI_SPECIES_DNA",
]
