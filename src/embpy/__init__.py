"""embpy -- infrastructure for biological foundation-model embeddings.

Importing this module is intentionally cheap: nothing heavy is loaded
eagerly. Public symbols (BioEmbedder, resolvers, errors, subpackages)
are resolved lazily via the PEP-562 ``__getattr__`` hook below. Touching
``embpy.BioEmbedder`` for the first time triggers the full
``embpy.embedder`` import (~3,600 lines, transformers + torch + numpy
heavy), but ``import embpy`` itself does not.

Why this matters in practice (see ``docs/audit/embpy_audit.md`` step 1):

* Cold ``import embpy`` on lustre used to take minutes because
  ``from . import dt, models, pl, pp, resources, tl`` plus
  ``from .embedder import BioEmbedder`` pulled the entire model-wrapper
  tree, ``anndata``, ``scanpy``, ``transformers``, ``torch``,
  ``pyarrow``, and triggered ABI checks at every level.
* With lazy loading, ``import embpy`` only reads this file. Heavy
  dependencies are paid for when -- and only when -- the user touches
  the relevant symbol.

Backward compatibility:

* ``embpy.BioEmbedder``, ``embpy.GeneResolver``, ``embpy.tl`` etc. all
  still work, including in ``from embpy import X`` form (PEP 562 makes
  ``from`` fall back to ``__getattr__`` when ``X`` is not yet in the
  module dict).
* ``embpy.world_model`` keeps the Part C deprecation shim semantics
  (warns + re-exports the top-level ``world_model`` package).
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

__version__ = version("embpy")

# Static-analysis aid: declare the lazy-exported names so type checkers
# (basedpyright, mypy) and IDE intellisense see the public surface.
# These imports are NOT executed at runtime; `__getattr__` below is the
# real load path.
if TYPE_CHECKING:
    from . import dt, models, pl, pp, resources, tl, world_model
    from .embedder import BioEmbedder
    from .errors import (
        ConfigError,
        DataError,
        DependencyError,
        EmbeddingError,
        EmbpyError,
        GeneNotInGraphError,
        GraphNotBuiltError,
        IdentifierError,
        InvalidPoolingError,
        InvalidSMILESError,
        ModelNotFoundError,
        ModelNotLoadedError,
    )
    from .resources import DrugResolver, GeneResolver

# Map of public_name -> import target. Targets are either a module path
# ("embpy.tl") or a "module:attr" pair ("embpy.embedder:BioEmbedder").
# Adding a new public symbol means a one-line entry here -- no eager
# import is taken on `import embpy`.
_LAZY: dict[str, str] = {
    "dt": "embpy.dt",
    "models": "embpy.models",
    "pl": "embpy.pl",
    "pp": "embpy.pp",
    "resources": "embpy.resources",
    "tl": "embpy.tl",
    "world_model": "embpy.world_model",
    "BioEmbedder": "embpy.embedder:BioEmbedder",
    "ConfigError": "embpy.errors:ConfigError",
    "DataError": "embpy.errors:DataError",
    "DependencyError": "embpy.errors:DependencyError",
    "EmbeddingError": "embpy.errors:EmbeddingError",
    "EmbpyError": "embpy.errors:EmbpyError",
    "GeneNotInGraphError": "embpy.errors:GeneNotInGraphError",
    "GraphNotBuiltError": "embpy.errors:GraphNotBuiltError",
    "IdentifierError": "embpy.errors:IdentifierError",
    "InvalidPoolingError": "embpy.errors:InvalidPoolingError",
    "InvalidSMILESError": "embpy.errors:InvalidSMILESError",
    "ModelNotFoundError": "embpy.errors:ModelNotFoundError",
    "ModelNotLoadedError": "embpy.errors:ModelNotLoadedError",
    "DrugResolver": "embpy.resources:DrugResolver",
    "GeneResolver": "embpy.resources:GeneResolver",
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access for the embpy public API.

    Triggers a single ``import_module`` (and, for ``module:attr``
    targets, one ``getattr``) on first access. The result is cached in
    the module's ``__dict__`` by Python's normal import machinery, so
    subsequent accesses pay no overhead.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if ":" in target:
        module, attr = target.split(":", 1)
        return getattr(import_module(module), attr)
    return import_module(target)


def __dir__() -> list[str]:
    """Make tab-completion / `dir(embpy)` show the lazy-export surface."""
    return sorted({*_LAZY, "__version__"})


__all__ = [
    "BioEmbedder",
    "ConfigError",
    "DataError",
    "DependencyError",
    "DrugResolver",
    "EmbeddingError",
    "EmbpyError",
    "GeneNotInGraphError",
    "GeneResolver",
    "GraphNotBuiltError",
    "IdentifierError",
    "InvalidPoolingError",
    "InvalidSMILESError",
    "ModelNotFoundError",
    "ModelNotLoadedError",
    "dt",
    "models",
    "pl",
    "pp",
    "resources",
    "tl",
    "world_model",
]
