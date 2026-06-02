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
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

try:
    __version__ = version("embpy")
except PackageNotFoundError:
    # Source-tree imports during tests/development may not have installed
    # package metadata yet. Keep import embpy cheap and usable.
    __version__ = "0+unknown"

# Static-analysis aid: declare the lazy-exported names so type checkers
# (basedpyright, mypy) and IDE intellisense see the public surface.
# These imports are NOT executed at runtime; `__getattr__` below is the
# real load path.
if TYPE_CHECKING:
    from . import dt, models, pl, pp, resources, tl
    from .embedder import BioEmbedder
    from .errors import (
        ConfigError,
        ContextOverflowError,
        DataError,
        DependencyError,
        EmbeddingError,
        EmbpyError,
        GeneNotInGraphError,
        GraphNotBuiltError,
        IdentifierError,
        InvalidPoolingError,
        InvalidSMILESError,
        ModelLoadError,
        ModelNotFoundError,
        ModelNotLoadedError,
        ModelOOMError,
        ResolverError,
    )
    from .observability import log_event, time_block
    from .pp.static_embeddings import (
        StaticEmbeddingStore,
        load_static_embedding_package,
        render_static_embedding_dataset_card,
        write_static_embedding_dataset_card,
    )
    from .reporting import ResolutionRecord, ResolutionReport
    from .resources import DrugResolver, GeneResolver
    from .retry import embed_batch_with_oom_recovery, retry_with_backoff

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
    "BioEmbedder": "embpy.embedder:BioEmbedder",
    "StaticEmbeddingStore": "embpy.pp.static_embeddings:StaticEmbeddingStore",
    "ConfigError": "embpy.errors:ConfigError",
    "ContextOverflowError": "embpy.errors:ContextOverflowError",
    "DataError": "embpy.errors:DataError",
    "DependencyError": "embpy.errors:DependencyError",
    "EmbeddingError": "embpy.errors:EmbeddingError",
    "EmbpyError": "embpy.errors:EmbpyError",
    "GeneNotInGraphError": "embpy.errors:GeneNotInGraphError",
    "GraphNotBuiltError": "embpy.errors:GraphNotBuiltError",
    "IdentifierError": "embpy.errors:IdentifierError",
    "InvalidPoolingError": "embpy.errors:InvalidPoolingError",
    "InvalidSMILESError": "embpy.errors:InvalidSMILESError",
    "ModelLoadError": "embpy.errors:ModelLoadError",
    "ModelNotFoundError": "embpy.errors:ModelNotFoundError",
    "ModelNotLoadedError": "embpy.errors:ModelNotLoadedError",
    "ModelOOMError": "embpy.errors:ModelOOMError",
    "ResolverError": "embpy.errors:ResolverError",
    "DrugResolver": "embpy.resources:DrugResolver",
    "GeneResolver": "embpy.resources:GeneResolver",
    # Layer 2: per-input resolution reports.
    "ResolutionRecord": "embpy.reporting:ResolutionRecord",
    "ResolutionReport": "embpy.reporting:ResolutionReport",
    # Layer 3: retry + OOM bisection primitives.
    "retry_with_backoff": "embpy.retry:retry_with_backoff",
    "embed_batch_with_oom_recovery": "embpy.retry:embed_batch_with_oom_recovery",
    # Layer 4: structured logging helpers.
    "log_event": "embpy.observability:log_event",
    "time_block": "embpy.observability:time_block",
    "load_static_embedding_package": "embpy.pp.static_embeddings:load_static_embedding_package",
    "render_static_embedding_dataset_card": "embpy.pp.static_embeddings:render_static_embedding_dataset_card",
    "write_static_embedding_dataset_card": "embpy.pp.static_embeddings:write_static_embedding_dataset_card",
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
    "ContextOverflowError",
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
    "ModelLoadError",
    "ModelNotFoundError",
    "ModelNotLoadedError",
    "ModelOOMError",
    "ResolutionRecord",
    "ResolutionReport",
    "ResolverError",
    "StaticEmbeddingStore",
    "dt",
    "embed_batch_with_oom_recovery",
    "log_event",
    "load_static_embedding_package",
    "models",
    "pl",
    "pp",
    "resources",
    "retry_with_backoff",
    "time_block",
    "tl",
]
