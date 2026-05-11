from importlib.metadata import version

# NOTE on `world_model`: since the Part C package split, the world
# model lives at the top-level package `world_model` and is *not*
# imported eagerly here. `embpy.world_model` still resolves through
# the deprecation shim at `src/embpy/world_model/__init__.py`, but
# only when the user actually touches it (see `__getattr__` below).
# This keeps `import embpy` cheap and enforces the one-way arrow
# embpy -> {dt, pl, pp, tl, resources, models, embedder, errors}
# without `world_model` leaking into the import graph.
from . import dt, models, pl, pp, resources, tl
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

__all__ = [
    "dt",
    "models",
    "pl",
    "pp",
    "tl",
    "resources",
    "world_model",
    "BioEmbedder",
    "GeneResolver",
    "DrugResolver",
    "EmbpyError",
    "ConfigError",
    "IdentifierError",
    "InvalidSMILESError",
    "ModelNotFoundError",
    "ModelNotLoadedError",
    "InvalidPoolingError",
    "EmbeddingError",
    "GraphNotBuiltError",
    "GeneNotInGraphError",
    "DependencyError",
    "DataError",
]

__version__ = version("embpy")


def __getattr__(name: str):
    """Lazy attribute access for the `world_model` compatibility shim.

    Touching `embpy.world_model` triggers `import embpy.world_model`,
    which runs the deprecation shim and re-routes the submodule to the
    top-level `world_model` package. Doing this lazily (instead of an
    eager `from . import world_model` at module top) keeps `import
    embpy` byte-equivalent w.r.t. `sys.modules` to the pre-split
    behaviour as far as the world_model graph is concerned -- nothing
    is loaded until the user actually opts in.
    """
    if name == "world_model":
        import importlib

        return importlib.import_module("embpy.world_model")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
