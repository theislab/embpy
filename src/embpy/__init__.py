from importlib.metadata import version

from . import pl, pp, tl
from .embedder import BioEmbedder
from .errors import ConfigError, IdentifierError, ModelNotFoundError

__all__ = [
    "pl",
    "pp",
    "tl",
    "BioEmbedder",
    "ModelNotFoundError",
    "IdentifierError",
    "ConfigError",
]

__version__ = version("embpy")
