from importlib.metadata import version

from . import models, pl, pp, resources, tl
from .embedder import BioEmbedder
from .errors import ConfigError, IdentifierError, ModelNotFoundError
from .resources import DrugResolver, GeneResolver

__all__ = [
    "models",
    "pl",
    "pp",
    "tl",
    "resources",
    "BioEmbedder",
    "GeneResolver",
    "DrugResolver",
    "ModelNotFoundError",
    "IdentifierError",
    "ConfigError",
]

__version__ = version("embpy")
