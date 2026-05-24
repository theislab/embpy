# Backward compatibility -- code lives in resources/gene/resolver.py
from .gene.resolver import *  # noqa: F403
from .gene.resolver import _looks_like_smiles  # noqa: F401
