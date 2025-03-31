from importlib.metadata import version

from . import pl, pp, tl, torch

__all__ = ["pl", "pp", "tl", "torch"]

__version__ = version("embpy")
