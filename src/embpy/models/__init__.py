from .dna_models import BorzoiWrapper, EnformerWrapper
from .molecule_models import (
    ChembertaWrapper,
    MHGGNNWrapper,
    MiniMolWrapper,
    MolEWrapper,
    MolformerWrapper,
    RDKitWrapper,
)
from .protein_models import ESM2Wrapper, ESMCWrapper, ProtT5Wrapper, STRINGWrapper
from .text_models import TextLLMWrapper

# Evo (v1/v1.5) is optional (requires: pip install evo-model)
try:
    from .dna_models import EvoWrapper
except ImportError:
    EvoWrapper = None  # type: ignore

# Evo2 is optional (requires: pip install embpy[evo2])
try:
    from .dna_models import Evo2Wrapper
except ImportError:
    Evo2Wrapper = None  # type: ignore

# PPI GNN is optional (requires: pip install torch-geometric)
try:
    from .ppi_models import GNNEncoder, PPIGNNWrapper
except ImportError:
    GNNEncoder = None  # type: ignore
    PPIGNNWrapper = None  # type: ignore

__all__ = [
    "BorzoiWrapper",
    "EnformerWrapper",
    "EvoWrapper",
    "Evo2Wrapper",
    "ChembertaWrapper",
    "ESM2Wrapper",
    "ESMCWrapper",
    "GNNEncoder",
    "MHGGNNWrapper",
    "MiniMolWrapper",
    "MolEWrapper",
    "MolformerWrapper",
    "PPIGNNWrapper",
    "ProtT5Wrapper",
    "RDKitWrapper",
    "STRINGWrapper",
    "TextLLMWrapper",
]
