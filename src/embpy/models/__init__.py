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

# HuggingFace-based DNA models (requires: pip install transformers)
try:
    from .dna_models import (
        CaduceusWrapper,
        GENALMWrapper,
        HyenaDNAWrapper,
        NucleotideTransformerWrapper,
        NucleotideTransformerV3Wrapper,
    )
except ImportError:
    CaduceusWrapper = None  # type: ignore
    GENALMWrapper = None  # type: ignore
    HyenaDNAWrapper = None  # type: ignore
    NucleotideTransformerWrapper = None  # type: ignore
    NucleotideTransformerV3Wrapper = None  # type: ignore

# PPI precomputed embeddings (requires: pip install h5py)
try:
    from .ppi_models import PrecomputedPPIWrapper
except ImportError:
    PrecomputedPPIWrapper = None  # type: ignore

# Single-cell foundation model wrappers (requires: pip install helical)
from .singlecell_models import (
    Cell2SentenceWrapper,
    GeneformerWrapper,
    SCModelCard,
    ScGPTWrapper,
    SingleCellWrapper,
    TahoeWrapper,
    TranscriptFormerWrapper,
    UCEWrapper,
    get_singlecell_wrapper,
    list_singlecell_models,
    singlecell_info,
)

__all__ = [
    "BorzoiWrapper",
    "EnformerWrapper",
    "EvoWrapper",
    "Evo2Wrapper",
    "CaduceusWrapper",
    "Cell2SentenceWrapper",
    "GENALMWrapper",
    "GeneformerWrapper",
    "HyenaDNAWrapper",
    "NucleotideTransformerWrapper",
    "NucleotideTransformerV3Wrapper",
    "ChembertaWrapper",
    "ESM2Wrapper",
    "ESMCWrapper",
    "MHGGNNWrapper",
    "MiniMolWrapper",
    "MolEWrapper",
    "MolformerWrapper",
    "PrecomputedPPIWrapper",
    "ProtT5Wrapper",
    "RDKitWrapper",
    "SCModelCard",
    "ScGPTWrapper",
    "SingleCellWrapper",
    "STRINGWrapper",
    "TahoeWrapper",
    "TextLLMWrapper",
    "TranscriptFormerWrapper",
    "UCEWrapper",
    "get_singlecell_wrapper",
    "list_singlecell_models",
    "singlecell_info",
]
