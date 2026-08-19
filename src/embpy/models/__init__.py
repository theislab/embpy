"""Model wrapper namespace -- lazily resolved, so importing it is free.

Every wrapper module in this package imports ``torch`` (and most import
``transformers``) at module level: that is what they are for. Re-exporting
them eagerly, as this file used to, meant ``import embpy.models`` -- and
therefore ``from embpy.models.base import BaseModelWrapper``, since Python
executes a package's ``__init__`` before any of its submodules -- pulled the
entire deep-learning stack. On the lightweight core install
(``pip install embpy``, which CI's ``core`` jobs use via ``.[test]``) torch is
absent, so that first line raised ``ModuleNotFoundError`` and took
``embpy.embedder`` down with it.

Names are now resolved on first attribute access via the PEP 562
``__getattr__`` hook below, matching the pattern already used in
``embpy/__init__.py``. ``import embpy.models`` reads this file and nothing
else; ``embpy.models.ESM2Wrapper`` is what imports torch.

The public surface is unchanged. In particular the optional wrappers listed
in ``_OPTIONAL`` still resolve to ``None`` rather than raising when their
backend is missing -- previously a module-level ``try/except ImportError``,
now the same ``except ImportError`` around the deferred import. Resolved
values (including ``None``) are cached in the module dict, so a missing
optional backend costs one failed import, not one per access.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

# Static-analysis aid: declare the lazy-exported names so type checkers
# (basedpyright, mypy) and IDE intellisense see the public surface. These
# imports are NOT executed at runtime; `__getattr__` below is the real load
# path.
if TYPE_CHECKING:
    from .alphagenome_models import AlphaGenomeWrapper
    from .dna_models import (
        BorzoiWrapper,
        CaduceusWrapper,
        EnformerWrapper,
        Evo2Wrapper,
        EvoWrapper,
        GENALMWrapper,
        HyenaDNAWrapper,
        NucleotideTransformerV3Wrapper,
        NucleotideTransformerWrapper,
    )
    from .molecule_models import (
        ChembertaWrapper,
        MHGGNNWrapper,
        MiniMolWrapper,
        MolEWrapper,
        MolformerWrapper,
        RDKitWrapper,
    )
    from .ppi_models import PrecomputedPPIWrapper
    from .protein_models import ESM2Wrapper, ESM3Wrapper, ESMCWrapper, ProtT5Wrapper
    from .scooby_models import ScoobyWrapper
    from .singlecell_models import (
        Cell2SentenceWrapper,
        GeneformerWrapper,
        PCAEmbedding,
        ScGPTWrapper,
        SCModelCard,
        ScVIToolsWrapper,
        SingleCellWrapper,
        StackWrapper,
        StateEmbeddingWrapper,
        TahoeWrapper,
        TranscriptFormerWrapper,
        UCEWrapper,
        get_singlecell_wrapper,
        list_singlecell_models,
        singlecell_info,
    )
    from .text_models import TextLLMWrapper

# Map of public_name -> "submodule:attr". Adding a wrapper to the public
# surface means one entry here plus one line in `__all__` -- no eager import
# is taken on `import embpy.models`.
_LAZY: dict[str, str] = {
    # --- DNA ---
    "BorzoiWrapper": "dna_models:BorzoiWrapper",
    "EnformerWrapper": "dna_models:EnformerWrapper",
    "EvoWrapper": "dna_models:EvoWrapper",
    "Evo2Wrapper": "dna_models:Evo2Wrapper",
    "CaduceusWrapper": "dna_models:CaduceusWrapper",
    "GENALMWrapper": "dna_models:GENALMWrapper",
    "HyenaDNAWrapper": "dna_models:HyenaDNAWrapper",
    "NucleotideTransformerWrapper": "dna_models:NucleotideTransformerWrapper",
    "NucleotideTransformerV3Wrapper": "dna_models:NucleotideTransformerV3Wrapper",
    "AlphaGenomeWrapper": "alphagenome_models:AlphaGenomeWrapper",
    "ScoobyWrapper": "scooby_models:ScoobyWrapper",
    # --- Protein ---
    "ESM2Wrapper": "protein_models:ESM2Wrapper",
    "ESM3Wrapper": "protein_models:ESM3Wrapper",
    "ESMCWrapper": "protein_models:ESMCWrapper",
    "ProtT5Wrapper": "protein_models:ProtT5Wrapper",
    # --- Molecule ---
    "ChembertaWrapper": "molecule_models:ChembertaWrapper",
    "MHGGNNWrapper": "molecule_models:MHGGNNWrapper",
    "MiniMolWrapper": "molecule_models:MiniMolWrapper",
    "MolEWrapper": "molecule_models:MolEWrapper",
    "MolformerWrapper": "molecule_models:MolformerWrapper",
    "RDKitWrapper": "molecule_models:RDKitWrapper",
    # --- Text ---
    "TextLLMWrapper": "text_models:TextLLMWrapper",
    # --- PPI ---
    "PrecomputedPPIWrapper": "ppi_models:PrecomputedPPIWrapper",
    # --- Single-cell ---
    "Cell2SentenceWrapper": "singlecell_models:Cell2SentenceWrapper",
    "GeneformerWrapper": "singlecell_models:GeneformerWrapper",
    "PCAEmbedding": "singlecell_models:PCAEmbedding",
    "SCModelCard": "singlecell_models:SCModelCard",
    "ScGPTWrapper": "singlecell_models:ScGPTWrapper",
    "ScVIToolsWrapper": "singlecell_models:ScVIToolsWrapper",
    "SingleCellWrapper": "singlecell_models:SingleCellWrapper",
    "StackWrapper": "singlecell_models:StackWrapper",
    "StateEmbeddingWrapper": "singlecell_models:StateEmbeddingWrapper",
    "TahoeWrapper": "singlecell_models:TahoeWrapper",
    "TranscriptFormerWrapper": "singlecell_models:TranscriptFormerWrapper",
    "UCEWrapper": "singlecell_models:UCEWrapper",
    "get_singlecell_wrapper": "singlecell_models:get_singlecell_wrapper",
    "list_singlecell_models": "singlecell_models:list_singlecell_models",
    "singlecell_info": "singlecell_models:singlecell_info",
}

# Wrappers whose backend is an optional extra, not part of `embpy[cpu]` /
# `embpy[gpu]`. These resolve to `None` when the backend is missing, which is
# the contract the registry relies on: `MODEL_REGISTRY` slots hold `None` for
# unavailable models and `BioEmbedder._discover_models` drops them from the
# user-facing listing. Everything not listed here raises, as before.
_OPTIONAL: frozenset[str] = frozenset(
    {
        # requires: pip install evo-model
        "EvoWrapper",
        # requires: pip install embpy[evo2]
        "Evo2Wrapper",
        # requires: pip install transformers
        "CaduceusWrapper",
        "GENALMWrapper",
        "HyenaDNAWrapper",
        "NucleotideTransformerWrapper",
        "NucleotideTransformerV3Wrapper",
        # requires: pip install h5py
        "PrecomputedPPIWrapper",
        # requires: pip install alphagenome
        "AlphaGenomeWrapper",
        # requires: pip install snapatac2-scooby && pip install git+https://github.com/gagneurlab/scooby.git
        "ScoobyWrapper",
    }
)


#: Submodules of this package. Eager re-exports used to bind these as package
#: attributes as a side effect (importing `.dna_models` binds `dna_models`, and
#: its own `from .base import BaseModelWrapper` binds `base`), so
#: `embpy.models.base.BaseModelWrapper` worked after a bare `import embpy`.
#: Resolve them here so that keeps working -- lazily, like everything else.
_SUBMODULES: frozenset[str] = frozenset(
    {
        "alphagenome_models",
        "api_models",
        "base",
        "dna_models",
        "molecule_models",
        "morphology_models",
        "ppi_models",
        "protein_models",
        "scooby_models",
        "singlecell_models",
        "structure_models",
        "text_models",
    }
)


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access for the wrapper namespace.

    Triggers a single ``import_module`` (plus one ``getattr`` for wrapper
    names) on first access, then caches the result in the module dict so later
    accesses are ordinary attribute lookups that never reach this hook.
    """
    if name in _SUBMODULES:
        value = import_module(f".{name}", __name__)
        globals()[name] = value
        return value

    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module, attr = target.split(":", 1)
    try:
        value = getattr(import_module(f".{module}", __name__), attr)
    except ImportError:
        # The optional backends kept their old contract: missing means `None`,
        # not an exception. Cache the `None` so a missing backend costs one
        # failed import for the process, not one per attribute access.
        if name not in _OPTIONAL:
            raise
        value = None

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Make tab-completion / `dir(embpy.models)` show the lazy-export surface."""
    return sorted({*_LAZY, *_SUBMODULES})


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
    "ESM3Wrapper",
    "ESMCWrapper",
    "MHGGNNWrapper",
    "MiniMolWrapper",
    "MolEWrapper",
    "MolformerWrapper",
    "PCAEmbedding",
    "PrecomputedPPIWrapper",
    "ProtT5Wrapper",
    "RDKitWrapper",
    "SCModelCard",
    "ScGPTWrapper",
    "ScVIToolsWrapper",
    "SingleCellWrapper",
    "StackWrapper",
    "StateEmbeddingWrapper",
    "TahoeWrapper",
    "TextLLMWrapper",
    "TranscriptFormerWrapper",
    "UCEWrapper",
    "get_singlecell_wrapper",
    "list_singlecell_models",
    "singlecell_info",
    "AlphaGenomeWrapper",
    "ScoobyWrapper",
]
