"""Flat ``MODEL_REGISTRY`` merged from the per-modality submodules.

After audit step 3 this file is a 25-line merge, not a 285-line
verbatim dict. The per-modality submodules
(``dna.py``, ``protein.py``, ``molecule.py``, ``text.py``,
``morphology.py``, ``singlecell.py``, ``api.py``) are the source of
truth for entries; this file owns only:

* the public ``MODEL_REGISTRY`` flat dict (a merge of all per-modality
  dicts);
* re-exports of the DNA species sets ``HUMAN_ONLY_MODELS``,
  ``MOUSE_ONLY_MODELS``, ``MULTI_SPECIES_DNA`` so the public import
  path ``from embpy.embedder import HUMAN_ONLY_MODELS`` (audit step 2)
  keeps working.

Byte-equivalence with the pre-split snapshot is guarded by
``tests/embpy/test_registry_split.py``.
"""

from __future__ import annotations

from ..models.base import BaseModelWrapper
from .api import API_MODELS
from .dna import (
    DNA_MODELS,
    HUMAN_ONLY_MODELS,
    MOUSE_ONLY_MODELS,
    MULTI_SPECIES_DNA,
)
from .molecule import MOLECULE_MODELS
from .morphology import MORPHOLOGY_MODELS
from .protein import PROTEIN_MODELS
from .singlecell import SINGLECELL_MODELS
from .text import TEXT_MODELS


# Merge order: DNA, Protein, Molecule, Text, Morphology, Single-cell,
# API. This groups DNA entries together; in the pre-split flat dict
# GENA-LM, NT, HyenaDNA, and Caduceus were appended after morphology
# (a historical artifact of when they were added). Key SET and every
# (Wrapper, path) TUPLE remain byte-equivalent against the pre-split
# snapshot -- only the iteration order changes. Verified by
# `tests/embpy/test_registry_split.py::test_registry_set_equivalent_to_pre_split_snapshot`.
# `_discover_models` and `list_available_models` are order-insensitive
# (callers either filter or check membership, never index by position).
MODEL_REGISTRY: dict[str, tuple[type[BaseModelWrapper] | None, str | None]] = {
    **DNA_MODELS,
    **PROTEIN_MODELS,
    **MOLECULE_MODELS,
    **TEXT_MODELS,
    **MORPHOLOGY_MODELS,
    **SINGLECELL_MODELS,
    **API_MODELS,
}


__all__ = [
    "HUMAN_ONLY_MODELS",
    "MODEL_REGISTRY",
    "MOUSE_ONLY_MODELS",
    "MULTI_SPECIES_DNA",
]
