"""Model registry package for embpy (audit step 3).

After step 3 the registry is split across seven per-modality
submodules (``dna``, ``protein``, ``molecule``, ``text``,
``morphology``, ``singlecell``, ``api``). ``flat`` merges them into a
single public ``MODEL_REGISTRY`` so the existing user-facing import
path ``from embpy.embedder import MODEL_REGISTRY`` returns a
byte-equivalent dict object.

The DNA species sets (``HUMAN_ONLY_MODELS``, ``MOUSE_ONLY_MODELS``,
``MULTI_SPECIES_DNA``) live in ``dna.py`` per the audit
recommendation (section 1: "Currently consumed only inside
``embedder.py``; should live next to the DNA registry"). They are
re-exported here and through ``flat``.
"""

from __future__ import annotations

from .flat import (
    HUMAN_ONLY_MODELS,
    MODEL_REGISTRY,
    MOUSE_ONLY_MODELS,
    MULTI_SPECIES_DNA,
)

__all__ = [
    "HUMAN_ONLY_MODELS",
    "MODEL_REGISTRY",
    "MOUSE_ONLY_MODELS",
    "MULTI_SPECIES_DNA",
]
