"""Model registry package for embpy (audit step 2).

This is the new home for ``MODEL_REGISTRY`` and the DNA species sets
(``HUMAN_ONLY_MODELS``, ``MOUSE_ONLY_MODELS``, ``MULTI_SPECIES_DNA``).
Step 2 only relocates the flat dict; step 3 will replace ``flat.py``
with a per-modality merge (``dna.py``, ``protein.py``, ...). The
public import path stays at ``embpy.embedder`` so existing user code
of the form

    from embpy.embedder import MODEL_REGISTRY

keeps working unchanged.
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
