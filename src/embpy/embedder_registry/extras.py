"""Which pip extra installs the optional backend behind each wrapper.

An optional dependency belongs to a *wrapper class*, not to an individual
registry key -- ``boltz2``, ``boltz2_pairwise`` and ``boltz2_both`` all share
``Boltz2Wrapper`` and all need the same ``boltz`` extra. So this maps wrapper
class **name** to extra name.

Names are plain strings on purpose: importing this module must never import a
backend, since its whole job is to explain what happens when a backend is
missing.

Only wrappers whose backend lives behind an extra appear here. Wrappers covered
by ``embpy[cpu]`` / ``embpy[gpu]`` are absent, and so are the single-cell
wrappers, which are resolved through ``_SC_MODEL_REGISTRY`` rather than
``MODEL_REGISTRY``. A wrapper missing from this table is not an error: the
caller falls back to naming the module from the import traceback.

Used by :func:`embpy.embedder._get_model` to turn a failed optional import into
advice a user can act on.
"""

from __future__ import annotations

WRAPPER_EXTRAS: dict[str, str] = {
    "AlphaGenomeWrapper": "alphagenome",
    "Boltz2Wrapper": "boltz",
    "BorzoiWrapper": "seqmodels",
    # CaduceusWrapper was missing, so a machine without mamba-ssm got a bare
    # ModelLoadError with no install advice -- the failure that made the gene
    # notebook look as though Caduceus were simply unavailable.
    "CaduceusWrapper": "caduceus",
    "ESM3Wrapper": "esm3",
    "ESMCWrapper": "esm3",
    "EnformerWrapper": "seqmodels",
    "Evo2Wrapper": "evo2",
    "EvoWrapper": "evo",
    "MiniMolWrapper": "minimol",
    "NucleotideTransformerV3Wrapper": "ntv3",
    "ScoobyWrapper": "scooby",
}


__all__ = ["WRAPPER_EXTRAS"]
