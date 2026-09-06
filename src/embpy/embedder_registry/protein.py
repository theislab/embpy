"""Protein registry entries (ESM family + ProtT5 + Boltz-2).

Boltz-2 is a structure model but registered here per the existing
grouping in ``embpy.embedder``'s original flat dict, so the per-modality
split preserves byte-equivalence. If a future PR creates a dedicated
``structure.py``, move the Boltz-2 entries and update the merge in
``embedder_registry/flat.py``.
"""

from __future__ import annotations

from ..models.base import BaseModelWrapper
from ..models.protein_models import ESM2Wrapper, ESM3Wrapper, ESMCWrapper, ProtT5Wrapper

try:
    from ..models.structure_models import Boltz2Wrapper

    _HAVE_BOLTZ = True
except ImportError:
    _HAVE_BOLTZ = False
    Boltz2Wrapper = None  # type: ignore


PROTEIN_MODELS: dict[str, tuple[type[BaseModelWrapper] | None, str | None]] = {
    # ESM-1b (Meta AI, 650M params, HuggingFace). The short "facebook/esm-1b"
    # repo declares tokenizer_class "ESMTokenizer", a name transformers dropped
    # when it renamed the class to EsmTokenizer, so AutoTokenizer raises
    # "Tokenizer class ESMTokenizer does not exist". The versioned repo carries
    # the current name and loads.
    "esm1b": (ESM2Wrapper, "facebook/esm1b_t33_650M_UR50S"),
    # ESM-1v (Meta AI, 650M params, 5 random seeds, HuggingFace)
    "esm1v_1": (ESM2Wrapper, "facebook/esm1v_t33_650M_UR90S_1"),
    "esm1v_2": (ESM2Wrapper, "facebook/esm1v_t33_650M_UR90S_2"),
    "esm1v_3": (ESM2Wrapper, "facebook/esm1v_t33_650M_UR90S_3"),
    "esm1v_4": (ESM2Wrapper, "facebook/esm1v_t33_650M_UR90S_4"),
    "esm1v_5": (ESM2Wrapper, "facebook/esm1v_t33_650M_UR90S_5"),
    # ESM-2 (Meta AI, HuggingFace Transformers)
    "esm2_8M": (ESM2Wrapper, "facebook/esm2_t6_8M_UR50D"),
    "esm2_35M": (ESM2Wrapper, "facebook/esm2_t12_35M_UR50D"),
    "esm2_150M": (ESM2Wrapper, "facebook/esm2_t30_150M_UR50D"),
    "esm2_650M": (ESM2Wrapper, "facebook/esm2_t33_650M_UR50D"),
    "esm2_3B": (ESM2Wrapper, "facebook/esm2_t36_3B_UR50D"),
    "esm2_15B": (ESM2Wrapper, "facebook/esm2_t48_15B_UR50D"),
    # ESM-C (EvolutionaryScale SDK)
    "esmc_300m": (ESMCWrapper, "esmc_300m"),
    "esmc_600m": (ESMCWrapper, "esmc_600m"),
    "esmc_6b": (ESMCWrapper, "esmc-6b-2024-12"),
    # ESM3 (EvolutionaryScale SDK -- open weights or Forge API)
    "esm3_small": (ESM3Wrapper, "esm3-small-2024-08"),
    "esm3_medium": (ESM3Wrapper, "esm3-medium-2024-08"),
    "esm3_large": (ESM3Wrapper, "esm3-large-2024-03"),
    # ProtT5 Models (ProtTrans)
    "prot_t5_xl": (ProtT5Wrapper, "Rostlab/prot_t5_xl_uniref50"),
    "prot_t5_xl_half": (ProtT5Wrapper, "Rostlab/prot_t5_xl_half_uniref50-enc"),
    # Boltz-2 structure model (requires: pip install boltz[cuda])
    "boltz2": (Boltz2Wrapper if _HAVE_BOLTZ else None, "boltz2"),
    "boltz2_pairwise": (Boltz2Wrapper if _HAVE_BOLTZ else None, "boltz2_pairwise"),
    "boltz2_both": (Boltz2Wrapper if _HAVE_BOLTZ else None, "boltz2_both"),
}
