"""DNA registry entries + species sets.

Owns the per-modality dict ``DNA_MODELS`` plus the three species
sets that gate model selection on whether the user is asking for a
human, mouse, or multi-species genome (``HUMAN_ONLY_MODELS``,
``MOUSE_ONLY_MODELS``, ``MULTI_SPECIES_DNA``). The audit (section 1)
called out that the species sets are consumed only inside
``BioEmbedder.embed_gene``'s human/mouse guard and should live next to
the DNA registry; they do, now.

The optional ``EvoWrapper`` / ``Evo2Wrapper`` / ``AlphaGenomeWrapper`` /
``ScoobyWrapper`` gating uses the ``_HAVE_*`` flag pattern so the entire
dict can be constructed even when the optional dependency is missing --
the wrapper slot for unavailable models is ``None`` and the per-call
dispatch (see ``BioEmbedder._discover_models``) drops the entry from the
user-facing listing.
"""

from __future__ import annotations

from ..models.base import BaseModelWrapper
from ..models.dna_models import (
    BorzoiWrapper,
    CaduceusWrapper,
    EnformerWrapper,
    GENALMWrapper,
    HyenaDNAWrapper,
    NucleotideTransformerV3Wrapper,
    NucleotideTransformerWrapper,
)

try:
    from ..models.dna_models import EvoWrapper

    _HAVE_EVO = True
except ImportError:
    _HAVE_EVO = False
    EvoWrapper = None  # type: ignore

try:
    from ..models.dna_models import Evo2Wrapper

    _HAVE_EVO2 = True
except ImportError:
    _HAVE_EVO2 = False
    Evo2Wrapper = None  # type: ignore

try:
    from ..models.alphagenome_models import AlphaGenomeWrapper

    _HAVE_ALPHAGENOME = True
except ImportError:
    _HAVE_ALPHAGENOME = False
    AlphaGenomeWrapper = None  # type: ignore

try:
    from ..models.scooby_models import ScoobyWrapper

    _HAVE_SCOOBY = True
except ImportError:
    _HAVE_SCOOBY = False
    ScoobyWrapper = None  # type: ignore


DNA_MODELS: dict[str, tuple[type[BaseModelWrapper] | None, str | None]] = {
    "enformer_human_rough": (EnformerWrapper, "EleutherAI/enformer-official-rough"),
    # Borzoi (johahi/borzoi-pytorch, 4 replicates + mouse variants)
    "borzoi_v0": (BorzoiWrapper, "johahi/borzoi-replicate-0"),
    "borzoi_v1": (BorzoiWrapper, "johahi/borzoi-replicate-1"),
    "borzoi_v2": (BorzoiWrapper, "johahi/borzoi-replicate-2"),
    "borzoi_v3": (BorzoiWrapper, "johahi/borzoi-replicate-3"),
    "borzoi_v0_mouse": (BorzoiWrapper, "johahi/borzoi-replicate-0-mouse"),
    "borzoi_v1_mouse": (BorzoiWrapper, "johahi/borzoi-replicate-1-mouse"),
    "borzoi_v2_mouse": (BorzoiWrapper, "johahi/borzoi-replicate-2-mouse"),
    "borzoi_v3_mouse": (BorzoiWrapper, "johahi/borzoi-replicate-3-mouse"),
    # Flashzoi (3x faster Borzoi with FlashAttention-2)
    "flashzoi_v0": (BorzoiWrapper, "johahi/flashzoi-replicate-0"),
    "flashzoi_v1": (BorzoiWrapper, "johahi/flashzoi-replicate-1"),
    "flashzoi_v2": (BorzoiWrapper, "johahi/flashzoi-replicate-2"),
    "flashzoi_v3": (BorzoiWrapper, "johahi/flashzoi-replicate-3"),
    # Evo models (requires optional `evo-model` dependency: pip install embpy[evo])
    "evo1_8k": (EvoWrapper if _HAVE_EVO else None, "evo-1-8k-base"),
    "evo1_131k": (EvoWrapper if _HAVE_EVO else None, "evo-1-131k-base"),
    "evo1.5_8k": (EvoWrapper if _HAVE_EVO else None, "evo-1.5-8k-base"),
    "evo1_crispr": (EvoWrapper if _HAVE_EVO else None, "evo-1-8k-crispr"),
    "evo1_transposon": (EvoWrapper if _HAVE_EVO else None, "evo-1-8k-transposon"),
    # Evo2 models (requires optional `evo2` dependency: pip install embpy[evo2])
    "evo2_7b": (Evo2Wrapper if _HAVE_EVO2 else None, "evo2_7b"),
    "evo2_40b": (Evo2Wrapper if _HAVE_EVO2 else None, "evo2_40b"),
    "evo2_7b_base": (Evo2Wrapper if _HAVE_EVO2 else None, "evo2_7b_base"),
    "evo2_1b_base": (Evo2Wrapper if _HAVE_EVO2 else None, "evo2_1b_base"),
    # GENA-LM (AIRI-Institute) -- pip install transformers
    "gena_lm_bert_base": (GENALMWrapper, "AIRI-Institute/gena-lm-bert-base-t2t"),
    "gena_lm_bert_large": (GENALMWrapper, "AIRI-Institute/gena-lm-bert-large-t2t"),
    "gena_lm_bert_base_multi": (
        GENALMWrapper,
        "AIRI-Institute/gena-lm-bert-base-t2t-multi",
    ),
    "gena_lm_bigbird_base": (GENALMWrapper, "AIRI-Institute/gena-lm-bigbird-base-t2t"),
    # Nucleotide Transformer v1/v2 (InstaDeep) -- pip install transformers
    "nt_500m_human_ref": (
        NucleotideTransformerWrapper,
        "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    ),
    "nt_500m_1000g": (
        NucleotideTransformerWrapper,
        "InstaDeepAI/nucleotide-transformer-500m-1000g",
    ),
    "nt_2b5_1000g": (
        NucleotideTransformerWrapper,
        "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
    ),
    "nt_2b5_multi": (
        NucleotideTransformerWrapper,
        "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    ),
    "nt_v2_50m": (
        NucleotideTransformerWrapper,
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    ),
    "nt_v2_100m": (
        NucleotideTransformerWrapper,
        "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
    ),
    "nt_v2_250m": (
        NucleotideTransformerWrapper,
        "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
    ),
    "nt_v2_500m": (
        NucleotideTransformerWrapper,
        "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    ),
    # Nucleotide Transformer v3 (InstaDeep) -- pip install transformers
    "ntv3_8m_pre": (NucleotideTransformerV3Wrapper, "InstaDeepAI/NTv3_8M_pre"),
    "ntv3_100m_pre": (NucleotideTransformerV3Wrapper, "InstaDeepAI/NTv3_100M_pre"),
    "ntv3_100m_pos": (NucleotideTransformerV3Wrapper, "InstaDeepAI/NTv3_100M_pos"),
    "ntv3_650m_pre": (NucleotideTransformerV3Wrapper, "InstaDeepAI/NTv3_650M_pre"),
    "ntv3_650m_pos": (NucleotideTransformerV3Wrapper, "InstaDeepAI/NTv3_650M_pos"),
    # HyenaDNA (HazyResearch) -- pip install transformers
    "hyenadna_tiny_1k": (HyenaDNAWrapper, "LongSafari/hyenadna-tiny-1k-seqlen-hf"),
    "hyenadna_small_32k": (HyenaDNAWrapper, "LongSafari/hyenadna-small-32k-seqlen-hf"),
    "hyenadna_medium_160k": (
        HyenaDNAWrapper,
        "LongSafari/hyenadna-medium-160k-seqlen-hf",
    ),
    "hyenadna_medium_450k": (
        HyenaDNAWrapper,
        "LongSafari/hyenadna-medium-450k-seqlen-hf",
    ),
    "hyenadna_large_1m": (HyenaDNAWrapper, "LongSafari/hyenadna-large-1m-seqlen-hf"),
    # Caduceus (kuleshov-group) -- pip install embpy[caduceus]
    "caduceus_ph_131k": (
        CaduceusWrapper,
        "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
    ),
    "caduceus_ps_131k": (
        CaduceusWrapper,
        "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
    ),
    # AlphaGenome (Google DeepMind cloud API client) -- pip install embpy[alphagenome]
    "alphagenome": (AlphaGenomeWrapper if _HAVE_ALPHAGENOME else None, "alphagenome"),
    # Scooby (gagneurlab/scooby) -- pip install embpy[scooby]. Single-cell-resolution
    # DNA sequence model conditioned on a precomputed per-cell embedding; the model
    # path selects the checkpoint (and, via ScoobyWrapper.KNOWN_CHECKPOINTS, its
    # architecture hyperparameters).
    "scooby_onek1k": (ScoobyWrapper if _HAVE_SCOOBY else None, "lauradmartens/onek1k-scooby"),
    "scooby_neurips": (ScoobyWrapper if _HAVE_SCOOBY else None, "johahi/neurips-scooby"),
    "scooby_epicardioids": (
        ScoobyWrapper if _HAVE_SCOOBY else None,
        "lauradmartens/epicardioids-scooby",
    ),
}


HUMAN_ONLY_MODELS = frozenset(
    {
        "enformer_human_rough",
        "nt_500m_human_ref",
    }
)

MOUSE_ONLY_MODELS = frozenset(
    {
        "borzoi_v0_mouse",
        "borzoi_v1_mouse",
        "borzoi_v2_mouse",
        "borzoi_v3_mouse",
    }
)

MULTI_SPECIES_DNA = frozenset(
    {
        "nt_v2_50m",
        "nt_v2_100m",
        "nt_v2_250m",
        "nt_v2_500m",
        "nt_2b5_multi",
        "ntv3_8m_pre",
        "ntv3_100m_pre",
        "ntv3_100m_pos",
        "ntv3_650m_pre",
        "ntv3_650m_pos",
        "gena_lm_bert_base_multi",
        "hyenadna_tiny_1k",
        "hyenadna_small_32k",
        "hyenadna_medium_160k",
        "hyenadna_medium_450k",
        "hyenadna_large_1m",
        "caduceus_ph_131k",
        "caduceus_ps_131k",
    }
)
