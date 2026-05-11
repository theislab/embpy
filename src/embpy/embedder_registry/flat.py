"""Flat ``MODEL_REGISTRY`` for embpy (audit step 2).

This module is the source of truth for the mapping ``name -> (Wrapper,
path)``. It was extracted verbatim from ``embpy.embedder`` lines 1-301
of commit 882be61 so the byte-equivalence regression test
(``tests/embpy/test_registry_split.py``) can compare against a frozen
snapshot.

Step 3 (per-modality split) replaces this file with a merge of
``dna.py``, ``protein.py``, ``molecule.py``, ``text.py``,
``morphology.py``, ``singlecell.py``, and ``api.py``. The public
import path (``from embpy.embedder import MODEL_REGISTRY``) is
preserved across both steps.
"""

from __future__ import annotations

from ..models.api_models import APIEmbeddingWrapper
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
from ..models.molecule_models import (
    ChembertaWrapper,
    MHGGNNWrapper,
    MiniMolWrapper,
    MolEWrapper,
    MolformerWrapper,
    RDKitWrapper,
)
from ..models.morphology_models import SubCellWrapper
from ..models.protein_models import ESM2Wrapper, ESM3Wrapper, ESMCWrapper, ProtT5Wrapper
from ..models.text_models import LlamaEmbeddingWrapper, TextLLMWrapper

# Evo (v1/v1.5) is an optional dependency - import conditionally
try:
    from ..models.dna_models import EvoWrapper

    _HAVE_EVO = True
except ImportError:
    _HAVE_EVO = False
    EvoWrapper = None  # type: ignore

# Evo2 is an optional dependency - import conditionally
try:
    from ..models.dna_models import Evo2Wrapper

    _HAVE_EVO2 = True
except ImportError:
    _HAVE_EVO2 = False
    Evo2Wrapper = None  # type: ignore

# Boltz-2 structure model (optional: pip install boltz[cuda])
try:
    from ..models.structure_models import Boltz2Wrapper

    _HAVE_BOLTZ = True
except ImportError:
    _HAVE_BOLTZ = False
    Boltz2Wrapper = None  # type: ignore


# Define the model registry mapping user-facing names to Wrapper classes and model paths
# This could potentially be loaded from a config file or use entry points for extensibility
MODEL_REGISTRY: dict[str, tuple[type[BaseModelWrapper] | None, str | None]] = {
    # User-facing name: (WrapperClass, HuggingFace_or_Path_Identifier)
    # --- DNA Models ---
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
    # --- Protein Models ---
    # ESM-1b (Meta AI, 650M params, HuggingFace)
    "esm1b": (ESM2Wrapper, "facebook/esm-1b"),
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
    # --- Molecule Models ---
    "chemberta2MTR": (ChembertaWrapper, "DeepChem/ChemBERTa-77M-MTR"),
    "chemberta2MLM": (ChembertaWrapper, "DeepChem/ChemBERTa-100M-MLM"),
    "molformer_base": (MolformerWrapper, "ibm/MoLFormer-XL-both-10pct"),
    # RDKit Fingerprints (CPU-only, no download needed)
    "rdkit_fp": (RDKitWrapper, "rdkit"),
    "morgan_fp": (RDKitWrapper, "morgan"),
    "morgan_count_fp": (RDKitWrapper, "morgan_count"),
    "maccs_fp": (RDKitWrapper, "maccs"),
    "atom_pair_fp": (RDKitWrapper, "atom_pair"),
    "atom_pair_count_fp": (RDKitWrapper, "atom_pair_count"),
    "torsion_fp": (RDKitWrapper, "topological_torsion"),
    "torsion_count_fp": (RDKitWrapper, "topological_torsion_count"),
    # GNN-based molecule models (optional dependencies)
    "minimol": (MiniMolWrapper, "minimol"),
    "mhg_gnn": (MHGGNNWrapper, "ibm-research/materials.mhg-ged"),
    "mole": (MolEWrapper, "mole"),
    # --- Text Models ---
    "minilm_l6_v2": (TextLLMWrapper, "sentence-transformers/all-MiniLM-L6-v2"),
    "bert_base_uncased": (TextLLMWrapper, "bert-base-uncased"),
    # LLaMA decoder-only models (requires HF_TOKEN for gated access)
    "llama3.1_8b": (LlamaEmbeddingWrapper, "meta-llama/Llama-3.1-8B"),
    "llama3.2_3b": (LlamaEmbeddingWrapper, "meta-llama/Llama-3.2-3B"),
    "llama3.2_1b": (LlamaEmbeddingWrapper, "meta-llama/Llama-3.2-1B"),
    # API-based embedding models (require API keys via environment variables)
    "openai_small": (APIEmbeddingWrapper, "text-embedding-3-small"),
    "openai_large": (APIEmbeddingWrapper, "text-embedding-3-large"),
    "cohere_v3": (APIEmbeddingWrapper, "embed-english-v3.0"),
    "cohere_multilingual": (APIEmbeddingWrapper, "embed-multilingual-v3.0"),
    "voyage_3": (APIEmbeddingWrapper, "voyage-3"),
    "voyage_3_lite": (APIEmbeddingWrapper, "voyage-3-lite"),
    "google_embed": (APIEmbeddingWrapper, "text-embedding-005"),
    # --- Morphology Models (microscopy images) ---
    # SubCell ViT-MAE models (auto-downloaded from CZI S3)
    "subcell_mae_rybg": (SubCellWrapper, "subcell_mae_rybg"),
    "subcell_vit_rybg": (SubCellWrapper, "subcell_vit_rybg"),
    "subcell_mae_rbg": (SubCellWrapper, "subcell_mae_rbg"),
    "subcell_vit_rbg": (SubCellWrapper, "subcell_vit_rbg"),
    "subcell_mae_ybg": (SubCellWrapper, "subcell_mae_ybg"),
    "subcell_vit_ybg": (SubCellWrapper, "subcell_vit_ybg"),
    "subcell_mae_bg": (SubCellWrapper, "subcell_mae_bg"),
    "subcell_vit_bg": (SubCellWrapper, "subcell_vit_bg"),
    # Convenience aliases
    "subcell_mae": (SubCellWrapper, "subcell_mae"),
    "subcell_contrast": (SubCellWrapper, "subcell_contrast"),
    "subcell_vit": (SubCellWrapper, "subcell_vit"),
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
