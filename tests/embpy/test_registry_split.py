"""Byte-equivalence regression test for the per-modality registry split.

This test is the audit's step-3 regression bar (`docs/audit/embpy_audit.md`,
section 9): "The merged dict is byte-equivalent (every key + every
(Wrapper, path) tuple matches the pre-split flat dict)."

The snapshot below was captured from `src/embpy/embedder_registry/flat.py`
at commit `fde522f` (the result of audit step 2). It records, for every
registry entry:

    (registry_key, expected_wrapper_class_name, expected_path)

`expected_wrapper_class_name` is the textual name of the wrapper class
even for optional-dependency models (Evo, Evo2, Boltz-2). At runtime
those slots resolve to either the wrapper class (if the optional dep is
installed) or ``None`` (if not); the test compares against the wrapper
NAME so it passes regardless of the local optional-dep matrix.

If you add a new model entry to a per-modality submodule, the test
fails on the symmetric difference and tells you exactly which side is
out of sync. Update the snapshot below in the same PR.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Pre-split snapshot (commit fde522f). Order in this list matches the
# per-modality grouping in the new split (DNA, Protein, Molecule, Text,
# Morphology, API). Single-cell intentionally has no entries.
# ---------------------------------------------------------------------------
EXPECTED_ENTRIES: list[tuple[str, str, str]] = [
    # --- DNA ---
    ("enformer_human_rough", "EnformerWrapper", "EleutherAI/enformer-official-rough"),
    ("borzoi_v0", "BorzoiWrapper", "johahi/borzoi-replicate-0"),
    ("borzoi_v1", "BorzoiWrapper", "johahi/borzoi-replicate-1"),
    ("borzoi_v2", "BorzoiWrapper", "johahi/borzoi-replicate-2"),
    ("borzoi_v3", "BorzoiWrapper", "johahi/borzoi-replicate-3"),
    ("borzoi_v0_mouse", "BorzoiWrapper", "johahi/borzoi-replicate-0-mouse"),
    ("borzoi_v1_mouse", "BorzoiWrapper", "johahi/borzoi-replicate-1-mouse"),
    ("borzoi_v2_mouse", "BorzoiWrapper", "johahi/borzoi-replicate-2-mouse"),
    ("borzoi_v3_mouse", "BorzoiWrapper", "johahi/borzoi-replicate-3-mouse"),
    ("flashzoi_v0", "BorzoiWrapper", "johahi/flashzoi-replicate-0"),
    ("flashzoi_v1", "BorzoiWrapper", "johahi/flashzoi-replicate-1"),
    ("flashzoi_v2", "BorzoiWrapper", "johahi/flashzoi-replicate-2"),
    ("flashzoi_v3", "BorzoiWrapper", "johahi/flashzoi-replicate-3"),
    ("evo1_8k", "EvoWrapper", "evo-1-8k-base"),
    ("evo1_131k", "EvoWrapper", "evo-1-131k-base"),
    ("evo1.5_8k", "EvoWrapper", "evo-1.5-8k-base"),
    ("evo1_crispr", "EvoWrapper", "evo-1-8k-crispr"),
    ("evo1_transposon", "EvoWrapper", "evo-1-8k-transposon"),
    ("evo2_7b", "Evo2Wrapper", "evo2_7b"),
    ("evo2_40b", "Evo2Wrapper", "evo2_40b"),
    ("evo2_7b_base", "Evo2Wrapper", "evo2_7b_base"),
    ("evo2_1b_base", "Evo2Wrapper", "evo2_1b_base"),
    ("gena_lm_bert_base", "GENALMWrapper", "AIRI-Institute/gena-lm-bert-base-t2t"),
    ("gena_lm_bert_large", "GENALMWrapper", "AIRI-Institute/gena-lm-bert-large-t2t"),
    (
        "gena_lm_bert_base_multi",
        "GENALMWrapper",
        "AIRI-Institute/gena-lm-bert-base-t2t-multi",
    ),
    ("gena_lm_bigbird_base", "GENALMWrapper", "AIRI-Institute/gena-lm-bigbird-base-t2t"),
    (
        "nt_500m_human_ref",
        "NucleotideTransformerWrapper",
        "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    ),
    (
        "nt_500m_1000g",
        "NucleotideTransformerWrapper",
        "InstaDeepAI/nucleotide-transformer-500m-1000g",
    ),
    (
        "nt_2b5_1000g",
        "NucleotideTransformerWrapper",
        "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
    ),
    (
        "nt_2b5_multi",
        "NucleotideTransformerWrapper",
        "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    ),
    (
        "nt_v2_50m",
        "NucleotideTransformerWrapper",
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    ),
    (
        "nt_v2_100m",
        "NucleotideTransformerWrapper",
        "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
    ),
    (
        "nt_v2_250m",
        "NucleotideTransformerWrapper",
        "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
    ),
    (
        "nt_v2_500m",
        "NucleotideTransformerWrapper",
        "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    ),
    ("ntv3_8m_pre", "NucleotideTransformerV3Wrapper", "InstaDeepAI/NTv3_8M_pre"),
    ("ntv3_100m_pre", "NucleotideTransformerV3Wrapper", "InstaDeepAI/NTv3_100M_pre"),
    ("ntv3_100m_pos", "NucleotideTransformerV3Wrapper", "InstaDeepAI/NTv3_100M_pos"),
    ("ntv3_650m_pre", "NucleotideTransformerV3Wrapper", "InstaDeepAI/NTv3_650M_pre"),
    ("ntv3_650m_pos", "NucleotideTransformerV3Wrapper", "InstaDeepAI/NTv3_650M_pos"),
    ("hyenadna_tiny_1k", "HyenaDNAWrapper", "LongSafari/hyenadna-tiny-1k-seqlen-hf"),
    ("hyenadna_small_32k", "HyenaDNAWrapper", "LongSafari/hyenadna-small-32k-seqlen-hf"),
    (
        "hyenadna_medium_160k",
        "HyenaDNAWrapper",
        "LongSafari/hyenadna-medium-160k-seqlen-hf",
    ),
    (
        "hyenadna_medium_450k",
        "HyenaDNAWrapper",
        "LongSafari/hyenadna-medium-450k-seqlen-hf",
    ),
    ("hyenadna_large_1m", "HyenaDNAWrapper", "LongSafari/hyenadna-large-1m-seqlen-hf"),
    (
        "caduceus_ph_131k",
        "CaduceusWrapper",
        "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
    ),
    (
        "caduceus_ps_131k",
        "CaduceusWrapper",
        "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
    ),
    ("alphagenome", "AlphaGenomeWrapper", "alphagenome"),
    ("scooby_onek1k", "ScoobyWrapper", "lauradmartens/onek1k-scooby"),
    ("scooby_neurips", "ScoobyWrapper", "johahi/neurips-scooby"),
    ("scooby_epicardioids", "ScoobyWrapper", "lauradmartens/epicardioids-scooby"),
    # --- Protein ---
    # Deliberately diverges from the fde522f snapshot: "facebook/esm-1b" cannot
    # load on any current transformers (its tokenizer_class is the long-removed
    # "ESMTokenizer"), so the path was corrected to the versioned repo.
    ("esm1b", "ESM2Wrapper", "facebook/esm1b_t33_650M_UR50S"),
    ("esm1v_1", "ESM2Wrapper", "facebook/esm1v_t33_650M_UR90S_1"),
    ("esm1v_2", "ESM2Wrapper", "facebook/esm1v_t33_650M_UR90S_2"),
    ("esm1v_3", "ESM2Wrapper", "facebook/esm1v_t33_650M_UR90S_3"),
    ("esm1v_4", "ESM2Wrapper", "facebook/esm1v_t33_650M_UR90S_4"),
    ("esm1v_5", "ESM2Wrapper", "facebook/esm1v_t33_650M_UR90S_5"),
    ("esm2_8M", "ESM2Wrapper", "facebook/esm2_t6_8M_UR50D"),
    ("esm2_35M", "ESM2Wrapper", "facebook/esm2_t12_35M_UR50D"),
    ("esm2_150M", "ESM2Wrapper", "facebook/esm2_t30_150M_UR50D"),
    ("esm2_650M", "ESM2Wrapper", "facebook/esm2_t33_650M_UR50D"),
    ("esm2_3B", "ESM2Wrapper", "facebook/esm2_t36_3B_UR50D"),
    ("esm2_15B", "ESM2Wrapper", "facebook/esm2_t48_15B_UR50D"),
    ("esmc_300m", "ESMCWrapper", "esmc_300m"),
    ("esmc_600m", "ESMCWrapper", "esmc_600m"),
    ("esmc_6b", "ESMCWrapper", "esmc-6b-2024-12"),
    ("esm3_small", "ESM3Wrapper", "esm3-small-2024-08"),
    ("esm3_medium", "ESM3Wrapper", "esm3-medium-2024-08"),
    ("esm3_large", "ESM3Wrapper", "esm3-large-2024-03"),
    ("prot_t5_xl", "ProtT5Wrapper", "Rostlab/prot_t5_xl_uniref50"),
    ("prot_t5_xl_half", "ProtT5Wrapper", "Rostlab/prot_t5_xl_half_uniref50-enc"),
    ("boltz2", "Boltz2Wrapper", "boltz2"),
    ("boltz2_pairwise", "Boltz2Wrapper", "boltz2_pairwise"),
    ("boltz2_both", "Boltz2Wrapper", "boltz2_both"),
    # --- Molecule ---
    ("chemberta2MTR", "ChembertaWrapper", "DeepChem/ChemBERTa-77M-MTR"),
    ("chemberta2MLM", "ChembertaWrapper", "DeepChem/ChemBERTa-100M-MLM"),
    ("molformer_base", "MolformerWrapper", "ibm/MoLFormer-XL-both-10pct"),
    ("rdkit_fp", "RDKitWrapper", "rdkit"),
    ("morgan_fp", "RDKitWrapper", "morgan"),
    ("morgan_count_fp", "RDKitWrapper", "morgan_count"),
    ("maccs_fp", "RDKitWrapper", "maccs"),
    ("atom_pair_fp", "RDKitWrapper", "atom_pair"),
    ("atom_pair_count_fp", "RDKitWrapper", "atom_pair_count"),
    ("torsion_fp", "RDKitWrapper", "topological_torsion"),
    ("torsion_count_fp", "RDKitWrapper", "topological_torsion_count"),
    ("minimol", "MiniMolWrapper", "minimol"),
    ("mhg_gnn", "MHGGNNWrapper", "ibm-research/materials.mhg-ged"),
    ("mole", "MolEWrapper", "mole"),
    # --- Text ---
    ("minilm_l6_v2", "TextLLMWrapper", "sentence-transformers/all-MiniLM-L6-v2"),
    ("bert_base_uncased", "TextLLMWrapper", "bert-base-uncased"),
    ("llama3.1_8b", "LlamaEmbeddingWrapper", "meta-llama/Llama-3.1-8B"),
    ("llama3.2_3b", "LlamaEmbeddingWrapper", "meta-llama/Llama-3.2-3B"),
    ("llama3.2_1b", "LlamaEmbeddingWrapper", "meta-llama/Llama-3.2-1B"),
    # --- Morphology ---
    ("subcell_mae_rybg", "SubCellWrapper", "subcell_mae_rybg"),
    ("subcell_vit_rybg", "SubCellWrapper", "subcell_vit_rybg"),
    ("subcell_mae_rbg", "SubCellWrapper", "subcell_mae_rbg"),
    ("subcell_vit_rbg", "SubCellWrapper", "subcell_vit_rbg"),
    ("subcell_mae_ybg", "SubCellWrapper", "subcell_mae_ybg"),
    ("subcell_vit_ybg", "SubCellWrapper", "subcell_vit_ybg"),
    ("subcell_mae_bg", "SubCellWrapper", "subcell_mae_bg"),
    ("subcell_vit_bg", "SubCellWrapper", "subcell_vit_bg"),
    ("subcell_mae", "SubCellWrapper", "subcell_mae"),
    ("subcell_contrast", "SubCellWrapper", "subcell_contrast"),
    ("subcell_vit", "SubCellWrapper", "subcell_vit"),
    # --- API ---
    ("openai_small", "APIEmbeddingWrapper", "text-embedding-3-small"),
    ("openai_large", "APIEmbeddingWrapper", "text-embedding-3-large"),
    ("cohere_v3", "APIEmbeddingWrapper", "embed-english-v3.0"),
    ("cohere_multilingual", "APIEmbeddingWrapper", "embed-multilingual-v3.0"),
    ("voyage_3", "APIEmbeddingWrapper", "voyage-3"),
    ("voyage_3_lite", "APIEmbeddingWrapper", "voyage-3-lite"),
    ("google_embed", "APIEmbeddingWrapper", "text-embedding-005"),
]


def _wrapper_name(value: tuple[object, object]) -> str | None:
    """Return the wrapper class name, or None when the optional dep is absent."""
    cls = value[0]
    if cls is None:
        return None
    return getattr(cls, "__name__", str(cls))


@pytest.fixture(scope="module")
def live_registry():
    """Importing flat does not touch BioEmbedder; only the wrapper modules load."""
    from embpy.embedder_registry import flat

    return flat


@pytest.fixture(scope="module")
def modality_dicts():
    from embpy.embedder_registry import (
        api,
        dna,
        molecule,
        morphology,
        protein,
        singlecell,
        text,
    )

    return {
        "dna": dna.DNA_MODELS,
        "protein": protein.PROTEIN_MODELS,
        "molecule": molecule.MOLECULE_MODELS,
        "text": text.TEXT_MODELS,
        "morphology": morphology.MORPHOLOGY_MODELS,
        "singlecell": singlecell.SINGLECELL_MODELS,
        "api": api.API_MODELS,
    }


def test_registry_set_equivalent_to_pre_split_snapshot(live_registry):
    """Key set + every (wrapper-name, path) tuple matches the pre-split snapshot."""
    expected_map = {
        name: (wrapper_name, path) for name, wrapper_name, path in EXPECTED_ENTRIES
    }
    live = live_registry.MODEL_REGISTRY

    extra = sorted(set(live) - set(expected_map))
    missing = sorted(set(expected_map) - set(live))
    assert not extra, (
        f"Models added to the registry without updating the snapshot: {extra}"
    )
    assert not missing, f"Models removed from the registry: {missing}"

    mismatches = []
    for name, (expected_wrapper_name, expected_path) in expected_map.items():
        wrapper_cls, path = live[name]
        actual_wrapper_name = _wrapper_name((wrapper_cls, path))
        if path != expected_path:
            mismatches.append(f"{name}: path {actual_wrapper_name!r} vs {expected_path!r}")
        # If the optional dep is installed, the wrapper name must match the
        # snapshot. If the dep is missing, the slot is None -- the snapshot
        # still records the original name, but we accept None here.
        if actual_wrapper_name is not None and actual_wrapper_name != expected_wrapper_name:
            mismatches.append(
                f"{name}: wrapper {actual_wrapper_name!r} vs expected {expected_wrapper_name!r}"
            )
    assert not mismatches, "Tuple mismatches:\n  " + "\n  ".join(mismatches)


def test_per_modality_dicts_are_disjoint(modality_dicts):
    """Each registry key appears in exactly one per-modality dict."""
    owners: dict[str, str] = {}
    for mod_name, mod_dict in modality_dicts.items():
        for key in mod_dict:
            prev = owners.get(key)
            assert prev is None, (
                f"Model {key!r} duplicated across modalities: {prev} and {mod_name}"
            )
            owners[key] = mod_name


def test_per_modality_merge_equals_flat(live_registry, modality_dicts):
    """The union of per-modality dicts equals the flat MODEL_REGISTRY (set + values)."""
    merged: dict = {}
    for mod_dict in modality_dicts.values():
        merged.update(mod_dict)
    assert merged.keys() == live_registry.MODEL_REGISTRY.keys()
    for key, value in live_registry.MODEL_REGISTRY.items():
        assert merged[key] == value, f"value mismatch for {key!r}"


def test_species_sets_owned_by_dna(live_registry):
    """HUMAN_ONLY_MODELS / MOUSE_ONLY_MODELS / MULTI_SPECIES_DNA live in dna.py."""
    from embpy.embedder_registry import dna

    assert live_registry.HUMAN_ONLY_MODELS is dna.HUMAN_ONLY_MODELS
    assert live_registry.MOUSE_ONLY_MODELS is dna.MOUSE_ONLY_MODELS
    assert live_registry.MULTI_SPECIES_DNA is dna.MULTI_SPECIES_DNA


@pytest.mark.requires_torch
def test_embedder_re_export_is_same_object(live_registry):
    """`from embpy.embedder import MODEL_REGISTRY` returns the same dict.

    Audit step 2's regression bar -- still in force after step 3 because
    `embedder.py` keeps re-importing the four public symbols from
    `embpy.embedder_registry.flat`.
    """
    from embpy import embedder

    assert embedder.MODEL_REGISTRY is live_registry.MODEL_REGISTRY
    assert embedder.HUMAN_ONLY_MODELS is live_registry.HUMAN_ONLY_MODELS
    assert embedder.MOUSE_ONLY_MODELS is live_registry.MOUSE_ONLY_MODELS
    assert embedder.MULTI_SPECIES_DNA is live_registry.MULTI_SPECIES_DNA


def test_species_sets_reference_only_dna_entries(live_registry):
    """Every member of the species sets is a key in the DNA registry."""
    from embpy.embedder_registry import dna

    dna_keys = set(dna.DNA_MODELS)
    for set_name, frozen in [
        ("HUMAN_ONLY_MODELS", live_registry.HUMAN_ONLY_MODELS),
        ("MOUSE_ONLY_MODELS", live_registry.MOUSE_ONLY_MODELS),
        ("MULTI_SPECIES_DNA", live_registry.MULTI_SPECIES_DNA),
    ]:
        stray = sorted(set(frozen) - dna_keys)
        assert not stray, f"{set_name} references non-DNA keys: {stray}"
