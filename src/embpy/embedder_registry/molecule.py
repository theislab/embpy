"""Molecule registry entries (ChemBERTa + MolFormer + RDKit + GNN-based)."""

from __future__ import annotations

from ..models.base import BaseModelWrapper
from ..models.molecule_models import (
    ChembertaWrapper,
    MHGGNNWrapper,
    MiniMolWrapper,
    MolEWrapper,
    MolformerWrapper,
    RDKitWrapper,
)


MOLECULE_MODELS: dict[str, tuple[type[BaseModelWrapper] | None, str | None]] = {
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
}
