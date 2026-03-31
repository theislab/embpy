# This file makes 'resources' a package.
from .drug_resolver import DrugResolver
from .gene_annotator import GeneAnnotator
from .gene_resolver import GeneResolver, detect_identifier_type
from .molecule_annotator import MoleculeAnnotator
from .protein_annotator import ProteinAnnotator
from .protein_resolver import ProteinResolver
from .text_resolver import TextResolver

__all__ = [
    "DrugResolver",
    "GeneAnnotator",
    "GeneResolver",
    "MoleculeAnnotator",
    "ProteinAnnotator",
    "ProteinResolver",
    "TextResolver",
    "detect_identifier_type",
]
