# This file makes 'resources' a package.
from .drug_resolver import DrugResolver
from .gene_resolver import GeneResolver, detect_identifier_type

__all__ = ["GeneResolver", "DrugResolver", "detect_identifier_type"]
