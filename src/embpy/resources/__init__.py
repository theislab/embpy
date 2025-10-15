# This file makes 'resources' a package.
from .drug_resolver import DrugResolver
from .gene_resolver import GeneResolver

__all__ = ["GeneResolver", "DrugResolver"]
