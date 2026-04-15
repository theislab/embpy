# This file makes 'resources' a package.
#
# Domain subpackages:
#   resources.protein   -- ProteinResolver, ProteinAnnotator, OrthologResolver
#   resources.molecule  -- DrugResolver, MoleculeAnnotator
#   resources.gene      -- GeneResolver, GeneAnnotator
#   resources.morphology -- HPA images, JUMP metadata
#   resources.text      -- TextResolver
#   resources.cellline  -- CellLineAnnotator

from .cellline import CellLineAnnotator
from .gene import GeneAnnotator, GeneResolver, detect_identifier_type
from .molecule import DrugResolver, MoleculeAnnotator
from .morphology import (
    HPA_IF_CHANNELS,
    HPA_IMAGE_BASE,
    build_hpa_subcellular_catalog,
    download_hpa_subcellular_images,
    fetch_hpa_if_image,
    get_hpa_antibodies,
    load_hpa_if_image,
    strip_antibody_id,
    fetch_jump_fov,
    get_jump_gene_mapper,
    get_jump_item_location_metadata,
)
from .protein import (
    OrthologResolver,
    OrthologResult,
    ProteinAnnotator,
    ProteinResolver,
)
from .text import TextResolver

__all__ = [
    "CellLineAnnotator",
    "DrugResolver",
    "GeneAnnotator",
    "GeneResolver",
    "HPA_IF_CHANNELS",
    "HPA_IMAGE_BASE",
    "OrthologResolver",
    "OrthologResult",
    "ProteinAnnotator",
    "ProteinResolver",
    "MoleculeAnnotator",
    "TextResolver",
    "build_hpa_subcellular_catalog",
    "detect_identifier_type",
    "download_hpa_subcellular_images",
    "fetch_hpa_if_image",
    "fetch_jump_fov",
    "get_hpa_antibodies",
    "get_jump_gene_mapper",
    "get_jump_item_location_metadata",
    "load_hpa_if_image",
    "strip_antibody_id",
]
