# pp -- preprocessing subpackages
#
# Domain subpackage structure:
#   pp/morphology/   -- single-cell morphology image preprocessing
#   pp/singlecell/   -- single-cell RNA-seq preprocessing

from .basic import PerturbationProcessor, reduce_embeddings
from .singlecell.preprocessing import preprocess_counts
from .depmap_handler import (
    DepMapDatasetCard,
    depmap_info,
    list_depmap_datasets,
    load_depmap,
)
from .hf_handler import HFHandler
from .lamin_handler import (
    LaminDatasetCard,
    lamin_info,
    list_lamin_datasets,
    load_lamin,
)
from .morphology.preprocessing import (
    CELL_PAINTING_CHANNELS,
    CELL_PAINTING_COLORS,
    SUBCELL_CANVAS_HEIGHT,
    SUBCELL_CANVAS_WIDTH,
    SUBCELL_CHANNELS,
    SUBCELL_TARGET_NM_PER_PIXEL,
    bbox_from_mask,
    cell_painting_to_subcell,
    composite_cell_painting,
    crop_spatial,
    crop_to_mask,
    load_channels_from_pngs,
    max_projection_z,
    max_projection_z_multichannel,
    normalize_channels,
    prepare_subcell_canvas,
    rescale_to_target_nm_per_pixel,
    resize_to_canvas,
    save_channels_as_pngs,
)

__all__ = [
    # data loaders
    "DepMapDatasetCard",
    "HFHandler",
    "LaminDatasetCard",
    "PerturbationProcessor",
    "depmap_info",
    "lamin_info",
    "list_depmap_datasets",
    "list_lamin_datasets",
    "load_depmap",
    "load_lamin",
    "preprocess_counts",
    "reduce_embeddings",
    # morphology preprocessing
    "CELL_PAINTING_CHANNELS",
    "CELL_PAINTING_COLORS",
    "SUBCELL_CANVAS_HEIGHT",
    "SUBCELL_CANVAS_WIDTH",
    "SUBCELL_CHANNELS",
    "SUBCELL_TARGET_NM_PER_PIXEL",
    "bbox_from_mask",
    "cell_painting_to_subcell",
    "composite_cell_painting",
    "crop_spatial",
    "crop_to_mask",
    "load_channels_from_pngs",
    "max_projection_z",
    "max_projection_z_multichannel",
    "normalize_channels",
    "prepare_subcell_canvas",
    "rescale_to_target_nm_per_pixel",
    "resize_to_canvas",
    "save_channels_as_pngs",
]
