# pp/singlecell -- single-cell RNA-seq preprocessing
#
# Public API re-exported from the .preprocessing module.

from .preprocessing import preprocess_counts

__all__ = [
    "preprocess_counts",
]
