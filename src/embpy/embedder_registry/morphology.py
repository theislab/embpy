"""Morphology registry entries (SubCell microscopy-image ViT-MAE variants)."""

from __future__ import annotations

from ..models.base import BaseModelWrapper
from ..models.morphology_models import SubCellWrapper


MORPHOLOGY_MODELS: dict[str, tuple[type[BaseModelWrapper] | None, str | None]] = {
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
}
