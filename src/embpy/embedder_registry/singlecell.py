"""Single-cell foundation-model registry entries.

This file is the future home for entries that today live in
``embpy.models.singlecell_models.SINGLECELL_MODEL_REGISTRY``. Single-cell
models bypass the general ``MODEL_REGISTRY`` pathway in
``BioEmbedder`` (they are looked up via
``BioEmbedder._get_or_load_singlecell_wrapper`` instead of
``_get_model``), so ``SINGLECELL_MODELS`` is intentionally empty in this
PR -- the per-modality split (audit step 3) only needs to preserve
byte-equivalence of the existing flat dict, which had no single-cell
entries.

When the single-cell facade is extracted (audit step 5 / the
``SingleCellEmbedder`` proposal in section 2), the entries from
``embpy.models.singlecell_models.SINGLECELL_MODEL_REGISTRY`` will move
here and the merge in ``embedder_registry/flat.py`` will pick them up
automatically.
"""

from __future__ import annotations

from ..models.base import BaseModelWrapper


SINGLECELL_MODELS: dict[str, tuple[type[BaseModelWrapper] | None, str | None]] = {}
