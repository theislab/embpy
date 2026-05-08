"""AnnData-backed dataset for paired (basal, perturbed, condition) triplets.

The world model expects training tuples of the form

* ``z_basal``    -- latent of an unperturbed control cell,
* ``z_perturbed`` -- latent of a perturbed cell, and
* ``cond``       -- a perturbation embedding vector (e.g. a drug embedding,
  a one-hot gene-KO vector, a concatenated drug + dose vector, ...).

This module pairs cells from an :class:`anndata.AnnData` object that
already contains pre-computed cell embeddings in ``.obsm`` and a
perturbation embedding lookup, and produces PyTorch tensors. Because
true paired (basal, perturbed) measurements are rare, by default each
perturbed cell is matched against a *random* control cell at access
time; this approximates marginal-conditional matching used in cellflow-
style training.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _require_torch():  # type: ignore[no-untyped-def]
    try:
        import torch
        from torch.utils.data import Dataset  # noqa: F401

        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for embpy.world_model datasets. "
            "Install with: pip install torch"
        ) from exc


def _require_anndata():  # type: ignore[no-untyped-def]
    try:
        from anndata import AnnData

        return AnnData
    except ImportError as exc:
        raise ImportError(
            "anndata is required for embpy.world_model datasets. "
            "Install with: pip install anndata"
        ) from exc


class PerturbationLatentDataset:
    """Pairs perturbed cells with random control cells of the same context.

    Parameters
    ----------
    adata
        AnnData containing both control and perturbed cells.
    embedding_key
        Key in ``adata.obsm`` holding the per-cell latent matrix used as
        ``z`` (e.g. ``"X_scvi"``, ``"X_state"``).
    perturbation_key
        Column in ``adata.obs`` identifying the perturbation applied to
        each cell. Cells whose value equals ``control_label`` are treated
        as basal/controls.
    cond_embeddings
        Mapping from perturbation label to a 1-D embedding vector. Every
        non-control label appearing in ``adata.obs[perturbation_key]``
        must be present.
    control_label
        Value in ``adata.obs[perturbation_key]`` that marks control cells.
    context_key
        Optional column in ``adata.obs`` used to restrict pairing (e.g.
        ``"cell_type"`` or ``"batch"``): a perturbed cell is only paired
        with a control cell that shares the same context. Pass ``None``
        to disable.
    rng
        Random number generator used for control sampling. Defaults to a
        fresh :func:`numpy.random.default_rng`.

    Notes
    -----
    The class implements the :class:`torch.utils.data.Dataset` protocol
    (``__len__`` + ``__getitem__``) without inheriting from it, so the
    module can be imported even when PyTorch is missing. ``__getitem__``
    triggers a lazy ``torch`` import on first use.
    """

    def __init__(
        self,
        adata: Any,
        embedding_key: str,
        perturbation_key: str,
        cond_embeddings: Mapping[str, np.ndarray],
        control_label: str = "control",
        context_key: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        AnnData = _require_anndata()
        if not isinstance(adata, AnnData):
            raise TypeError(f"Expected AnnData, got {type(adata).__name__}")
        if embedding_key not in adata.obsm:
            raise KeyError(
                f"'{embedding_key}' not in adata.obsm. "
                f"Available: {list(adata.obsm.keys())}"
            )
        if perturbation_key not in adata.obs.columns:
            raise KeyError(
                f"'{perturbation_key}' not in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )

        self.embeddings: np.ndarray = np.asarray(adata.obsm[embedding_key], dtype=np.float32)
        self.perturbation_labels = np.asarray(adata.obs[perturbation_key].values)
        self.context_key = context_key
        self.contexts: np.ndarray | None = (
            np.asarray(adata.obs[context_key].values) if context_key is not None else None
        )
        self.control_label = control_label

        is_control = self.perturbation_labels == control_label
        self.control_idx = np.flatnonzero(is_control)
        self.perturbed_idx = np.flatnonzero(~is_control)

        if self.control_idx.size == 0:
            raise ValueError(
                f"No control cells found (no rows with {perturbation_key}=='{control_label}')."
            )
        if self.perturbed_idx.size == 0:
            raise ValueError("No perturbed cells found in this AnnData.")

        missing = set(self.perturbation_labels[self.perturbed_idx]) - set(cond_embeddings.keys())
        if missing:
            raise KeyError(
                f"Missing condition embeddings for {len(missing)} perturbations. "
                f"First few: {sorted(missing)[:5]}"
            )

        first_emb = next(iter(cond_embeddings.values()))
        cond_dim = int(np.asarray(first_emb).shape[-1])
        self.cond_dim = cond_dim

        self.cond_lookup: dict[str, np.ndarray] = {
            str(k): np.asarray(v, dtype=np.float32).reshape(-1) for k, v in cond_embeddings.items()
        }
        for k, v in self.cond_lookup.items():
            if v.shape[0] != cond_dim:
                raise ValueError(
                    f"Inconsistent condition embedding dim for '{k}': "
                    f"{v.shape[0]} vs expected {cond_dim}."
                )

        if self.contexts is not None:
            self._control_idx_by_context: dict[Any, np.ndarray] = {
                ctx: np.flatnonzero(is_control & (self.contexts == ctx))
                for ctx in np.unique(self.contexts)
            }
            empty = [ctx for ctx, idx in self._control_idx_by_context.items() if idx.size == 0]
            if empty:
                logger.warning(
                    "No controls available for contexts %s; those perturbed cells will be skipped.",
                    empty,
                )
        else:
            self._control_idx_by_context = {}

        self.rng = rng if rng is not None else np.random.default_rng()
        self.latent_dim = int(self.embeddings.shape[1])

        logger.info(
            "PerturbationLatentDataset: %d perturbed cells, %d controls, "
            "latent_dim=%d, cond_dim=%d.",
            self.perturbed_idx.size, self.control_idx.size, self.latent_dim, self.cond_dim,
        )

    # ------------------------------------------------------------------
    # PyTorch Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self.perturbed_idx.size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = _require_torch()
        pert_row = int(self.perturbed_idx[index])
        label = str(self.perturbation_labels[pert_row])

        if self.contexts is not None:
            ctx = self.contexts[pert_row]
            candidates = self._control_idx_by_context.get(ctx, self.control_idx)
            if candidates.size == 0:
                candidates = self.control_idx
        else:
            candidates = self.control_idx

        ctrl_row = int(self.rng.choice(candidates))

        return {
            "z_basal": torch.from_numpy(self.embeddings[ctrl_row]),
            "z_perturbed": torch.from_numpy(self.embeddings[pert_row]),
            "cond": torch.from_numpy(self.cond_lookup[label]),
            "perturbation": label,
        }


__all__ = ["PerturbationLatentDataset"]
