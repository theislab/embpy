"""Single-cell foundation model wrappers via the Helical package.

Provides a unified interface for extracting cell embeddings from
state-of-the-art single-cell RNA-seq foundation models:

* **scGPT** -- 33M cells, transformer-based
* **Geneformer** -- 30-104M cells, multiple sizes and cancer-tuned variants
* **UCE** -- 36M cells, cross-species universal cell embedding
* **TranscriptFormer** -- 112M cells, cross-species generative atlas (CZI)
* **Tahoe-x1** -- cell + gene embeddings, 70M/1B/3B
* **Cell2Sentence-Scale** -- LLM-based, 2B/27B

All models are accessed through the `helical <https://github.com/helicalAI/helical>`_
package, which must be installed separately::

    pip install helical
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


def _require_helical():  # type: ignore[no-untyped-def]
    """Lazily import helical, raising a clear error if not installed."""
    try:
        import helical  # type: ignore[import-not-found]

        return helical
    except ImportError as exc:
        raise ImportError(
            "The 'helical' package is required for single-cell foundation "
            "models. Install with: pip install helical"
        ) from exc


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SCModelCard:
    """Metadata for a registered single-cell foundation model."""

    key: str
    wrapper_class_name: str
    description: str
    default_model_name: str | None = None
    embedding_dim: int | None = None
    variants: list[str] = field(default_factory=list)
    reference: str = ""


_SC_MODEL_REGISTRY: dict[str, SCModelCard] = {
    # --- scGPT ---
    "scgpt": SCModelCard(
        key="scgpt",
        wrapper_class_name="ScGPTWrapper",
        description="Transformer pre-trained on 33M+ human cells (Bo Wang Lab).",
        embedding_dim=512,
        reference="https://doi.org/10.1038/s41592-024-02201-0",
    ),
    # --- Geneformer v1 ---
    "geneformer_v1_6L": SCModelCard(
        key="geneformer_v1_6L",
        wrapper_class_name="GeneformerWrapper",
        description="Geneformer v1 (6-layer, 10M params, 2048 input).",
        default_model_name="gf-6L-10M-i2048",
        variants=["gf-6L-10M-i2048"],
        reference="https://doi.org/10.1038/s41586-023-06139-9",
    ),
    "geneformer_v1_12L": SCModelCard(
        key="geneformer_v1_12L",
        wrapper_class_name="GeneformerWrapper",
        description="Geneformer v1 (12-layer, 40M params, 2048 input).",
        default_model_name="gf-12L-40M-i2048",
        variants=["gf-12L-40M-i2048"],
        reference="https://doi.org/10.1038/s41586-023-06139-9",
    ),
    "geneformer_v1_12L_czi": SCModelCard(
        key="geneformer_v1_12L_czi",
        wrapper_class_name="GeneformerWrapper",
        description="Geneformer v1 (12-layer) fine-tuned by CZI CELLxGENE.",
        default_model_name="gf-12L-40M-i2048-CZI-CellxGene",
        variants=["gf-12L-40M-i2048-CZI-CellxGene"],
        reference="https://doi.org/10.1038/s41586-023-06139-9",
    ),
    # --- Geneformer v2 ---
    "geneformer_v2_12L": SCModelCard(
        key="geneformer_v2_12L",
        wrapper_class_name="GeneformerWrapper",
        description="Geneformer v2 (12-layer, 38M params, 4096 input, 95M cells).",
        default_model_name="gf-12L-38M-i4096",
        variants=["gf-12L-38M-i4096"],
        reference="https://doi.org/10.1101/2024.08.16.608180",
    ),
    "geneformer_v2_20L": SCModelCard(
        key="geneformer_v2_20L",
        wrapper_class_name="GeneformerWrapper",
        description="Geneformer v2 (20-layer, 151M params, 4096 input).",
        default_model_name="gf-20L-151M-i4096",
        variants=["gf-20L-151M-i4096"],
        reference="https://doi.org/10.1101/2024.08.16.608180",
    ),
    "geneformer_v2_12L_cancer": SCModelCard(
        key="geneformer_v2_12L_cancer",
        wrapper_class_name="GeneformerWrapper",
        description="Geneformer v2 (12-layer) cancer-tuned variant.",
        default_model_name="gf-12L-38M-i4096-CLcancer",
        variants=["gf-12L-38M-i4096-CLcancer"],
        reference="https://doi.org/10.1101/2024.08.16.608180",
    ),
    "geneformer_v2_12L_104M": SCModelCard(
        key="geneformer_v2_12L_104M",
        wrapper_class_name="GeneformerWrapper",
        description="Geneformer v2 (12-layer, 104M cells).",
        default_model_name="gf-12L-104M-i4096",
        variants=["gf-12L-104M-i4096"],
        reference="https://doi.org/10.1101/2024.08.16.608180",
    ),
    "geneformer_v2_12L_104M_cancer": SCModelCard(
        key="geneformer_v2_12L_104M_cancer",
        wrapper_class_name="GeneformerWrapper",
        description="Geneformer v2 (12-layer, 104M cells) cancer-tuned.",
        default_model_name="gf-12L-104M-i4096-CLcancer",
        variants=["gf-12L-104M-i4096-CLcancer"],
        reference="https://doi.org/10.1101/2024.08.16.608180",
    ),
    "geneformer_v2_18L": SCModelCard(
        key="geneformer_v2_18L",
        wrapper_class_name="GeneformerWrapper",
        description="Geneformer v2 (18-layer, 316M params, largest).",
        default_model_name="gf-18L-316M-i4096",
        variants=["gf-18L-316M-i4096"],
        reference="https://doi.org/10.1101/2024.08.16.608180",
    ),
    # --- UCE ---
    "uce": SCModelCard(
        key="uce",
        wrapper_class_name="UCEWrapper",
        description="Universal Cell Embedding (36M+ cells, cross-species).",
        embedding_dim=1280,
        reference="https://doi.org/10.1101/2023.11.28.568918",
    ),
    # --- TranscriptFormer ---
    "transcriptformer_metazoa": SCModelCard(
        key="transcriptformer_metazoa",
        wrapper_class_name="TranscriptFormerWrapper",
        description="TranscriptFormer-Metazoa (112M cells, 12 species, 444M params).",
        default_model_name="tf_metazoa",
        variants=["tf_metazoa"],
        reference="https://doi.org/10.1101/2025.04.25.650731",
    ),
    "transcriptformer_exemplar": SCModelCard(
        key="transcriptformer_exemplar",
        wrapper_class_name="TranscriptFormerWrapper",
        description="TranscriptFormer-Exemplar (110M cells, 5 species, 542M params).",
        default_model_name="tf_exemplar",
        variants=["tf_exemplar"],
        reference="https://doi.org/10.1101/2025.04.25.650731",
    ),
    "transcriptformer_sapiens": SCModelCard(
        key="transcriptformer_sapiens",
        wrapper_class_name="TranscriptFormerWrapper",
        description="TranscriptFormer-Sapiens (57M human cells, 368M params).",
        default_model_name="tf_sapiens",
        variants=["tf_sapiens"],
        reference="https://doi.org/10.1101/2025.04.25.650731",
    ),
    # --- Tahoe-x1 ---
    "tahoe_70m": SCModelCard(
        key="tahoe_70m",
        wrapper_class_name="TahoeWrapper",
        description="Tahoe-x1 70M parameter model.",
        default_model_name="70m",
        variants=["70m"],
    ),
    "tahoe_1b": SCModelCard(
        key="tahoe_1b",
        wrapper_class_name="TahoeWrapper",
        description="Tahoe-x1 1B parameter model.",
        default_model_name="1b",
        variants=["1b"],
    ),
    "tahoe_3b": SCModelCard(
        key="tahoe_3b",
        wrapper_class_name="TahoeWrapper",
        description="Tahoe-x1 3B parameter model.",
        default_model_name="3b",
        variants=["3b"],
    ),
    # --- Cell2Sentence-Scale ---
    "cell2sentence_2b": SCModelCard(
        key="cell2sentence_2b",
        wrapper_class_name="Cell2SentenceWrapper",
        description="Cell2Sentence-Scale 2B (Gemma-2-based LLM).",
        default_model_name="2B",
        variants=["2B"],
    ),
    "cell2sentence_27b": SCModelCard(
        key="cell2sentence_27b",
        wrapper_class_name="Cell2SentenceWrapper",
        description="Cell2Sentence-Scale 27B (Gemma-2-based LLM).",
        default_model_name="27B",
        variants=["27B"],
    ),
}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SingleCellWrapper(ABC):
    """Abstract base for single-cell foundation model wrappers.

    Unlike :class:`~embpy.models.base.BaseModelWrapper` (designed for
    string inputs like sequences or SMILES), single-cell models operate
    on :class:`~anndata.AnnData` gene-expression matrices and return
    per-cell embedding vectors.

    Subclasses must implement :meth:`load` and :meth:`embed_cells`.
    """

    model_type: Literal["single_cell"] = "single_cell"

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self.batch_size = batch_size
        self.device: str = "cpu"
        self._model: Any = None
        self._config: Any = None
        self._kwargs = kwargs

    @abstractmethod
    def load(self, device: str = "cpu") -> None:
        """Initialise the underlying helical model and move to *device*."""

    @abstractmethod
    def embed_cells(self, adata: Any) -> np.ndarray:
        """Compute cell embeddings from an AnnData.

        Parameters
        ----------
        adata : anndata.AnnData
            Gene-expression matrix (cells x genes). Raw counts are
            expected by most models.

        Returns
        -------
        np.ndarray of shape ``(n_cells, embedding_dim)``
        """

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the cell embeddings (0 if not loaded)."""
        return 0

    @property
    def model_name(self) -> str | None:
        """User-facing model name or variant identifier."""
        return self._model_name

    def __repr__(self) -> str:
        name = self._model_name or type(self).__name__
        return f"{type(self).__name__}(model_name={name!r}, device={self.device!r})"


# ---------------------------------------------------------------------------
# Concrete wrappers
# ---------------------------------------------------------------------------


class ScGPTWrapper(SingleCellWrapper):
    """Wrapper for the scGPT single-cell foundation model.

    Pre-trained on 33M+ human cells. Produces 512-dim cell embeddings.

    Example::

        wrapper = ScGPTWrapper(batch_size=10)
        wrapper.load("cuda")
        embs = wrapper.embed_cells(adata)  # (n_cells, 512)
    """

    def load(self, device: str = "cpu") -> None:  # noqa: D102
        _require_helical()
        from helical.models.scgpt import scGPT, scGPTConfig  # type: ignore[import-not-found]

        self.device = device
        self._config = scGPTConfig(batch_size=self.batch_size, device=device)
        self._model = scGPT(configurer=self._config)
        logger.info("Loaded scGPT on %s", device)

    def embed_cells(self, adata: Any) -> np.ndarray:  # noqa: D102
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        dataset = self._model.process_data(adata)
        embs = self._model.get_embeddings(dataset)
        return np.asarray(embs, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:  # noqa: D102
        return 512 if self._model is not None else 0


class GeneformerWrapper(SingleCellWrapper):
    """Wrapper for the Geneformer single-cell foundation model.

    Supports v1 (30M cells) and v2 (95-104M cells) variants, including
    cancer-tuned models. Pass the variant name via ``model_name``.

    Example::

        wrapper = GeneformerWrapper(model_name="gf-12L-38M-i4096")
        wrapper.load("cuda")
        embs = wrapper.embed_cells(adata)
    """

    def load(self, device: str = "cpu") -> None:  # noqa: D102
        _require_helical()
        from helical.models.geneformer import Geneformer, GeneformerConfig  # type: ignore[import-not-found]

        self.device = device
        model_name = self._model_name or "gf-12L-38M-i4096"
        self._config = GeneformerConfig(
            model_name=model_name,
            batch_size=self.batch_size,
            device=device,
        )
        self._model = Geneformer(configurer=self._config)
        logger.info("Loaded Geneformer '%s' on %s", model_name, device)

    def embed_cells(self, adata: Any) -> np.ndarray:  # noqa: D102
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        dataset = self._model.process_data(adata)
        embs = self._model.get_embeddings(dataset)
        return np.asarray(embs, dtype=np.float32)


class UCEWrapper(SingleCellWrapper):
    """Wrapper for Universal Cell Embedding (UCE).

    Trained on 36M+ cells across 8 species. Produces 1280-dim embeddings.

    Example::

        wrapper = UCEWrapper(batch_size=10)
        wrapper.load("cuda")
        embs = wrapper.embed_cells(adata)  # (n_cells, 1280)
    """

    def load(self, device: str = "cpu") -> None:  # noqa: D102
        _require_helical()
        from helical.models.uce import UCE, UCEConfig  # type: ignore[import-not-found]

        self.device = device
        self._config = UCEConfig(batch_size=self.batch_size, device=device)
        self._model = UCE(configurer=self._config)
        logger.info("Loaded UCE on %s", device)

    def embed_cells(self, adata: Any) -> np.ndarray:  # noqa: D102
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        dataset = self._model.process_data(adata)
        embs = self._model.get_embeddings(dataset)
        return np.asarray(embs, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:  # noqa: D102
        return 1280 if self._model is not None else 0


class TranscriptFormerWrapper(SingleCellWrapper):
    """Wrapper for TranscriptFormer (CZI cross-species generative model).

    Variants: ``tf_metazoa`` (112M cells, 12 species),
    ``tf_exemplar`` (5 species), ``tf_sapiens`` (human only).

    Example::

        wrapper = TranscriptFormerWrapper(model_name="tf_metazoa")
        wrapper.load("cuda")
        embs = wrapper.embed_cells(adata)
    """

    def load(self, device: str = "cpu") -> None:  # noqa: D102
        _require_helical()
        from helical.models.transcriptformer.model import TranscriptFormer  # type: ignore[import-not-found]
        from helical.models.transcriptformer.transcriptformer_config import (  # type: ignore[import-not-found]
            TranscriptFormerConfig,
        )

        self.device = device
        model_name = self._model_name or "tf_metazoa"
        self._config = TranscriptFormerConfig(
            model_name=model_name,
            batch_size=self.batch_size,
        )
        self._model = TranscriptFormer(configurer=self._config)
        logger.info("Loaded TranscriptFormer '%s' on %s", model_name, device)

    def embed_cells(self, adata: Any) -> np.ndarray:  # noqa: D102
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        dataset = self._model.process_data([adata])
        embs = self._model.get_embeddings(dataset)
        return np.asarray(embs, dtype=np.float32)


class TahoeWrapper(SingleCellWrapper):
    """Wrapper for Tahoe-x1 single-cell foundation model.

    Supports ``70m``, ``1b``, and ``3b`` model sizes. Extracts both
    cell and gene embeddings from raw count data.

    Example::

        wrapper = TahoeWrapper(model_name="1b")
        wrapper.load("cuda")
        embs = wrapper.embed_cells(adata)
    """

    def load(self, device: str = "cpu") -> None:  # noqa: D102
        _require_helical()
        from helical.models.tahoe import Tahoe, TahoeConfig  # type: ignore[import-not-found]

        self.device = device
        model_size = self._model_name or "70m"
        self._config = TahoeConfig(
            model_size=model_size,
            batch_size=self.batch_size,
            device=device,
        )
        self._model = Tahoe(configurer=self._config)
        logger.info("Loaded Tahoe-x1 '%s' on %s", model_size, device)

    def embed_cells(self, adata: Any) -> np.ndarray:  # noqa: D102
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        dataset = self._model.process_data(adata)
        embs = self._model.get_embeddings(dataset)
        return np.asarray(embs, dtype=np.float32)


class Cell2SentenceWrapper(SingleCellWrapper):
    """Wrapper for Cell2Sentence-Scale (LLM-based cell embeddings).

    Converts gene expression into ranked gene sentences and embeds them
    with a large language model. Variants: ``2B`` (Gemma-2 2B) and
    ``27B`` (Gemma-2 27B).

    Example::

        wrapper = Cell2SentenceWrapper(model_name="2B")
        wrapper.load("cuda")
        embs = wrapper.embed_cells(adata)
    """

    def load(self, device: str = "cpu") -> None:  # noqa: D102
        _require_helical()
        from helical.models.c2s import Cell2Sen, Cell2SenConfig  # type: ignore[import-not-found]

        self.device = device
        model_size = self._model_name or "2B"
        self._config = Cell2SenConfig(
            model_size=model_size,
            batch_size=self.batch_size,
            device=device,
        )
        self._model = Cell2Sen(configurer=self._config)
        logger.info("Loaded Cell2Sentence-Scale '%s' on %s", model_size, device)

    def embed_cells(self, adata: Any) -> np.ndarray:  # noqa: D102
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        dataset = self._model.process_data(adata)
        embs = self._model.get_embeddings(dataset)
        return np.asarray(embs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory / discovery
# ---------------------------------------------------------------------------

_WRAPPER_MAP: dict[str, type[SingleCellWrapper]] = {
    "ScGPTWrapper": ScGPTWrapper,
    "GeneformerWrapper": GeneformerWrapper,
    "UCEWrapper": UCEWrapper,
    "TranscriptFormerWrapper": TranscriptFormerWrapper,
    "TahoeWrapper": TahoeWrapper,
    "Cell2SentenceWrapper": Cell2SentenceWrapper,
}


def list_singlecell_models() -> list[str]:
    """Return the keys of all registered single-cell foundation models."""
    return list(_SC_MODEL_REGISTRY.keys())


def singlecell_info(key: str) -> SCModelCard:
    """Return the :class:`SCModelCard` for a registered model.

    Parameters
    ----------
    key
        Model key (e.g. ``"scgpt"``, ``"geneformer_v2_12L"``).
    """
    if key not in _SC_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown single-cell model {key!r}. "
            f"Available: {list_singlecell_models()}"
        )
    return _SC_MODEL_REGISTRY[key]


def get_singlecell_wrapper(
    key: str,
    *,
    batch_size: int = 32,
    **kwargs: Any,
) -> SingleCellWrapper:
    """Instantiate a single-cell model wrapper by registry key.

    Parameters
    ----------
    key
        Model key from :func:`list_singlecell_models`.
    batch_size
        Batch size for the helical model.
    **kwargs
        Additional arguments forwarded to the wrapper constructor.

    Returns
    -------
    :class:`SingleCellWrapper`
        An uninitialised wrapper. Call ``.load(device)`` before use.
    """
    card = singlecell_info(key)
    wrapper_cls = _WRAPPER_MAP[card.wrapper_class_name]
    return wrapper_cls(
        model_name=card.default_model_name,
        batch_size=batch_size,
        **kwargs,
    )
