"""Single-cell foundation model wrappers.

Provides a unified interface for extracting cell embeddings from
state-of-the-art single-cell RNA-seq foundation models as well as for
decoding embeddings back into gene-expression space (for encoder-decoder
models).

Encoder-only foundation models (via `helical <https://github.com/helicalAI/helical>`_,
``pip install helical``):

* **scGPT** -- 33M cells, transformer-based
* **Geneformer** -- 30-104M cells, multiple sizes and cancer-tuned variants
* **UCE** -- 36M cells, cross-species universal cell embedding
* **TranscriptFormer** -- 112M cells, cross-species generative atlas (CZI)
* **Tahoe-x1** -- cell + gene embeddings, 70M/1B/3B
* **Cell2Sentence-Scale** -- LLM-based, 2B/27B

Encoder-decoder models (usable in a basal -> perturbed flow-matching /
``cellflow``-style setup: encode cells to latent, predict latent shift,
then decode back to expression):

* **PCA** -- classical baseline (sklearn / cuml)
* **scVI family** (scvi-tools) -- scVI, scANVI, totalVI
* **STATE** (Arc Institute, ``pip install arc-state``) -- SE-600M
  transformer encoder with a binary decoder head that maps latents to
  per-gene log-probabilities.
* **Stack** (Arc Institute, ``pip install arc-stack``) -- tabular-attention
  encoder plus an in-context generation head that synthesises new cell
  profiles conditioned on a base-context AnnData.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
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


def _install_flash_attn_shim() -> None:
    """Install a pure-PyTorch shim for the subset of `flash_attn` Tahoe uses.

    Tahoe-x1 vendors LLM-Foundry code whose `modeling_mpt.py` does
    `from flash_attn import bert_padding` at module import time and then,
    inside every forward pass, calls `bert_padding.unpad_input(...)` --
    even when `attn_impl="torch"`. If `flash_attn` is absent, Tahoe cannot
    even load; if it is present as an empty stub, loading succeeds but
    inference crashes with `AttributeError: module 'flash_attn.bert_padding'
    has no attribute 'unpad_input'`.

    Installing real `flash_attn` is impractical in generic environments
    (requires custom CUDA build), so we install a minimal functional shim
    that implements the two functions Tahoe actually calls
    (`unpad_input` and `pad_input`) in pure PyTorch. The fast CUDA path is
    never taken under `attn_impl="torch"`, so this is exercised only as a
    padding utility and is numerically identical to the real thing.
    """
    import sys
    import types

    try:
        import flash_attn  # type: ignore[import-not-found]

        # Real flash_attn present -- only shim if bert_padding is broken.
        if hasattr(flash_attn, "bert_padding") and hasattr(
            flash_attn.bert_padding, "unpad_input"
        ):
            return
    except ImportError:
        pass

    import torch

    def unpad_input(hidden_states: "torch.Tensor", attention_mask: "torch.Tensor"):
        """Flatten padded (batch, seqlen, ...) tensors down to non-pad tokens.

        Mirrors ``flash_attn.bert_padding.unpad_input``. Returns
        ``(hidden_states_flat, indices, cu_seqlens, max_seqlen_in_batch)``.
        """
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = int(seqlens_in_batch.max().item()) if seqlens_in_batch.numel() else 0
        cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
        )
        flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
        flat_unpad = flat.index_select(0, indices)
        return flat_unpad, indices, cu_seqlens, max_seqlen_in_batch

    def pad_input(hidden_states: "torch.Tensor", indices: "torch.Tensor", batch: int, seqlen: int):
        """Inverse of ``unpad_input``. Mirrors ``flash_attn.bert_padding.pad_input``."""
        output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
        output.index_copy_(0, indices, hidden_states)
        return output.reshape(batch, seqlen, *hidden_states.shape[1:])

    def index_first_axis(hidden_states: "torch.Tensor", indices: "torch.Tensor"):
        """Mirrors ``flash_attn.bert_padding.index_first_axis``."""
        return hidden_states.index_select(0, indices)

    def unpad_input_for_concatenated_sequences(*args, **kwargs):  # pragma: no cover
        raise NotImplementedError(
            "unpad_input_for_concatenated_sequences is not implemented in the "
            "flash_attn shim. This path is only triggered by packed-sequence "
            "inputs, which Tahoe does not use with attn_impl='torch'."
        )

    bert_padding = types.ModuleType("flash_attn.bert_padding")
    bert_padding.unpad_input = unpad_input  # type: ignore[attr-defined]
    bert_padding.pad_input = pad_input  # type: ignore[attr-defined]
    bert_padding.index_first_axis = index_first_axis  # type: ignore[attr-defined]
    bert_padding.unpad_input_for_concatenated_sequences = (  # type: ignore[attr-defined]
        unpad_input_for_concatenated_sequences
    )

    flash_attn_mod = types.ModuleType("flash_attn")
    flash_attn_mod.bert_padding = bert_padding  # type: ignore[attr-defined]

    sys.modules["flash_attn"] = flash_attn_mod
    sys.modules["flash_attn.bert_padding"] = bert_padding
    logger.debug("Installed pure-torch flash_attn.bert_padding shim for Tahoe-x1.")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SCModelCard:
    """Metadata for a registered single-cell foundation model.

    Attributes
    ----------
    vocab_type
        Gene-identifier convention expected by the underlying model:

        - ``"symbol"``  -- wants gene symbols (e.g. ``TP53``). Used by
          scGPT, UCE, STATE, STACK, Cell2Sentence. Ensembl IDs produce
          zero matches and the model silently returns empty embeddings.
        - ``"ensembl_id"`` -- wants Ensembl gene IDs (e.g. ``ENSG00000141510``).
        - ``"either"`` -- model accepts either format, typically because
          helical performs an internal symbol->Ensembl mapping
          (Geneformer, TranscriptFormer, Tahoe).
        - ``"any"`` -- model is gene-identifier agnostic (PCA, scVI family).

        This metadata drives the automatic ``BioEmbedder.embed_cells``
        vocabulary-conversion step.
    supports_decode
        ``True`` when the wrapper implements :meth:`SingleCellWrapper.decode_cells`
        (i.e. the model is an encoder-decoder whose latent space can be
        projected back to gene-expression). Currently: PCA, scVI family,
        STATE.
    supports_generation
        ``True`` when the wrapper implements :meth:`SingleCellWrapper.generate_cells`
        (i.e. the model can synthesise whole cell profiles, typically
        conditioned on a base context). Currently: Stack.
    """

    key: str
    wrapper_class_name: str
    description: str
    default_model_name: str | None = None
    embedding_dim: int | None = None
    variants: list[str] = field(default_factory=list)
    reference: str = ""
    vocab_type: Literal["symbol", "ensembl_id", "either", "any"] = "symbol"
    supports_decode: bool = False
    supports_generation: bool = False


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
        vocab_type="either",
        description="Geneformer v1 (6-layer, 10M params, 2048 input).",
        default_model_name="gf-6L-10M-i2048",
        variants=["gf-6L-10M-i2048"],
        reference="https://doi.org/10.1038/s41586-023-06139-9",
    ),
    "geneformer_v1_12L": SCModelCard(
        key="geneformer_v1_12L",
        wrapper_class_name="GeneformerWrapper",
        vocab_type="either",
        description="Geneformer v1 (12-layer, 40M params, 2048 input).",
        default_model_name="gf-12L-40M-i2048",
        variants=["gf-12L-40M-i2048"],
        reference="https://doi.org/10.1038/s41586-023-06139-9",
    ),
    "geneformer_v1_12L_czi": SCModelCard(
        key="geneformer_v1_12L_czi",
        wrapper_class_name="GeneformerWrapper",
        vocab_type="either",
        description="Geneformer v1 (12-layer) fine-tuned by CZI CELLxGENE.",
        default_model_name="gf-12L-40M-i2048-CZI-CellxGene",
        variants=["gf-12L-40M-i2048-CZI-CellxGene"],
        reference="https://doi.org/10.1038/s41586-023-06139-9",
    ),
    # --- Geneformer v2 ---
    "geneformer_v2_12L": SCModelCard(
        key="geneformer_v2_12L",
        wrapper_class_name="GeneformerWrapper",
        vocab_type="either",
        description="Geneformer v2 (12-layer, 38M params, 4096 input, 95M cells).",
        default_model_name="gf-12L-38M-i4096",
        variants=["gf-12L-38M-i4096"],
        reference="https://doi.org/10.1101/2024.08.16.608180",
    ),
    "geneformer_v2_20L": SCModelCard(
        key="geneformer_v2_20L",
        wrapper_class_name="GeneformerWrapper",
        vocab_type="either",
        description="Geneformer v2 (20-layer, 151M params, 4096 input).",
        default_model_name="gf-20L-151M-i4096",
        variants=["gf-20L-151M-i4096"],
        reference="https://doi.org/10.1101/2024.08.16.608180",
    ),
    "geneformer_v2_12L_cancer": SCModelCard(
        key="geneformer_v2_12L_cancer",
        wrapper_class_name="GeneformerWrapper",
        vocab_type="either",
        description="Geneformer v2 (12-layer) cancer-tuned variant.",
        default_model_name="gf-12L-38M-i4096-CLcancer",
        variants=["gf-12L-38M-i4096-CLcancer"],
        reference="https://doi.org/10.1101/2024.08.16.608180",
    ),
    "geneformer_v2_12L_104M": SCModelCard(
        key="geneformer_v2_12L_104M",
        wrapper_class_name="GeneformerWrapper",
        vocab_type="either",
        description="Geneformer v2 (12-layer, 104M cells).",
        default_model_name="gf-12L-104M-i4096",
        variants=["gf-12L-104M-i4096"],
        reference="https://doi.org/10.1101/2024.08.16.608180",
    ),
    "geneformer_v2_12L_104M_cancer": SCModelCard(
        key="geneformer_v2_12L_104M_cancer",
        wrapper_class_name="GeneformerWrapper",
        vocab_type="either",
        description="Geneformer v2 (12-layer, 104M cells) cancer-tuned.",
        default_model_name="gf-12L-104M-i4096-CLcancer",
        variants=["gf-12L-104M-i4096-CLcancer"],
        reference="https://doi.org/10.1101/2024.08.16.608180",
    ),
    "geneformer_v2_18L": SCModelCard(
        key="geneformer_v2_18L",
        wrapper_class_name="GeneformerWrapper",
        vocab_type="either",
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
        vocab_type="either",
        description="TranscriptFormer-Metazoa (112M cells, 12 species, 444M params).",
        default_model_name="tf_metazoa",
        variants=["tf_metazoa"],
        reference="https://doi.org/10.1101/2025.04.25.650731",
    ),
    "transcriptformer_exemplar": SCModelCard(
        key="transcriptformer_exemplar",
        wrapper_class_name="TranscriptFormerWrapper",
        vocab_type="either",
        description="TranscriptFormer-Exemplar (110M cells, 5 species, 542M params).",
        default_model_name="tf_exemplar",
        variants=["tf_exemplar"],
        reference="https://doi.org/10.1101/2025.04.25.650731",
    ),
    "transcriptformer_sapiens": SCModelCard(
        key="transcriptformer_sapiens",
        wrapper_class_name="TranscriptFormerWrapper",
        vocab_type="either",
        description="TranscriptFormer-Sapiens (57M human cells, 368M params).",
        default_model_name="tf_sapiens",
        variants=["tf_sapiens"],
        reference="https://doi.org/10.1101/2025.04.25.650731",
    ),
    # --- Tahoe-x1 ---
    "tahoe_70m": SCModelCard(
        key="tahoe_70m",
        wrapper_class_name="TahoeWrapper",
        vocab_type="either",
        description="Tahoe-x1 70M parameter model.",
        default_model_name="70m",
        variants=["70m"],
    ),
    "tahoe_1b": SCModelCard(
        key="tahoe_1b",
        wrapper_class_name="TahoeWrapper",
        vocab_type="either",
        description="Tahoe-x1 1B parameter model.",
        default_model_name="1b",
        variants=["1b"],
    ),
    "tahoe_3b": SCModelCard(
        key="tahoe_3b",
        wrapper_class_name="TahoeWrapper",
        vocab_type="either",
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
    # --- STATE (Arc Institute) ---
    "state": SCModelCard(
        key="state",
        wrapper_class_name="StateEmbeddingWrapper",
        description="STATE embedding model (Arc Institute, SE-600M).",
        reference="https://github.com/ArcInstitute/state",
        supports_decode=True,
    ),
    # --- Stack (Arc Institute) ---
    "stack": SCModelCard(
        key="stack",
        wrapper_class_name="StackWrapper",
        description="Stack encoder-decoder (150M+ cells, tabular attention, Arc Institute).",
        reference="https://github.com/ArcInstitute/stack",
        supports_generation=True,
    ),
    # --- PCA (classical baseline) ---
    "pca": SCModelCard(
        key="pca",
        wrapper_class_name="PCAEmbedding",
        vocab_type="any",
        description="PCA on the expression matrix (classical baseline).",
        supports_decode=True,
    ),
    # --- scvi-tools ---
    "scvi": SCModelCard(
        key="scvi",
        wrapper_class_name="ScVIToolsWrapper",
        vocab_type="any",
        description="scVI variational autoencoder (scvi-tools).",
        default_model_name="SCVI",
        supports_decode=True,
    ),
    "scanvi": SCModelCard(
        key="scanvi",
        wrapper_class_name="ScVIToolsWrapper",
        vocab_type="any",
        description="scANVI semi-supervised VAE (scvi-tools).",
        default_model_name="SCANVI",
        supports_decode=True,
    ),
    "totalvi": SCModelCard(
        key="totalvi",
        wrapper_class_name="ScVIToolsWrapper",
        vocab_type="any",
        description="totalVI joint RNA+protein VAE (scvi-tools).",
        default_model_name="TOTALVI",
        supports_decode=True,
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
    Encoder-decoder models may additionally override :meth:`decode_cells`
    (latent -> gene-expression) and/or :meth:`generate_cells` (in-context
    synthesis of whole cell profiles).
    """

    model_type: Literal["single_cell"] = "single_cell"

    # Capability flags -- subclasses override. Keeps the runtime check
    # consistent with the SCModelCard metadata.
    supports_decode: bool = False
    supports_generation: bool = False

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

    def decode_cells(
        self,
        latent: np.ndarray,
        *,
        gene_names: Sequence[str] | None = None,
        adata: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Decode latent cell representations back to gene-expression space.

        This method is only implemented by encoder-decoder wrappers
        (PCA, scVI family, STATE). For foundation encoders without a
        reusable decoder head (scGPT, Geneformer, UCE, ...), calling
        this raises :class:`NotImplementedError`.

        Parameters
        ----------
        latent : np.ndarray of shape ``(n_cells, embedding_dim)``
            Cell-level latents produced by :meth:`embed_cells` or by an
            external model (e.g. a flow-matching network trained in the
            latent space).
        gene_names : sequence of str or None
            Genes to decode to. ``None`` means "whatever the fitted
            decoder produces" (i.e. the genes used at training/embed
            time). Required for STATE because its decoder is
            parametrised by per-gene protein embeddings.
        adata : anndata.AnnData or None
            Optional original AnnData (used by scvi-tools to recover
            per-cell library size and batch/label obs columns when
            decoding).
        **kwargs
            Additional model-specific options.

        Returns
        -------
        np.ndarray of shape ``(n_cells, n_genes)``
            Decoded gene-expression (normalized-expression, log-probs, or
            reconstructed linear-scale values; see subclass docstring).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement decode_cells(). "
            "Only encoder-decoder wrappers (pca, scvi, scanvi, totalvi, "
            "state) expose a decoder."
        )

    def generate_cells(
        self,
        base_adata: Any,
        test_adata: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate new cell profiles via in-context generation.

        Currently only :class:`StackWrapper` implements this: it uses
        ``base_adata`` as a donor-conditioned context and predicts
        gene-expression for each cell in ``test_adata`` using the same
        gene list as at training time.

        Returns
        -------
        np.ndarray of shape ``(n_test_cells, n_genes)``
            Generated gene-expression values aligned to ``test_adata.var``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate_cells(). "
            "Only in-context generation wrappers (stack) expose this."
        )

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
        _install_flash_attn_shim()

        from helical.models.tahoe import Tahoe, TahoeConfig  # type: ignore[import-not-found]

        self.device = device
        model_size = self._model_name or "70m"
        self._config = TahoeConfig(
            model_size=model_size,
            batch_size=self.batch_size,
            device=device,
            attn_impl="torch",
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


class StateEmbeddingWrapper(SingleCellWrapper):
    """Wrapper for STATE embedding model (Arc Institute).

    STATE (State Transition / Embedding) is a foundation model for
    predicting cellular perturbation responses. The SE (State Embedding)
    component produces cell embeddings from scRNA-seq count data and
    exposes a *binary decoder* that maps latents back to per-gene
    log-probabilities, which is what makes STATE usable as the encoder/
    decoder pair in flow-matching (cellflow-style) perturbation
    pipelines.

    Encoder path: :meth:`embed_cells` -> ``state.emb.Inference.encode_adata``
    (concatenates the per-cell ``emb`` with the per-dataset ``ds_emb``).

    Decoder path: :meth:`decode_cells` -> ``Inference.decode_from_adata``.
    The decoder takes a latent matrix and a list of target genes and
    returns per-cell log-probabilities of shape ``(n_cells, n_genes)``.
    The target genes are embedded on-the-fly via the ``protein_embeds``
    dictionary loaded at :meth:`load` time, so you can decode to
    arbitrary gene panels (not only the genes that appeared at
    encode-time).

    Requires the ``arc-state`` package::

        pip install arc-state

    Parameters
    ----------
    checkpoint : str
        Path to the trained STATE ``.ckpt`` checkpoint file.
    model_folder : str or None
        Path to the model folder containing the checkpoint and optional
        ``protein_embeddings.pt``. If provided, checkpoint is auto-detected.
    protein_embeddings : str or None
        Path to a ``.pt`` file with protein embeddings. Auto-detected
        from ``model_folder`` if not given.
    config : str or None
        Path to a YAML config override. If omitted, uses config
        embedded in the checkpoint.

    Example::

        wrapper = StateEmbeddingWrapper(
            checkpoint="/path/to/SE-600M/se600m_epoch15.ckpt",
            model_folder="/path/to/SE-600M",
        )
        wrapper.load("cuda")
        z = wrapper.embed_cells(adata)
        # decode to the same gene panel that was used at encode time
        logprobs = wrapper.decode_cells(z, gene_names=adata.var_names)
    """

    supports_decode: bool = True

    def __init__(
        self,
        checkpoint: str | None = None,
        model_folder: str | None = None,
        protein_embeddings: str | None = None,
        config: str | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name or "state", **kwargs)
        self._checkpoint = checkpoint
        self._model_folder = model_folder
        self._protein_embeddings_path = protein_embeddings
        self._config_path = config
        self._inferer: Any = None

    def load(self, device: str = "cpu") -> None:  # noqa: D102
        try:
            from state.emb import Inference  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "The 'arc-state' package is required for STATE embeddings. "
                "Install with: pip install arc-state"
            ) from exc

        import glob
        import os
        import torch
        from omegaconf import OmegaConf  # type: ignore[import-not-found]

        self.device = device

        protein_embeds = None
        if self._protein_embeddings_path:
            protein_embeds = torch.load(
                self._protein_embeddings_path, weights_only=False, map_location="cpu",
            )
        elif self._model_folder:
            pe_path = os.path.join(self._model_folder, "protein_embeddings.pt")
            if os.path.exists(pe_path):
                protein_embeds = torch.load(pe_path, weights_only=False, map_location="cpu")

        cfg = OmegaConf.load(self._config_path) if self._config_path else None
        self._inferer = Inference(cfg=cfg, protein_embeds=protein_embeds)

        checkpoint = self._checkpoint
        if checkpoint is None and self._model_folder:
            ckpts = sorted(glob.glob(os.path.join(self._model_folder, "*.ckpt")))
            if not ckpts:
                raise FileNotFoundError(
                    f"No .ckpt files found in {self._model_folder}"
                )
            checkpoint = ckpts[-1]
        if checkpoint is None:
            raise ValueError(
                "Either checkpoint or model_folder must be provided."
            )

        self._inferer.load_model(checkpoint)
        self._model = self._inferer.model
        logger.info("Loaded STATE embedding model from %s on %s", checkpoint, device)

    def embed_cells(self, adata: Any) -> np.ndarray:  # noqa: D102
        if self._inferer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "input.h5ad")
            adata.write_h5ad(tmp_path)
            embeddings = self._inferer.encode_adata(
                input_adata_path=tmp_path,
                output_adata_path=None,
                emb_key="X_state",
            )
        return np.asarray(embeddings, dtype=np.float32)

    def decode_cells(  # noqa: D102
        self,
        latent: np.ndarray,
        *,
        gene_names: Sequence[str] | None = None,
        adata: Any = None,
        read_depth: float = 4.0,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> np.ndarray:
        """Decode STATE latents back to per-gene log-probabilities.

        Wraps :meth:`state.emb.Inference.decode_from_adata`. The latent
        matrix should be the exact output of :meth:`embed_cells` (i.e.
        per-cell ``emb`` concatenated with per-dataset ``ds_emb``) or a
        predicted latent from a downstream flow-matching model.

        Parameters
        ----------
        latent : np.ndarray of shape ``(n_cells, emb_dim + ds_emb_dim)``
            Cell-level latents. The last ``model.z_dim_ds`` columns are
            treated as the dataset embedding, consistent with how STATE
            concatenates them at encode time.
        gene_names : sequence of str
            Target genes to decode to. Must be provided because STATE's
            decoder is parametrised by per-gene protein embeddings; the
            loader looks each gene up in ``self.protein_embeds`` (missing
            genes get a zero vector).
        adata : anndata.AnnData, optional
            If supplied its ``var_names`` are used when ``gene_names`` is
            ``None``.
        read_depth : float
            Desired task read-depth passed to the decoder (default 4.0,
            matching STATE's own RDA default).
        batch_size : int
            Decoder batch size.
        """
        del kwargs  # unused
        if self._inferer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if gene_names is None and adata is not None:
            gene_names = list(adata.var_names)
        if gene_names is None:
            raise ValueError(
                "StateEmbeddingWrapper.decode_cells requires `gene_names` "
                "(or an `adata` whose var_names will be used). STATE's "
                "decoder is gene-parametric and cannot infer a target "
                "panel on its own."
            )

        import anndata as ad
        import pandas as pd

        latent_arr = np.asarray(latent, dtype=np.float32)
        n_cells = latent_arr.shape[0]

        # Build a minimal AnnData scaffold whose obsm carries the latent
        # and whose var_names are the target genes.
        scaffold = ad.AnnData(
            X=np.zeros((n_cells, len(gene_names)), dtype=np.float32),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
            var=pd.DataFrame(index=list(gene_names)),
        )
        emb_key = "X_state_decode"
        scaffold.obsm[emb_key] = latent_arr

        batches = list(
            self._inferer.decode_from_adata(
                scaffold,
                gene_names,
                emb_key=emb_key,
                read_depth=read_depth,
                batch_size=batch_size,
            )
        )
        if not batches:
            return np.empty((n_cells, len(gene_names)), dtype=np.float32)

        concatenated = np.concatenate(
            [np.atleast_2d(np.asarray(b, dtype=np.float32)) for b in batches],
            axis=0,
        )
        # Some checkpoints emit a trailing singleton dim; normalise.
        if concatenated.ndim == 3 and concatenated.shape[-1] == 1:
            concatenated = concatenated[..., 0]
        return concatenated


class StackWrapper(SingleCellWrapper):
    """Wrapper for Stack single-cell foundation model (Arc Institute).

    Stack is a large-scale encoder-decoder model trained on 150M+
    single cells using tabular attention. It supports in-context
    learning and produces cell embeddings from raw count data.

    Encoder path: :meth:`embed_cells` -> ``stack.cli.embedding.extract_embeddings``.

    Generation path: :meth:`generate_cells` -> Stack's in-context
    generation head (``model.get_incontext_generation``) via the same
    wrapper that ``stack-generation`` uses on the CLI. Unlike a pure
    latent-to-expression decoder, Stack synthesises cell profiles by
    conditioning on a *base-context* AnnData: give it a donor-specific
    base adata and a *test* adata whose cells you want to predict, and
    it returns a ``(n_test_cells, n_genes)`` matrix aligned to the
    training gene list.

    Requires the ``arc-stack`` package::

        pip install arc-stack

    Parameters
    ----------
    checkpoint : str
        Path to the trained Stack ``.ckpt`` checkpoint file.
    genelist : str
        Path to the pickled gene list used during training.
    gene_name_col : str or None
        Column in ``adata.var`` containing gene symbols. If ``None``,
        auto-detected from the overlap with the model's gene list.

    Example::

        wrapper = StackWrapper(
            checkpoint="/path/to/bc_large.ckpt",
            genelist="/path/to/basecount_1000per_15000max.pkl",
        )
        wrapper.load("cuda")
        embs = wrapper.embed_cells(adata)
        preds = wrapper.generate_cells(base_adata, test_adata)
    """

    supports_generation: bool = True

    def __init__(
        self,
        checkpoint: str | None = None,
        genelist: str | None = None,
        gene_name_col: str | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name or "stack", **kwargs)
        self._checkpoint = checkpoint
        self._genelist = genelist
        self._gene_name_col = gene_name_col

    def load(self, device: str = "cpu") -> None:  # noqa: D102
        try:
            from stack.cli.embedding import _ensure_deps, _load_model, _resolve_device  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "The 'arc-stack' package is required for Stack embeddings. "
                "Install with: pip install arc-stack"
            ) from exc

        if self._checkpoint is None:
            raise ValueError("checkpoint path is required for StackWrapper.")
        if self._genelist is None:
            raise ValueError("genelist path is required for StackWrapper.")

        self.device = device
        _ensure_deps()
        resolved = _resolve_device(device)
        self._model = _load_model(self._checkpoint, device=resolved)
        logger.info("Loaded Stack model from %s on %s", self._checkpoint, device)

    # STACK's TestSamplerDataset filters obs["organism"] against the
    # *exact* string "Homo sapiens" (see
    # stack/data/training/datasets.py:817 in arc-stack 0.1.x). Datasets
    # that encode the species with a common shorthand -- e.g. Replogle
    # K562 stores "human", many in-house pipelines write "Mouse" --
    # would otherwise produce zero matches and STACK aborts with
    # ValueError("No Homo sapiens cells found in the file"). We
    # canonicalise these aliases on a captured copy of the obs column
    # before writing the temp h5ad and restore the caller's value on the
    # way out so their in-memory AnnData is not mutated.
    _STACK_ORGANISM_ALIASES: dict[str, str] = {
        "human": "Homo sapiens",
        "homo sapiens": "Homo sapiens",
        "homo_sapiens": "Homo sapiens",
        "h sapiens": "Homo sapiens",
        "h. sapiens": "Homo sapiens",
        "h_sapiens": "Homo sapiens",
        "mouse": "Mus musculus",
        "mus musculus": "Mus musculus",
        "mus_musculus": "Mus musculus",
        "m musculus": "Mus musculus",
        "m. musculus": "Mus musculus",
    }

    def embed_cells(self, adata: Any) -> np.ndarray:  # noqa: D102
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        try:
            from stack.cli.embedding import extract_embeddings  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "The 'arc-stack' package is required. "
                "Install with: pip install arc-stack"
            ) from exc

        import tempfile
        import os

        # Capture-and-restore so we never silently mutate the caller's
        # AnnData. We only touch the obs frame, not X / var / layers, so
        # the operation is O(n_obs) in time and memory regardless of the
        # underlying matrix size.
        original_organism = None
        had_organism_col = (
            hasattr(adata, "obs")
            and hasattr(adata.obs, "columns")
            and "organism" in adata.obs.columns
        )
        if had_organism_col:
            original_organism = adata.obs["organism"].copy()
            raw = adata.obs["organism"].astype(str)
            lowered = raw.str.strip().str.lower()
            mapped = lowered.map(self._STACK_ORGANISM_ALIASES).fillna(raw)
            if not mapped.equals(raw):
                logger.info(
                    "StackWrapper: canonicalising obs['organism'] aliases "
                    "(e.g. 'human' -> 'Homo sapiens') for STACK's "
                    "filter_organism check."
                )
                adata.obs["organism"] = mapped.astype("category")

        # STACK's get_gene_names_from_h5 (arc-stack 0.1.x) only looks
        # for the literal keys "_index" / "index" inside the h5ad's var
        # group. AnnData, by contrast, stores the index data under a key
        # named after var.index.name (e.g. "gene_name" for Replogle) and
        # records the actual name in var.attrs["_index"]. STACK does not
        # read that attribute, so when var.index.name is anything other
        # than "_index" / "index" / None, STACK raises
        # ValueError("Could not find gene names in the file"). We detect
        # this and pass the index's real name through as gene_name_col
        # so STACK's first lookup branch (gene_name_col in var_group)
        # succeeds. Users who explicitly set gene_name_col override this
        # auto-detect.
        effective_gene_name_col = self._gene_name_col
        if effective_gene_name_col is None and hasattr(adata, "var"):
            idx = getattr(adata.var, "index", None)
            idx_name = getattr(idx, "name", None)
            if idx_name and idx_name not in {"_index", "index"}:
                effective_gene_name_col = idx_name
                logger.info(
                    "StackWrapper: forwarding adata.var.index.name=%r as "
                    "gene_name_col to STACK to bypass its naive index "
                    "lookup (AnnData stores the index under that key).",
                    idx_name,
                )

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = os.path.join(tmpdir, "input.h5ad")
                adata.write_h5ad(tmp_path)
                embeddings, _ = extract_embeddings(
                    checkpoint_path=self._checkpoint,
                    adata_path=tmp_path,
                    genelist_path=self._genelist,
                    gene_name_col=effective_gene_name_col,
                    batch_size=self.batch_size,
                    device=self.device,
                )
        finally:
            if had_organism_col and original_organism is not None:
                adata.obs["organism"] = original_organism

        return np.asarray(embeddings, dtype=np.float32)

    def generate_cells(  # noqa: D102
        self,
        base_adata: Any,
        test_adata: Any,
        *,
        split_column: str | None = None,
        split_values: Sequence[str] | None = None,
        gene_name_col: str | None = None,
        prompt_ratio: float = 0.25,
        context_ratio: float = 0.4,
        context_ratio_min: float = 0.2,
        mask_rate: float = 1.0,
        num_steps: int | None = 5,
        mode: str = "mdm",
        num_workers: int = 4,
        random_seed: int | None = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Run Stack's in-context generation for ``test_adata``.

        This is the *decoder-analog* for Stack: rather than mapping a
        latent code back to expression, Stack conditions on a donor
        "base context" (e.g. cells from a specific donor in
        ``base_adata.obs[split_column]``) and synthesises gene-expression
        for each cell in ``test_adata`` using the same gene list as at
        training time.

        Parameters
        ----------
        base_adata
            Donor-specific context AnnData. Must contain ``split_column``
            in ``obs`` unless only one donor is present.
        test_adata
            Cells to synthesise predictions for. Its ``var`` should
            overlap the training gene list.
        split_column
            Column in ``base_adata.obs`` that identifies donors /
            contexts. Required if ``base_adata`` contains multiple
            donors.
        split_values
            Optional subset of donor identifiers to run generation for.
        gene_name_col, prompt_ratio, context_ratio, context_ratio_min,
        mask_rate, num_steps, mode, num_workers, random_seed,
        show_progress
            Forwarded to ``stack.cli.generation.generate``.

        Returns
        -------
        np.ndarray of shape ``(n_test_cells, n_genes)``
            Predictions concatenated across donor splits, in the order
            ``generate`` returned them.
        """
        del kwargs  # unused
        if self._checkpoint is None or self._genelist is None:
            raise ValueError(
                "StackWrapper.generate_cells requires both 'checkpoint' "
                "and 'genelist' to be set at construction time."
            )
        try:
            from stack.cli.generation import generate  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "The 'arc-stack' package is required for Stack in-context "
                "generation. Install with: pip install arc-stack"
            ) from exc

        import tempfile
        import os

        # generate() wants file paths, so persist the in-memory AnnDatas
        # to a temp dir and hand it the paths. This also matches the
        # default backed="r" read path that ``generate`` uses to keep
        # memory bounded on large donor contexts.
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "base.h5ad")
            test_path = os.path.join(tmpdir, "test.h5ad")
            base_adata.write_h5ad(base_path)
            test_adata.write_h5ad(test_path)

            if split_column is None:
                # Single-donor shortcut: inject a placeholder split.
                import anndata as ad
                import pandas as pd
                placeholder = ad.read_h5ad(base_path)
                placeholder.obs["__embpy_split__"] = "ALL"
                placeholder.write_h5ad(base_path)
                split_column = "__embpy_split__"
                split_values = ["ALL"]

            generations = generate(
                checkpoint_path=self._checkpoint,
                base_adata_path=base_path,
                test_adata_path=test_path,
                genelist_path=self._genelist,
                split_column=split_column,
                split_values=list(split_values) if split_values else None,
                gene_name_col=gene_name_col or self._gene_name_col,
                prompt_ratio=prompt_ratio,
                context_ratio=context_ratio,
                context_ratio_min=context_ratio_min,
                mask_rate=mask_rate,
                num_steps=num_steps,
                mode=mode,
                batch_size=self.batch_size,
                num_workers=num_workers,
                random_seed=random_seed,
                device=self.device,
                show_progress=show_progress,
            )

        if not generations:
            raise RuntimeError(
                "stack.cli.generation.generate() returned no splits; "
                "check that `split_column` and `split_values` match "
                "values in base_adata.obs."
            )

        pieces = []
        for split_val, pred_adata in generations.items():
            logger.info("Stack generation: split=%s -> %s", split_val, pred_adata.shape)
            pieces.append(np.asarray(pred_adata.X, dtype=np.float32))
        return np.concatenate(pieces, axis=0)


# ---------------------------------------------------------------------------
# Classical / statistical wrappers
# ---------------------------------------------------------------------------


class PCAEmbedding(SingleCellWrapper):
    """PCA on the expression matrix as a classical encoder-decoder baseline.

    Runs sklearn PCA (CPU) or rapids_singlecell/cuml PCA (GPU) on
    log-normalized counts (or raw counts if no processed layer is
    available).  Optionally restricts to highly variable genes.

    After :meth:`embed_cells`, the fitted PCA object, optional
    ``StandardScaler`` and HVG column mask are cached on the wrapper so
    that :meth:`decode_cells` can invert the transform and produce
    (n_cells, n_genes) reconstructions. The decoder writes zeros into
    non-HVG columns when HVG restriction was used at fit time.

    Parameters
    ----------
    n_components : int
        Number of principal components.
    use_hvg : bool
        If ``True`` and ``adata.var["highly_variable"]`` exists, restrict
        to those genes before PCA.
    layer : str or None
        AnnData layer to use as input.  ``None`` uses ``.X``.
        ``"log_normalized"`` uses the standard-pipeline output.
    scale : bool
        Whether to zero-center and unit-scale before PCA.
    backend : {"cpu", "gpu"}
        ``"cpu"`` uses sklearn PCA (default).
        ``"gpu"`` uses ``rapids_singlecell.pp.pca`` or ``cuml`` PCA.

    Example::

        wrapper = PCAEmbedding(n_components=50, backend="cpu")
        wrapper.load()
        embs = wrapper.embed_cells(adata)        # (n_cells, 50)
        recon = wrapper.decode_cells(embs)       # (n_cells, n_genes)
    """

    supports_decode: bool = True

    def __init__(
        self,
        n_components: int = 50,
        use_hvg: bool = True,
        layer: str | None = "log_normalized",
        scale: bool = True,
        backend: str = "cpu",
        model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name or "pca", **kwargs)
        self.n_components = n_components
        self.use_hvg = use_hvg
        self.layer = layer
        self.scale = scale
        self.backend = backend
        # Cached fit state for decode_cells.
        self._pca: Any = None
        self._scaler: Any = None
        self._hvg_mask: np.ndarray | None = None
        self._n_genes_full: int | None = None
        self._var_names: Sequence[str] | None = None

    def load(self, device: str = "cpu") -> None:  # noqa: D102
        self.device = device

    def embed_cells(self, adata: Any) -> np.ndarray:  # noqa: D102
        import scipy.sparse as sp

        if self.layer and self.layer in adata.layers:
            X = adata.layers[self.layer]
        else:
            X = adata.X

        if sp.issparse(X):
            X = np.asarray(X.toarray(), dtype=np.float64)
        else:
            X = np.asarray(X, dtype=np.float64)

        self._n_genes_full = X.shape[1]
        self._var_names = list(adata.var_names)

        hvg_mask: np.ndarray | None = None
        if self.use_hvg and "highly_variable" in adata.var.columns:
            hvg_mask = adata.var["highly_variable"].values.astype(bool)
            X = X[:, hvg_mask]
        self._hvg_mask = hvg_mask

        n_comp = min(self.n_components, X.shape[0], X.shape[1])

        if self.backend == "gpu":
            try:
                from cuml.decomposition import PCA as cuPCA  # type: ignore[import-untyped]
                from cuml.preprocessing import StandardScaler as cuScaler  # type: ignore[import-untyped]
                scaler = cuScaler() if self.scale else None
                if scaler is not None:
                    X = scaler.fit_transform(X)
                pca = cuPCA(n_components=n_comp, random_state=0)
                result = pca.fit_transform(X)
                result = np.asarray(result, dtype=np.float32)
                var_explained = float(pca.explained_variance_ratio_.sum()) * 100
                self._pca = pca
                self._scaler = scaler
            except ImportError:
                import rapids_singlecell as rsc  # type: ignore[import-untyped]
                import anndata as ad
                adata_tmp = ad.AnnData(X=X.astype(np.float32))
                if self.scale:
                    rsc.pp.scale(adata_tmp)
                rsc.pp.pca(adata_tmp, n_comps=n_comp)
                result = np.asarray(adata_tmp.obsm["X_pca"], dtype=np.float32)
                var_explained = float(
                    adata_tmp.uns["pca"]["variance_ratio"].sum()
                ) * 100
                # rapids_singlecell path does not expose a reusable fitted
                # PCA object, so decode_cells is not available here.
                self._pca = None
                self._scaler = None
        else:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler() if self.scale else None
            if scaler is not None:
                X = scaler.fit_transform(X)
            pca = PCA(n_components=n_comp, random_state=0)
            result = pca.fit_transform(X).astype(np.float32)
            var_explained = pca.explained_variance_ratio_.sum() * 100
            self._pca = pca
            self._scaler = scaler

        logger.info(
            "PCA (backend=%s): %d -> %d components (%.1f%% variance explained)",
            self.backend, X.shape[1] if hasattr(X, "shape") else 0,
            n_comp, var_explained,
        )
        return result

    def decode_cells(  # noqa: D102
        self,
        latent: np.ndarray,
        *,
        gene_names: Sequence[str] | None = None,
        adata: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        del gene_names, adata, kwargs  # unused
        if self._pca is None:
            raise RuntimeError(
                "PCAEmbedding.decode_cells requires a previous call to "
                "embed_cells() with backend='cpu' or backend='gpu' "
                "(cuml path). The rapids_singlecell fallback does not "
                "expose a reusable fitted PCA object."
            )

        latent = np.asarray(latent, dtype=np.float64)
        X_hvg = self._pca.inverse_transform(latent)
        if self._scaler is not None:
            X_hvg = self._scaler.inverse_transform(X_hvg)
        X_hvg = np.asarray(X_hvg, dtype=np.float32)

        if self._hvg_mask is not None and self._n_genes_full is not None:
            full = np.zeros(
                (latent.shape[0], self._n_genes_full), dtype=np.float32
            )
            full[:, self._hvg_mask] = X_hvg
            return full
        return X_hvg

    @property
    def embedding_dim(self) -> int:  # noqa: D102
        return self.n_components


class ScVIToolsWrapper(SingleCellWrapper):
    """Flexible wrapper around scvi-tools models (encoder + decoder).

    Supports scVI, scANVI, totalVI, and any future scvi-tools model
    that follows the ``setup_anndata`` / ``train`` /
    ``get_latent_representation`` pattern.

    After :meth:`embed_cells`, the trained model and the processed
    AnnData are cached so that :meth:`decode_cells` can route arbitrary
    latent points ``z`` through the generative module, i.e.
    ``module.generative(z, library, batch_index)``, and return the
    (n_cells, n_genes) expected-expression matrix (NB mean / ``px_rate``).
    This is exactly the decoder hook that flow-matching / cellflow
    setups use in the perturbation literature.

    Parameters
    ----------
    model_class : str
        scvi-tools model class name: ``"SCVI"``, ``"SCANVI"``,
        ``"TOTALVI"``.
    n_latent : int
        Dimensionality of the latent space.
    n_layers : int
        Number of hidden layers in encoder/decoder.
    n_hidden : int
        Number of nodes per hidden layer.
    max_epochs : int
        Maximum training epochs.
    early_stopping : bool
        Whether to use early stopping during training.
    batch_key : str or None
        Column in ``adata.obs`` for batch correction.
    labels_key : str or None
        Column in ``adata.obs`` for cell-type labels (scANVI).
    protein_expression_obsm_key : str or None
        Key in ``adata.obsm`` with protein counts (totalVI).
    layer : str or None
        AnnData layer containing raw counts for model input.
        ``None`` uses ``.X``; ``"counts"`` uses the counts layer.

    Example::

        wrapper = ScVIToolsWrapper(model_class="SCVI", n_latent=30)
        wrapper.load("cuda")
        z = wrapper.embed_cells(adata)            # (n_cells, 30)
        expr = wrapper.decode_cells(z)            # (n_cells, n_genes)
    """

    supports_decode: bool = True

    def __init__(
        self,
        model_class: str = "SCVI",
        n_latent: int = 30,
        n_layers: int = 2,
        n_hidden: int = 128,
        max_epochs: int = 200,
        early_stopping: bool = True,
        batch_key: str | None = None,
        labels_key: str | None = None,
        protein_expression_obsm_key: str | None = None,
        layer: str | None = "counts",
        model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name or model_class.lower(), **kwargs)
        self.model_class_name = model_class.upper()
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.batch_key = batch_key
        self.labels_key = labels_key
        self.protein_expression_obsm_key = protein_expression_obsm_key
        self.layer = layer
        self._scvi_kwargs = kwargs
        # Cached state for decode_cells.
        self._trained_model: Any = None
        self._trained_adata: Any = None

    def load(self, device: str = "cpu") -> None:  # noqa: D102
        self.device = device

    def _get_model_cls(self):  # type: ignore[no-untyped-def]
        """Import and return the scvi-tools model class."""
        try:
            import scvi  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "scvi-tools is required for ScVIToolsWrapper. "
                "Install with: pip install scvi-tools"
            ) from exc

        model_map = {
            "SCVI": scvi.model.SCVI,
            "SCANVI": scvi.model.SCANVI,
            "TOTALVI": scvi.model.TOTALVI,
        }
        if self.model_class_name not in model_map:
            raise ValueError(
                f"Unknown scvi-tools model '{self.model_class_name}'. "
                f"Available: {list(model_map.keys())}"
            )
        return model_map[self.model_class_name]

    def embed_cells(self, adata: Any) -> np.ndarray:  # noqa: D102
        import scvi as scvi_module  # type: ignore[import-untyped]

        model_cls = self._get_model_cls()
        adata_work = adata.copy()

        # Determine which layer has raw counts
        layer = self.layer
        if layer and layer not in adata_work.layers:
            logger.warning(
                "Layer '%s' not found in adata.layers; falling back to .X",
                layer,
            )
            layer = None

        # Setup anndata for scvi-tools
        setup_kwargs: dict[str, Any] = {}
        if layer:
            setup_kwargs["layer"] = layer
        if self.batch_key and self.batch_key in adata_work.obs.columns:
            setup_kwargs["batch_key"] = self.batch_key

        if self.model_class_name == "SCANVI":
            if self.labels_key and self.labels_key in adata_work.obs.columns:
                setup_kwargs["labels_key"] = self.labels_key
            else:
                raise ValueError(
                    "scANVI requires labels_key pointing to a cell-type "
                    "column in adata.obs"
                )

        if self.model_class_name == "TOTALVI":
            if self.protein_expression_obsm_key:
                setup_kwargs["protein_expression_obsm_key"] = (
                    self.protein_expression_obsm_key
                )

        model_cls.setup_anndata(adata_work, **setup_kwargs)

        # Build model
        model_kwargs: dict[str, Any] = {
            "n_latent": self.n_latent,
            "n_layers": self.n_layers,
            "n_hidden": self.n_hidden,
        }

        if self.model_class_name == "SCANVI":
            # scANVI uses a two-step approach: train SCVI first, then SCANVI
            scvi_model = scvi_module.model.SCVI(adata_work, **model_kwargs)
            scvi_model.train(
                max_epochs=max(self.max_epochs // 2, 50),
                early_stopping=self.early_stopping,
            )
            model = scvi_module.model.SCANVI.from_scvi_model(
                scvi_model,
                unlabeled_category="Unknown",
            )
            model.train(
                max_epochs=self.max_epochs // 2,
                early_stopping=self.early_stopping,
            )
        else:
            model = model_cls(adata_work, **model_kwargs)
            model.train(
                max_epochs=self.max_epochs,
                early_stopping=self.early_stopping,
            )

        latent = model.get_latent_representation()
        logger.info(
            "%s: trained %d epochs, latent shape %s",
            self.model_class_name, model.history_["elbo_train"].shape[0],
            latent.shape,
        )
        # Cache so decode_cells can reuse the trained generative module.
        self._trained_model = model
        self._trained_adata = adata_work
        return np.asarray(latent, dtype=np.float32)

    def decode_cells(  # noqa: D102
        self,
        latent: np.ndarray,
        *,
        gene_names: Sequence[str] | None = None,
        adata: Any = None,
        library_size: float | np.ndarray | None = None,
        batch_index: int | np.ndarray | None = None,
        labels: int | np.ndarray | None = None,
        return_raw: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Decode a latent matrix to expected gene-expression.

        Routes ``z`` through the trained generative module. The returned
        matrix is the NB/ZINB mean ``px_rate`` (i.e.
        ``library_size * px_scale``), the same object that
        ``model.get_normalized_expression`` scales to a target library
        size. Use ``return_raw=True`` to get the raw per-cell NB mean
        without enforcing a library size.

        Parameters
        ----------
        latent
            Latent codes, shape ``(n_cells, n_latent)``.
        gene_names
            Ignored (scvi-tools always decodes to the full training gene set).
        adata
            Ignored here; kept for API compatibility.
        library_size
            Scalar or per-cell log-library-size used by the decoder. If
            ``None``, defaults to ``log(1e4)`` (as in
            ``get_normalized_expression``).
        batch_index
            Scalar or per-cell batch index fed to the decoder (defaults
            to 0, matching the behaviour of ``transform_batch=None``).
        labels
            Only used by scANVI (per-cell class index). Defaults to 0.
        return_raw
            If ``True``, return ``px_scale`` instead of ``px_rate``
            (i.e. the decoder's multinomial logits before library
            scaling).
        """
        del gene_names, adata, kwargs  # unused
        if self._trained_model is None:
            raise RuntimeError(
                "ScVIToolsWrapper.decode_cells requires a prior call to "
                "embed_cells() so the generative module is trained and "
                "cached."
            )

        import torch

        model = self._trained_model
        module = model.module
        device = next(module.parameters()).device
        module.eval()

        z = torch.as_tensor(np.asarray(latent), device=device, dtype=torch.float32)
        n = z.shape[0]

        if library_size is None:
            lib = torch.full(
                (n, 1), float(np.log(1e4)), device=device, dtype=torch.float32,
            )
        elif np.ndim(library_size) == 0:
            lib = torch.full(
                (n, 1), float(library_size), device=device, dtype=torch.float32,
            )
        else:
            lib = torch.as_tensor(
                np.asarray(library_size).reshape(n, 1),
                device=device, dtype=torch.float32,
            )

        if batch_index is None:
            bidx = torch.zeros((n, 1), device=device, dtype=torch.long)
        elif np.ndim(batch_index) == 0:
            bidx = torch.full(
                (n, 1), int(batch_index), device=device, dtype=torch.long,
            )
        else:
            bidx = torch.as_tensor(
                np.asarray(batch_index).reshape(n, 1),
                device=device, dtype=torch.long,
            )

        gen_kwargs: dict[str, Any] = {
            "z": z,
            "library": lib,
            "batch_index": bidx,
        }

        # scANVI also requires a label index y.
        if self.model_class_name == "SCANVI":
            if labels is None:
                yidx = torch.zeros((n, 1), device=device, dtype=torch.long)
            elif np.ndim(labels) == 0:
                yidx = torch.full(
                    (n, 1), int(labels), device=device, dtype=torch.long,
                )
            else:
                yidx = torch.as_tensor(
                    np.asarray(labels).reshape(n, 1),
                    device=device, dtype=torch.long,
                )
            gen_kwargs["y"] = yidx

        with torch.no_grad():
            outputs = module.generative(**gen_kwargs)

        # scvi-tools returns either a dict with ``px`` (torch distribution)
        # or a ``px_rate`` / ``px_scale`` tensor depending on the version.
        # NOTE: torch.Tensor also has a ``.mean`` method, so we must not use
        # ``hasattr(val, "mean")`` as the branch predicate. Plain tensors
        # are forwarded as-is; only objects that are NOT a tensor (i.e.
        # torch.distributions.Distribution instances) have ``.mean`` read
        # as the distribution mean tensor.
        if return_raw:
            key_candidates = ("px_scale", "px")
        else:
            key_candidates = ("px_rate", "px")
        tensor: torch.Tensor | None = None
        for key in key_candidates:
            if key in outputs:
                val = outputs[key]
                if isinstance(val, torch.Tensor):
                    tensor = val
                elif hasattr(val, "mean"):  # torch Distribution or similar
                    mean_attr = val.mean
                    tensor = mean_attr() if callable(mean_attr) else mean_attr
                else:
                    tensor = val
                break
        if tensor is None:
            raise RuntimeError(
                "scvi-tools generative output did not expose 'px_rate' / "
                f"'px_scale' / 'px'; got keys: {list(outputs.keys())}"
            )
        return tensor.detach().cpu().float().numpy()

    @property
    def embedding_dim(self) -> int:  # noqa: D102
        return self.n_latent


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
    "StateEmbeddingWrapper": StateEmbeddingWrapper,
    "StackWrapper": StackWrapper,
    "PCAEmbedding": PCAEmbedding,
    "ScVIToolsWrapper": ScVIToolsWrapper,
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
