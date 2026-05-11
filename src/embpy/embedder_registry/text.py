"""Text registry entries (sentence-transformers + BERT + LLaMA)."""

from __future__ import annotations

from ..models.base import BaseModelWrapper
from ..models.text_models import LlamaEmbeddingWrapper, TextLLMWrapper


TEXT_MODELS: dict[str, tuple[type[BaseModelWrapper] | None, str | None]] = {
    "minilm_l6_v2": (TextLLMWrapper, "sentence-transformers/all-MiniLM-L6-v2"),
    "bert_base_uncased": (TextLLMWrapper, "bert-base-uncased"),
    # LLaMA decoder-only models (requires HF_TOKEN for gated access)
    "llama3.1_8b": (LlamaEmbeddingWrapper, "meta-llama/Llama-3.1-8B"),
    "llama3.2_3b": (LlamaEmbeddingWrapper, "meta-llama/Llama-3.2-3B"),
    "llama3.2_1b": (LlamaEmbeddingWrapper, "meta-llama/Llama-3.2-1B"),
}
