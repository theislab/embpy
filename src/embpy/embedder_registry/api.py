"""API-based embedding registry entries (OpenAI / Cohere / Voyage / Google).

These models do not download weights; they call third-party HTTP APIs
and require credentials via environment variables (see
``embpy.models.api_models.APIEmbeddingWrapper``).
"""

from __future__ import annotations

from ..models.api_models import APIEmbeddingWrapper
from ..models.base import BaseModelWrapper


API_MODELS: dict[str, tuple[type[BaseModelWrapper] | None, str | None]] = {
    "openai_small": (APIEmbeddingWrapper, "text-embedding-3-small"),
    "openai_large": (APIEmbeddingWrapper, "text-embedding-3-large"),
    "cohere_v3": (APIEmbeddingWrapper, "embed-english-v3.0"),
    "cohere_multilingual": (APIEmbeddingWrapper, "embed-multilingual-v3.0"),
    "voyage_3": (APIEmbeddingWrapper, "voyage-3"),
    "voyage_3_lite": (APIEmbeddingWrapper, "voyage-3-lite"),
    "google_embed": (APIEmbeddingWrapper, "text-embedding-005"),
}
