"""API-based embedding models (OpenAI, Cohere, Voyage AI, Google, generic).

Supports any provider that exposes an embedding endpoint. For
OpenAI-compatible APIs (vLLM, Ollama, Together, Anyscale), set
``provider="openai"`` and ``base_url`` to the custom endpoint.

API keys are resolved in this order:
1. Explicit ``api_key`` argument
2. Provider-specific environment variable (see table below)
3. ``LLM_API_KEY`` fallback

| Provider | Env var | Default base URL |
|----------|---------|------------------|
| openai   | ``OPENAI_API_KEY`` | ``https://api.openai.com/v1`` |
| cohere   | ``COHERE_API_KEY`` | ``https://api.cohere.com/v2`` |
| voyage   | ``VOYAGE_API_KEY`` | ``https://api.voyageai.com/v1`` |
| google   | ``GOOGLE_API_KEY`` | Vertex AI endpoint |
| generic  | ``LLM_API_KEY``    | user must supply ``base_url`` |
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import requests
import torch

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "cohere": "COHERE_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "google": "GOOGLE_API_KEY",
    "generic": "LLM_API_KEY",
}

PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "cohere": "https://api.cohere.com/v2",
    "voyage": "https://api.voyageai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",
}

PROVIDER_BATCH_LIMITS = {
    "openai": 2048,
    "cohere": 96,
    "voyage": 128,
    "google": 100,
    "generic": 128,
}


class APIEmbeddingWrapper(BaseModelWrapper):
    """Wrapper for API-based embedding models.

    Calls external embedding APIs (OpenAI, Cohere, Voyage AI, Google,
    or any OpenAI-compatible endpoint) and returns the embedding vectors.

    Parameters
    ----------
    model_path_or_name
        Model identifier as expected by the API (e.g.
        ``"text-embedding-3-small"`` for OpenAI).
    provider
        API provider: ``"openai"``, ``"cohere"``, ``"voyage"``,
        ``"google"``, or ``"generic"`` (OpenAI-compatible).
    api_key
        API key. If ``None``, reads from the provider's environment
        variable or ``LLM_API_KEY``.
    base_url
        Override the default API base URL. Required for ``"generic"``
        provider; useful for local endpoints (Ollama, vLLM).
    rate_limit_delay
        Seconds to wait between API calls.
    """

    model_type = "text"
    available_pooling_strategies = ["api"]

    def __init__(
        self,
        model_path_or_name: str = "text-embedding-3-small",
        provider: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        rate_limit_delay: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(model_path_or_name, **kwargs)
        self.provider = provider.lower()
        self._api_key = api_key
        self._base_url = base_url
        self.delay = rate_limit_delay
        self._resolved_key: str | None = None
        self._resolved_url: str | None = None

    def _resolve_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        env_key = PROVIDER_ENV_KEYS.get(self.provider, "LLM_API_KEY")
        key = os.environ.get(env_key) or os.environ.get("LLM_API_KEY")
        if not key:
            raise ValueError(
                f"No API key found for provider '{self.provider}'. "
                f"Set {env_key} or LLM_API_KEY environment variable, "
                f"or pass api_key explicitly."
            )
        return key

    def _resolve_base_url(self) -> str:
        if self._base_url:
            return self._base_url.rstrip("/")
        url = PROVIDER_BASE_URLS.get(self.provider)
        if not url:
            raise ValueError(
                f"No default base URL for provider '{self.provider}'. "
                f"Pass base_url explicitly."
            )
        return url

    def load(self, device: torch.device) -> None:
        """Validate API key and URL. No model download needed."""
        self._resolved_key = self._resolve_api_key()
        self._resolved_url = self._resolve_base_url()
        self.device = device
        logger.info(
            "API embedding ready: provider=%s, model=%s, url=%s",
            self.provider, self.model_name, self._resolved_url,
        )

    def _call_openai_compatible(
        self, texts: list[str],
    ) -> list[np.ndarray]:
        """Call an OpenAI-compatible /embeddings endpoint."""
        resp = requests.post(
            f"{self._resolved_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self._resolved_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "input": texts,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = sorted(data["data"], key=lambda x: x["index"])
        return [np.array(e["embedding"], dtype=np.float32) for e in embeddings]

    def _call_cohere(self, texts: list[str]) -> list[np.ndarray]:
        """Call Cohere embed API."""
        resp = requests.post(
            f"{self._resolved_url}/embed",
            headers={
                "Authorization": f"Bearer {self._resolved_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "texts": texts,
                "input_type": "search_document",
                "embedding_types": ["float"],
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        vecs = data.get("embeddings", {}).get("float", [])
        return [np.array(v, dtype=np.float32) for v in vecs]

    def _call_google(self, texts: list[str]) -> list[np.ndarray]:
        """Call Google Generative AI embedding API."""
        results = []
        for text in texts:
            resp = requests.post(
                f"{self._resolved_url}/models/{self.model_name}:embedContent",
                params={"key": self._resolved_key},
                json={
                    "model": f"models/{self.model_name}",
                    "content": {"parts": [{"text": text}]},
                },
                timeout=60,
            )
            resp.raise_for_status()
            values = resp.json()["embedding"]["values"]
            results.append(np.array(values, dtype=np.float32))
            if self.delay > 0:
                time.sleep(self.delay)
        return results

    def _call_api(self, texts: list[str]) -> list[np.ndarray]:
        """Dispatch to the correct provider API."""
        if self.provider in ("openai", "voyage", "generic"):
            return self._call_openai_compatible(texts)
        elif self.provider == "cohere":
            return self._call_cohere(texts)
        elif self.provider == "google":
            return self._call_google(texts)
        else:
            raise ValueError(f"Unknown provider '{self.provider}'")

    def embed(
        self,
        input: str,
        pooling_strategy: str = "api",
        **kwargs: Any,
    ) -> np.ndarray:
        """Get embedding for a single text via API.

        Parameters
        ----------
        input
            Text string to embed.
        pooling_strategy
            Ignored -- the API handles pooling internally.
        **kwargs
            Ignored.

        Returns
        -------
        np.ndarray
            Embedding vector.
        """
        if self._resolved_key is None:
            raise RuntimeError("API not initialized. Call load() first.")

        results = self._call_api([input])
        return results[0]

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "api",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Get embeddings for a batch of texts via API.

        Automatically chunks into provider-specific batch size limits.

        Parameters
        ----------
        inputs
            List of text strings.
        pooling_strategy
            Ignored.
        **kwargs
            Ignored.

        Returns
        -------
        list of np.ndarray
            One embedding per input.
        """
        if self._resolved_key is None:
            raise RuntimeError("API not initialized. Call load() first.")

        max_batch = PROVIDER_BATCH_LIMITS.get(self.provider, 128)
        all_results: list[np.ndarray] = []
        texts = list(inputs)

        for i in range(0, len(texts), max_batch):
            chunk = texts[i : i + max_batch]
            try:
                chunk_results = self._call_api(chunk)
                all_results.extend(chunk_results)
            except Exception as e:  # noqa: BLE001
                logger.error("API batch call failed for chunk %d: %s", i, e)
                all_results.extend([np.zeros(1, dtype=np.float32)] * len(chunk))

            if self.delay > 0 and i + max_batch < len(texts):
                time.sleep(self.delay)

        return all_results
