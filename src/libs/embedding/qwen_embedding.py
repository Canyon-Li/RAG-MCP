"""Qwen (Alibaba Cloud DashScope) Embedding implementation (OpenAI-compatible).

Qwen exposes an OpenAI-compatible Embeddings endpoint through DashScope. This
module subclasses :class:`OpenAIEmbedding` and only overrides the default base
URL and API-key resolution, so all embedding logic is inherited unchanged.

Configuration (settings.yaml)::

    embedding:
      provider: "qwen"
      model: "text-embedding-v3"
      dimensions: 1024
      api_key: "<DASHSCOPE_API_KEY>"
      base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
"""

from __future__ import annotations

import os
from typing import Any, Optional

from src.libs.embedding.openai_embedding import OpenAIEmbedding


class QwenEmbeddingError(RuntimeError):
    """Raised when the Qwen (DashScope) Embeddings API call fails."""


class QwenEmbedding(OpenAIEmbedding):
    """Qwen Embedding provider — OpenAI-compatible DashScope endpoint.

    Inherits all embedding logic from :class:`OpenAIEmbedding`; overrides the
    base URL and resolves the API key from ``settings.embedding.api_key`` or
    the ``DASHSCOPE_API_KEY`` environment variable.
    """

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        resolved_base = (
            base_url
            or getattr(settings.embedding, "base_url", None)
            or self.DEFAULT_BASE_URL
        )
        resolved_key = (
            api_key
            or getattr(settings.embedding, "api_key", None)
            or os.environ.get("DASHSCOPE_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "Qwen API key not provided. Set in settings.yaml "
                "(embedding.api_key) or the DASHSCOPE_API_KEY environment variable."
            )
        super().__init__(
            settings, api_key=resolved_key, base_url=resolved_base, **kwargs
        )
