"""Zhipu (智谱 GLM) Embedding implementation (OpenAI-compatible).

Zhipu AI exposes an OpenAI-compatible Embeddings endpoint (BigModel v4). This
module subclasses :class:`OpenAIEmbedding` and only overrides the default base
URL and API-key resolution, so all embedding logic is inherited unchanged.

Configuration (settings.yaml)::

    embedding:
      provider: "zhipu"
      model: "embedding-3"
      dimensions: 2048
      api_key: ""                              # read from ZHIPUAI_API_KEY env var
      base_url: "https://open.bigmodel.cn/api/paas/v4"
"""

from __future__ import annotations

import os
from typing import Any, Optional

from src.libs.embedding.openai_embedding import OpenAIEmbedding


class ZhipuEmbeddingError(RuntimeError):
    """Raised when the Zhipu (GLM) Embeddings API call fails."""


class ZhipuEmbedding(OpenAIEmbedding):
    """Zhipu Embedding provider — OpenAI-compatible BigModel v4 endpoint.

    Inherits all embedding logic from :class:`OpenAIEmbedding`; overrides the
    base URL and resolves the API key from ``settings.embedding.api_key`` or
    the ``ZHIPUAI_API_KEY`` environment variable.

    Note:
        ``embedding-3`` defaults to 2048 dimensions. The inherited
        :meth:`OpenAIEmbedding.embed` only forwards ``dimensions`` for
        ``text-embedding-3-*`` models, so the Zhipu default dimension is used
        by the API; set ``embedding.dimensions`` in config so
        :meth:`get_dimension` reports the correct value to the vector store.
    """

    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

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
            or os.environ.get("ZHIPUAI_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "Zhipu API key not provided. Set in settings.yaml "
                "(embedding.api_key) or the ZHIPUAI_API_KEY environment variable."
            )
        super().__init__(
            settings, api_key=resolved_key, base_url=resolved_base, **kwargs
        )
