"""Zhipu (智谱 GLM) LLM implementation (OpenAI-compatible).

Zhipu AI exposes an OpenAI-compatible Chat Completions endpoint (BigModel v4).
This module subclasses :class:`OpenAILLM` and only overrides the default base
URL and API-key resolution, so all chat/HTTP logic is inherited unchanged.

Configuration (settings.yaml)::

    llm:
      provider: "zhipu"
      model: "glm-4.7"
      api_key: ""                              # read from ZHIPUAI_API_KEY env var
      base_url: "https://open.bigmodel.cn/api/paas/v4"
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.openai_llm import OpenAILLM, OpenAILLMError


class ZhipuLLMError(RuntimeError):
    """Raised when the Zhipu (GLM) API call fails."""


class ZhipuLLM(OpenAILLM):
    """Zhipu LLM provider — OpenAI-compatible BigModel v4 endpoint.

    Inherits all chat logic from :class:`OpenAILLM`; overrides the base URL and
    resolves the API key from ``settings.llm.api_key`` or the
    ``ZHIPUAI_API_KEY`` environment variable. API errors raised by the inherited
    chat flow are translated to :class:`ZhipuLLMError`.
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
            or getattr(settings.llm, "base_url", None)
            or self.DEFAULT_BASE_URL
        )
        resolved_key = (
            api_key
            or getattr(settings.llm, "api_key", None)
            or os.environ.get("ZHIPUAI_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "Zhipu API key not provided. Set in settings.yaml (llm.api_key) "
                "or the ZHIPUAI_API_KEY environment variable."
            )
        super().__init__(
            settings, api_key=resolved_key, base_url=resolved_base, **kwargs
        )

    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a chat completion via the inherited OpenAI-compatible flow.

        Args:
            messages: List of conversation messages.
            trace: Optional TraceContext for observability (reserved for Stage F).
            **kwargs: Override parameters (temperature, max_tokens, etc.).

        Returns:
            ChatResponse with generated content and metadata.

        Raises:
            ValueError: If messages are invalid.
            ZhipuLLMError: If the API call fails.
        """
        try:
            return super().chat(messages, trace=trace, **kwargs)
        except OpenAILLMError as e:
            raise ZhipuLLMError(
                str(e).replace("[OpenAI]", "[Zhipu]")
            ) from e
