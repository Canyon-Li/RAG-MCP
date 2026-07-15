"""DeepSeek LLM implementation (OpenAI-compatible).

DeepSeek exposes an OpenAI-compatible Chat Completions endpoint. This module
subclasses :class:`OpenAILLM` and only overrides the default base URL, API-key
resolution, and error type, so all chat/HTTP logic is inherited unchanged.

Configuration (settings.yaml)::

    llm:
      provider: "deepseek"
      model: "deepseek-chat"
      api_key: "<DEEPSEEK_API_KEY>"
      base_url: "https://api.deepseek.com"   # optional, this is the default
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.openai_llm import OpenAILLM, OpenAILLMError


class DeepSeekLLMError(RuntimeError):
    """Raised when the DeepSeek API call fails."""


class DeepSeekLLM(OpenAILLM):
    """DeepSeek LLM provider — OpenAI-compatible endpoint.

    Inherits all chat logic from :class:`OpenAILLM`; overrides the base URL and
    resolves the API key from ``settings.llm.api_key`` or the
    ``DEEPSEEK_API_KEY`` environment variable. API errors raised by the inherited
    chat flow are translated to :class:`DeepSeekLLMError` to preserve this
    provider's public error contract.
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com"

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
            or os.environ.get("DEEPSEEK_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "DeepSeek API key not provided. Set in settings.yaml (llm.api_key), "
                "the DEEPSEEK_API_KEY environment variable, or pass api_key parameter."
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
            DeepSeekLLMError: If the API call fails.
        """
        try:
            return super().chat(messages, trace=trace, **kwargs)
        except OpenAILLMError as e:
            raise DeepSeekLLMError(
                str(e).replace("[OpenAI]", "[DeepSeek]")
            ) from e
