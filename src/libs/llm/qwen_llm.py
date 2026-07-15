"""Qwen (Alibaba Cloud DashScope) LLM implementation (OpenAI-compatible).

Qwen exposes an OpenAI-compatible Chat Completions endpoint through DashScope.
This module subclasses :class:`OpenAILLM` and only overrides the default base
URL and API-key resolution, so all chat logic is inherited unchanged.

Configuration (settings.yaml)::

    llm:
      provider: "qwen"
      model: "qwen-turbo"            # qwen-turbo | qwen-plus | qwen-max
      api_key: "<DASHSCOPE_API_KEY>"
      base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
"""

from __future__ import annotations

import os
from typing import Any, Optional

from src.libs.llm.openai_llm import OpenAILLM


class QwenLLMError(RuntimeError):
    """Raised when the Qwen (DashScope) API call fails."""


class QwenLLM(OpenAILLM):
    """Qwen LLM provider — OpenAI-compatible DashScope endpoint.

    Inherits all chat logic from :class:`OpenAILLM`; overrides the base URL and
    resolves the API key from ``settings.llm.api_key`` or the
    ``DASHSCOPE_API_KEY`` environment variable.
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
            or getattr(settings.llm, "base_url", None)
            or self.DEFAULT_BASE_URL
        )
        resolved_key = (
            api_key
            or getattr(settings.llm, "api_key", None)
            or os.environ.get("DASHSCOPE_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "Qwen API key not provided. Set in settings.yaml (llm.api_key) "
                "or the DASHSCOPE_API_KEY environment variable."
            )
        super().__init__(
            settings, api_key=resolved_key, base_url=resolved_base, **kwargs
        )
