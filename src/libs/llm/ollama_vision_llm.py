"""Ollama Vision LLM implementation for local multimodal inference.

This module provides an Ollama Vision LLM that works with locally running
Ollama instances exposing an OpenAI-compatible API (/v1/chat/completions).
Supports vision models like llava-phi3, bakllava, llava, etc.

Ollama's /v1 endpoint is fully OpenAI-compatible, so this class inherits from
OpenAIVisionLLM and overrides only:
- Initialization: API key is not required, base URL defaults to localhost
- HTTP client: trust_env=False to bypass system proxy (D-014)
- Timeout: 120s for local inference
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx

from src.libs.llm.openai_vision_llm import OpenAIVisionLLM, OpenAIVisionLLMError


class OllamaVisionLLMError(OpenAIVisionLLMError):
    """Raised when Ollama Vision API call fails."""


class OllamaVisionLLM(OpenAIVisionLLM):
    """Ollama Vision LLM provider for local multimodal inference.

    Inherits from OpenAIVisionLLM to reuse the OpenAI-compatible message
    format (base64 images in image_url content type), overriding only
    initialization defaults and the HTTP transport layer.

    Attributes:
        base_url: Ollama OpenAI-compatible endpoint (default http://localhost:11434/v1).
        model: Vision model identifier (e.g., 'llava-phi3:3.8b').
        api_key: Placeholder — Ollama does not require authentication.
    """

    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    DEFAULT_TIMEOUT = 120.0

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_image_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama Vision LLM provider.

        Args:
            settings: Application settings containing vision_llm configuration.
            api_key: Ignored (Ollama does not require authentication).
            base_url: Optional base URL override.
            max_image_size: Maximum image dimension in pixels for auto-compression.
            **kwargs: Additional configuration overrides.
        """
        vision_settings = getattr(settings, "vision_llm", None)

        # Temperature / max_tokens from LLM section defaults
        self.default_temperature = getattr(settings.llm, "temperature", 0.0)
        self.default_max_tokens = getattr(settings.llm, "max_tokens", 4096)

        # Model name: vision_llm.model > llm.model fallback
        vision_model = (
            getattr(vision_settings, "model", None) if vision_settings else None
        )
        self.model = vision_model or settings.llm.model

        # Max image size: param > settings > class default
        vision_max_size = (
            getattr(vision_settings, "max_image_size", None)
            if vision_settings
            else None
        )
        self.max_image_size = (
            max_image_size or vision_max_size or self.DEFAULT_MAX_IMAGE_SIZE
        )

        # API key: not required for Ollama, use placeholder
        self.api_key = "ollama"

        # Base URL: explicit > vision_llm.base_url > OLLAMA_BASE_URL env > default
        vision_base_url = (
            getattr(vision_settings, "base_url", None) if vision_settings else None
        )
        _ollama_env = os.environ.get("OLLAMA_BASE_URL", "").rstrip("/")
        if base_url:
            self.base_url = base_url
        elif vision_base_url:
            self.base_url = vision_base_url
        elif _ollama_env:
            # OLLAMA_BASE_URL typically does NOT include /v1 suffix
            self.base_url = (
                _ollama_env
                if _ollama_env.endswith("/v1")
                else f"{_ollama_env}/v1"
            )
        else:
            self.base_url = self.DEFAULT_BASE_URL

        # Disable Azure-specific behavior
        self.api_version = None
        self._use_azure_auth = False

        self._extra_config = kwargs

    def _call_api(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """Make HTTP request to the Ollama OpenAI-compatible endpoint.

        Uses trust_env=False to avoid system proxy interception of localhost
        traffic (see D-014 in DEV_CHANGELOG).

        Args:
            messages: List of API-formatted messages.
            temperature: Generation temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            API response as dictionary.

        Raises:
            OllamaVisionLLMError: If API call fails.
        """
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            with httpx.Client(
                timeout=self.DEFAULT_TIMEOUT, trust_env=False
            ) as client:
                response = client.post(url, json=payload, headers=headers)

                if response.status_code != 200:
                    error_detail = self._parse_error_response(response)
                    raise OllamaVisionLLMError(
                        f"[Ollama Vision] API error (HTTP {response.status_code}): {error_detail}"
                    )

                return response.json()
        except httpx.TimeoutException as e:
            raise OllamaVisionLLMError(
                f"[Ollama Vision] Request timed out after {self.DEFAULT_TIMEOUT} seconds"
            ) from e
        except httpx.RequestError as e:
            raise OllamaVisionLLMError(
                f"[Ollama Vision] Connection failed. Ensure Ollama is running locally: {e}"
            ) from e
