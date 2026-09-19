"""Ollama Embedding implementation for local embedding models.

This module provides the Ollama Embedding implementation that works with
locally running Ollama instances. Ollama enables running embedding models like
nomic-embed-text, mxbai-embed-large, etc. on local hardware.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.libs.embedding.base_embedding import BaseEmbedding


class OllamaEmbeddingError(RuntimeError):
    """Raised when Ollama Embeddings API call fails.
    
    This exception provides clear error messages without exposing
    sensitive configuration details like internal URLs.
    """


class OllamaEmbedding(BaseEmbedding):
    """Ollama Embedding provider implementation for local embedding.
    
    This class implements the BaseEmbedding interface for Ollama's embeddings API,
    enabling local embedding generation without cloud dependencies.
    
    Attributes:
        base_url: The base URL for the Ollama server (default: http://localhost:11434).
        model: The model identifier to use (e.g., 'nomic-embed-text', 'mxbai-embed-large').
        timeout: Request timeout in seconds.
        dimension: The dimensionality of embeddings produced by this model.
    
    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> embedding = OllamaEmbedding(settings)
        >>> vectors = embedding.embed(["hello world", "test"])
    """
    
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 120.0  # Longer timeout for local inference
    DEFAULT_DIMENSION = 768  # Common dimension for local embedding models
    
    def __init__(
        self,
        settings: Any,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama Embedding provider.
        
        Args:
            settings: Application settings containing Embedding configuration.
            base_url: Optional base URL override (falls back to env var OLLAMA_BASE_URL).
            timeout: Optional timeout override for requests.
            **kwargs: Additional configuration overrides.
        
        Raises:
            ValueError: If required configuration is missing.
        """
        self.model = settings.embedding.model
        
        # Base URL: explicit > env var > default
        self.base_url = (
            base_url 
            or os.environ.get("OLLAMA_BASE_URL") 
            or self.DEFAULT_BASE_URL
        )
        
        # Timeout: explicit > default
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        
        # Dimension: settings > default
        self.dimension = getattr(settings.embedding, 'dimensions', self.DEFAULT_DIMENSION)
        
        # Store any additional kwargs for future use
        self._extra_config = kwargs
    
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts using Ollama API.
        
        Args:
            texts: List of text strings to embed. Must not be empty.
            trace: Optional TraceContext for observability (reserved for Stage F).
            **kwargs: Additional parameters (currently unused, reserved for future).
        
        Returns:
            List of embedding vectors, where each vector is a list of floats.
        
        Raises:
            ValueError: If texts list is empty or contains invalid entries.
            OllamaEmbeddingError: If API call fails.
        
        Example:
            >>> embeddings = embedding.embed(["hello", "world"])
            >>> len(embeddings)  # 2 vectors
            >>> len(embeddings[0])  # dimension (e.g., 768)
        """
        # Validate input
        self.validate_texts(texts)
        
        try:
            import httpx
        except ImportError as e:
            raise OllamaEmbeddingError(
                "httpx library is required for Ollama Embedding. "
                "Install with: pip install httpx"
            ) from e
        
        # Prepare API request — /api/embed, NOT the legacy /api/embeddings.
        # The legacy endpoint rejects prompts longer than the model's
        # context window with a bare 500 (no truncation), which on
        # 2026-09-19 silently dropped a whole encode batch: a HybridChunker
        # table chunk stitched with two VLM captions hit 8.7k chars ≈ 2.2k
        # tokens against nomic-embed-text's 2048-token ceiling. /api/embed
        # truncates to num_ctx server-side instead. Vector direction is
        # identical between the endpoints (scale differs; the vector store
        # uses cosine space, which is scale-invariant).
        url = f"{self.base_url}/api/embed"

        embeddings: List[List[float]] = []

        # Process each text individually (one input per request keeps the
        # response-order contract trivially exact)
        for text in texts:
            payload = {
                "model": self.model,
                "input": text,
            }
            
            try:
                # trust_env=False: Ollama runs locally — never route requests
                # through an HTTP proxy. When a system proxy (clash/v2ray/...) is
                # active and localhost isn't in its no_proxy list, the default
                # trust_env=True yields a 502 from the proxy instead of hitting
                # the Ollama server.
                with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                    response = client.post(url, json=payload)
                    response.raise_for_status()
                    
                    result = response.json()

                    # Extract embedding from response
                    if "embeddings" not in result or not result["embeddings"]:
                        raise OllamaEmbeddingError(
                            f"Unexpected response format from Ollama API. "
                            f"Expected 'embeddings' field but got: {list(result.keys())}"
                        )

                    embeddings.append(result["embeddings"][0])
                    
            except httpx.HTTPStatusError as e:
                # HTTP error (4xx, 5xx)
                raise OllamaEmbeddingError(
                    f"Ollama API request failed with status {e.response.status_code}. "
                    f"Ensure Ollama is running and model '{self.model}' is available."
                ) from e
            except httpx.ConnectError as e:
                # Connection error (server not reachable)
                raise OllamaEmbeddingError(
                    f"Failed to connect to Ollama server at {self.base_url}. "
                    f"Ensure Ollama is running (try: ollama serve)"
                ) from e
            except httpx.TimeoutException as e:
                # Request timeout
                raise OllamaEmbeddingError(
                    f"Ollama API request timed out after {self.timeout}s. "
                    f"The model may be loading or the request is too large."
                ) from e
            except httpx.RequestError as e:
                # Other request errors
                raise OllamaEmbeddingError(
                    f"Ollama API request failed: {str(e)}"
                ) from e
            except (KeyError, ValueError, TypeError) as e:
                # JSON parsing or data extraction error
                raise OllamaEmbeddingError(
                    f"Failed to parse Ollama API response: {str(e)}"
                ) from e
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get the dimensionality of embeddings produced by this provider.
        
        Returns:
            The vector dimension configured for this instance.
        
        Note:
            The actual dimension may vary by model. Common dimensions:
            - nomic-embed-text: 768
            - mxbai-embed-large: 1024
            Configure via settings.embedding.dimensions or accepts default 768.
        """
        return self.dimension
