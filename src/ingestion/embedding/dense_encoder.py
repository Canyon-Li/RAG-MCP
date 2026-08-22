"""Dense Encoder for generating embeddings from text chunks.

This module implements the Dense Encoder component of the Ingestion Pipeline,
responsible for converting text chunks into dense vector representations using
configurable embedding providers.

Design Principles:
- Config-Driven: Uses factory pattern to obtain embedding provider from settings
- Batch Processing: Optimizes API calls through batching
- Observable: Accepts TraceContext for future observability integration
- Error Handling: Individual failures shouldn't crash entire batch
- Deterministic: Same inputs produce same outputs
- Cache-Friendly (D-031): optional persistent (model, content-hash) cache —
  unchanged chunk texts skip the embedding call on re-ingest
"""

import hashlib
import logging
from typing import List, Optional, Any

from src.core.types import Chunk
from src.libs.embedding.base_embedding import BaseEmbedding

logger = logging.getLogger(__name__)


class DenseEncoder:
    """Encodes text chunks into dense vectors using BaseEmbedding provider.

    This encoder acts as a bridge between the ingestion pipeline and the
    pluggable embedding layer. It handles batching, error recovery, and
    maintains alignment between input chunks and output vectors.

    Design:
    - Dependency Injection: Receives BaseEmbedding instance (no direct factory call)
    - Batch-First: Processes only cache-MISSED chunks in configurable batch sizes
    - Cache layer (D-031): before calling the provider, every chunk text is
      hashed and looked up in the optional persistent cache; hits are served
      directly. Cache failures never break ingestion (warn + bypass).
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        batch_size: int = 100,
        cache: Optional[Any] = None,
    ):
        """Initialize DenseEncoder.

        Args:
            embedding: Embedding provider instance (from EmbeddingFactory)
            batch_size: Number of chunks to process per API call (default: 100)
            cache: Optional persistent embedding cache (D-031) exposing
                get_many(model, hashes) / put_many(model, items)

        Raises:
            ValueError: If batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.embedding = embedding
        self.batch_size = batch_size
        self.cache = cache
        # Refreshed after every encode() call; the pipeline surfaces it in
        # the embed trace stage (observability for incremental ingestion).
        self.last_cache_stats: Optional[dict] = None

    def _cache_model(self) -> str:
        return getattr(self.embedding, "model", type(self.embedding).__name__)

    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> List[List[float]]:
        """Encode chunks into dense vectors.

        Steps: hash texts → serve cache hits → batch-embed only misses →
        write misses back to cache → return vectors in input order.

        Args:
            chunks: List of Chunk objects to encode
            trace: Optional TraceContext for observability

        Returns:
            List of dense vectors (one per chunk, in same order).

        Raises:
            ValueError: If chunks list is empty
            RuntimeError: If embedding provider fails
        """
        if not chunks:
            raise ValueError("Cannot encode empty chunks list")

        texts = [chunk.text for chunk in chunks]

        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(
                    f"Chunk at index {i} (id={chunks[i].id}) has empty or whitespace-only text"
                )

        # ── D-031 cache lookup ─────────────────────────────────────────
        text_hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        cached: dict = {}
        if self.cache is not None:
            try:
                cached = self.cache.get_many(self._cache_model(), text_hashes)
            except Exception as e:  # cache must never break ingestion
                logger.warning(f"Embedding cache lookup failed (ignored): {e}")
                cached = {}

        miss_positions = [i for i, h in enumerate(text_hashes) if h not in cached]
        self.last_cache_stats = {
            "total": len(texts),
            "hits": len(texts) - len(miss_positions),
            "misses": len(miss_positions),
        }
        if cached:
            logger.info(
                f"  Embedding cache: {self.last_cache_stats['hits']} hits / "
                f"{self.last_cache_stats['misses']} misses"
            )

        all_vectors: List[Optional[List[float]]] = [None] * len(texts)
        for i, h in enumerate(text_hashes):
            if h in cached:
                all_vectors[i] = cached[h]

        # ── batch-embed only the misses ────────────────────────────────
        miss_texts = [texts[i] for i in miss_positions]
        fresh_vectors: List[List[float]] = []

        for batch_start in range(0, len(miss_texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(miss_texts))
            batch_texts = miss_texts[batch_start:batch_end]

            try:
                batch_vectors = self.embedding.embed(
                    texts=batch_texts,
                    trace=trace,
                )

                if len(batch_vectors) != len(batch_texts):
                    raise RuntimeError(
                        f"Embedding provider returned {len(batch_vectors)} vectors "
                        f"for {len(batch_texts)} texts in batch {batch_start}-{batch_end}"
                    )

                fresh_vectors.extend(batch_vectors)

            except Exception as e:
                raise RuntimeError(
                    f"Failed to encode batch {batch_start}-{batch_end}: {str(e)}"
                ) from e

        # write fresh vectors back to the cache (best-effort)
        if self.cache is not None and fresh_vectors:
            try:
                self.cache.put_many(
                    self._cache_model(),
                    zip((text_hashes[i] for i in miss_positions), fresh_vectors),
                )
            except Exception as e:
                logger.warning(f"Embedding cache write failed (ignored): {e}")

        for pos, vec in zip(miss_positions, fresh_vectors):
            all_vectors[pos] = vec

        if any(v is None for v in all_vectors):
            missing = sum(1 for v in all_vectors if v is None)
            raise RuntimeError(
                f"Vector count mismatch: {missing} of {len(chunks)} chunks "
                f"ended up without a vector"
            )
        final_vectors: List[List[float]] = [v for v in all_vectors if v is not None]

        # Validate vector dimensions are consistent
        if final_vectors:
            expected_dim = len(final_vectors[0])
            for i, vec in enumerate(final_vectors):
                if len(vec) != expected_dim:
                    raise RuntimeError(
                        f"Inconsistent vector dimensions: vector {i} has "
                        f"{len(vec)} dimensions, expected {expected_dim}"
                    )

        return final_vectors

    def get_batch_count(self, num_chunks: int) -> int:
        """Calculate number of batches needed for given chunk count.

        Utility method for logging/progress tracking. With the D-031 cache
        the actual number of embedding CALLS is the miss count's batches;
        this remains the worst-case (all-miss) batch count.

        Args:
            num_chunks: Number of chunks to encode

        Returns:
            Number of batches required
        """
        if num_chunks <= 0:
            return 0
        return (num_chunks + self.batch_size - 1) // self.batch_size
