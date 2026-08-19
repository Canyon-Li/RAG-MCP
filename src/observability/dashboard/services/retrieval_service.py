"""RetrievalService – retrieval facade for the Retrieval Playground page.

The single retrieval entry point for the Playground (A3 extension seam):
future knobs (rerank toggle, per-path top_k, …) land here, pages only
render. Wraps ``HybridSearch.search(return_details=True)`` and converts
``HybridSearchResult`` into a UI-friendly ``PlaygroundResult``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class PlaygroundResult:
    """UI-friendly view of one hybrid search execution.

    Attributes:
        fused / dense / sparse: chunk dicts ``{chunk_id, score, text,
            source, title}`` in rank order (score rounded to 4 decimals).
        dense_error / sparse_error: error text per path, None on success.
        used_fallback: whether one path failed and the other was used raw.
        keywords: tokenizer output from ProcessedQuery (index-side aligned).
        intersection: chunk_ids present in BOTH dense and sparse results.
        timings: elapsed ms per stage (dense / sparse / fusion / total).
    """

    fused: List[Dict[str, Any]] = field(default_factory=list)
    dense: List[Dict[str, Any]] = field(default_factory=list)
    sparse: List[Dict[str, Any]] = field(default_factory=list)
    dense_error: Optional[str] = None
    sparse_error: Optional[str] = None
    used_fallback: bool = False
    keywords: List[str] = field(default_factory=list)
    intersection: set = field(default_factory=set)
    timings: Dict[str, float] = field(default_factory=dict)


def _to_chunk_dicts(results: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """Convert RetrievalResult objects to plain dicts for rendering."""
    if not results:
        return []
    return [
        {
            "chunk_id": r.chunk_id,
            "score": round(r.score, 4),
            "text": r.text or "",
            "source": r.metadata.get("source_path", r.metadata.get("source", "")),
            "title": r.metadata.get("title", ""),
        }
        for r in results
    ]


@st.cache_resource
def _get_hybrid(collection: str) -> Any:
    """Build a HybridSearch for ``collection``; cached until collection changes.

    Assembly mirrors ``QueryKnowledgeHubTool._ensure_initialized``
    (src/mcp_server/tools/query_knowledge_hub.py:150).
    """
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.hybrid_search import create_hybrid_search
    from src.core.query_engine.dense_retriever import create_dense_retriever
    from src.core.query_engine.sparse_retriever import create_sparse_retriever
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory
    from src.core.settings import load_settings, resolve_path

    settings = load_settings()
    embedding_client = EmbeddingFactory.create(settings)
    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    dense = create_dense_retriever(
        settings=settings, embedding_client=embedding_client, vector_store=vector_store,
    )
    bm25 = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
    sparse = create_sparse_retriever(
        settings=settings, bm25_indexer=bm25, vector_store=vector_store,
    )
    sparse.default_collection = collection  # BM25 must read this collection's dir
    return create_hybrid_search(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=dense,
        sparse_retriever=sparse,
    )


def _clear_hybrid_cache() -> None:
    """Test hook: drop cached HybridSearch instances."""
    _get_hybrid.clear()


class RetrievalService:
    """Facade the Playground page calls; holds no state besides settings."""

    def __init__(self, settings: Any = None) -> None:
        self._settings = settings

    def list_collections(self) -> List[str]:
        """List Chroma collection names; empty list on any failure."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            from src.core.settings import load_settings, resolve_path

            settings = self._settings or load_settings()
            persist_dir = str(resolve_path(settings.vector_store.persist_directory))
            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            names = []
            for col in client.list_collections():
                names.append(col.name if hasattr(col, "name") else str(col))
            return sorted(names)
        except Exception:
            return []

    def search(
        self,
        query: str,
        top_k: int = 10,
        collection: str = "default",
    ) -> PlaygroundResult:
        """Run one hybrid search and return the UI-friendly result.

        Raises:
            ValueError: If query is empty/whitespace.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

        t_total = time.monotonic()
        hybrid = _get_hybrid(collection)
        details = hybrid.search(query=query, top_k=top_k, return_details=True)

        dense_ids = {r.chunk_id for r in (details.dense_results or [])}
        sparse_ids = {r.chunk_id for r in (details.sparse_results or [])}
        timings = {"total": (time.monotonic() - t_total) * 1000.0}

        return PlaygroundResult(
            fused=_to_chunk_dicts(details.results),
            dense=_to_chunk_dicts(details.dense_results),
            sparse=_to_chunk_dicts(details.sparse_results),
            dense_error=details.dense_error,
            sparse_error=details.sparse_error,
            used_fallback=details.used_fallback,
            keywords=list(details.processed_query.keywords) if details.processed_query else [],
            intersection=dense_ids & sparse_ids,
            timings=timings,
        )
