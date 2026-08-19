"""Unit tests for RetrievalService (Playground's retrieval facade)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _rr(chunk_id: str, score: float) -> Any:
    """Build a RetrievalResult-like object (duck-typed)."""
    from src.core.types import RetrievalResult
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=f"text-{chunk_id}",
        metadata={"source_path": "a.pdf", "title": "T"},
    )


def _mock_hybrid() -> MagicMock:
    """HybridSearch stand-in returning a HybridSearchResult-like object."""
    from src.core.query_engine.hybrid_search import HybridSearchResult
    h = MagicMock()
    h.search.return_value = HybridSearchResult(
        results=[_rr("f1", 0.9), _rr("d1", 0.8)],
        dense_results=[_rr("d1", 0.8), _rr("both", 0.7)],
        sparse_results=[_rr("both", 0.7), _rr("s1", 0.6)],
        dense_error=None,
        sparse_error=None,
        used_fallback=False,
        processed_query=MagicMock(keywords=["quantum", "aes"]),
    )
    return h


class TestRetrievalService:
    @patch("src.observability.dashboard.services.retrieval_service._get_hybrid")
    def test_search_converts_details(self, mock_get: MagicMock) -> None:
        from src.observability.dashboard.services.retrieval_service import RetrievalService

        mock_get.return_value = _mock_hybrid()
        svc = RetrievalService()
        result = svc.search("quantum AES", top_k=5, collection="papers")

        assert [c["chunk_id"] for c in result.fused] == ["f1", "d1"]
        assert [c["chunk_id"] for c in result.dense] == ["d1", "both"]
        assert result.intersection == {"both"}          # dense∩sparse
        assert result.keywords == ["quantum", "aes"]
        assert result.used_fallback is False
        assert result.dense_error is None
        # chunk dict shape
        assert result.fused[0]["source"] == "a.pdf"
        assert result.fused[0]["title"] == "T"
        assert isinstance(result.timings["total"], float)

    @patch("src.observability.dashboard.services.retrieval_service._get_hybrid")
    def test_search_empty_query_raises(self, mock_get: MagicMock) -> None:
        from src.observability.dashboard.services.retrieval_service import RetrievalService

        svc = RetrievalService()
        with pytest.raises(ValueError):
            svc.search("   ", top_k=5)

    @patch("src.observability.dashboard.services.retrieval_service._get_hybrid")
    def test_search_passes_params(self, mock_get: MagicMock) -> None:
        from src.observability.dashboard.services.retrieval_service import RetrievalService

        h = _mock_hybrid()
        mock_get.return_value = h
        svc = RetrievalService()
        svc.search("q", top_k=7, collection="papers")
        h.search.assert_called_once_with(query="q", top_k=7, return_details=True)

    def test_list_collections(self) -> None:
        from src.observability.dashboard.services.retrieval_service import RetrievalService

        col = MagicMock()
        col.name = "papers"
        mock_chroma = MagicMock()
        client = mock_chroma.PersistentClient.return_value
        client.list_collections.return_value = [col]
        # list_collections() imports chromadb locally → patch sys.modules
        with patch.dict("sys.modules",
                        {"chromadb": mock_chroma, "chromadb.config": MagicMock()}):
            svc = RetrievalService()
            assert svc.list_collections() == ["papers"]

    def test_list_collections_failure_returns_empty(self) -> None:
        from src.observability.dashboard.services.retrieval_service import RetrievalService

        svc = RetrievalService()
        # chromadb 未初始化/目录缺失 → 空列表而非异常
        mock_chroma = MagicMock()
        mock_chroma.PersistentClient.side_effect = Exception("boom")
        with patch.dict("sys.modules",
                        {"chromadb": mock_chroma, "chromadb.config": MagicMock()}):
            assert svc.list_collections() == []
