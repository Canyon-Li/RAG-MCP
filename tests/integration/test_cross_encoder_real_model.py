"""Integration smoke: CrossEncoderReranker with the real configured model (T15).

Loads the actual cross-encoder configured in settings.yaml and exercises the
paths the mock-based unit tests never touch: model loading from
settings.rerank.model, real scoring, ordering, and the CoreReranker trace
stage (method/provider recorded for the dashboard).

Marked slow — loads a ~2GB torch model. Skips when the cross-encoder extra
is not installed, rerank is not enabled as cross_encoder in settings, or the
model artifact is unavailable locally (env artifact, not a code failure).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from src.core.settings import Settings, load_settings
from src.core.types import RetrievalResult

pytestmark = [pytest.mark.integration, pytest.mark.slow]

QUERY = "How does BB84 quantum key distribution detect eavesdropping?"

CANDIDATES: list[dict[str, Any]] = [
    {
        "id": "irrelevant_weather",
        "text": "Average monthly temperatures in New York show a clear seasonal "
        "pattern, peaking in July and reaching their minimum in late January.",
    },
    {
        "id": "relevant_bb84",
        "text": "The BB84 protocol detects eavesdropping because an interceptor "
        "measuring qubits in the wrong basis disturbs their state, introducing "
        "a detectable error rate into the shared key.",
    },
    {
        "id": "irrelevant_python",
        "text": "Python dictionaries are implemented as hash maps, providing "
        "average O(1) lookup, insertion, and deletion.",
    },
]


@pytest.fixture(scope="module")
def settings() -> Settings:
    return load_settings()


@pytest.fixture(scope="module")
def reranker(settings: Settings):
    pytest.importorskip("sentence_transformers")
    if not settings.rerank.enabled or settings.rerank.provider != "cross_encoder":
        pytest.skip(
            f"rerank not enabled as cross_encoder in settings "
            f"(provider={settings.rerank.provider!r}, enabled={settings.rerank.enabled})"
        )
    from src.libs.reranker.reranker_factory import RerankerFactory

    try:
        return RerankerFactory.create(settings)
    except Exception as exc:  # model missing / hub unreachable — env artifact
        pytest.skip(f"cross-encoder model unavailable: {exc}")


class TestRealCrossEncoder:
    def test_factory_returns_cross_encoder(self, reranker) -> None:
        from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker

        assert isinstance(reranker, CrossEncoderReranker)

    def test_rerank_scores_and_orders_relevant_first(self, reranker) -> None:
        out = reranker.rerank(QUERY, CANDIDATES)

        assert [c["id"] for c in out][0] == "relevant_bb84"
        scores = [c["rerank_score"] for c in out]
        assert scores == sorted(scores, reverse=True)
        assert all(isinstance(s, float) for s in scores)

    def test_core_reranker_records_trace_stage(self, reranker, settings) -> None:
        from src.core.query_engine.reranker import CoreReranker

        core = CoreReranker(settings=settings, reranker=reranker)
        trace = Mock()
        results = [
            RetrievalResult(chunk_id=c["id"], score=0.5 - i * 0.1, text=c["text"], metadata={})
            for i, c in enumerate(CANDIDATES)
        ]

        rr = core.rerank(QUERY, results, top_k=2, trace=trace)

        assert rr.reranker_type == "cross_encoder"
        assert not rr.used_fallback
        assert len(rr.results) == 2
        assert rr.results[0].chunk_id == "relevant_bb84"
        assert all(r.metadata.get("reranked") for r in rr.results)

        trace.record_stage.assert_called_once()
        stage_name, details = trace.record_stage.call_args[0]
        assert stage_name == "rerank"
        assert details["method"] == "cross_encoder"
        assert details["input_count"] == 3
        assert details["output_count"] == 2
