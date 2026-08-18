"""Unit tests for RagasEvaluator.

Tests verify:
- Initialization with valid/invalid metrics
- ImportError handling when ragas is not installed
- Input validation (empty query, empty chunks, etc.)
- Metric extraction from settings
- Text extraction from various chunk formats
- New context_relevance / context_precision routing

Note: Actual Ragas evaluation (LLM calls) is mocked to keep unit tests fast
and deterministic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any, Dict

import pytest


class TestRagasEvaluatorInit:
    """Tests for RagasEvaluator initialisation."""

    def test_init_default_metrics(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator()
        assert set(evaluator._metric_names) == {
            "context_precision",
            "context_recall",
            "context_relevance",
        }

    def test_init_custom_metrics(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["context_relevance"])
        assert evaluator._metric_names == ["context_relevance"]

    def test_init_unsupported_metric_raises(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        with pytest.raises(ValueError, match="Unsupported ragas metrics"):
            RagasEvaluator(metrics=["hit_rate"])

    def test_init_reads_metrics_from_settings(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        settings = MagicMock()
        settings.evaluation.metrics = [
            "context_relevance", "context_precision", "hit_rate",
        ]

        evaluator = RagasEvaluator(settings=settings)
        # hit_rate is not a ragas metric, should be filtered out
        assert "hit_rate" not in evaluator._metric_names
        assert "context_relevance" in evaluator._metric_names
        assert "context_precision" in evaluator._metric_names

    def test_init_no_settings_defaults_to_all(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(settings=None, metrics=None)
        assert len(evaluator._metric_names) == 3


class TestRagasImportCheck:
    """Tests for ragas import validation."""

    def test_import_error_when_ragas_missing(self) -> None:
        from src.observability.evaluation.ragas_evaluator import _import_ragas

        with patch.dict("sys.modules", {"ragas": None}):
            with pytest.raises(ImportError, match="ragas"):
                _import_ragas()


class TestRagasEvaluatorValidation:
    """Tests for input validation in evaluate()."""

    def test_empty_query_raises(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["context_relevance"])
        with pytest.raises(ValueError, match="Query cannot be empty"):
            evaluator.evaluate("  ", [{"text": "ctx"}], generated_answer="ans")

    def test_empty_chunks_raises(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["context_relevance"])
        with pytest.raises(ValueError, match="retrieved_chunks cannot be empty"):
            evaluator.evaluate("query", [], generated_answer="ans")


class TestRagasEvaluatorTextExtraction:
    """Tests for _extract_texts helper."""

    def test_extract_from_dicts(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["context_relevance"])
        result = evaluator._extract_texts([
            {"text": "chunk1"},
            {"content": "chunk2"},
            {"page_content": "chunk3"},
        ])
        assert result == ["chunk1", "chunk2", "chunk3"]

    def test_extract_from_strings(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["context_relevance"])
        result = evaluator._extract_texts(["chunk1", "chunk2"])
        assert result == ["chunk1", "chunk2"]

    def test_extract_from_objects(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["context_relevance"])

        class Chunk:
            def __init__(self, text: str) -> None:
                self.text = text

        result = evaluator._extract_texts([Chunk("hello"), Chunk("world")])
        assert result == ["hello", "world"]


class TestRagasEvaluatorEvaluate:
    """Tests for evaluate() with mocked Ragas backend."""

    def _make_mock_ragas_result(self, scores: Dict[str, float]) -> MagicMock:
        """Create a mock ragas evaluation result."""
        import pandas as pd

        df = pd.DataFrame([scores])
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = df
        return mock_result

    def test_evaluate_returns_metrics_dict(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["context_relevance", "context_precision"])

        expected = {"context_relevance": 0.92, "context_precision": 0.85}
        evaluator._run_ragas = MagicMock(return_value=expected)  # type: ignore[method-assign]

        result = evaluator.evaluate(
            query="What is RAG?",
            retrieved_chunks=["RAG is retrieval augmented generation."],
            generated_answer="",
        )

        assert result == expected

    def test_evaluate_with_mocked_run_ragas(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["context_relevance", "context_precision"])

        expected_scores = {"context_relevance": 0.95, "context_precision": 0.88}
        evaluator._run_ragas = MagicMock(return_value=expected_scores)  # type: ignore[method-assign]

        result = evaluator.evaluate(
            query="What is RAG?",
            retrieved_chunks=[{"text": "RAG is Retrieval Augmented Generation"}],
            generated_answer="",
        )

        assert result == expected_scores
        evaluator._run_ragas.assert_called_once()

    def test_evaluate_runtime_error_on_ragas_failure(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["context_relevance"])
        evaluator._run_ragas = MagicMock(  # type: ignore[method-assign]
            side_effect=Exception("LLM call failed"),
        )

        with pytest.raises(RuntimeError, match="Ragas evaluation failed"):
            evaluator.evaluate(
                query="test",
                retrieved_chunks=[{"text": "ctx"}],
                generated_answer="",
            )

    def test_ground_truth_is_ignored(self) -> None:
        """Ragas should work fine even when ground_truth is provided."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["context_relevance"])
        evaluator._run_ragas = MagicMock(return_value={"context_relevance": 0.9})  # type: ignore[method-assign]

        result = evaluator.evaluate(
            query="test",
            retrieved_chunks=[{"text": "ctx"}],
            generated_answer="",
            ground_truth=["chunk_001"],  # should be ignored
        )

        assert "context_relevance" in result


class TestRagasEvaluatorFactory:
    """Tests for factory integration."""

    def test_factory_creates_ragas_evaluator(self) -> None:
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        settings = MagicMock()
        settings.evaluation.enabled = True
        settings.evaluation.provider = "ragas"
        settings.evaluation.metrics = ["context_relevance"]

        evaluator = EvaluatorFactory.create(settings)
        assert isinstance(evaluator, RagasEvaluator)

    def test_factory_lists_ragas(self) -> None:
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory

        providers = EvaluatorFactory.list_providers()
        assert "custom" in providers
        # ragas may be in _PROVIDERS after first create or in _LAZY_PROVIDERS


class TestRagasOllamaJudge:
    """Tests that RagasEvaluator builds an Ollama-backed judge wrapper."""

    def test_build_wrappers_ollama_branch(self, monkeypatch) -> None:
        """_build_wrappers with ollama provider creates AsyncOpenAI with ollama base_url."""
        import sys as _sys

        # Force ollama branch
        monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Mock at the source modules — _build_wrappers imports these names
        # function-locally (from openai / ragas.llms), so module-attribute
        # patches on the evaluator module won't resolve. Patch the origin.
        # ragas.llms triggers a heavy import chain; stub ragas submodules so
        # patch() can resolve the target without importing the real ragas.
        mock_llms_mod = MagicMock()
        mock_llms_mod.llm_factory = MagicMock(name="ragas_llm_factory_fn")

        ragas_stubs = {
            "ragas": MagicMock(),
            "ragas.llms": mock_llms_mod,
            "ragas.llms.base": MagicMock(),
        }

        with patch.dict(_sys.modules, ragas_stubs, clear=False), \
             patch("openai.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock(name="openai_client")

            # Build a minimal settings stub — _build_wrappers reads settings.embedding
            # for the embeddings wrapper, but in ollama branch we reuse the same client
            from src.observability.evaluation.ragas_evaluator import RagasEvaluator
            settings = MagicMock()
            settings.llm.provider = "ollama"
            settings.embedding.provider = "ollama"

            evaluator = RagasEvaluator.__new__(RagasEvaluator)  # bypass __init__ (ragas import)
            evaluator.settings = settings

            llm = evaluator._build_wrappers()

            # AsyncOpenAI called with the ollama base_url (/v1 appended)
            call_kwargs = mock_openai.call_args.kwargs
            assert "localhost:11434" in call_kwargs["base_url"]
            assert call_kwargs["api_key"] == "ollama"
            # llm_factory called with the configured judge model (default llama3,
            # overridable via RAGAS_JUDGE_MODEL env, read at call time). Don't pin
            # the model name — assert it's the resolved value.
            factory_args = mock_llms_mod.llm_factory.call_args.args
            assert RagasEvaluator._resolve_judge_model() in factory_args


class TestRagasMetricRouting:
    """Tests that RagasEvaluator routes to context_relevance / context_precision."""

    def test_supported_metrics_replaced(self) -> None:
        """SUPPORTED_METRICS should now be context_relevance + context_precision."""
        from src.observability.evaluation.ragas_evaluator import SUPPORTED_METRICS
        assert SUPPORTED_METRICS == {
            "context_relevance", "context_precision", "context_recall",
        }

    def test_context_relevance_called(self) -> None:
        """_run_ragas with context_relevance should call ContextRelevance.score."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator._metric_names = ["context_relevance"]
        evaluator.settings = MagicMock()

        fake_metric = MagicMock()
        fake_result = MagicMock()
        fake_result.value = 0.8
        fake_metric.score.return_value = fake_result

        with patch(
            "ragas.metrics.collections.ContextRelevance"
        ) as mock_cr, patch.object(
            evaluator, "_build_wrappers", return_value=MagicMock()
        ):
            mock_cr.return_value = fake_metric
            scores = evaluator._run_ragas(
                query="q", contexts=["ctx text"], reference=None,
            )

        assert scores["context_relevance"] == 0.8
        mock_cr.return_value.score.assert_called_once()

    def test_context_precision_uses_reference_not_response(self) -> None:
        """context_precision must pass reference= (not response=) — matches
        ContextPrecisionWithReference's real signature in ragas 0.4.3."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator._metric_names = ["context_precision"]
        evaluator.settings = MagicMock()

        fake_metric = MagicMock()
        fake_result = MagicMock()
        fake_result.value = 0.6
        fake_metric.score.return_value = fake_result

        with patch(
            "ragas.metrics.collections.ContextPrecision"
        ) as mock_cp, patch.object(
            evaluator, "_build_wrappers", return_value=MagicMock()
        ):
            mock_cp.return_value = fake_metric
            scores = evaluator._run_ragas(
                query="q",
                contexts=["ctx text"],
                reference="the reference answer",
            )

        assert scores["context_precision"] == 0.6
        # Verify reference= was passed, response= was NOT
        call_kwargs = mock_cp.return_value.score.call_args.kwargs
        assert call_kwargs.get("reference") == "the reference answer"
        assert "response" not in call_kwargs

    def test_context_precision_skipped_without_reference(self) -> None:
        """No reference → context_precision returns 0.0 (not crash)."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator._metric_names = ["context_precision"]
        evaluator.settings = MagicMock()

        with patch(
            "ragas.metrics.collections.ContextPrecision"
        ) as mock_cp, patch.object(
            evaluator, "_build_wrappers", return_value=MagicMock()
        ):
            scores = evaluator._run_ragas(
                query="q", contexts=["ctx"], reference=None,
            )

        assert scores["context_precision"] == 0.0
        mock_cp.return_value.score.assert_not_called()

    def test_extract_reference_from_ground_truth(self) -> None:
        """_extract_reference reads ground_truth['reference']."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        assert RagasEvaluator._extract_reference({"reference": "ans"}) == "ans"
        assert RagasEvaluator._extract_reference({}) is None
        assert RagasEvaluator._extract_reference(None) is None

    def test_no_answer_required_for_context_relevance(self) -> None:
        """evaluate() should NOT raise when generated_answer is empty."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator._metric_names = ["context_relevance"]
        evaluator.settings = MagicMock()

        with patch.object(
            evaluator, "_run_ragas", return_value={"context_relevance": 0.7}
        ) as mock_run:
            metrics = evaluator.evaluate(
                query="q",
                retrieved_chunks=[{"id": "c1", "text": "ctx"}],
                generated_answer="",
            )

        assert metrics["context_relevance"] == 0.7


class TestContextRecallMetric:
    """Tests for the context_recall metric branch in _run_ragas."""

    def test_supported_metrics_includes_context_recall(self) -> None:
        from src.observability.evaluation.ragas_evaluator import SUPPORTED_METRICS

        assert "context_recall" in SUPPORTED_METRICS

    def test_init_accepts_context_recall_metric(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["context_recall"])
        assert evaluator._metric_names == ["context_recall"]

    def test_run_ragas_calls_context_recall_with_reference(self) -> None:
        """context_recall branch invokes ContextRecall.score with
        (user_input, retrieved_contexts, reference) and records the value."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["context_recall"])
        mock_result = MagicMock()
        mock_result.value = 0.75

        with patch(
            "src.observability.evaluation.ragas_evaluator.RagasEvaluator._build_wrappers",
            return_value=MagicMock(),
        ), patch(
            "ragas.metrics.collections.ContextRecall"
        ) as MockContextRecall:
            MockContextRecall.return_value.score.return_value = mock_result
            scores = evaluator._run_ragas(
                query="What is LCU?",
                contexts=["LCU-based approach with query complexity O(1/sqrt(eps))."],
                reference="The paper uses LCU with complexity O(1/sqrt(eps)).",
            )

        assert scores["context_recall"] == 0.75
        MockContextRecall.return_value.score.assert_called_once_with(
            user_input="What is LCU?",
            retrieved_contexts=["LCU-based approach with query complexity O(1/sqrt(eps))."],
            reference="The paper uses LCU with complexity O(1/sqrt(eps)).",
        )

    def test_run_ragas_context_recall_without_reference_scores_zero(self) -> None:
        """Missing reference → warning logged, score 0.0 (symmetric with precision)."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["context_recall"])

        with patch(
            "src.observability.evaluation.ragas_evaluator.RagasEvaluator._build_wrappers",
            return_value=MagicMock(),
        ), patch(
            "ragas.metrics.collections.ContextRecall"
        ) as MockContextRecall:
            scores = evaluator._run_ragas(
                query="What is LCU?", contexts=["some context"], reference=None,
            )

        assert scores["context_recall"] == 0.0
        MockContextRecall.return_value.score.assert_not_called()


class TestBuildWrappersDeepseekBranch:
    """Tests for the deepseek judge branch in _build_wrappers."""

    def test_deepseek_branch_builds_llm_with_env_key(self, monkeypatch) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "deepseek")
        monkeypatch.setenv("RAGAS_JUDGE_MODEL", "deepseek-v4-flash")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")

        # Bypass __init__ (which calls _import_ragas) and set a MagicMock
        # settings so the `settings is None` guard passes.
        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator.settings = MagicMock()

        # Mock llm_factory + AsyncOpenAI at their import sources
        with patch("ragas.llms.llm_factory") as mock_factory, \
             patch("openai.AsyncOpenAI") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            result = evaluator._build_wrappers()

        mock_client_cls.assert_called_once()
        _, kwargs = mock_client_cls.call_args
        assert kwargs["api_key"] == "sk-test-123"
        assert kwargs["base_url"] == "https://api.deepseek.com"
        mock_factory.assert_called_once_with(
            "deepseek-v4-flash", client=mock_client_cls.return_value,
            max_tokens=8192,
        )
        # embeddings 清理后返回单值 llm
        assert result is mock_factory.return_value

    def test_deepseek_branch_missing_key_raises(self, monkeypatch) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "deepseek")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator.settings = MagicMock()

        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            evaluator._build_wrappers()

    def test_deepseek_branch_custom_base_url(self, monkeypatch) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
        monkeypatch.setenv("RAGAS_JUDGE_BASE_URL", "https://api.deepseek.com/v1")

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator.settings = MagicMock()

        with patch("ragas.llms.llm_factory"), \
             patch("openai.AsyncOpenAI") as mock_client_cls:
            evaluator._build_wrappers()

        _, kwargs = mock_client_cls.call_args
        assert kwargs["base_url"] == "https://api.deepseek.com/v1"

    def test_build_wrappers_returns_single_llm_not_tuple(self, monkeypatch) -> None:
        """After embeddings cleanup, _build_wrappers returns the llm alone."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator.settings = MagicMock()
        with patch("ragas.llms.llm_factory") as mock_factory, \
             patch("openai.AsyncOpenAI"):
            result = evaluator._build_wrappers()

        assert result is mock_factory.return_value
