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
            "faithfulness",
            "answer_relevancy",
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
        assert len(evaluator._metric_names) == 5


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

        # Force ollama branch; cache off so no real DiskCacheBackend is built
        monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("RAGAS_JUDGE_CACHE", "0")

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
        """T20: 5 supported — 3 retrieval-era + faithfulness/answer_relevancy."""
        from src.observability.evaluation.ragas_evaluator import SUPPORTED_METRICS
        assert SUPPORTED_METRICS == {
            "context_relevance", "context_precision", "context_recall",
            "faithfulness", "answer_relevancy",
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
        """No reference → context_precision is EXCLUDED (absent key), not a
        pseudo-0.0 that would drag the aggregate mean down (T10)."""
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

        assert "context_precision" not in scores
        mock_cp.return_value.score.assert_not_called()

    def test_context_recall_skipped_without_reference(self) -> None:
        """No reference → context_recall excluded (absent key), symmetric
        with precision (T10)."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator._metric_names = ["context_recall"]
        evaluator.settings = MagicMock()

        with patch(
            "ragas.metrics.collections.ContextRecall"
        ) as mock_crc, patch.object(
            evaluator, "_build_wrappers", return_value=MagicMock()
        ):
            scores = evaluator._run_ragas(
                query="q", contexts=["ctx"], reference=None,
            )

        assert "context_recall" not in scores
        mock_crc.return_value.score.assert_not_called()

    def test_nan_value_excluded(self) -> None:
        """NaN result.value → metric excluded, never recorded (one NaN would
        poison the aggregate mean via sum/len — T10)."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator._metric_names = ["context_relevance"]
        evaluator.settings = MagicMock()

        fake_metric = MagicMock()
        fake_result = MagicMock()
        fake_result.value = float("nan")
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

        assert "context_relevance" not in scores

    def test_none_value_excluded(self) -> None:
        """None result.value → metric excluded (not pseudo-0.0) — T10."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator._metric_names = ["context_relevance"]
        evaluator.settings = MagicMock()

        fake_metric = MagicMock()
        fake_result = MagicMock()
        fake_result.value = None
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

        assert "context_relevance" not in scores

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
        """Missing reference → warning logged, metric EXCLUDED (absent key,
        not pseudo-0.0) — symmetric with precision (T10)."""
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

        assert "context_recall" not in scores
        MockContextRecall.return_value.score.assert_not_called()


class TestBuildWrappersDeepseekBranch:
    """Tests for the deepseek judge branch in _build_wrappers."""

    def test_deepseek_branch_builds_llm_with_env_key(self, monkeypatch) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "deepseek")
        monkeypatch.setenv("RAGAS_JUDGE_MODEL", "deepseek-v4-flash")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
        # T11: cache off — this test pins the exact llm_factory kwargs
        monkeypatch.setenv("RAGAS_JUDGE_CACHE", "0")
        # T22: judge params resolve from env with frozen defaults — delenv
        # so the pinned kwargs below reflect the defaults, not outer env.
        monkeypatch.delenv("RAGAS_JUDGE_MAX_TOKENS", raising=False)
        monkeypatch.delenv("RAGAS_JUDGE_THINKING", raising=False)

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
            max_tokens=16384, cache=None,
            extra_body={"thinking": {"type": "disabled"}},
        )
        # embeddings 清理后返回单值 llm
        assert result is mock_factory.return_value

    def test_deepseek_branch_missing_key_raises(self, monkeypatch) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "deepseek")
        monkeypatch.setenv("RAGAS_JUDGE_CACHE", "0")
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
        monkeypatch.setenv("RAGAS_JUDGE_CACHE", "0")

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
        monkeypatch.setenv("RAGAS_JUDGE_CACHE", "0")

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator.settings = MagicMock()
        with patch("ragas.llms.llm_factory") as mock_factory, \
             patch("openai.AsyncOpenAI"):
            result = evaluator._build_wrappers()

        assert result is mock_factory.return_value


class TestJudgeParamsT22:
    """T22: judge max_tokens / thinking parameterisation (gauge freeze).

    T20 lesson: the faithfulness decomposition of number-dense answers
    overflowed the hardcoded max_tokens=8192 (IncompleteOutputException —
    decomposition inflation, NOT answer length). T21 lesson: deepseek-flash
    defaults thinking ON (billed reasoning tokens, slower). Both frozen at
    the T22 baseline: 16384 + thinking off, env-swappable for rollback
    without code edits. Changing max_tokens auto-invalidates the judge
    DiskCache (the cache key includes it) — expected, baseline runs are
    --no-judge-cache anyway.
    """

    # ── max_tokens resolution ──────────────────────────────────────

    def test_max_tokens_default_16384(self, monkeypatch) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.delenv("RAGAS_JUDGE_MAX_TOKENS", raising=False)
        assert RagasEvaluator._resolve_judge_max_tokens() == 16384

    def test_max_tokens_env_override(self, monkeypatch) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_MAX_TOKENS", "12000")
        assert RagasEvaluator._resolve_judge_max_tokens() == 12000

    @pytest.mark.parametrize("bad", ["abc", "0", "-5", "8.5"])
    def test_max_tokens_invalid_raises(self, monkeypatch, bad) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_MAX_TOKENS", bad)
        with pytest.raises(ValueError, match="RAGAS_JUDGE_MAX_TOKENS"):
            RagasEvaluator._resolve_judge_max_tokens()

    def test_max_tokens_whitespace_means_unset(self, monkeypatch) -> None:
        """Whitespace-only value strips to empty → treated as unset (default),
        not garbage: the strip-to-empty-as-unset convention."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_MAX_TOKENS", "   ")
        assert RagasEvaluator._resolve_judge_max_tokens() == 16384

    # ── thinking resolution ────────────────────────────────────────

    def test_thinking_default_off(self, monkeypatch) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.delenv("RAGAS_JUDGE_THINKING", raising=False)
        assert RagasEvaluator._resolve_judge_thinking() == {
            "thinking": {"type": "disabled"},
        }

    def test_thinking_on_maps_enabled(self, monkeypatch) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_THINKING", "on")
        assert RagasEvaluator._resolve_judge_thinking() == {
            "thinking": {"type": "enabled"},
        }

    def test_thinking_invalid_raises(self, monkeypatch) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_THINKING", "maybe")
        with pytest.raises(ValueError, match="RAGAS_JUDGE_THINKING"):
            RagasEvaluator._resolve_judge_thinking()

    # ── wiring into llm_factory ────────────────────────────────────

    def test_deepseek_branch_env_override_params(self, monkeypatch) -> None:
        """RAGAS_JUDGE_MAX_TOKENS / RAGAS_JUDGE_THINKING override the frozen
        defaults end-to-end into the llm_factory call."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
        monkeypatch.setenv("RAGAS_JUDGE_CACHE", "0")
        monkeypatch.setenv("RAGAS_JUDGE_MAX_TOKENS", "12000")
        monkeypatch.setenv("RAGAS_JUDGE_THINKING", "on")

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator.settings = MagicMock()
        with patch("ragas.llms.llm_factory") as mock_factory, \
             patch("openai.AsyncOpenAI") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            evaluator._build_wrappers()

        kwargs = mock_factory.call_args.kwargs
        assert kwargs["max_tokens"] == 12000
        assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}

    def test_ollama_branch_forwards_no_thinking(self, monkeypatch) -> None:
        """thinking rides extra_body — a deepseek-specific param. The ollama
        branch must not forward it (foreign endpoints may reject unknown
        params). max_tokens IS shared: the 8192 overflow was judge-side,
        provider-independent."""
        import sys as _sys

        monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("RAGAS_JUDGE_CACHE", "0")
        monkeypatch.delenv("RAGAS_JUDGE_MAX_TOKENS", raising=False)
        monkeypatch.delenv("RAGAS_JUDGE_THINKING", raising=False)

        mock_llms_mod = MagicMock()
        mock_llms_mod.llm_factory = MagicMock(name="ragas_llm_factory_fn")
        ragas_stubs = {
            "ragas": MagicMock(),
            "ragas.llms": mock_llms_mod,
            "ragas.llms.base": MagicMock(),
        }

        with patch.dict(_sys.modules, ragas_stubs, clear=False), \
             patch("openai.AsyncOpenAI"):
            from src.observability.evaluation.ragas_evaluator import RagasEvaluator

            settings = MagicMock()
            settings.llm.provider = "ollama"
            settings.embedding.provider = "ollama"

            evaluator = RagasEvaluator.__new__(RagasEvaluator)
            evaluator.settings = settings
            evaluator._build_wrappers()

        kwargs = mock_llms_mod.llm_factory.call_args.kwargs
        assert "extra_body" not in kwargs
        assert kwargs["max_tokens"] == 16384


class TestAnswerSideMetrics:
    """T20: faithfulness / answer_relevancy — the generation-era metrics.

    ragas 0.4.3 collections metrics verified live before implementation:
    Faithfulness(llm) scores (user_input, response, retrieved_contexts);
    AnswerRelevancy(llm, embeddings) scores (user_input, response) and its
    embeddings arg is REQUIRED at __init__ (missing → TypeError).
    """

    def test_supported_metrics_expanded(self) -> None:
        from src.observability.evaluation.ragas_evaluator import SUPPORTED_METRICS
        assert SUPPORTED_METRICS == {
            "context_relevance", "context_precision", "context_recall",
            "faithfulness", "answer_relevancy",
        }

    def test_init_accepts_new_metrics(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["faithfulness", "answer_relevancy"])
        assert evaluator._metric_names == ["faithfulness", "answer_relevancy"]

    def test_faithfulness_called_with_response(self) -> None:
        """faithfulness passes response= (the generated answer) plus contexts."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator._metric_names = ["faithfulness"]
        evaluator.settings = MagicMock()

        fake_metric = MagicMock()
        fake_result = MagicMock()
        fake_result.value = 0.9
        fake_metric.score.return_value = fake_result

        with patch(
            "ragas.metrics.collections.Faithfulness"
        ) as mock_f, patch.object(
            evaluator, "_build_wrappers", return_value=MagicMock(name="llm")
        ):
            mock_f.return_value = fake_metric
            scores = evaluator._run_ragas(
                query="q", contexts=["ctx"], reference="ref", response="ans",
            )

        assert scores["faithfulness"] == 0.9
        mock_f.return_value.score.assert_called_once_with(
            user_input="q", response="ans", retrieved_contexts=["ctx"],
        )
        # constructed with llm= (required positional in ragas 0.4.3)
        assert mock_f.call_args.kwargs.get("llm") is not None

    def test_faithfulness_skipped_without_response(self) -> None:
        """No generated answer → excluded (absent key), never a pseudo-0.0 —
        symmetric with the reference-missing rule (T10)."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator._metric_names = ["faithfulness"]
        evaluator.settings = MagicMock()

        with patch(
            "ragas.metrics.collections.Faithfulness"
        ) as mock_f, patch.object(
            evaluator, "_build_wrappers", return_value=MagicMock()
        ):
            scores = evaluator._run_ragas(
                query="q", contexts=["ctx"], reference="ref", response=None,
            )

        assert "faithfulness" not in scores
        mock_f.return_value.score.assert_not_called()

    def test_answer_relevancy_constructed_with_embeddings(self) -> None:
        """answer_relevancy needs llm + embeddings at construction and scores
        with (user_input, response) only — no retrieved_contexts."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator._metric_names = ["answer_relevancy"]
        evaluator.settings = MagicMock()

        fake_metric = MagicMock()
        fake_result = MagicMock()
        fake_result.value = 0.7
        fake_metric.score.return_value = fake_result

        with patch(
            "ragas.metrics.collections.AnswerRelevancy"
        ) as mock_ar, patch.object(
            evaluator, "_build_wrappers", return_value=MagicMock(name="llm")
        ), patch.object(
            evaluator, "_build_judge_embeddings",
            return_value=MagicMock(name="emb"),
        ) as mock_emb_build:
            mock_ar.return_value = fake_metric
            scores = evaluator._run_ragas(
                query="q", contexts=["ctx"], reference="ref", response="ans",
            )

        assert scores["answer_relevancy"] == 0.7
        kwargs = mock_ar.call_args.kwargs
        assert kwargs.get("llm") is not None
        assert kwargs.get("embeddings") is not None
        mock_ar.return_value.score.assert_called_once_with(
            user_input="q", response="ans",
        )
        # embeddings are only built when this metric is in the list
        mock_emb_build.assert_called_once()

    def test_answer_relevancy_skipped_without_response(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator._metric_names = ["answer_relevancy"]
        evaluator.settings = MagicMock()

        with patch(
            "ragas.metrics.collections.AnswerRelevancy"
        ) as mock_ar, patch.object(
            evaluator, "_build_wrappers", return_value=MagicMock()
        ):
            scores = evaluator._run_ragas(
                query="q", contexts=["ctx"], reference="ref", response=None,
            )

        assert "answer_relevancy" not in scores
        mock_ar.return_value.score.assert_not_called()

    def test_context_metrics_do_not_build_embeddings(self) -> None:
        """Retrieval-only metric mix must not require a judge embedding —
        answer_relevancy drives the only emb construction."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator._metric_names = ["context_precision"]
        evaluator.settings = MagicMock()

        fake_metric = MagicMock()
        fake_result = MagicMock()
        fake_result.value = 0.5
        fake_metric.score.return_value = fake_result

        with patch(
            "ragas.metrics.collections.ContextPrecision"
        ) as mock_cp, patch.object(
            evaluator, "_build_wrappers", return_value=MagicMock()
        ), patch.object(
            evaluator, "_build_judge_embeddings",
        ) as mock_emb_build:
            mock_cp.return_value = fake_metric
            evaluator._run_ragas(
                query="q", contexts=["ctx"], reference="ref", response="ans",
            )

        mock_emb_build.assert_not_called()

    def test_evaluate_forwards_generated_answer(self) -> None:
        """evaluate() threads generated_answer into _run_ragas as response."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator._metric_names = ["faithfulness"]
        evaluator.settings = MagicMock()

        with patch.object(
            evaluator, "_run_ragas", return_value={"faithfulness": 0.9}
        ) as mock_run:
            evaluator.evaluate(
                query="q",
                retrieved_chunks=[{"text": "ctx"}],
                generated_answer="the answer",
                ground_truth={"reference": "ref"},
            )

        assert mock_run.call_args.kwargs.get("response") == "the answer"


class TestJudgeEmbeddingsBuilder:
    """T20: local Ollama embeddings for AnswerRelevancy (deepseek has no
    embedding API — the judge's only local component)."""

    def test_builds_openai_embeddings_against_ollama(self, monkeypatch) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        monkeypatch.delenv("RAGAS_JUDGE_EMB_MODEL", raising=False)

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator.settings = MagicMock()

        with patch(
            "ragas.embeddings.OpenAIEmbeddings"
        ) as mock_emb_cls, patch(
            "openai.AsyncOpenAI"
        ) as mock_client_cls:
            mock_client_cls.return_value = MagicMock(name="client")
            evaluator._build_judge_embeddings()

        client_kwargs = mock_client_cls.call_args.kwargs
        assert "localhost:11434/v1" in client_kwargs["base_url"]
        assert client_kwargs["api_key"] == "ollama"
        emb_kwargs = mock_emb_cls.call_args.kwargs
        assert emb_kwargs["model"] == "nomic-embed-text"
        assert emb_kwargs["client"] is mock_client_cls.return_value

    def test_emb_model_env_override(self, monkeypatch) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_EMB_MODEL", "bge-m3")

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator.settings = MagicMock()

        with patch(
            "ragas.embeddings.OpenAIEmbeddings"
        ) as mock_emb_cls, patch(
            "openai.AsyncOpenAI"
        ):
            evaluator._build_judge_embeddings()

        assert mock_emb_cls.call_args.kwargs["model"] == "bge-m3"


class TestPerMetricIsolation:
    """T20 first-run lesson: a single judge-call transport failure (deepseek
    connect timeout under the ~7-calls/question 4-metric volume) must drop
    ONLY that metric — not zero out all four via exception propagation."""

    def _make_evaluator(self, metric_names):
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator._metric_names = metric_names
        evaluator.settings = MagicMock()
        return evaluator

    def test_one_metric_raising_excludes_only_itself(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = self._make_evaluator(
            ["faithfulness", "context_precision"]
        )
        good_result = MagicMock()
        good_result.value = 0.8

        with patch(
            "ragas.metrics.collections.Faithfulness"
        ) as mock_f, patch(
            "ragas.metrics.collections.ContextPrecision"
        ) as mock_cp, patch.object(
            evaluator, "_build_wrappers", return_value=MagicMock()
        ):
            mock_f.return_value.score.side_effect = TimeoutError(
                "APITimeoutError"
            )
            mock_cp.return_value.score.return_value = good_result
            scores = evaluator._run_ragas(
                query="q", contexts=["ctx"], reference="ref", response="ans",
            )

        assert scores == {"context_precision": 0.8}

    def test_all_metrics_raising_returns_empty_not_raise(self) -> None:
        evaluator = self._make_evaluator(["faithfulness", "context_recall"])

        with patch(
            "ragas.metrics.collections.Faithfulness"
        ) as mock_f, patch(
            "ragas.metrics.collections.ContextRecall"
        ) as mock_crc, patch.object(
            evaluator, "_build_wrappers", return_value=MagicMock()
        ):
            mock_f.return_value.score.side_effect = TimeoutError("t")
            mock_crc.return_value.score.side_effect = TimeoutError("t")
            scores = evaluator._run_ragas(
                query="q", contexts=["ctx"], reference="ref", response="ans",
            )

        assert scores == {}


class TestJudgeDiskCache:
    """T11: judge disk cache — default on, env off-switch, llm_factory wiring.

    The cache replays exact-match judge calls (same prompt + model params +
    response model) at zero cost/noise; only retrieval-changed questions get
    re-judged. Fresh-sample runs (final gate) must disable it.
    """

    def test_cache_default_on_returns_backend(self, monkeypatch, tmp_path) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.delenv("RAGAS_JUDGE_CACHE", raising=False)
        monkeypatch.setenv("RAGAS_JUDGE_CACHE_DIR", str(tmp_path / "jc"))

        cache = RagasEvaluator._resolve_judge_cache()

        assert cache is not None
        # DiskCacheBackend wraps a diskcache.Cache
        assert hasattr(cache, "cache")

    def test_cache_respects_custom_dir(self, monkeypatch, tmp_path) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_CACHE_DIR", str(tmp_path / "custom"))

        cache = RagasEvaluator._resolve_judge_cache()

        assert cache is not None
        assert str(tmp_path / "custom") in str(getattr(cache.cache, "directory", ""))

    @pytest.mark.parametrize("off", ["0", "false", "off", "no"])
    def test_cache_disabled_by_env(self, monkeypatch, off) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_CACHE", off)

        assert RagasEvaluator._resolve_judge_cache() is None

    def test_build_wrappers_passes_cache_when_on(self, monkeypatch, tmp_path) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
        monkeypatch.delenv("RAGAS_JUDGE_CACHE", raising=False)
        monkeypatch.setenv("RAGAS_JUDGE_CACHE_DIR", str(tmp_path / "jc"))

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator.settings = MagicMock()

        with patch("ragas.llms.llm_factory") as mock_factory, \
             patch("openai.AsyncOpenAI") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            evaluator._build_wrappers()

        assert mock_factory.call_args.kwargs.get("cache") is not None

    def test_build_wrappers_cache_none_when_off(self, monkeypatch) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
        monkeypatch.setenv("RAGAS_JUDGE_CACHE", "0")

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator.settings = MagicMock()

        with patch("ragas.llms.llm_factory") as mock_factory, \
             patch("openai.AsyncOpenAI"):
            evaluator._build_wrappers()

        assert mock_factory.call_args.kwargs.get("cache") is None
