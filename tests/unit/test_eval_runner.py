"""Unit tests for EvalRunner and golden test set loading."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.core.types import RetrievalResult
from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.observability.evaluation.eval_runner import (
    EvalRunner,
    EvalReport,
    GoldenTestCase,
    QueryResult,
    load_test_set,
)


# ── Fixtures / Helpers ────────────────────────────────────────────


class StubEvaluator(BaseEvaluator):
    """Evaluator that returns fixed metrics for testing."""

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        return {"hit_rate": 1.0, "mrr": 0.5}


def _write_golden_json(path: Path, test_cases: List[Dict]) -> None:
    data = {"test_cases": test_cases}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ── Tests: load_test_set ──────────────────────────────────────────


class TestLoadTestSet:
    def test_load_valid_file(self, tmp_path: Path) -> None:
        f = tmp_path / "golden.json"
        _write_golden_json(f, [
            {"query": "What is RAG?", "expected_chunk_ids": ["c1"]},
            {"query": "How does BM25 work?"},
        ])

        cases = load_test_set(f)

        assert len(cases) == 2
        assert cases[0].query == "What is RAG?"
        assert cases[0].expected_chunk_ids == ["c1"]
        assert cases[1].expected_chunk_ids == []

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_test_set("nonexistent.json")

    def test_load_invalid_format_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text('{"wrong_key": []}', encoding="utf-8")

        with pytest.raises(ValueError, match="missing 'test_cases'"):
            load_test_set(f)


class TestLoadTestSetJsonl:
    """T20: ragas-native SingleTurnSample jsonl (one sample per line).

    The jsonl dialect is produced by ragas ``EvaluationDataset.to_jsonl``
    (verified round-trip against ragas 0.4.3) and parsed here with stdlib
    json only — importing ragas just to read two fields would drag its
    vertexai stub into every eval_runner consumer.
    """

    def test_load_ragas_jsonl(self, tmp_path: Path) -> None:
        f = tmp_path / "golden.jsonl"
        f.write_text(
            '{"user_input": "What is RAG?", "reference": "RAG is ..."}\n'
            '{"user_input": "How does BM25 work?"}\n',
            encoding="utf-8",
        )

        cases = load_test_set(f)

        assert len(cases) == 2
        assert cases[0].query == "What is RAG?"
        assert cases[0].reference_answer == "RAG is ..."
        assert cases[0].expected_chunk_ids == []
        assert cases[0].expected_sources == []
        assert cases[1].query == "How does BM25 work?"
        assert cases[1].reference_answer is None

    def test_jsonl_blank_lines_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "golden.jsonl"
        f.write_text(
            '\n{"user_input": "q1"}\n\n\n{"user_input": "q2"}\n',
            encoding="utf-8",
        )

        cases = load_test_set(f)

        assert [c.query for c in cases] == ["q1", "q2"]

    def test_jsonl_line_missing_user_input_raises(
        self, tmp_path: Path,
    ) -> None:
        f = tmp_path / "golden.jsonl"
        f.write_text('{"user_input": "q1"}\n{"reference": "no query"}\n',
                     encoding="utf-8")

        with pytest.raises(ValueError, match="line 2.*user_input"):
            load_test_set(f)

    def test_jsonl_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        f = tmp_path / "golden.jsonl"
        f.write_text("", encoding="utf-8")

        assert load_test_set(f) == []


# ── Tests: TestCase ───────────────────────────────────────────────


class TestGoldenTestCase:
    def test_from_dict_full(self) -> None:
        tc = GoldenTestCase.from_dict({
            "query": "Q",
            "expected_chunk_ids": ["a", "b"],
            "expected_sources": ["doc.pdf"],
            "reference_answer": "Answer",
        })
        assert tc.query == "Q"
        assert tc.expected_chunk_ids == ["a", "b"]
        assert tc.expected_sources == ["doc.pdf"]
        assert tc.reference_answer == "Answer"

    def test_from_dict_minimal(self) -> None:
        tc = GoldenTestCase.from_dict({"query": "Q"})
        assert tc.expected_chunk_ids == []
        assert tc.reference_answer is None


# ── Tests: EvalRunner ─────────────────────────────────────────────


class TestEvalRunner:
    def test_run_without_evaluator_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "g.json"
        _write_golden_json(f, [{"query": "Q"}])

        runner = EvalRunner(evaluator=None)
        with pytest.raises(ValueError, match="requires an evaluator"):
            runner.run(f)

    def test_run_empty_test_set_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "g.json"
        _write_golden_json(f, [])

        runner = EvalRunner(evaluator=StubEvaluator())
        with pytest.raises(ValueError, match="empty"):
            runner.run(f)

    def test_run_with_stub_evaluator(self, tmp_path: Path) -> None:
        f = tmp_path / "g.json"
        _write_golden_json(f, [
            {"query": "What is RAG?"},
            {"query": "How does hybrid search work?"},
        ])

        runner = EvalRunner(evaluator=StubEvaluator())
        report = runner.run(f)

        assert isinstance(report, EvalReport)
        assert len(report.query_results) == 2
        assert report.aggregate_metrics["hit_rate"] == 1.0
        assert report.aggregate_metrics["mrr"] == 0.5
        # total_elapsed_ms is a real wall-clock delta; on a fast stub it may
        # read 0.0. Assert non-negative (monotonic guarantee) rather than > 0
        # to avoid a flaky timing assertion.
        assert report.total_elapsed_ms >= 0

    def test_run_with_hybrid_search(self, tmp_path: Path) -> None:
        f = tmp_path / "g.json"
        _write_golden_json(f, [
            {"query": "RAG", "expected_chunk_ids": ["c1"]},
        ])

        mock_search = MagicMock()
        mock_search.search.return_value = [
            MagicMock(chunk_id="c1", text="RAG is...", score=0.9),
        ]

        runner = EvalRunner(
            hybrid_search=mock_search,
            evaluator=StubEvaluator(),
        )
        report = runner.run(f)

        assert len(report.query_results) == 1
        assert report.query_results[0].retrieved_chunk_ids == ["c1"]

    def test_run_with_answer_generator(self, tmp_path: Path) -> None:
        f = tmp_path / "g.json"
        _write_golden_json(f, [{"query": "Q"}])

        def gen(query, chunks):
            return f"Generated answer for: {query}"

        runner = EvalRunner(
            evaluator=StubEvaluator(),
            answer_generator=gen,
        )
        report = runner.run(f)

        assert "Generated answer for: Q" == report.query_results[0].generated_answer

    def test_failing_generator_yields_none_not_concat(self, tmp_path: Path) -> None:
        """T20 decision: generation failure → generated_answer None so
        answer-side metrics skip. The chunk-concat fallback would score
        faithfulness ≈ 1.0 (the "answer" IS the context) and fake the gauge."""
        f = tmp_path / "g.json"
        _write_golden_json(f, [{"query": "Q"}])

        def gen(query, chunks):
            raise RuntimeError("ollama down")

        runner = EvalRunner(evaluator=StubEvaluator(), answer_generator=gen)
        report = runner.run(f)

        assert report.query_results[0].generated_answer is None

    def test_no_generator_falls_back_to_concat(self, tmp_path: Path) -> None:
        """Legacy no-generator mode keeps its placeholder concatenation."""
        f = tmp_path / "g.json"
        _write_golden_json(f, [{"query": "Q"}])

        mock_search = MagicMock()
        mock_search.search.return_value = [
            MagicMock(chunk_id="c1", text="chunk text", score=0.9),
        ]

        runner = EvalRunner(
            hybrid_search=mock_search,
            evaluator=StubEvaluator(),
        )
        report = runner.run(f)

        answer = report.query_results[0].generated_answer
        assert answer is not None and "chunk text" in answer

    def test_report_to_dict(self, tmp_path: Path) -> None:
        f = tmp_path / "g.json"
        _write_golden_json(f, [{"query": "Q"}])

        runner = EvalRunner(evaluator=StubEvaluator())
        report = runner.run(f)

        d = report.to_dict()
        assert "aggregate_metrics" in d
        assert "query_results" in d
        assert d["query_count"] == 1
        assert d["evaluator_name"] == "StubEvaluator"


class TestEvalRunnerAggregation:
    """Test metric aggregation logic."""

    def test_aggregate_averages_correctly(self) -> None:
        results = [
            QueryResult(query="q1", metrics={"hit_rate": 1.0, "mrr": 1.0}),
            QueryResult(query="q2", metrics={"hit_rate": 0.0, "mrr": 0.5}),
        ]

        avg = EvalRunner._aggregate_metrics(results)

        assert avg["hit_rate"] == pytest.approx(0.5)
        assert avg["mrr"] == pytest.approx(0.75)

    def test_aggregate_empty_returns_empty(self) -> None:
        assert EvalRunner._aggregate_metrics([]) == {}

    def test_aggregate_partial_metrics(self) -> None:
        """When some queries have metrics that others don't."""
        results = [
            QueryResult(query="q1", metrics={"hit_rate": 1.0}),
            QueryResult(query="q2", metrics={"faithfulness": 0.9}),
        ]

        avg = EvalRunner._aggregate_metrics(results)

        # Each metric averaged over only the queries that produced it
        assert avg["hit_rate"] == 1.0
        assert avg["faithfulness"] == 0.9

    def test_aggregate_filters_nan(self) -> None:
        """A single NaN must not poison the mean — it's excluded, and the
        remaining finite values still average (T10 gauge calibration)."""
        results = [
            QueryResult(query="q1", metrics={"context_recall": 0.5}),
            QueryResult(query="q2", metrics={"context_recall": float("nan")}),
            QueryResult(query="q3", metrics={"context_recall": 1.0}),
        ]

        avg = EvalRunner._aggregate_metrics(results)

        assert avg["context_recall"] == pytest.approx(0.75)

    def test_aggregate_filters_none_and_inf(self) -> None:
        """None / inf values are excluded from the mean as well."""
        results = [
            QueryResult(query="q1", metrics={"mrr": 0.4}),
            QueryResult(query="q2", metrics={"mrr": None}),
            QueryResult(query="q3", metrics={"mrr": float("inf")}),
        ]

        avg = EvalRunner._aggregate_metrics(results)

        assert avg["mrr"] == pytest.approx(0.4)


# ── Tests: Golden test set fixture ────────────────────────────────


class TestGoldenTestSetFixture:
    """Validate the actual golden test set file exists and is valid."""

    def test_golden_set_loads(self) -> None:
        golden_path = Path("tests/fixtures/golden_test_set.json")
        if not golden_path.exists():
            pytest.skip("Golden test set not present")

        cases = load_test_set(golden_path)
        assert len(cases) >= 1
        for tc in cases:
            assert tc.query.strip(), "Query must be non-empty"

    def test_ragas_jsonl_fixture_loads(self) -> None:
        """T20: converted ragas-native fixture mirrors the v4.0 json set."""
        jsonl_path = Path("tests/fixtures/golden_test_set_ragas.jsonl")
        if not jsonl_path.exists():
            pytest.skip("Ragas jsonl fixture not present")

        jsonl_cases = load_test_set(jsonl_path)
        json_cases = load_test_set("tests/fixtures/golden_test_set.json")

        assert [c.query for c in jsonl_cases] == [c.query for c in json_cases]
        assert [c.reference_answer for c in jsonl_cases] == [
            c.reference_answer for c in json_cases
        ]


# ── Tests: EvalRunner source ground truth forwarding ────────────────


class TestEvalRunnerSourceGT:
    """Tests that EvalRunner forwards expected_sources to the evaluator."""

    def test_source_ground_truth_forwarded(self) -> None:
        """EvalRunner should include expected_sources in ground_truth dict."""
        from src.observability.evaluation.eval_runner import (
            EvalRunner, GoldenTestCase,
        )

        # A fake hybrid_search returning one chunk with a known source
        fake_chunk = {"id": "c1", "source": "paperA.pdf", "text": "some text"}
        fake_search = MagicMock()
        fake_search.search.return_value = [fake_chunk]

        evaluator = CustomEvaluator(
            metrics=["source_recall_at_k"], source_top_k=5,
        )
        runner = EvalRunner(
            settings=None, hybrid_search=fake_search, evaluator=evaluator,
        )

        # Write a tiny golden set to a temp file
        import json, tempfile, os
        golden = {
            "test_cases": [
                {
                    "query": "which paper?",
                    "expected_sources": ["paperA.pdf"],
                    "expected_chunk_ids": [],
                }
            ]
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(golden, f)
            tmp_path = f.name

        try:
            report = runner.run(tmp_path, top_k=5)
        finally:
            os.unlink(tmp_path)

        # source_recall should be 1.0 because paperA.pdf was retrieved
        assert report.query_results[0].metrics["source_recall_at_k"] == 1.0


class TestEvalRunnerRerankProtocol:
    """Pin the T15 A/B retrieval protocol: with an enabled reranker,
    _retrieve takes 2x candidates from hybrid search, reranks, then
    truncates back to the requested top_k (context depth stays constant,
    so a rerank on/off comparison isolates rerank quality alone)."""

    def _make_results(self, n: int) -> list[Any]:
        return [
            RetrievalResult(chunk_id=f"c{i}", score=1.0 - i * 0.01, text=f"t{i}", metadata={})
            for i in range(n)
        ]

    def _make_runner(self, reranker: Any) -> EvalRunner:
        mock_search = MagicMock()
        mock_search.search.side_effect = lambda query, top_k: self._make_results(top_k)
        return EvalRunner(hybrid_search=mock_search, evaluator=StubEvaluator(), reranker=reranker)

    def test_enabled_reranker_doubles_pool_then_truncates(self) -> None:
        reranked = self._make_results(3)  # reranker "returns" fewer than top_k
        mock_reranker = MagicMock()
        mock_reranker.is_enabled = True
        mock_reranker.rerank.return_value = MagicMock(results=reranked)

        runner = self._make_runner(mock_reranker)
        out = runner._retrieve("qkd security", 10, None)

        # 2x candidate pool requested from hybrid search
        runner.hybrid_search.search.assert_called_once_with(query="qkd security", top_k=20)
        # rerank applied at the requested depth (10), not rerank.top_k (5)
        _, rerank_kwargs = mock_reranker.rerank.call_args
        assert rerank_kwargs["top_k"] == 10
        assert out == reranked

    def test_disabled_reranker_keeps_plain_depth(self) -> None:
        mock_reranker = MagicMock()
        mock_reranker.is_enabled = False

        runner = self._make_runner(mock_reranker)
        runner._retrieve("qkd security", 10, None)

        runner.hybrid_search.search.assert_called_once_with(query="qkd security", top_k=10)
        mock_reranker.rerank.assert_not_called()

    def test_no_reranker_keeps_plain_depth(self) -> None:
        runner = self._make_runner(None)
        runner._retrieve("qkd security", 10, None)

        runner.hybrid_search.search.assert_called_once_with(query="qkd security", top_k=10)
