"""Evaluation runner for batch quality assessment.

EvalRunner reads a golden test set, runs HybridSearch for each test case,
optionally generates answers, then invokes the configured Evaluator(s) to
produce a structured evaluation report.

Design Principles:
- Config-Driven: Evaluator selected via settings.yaml.
- Observable: Produces EvalReport with per-query details.
- Decoupled: Works with any BaseEvaluator implementation.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


@dataclass
class GoldenTestCase:
    """A single evaluation test case from the golden test set.

    Attributes:
        query: The test query string.
        expected_chunk_ids: Ground-truth chunk IDs for IR metrics.
        expected_sources: Ground-truth source file names (optional).
        reference_answer: Reference answer text for LLM-as-Judge (optional).
    """

    query: str
    expected_chunk_ids: List[str] = field(default_factory=list)
    expected_sources: List[str] = field(default_factory=list)
    reference_answer: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GoldenTestCase:
        return cls(
            query=data["query"],
            expected_chunk_ids=data.get("expected_chunk_ids", []),
            expected_sources=data.get("expected_sources", []),
            reference_answer=data.get("reference", data.get("reference_answer")),
        )


@dataclass
class QueryResult:
    """Result of evaluating a single test case.

    Attributes:
        query: The test query.
        retrieved_chunk_ids: IDs of chunks actually retrieved.
        generated_answer: The generated answer (if applicable).
        metrics: Evaluation metrics for this query.
        elapsed_ms: Time taken for retrieval + evaluation.
    """

    query: str
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    generated_answer: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class EvalReport:
    """Aggregated evaluation report across all test cases.

    Attributes:
        query_results: Per-query evaluation results.
        aggregate_metrics: Averaged metrics across all queries.
        total_elapsed_ms: Total time for the entire evaluation.
        evaluator_name: Name of the evaluator used.
        test_set_path: Path to the golden test set file.
    """

    query_results: List[QueryResult] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    total_elapsed_ms: float = 0.0
    evaluator_name: str = ""
    test_set_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise report to dictionary."""
        return {
            "evaluator_name": self.evaluator_name,
            "test_set_path": self.test_set_path,
            "total_elapsed_ms": round(self.total_elapsed_ms, 1),
            "aggregate_metrics": {
                k: round(v, 4) for k, v in self.aggregate_metrics.items()
            },
            "query_count": len(self.query_results),
            "query_results": [
                {
                    "query": qr.query,
                    "retrieved_chunk_ids": qr.retrieved_chunk_ids,
                    "generated_answer": qr.generated_answer,
                    "metrics": {k: round(v, 4) for k, v in qr.metrics.items()},
                    "elapsed_ms": round(qr.elapsed_ms, 1),
                }
                for qr in self.query_results
            ],
        }


def load_test_set(path: str | Path) -> List[GoldenTestCase]:
    """Load golden test set from a JSON or ragas-native jsonl file.

    Three dialects, selected by file suffix + top-level shape:

    - ``.jsonl`` — ragas ``SingleTurnSample`` per line (what
      ``EvaluationDataset.to_jsonl`` writes; verified round-trip against
      ragas 0.4.3). Parsed with stdlib json — importing ragas just to read
      two fields would drag its vertexai import-stub into every consumer.
    - ``.json`` top-level array — the v5.0 exam (T21): an indented JSON
      array of ragas ``SingleTurnSample`` dicts written by the synthesis
      finaliser. Same per-sample mapping as the jsonl dialect.
    - ``.json`` top-level dict — the v1/v4.0 house format
      (``{"test_cases": [...]}``).

    Args:
        path: Path to the golden test set file.

    Returns:
        List of TestCase instances.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Golden test set not found: {file_path}")

    if file_path.suffix.lower() == ".jsonl":
        return _load_test_set_jsonl(file_path)

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return _load_test_set_json_array(file_path, data)

    if "test_cases" not in data:
        raise ValueError(
            "Invalid golden test set format: missing 'test_cases' key."
        )

    return [GoldenTestCase.from_dict(tc) for tc in data["test_cases"]]


def _load_test_set_jsonl(file_path: Path) -> list[GoldenTestCase]:
    """Load a ragas-native jsonl test set (SingleTurnSample per line).

    Field mapping: ``user_input`` → query, ``reference`` → reference_answer.
    Everything else (retrieved_contexts / response / reference_contexts …)
    belongs to the ragas scoring pipeline, not this runner, and is ignored.
    """
    cases: list[GoldenTestCase] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                sample = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid jsonl on line {line_no} of {file_path}: {exc}"
                ) from exc
            if not isinstance(sample, dict) or not sample.get("user_input"):
                raise ValueError(
                    f"Invalid jsonl sample at line {line_no} of {file_path}: "
                    "missing 'user_input'"
                )
            cases.append(
                GoldenTestCase(
                    query=sample["user_input"],
                    reference_answer=sample.get("reference") or None,
                )
            )
    return cases


def _load_test_set_json_array(
    file_path: Path, data: list[Any],
) -> list[GoldenTestCase]:
    """Load a ragas SingleTurnSample JSON array (the v5.0 exam, T21).

    Same field mapping as the jsonl dialect: ``user_input`` → query,
    ``reference`` → reference_answer; everything else belongs to the
    ragas scoring pipeline and is ignored.
    """
    cases: list[GoldenTestCase] = []
    for idx, sample in enumerate(data):
        if not isinstance(sample, dict) or not sample.get("user_input"):
            raise ValueError(
                f"Invalid sample at index {idx} of {file_path}: "
                "missing 'user_input'"
            )
        cases.append(
            GoldenTestCase(
                query=sample["user_input"],
                reference_answer=sample.get("reference") or None,
            )
        )
    return cases


class EvalRunner:
    """Runs batch evaluation against a golden test set.

    This class orchestrates:
    1. Loading the golden test set
    2. Running HybridSearch for each query
    3. Optionally generating answers
    4. Invoking the evaluator to score each result
    5. Aggregating metrics into an EvalReport

    Example::

        runner = EvalRunner(
            settings=settings,
            hybrid_search=hybrid_search,
            evaluator=evaluator,
        )
        report = runner.run("tests/fixtures/golden_test_set.json")
        print(report.aggregate_metrics)
    """

    def __init__(
        self,
        settings: Any = None,
        hybrid_search: Any = None,
        evaluator: Optional[BaseEvaluator] = None,
        answer_generator: Any = None,
        answer_overrides: Optional[Dict[int, str]] = None,
        reranker: Any = None,
    ) -> None:
        """Initialize EvalRunner.

        Args:
            settings: Application settings.
            hybrid_search: HybridSearch instance for retrieval.
            evaluator: BaseEvaluator instance for scoring.
            answer_generator: Optional callable(query, chunks) -> str
                for generating answers. If None, a simple concatenation
                is used as a placeholder.
            answer_overrides: Optional dict mapping test case index (0-based)
                to a user-provided answer string. When present, the override
                answer is used instead of auto-generation for that test case.
            reranker: Optional CoreReranker instance for reranking results.
        """
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.evaluator = evaluator
        self.answer_generator = answer_generator
        self.answer_overrides = answer_overrides or {}
        self.reranker = reranker

    def run(
        self,
        test_set_path: str | Path,
        top_k: int = 10,
        collection: Optional[str] = None,
    ) -> EvalReport:
        """Run evaluation on the golden test set.

        Args:
            test_set_path: Path to golden_test_set.json.
            top_k: Number of chunks to retrieve per query.
            collection: Optional collection name filter.

        Returns:
            EvalReport with per-query and aggregate metrics.

        Raises:
            FileNotFoundError: If test set file doesn't exist.
            ValueError: If evaluator or hybrid_search is not set.
        """
        if self.evaluator is None:
            raise ValueError("EvalRunner requires an evaluator.")

        test_cases = load_test_set(test_set_path)
        if not test_cases:
            raise ValueError("Golden test set is empty.")

        logger.info(
            "Starting evaluation: %d test cases, evaluator=%s",
            len(test_cases),
            type(self.evaluator).__name__,
        )

        report = EvalReport(
            evaluator_name=type(self.evaluator).__name__,
            test_set_path=str(test_set_path),
        )

        t0 = time.monotonic()

        for idx, tc in enumerate(test_cases):
            logger.info("Evaluating [%d/%d]: %s", idx + 1, len(test_cases), tc.query[:60])
            # Use user-provided answer override if available for this index
            answer_override = self.answer_overrides.get(idx)
            qr = self._evaluate_single(
                tc, top_k=top_k, collection=collection,
                answer_override=answer_override,
            )
            report.query_results.append(qr)

        report.total_elapsed_ms = (time.monotonic() - t0) * 1000.0
        report.aggregate_metrics = self._aggregate_metrics(report.query_results)

        logger.info(
            "Evaluation complete: %d queries, aggregate=%s",
            len(report.query_results),
            report.aggregate_metrics,
        )

        return report

    def _evaluate_single(
        self,
        test_case: GoldenTestCase,
        top_k: int = 10,
        collection: Optional[str] = None,
        answer_override: Optional[str] = None,
    ) -> QueryResult:
        """Evaluate a single test case.

        Args:
            test_case: The test case to evaluate.
            top_k: Number of results to retrieve.
            collection: Optional collection filter.
            answer_override: User-provided answer text. When set, used
                instead of auto-generated answer from chunks.

        Returns:
            QueryResult with metrics for this test case.
        """
        t0 = time.monotonic()
        qr = QueryResult(query=test_case.query)

        # Step 1: Retrieve chunks
        retrieved_chunks = self._retrieve(test_case.query, top_k, collection)
        qr.retrieved_chunk_ids = [
            self._get_chunk_id(c) for c in retrieved_chunks
        ]

        # Step 2: Generate answer — user override wins; else generate (None on
        # failure → answer-side metrics skip, T20 policy).
        answer: str | None = answer_override or None
        if answer is None:
            answer = self._generate_answer(test_case.query, retrieved_chunks)
        qr.generated_answer = answer

        # Step 3: Build ground truth — always a dict so source-level metrics
        # (which read ground_truth["sources"]) work even when chunk ids absent.
        # reference_answer is forwarded as ground_truth["reference"] for RAGAS
        # context_precision (ContextPrecisionWithReference needs 'reference',
        # not the generated answer — this system doesn't generate answers).
        ground_truth: Dict[str, Any] = {}
        if test_case.expected_chunk_ids:
            ground_truth["ids"] = test_case.expected_chunk_ids
        if test_case.expected_sources:
            ground_truth["sources"] = test_case.expected_sources
        if test_case.reference_answer:
            ground_truth["reference"] = test_case.reference_answer
        if not ground_truth:
            ground_truth = None

        # Step 4: Evaluate
        try:
            metrics = self.evaluator.evaluate(  # type: ignore[union-attr]
                query=test_case.query,
                retrieved_chunks=retrieved_chunks,
                generated_answer=answer,
                ground_truth=ground_truth,
            )
            qr.metrics = metrics
        except Exception as exc:
            logger.warning("Evaluation failed for '%s': %s", test_case.query[:40], exc)
            qr.metrics = {}

        qr.elapsed_ms = (time.monotonic() - t0) * 1000.0
        return qr

    def _retrieve(
        self,
        query: str,
        top_k: int,
        collection: Optional[str],
    ) -> List[Any]:
        """Retrieve chunks using HybridSearch + optional Reranking.

        Falls back to an empty list if search is not configured.
        """
        if self.hybrid_search is None:
            logger.warning("No HybridSearch configured; returning empty results.")
            return []

        try:
            # Retrieve more candidates if reranker is enabled
            has_reranker = self.reranker is not None and getattr(self.reranker, 'is_enabled', False)
            initial_top_k = top_k * 2 if has_reranker else top_k

            results = self.hybrid_search.search(
                query=query,
                top_k=initial_top_k,
            )
            results = results if isinstance(results, list) else results.results

            # Apply reranking if enabled
            if has_reranker and results:
                rerank_result = self.reranker.rerank(query=query, results=results, top_k=top_k)
                results = rerank_result.results

            return results
        except Exception as exc:
            logger.warning("Retrieval failed for '%s': %s", query[:40], exc)
            return []

    def _generate_answer(self, query: str, chunks: List[Any]) -> str | None:
        """Generate an answer from retrieved chunks.

        With a generator configured (T20 era): its return value wins —
        including None on failure, so answer-side metrics skip instead of
        scoring garbage. No concat fallback here: a concatenated pseudo-answer
        would score faithfulness ≈ 1.0 (the "answer" IS the context) and fake
        the gauge.

        Without a generator (legacy retrieval-only runs): concatenate chunk
        texts as a placeholder.
        """
        if self.answer_generator is not None:
            try:
                return self.answer_generator(query, chunks)
            except Exception as exc:
                logger.warning(
                    "Answer generation failed: %s — answer-side metrics "
                    "skipped for this query.", exc,
                )
                return None

        # Legacy fallback: concatenate chunk texts
        texts = []
        for c in chunks:
            if isinstance(c, str):
                texts.append(c)
            elif isinstance(c, dict):
                texts.append(c.get("text", str(c)))
            elif hasattr(c, "text"):
                texts.append(str(getattr(c, "text")))
            else:
                texts.append(str(c))

        return " ".join(texts[:5])  # first 5 chunks

    def _get_chunk_id(self, chunk: Any) -> str:
        """Extract chunk ID from various representations."""
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict):
            for key in ("id", "chunk_id"):
                if key in chunk:
                    return str(chunk[key])
            return str(chunk)
        if hasattr(chunk, "chunk_id"):
            return str(getattr(chunk, "chunk_id"))
        if hasattr(chunk, "id"):
            return str(getattr(chunk, "id"))
        return str(chunk)

    @staticmethod
    def _aggregate_metrics(results: List[QueryResult]) -> Dict[str, float]:
        """Compute average metrics across all query results.

        Args:
            results: List of QueryResult with per-query metrics.

        Returns:
            Dictionary of average metric values.
        """
        if not results:
            return {}

        # Collect all metric keys
        all_keys: set[str] = set()
        for qr in results:
            all_keys.update(qr.metrics.keys())

        # Average each metric. Non-finite values (NaN from a judge failure,
        # inf) are filtered before the mean — a single NaN would otherwise
        # poison the entire aggregated metric via sum/len.
        averages: Dict[str, float] = {}
        for key in sorted(all_keys):
            values = [
                v for v in (qr.metrics.get(key) for qr in results)
                if v is not None and math.isfinite(v)
            ]
            averages[key] = sum(values) / len(values) if values else 0.0

        return averages
