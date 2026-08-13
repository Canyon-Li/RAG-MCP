"""Composite evaluator that combines multiple evaluators.

This evaluator implements the Composite pattern: it holds a list of
BaseEvaluator instances, runs them all, and merges their metric
dictionaries into a single result.

Design Principles:
- Pluggable: Any BaseEvaluator can be composed.
- Config-Driven: `evaluation.backends: [ragas, custom]` drives composition.
- Observable: Logs individual evaluator successes/failures.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class CompositeEvaluator(BaseEvaluator):
    """Evaluator that composes multiple evaluators and merges metrics.

    Each sub-evaluator is invoked with the same arguments.  Results are
    merged into a single metrics dictionary.  If two evaluators produce
    the same metric key, the later one's value wins (with a warning).

    Partial failure is tolerated: if one sub-evaluator fails, its error
    is logged and the rest still execute.

    Example::

        composite = CompositeEvaluator(evaluators=[
            CustomEvaluator(metrics=["source_recall_at_k", "source_precision_at_k"]),
            RagasEvaluator(metrics=["context_relevance", "context_precision"]),
        ])
        metrics = composite.evaluate(
            query="test", retrieved_chunks=[...],
            ground_truth={"sources": ["paper.pdf"], "reference": "..."},
        )
        # metrics == {"source_recall_at_k": 1.0, "source_precision_at_k": 0.5,
        #             "context_relevance": 0.95, "context_precision": 0.97}
    """

    def __init__(
        self,
        evaluators: Optional[Sequence[BaseEvaluator]] = None,
        settings: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CompositeEvaluator.

        Args:
            evaluators: Pre-built evaluator instances. If None, built from settings.
            settings: Application settings (used for config-driven composition).
            **kwargs: Additional parameters forwarded to sub-evaluators.

        Raises:
            ValueError: If no evaluators are provided and settings don't
                specify backends.
        """
        self.settings = settings
        self.kwargs = kwargs

        if evaluators is not None:
            self._evaluators: List[BaseEvaluator] = list(evaluators)
        else:
            self._evaluators = self._build_from_settings(settings, **kwargs)

        if not self._evaluators:
            raise ValueError(
                "CompositeEvaluator requires at least one sub-evaluator. "
                "Provide evaluators directly or configure "
                "'evaluation.backends' in settings.yaml."
            )

        logger.info(
            "CompositeEvaluator initialised with %d evaluator(s): %s",
            len(self._evaluators),
            [type(e).__name__ for e in self._evaluators],
        )

    @property
    def evaluators(self) -> List[BaseEvaluator]:
        """Return the list of composed evaluators."""
        return list(self._evaluators)

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Run all sub-evaluators and merge their metrics.

        Args:
            query: The user query string.
            retrieved_chunks: Retrieved chunks or records.
            generated_answer: Optional generated answer text.
            ground_truth: Optional ground truth data.
            trace: Optional TraceContext for observability.
            **kwargs: Additional parameters.

        Returns:
            Merged dictionary of all metric names to float values.

        Raises:
            RuntimeError: If ALL sub-evaluators fail.
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        merged: Dict[str, float] = {}
        errors: List[str] = []

        for evaluator in self._evaluators:
            name = type(evaluator).__name__
            try:
                metrics = evaluator.evaluate(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    generated_answer=generated_answer,
                    ground_truth=ground_truth,
                    trace=trace,
                    **kwargs,
                )
                for key, value in metrics.items():
                    if key in merged:
                        logger.warning(
                            "Metric '%s' produced by multiple evaluators; "
                            "overwriting with value from %s",
                            key,
                            name,
                        )
                    merged[key] = value

                logger.debug(
                    "%s produced %d metric(s): %s",
                    name,
                    len(metrics),
                    list(metrics.keys()),
                )

            except Exception as exc:
                msg = f"{name} failed: {exc}"
                logger.warning(msg)
                errors.append(msg)

        if not merged and errors:
            raise RuntimeError(
                "All sub-evaluators failed:\n" + "\n".join(errors)
            )

        return merged

    # ── config-driven builder ────────────────────────────────────

    @staticmethod
    def _build_from_settings(
        settings: Any,
        **kwargs: Any,
    ) -> List[BaseEvaluator]:
        """Build sub-evaluators from settings.evaluation.backends.

        Expected config::

            evaluation:
              enabled: true
              provider: composite
              backends:
                - custom
                - ragas
              metrics:
                - source_recall_at_k
                - source_precision_at_k
                - context_relevance
                - context_precision

        Args:
            settings: Application settings.
            **kwargs: Forwarded to each sub-evaluator constructor.

        Returns:
            List of BaseEvaluator instances.
        """
        if settings is None:
            return []

        evaluation = getattr(settings, "evaluation", None)
        if evaluation is None:
            return []

        backends = getattr(evaluation, "backends", None)
        if not backends:
            return []

        from src.libs.evaluator.evaluator_factory import EvaluatorFactory

        all_metrics = list(getattr(evaluation, "metrics", []))

        evaluators: List[BaseEvaluator] = []
        for backend_name in backends:
            backend_name = str(backend_name).strip().lower()
            if backend_name in {"composite", "none", "disabled"}:
                continue  # avoid infinite recursion / no-ops

            try:
                # Route only the metrics this backend supports. Without this,
                # the full metrics list is passed to every sub-evaluator and the
                # ones whose SUPPORTED_METRICS doesn't include a foreign metric
                # reject construction (e.g. CustomEvaluator rejects ragas metrics).
                supported = CompositeEvaluator._backend_supported_metrics(backend_name)
                routed = [m for m in all_metrics if m in supported] if supported else all_metrics

                # Create a mock settings with provider overridden
                from unittest.mock import MagicMock

                sub_settings = MagicMock(wraps=settings)
                sub_eval = MagicMock()
                sub_eval.enabled = True
                sub_eval.provider = backend_name
                sub_eval.metrics = routed
                sub_eval.backends = []  # prevent recursion
                sub_settings.evaluation = sub_eval

                evaluator = EvaluatorFactory.create(sub_settings, **kwargs)
                evaluators.append(evaluator)
                logger.info("CompositeEvaluator: loaded backend '%s'", backend_name)
            except Exception as exc:
                logger.warning(
                    "CompositeEvaluator: failed to load backend '%s': %s",
                    backend_name,
                    exc,
                )

        return evaluators

    @staticmethod
    def _backend_supported_metrics(backend_name: str) -> set:
        """Resolve a backend provider's SUPPORTED_METRICS without instantiating it.

        Returns an empty set if the class can't be peeked (e.g. lazy-loaded and
        its dependency is missing) — callers then pass the full metric list and
        let the backend's own validation raise a clear error.
        """
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory

        cls = EvaluatorFactory._PROVIDERS.get(backend_name)
        if cls is None and backend_name in EvaluatorFactory._LAZY_PROVIDERS:
            try:
                cls = EvaluatorFactory._LAZY_PROVIDERS[backend_name]()
            except ImportError:
                return set()
        if cls is not None and hasattr(cls, "SUPPORTED_METRICS"):
            return set(cls.SUPPORTED_METRICS)
        return set()
