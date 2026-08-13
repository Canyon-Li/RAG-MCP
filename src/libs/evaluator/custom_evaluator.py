"""Custom evaluator implementation for lightweight metrics.

This evaluator computes simple, deterministic metrics such as hit rate and MRR.
It is designed for fast regression checks and sanity validation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """Custom evaluator for lightweight metrics (hit_rate, mrr).

    The evaluator expects retrieved chunks to contain an identifier field.
    Supported id fields: id, chunk_id, document_id, doc_id.
    """

    SUPPORTED_METRICS = {"hit_rate", "mrr", "source_recall_at_k", "source_precision_at_k"}
    _ID_FIELDS = ("id", "chunk_id", "document_id", "doc_id")
    _SOURCE_FIELDS = ("source", "source_path")

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        source_top_k: int = 5,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.kwargs = kwargs
        self.source_top_k = source_top_k

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalized = [str(metric).strip().lower() for metric in (metrics or [])]
        if not normalized:
            normalized = ["hit_rate", "mrr"]

        unsupported = [metric for metric in normalized if metric not in self.SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                "Unsupported custom metrics: "
                f"{', '.join(unsupported)}. Supported: {', '.join(sorted(self.SUPPORTED_METRICS))}"
            )

        self.metrics = normalized

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute requested metrics for the given retrieval results.

        Args:
            query: The user query string.
            retrieved_chunks: Retrieved chunks or records.
            generated_answer: Optional generated answer (unused).
            ground_truth: Ground truth ids or structure.
            trace: Optional TraceContext (unused).
            **kwargs: Additional parameters (unused).

        Returns:
            Dictionary of metric name to float value.
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        results: Dict[str, float] = {}

        if "hit_rate" in self.metrics or "mrr" in self.metrics:
            retrieved_ids = self._extract_ids(retrieved_chunks, label="retrieved_chunks")
            ground_truth_ids = self._extract_ground_truth_ids(ground_truth)

            if "hit_rate" in self.metrics:
                results["hit_rate"] = self._compute_hit_rate(retrieved_ids, ground_truth_ids)
            if "mrr" in self.metrics:
                results["mrr"] = self._compute_mrr(retrieved_ids, ground_truth_ids)

        if "source_recall_at_k" in self.metrics or "source_precision_at_k" in self.metrics:
            retrieved_sources = self._extract_sources(retrieved_chunks)
            gt_sources = self._extract_ground_truth_sources(ground_truth)
            top_k_sources = retrieved_sources[: self.source_top_k]

            if "source_recall_at_k" in self.metrics:
                results["source_recall_at_k"] = self._compute_source_recall(
                    top_k_sources, gt_sources
                )
            if "source_precision_at_k" in self.metrics:
                results["source_precision_at_k"] = self._compute_source_precision(
                    top_k_sources, gt_sources
                )

        return results

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        """Extract metrics list from settings if available."""
        if settings is None:
            return []
        metrics = getattr(getattr(settings, "evaluation", None), "metrics", None)
        if metrics is None:
            return []
        return [str(metric) for metric in metrics]

    def _extract_ground_truth_ids(self, ground_truth: Optional[Any]) -> List[str]:
        """Extract ground truth ids from various input shapes."""
        if ground_truth is None:
            return []
        if isinstance(ground_truth, str):
            return [ground_truth]
        if isinstance(ground_truth, dict):
            if "ids" in ground_truth and isinstance(ground_truth["ids"], list):
                return self._extract_ids(ground_truth["ids"], label="ground_truth.ids")
            return self._extract_ids([ground_truth], label="ground_truth")
        if isinstance(ground_truth, list):
            return self._extract_ids(ground_truth, label="ground_truth")

        raise ValueError(
            f"Unsupported ground_truth type: {type(ground_truth).__name__}. "
            "Expected str, dict, list, or None."
        )

    def _extract_ids(self, items: Iterable[Any], label: str) -> List[str]:
        """Extract ids from a list of items."""
        ids: List[str] = []
        for index, item in enumerate(items):
            if isinstance(item, str):
                ids.append(item)
                continue
            if isinstance(item, dict):
                for field in self._ID_FIELDS:
                    if field in item:
                        ids.append(str(item[field]))
                        break
                else:
                    raise ValueError(
                        f"Missing id field in {label}[{index}]. "
                        f"Expected one of {', '.join(self._ID_FIELDS)}"
                    )
                continue
            if hasattr(item, "id"):
                ids.append(str(getattr(item, "id")))
                continue

            raise ValueError(
                f"Unable to extract id from {label}[{index}] of type "
                f"{type(item).__name__}"
            )

        return ids

    def _compute_hit_rate(self, retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
        """Compute hit rate (binary)."""
        if not ground_truth_ids:
            return 0.0
        return 1.0 if any(item in ground_truth_ids for item in retrieved_ids) else 0.0

    def _compute_mrr(self, retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
        """Compute Mean Reciprocal Rank (MRR)."""
        if not ground_truth_ids:
            return 0.0
        for rank, item in enumerate(retrieved_ids, start=1):
            if item in ground_truth_ids:
                return 1.0 / rank
        return 0.0

    def _extract_sources(self, chunks: Iterable[Any]) -> List[str]:
        """Extract source file identifiers from retrieved chunks.

        Reads the ``source`` (or ``source_path``) field from each chunk dict.
        Missing fields yield an empty string so the slot count matches the
        retrieval order.
        """
        sources: List[str] = []
        for item in chunks:
            if isinstance(item, dict):
                value = ""
                for field in self._SOURCE_FIELDS:
                    if field in item and item[field]:
                        value = str(item[field])
                        break
                sources.append(value)
            elif hasattr(item, "source"):
                sources.append(str(getattr(item, "source")))
            else:
                sources.append("")
        return sources

    def _extract_ground_truth_sources(self, ground_truth: Optional[Any]) -> List[str]:
        """Extract expected source identifiers from ground_truth.

        Accepts a dict shaped as ``{"sources": [<file>, ...]}`` or a bare list.
        """
        if ground_truth is None:
            return []
        if isinstance(ground_truth, dict):
            sources = ground_truth.get("sources", [])
            return [str(s) for s in sources] if sources else []
        if isinstance(ground_truth, list):
            return [str(s) for s in ground_truth]
        return []

    @staticmethod
    def _compute_source_recall(
        retrieved_sources: Sequence[str],
        gt_sources: Sequence[str],
    ) -> float:
        """Source Recall@k: 1.0 if any expected source appears in top-k, else 0.0.

        Binary recall — 'did we find the right paper at all'.
        """
        if not gt_sources:
            return 0.0
        gt_set = set(gt_sources)
        return 1.0 if any(s in gt_set for s in retrieved_sources if s) else 0.0

    @staticmethod
    def _compute_source_precision(
        retrieved_sources: Sequence[str],
        gt_sources: Sequence[str],
    ) -> float:
        """Source Precision@k: fraction of top-k chunks whose source is expected."""
        if not retrieved_sources:
            return 0.0
        gt_set = set(gt_sources)
        hits = sum(1 for s in retrieved_sources if s and s in gt_set)
        return hits / len(retrieved_sources)