"""Ragas-based evaluator for RAG quality assessment.

This evaluator wraps the Ragas framework to compute LLM-as-Judge metrics:
- Context Relevance: Are the retrieved contexts relevant to the query?
- Context Precision: Are the retrieved chunks relevant and well-ordered?
- Context Recall: Does the reference answer appear in retrieved contexts?

Design Principles:
- Pluggable: Implements BaseEvaluator interface, swappable via factory.
- Judge-decoupled: The judge LLM is NOT the retrieval pipeline's settings.llm —
  it is configured via env vars (RAGAS_JUDGE_PROVIDER / RAGAS_JUDGE_MODEL /
  OLLAMA_BASE_URL) and defaults to a local Ollama model, so swapping retrieval
  backends never affects the judge.
- Graceful Degradation: Clear ImportError if ragas not installed.
"""

from __future__ import annotations

# ── ragas 0.4.3 import workaround ──────────────────────────────────
# ragas eagerly imports langchain_community.chat_models.vertexai during its
# own import; that package is not installed here. Inject a stub so the import
# succeeds. (Mirrors the agentic-rag-for-dummies evaluation notebook.)
import sys as _sys
import types as _types
try:  # pragma: no cover — only triggers when langchain_community is missing vertexai
    import langchain_community.chat_models.vertexai  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    if "langchain_community.chat_models.vertexai" not in _sys.modules:
        _stub = _types.ModuleType("langchain_community.chat_models.vertexai")

        class _ChatVertexAI:  # minimal stub
            pass

        _stub.ChatVertexAI = _ChatVertexAI  # type: ignore
        _sys.modules["langchain_community.chat_models.vertexai"] = _stub

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Metric name constants
CONTEXT_RELEVANCE = "context_relevance"
CONTEXT_PRECISION = "context_precision"
CONTEXT_RECALL = "context_recall"

SUPPORTED_METRICS = {CONTEXT_RELEVANCE, CONTEXT_PRECISION, CONTEXT_RECALL}


def _import_ragas() -> None:
    """Validate that ragas is importable, raising a clear error if not."""
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'ragas' package is required for RagasEvaluator. "
            "Install it with: pip install ragas datasets"
        ) from exc


class RagasEvaluator(BaseEvaluator):
    """Evaluator that uses the Ragas framework for LLM-as-Judge metrics.

    Ragas does NOT require ground-truth labels.  It uses an LLM to judge
    the quality of the retrieved context against the query.

    Supported metrics:
        - context_relevance: Measures relevance of retrieved contexts to query.
        - context_precision: Measures relevance/ordering of retrieved chunks.

    Example::

        evaluator = RagasEvaluator(settings=settings)
        metrics = evaluator.evaluate(
            query="What is RAG?",
            retrieved_chunks=[{"id": "c1", "text": "RAG is ..."}],
        )
        # metrics == {"context_relevance": 0.95, "context_precision": 0.88, ...}
    """

    # Route-discovery: CompositeEvaluator._backend_supported_metrics() peeks
    # this CLASS attribute (hasattr(cls, "SUPPORTED_METRICS")) to route metrics
    # per backend. Must stay a class attribute — a bare module-level constant
    # alone is invisible to that lookup (CustomEvaluator does the same).
    SUPPORTED_METRICS = {CONTEXT_RELEVANCE, CONTEXT_PRECISION, CONTEXT_RECALL}

    # Judge LLM is decoupled from the retrieval pipeline's settings.llm.
    # Configured via env vars so the judge is stable across provider swaps.
    _JUDGE_DEFAULT_BASE_URL = "http://localhost:11434/v1"
    _DEFAULT_JUDGE_MODEL = "llama3"

    @staticmethod
    def _resolve_judge_provider() -> str:
        """Which provider family to use for the judge LLM.

        Reads RAGAS_JUDGE_PROVIDER (default 'ollama'). Set to 'azure' or
        'openai' to use a cloud judge instead.
        """
        return os.environ.get("RAGAS_JUDGE_PROVIDER", "ollama").lower()

    @classmethod
    def _resolve_judge_model(cls) -> str:
        """Which model to use for the judge LLM.

        Reads RAGAS_JUDGE_MODEL at call time (default llama3). Read at call
        time — not class-definition time — so it can be hot-swapped, matching
        _resolve_judge_provider's pattern.
        """
        return os.environ.get("RAGAS_JUDGE_MODEL", cls._DEFAULT_JUDGE_MODEL)

    def _resolve_ollama_base_url(self) -> str:
        """Ollama endpoint: OLLAMA_BASE_URL env > default, with /v1 suffix."""
        env_url = os.environ.get("OLLAMA_BASE_URL", "").rstrip("/")
        if not env_url:
            return self._JUDGE_DEFAULT_BASE_URL
        if env_url.endswith("/v1"):
            return env_url
        return f"{env_url}/v1"

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RagasEvaluator.

        Args:
            settings: Application settings (used to configure LLM backend).
            metrics: Metric names to compute. Defaults to all supported.
            **kwargs: Additional parameters (reserved).

        Raises:
            ImportError: If ragas is not installed.
            ValueError: If unsupported metric names are requested.
        """
        _import_ragas()

        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalised = [m.strip().lower() for m in (metrics or [])]
        if not normalised:
            normalised = sorted(SUPPORTED_METRICS)

        unsupported = [m for m in normalised if m not in SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                f"Unsupported ragas metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(SUPPORTED_METRICS))}"
            )

        self._metric_names = normalised

    # ── public API ────────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Evaluate RAG quality using Ragas LLM-as-Judge metrics.

        Args:
            query: The user query string.
            retrieved_chunks: Retrieved chunks (dicts with 'text' key or strings).
            generated_answer: Unused — this system does not generate answers.
                Retained for BaseEvaluator interface compatibility.
            ground_truth: Consumed for context_precision — the golden set's
                reference answer is read from ``ground_truth["reference"]``
                (forwarded by EvalRunner). context_relevance needs no ground truth.
            trace: Optional TraceContext for observability.
            **kwargs: Additional parameters.

        Returns:
            Dictionary mapping metric names to float scores (0.0 – 1.0).

        Raises:
            ValueError: If query/chunks are invalid.
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        # This system does not generate answers (retrieval + citation only).
        # context_relevance needs no answer. context_precision needs a reference
        # answer (ContextPrecisionWithReference); it is read from
        # ground_truth["reference"], forwarded by EvalRunner from the golden
        # set's reference_answer. Empty generated_answer is allowed.

        contexts = self._extract_texts(retrieved_chunks)
        reference = self._extract_reference(ground_truth)

        try:
            result = self._run_ragas(query, contexts, reference)
        except Exception as exc:
            logger.error("Ragas evaluation failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Ragas evaluation failed: {exc}") from exc

        return result

    # ── private helpers ───────────────────────────────────────────

    def _run_ragas(
        self,
        query: str,
        contexts: List[str],
        reference: Optional[str],
    ) -> Dict[str, float]:
        """Execute Ragas collections metrics and return normalised scores.

        Ragas 0.4+ collections metrics use per-metric ``score()``.  Signatures:
        - ContextRelevance: (user_input, retrieved_contexts)
        - ContextPrecision (= ContextPrecisionWithReference):
          (user_input, retrieved_contexts, reference)
        - ContextRecall: (user_input, retrieved_contexts, reference)
        """
        from ragas.metrics.collections import (
            ContextRelevance,
            ContextPrecision,
            ContextRecall,
        )

        # Build the judge LLM wrapper from env-driven provider config
        llm = self._build_wrappers()

        scores: Dict[str, float] = {}

        for metric_name in self._metric_names:
            if metric_name == CONTEXT_RELEVANCE:
                m = ContextRelevance(llm=llm)
                result = m.score(
                    user_input=query, retrieved_contexts=contexts,
                )
            elif metric_name == CONTEXT_PRECISION:
                m = ContextPrecision(llm=llm)
                # ContextPrecisionWithReference ranks retrieved contexts by
                # whether each is needed to answer the query, judged against the
                # reference answer. Requires a non-empty reference.
                if not reference:
                    logger.warning(
                        "context_precision skipped: no reference answer provided "
                        "(golden set missing 'reference')."
                    )
                    scores[metric_name] = 0.0
                    continue
                result = m.score(
                    user_input=query,
                    retrieved_contexts=contexts,
                    reference=reference,
                )
            elif metric_name == CONTEXT_RECALL:
                m = ContextRecall(llm=llm)
                # ContextRecall decomposes the reference into atomic claims
                # and scores the fraction supported by the retrieved contexts.
                # Symmetric with precision: missing reference → warn + 0.0.
                if not reference:
                    logger.warning(
                        "context_recall skipped: no reference answer provided "
                        "(golden set missing 'reference')."
                    )
                    scores[metric_name] = 0.0
                    continue
                result = m.score(
                    user_input=query,
                    retrieved_contexts=contexts,
                    reference=reference,
                )
            else:
                continue

            scores[metric_name] = float(result.value) if result.value is not None else 0.0

        return scores

    @staticmethod
    def _extract_reference(ground_truth: Optional[Any]) -> Optional[str]:
        """Pull the reference answer out of ground_truth.

        EvalRunner forwards the golden set's reference_answer as
        ``ground_truth["reference"]``. Returns None when absent.
        """
        if ground_truth is None:
            return None
        if isinstance(ground_truth, dict):
            ref = ground_truth.get("reference")
            return str(ref) if ref else None
        return None

    def _build_wrappers(self) -> Any:
        """Build the Ragas judge LLM wrapper.

        Judge provider is env-driven (RAGAS_JUDGE_PROVIDER: ollama | deepseek
        | azure | openai), decoupled from settings.llm so swapping the
        retrieval LLM never affects the judge.

        The three active metrics (relevance / precision / recall) consume no
        embeddings — the historical embeddings wrapper was dead code and has
        been removed (spec 2026-08-16 §2).
        """
        from openai import AsyncAzureOpenAI, AsyncOpenAI
        from ragas.llms import llm_factory

        if self.settings is None:
            raise ValueError("Settings required to create LLM for Ragas evaluation")

        judge_provider = self._resolve_judge_provider()
        if judge_provider == "ollama":
            base_url = self._resolve_ollama_base_url()
            # trust_env=False bypasses the system proxy intercepting localhost
            # traffic (D-014). AsyncOpenAI reads proxy env vars by default; an
            # explicit httpx client is the robust way to disable that.
            import httpx
            http_client = httpx.AsyncClient(trust_env=False, timeout=300.0)
            client = AsyncOpenAI(
                base_url=base_url, api_key="ollama", http_client=http_client,
            )
            llm = llm_factory(self._resolve_judge_model(), client=client, max_tokens=8192)
            return llm

        if judge_provider == "deepseek":
            # Cloud judge via DeepSeek's OpenAI-compatible endpoint.
            # NOTE: unlike the ollama branch, trust_env is deliberately NOT
            # disabled here — deepseek is a REMOTE endpoint, so going through
            # the system proxy is the correct path (D-014 only applies to
            # localhost traffic).
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError(
                    "RAGAS_JUDGE_PROVIDER=deepseek but DEEPSEEK_API_KEY "
                    "is not set"
                )
            base_url = os.environ.get(
                "RAGAS_JUDGE_BASE_URL", "https://api.deepseek.com"
            )
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            llm = llm_factory(
                self._resolve_judge_model(), client=client, max_tokens=8192,
            )
            return llm

        # Fallback: original cloud-judge logic (azure/openai) below

        # ── LLM ──
        llm_cfg = self.settings.llm
        provider = llm_cfg.provider.lower()
        llm_azure_endpoint = getattr(llm_cfg, "azure_endpoint", None)

        # Azure-compatible mode: if azure_endpoint is configured, use Azure
        # client even when provider is "openai" (matches project convention).
        use_azure_llm = (
            provider == "azure"
            or (provider == "openai" and llm_azure_endpoint)
        )

        if use_azure_llm:
            llm_client = AsyncAzureOpenAI(
                api_key=llm_cfg.api_key,
                azure_endpoint=llm_azure_endpoint or llm_cfg.azure_endpoint,
                api_version=getattr(llm_cfg, "api_version", None) or "2024-02-15-preview",
            )
        elif provider == "openai":
            llm_client = AsyncOpenAI(api_key=llm_cfg.api_key)
        else:
            raise ValueError(
                f"Unsupported LLM provider for Ragas: '{provider}'. "
                "Supported: azure, openai"
            )

        llm = llm_factory(llm_cfg.model, client=llm_client, max_tokens=8192)

        return llm

    def _extract_texts(self, chunks: List[Any]) -> List[str]:
        """Extract text strings from various chunk representations.

        Args:
            chunks: List of chunk dicts, strings, or objects with .text.

        Returns:
            List of text strings.
        """
        texts: List[str] = []
        for chunk in chunks:
            if isinstance(chunk, str):
                texts.append(chunk)
            elif isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content") or chunk.get("page_content", "")
                texts.append(str(text))
            elif hasattr(chunk, "text"):
                texts.append(str(getattr(chunk, "text")))
            else:
                texts.append(str(chunk))
        return texts

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        """Extract metrics list from settings if available."""
        if settings is None:
            return []
        evaluation = getattr(settings, "evaluation", None)
        if evaluation is None:
            return []
        raw_metrics = getattr(evaluation, "metrics", None)
        if raw_metrics is None:
            return []
        # Filter to only ragas-supported metrics
        return [m for m in raw_metrics if m.lower() in SUPPORTED_METRICS]
