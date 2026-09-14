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

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.observability.evaluation.ragas_compat import (
    build_deepseek_client,
    install_vertexai_stub,
)

# ── ragas 0.4.3 import workaround ──────────────────────────────────
# Stub installed at module import so ragas's lazy imports (inside
# _run_ragas) never see the missing vertexai package. The stub itself
# lives in ragas_compat, shared with the T21 synthesis script.
install_vertexai_stub()

logger = logging.getLogger(__name__)

# Metric name constants
CONTEXT_RELEVANCE = "context_relevance"
CONTEXT_PRECISION = "context_precision"
CONTEXT_RECALL = "context_recall"
FAITHFULNESS = "faithfulness"
ANSWER_RELEVANCY = "answer_relevancy"


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
    # context_relevance is v4.0-era: compute path kept (historical runs stay
    # reproducible) but dropped from default metric lists (T20).
    SUPPORTED_METRICS = {
        CONTEXT_RELEVANCE,
        CONTEXT_PRECISION,
        CONTEXT_RECALL,
        FAITHFULNESS,
        ANSWER_RELEVANCY,
    }

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

    # T11: judge disk cache lives under gitignored data/ — NOT ragas's
    # default .cache/ (that would dirty the repo root).
    _DEFAULT_JUDGE_CACHE_DIR = (
        Path(__file__).resolve().parents[3] / "data" / "eval_judge_cache"
    )

    @classmethod
    def _resolve_judge_cache(cls) -> Any | None:
        """Disk cache for judge LLM calls (exact-match replay, T11).

        Re-running an eval with unchanged (query, contexts, reference)
        replays the previous verdict from disk — zero API cost, zero
        re-roll noise; only questions whose retrieval actually changed get
        re-judged. Turn OFF for fresh-sample runs (the final gate's 3
        independent runs must replay nothing): RAGAS_JUDGE_CACHE=0 or
        scripts/evaluate.py --no-judge-cache.

        Boundaries: the cache removes re-sampling variance, NOT judge
        bias; the cache key includes model params (temperature /
        max_tokens), so changing judge params auto-invalidates; failed
        judge calls (e.g. max_tokens overflow drops) are not cached.
        """
        if os.environ.get("RAGAS_JUDGE_CACHE", "1").strip().lower() in {
            "0", "false", "off", "no",
        }:
            return None
        try:
            from ragas.cache import DiskCacheBackend
        except ImportError:  # pragma: no cover — diskcache ships with ragas
            logger.warning(
                "ragas.cache.DiskCacheBackend unavailable; "
                "judge calls run uncached."
            )
            return None
        cache_dir = os.environ.get("RAGAS_JUDGE_CACHE_DIR") or str(
            cls._DEFAULT_JUDGE_CACHE_DIR
        )
        return DiskCacheBackend(cache_dir=cache_dir)

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
            generated_answer: Consumed by faithfulness / answer_relevancy
                (T20 generation era). Missing/empty answer → those two
                metrics are skipped with a warning, never a pseudo-0.0.
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

        # context_* metrics need no answer (precision/recall need a reference
        # answer via ground_truth["reference"], forwarded by EvalRunner).
        # faithfulness / answer_relevancy need the GENERATED answer; an empty
        # one skips them rather than scoring garbage.

        contexts = self._extract_texts(retrieved_chunks)
        reference = self._extract_reference(ground_truth)

        try:
            result = self._run_ragas(
                query, contexts, reference, response=generated_answer,
            )
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
        response: str | None = None,
    ) -> Dict[str, float]:
        """Execute Ragas collections metrics and return normalised scores.

        Ragas 0.4+ collections metrics use per-metric ``score()``.  Signatures:
        - ContextRelevance: (user_input, retrieved_contexts)
        - ContextPrecision (= ContextPrecisionWithReference):
          (user_input, retrieved_contexts, reference)
        - ContextRecall: (user_input, retrieved_contexts, reference)
        - Faithfulness: (user_input, response, retrieved_contexts) — T20
        - AnswerRelevancy: (user_input, response) — T20, needs embeddings
        """
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextRelevance,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        # Build the judge LLM wrapper from env-driven provider config
        llm = self._build_wrappers()

        # Judge embeddings: only AnswerRelevancy consumes them — build lazily
        # so retrieval-only metric mixes never require a local embedding model.
        emb: Any = None

        scores: Dict[str, float] = {}

        for metric_name in self._metric_names:
            # Per-metric isolation: a transport failure on ONE judge call
            # (e.g. deepseek connect timeout under the 4-metric volume,
            # first T20 run: 13/23 questions zeroed by single timeouts)
            # excludes only that metric — never the whole question.
            try:
                if metric_name == CONTEXT_RELEVANCE:
                    result = ContextRelevance(llm=llm).score(
                        user_input=query, retrieved_contexts=contexts,
                    )
                elif metric_name == CONTEXT_PRECISION:
                    # ContextPrecisionWithReference ranks retrieved contexts
                    # by whether each is needed to answer the query, judged
                    # against the reference answer. Missing reference → skip
                    # entirely (absent key) — a recorded 0.0 is a pseudo-zero
                    # that drags the aggregate mean down.
                    if not reference:
                        logger.warning(
                            "context_precision skipped: no reference answer "
                            "provided (golden set missing 'reference')."
                        )
                        continue
                    result = ContextPrecision(llm=llm).score(
                        user_input=query,
                        retrieved_contexts=contexts,
                        reference=reference,
                    )
                elif metric_name == CONTEXT_RECALL:
                    # ContextRecall decomposes the reference into atomic
                    # claims and scores the fraction supported by the
                    # retrieved contexts. Symmetric with precision.
                    if not reference:
                        logger.warning(
                            "context_recall skipped: no reference answer "
                            "provided (golden set missing 'reference')."
                        )
                        continue
                    result = ContextRecall(llm=llm).score(
                        user_input=query,
                        retrieved_contexts=contexts,
                        reference=reference,
                    )
                elif metric_name == FAITHFULNESS:
                    # Faithfulness decomposes the GENERATED response into
                    # atomic statements and NLI-checks each against the
                    # retrieved contexts. Missing response → skip (T10 rule).
                    if not response:
                        logger.warning(
                            "faithfulness skipped: no generated answer provided."
                        )
                        continue
                    result = Faithfulness(llm=llm).score(
                        user_input=query,
                        response=response,
                        retrieved_contexts=contexts,
                    )
                elif metric_name == ANSWER_RELEVANCY:
                    # AnswerRelevancy reverse-generates questions from the
                    # response and cosine-compares them to the original
                    # question — hence embeddings. Missing response → skip.
                    if not response:
                        logger.warning(
                            "answer_relevancy skipped: no generated answer "
                            "provided."
                        )
                        continue
                    if emb is None:
                        emb = self._build_judge_embeddings()
                    result = AnswerRelevancy(llm=llm, embeddings=emb).score(
                        user_input=query,
                        response=response,
                    )
                else:
                    continue

                # result.value can be None (metric produced nothing) or NaN/inf
                # (e.g. 0/0 inside ragas when no statements parse). Either way
                # the metric is excluded (absent key), never recorded as 0.0 —
                # one NaN through sum/len would poison the whole aggregate.
                raw = result.value
                if raw is None:
                    logger.warning(
                        "%s produced no value; excluded from metrics.",
                        metric_name,
                    )
                    continue
                value = float(raw)
                if not math.isfinite(value):
                    logger.warning(
                        "%s produced non-finite value %s; excluded.",
                        metric_name, raw,
                    )
                    continue
            except Exception as exc:
                logger.warning(
                    "%s failed (%s: %s); excluded from metrics.",
                    metric_name, type(exc).__name__, exc,
                )
                continue
            scores[metric_name] = value

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

        # T11: judge disk cache — see _resolve_judge_cache for boundaries.
        cache = self._resolve_judge_cache()

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
            llm = llm_factory(
                self._resolve_judge_model(), client=client, max_tokens=8192,
                cache=cache,
            )
            return llm

        if judge_provider == "deepseek":
            # Cloud judge via DeepSeek's OpenAI-compatible endpoint.
            # trust_env deliberately NOT disabled — deepseek is a REMOTE
            # endpoint, going through the system proxy is correct (D-014
            # only applies to localhost). Transport params (T20 timeouts/
            # retries rationale) live in ragas_compat.build_deepseek_client,
            # shared with the T21 synthesiser so a retune lands in both.
            if not os.environ.get("DEEPSEEK_API_KEY"):
                raise ValueError(
                    "RAGAS_JUDGE_PROVIDER=deepseek but DEEPSEEK_API_KEY "
                    "is not set"
                )
            client = build_deepseek_client()
            llm = llm_factory(
                self._resolve_judge_model(), client=client, max_tokens=8192,
                cache=cache,
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

        llm = llm_factory(
            llm_cfg.model, client=llm_client, max_tokens=8192, cache=cache
        )

        return llm

    # Judge-side embeddings: the ONLY local component of the judge stack —
    # the judge LLM is cloud (deepseek), which has no embedding API (T20).
    _DEFAULT_JUDGE_EMB_MODEL = "nomic-embed-text"

    @classmethod
    def _resolve_judge_emb_model(cls) -> str:
        """Local embedding model for AnswerRelevancy (env-swappable)."""
        return os.environ.get(
            "RAGAS_JUDGE_EMB_MODEL", cls._DEFAULT_JUDGE_EMB_MODEL
        )

    def _build_judge_embeddings(self) -> Any:
        """Build a ragas BaseRagasEmbedding backed by local Ollama.

        Reuses ragas's modern OpenAIEmbeddings pointed at Ollama's
        OpenAI-compatible /v1 endpoint (verified live: nomic-embed-text,
        768-d, .wayfinder/tmp/t20_emb_verify.log). trust_env=False on the
        httpx client bypasses the system proxy for localhost traffic (D-014).
        Embeddings are deterministic — no judge cache on this side.
        """
        from openai import AsyncOpenAI
        from ragas.embeddings import OpenAIEmbeddings

        base_url = self._resolve_ollama_base_url()
        # Same proxy bypass as the ollama judge branch (D-014).
        import httpx
        http_client = httpx.AsyncClient(trust_env=False, timeout=60.0)
        client = AsyncOpenAI(
            base_url=base_url, api_key="ollama", http_client=http_client,
        )
        return OpenAIEmbeddings(
            client=client, model=self._resolve_judge_emb_model(),
        )

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


# Module-level alias (single source = the class attribute; tests and
# _metrics_from_settings import the name from either location).
SUPPORTED_METRICS = RagasEvaluator.SUPPORTED_METRICS
