"""Core layer Generator: LLM answer synthesis over retrieved chunks.

This module implements the CoreGenerator class that:
1. Takes the fused (and optionally reranked) retrieval results and the query
2. Builds a numbered context block whose [n] indices match the citations
   emitted by ``CitationGenerator`` (1-based, result order)
3. Calls the configured LLM (``settings.llm`` via LLMFactory) with the
   prompt from ``config/prompts/generation.txt``
4. Degrades gracefully — any failure returns ``answer=None`` and the caller
   falls back to the plain retrieval response, mirroring CoreReranker

Design Principles:
- Config-Driven: optional ``generation:`` section in settings.yaml
- Graceful Degradation: disabled config / LLM init failure / call failure
  all yield a fallback result instead of raising
- Observable: ``trace.record_stage("generation", {...})`` with method/provider
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.types import RetrievalResult
from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.trace import TraceContext

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = "config/prompts/generation.txt"


@dataclass
class GenerationConfig:
    """Operational configuration for CoreGenerator.

    Attributes:
        enabled: Whether LLM answer generation is enabled
        max_chunks: Maximum number of retrieved chunks fed to the LLM
        max_chunk_chars: Per-chunk character cap in the context block
        max_context_chars: Total context character cap (chunks truncated
            front-first once exceeded)
    """
    enabled: bool = False
    max_chunks: int = 10
    max_chunk_chars: int = 1500
    max_context_chars: int = 12000


@dataclass
class GenerationResult:
    """Result of a generate operation.

    Attributes:
        answer: Generated answer text with [n] citation markers,
            or None when generation was skipped/failed.
        used_fallback: Whether the answer is absent due to a fallback
        fallback_reason: Why the fallback happened (if applicable)
        model: LLM model identifier used (for trace/metadata)
        chunk_count: Number of chunks actually fed to the LLM
    """
    answer: Optional[str] = None
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    model: Optional[str] = None
    chunk_count: int = 0


class CoreGenerator:
    """Core layer LLM generator with fallback support.

    Wraps the ``settings.llm`` backend (LLMFactory) and produces a grounded
    answer whose [n] markers align with the response citation list.

    Example:
        >>> generator = CoreGenerator(settings)
        >>> result = generator.generate("query", results)
        >>> print(result.answer)  # text with [1], [2] markers, or None
    """

    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        config: Optional[GenerationConfig] = None,
        prompt_path: Optional[str] = None,
    ) -> None:
        """Initialize CoreGenerator.

        Args:
            settings: Application settings (llm + optional generation section).
            llm: Optional LLM backend override (tests / DI).
            config: Optional GenerationConfig override.
            prompt_path: Optional prompt file override (tests / DI).
        """
        self.settings = settings
        self.config = config or self._extract_config(settings)
        self._prompt_path = str(prompt_path or DEFAULT_PROMPT_PATH)
        self._prompt_template: Optional[str] = None

        self._llm: Optional[BaseLLM] = None
        if llm is not None:
            self._llm = llm
        elif self.config.enabled:
            try:
                from src.libs.llm.llm_factory import LLMFactory
                self._llm = LLMFactory.create(settings)
            except Exception as e:
                logger.warning(f"Failed to create LLM for generation, disabling: {e}")
                self._llm = None
        self._model_name: Optional[str] = None
        if self._llm is not None:
            self._model_name = getattr(
                getattr(settings, "llm", None), "model", None
            )

    def _extract_config(self, settings: Settings) -> GenerationConfig:
        """Extract GenerationConfig from the optional settings.generation section."""
        generation = getattr(settings, "generation", None)
        if generation is None:
            return GenerationConfig(enabled=False)
        return GenerationConfig(
            enabled=bool(getattr(generation, "enabled", False)),
            max_chunks=int(getattr(generation, "max_chunks", 10)),
            max_chunk_chars=int(getattr(generation, "max_chunk_chars", 1500)),
            max_context_chars=int(getattr(generation, "max_context_chars", 12000)),
        )

    @property
    def is_enabled(self) -> bool:
        """Whether generation will actually run (config on + LLM available)."""
        return bool(self.config.enabled and self._llm is not None)

    def generate(
        self,
        query: str,
        results: List[RetrievalResult],
        trace: Optional["TraceContext"] = None,
    ) -> GenerationResult:
        """Generate a grounded answer for the query over the retrieved chunks.

        Never raises: any failure degrades to ``answer=None`` + fallback info.

        Args:
            query: Original user query.
            results: Retrieval results (order defines citation numbering).
            trace: Optional TraceContext for observability.

        Returns:
            GenerationResult with the answer or fallback details.
        """
        if not self.config.enabled:
            return GenerationResult(used_fallback=True, fallback_reason="disabled")
        if self._llm is None:
            return GenerationResult(used_fallback=True, fallback_reason="llm_unavailable")
        if not results:
            return GenerationResult(used_fallback=True, fallback_reason="no_results")

        chunks = results[: self.config.max_chunks]
        context = self._build_context(chunks)
        prompt_template = self._load_prompt()
        if not prompt_template:
            return self._fallback(
                chunks, trace, "prompt_missing",
                f"Prompt file not found or unreadable: {self._prompt_path}",
            )
        if "{query}" not in prompt_template or "{context}" not in prompt_template:
            return self._fallback(
                chunks, trace, "prompt_invalid",
                "Prompt template missing {query} or {context} placeholder",
            )

        prompt = prompt_template.replace("{query}", query).replace("{context}", context)
        messages = [Message(role="user", content=prompt)]

        _t0 = time.monotonic()
        try:
            response: ChatResponse = self._llm.chat(messages, trace=trace)
        except Exception as e:
            return self._fallback(chunks, trace, "llm_failed", str(e), _t0)

        answer = response.content.strip() if response and response.content else ""
        if not answer:
            return self._fallback(chunks, trace, "llm_empty", "LLM returned empty answer", _t0)

        elapsed_ms = (time.monotonic() - _t0) * 1000.0
        if trace is not None:
            trace.record_stage("generation", {
                "method": "llm",
                "provider": getattr(getattr(self.settings, "llm", None), "provider", "unknown"),
                "model": self._model_name,
                "input_count": len(chunks),
                "answer_chars": len(answer),
            }, elapsed_ms=elapsed_ms)
        return GenerationResult(
            answer=answer,
            model=self._model_name or response.model,
            chunk_count=len(chunks),
        )

    def _fallback(
        self,
        chunks: List[RetrievalResult],
        trace: Optional["TraceContext"],
        reason: str,
        detail: str,
        _t0: Optional[float] = None,
    ) -> GenerationResult:
        """Record a fallback trace stage and build the fallback result."""
        logger.warning(f"Generation fallback ({reason}): {detail}")
        elapsed_ms = ((time.monotonic() - _t0) * 1000.0) if _t0 is not None else None
        if trace is not None:
            data: Dict[str, Any] = {
                "method": "llm",
                "fallback": True,
                "error": reason,
                "input_count": len(chunks),
            }
            trace.record_stage("generation", data, elapsed_ms=elapsed_ms)
        return GenerationResult(
            used_fallback=True,
            fallback_reason=f"{reason}: {detail}",
            model=self._model_name,
            chunk_count=len(chunks),
        )

    def _build_context(self, chunks: List[RetrievalResult]) -> str:
        """Build the numbered context block.

        Chunk numbering is 1-based in result order so the LLM's [n] markers
        align with the CitationGenerator indices in the final response.
        Chunks are truncated per-chunk first, then front-first once the
        total context cap is exceeded.
        """
        parts: List[str] = []
        total = 0
        for i, result in enumerate(chunks, start=1):
            text = " ".join((result.text or "").split())
            if len(text) > self.config.max_chunk_chars:
                text = text[: self.config.max_chunk_chars] + "..."
            source = (result.metadata or {}).get("source_path", "") or ""
            page = (result.metadata or {}).get("page_num", None)
            source_label = source or "unknown source"
            if page not in (None, ""):
                source_label += f" (p.{page})"
            part = f"[{i}] ({source_label})\n{text}"
            total += len(part)
            if total > self.config.max_context_chars and i > 1:
                logger.info(
                    f"Context cap {self.config.max_context_chars} reached, "
                    f"truncated to {i - 1} chunks"
                )
                break
            parts.append(part)
        return "\n\n".join(parts)

    def _load_prompt(self) -> Optional[str]:
        """Load and cache the prompt template (None on failure)."""
        if self._prompt_template is not None:
            return self._prompt_template
        try:
            from src.core.settings import resolve_path
            prompt_path = Path(resolve_path(self._prompt_path))
            if not prompt_path.exists():
                logger.warning(f"Prompt file not found: {prompt_path}")
                return None
            self._prompt_template = prompt_path.read_text(encoding="utf-8")
            return self._prompt_template
        except Exception as e:
            logger.error(f"Failed to load prompt template: {e}")
            return None


def create_core_generator(settings: Settings) -> CoreGenerator:
    """Create a CoreGenerator from settings (mirrors create_core_reranker)."""
    return CoreGenerator(settings=settings)
