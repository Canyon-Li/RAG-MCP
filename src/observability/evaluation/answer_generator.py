"""Eval-side RAG answer generation (T20).

Turns (query, retrieved chunks) into a grounded answer via the configured
``settings.llm`` — local Ollama granite at temperature 0 — so evaluation
exercises the same retrieve → generate shape the MCP tool exposes. The
prompt lives in ``config/prompts/rag_answer.txt`` (grounding / no-fabrication
rules; edit there, not here).

Failure policy: an LLM failure returns ``None`` — EvalRunner then skips the
answer-side metrics for that query. Falling back to concatenated chunks
would fake faithfulness ≈ 1.0 (the "answer" would literally be the context)
and corrupt the gauge.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.core.settings import resolve_path
from src.libs.llm.base_llm import BaseLLM, Message
from src.libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = "config/prompts/rag_answer.txt"


class RagAnswerGenerator:
    """Callable answer generator: ``(query, chunks) -> Optional[str]``.

    Attributes:
        llm: BaseLLM instance (settings.llm — local granite, temperature 0).
        prompt_template: Loaded from ``config/prompts/rag_answer.txt`` with
            ``{query}`` and ``{context}`` placeholders.
    """

    def __init__(
        self,
        llm: BaseLLM,
        prompt_path: str | None = None,
    ) -> None:
        self.llm = llm
        path = Path(prompt_path) if prompt_path else Path(
            resolve_path(DEFAULT_PROMPT_PATH)
        )
        self.prompt_template = path.read_text(encoding="utf-8").strip()

    def __call__(self, query: str, chunks: list[Any]) -> str | None:
        """Generate an answer; None on failure (see module docstring)."""
        context = self._assemble_context(chunks)
        prompt = self.prompt_template.format(query=query, context=context)
        try:
            response = self.llm.chat([Message(role="user", content=prompt)])
        except Exception as exc:
            logger.warning(
                "Answer generation failed for '%s': %s — answer-side metrics "
                "will be skipped for this query.",
                query[:40], exc,
            )
            return None
        answer = (response.content or "").strip()
        return answer or None

    # ── context assembly ───────────────────────────────────────────

    @staticmethod
    def _assemble_context(chunks: list[Any]) -> str:
        """Number the passages; no source labels.

        T22 (2026-09-15): the previous "[i] (Source: basename)" format was
        the mechanical source of citation welding — granite mirrors the
        assembled format, and file names inside answer statements are
        unverifiable for the faithfulness judge (old-exam q20 scored 0.00
        on all-factual, welded content). Passage numbering stays: it backs
        the prompt's conflict rule (report each version separately).
        Answer-side source citation remains deferred work (2026-09-13) —
        when it lands, attribution must be deliberate, not mirrored.
        """
        blocks: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            blocks.append(f"[{i}]\n{_chunk_text(chunk)}")
        return "\n\n".join(blocks)


def _chunk_text(chunk: Any) -> str:
    """Extract text from the chunk shapes the pipeline produces."""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        return str(chunk.get("text") or chunk.get("content") or chunk)
    if hasattr(chunk, "text"):
        return str(getattr(chunk, "text"))
    return str(chunk)


def build_answer_generator(
    settings: Any,
    llm: BaseLLM | None = None,
) -> Callable[[str, list[Any]], str | None]:
    """Build the answer generator evaluate.py wires into EvalRunner.

    Uses ``LLMFactory.create(settings)`` — i.e. the same provider/model/
    temperature block as production (``settings.llm``), config-only swaps.
    """
    generator = RagAnswerGenerator(llm=llm or LLMFactory.create(settings))
    logger.info(
        "Answer generator ready (llm=%s, prompt=%s)",
        type(generator.llm).__name__, DEFAULT_PROMPT_PATH,
    )
    return generator
