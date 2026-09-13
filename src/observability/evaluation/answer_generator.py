"""Eval-side RAG answer generation (T20).

Turns (query, retrieved chunks) into a grounded answer via the configured
``settings.llm`` — local Ollama granite at temperature 0 — so evaluation
exercises the same retrieve → generate shape the MCP tool exposes. The
prompt lives in ``config/prompts/rag_answer.txt`` (grounding / no-fabrication
/ source-citation rules; edit there, not here).

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

# Source fields mirrored from CustomEvaluator._SOURCE_FIELDS — the real
# EvalRunner path is RetrievalResult.metadata["source_path"]; dicts carry
# source/source_path directly (trace snapshots, tests).
_SOURCE_FIELDS = ("source", "source_path")


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
        """Number the passages and attribute each to its source basename.

        Passage numbering backs the prompt's conflict rule (rag_answer.txt
        rule 3: report each version, noting which passage it came from);
        answer-side citation formatting is deferred work (2026-09-13).
        """
        blocks: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            text = _chunk_text(chunk)
            source = _chunk_source(chunk)
            blocks.append(f"[{i}] (Source: {source})\n{text}")
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


def _chunk_source(chunk: Any) -> str:
    """Extract a display source name (basename) from a chunk."""
    raw: str | None = None
    if isinstance(chunk, dict):
        for field in _SOURCE_FIELDS:
            if chunk.get(field):
                raw = str(chunk[field])
                break
    elif hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
        for field in _SOURCE_FIELDS:
            if chunk.metadata.get(field):
                raw = str(chunk.metadata[field])
                break
    if not raw:
        return "unknown"
    return Path(raw.replace("\\", "/")).name or "unknown"


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
