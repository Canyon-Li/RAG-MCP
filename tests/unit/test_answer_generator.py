"""Unit tests for the T20 RAG answer generator (eval-side generation).

The generator turns (query, retrieved chunks) into a grounded answer via the
configured settings.llm (local granite, temperature 0). Failure policy: an
LLM failure returns None — the runner then skips answer-side metrics rather
than falling back to chunk concatenation, which would fake faithfulness ≈ 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


def _read_prompt_template() -> str:
    from src.core.settings import resolve_path

    p = Path(resolve_path("config/prompts/rag_answer.txt"))
    return p.read_text(encoding="utf-8")


def _make_chunk(text: str, source: str = "paper.pdf") -> Any:
    """RetrievalResult-shaped stub: .text + .metadata['source_path']."""
    chunk = MagicMock()
    chunk.text = text
    chunk.metadata = {"source_path": f"C:/corpus/{source}"}
    return chunk


# ── Prompt file ────────────────────────────────────────────────────


class TestPromptFile:
    def test_prompt_file_exists_with_placeholders(self) -> None:
        template = _read_prompt_template()
        assert "{query}" in template
        assert "{context}" in template

    def test_prompt_carries_grounding_rules(self) -> None:
        """User-approved draft: only-retrieved / no-fabrication / no-preamble.
        Source citation was deferred by the user (2026-09-13, faithfulness
        overflow triage) — no assertion on it until that work lands."""
        template = _read_prompt_template().lower()
        assert "only" in template
        assert "never fabricate" in template
        assert "do not cover this" in template

    def test_prompt_caps_answer_length(self) -> None:
        """Length cap targets the faithfulness judge overflow: statement
        count scales with answer length, judge output capped at 8192."""
        template = _read_prompt_template()
        assert "900 characters" in template


# ── Generator behaviour ────────────────────────────────────────────


class TestRagAnswerGenerator:
    def _make_generator(self, llm: Any):
        from src.observability.evaluation.answer_generator import RagAnswerGenerator

        return RagAnswerGenerator(llm=llm, prompt_path=None)

    def test_call_formats_prompt_with_query_and_context(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = MagicMock(content="The answer.")

        gen = self._make_generator(llm)
        out = gen("How many qubits?", [_make_chunk("264/328/392 qubits.")])

        assert out == "The answer."
        messages = llm.chat.call_args.args[0]
        assert len(messages) == 1
        content = messages[0].content
        assert "How many qubits?" in content
        assert "264/328/392 qubits." in content

    def test_context_passages_numbered_with_source_basenames(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = MagicMock(content="ans")

        gen = self._make_generator(llm)
        gen("q", [
            _make_chunk("text one", source="alpha-paper.pdf"),
            _make_chunk("text two", source="beta-paper.pdf"),
        ])

        content = llm.chat.call_args.args[0][0].content
        assert "[1]" in content and "alpha-paper.pdf" in content
        assert "[2]" in content and "beta-paper.pdf" in content
        # absolute path noise must not leak into the prompt
        assert "C:/corpus" not in content

    def test_dict_chunks_supported(self) -> None:
        """dict chunks (text/source keys) assemble like object chunks."""
        llm = MagicMock()
        llm.chat.return_value = MagicMock(content="ans")

        gen = self._make_generator(llm)
        gen("q", [{"text": "plain dict text", "source_path": "x.pdf"}])

        content = llm.chat.call_args.args[0][0].content
        assert "plain dict text" in content
        assert "x.pdf" in content

    def test_llm_failure_returns_none(self) -> None:
        """Decision T20: generation failure → None (skip metrics), never a
        concatenated-chunks pseudo-answer."""
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("ollama down")

        gen = self._make_generator(llm)
        assert gen("q", [_make_chunk("t")]) is None

    def test_string_chunks_fall_back_to_unknown_source(self) -> None:
        llm = MagicMock()
        llm.chat.return_value = MagicMock(content="ans")

        gen = self._make_generator(llm)
        gen("q", ["just a string chunk"])

        content = llm.chat.call_args.args[0][0].content
        assert "just a string chunk" in content


# ── Factory wiring (evaluate.py path) ──────────────────────────────


class TestBuildAnswerGenerator:
    def test_build_uses_llm_factory(self) -> None:
        from src.observability.evaluation.answer_generator import (
            build_answer_generator,
        )

        settings = MagicMock()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "src.libs.llm.llm_factory.LLMFactory.create",
                staticmethod(lambda s: MagicMock(name="llm")),
            )
            gen = build_answer_generator(settings)

        assert callable(gen)

    def test_build_accepts_injected_llm(self) -> None:
        from src.observability.evaluation.answer_generator import (
            build_answer_generator,
        )

        llm = MagicMock(name="injected")
        gen = build_answer_generator(MagicMock(), llm=llm)

        llm.chat.return_value = MagicMock(content="ok")
        assert gen("q", [_make_chunk("t")]) == "ok"
