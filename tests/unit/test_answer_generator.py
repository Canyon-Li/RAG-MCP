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

    def test_prompt_grounding_rules_present(self) -> None:
        """T22 prompt surgery (2026-09-15): the grounding contract gained
        three invariants born from the run1/oldexam faithfulness attribution
        — the fixed one-sentence hedge form, the ban on mentioning passage
        labels/file names (citation welding zeroed q20), and the ban on
        computing numbers the passages don't state (q6's "111 fewer")."""
        template = _read_prompt_template().lower()
        assert "only" in template
        assert "never fabricate" in template
        assert "not covered by the passages" in template
        assert "do not mention the passages" in template
        assert "do not compute new numbers" in template

    def test_prompt_caps_answer_length(self) -> None:
        """Length cap targets the faithfulness judge overflow: statement
        count scales with answer length, judge output capped at 8192."""
        template = _read_prompt_template()
        assert "900 characters" in template

    def test_prompt_hard_refusal_rule(self) -> None:
        """T22 layer-1 refusal (2026-09-15): when the fact the question
        actually asks for is absent from the passages, the model must
        hard-refuse with one exact sentence instead of parametric-memory
        filling (old-exam q13/q11 family). Three contract pins: the exact
        refusal sentence, the literal trigger (the asked-for fact absent
        "in its own words" — a check granite 8B can actually perform), and
        the partial-coverage escape (answer what IS covered, don't refuse)."""
        template = _read_prompt_template().lower()
        assert "i cannot answer based on the given passages" in template
        assert "in its own words" in template
        assert "answer those parts" in template


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

    def test_context_passages_numbered_without_source_labels(self) -> None:
        """T22 welding fix: passages carry only "[i]" numbering — NO source
        file names. granite mirrors the assembled format, and "(Source:
        x.pdf)" labels got welded into answer statements the faithfulness
        judge cannot verify (old-exam q20: 0.00 on all-factual content)."""
        llm = MagicMock()
        llm.chat.return_value = MagicMock(content="ans")

        gen = self._make_generator(llm)
        gen("q", [
            _make_chunk("text one", source="alpha-paper.pdf"),
            _make_chunk("text two", source="beta-paper.pdf"),
        ])

        content = llm.chat.call_args.args[0][0].content
        assert "[1]" in content and "text one" in content
        assert "[2]" in content and "text two" in content
        assert "alpha-paper.pdf" not in content
        assert "beta-paper.pdf" not in content
        assert "Source:" not in content
        # absolute path noise must not leak into the prompt
        assert "C:/corpus" not in content

    def test_dict_chunks_supported(self) -> None:
        """dict chunks (text/source keys) assemble like object chunks
        (source metadata is ignored — see the welding-fix test above)."""
        llm = MagicMock()
        llm.chat.return_value = MagicMock(content="ans")

        gen = self._make_generator(llm)
        gen("q", [{"text": "plain dict text", "source_path": "x.pdf"}])

        content = llm.chat.call_args.args[0][0].content
        assert "plain dict text" in content
        assert "x.pdf" not in content

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
