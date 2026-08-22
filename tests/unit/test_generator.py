"""Unit tests for CoreGenerator (query-side LLM answer generation, D-027).

Mirrors test_reranker_fallback.py patterns: MockSettings with an optional
``generation`` attribute, an injected fake BaseLLM, and RetrievalResult
fixtures. Covers the happy path plus every graceful-degradation branch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.query_engine.generator import (
    CoreGenerator,
    GenerationConfig,
    GenerationResult,
)
from src.core.trace import TraceContext
from src.core.types import RetrievalResult
from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


def make_settings(enabled: bool = True, **overrides) -> Mock:
    """Mock settings with a generation section (or none when enabled is None)."""
    settings = Mock()
    settings.llm = Mock(provider="zhipu", model="glm-4-flash")
    if enabled is None:
        # Settings.generation is Optional — None means "section absent"
        settings.generation = None
        return settings
    generation = Mock(
        enabled=enabled,
        max_chunks=overrides.get("max_chunks", 10),
        max_chunk_chars=overrides.get("max_chunk_chars", 1500),
        max_context_chars=overrides.get("max_context_chars", 12000),
    )
    settings.generation = generation
    return settings


class FakeLLM(BaseLLM):
    """Fake LLM backend returning a canned answer."""

    def __init__(self, answer: str = "总结内容 [1]") -> None:
        self.answer = answer
        self.calls: list[list[Message]] = []

    def chat(self, messages, trace=None, **kwargs) -> ChatResponse:
        self.calls.append(messages)
        return ChatResponse(content=self.answer, model="glm-4-flash")


class FailingLLM(BaseLLM):
    """Fake LLM backend that always raises."""

    def chat(self, messages, trace=None, **kwargs) -> ChatResponse:
        raise RuntimeError("API down")


@pytest.fixture
def sample_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=f"doc1_{i:04d}_ab",
            score=0.9 - i * 0.05,
            text=f"第 {i} 段内容，讲述新材料进展。" * 5,
            metadata={"source_path": "docs/report.pdf", "page_num": i + 1},
        )
        for i in range(3)
    ]


def _prompt_file(tmp_path, content: str) -> str:
    p = tmp_path / "generation.txt"
    p.write_text(content, encoding="utf-8")
    return str(p)


PROMPT = "Question: {query}\nContext:\n{context}\nAnswer:"


class TestCoreGeneratorEnabled:
    def test_happy_path_returns_answer(self, tmp_path, sample_results):
        gen = CoreGenerator(
            make_settings(),
            llm=FakeLLM("答案 [1][2]"),
            prompt_path=_prompt_file(tmp_path, PROMPT),
        )
        assert gen.is_enabled

        result = gen.generate("新材料进展", sample_results)

        assert isinstance(result, GenerationResult)
        assert result.answer == "答案 [1][2]"
        assert not result.used_fallback
        assert result.model == "glm-4-flash"
        assert result.chunk_count == 3

    def test_prompt_contains_numbered_context(self, tmp_path, sample_results):
        llm = FakeLLM()
        gen = CoreGenerator(
            make_settings(),
            llm=llm,
            prompt_path=_prompt_file(tmp_path, PROMPT),
        )
        gen.generate("查询", sample_results)

        sent = llm.calls[0][0].content
        assert "[1] (docs/report.pdf (p.1))" in sent
        assert "[2] (docs/report.pdf (p.2))" in sent
        assert "[3] (docs/report.pdf (p.3))" in sent
        assert "查询" in sent

    def test_max_chunks_limits_context(self, tmp_path):
        results = [
            RetrievalResult(chunk_id=f"d_{i}", score=0.5, text=f"内容{i}", metadata={})
            for i in range(5)
        ]
        llm = FakeLLM()
        gen = CoreGenerator(
            make_settings(max_chunks=2),
            llm=llm,
            prompt_path=_prompt_file(tmp_path, PROMPT),
        )
        result = gen.generate("q", results)

        assert result.chunk_count == 2
        sent = llm.calls[0][0].content
        assert "内容0" in sent and "内容1" in sent
        assert "内容2" not in sent

    def test_trace_records_generation_stage(self, tmp_path, sample_results):
        gen = CoreGenerator(
            make_settings(),
            llm=FakeLLM(),
            prompt_path=_prompt_file(tmp_path, PROMPT),
        )
        trace = TraceContext(trace_type="query")
        gen.generate("查询", sample_results, trace=trace)

        stage = next(s for s in trace.stages if s["stage"] == "generation")
        assert stage["data"]["method"] == "llm"
        assert stage["data"]["input_count"] == 3
        assert "elapsed_ms" in stage


class TestCoreGeneratorDegradation:
    def test_disabled_config_returns_fallback(self, tmp_path, sample_results):
        gen = CoreGenerator(
            make_settings(enabled=False),
            prompt_path=_prompt_file(tmp_path, PROMPT),
        )
        assert not gen.is_enabled

        result = gen.generate("q", sample_results)
        assert result.answer is None
        assert result.used_fallback
        assert result.fallback_reason == "disabled"

    def test_missing_generation_section_disables(self, tmp_path, sample_results):
        gen = CoreGenerator(
            make_settings(enabled=None),
            prompt_path=_prompt_file(tmp_path, PROMPT),
        )
        assert not gen.is_enabled
        assert gen.generate("q", sample_results).fallback_reason == "disabled"

    def test_llm_failure_falls_back(self, tmp_path, sample_results):
        gen = CoreGenerator(
            make_settings(),
            llm=FailingLLM(),
            prompt_path=_prompt_file(tmp_path, PROMPT),
        )
        trace = TraceContext(trace_type="query")
        result = gen.generate("q", sample_results, trace=trace)

        assert result.answer is None
        assert result.used_fallback
        assert "llm_failed" in result.fallback_reason
        stage = next(s for s in trace.stages if s["stage"] == "generation")
        assert stage["data"]["error"] == "llm_failed"

    def test_empty_results_fall_back(self, tmp_path):
        gen = CoreGenerator(
            make_settings(),
            llm=FakeLLM(),
            prompt_path=_prompt_file(tmp_path, PROMPT),
        )
        result = gen.generate("q", [])
        assert result.used_fallback
        assert result.fallback_reason == "no_results"

    def test_missing_prompt_file_falls_back(self, tmp_path, sample_results):
        gen = CoreGenerator(
            make_settings(),
            llm=FakeLLM(),
            prompt_path=str(tmp_path / "nonexistent.txt"),
        )
        result = gen.generate("q", sample_results)
        assert result.answer is None
        assert "prompt_missing" in result.fallback_reason

    def test_prompt_without_placeholder_falls_back(self, tmp_path, sample_results):
        gen = CoreGenerator(
            make_settings(),
            llm=FakeLLM(),
            prompt_path=_prompt_file(tmp_path, "no placeholders here"),
        )
        result = gen.generate("q", sample_results)
        assert "prompt_invalid" in result.fallback_reason

    def test_empty_llm_answer_falls_back(self, tmp_path, sample_results):
        gen = CoreGenerator(
            make_settings(),
            llm=FakeLLM(answer="   "),
            prompt_path=_prompt_file(tmp_path, PROMPT),
        )
        result = gen.generate("q", sample_results)
        assert result.used_fallback
        assert "llm_empty" in result.fallback_reason

    def test_llm_unavailable_falls_back(self, tmp_path, sample_results):
        # Build with generation disabled so the factory isn't called, then
        # flip the flag — simulates "LLM init failed → _llm is None".
        gen = CoreGenerator(
            make_settings(enabled=False),
            llm=None,
            config=GenerationConfig(enabled=False),
            prompt_path=_prompt_file(tmp_path, PROMPT),
        )
        gen.config.enabled = True
        assert not gen.is_enabled

        result = gen.generate("q", sample_results)
        assert result.answer is None
        assert result.fallback_reason == "llm_unavailable"


class TestGenerationSettingsParsing:
    def test_settings_yaml_parsing(self):
        from src.core.settings import Settings

        data = {
            "llm": {"provider": "zhipu", "model": "glm-4-flash",
                    "temperature": 0.0, "max_tokens": 1024},
            "embedding": {"provider": "ollama", "model": "n",
                          "dimensions": 768},
            "vector_store": {"provider": "chroma",
                             "persist_directory": "./d", "collection_name": "c"},
            "retrieval": {"dense_top_k": 20, "sparse_top_k": 20,
                          "fusion_top_k": 10, "rrf_k": 60},
            "rerank": {"enabled": False, "provider": "none",
                       "model": "none", "top_k": 5},
            "evaluation": {"enabled": True, "provider": "composite",
                           "metrics": ["a"]},
            "observability": {"log_level": "INFO", "trace_enabled": True,
                              "trace_file": "./t.jsonl",
                              "structured_logging": True},
            "generation": {"enabled": True, "max_chunks": 7},
        }
        settings = Settings.from_dict(data)
        assert settings.generation is not None
        assert settings.generation.enabled is True
        assert settings.generation.max_chunks == 7
        assert settings.generation.max_context_chars == 12000

    def test_generation_section_optional(self):
        from src.core.settings import Settings

        data = {
            "llm": {"provider": "zhipu", "model": "glm-4-flash",
                    "temperature": 0.0, "max_tokens": 1024},
            "embedding": {"provider": "ollama", "model": "n",
                          "dimensions": 768},
            "vector_store": {"provider": "chroma",
                             "persist_directory": "./d", "collection_name": "c"},
            "retrieval": {"dense_top_k": 20, "sparse_top_k": 20,
                          "fusion_top_k": 10, "rrf_k": 60},
            "rerank": {"enabled": False, "provider": "none",
                       "model": "none", "top_k": 5},
            "evaluation": {"enabled": True, "provider": "composite",
                           "metrics": ["a"]},
            "observability": {"log_level": "INFO", "trace_enabled": True,
                              "trace_file": "./t.jsonl",
                              "structured_logging": True},
        }
        settings = Settings.from_dict(data)
        assert settings.generation is None
