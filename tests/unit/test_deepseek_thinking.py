"""DeepSeek thinking-mode default (2026-09-19).

deepseek-flash ships with thinking ON. Ingestion-time callers with tight
completion caps (table summarizer: max_tokens=120) had their entire budget
consumed by reasoning tokens — the API returned an empty ``content``, every
summary silently degraded to None (8/8 failures on the first re-ingest run).
Decision (D-041 追记, DEV_CHANGELOG): turn thinking OFF for **all**
deepseek-flash calls made through :class:`DeepSeekLLM` — generation, table
summaries, everything. The judge builds its own client (ragas_evaluator) and
already disables thinking.

Mechanism: ``DeepSeekLLM.chat`` injects ``extra_body={"thinking":
{"type": "disabled"}}`` unless the caller passes one explicitly;
``OpenAILLM`` just learns to forward ``extra_body`` into the JSON payload
(foreign OpenAI-compatible endpoints must NOT get unknown params injected —
that stays the caller's explicit choice).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from src.libs.llm import DeepSeekLLM, Message, OpenAILLM


@dataclass
class MockLLMSettings:
    provider: str = "deepseek"
    model: str = "deepseek-flash"
    temperature: float = 0.0
    max_tokens: int = 8192


@dataclass
class MockSettings:
    llm: MockLLMSettings | None = None

    def __post_init__(self) -> None:
        if self.llm is None:
            self.llm = MockLLMSettings()


FAKE_RESPONSE: Dict[str, Any] = {
    "choices": [{"message": {"content": "ok"}}],
    "model": "deepseek-flash",
    "usage": {"total_tokens": 3},
}


def _capturing_llm(cls: type, **ctor_kwargs: Any) -> tuple[Any, Dict[str, Any]]:
    """Build an LLM whose _call_api records kwargs and returns a canned body."""
    llm = cls(MockSettings(), **ctor_kwargs)
    captured: Dict[str, Any] = {}
    mock = MagicMock(return_value=FAKE_RESPONSE)

    def spy(**kw: Any) -> Dict[str, Any]:
        captured.update(kw)
        return mock(**kw)

    llm._call_api = spy  # type: ignore[method-assign]
    return llm, captured


class TestDeepSeekThinkingDefault:
    def test_thinking_disabled_by_default(self) -> None:
        llm, captured = _capturing_llm(
            DeepSeekLLM, api_key="k", base_url="https://api.deepseek.com"
        )
        llm.chat([Message(role="user", content="hi")], max_tokens=120)
        assert captured["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_explicit_extra_body_wins(self) -> None:
        llm, captured = _capturing_llm(
            DeepSeekLLM, api_key="k", base_url="https://api.deepseek.com"
        )
        llm.chat(
            [Message(role="user", content="hi")],
            max_tokens=120,
            extra_body={"thinking": {"type": "enabled"}},
        )
        assert captured["extra_body"] == {"thinking": {"type": "enabled"}}

    def test_extra_body_keys_merge_with_default(self) -> None:
        llm, captured = _capturing_llm(
            DeepSeekLLM, api_key="k", base_url="https://api.deepseek.com"
        )
        llm.chat(
            [Message(role="user", content="hi")],
            max_tokens=120,
            extra_body={"foo": 1},
        )
        assert captured["extra_body"] == {
            "foo": 1,
            "thinking": {"type": "disabled"},
        }


class TestOpenAIExtraBodyPassthrough:
    def test_openai_never_injects_thinking(self) -> None:
        llm, captured = _capturing_llm(OpenAILLM, api_key="k")
        llm.chat([Message(role="user", content="hi")], max_tokens=64)
        assert captured.get("extra_body") is None

    def test_openai_forwards_explicit_extra_body(self) -> None:
        llm, captured = _capturing_llm(OpenAILLM, api_key="k")
        llm.chat(
            [Message(role="user", content="hi")],
            max_tokens=64,
            extra_body={"custom_param": True},
        )
        assert captured["extra_body"] == {"custom_param": True}


if __name__ == "__main__":
    pytest.main([__file__])
