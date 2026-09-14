"""Unit tests for the shared ragas compatibility shims (T21)."""

from __future__ import annotations

import sys

import pytest

from src.observability.evaluation.ragas_compat import (
    build_deepseek_client,
    install_vertexai_stub,
    resolve_deepseek_config,
)


class TestVertexaiStub:
    def test_stub_installed_and_idempotent(self) -> None:
        install_vertexai_stub()
        assert "langchain_community.chat_models.vertexai" in sys.modules
        # Second call must not raise (or replace a real install).
        install_vertexai_stub()
        assert "langchain_community.chat_models.vertexai" in sys.modules


class TestResolveDeepseekConfig:
    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            resolve_deepseek_config()

    def test_defaults_with_key(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.delenv("RAGAS_JUDGE_BASE_URL", raising=False)
        monkeypatch.delenv("RAGAS_JUDGE_MODEL", raising=False)
        cfg = resolve_deepseek_config()
        assert cfg == {
            "api_key": "sk-test",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-flash",
        }


class TestBuildDeepseekClient:
    def test_constructs_with_env(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        client = build_deepseek_client()
        assert client.api_key == "sk-test"
        assert str(client.base_url).startswith("https://api.deepseek.com")

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            build_deepseek_client()
