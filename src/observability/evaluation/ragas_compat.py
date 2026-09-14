"""Shared ragas 0.4.3 compatibility shims (T21 code-review dedup).

Two things every ragas consumer in this repo needs; previously duplicated
between ragas_evaluator (module-level stub + inline client) and
scripts/synthesize_testset.py:

- the vertexai import stub: ragas eagerly imports
  ``langchain_community.chat_models.vertexai`` during its own import;
  that package is not installed here, so a stub module is injected into
  ``sys.modules`` first (mirrors the agentic-rag-for-dummies evaluation
  notebook workaround).
- the DeepSeek OpenAI-compatible client with the T20 transport params
  (connect=15s / read=300s / SDK retries=4 — short SDK defaults dropped
  13/23 questions at 4-metric volume). The judge and the T21 synthesiser
  share one construction so a future retune lands in both. trust_env is
  deliberately NOT disabled: deepseek is a remote endpoint, going
  through the system proxy is correct (D-014 only applies to localhost).
"""

from __future__ import annotations

import os
from typing import Any

_STUB_MODULE = "langchain_community.chat_models.vertexai"


def install_vertexai_stub() -> None:
    """Inject the vertexai stub so ``import ragas`` succeeds (idempotent)."""
    import sys
    import types

    try:
        import langchain_community.chat_models.vertexai  # type: ignore  # noqa: F401
        return
    except ImportError:
        pass
    if _STUB_MODULE in sys.modules:
        return
    stub = types.ModuleType(_STUB_MODULE)

    class _ChatVertexAI:  # minimal stub matching ragas's attribute access
        pass

    stub.ChatVertexAI = _ChatVertexAI  # type: ignore[attr-defined]
    sys.modules[_STUB_MODULE] = stub


def import_ragas() -> None:
    """Install the stub, then import ragas (ImportError if not installed)."""
    install_vertexai_stub()
    import ragas  # type: ignore  # noqa: F401


def resolve_deepseek_config() -> dict[str, str]:
    """Env-driven DeepSeek endpoint config (judge env reused by design:
    the T21 synthesiser is the same model and source as the judge)."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set (load .env or export it)")
    return {
        "api_key": api_key,
        "base_url": os.environ.get(
            "RAGAS_JUDGE_BASE_URL", "https://api.deepseek.com"
        ),
        "model": os.environ.get("RAGAS_JUDGE_MODEL", "deepseek-flash"),
    }


def build_deepseek_client(http_client: Any = None) -> Any:
    """AsyncOpenAI client for DeepSeek with the shared T20 transport params.

    ``http_client``: optional pre-built ``httpx.AsyncClient`` — the T21
    synthesiser passes one carrying a usage-logging response hook; the
    judge passes None and lets the SDK build its own.
    """
    import httpx
    from openai import AsyncOpenAI

    cfg = resolve_deepseek_config()
    return AsyncOpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        timeout=httpx.Timeout(300.0, connect=15.0),
        max_retries=4,
        http_client=http_client,
    )
