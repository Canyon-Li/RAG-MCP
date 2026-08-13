"""Verify settings.yaml reflects the local-English-literature-RAG scenario."""

from src.core.settings import load_settings


def test_llm_provider_is_local_ollama():
    s = load_settings()
    assert s.llm.provider == "ollama", f"expected ollama, got {s.llm.provider}"
    assert "granite" in s.llm.model.lower(), f"expected granite model, got {s.llm.model}"


def test_chunk_refiner_llm_disabled():
    """规则层保留,关闭会改写原文的 LLM 精炼层(溯源确定性)。"""
    s = load_settings()
    cr = s.ingestion.chunk_refiner or {}
    assert cr.get("use_llm") is False, f"expected use_llm=False, got {cr.get('use_llm')}"
