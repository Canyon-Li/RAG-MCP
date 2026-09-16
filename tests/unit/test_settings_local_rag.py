"""Verify settings.yaml reflects the local-English-literature-RAG scenario."""

import re
from pathlib import Path

import pytest

from src.core.settings import SettingsError, load_settings

_REPO = Path(__file__).resolve().parents[2]


def test_llm_provider_is_local_ollama():
    s = load_settings()
    assert s.llm.provider == "ollama", f"expected ollama, got {s.llm.provider}"
    assert "granite" in s.llm.model.lower(), f"expected granite model, got {s.llm.model}"


def test_chunk_refiner_llm_disabled():
    """规则层保留,关闭会改写原文的 LLM 精炼层(溯源确定性)。"""
    s = load_settings()
    cr = s.ingestion.chunk_refiner or {}
    assert cr.get("use_llm") is False, f"expected use_llm=False, got {cr.get('use_llm')}"


def _yaml_with_multiplier(tmp_path: Path, value: int | None) -> Path:
    """复制真实 settings.yaml，改写/删除 rerank_pool_multiplier 行。"""
    src = _REPO / "config" / "settings.yaml"
    text = src.read_text(encoding="utf-8")
    if value is None:
        text = re.sub(r"^\s*rerank_pool_multiplier:.*\n", "", text, flags=re.M)
    else:
        text = re.sub(
            r"^(\s*rerank_pool_multiplier:)\s*\S+",
            rf"\1 {value}",
            text,
            flags=re.M,
        )
    out = tmp_path / "settings.yaml"
    out.write_text(text, encoding="utf-8")
    return out


def test_retrieval_pool_multiplier_landed():
    """T22 D1 落地 pin：dense 40 / sparse 35 / 候选池倍数 5。"""
    s = load_settings()
    assert s.retrieval.dense_top_k == 40
    assert s.retrieval.sparse_top_k == 35
    assert s.retrieval.rerank_pool_multiplier == 5


def test_pool_multiplier_defaults_to_2_when_absent(tmp_path):
    s = load_settings(_yaml_with_multiplier(tmp_path, None))
    assert s.retrieval.rerank_pool_multiplier == 2


def test_pool_multiplier_rejects_below_1(tmp_path):
    with pytest.raises(SettingsError):
        load_settings(_yaml_with_multiplier(tmp_path, 0))
