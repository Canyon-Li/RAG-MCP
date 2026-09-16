"""HybridSearchConfig 提取 rerank_pool_multiplier（T22 D1：重排候选池可配）。

候选池深度原先在 eval/MCP/dashboard 三处硬编码 top_k×2；D1 把它提成
retrieval.rerank_pool_multiplier，经 HybridSearchConfig 传递，缺省 2 =
与旧硬编码行为逐字节一致。
"""

from types import SimpleNamespace

from src.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig
from src.core.settings import RetrievalSettings


def _settings_with(mult: int) -> SimpleNamespace:
    return SimpleNamespace(retrieval=RetrievalSettings(
        dense_top_k=40, sparse_top_k=35, fusion_top_k=10, rrf_k=60,
        filter_references=True, rerank_pool_multiplier=mult,
    ))


def test_extract_propagates_multiplier():
    hs = HybridSearch(settings=_settings_with(5))
    assert hs.config.rerank_pool_multiplier == 5
    assert hs.config.dense_top_k == 40


def test_default_multiplier_is_2():
    assert HybridSearchConfig().rerank_pool_multiplier == 2
    assert HybridSearch(settings=None).config.rerank_pool_multiplier == 2


def test_explicit_config_wins_over_settings():
    cfg = HybridSearchConfig(rerank_pool_multiplier=7)
    hs = HybridSearch(settings=_settings_with(5), config=cfg)
    assert hs.config.rerank_pool_multiplier == 7
