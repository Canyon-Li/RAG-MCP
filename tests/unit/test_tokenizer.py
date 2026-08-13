"""Unit tests for the shared Tokenizer."""

import pytest
from src.core.text.tokenizer import tokenize
from src.core.query_engine.query_processor import ENGLISH_STOPWORDS


class TestTokenizerEnglish:
    def test_lowercase(self):
        assert tokenize("Hello WORLD") == tokenize("hello world")

    def test_stemming_converges_family(self):
        # 场景核心:同源词收敛
        toks_opt = set(tokenize("optimization", stopwords=ENGLISH_STOPWORDS))
        toks_ize = set(tokenize("optimize", stopwords=ENGLISH_STOPWORDS))
        assert toks_opt == toks_ize

    def test_stopwords_removed(self):
        toks = tokenize("the optimization of the model", stopwords=ENGLISH_STOPWORDS)
        assert "the" not in toks
        assert "of" not in toks
        assert len(toks) == 2  # optimization, model

    def test_punctuation_dropped(self):
        toks = tokenize("hello, world!", stopwords=ENGLISH_STOPWORDS)
        assert "," not in toks
        assert "!" not in toks

    def test_min_term_length(self):
        toks = tokenize("a big cat", stopwords=ENGLISH_STOPWORDS, min_term_length=3)
        # "a"(停用词/长度1)、"big"(长度3)→ 保留 big
        assert "big" in toks
        assert all(len(t) >= 3 for t in toks)


class TestTokenizerChinese:
    def test_chinese_segmented(self):
        toks = tokenize("机器学习方法")
        # jieba 分词后应包含语义词
        assert "机器学习" in toks or "机器" in toks

    def test_mixed_language(self):
        toks = tokenize("用 Transformer 做优化", stopwords=ENGLISH_STOPWORDS)
        # 英文 Transformer 经 stem → transform(或 transformer,取决于算法);中文保留
        assert any(t.startswith("transform") for t in toks)


class TestTokenizerEdgeCases:
    def test_empty(self):
        assert tokenize("") == []

    def test_whitespace_only(self):
        assert tokenize("   ") == []

    def test_numbers_passthrough(self):
        # 纯数字 token 不走 stem,原样保留(长度过滤后)
        toks = tokenize("layer 256 attention", stopwords=ENGLISH_STOPWORDS)
        assert "256" in toks
