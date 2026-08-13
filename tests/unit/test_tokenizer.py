"""Unit tests for the shared Tokenizer."""

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


class TestTokenizerCodeReviewFixes:
    """Regression tests for code-review findings V2/V4/V3."""

    def test_technical_tokens_kept_whole(self):
        """V2: symbol-suffixed tokens that jieba keeps whole (C++, C#) must
        survive as searchable terms. Old _SPLIT_RE shattered 'C++' into a
        dropped 'c' + '++'."""
        toks = tokenize("Modern C++ beats C#", stopwords=ENGLISH_STOPWORDS)
        toks_set = set(toks)
        assert "c++" in toks_set
        assert "c#" in toks_set
        # 'modern' still stems normally (not captured by _TECH_TOKEN_RE)
        assert "modern" in toks_set

    def test_hyphenated_word_known_limitation(self):
        """Known limitation: jieba splits 'K-Means' into ['K','-','Means'] at the
        hyphen, so it cannot be reassembled into one term without a pre-merge
        pass. Document the actual behavior — 'Means' survives (stemmed to
        'mean'), 'K' is dropped by length filter. Acceptable: the document is
        still retrievable via 'mean'. A full fix would need hyphen-aware
        pre-processing, out of scope for the V2 symbol-token fix."""
        toks = tokenize("K-Means clustering", stopwords=ENGLISH_STOPWORDS)
        toks_set = set(toks)
        assert "mean" in toks_set  # 'Means' stemmed — doc still retrievable
        assert "k-means" not in toks_set  # known: not reassembled

    def test_pure_letter_words_still_stem(self):
        """V2 guard: _TECH_TOKEN_RE must NOT swallow pure-letter words,
        or they bypass stemming. 'optimization' must still stem to 'optim'."""
        toks = tokenize("optimization", stopwords=ENGLISH_STOPWORDS)
        assert toks == ["optim"]

    def test_use_family_filtered_as_stopword_stems(self):
        """V4: used/using/uses Porter-stem to 'us' (not stopword 'use').
        'us' is a near-zero-IDF garbage term — must be filtered via the
        stemmed-stopword expansion, not leak into the index."""
        for word in ["used", "using", "uses"]:
            toks = tokenize(f"{word} quantum gates", stopwords=ENGLISH_STOPWORDS)
            assert "us" not in toks, f"'us' leaked from '{word}': {toks}"
            # content words must still survive
            assert "quantum" in toks and "gate" in toks

    def test_becaus_still_filtered(self):
        """V4 regression: the prior pre-stem fix caught because→becaus;
        the stemmed-stopword expansion must continue to catch it."""
        toks = tokenize("optimization because these networks", stopwords=ENGLISH_STOPWORDS)
        assert "becaus" not in toks
        assert "thes" not in toks
