# 纯本地英文文献 RAG 场景适配 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让现有 RAG 项目支撑「绝对本地 + 英文学术论文 + 溯源检索」场景——LLM 全本地化、BM25 英文分词加 stemming/停用词、关闭会改写原文的 LLM 精炼层。

**Architecture:** 不动四层架构。三处适配:(1) 配置切换 LLM 到本地 ollama;(2) 抽取公共 tokenizer 统一索引侧/查询侧分词,英文走 Porter stem + 停用词;(3) 配置关闭 chunk_refiner 的 LLM 层(规则层保留)。

**Tech Stack:** Python 3.10、jieba(中文分词)、纯 Python Porter stemmer(零依赖)、PyYAML 配置、pytest(已有 marker:`unit`)。

## Global Constraints

- **绝对本地**:所有 LLM/embedding 调用走本地 ollama(`trust_env=False` 已就绪),不得引入任何云端 API 依赖。
- **分词一致性(纪律级)**:`SparseEncoder`(索引侧)和 `QueryProcessor`(查询侧)的最终 term 产出**必须完全一致**——同一文本两侧 tokenized 后 term 序列相同,否则 BM25 召不回。公共 tokenizer 是落地手段。
- **零新依赖**:不引入 nltk / snowballstemmer。Porter 算法用本仓库内的纯 Python 实现。
- **溯源确定性**:chunk 文本不得被概率性 LLM 改写——chunk_refiner 只保留确定性规则层。
- **Windows + 中文输出**:任何新脚本/print 若含非 ASCII,按 CLAUDE.md 约定处理 stdout 编码(本计划不涉及新脚本)。
- **测试约定**:repo root 跑 `pytest`,`from src.…` 已由 `conftest.py` 注入 `sys.path`;新单元测试用 `unit` marker(实际 `tests/unit/` 默认即跑)。

**Spec 参考**:[docs/superpowers/specs/2026-08-13-local-english-literature-rag-design.md](../specs/2026-08-13-local-english-literature-rag-design.md)

---

## File Structure

| 文件 | 责任 | 动作 |
|---|---|---|
| `src/core/text/__init__.py` | text 子包入口,导出公共 tokenizer | Create |
| `src/core/text/porter_stemmer.py` | 零依赖纯 Python Porter stemmer(单文件) | Create |
| `src/core/text/tokenizer.py` | 公共 `tokenize(text)`——中文 jieba + 英文(lowercase+stem+停用词+长度过滤) | Create |
| `src/ingestion/embedding/sparse_encoder.py` | `_tokenize` 改为调用公共 tokenizer | Modify |
| `src/core/query_engine/query_processor.py` | `_tokenize` 改为调用公共 tokenizer;停用词表迁移/复用 | Modify |
| `config/settings.yaml` | LLM provider→ollama granite4.1:8b;chunk_refiner.use_llm→false | Modify |
| `tests/unit/test_porter_stemmer.py` | Porter 算法正确性测试 | Create |
| `tests/unit/test_tokenizer.py` | 公共 tokenizer 双语 + 一致性测试 | Create |
| `tests/unit/test_sparse_encoder.py` | 更新:英文 stemming 行为测试 | Modify |
| `tests/unit/test_query_processor.py` | 更新:确认 query 侧与索引侧一致 | Modify |

---

## Task 1: 零依赖 Porter Stemmer

**Files:**
- Create: `src/core/text/porter_stemmer.py`
- Test: `tests/unit/test_porter_stemmer.py`

**Interfaces:**
- Produces: `stem(word: str) -> str` —— 输入小写英文单词,返回其 Porter 词干。空串/非字母原样返回。

- [ ] **Step 1: 写失败测试**

`tests/unit/test_porter_stemmer.py`:
```python
"""Unit tests for the zero-dependency Porter stemmer."""

import pytest
from src.core.text.porter_stemmer import stem


class TestPorterStemmer:
    def test_basic_plural(self):
        assert stem("cats") == "cat"
        assert stem("dogs") == "dog"

    def test_ization_suffix(self):
        # optimization / optimize / optimized 收敛到同一词干
        assert stem("optimization") == stem("optimize")
        assert stem("optimized") == stem("optimize")

    def test_ational_suffix(self):
        assert stem("relational") == "relat"

    def test_empty_and_non_alpha(self):
        assert stem("") == ""
        assert stem("123") == "123"

    def test_already_stem(self):
        assert stem("the") == "the"

    def test_case_insensitive_input(self):
        # 实现内部应 lower();大写输入不应报错且与小写同结果
        assert stem("Cats") == stem("cats")

    def test_convergence_corpus(self):
        """学术文献高频同源词应收敛到同一词干(场景核心诉求)。"""
        family = ["optimize", "optimization", "optimized", "optimizing", "optimal"]
        stems = {stem(w) for w in family}
        # 至少 optimize/optimization/optimized/optimizing 四者收敛
        assert stem("optimize") in stems
        assert len({stem(w) for w in ["optimize", "optimization", "optimized"]}) == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/test_porter_stemmer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.core.text.porter_stemmer'`(且 `src.core.text` 包尚不存在)。

- [ ] **Step 3: 实现包入口 + Porter stemmer**

`src/core/text/__init__.py`:
```python
"""Text-processing utilities shared across ingestion and query layers."""
```

`src/core/text/porter_stemmer.py` —— 采用 Martin Porter 经典算法的精简纯 Python 实现(公有领域算法)。完整实现如下:
```python
"""Zero-dependency Porter stemming algorithm (Porter 1980).

Public domain algorithm; this is a compact, dependency-free Python
implementation used by the shared tokenizer so BM25 term matching
converges English inflections (optimize/optimization/optimized ...).

Reference: https://snowballstem.org/algorithms/porter/stemmer.html
"""

from __future__ import annotations

import re

_VOWELS = "aeiou"


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(word: str) -> int:
    """Count VC sequences (the Porter 'm' measure)."""
    # Build a C/V pattern string: replace consonant runs with 'C', vowel runs with 'V'
    forms = []
    for i in range(len(word)):
        forms.append("C" if _is_consonant(word, i) else "V")
    form = "".join(forms)
    # (VC){m} — count m
    m = 0
    # Collapse to VC pairs
    compressed = re.sub(r"V+", "V", re.sub(r"C+", "C", form))
    m = compressed.count("VC")
    return m


def _contains_vowel(word: str) -> bool:
    return any(not _is_consonant(word, i) for i in range(len(word)))


def _ends_double_consonant(word: str) -> bool:
    if len(word) < 2:
        return False
    return word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    """True if word ends consonant-vowel-consonant where last consonant != w/x/y."""
    if len(word) < 3:
        return False
    n = len(word)
    return (
        _is_consonant(word, n - 3)
        and not _is_consonant(word, n - 2)
        and _is_consonant(word, n - 1)
        and word[-1] not in "wxy"
    )


def _apply_rules(word: str, rules) -> str:
    for suffix, replacement, condition in rules:
        if word.endswith(suffix):
            stem = word[: len(word) - len(suffix)]
            if condition(stem):
                return stem + replacement
            return word  # suffix matches but condition fails → no change
    return word


# Step 1a
_S1A = [("sses", "ss"), ("ies", "i"), ("ss", "ss"), ("s", "")]


def _step1a(word: str) -> str:
    for suffix, repl in _S1A:
        if word.endswith(suffix):
            return word[: len(word) - len(suffix)] + repl
    return word


def _step1b(word: str) -> str:
    if word.endswith("eed"):
        stem = word[:-3]
        if _measure(stem) > 0:
            return stem + "ee"
        return word
    if word.endswith("ed") and _contains_vowel(word[:-2]):
        return _step1b2(word[:-2])
    if word.endswith("ing") and _contains_vowel(word[:-3]):
        return _step1b2(word[:-3])
    return word


def _step1b2(word: str) -> str:
    if word.endswith(("at", "bl", "iz")):
        return word + "e"
    if _ends_double_consonant(word) and not word.endswith(("l", "s", "z")):
        return word[:-1]
    if _measure(word) == 1 and _ends_cvc(word):
        return word + "e"
    return word


def _step1c(word: str) -> str:
    # y -> i if stem contains a vowel
    if word.endswith("y") and _contains_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


def _step2(word: str) -> str:
    rules = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("abli", "able"),
        ("alli", "al"), ("entli", "ent"), ("eli", "e"),
        ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
        ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
        ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
        ("iviti", "ive"), ("biliti", "ble"),
    ]
    for suffix, repl in rules:
        if word.endswith(suffix):
            stem = word[: len(word) - len(suffix)]
            if _measure(stem) > 0:
                return stem + repl
            return word
    return word


def _step3(word: str) -> str:
    rules = [
        ("icate", "ic"), ("ative", ""), ("alize", "al"),
        ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""),
    ]
    for suffix, repl in rules:
        if word.endswith(suffix):
            stem = word[: len(word) - len(suffix)]
            if _measure(stem) > 0:
                return stem + repl
            return word
    return word


def _step4(word: str) -> str:
    suffixes = [
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
        "ement", "ment", "ent", "ou", "ism", "ate", "iti", "ous",
        "ive", "ize", "ion",
    ]
    for suffix in suffixes:
        if word.endswith(suffix):
            stem = word[: len(word) - len(suffix)]
            if suffix == "ion":
                # only if stem ends in s or t
                if _measure(stem) > 0 and stem.endswith(("s", "t")):
                    return stem
                return word
            if _measure(stem) > 1:
                return stem
            return word
    return word


def _step5(word: str) -> str:
    # 5a
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1:
            word = stem
        elif m == 1 and not _ends_cvc(stem):
            word = stem
    # 5b
    if _measure(word) > 1 and _ends_double_consonant(word) and word.endswith("l"):
        word = word[:-1]
    return word


def stem(word: str) -> str:
    """Return the Porter stem of a single English word.

    Args:
        word: An English word (case-insensitive; non-alpha passthrough).

    Returns:
        The stemmed form, lowercased. Empty / non-letter input is returned
        unchanged so callers can feed mixed token streams safely.
    """
    if not word:
        return word
    if not word.isalpha():
        return word.lower()
    word = word.lower()
    if len(word) <= 2:
        return word
    word = _step1a(word)
    word = _step1b(word)
    word = _step1c(word)
    word = _step2(word)
    word = _step3(word)
    word = _step4(word)
    word = _step5(word)
    return word
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/test_porter_stemmer.py -v`
Expected: 所有 TestPorterStemmer 用例 PASS。若 `test_convergence_corpus` 失败,检查 `_step2`/`_step4` 的 `ization`/`ation` 规则与 measure 条件。

- [ ] **Step 5: 提交**

```bash
git add src/core/text/__init__.py src/core/text/porter_stemmer.py tests/unit/test_porter_stemmer.py
git commit -m "feat(text): 零依赖纯 Python Porter stemmer
BM25 英文分词用,让 optimize/optimization/optimized 收敛到同一词干。
属 P1 英文检索质量改造的基础组件。"
```

---

## Task 2: 公共 Tokenizer(双语 + 一致性)

**Files:**
- Create: `src/core/text/tokenizer.py`
- Test: `tests/unit/test_tokenizer.py`

**Interfaces:**
- Consumes: `src.core.text.porter_stemmer.stem`
- Produces: `tokenize(text: str, *, lowercase=True, min_term_length=2, stem_english=True, stopwords=frozenset()) -> list[str]`
  - 中文走 jieba(`jieba.lcut`),保留长度 ≥ min_term_length 的中文 token,**不做 stemming**(中文无需)。
  - 英文走:lowercase → 去纯标点 → 去停用词 → Porter stem → 长度过滤。
  - 返回的 term 已是最终形态(索引侧直接入倒排表,查询侧直接做 BM25 匹配)。
  - `stopwords` 默认空集;调用方传入(复用 QueryProcessor 现有 `ENGLISH_STOPWORDS`)。

- [ ] **Step 1: 写失败测试**

`tests/unit/test_tokenizer.py`:
```python
"""Unit tests for the shared tokenizer."""

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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/test_tokenizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.core.text.tokenizer'`。

- [ ] **Step 3: 实现 tokenizer**

`src/core/text/tokenizer.py`:
```python
"""Shared tokenizer for BM25 index-side and query-side consistency.

The single source of truth for how text becomes BM25 terms. Both
``SparseEncoder`` (ingestion) and ``QueryProcessor`` (query) MUST call
``tokenize`` so the terms stored in the inverted index match the terms
looked up at query time — otherwise BM25 silently never recalls.

Language handling:
- Chinese: jieba segmentation, length-filtered, NO stemming.
- English (ASCII words): lowercase → drop pure-punctuation → stopword
  filter → Porter stem → length filter.
- Numbers and mixed alphanumeric tokens: passthrough (no stem), length filter.

Zero new dependencies: jieba (already used) + the local Porter stemmer.
"""

from __future__ import annotations

import re
from typing import FrozenSet, List

import jieba

from src.core.text.porter_stemmer import stem

# A token is "English/ASCII-word-like" if it is purely ASCII letters.
# Anything with CJK or digits takes a non-stem path.
_ASCII_WORD_RE = re.compile(r"^[a-zA-Z]+$")
# Split jieba tokens further on whitespace/punctuation so "foo,bar" → foo, bar.
_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)


def tokenize(
    text: str,
    *,
    lowercase: bool = True,
    min_term_length: int = 2,
    stem_english: bool = True,
    stopwords: FrozenSet[str] = frozenset(),
) -> List[str]:
    """Tokenize text into final BM25 terms.

    Args:
        text: Input text (Chinese, English, or mixed).
        lowercase: Lowercase all output terms (default True).
        min_term_length: Drop terms shorter than this (default 2).
        stem_english: Apply Porter stemmer to ASCII-letter words (default True).
        stopwords: Stopword set; matched case-insensitively after lowercasing.

    Returns:
        Ordered list of final terms (post stemming/stopword/length filtering).
    """
    if not text or not text.strip():
        return []

    terms: List[str] = []
    seen_lower = set()  # dedup while preserving order

    for raw in jieba.lcut(text):
        # jieba may keep punctuation glued; split on non-word chars.
        for piece in _SPLIT_RE.split(raw):
            piece = piece.strip()
            if not piece:
                continue

            tok = piece.lower() if lowercase else piece

            # Pure punctuation / whitespace after split → skip
            if not re.search(r"\w", tok, re.UNICODE):
                continue

            # Stopword (check pre-stem form)
            if tok in stopwords:
                continue

            # English ASCII word → stem
            if _ASCII_WORD_RE.match(tok):
                final = stem(tok) if stem_english else tok
                if final in stopwords:
                    continue
            else:
                # Chinese / numbers / mixed alphanumeric → passthrough
                final = tok

            if len(final) < min_term_length:
                continue
            if final in seen_lower:
                continue
            seen_lower.add(final)
            terms.append(final)

    return terms
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/test_tokenizer.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: 导出公共入口并提交**

`src/core/text/__init__.py` 更新为:
```python
"""Text-processing utilities shared across ingestion and query layers."""

from src.core.text.tokenizer import tokenize
from src.core.text.porter_stemmer import stem

__all__ = ["tokenize", "stem"]
```

```bash
git add src/core/text/tokenizer.py src/core/text/__init__.py tests/unit/test_tokenizer.py
git commit -m "feat(text): 公共 tokenizer(中文 jieba + 英文 Porter stem + 停用词)

索引侧 SparseEncoder 与查询侧 QueryProcessor 的单一分词真相源,
保证两侧 term 一致(BM25 召回前提)。"
```

---

## Task 3: SparseEncoder 接入公共 Tokenizer

**Files:**
- Modify: `src/ingestion/embedding/sparse_encoder.py:134-169`(`_tokenize` 方法)
- Test: `tests/unit/test_sparse_encoder.py`(追加英文 stemming 测试)

**Interfaces:**
- Consumes: `src.core.text.tokenize`、`QueryProcessor.ENGLISH_STOPWORDS`(复用停用词表)
- Produces: `SparseEncoder._tokenize` 内部改为调用 `tokenize`,对外签名不变,encode 行为升级为英文 stem 化。

- [ ] **Step 1: 追加失败测试**

在 `tests/unit/test_sparse_encoder.py` 末尾追加:
```python
from src.core.query_engine.query_processor import ENGLISH_STOPWORDS


def test_encode_english_stemming_converges():
    """同源英文词经 encode 后应落到同一 term(场景核心:optimization 召回 optimal)。"""
    encoder = SparseEncoder()
    chunk_a = Chunk(id="a", text="optimization", metadata={})
    chunk_b = Chunk(id="b", text="optimize", metadata={})
    stats_a = encoder.encode([chunk_a])[0]
    stats_b = encoder.encode([chunk_b])[0]
    # 两个 chunk 的 term_frequencies 键集应有交集(同一词干)
    common = set(stats_a["term_frequencies"]) & set(stats_b["term_frequencies"])
    assert common, f"expected shared stem term, got {stats_a} vs {stats_b}"


def test_encode_drops_english_stopwords():
    """索引侧应剔除英文停用词,避免倒排表膨胀。"""
    encoder = SparseEncoder()
    chunk = Chunk(id="x", text="the model is good", metadata={})
    stats = encoder.encode([chunk])[0]
    terms = set(stats["term_frequencies"])
    assert "the" not in terms and "is" not in terms
```

- [ ] **Step 2: 运行确认新测试失败**

Run: `pytest tests/unit/test_sparse_encoder.py::test_encode_english_stemming_converges tests/unit/test_sparse_encoder.py::test_encode_drops_english_stopwords -v`
Expected: FAIL —— 当前 `_tokenize` 不做 stem/停用词,两个同源词 term 不交集,停用词仍出现。

- [ ] **Step 3: 改造 `_tokenize`**

`src/ingestion/embedding/sparse_encoder.py` 顶部 import 区,把 `import jieba` 替换为:
```python
from src.core.text.tokenizer import tokenize
from src.core.query_engine.query_processor import ENGLISH_STOPWORDS
```
(保留 `re`、`Counter` 等其它 import 不变。)

将 `_tokenize` 方法整体替换为:
```python
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into BM25 terms via the shared tokenizer.

        Delegates to ``src.core.text.tokenizer.tokenize`` so the index-side
        terms are identical to the query-side (QueryProcessor) terms — the
        hard requirement for BM25 recall. English terms are Porter-stemmed
        and stopwords removed; Chinese is jieba-segmented.

        Args:
            text: Input text to tokenize

        Returns:
            List of final terms (stemmed / stopword-filtered / length-filtered).
        """
        return tokenize(
            text,
            lowercase=self.lowercase,
            min_term_length=self.min_term_length,
            stem_english=True,
            stopwords=frozenset(ENGLISH_STOPWORDS),
        )
```

并在文件顶部确保有 `from typing import ... FrozenSet` 不需要(用内置 `frozenset()` 即可,无需新增 import)。

- [ ] **Step 4: 运行 sparse_encoder 全部测试**

Run: `pytest tests/unit/test_sparse_encoder.py -v`
Expected: 全部 PASS,包括新追加的两条。若旧测试因 stem 改变断言而失败,**逐一核对**:旧测试断言的是未 stem 的原词(如 `"hello"`),经 Porter 后短词通常不变(`hello`→`hello`),应仍通过;若有断言被破坏,更新断言为 stem 后形态并在 commit 说明。

- [ ] **Step 5: 提交**

```bash
git add src/ingestion/embedding/sparse_encoder.py tests/unit/test_sparse_encoder.py
git commit -m "refactor(sparse): SparseEncoder._tokenize 接入公共 tokenizer

英文 term 走 Porter stem + 停用词,与 QueryProcessor 侧一致。
BM25 终于能让 optimization 召回 optimal。"
```

---

## Task 4: QueryProcessor 接入公共 Tokenizer(两侧一致性)

**Files:**
- Modify: `src/core/query_engine/query_processor.py:212-239`(`_tokenize` 方法)
- Test: `tests/unit/test_query_processor.py`(追加跨层一致性测试)

**Interfaces:**
- Consumes: `src.core.text.tokenizer.tokenize`
- Produces: `QueryProcessor._tokenize` 调用公共 tokenizer;`ENGLISH_STOPWORDS` 保持导出(供 sparse 侧复用,不破坏现有 import)。

- [ ] **Step 1: 追加失败测试 —— 跨层一致性**

在 `tests/unit/test_query_processor.py` 末尾追加:
```python
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.core.types import Chunk


def test_query_and_index_tokenization_match():
    """同一文本,query 侧 tokenized 后的关键词 ⊆ 索引侧 term。
    这是 BM25 召回的充要条件(spec 纪律级约束)。"""
    text = "the optimization of neural networks"
    encoder = SparseEncoder()
    stats = encoder.encode([Chunk(id="t", text=text, metadata={})])[0]
    index_terms = set(stats["term_frequencies"])

    processor = QueryProcessor()
    result = processor.process(text)
    query_terms = set(result.keywords) | {k.lower() for k in result.keywords}

    # 查询侧关键词必须在索引侧 term 集合里
    missing = query_terms - index_terms
    # 允许 query 侧 _filter_keywords 因 max_keywords 截断,但不能出现"索引没有"的词
    assert not missing, f"query keywords not in index terms: {missing}"


def test_query_side_stems_english():
    """query 侧英文也应 stem,否则查 optimization 召不回索引里的 optim 词干。"""
    processor = QueryProcessor()
    result = processor.process("optimization")
    # 经 stem 后的形态应能与索引侧一致(具体词干由 Porter 决定)
    encoder = SparseEncoder()
    idx = set(encoder.encode([Chunk(id="t", text="optimization", metadata={})])[0]["term_frequencies"])
    assert set(k.lower() for k in result.keywords) & idx, "query stem != index stem"
```

- [ ] **Step 2: 运行确认新测试失败**

Run: `pytest tests/unit/test_query_processor.py::test_query_and_index_tokenization_match tests/unit/test_query_processor.py::test_query_side_stems_english -v`
Expected: 至少 `test_query_side_stems_english` FAIL —— 当前 query 侧 `_tokenize` 不做 stem。

- [ ] **Step 3: 改造 `_tokenize`**

`src/core/query_engine/query_processor.py` 顶部 import 区增加:
```python
from src.core.text.tokenizer import tokenize as shared_tokenize
```
(保留现有 `import jieba` —— 若 `_tokenize` 改造后 jieba 不再直接使用,可移除;但 `jieba` 可能被同文件其它代码引用,**移除前先 grep 确认无其它引用**,稳妥起见保留 import。)

将 `_tokenize` 方法整体替换为:
```python
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize query text via the shared tokenizer.

        Delegates to ``src.core.text.tokenizer.tokenize`` — the SAME function
        the index-side ``SparseEncoder`` uses — so query terms land on the
        exact stems stored in the BM25 inverted index. English is Porter-
        stemmed; stopwords are removed in ``_filter_keywords`` downstream
        (the shared tokenizer also receives them for defense-in-depth).

        Args:
            text: Query text (filters already stripped by caller)

        Returns:
            List of tokenized terms (stemmed for English, segmented for Chinese).
        """
        return shared_tokenize(
            text,
            lowercase=True,
            min_term_length=1,  # length filter applied in _filter_keywords
            stem_english=True,
            stopwords=frozenset(),  # stopword filtering kept in _filter_keywords
        )
```

> 设计说明:`min_term_length=1` + `stopwords=frozenset()` 是刻意的——QueryProcessor 的 `_filter_keywords` 已有自己的长度/停用词过滤逻辑(`self.config.stopwords`、`min_keyword_length`),这里把 tokenizer 的过滤"放开",避免双重过滤导致 query 侧比索引侧更严格而产生不一致。索引侧(SparseEncoder)用 `min_term_length=2` + `ENGLISH_STOPWORDS`,这是两侧唯一的"不对称"——但它**不会破坏召回**,因为索引侧只是更"干净",query 侧的词只要在索引侧出现过就能命中;反过来若 query 侧过滤更严则会漏召。

- [ ] **Step 4: 运行 query_processor 全部测试**

Run: `pytest tests/unit/test_query_processor.py -v`
Expected: 全部 PASS。重点核对:
- `test_simple_english_query`:`"Azure"` / `"OpenAI"` / `"configuration"` —— 经 stem 后变 `azur`/`openai`/`configur`,旧断言 `"Azure" in result.keywords` 可能失败。**若失败,把断言更新为 stem 后形态**(如 `"azur" in [k.lower() for k in result.keywords]`),并在 commit message 注明"query 侧关键词现以 stem 形态返回,这是与索引侧对齐的必要变化"。

- [ ] **Step 5: 提交**

```bash
git add src/core/query_engine/query_processor.py tests/unit/test_query_processor.py
git commit -m "refactor(query): QueryProcessor._tokenize 接入公共 tokenizer

query 侧与索引侧分词完全一致,英文关键词以 stem 形态返回。
更新受影响断言以反映 stem 化后的期望。"
```

---

## Task 5: 配置切换 —— LLM 本地化 + 关闭 chunk_refiner LLM 层

**Files:**
- Modify: `config/settings.yaml`(L8-17 llm 段、L107-109 chunk_refiner 段)

**Interfaces:**
- 无代码接口变化;`settings.py` 的 `IngestionSettings.chunk_refiner` 已是 `Optional[Dict]`,`.get('use_llm')` 读取,改 yaml 即生效。
- LLM 段:`OllamaLLM` 已注册(见 `src/libs/llm/__init__.py:28`),`base_url`/`trust_env` 已就绪。

**前置(运行时,非代码)**:执行 `ollama pull granite4.1:8b`(本计划不负责,但实施者需确认模型已拉取)。

- [ ] **Step 1: 写验证测试 —— 配置正确性**

新建 `tests/unit/test_settings_local_rag.py`:
```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/unit/test_settings_local_rag.py -v`
Expected: 两条均 FAIL —— 当前 `llm.provider=zhipu`、`chunk_refiner.use_llm=true`。

- [ ] **Step 3: 修改 settings.yaml**

LLM 段(L8-17)改为:
```yaml
llm:
  provider: "ollama"  # 本地 LLM(绝对本地/防泄露);granite4.1:8b 决策见场景适配 spec
  model: "granite4.1:8b"
  deployment_name: ""
  azure_endpoint: ""
  api_version: ""
  api_key: ""  # 本地 ollama 无需 key
  base_url: "http://localhost:11434/v1"  # httpx 需 trust_env=False 绕系统代理(已就绪)
  temperature: 0.0
  max_tokens: 4096
```

chunk_refiner 段(L107-109)改为:
```yaml
  chunk_refiner:
    use_llm: false  # 策略 C:先关 LLM 精炼层(溯源优先确定性),规则去噪层仍恒执行;不满意再开
    # When use_llm is false or LLM call fails, falls back to rule-based refinement
```

- [ ] **Step 4: 运行配置测试确认通过**

Run: `pytest tests/unit/test_settings_local_rag.py -v`
Expected: 两条 PASS。

- [ ] **Step 5: 烟测本地 LLM 链路(非 e2e,只验证 settings 可加载且不触发云端)**

Run:
```bash
python -c "from src.core.settings import load_settings; s=load_settings(); print(f'llm={s.llm.provider}/{s.llm.model}, refiner.use_llm={(s.ingestion.chunk_refiner or {}).get(\"use_llm\")}')"
```
Expected: 输出 `llm=ollama/granite4.1:8b, refiner.use_llm=False`,**无网络请求**(证明配置层已脱离云端)。

- [ ] **Step 6: 提交**

```bash
git add config/settings.yaml tests/unit/test_settings_local_rag.py
git commit -m "feat(config): LLM 切本地 ollama granite4.1:8b + 关闭 chunk_refiner LLM 层

P0 隐私合规(绝对本地)+ P2 减负(溯源优先确定性)。
规则去噪层恒执行,关的只是概率性 LLM 精炼。"
```

---

## Task 6: 全量回归 + 索引侧/查询侧一致性集成验证

**Files:**
- Test: 复用 `tests/unit/`,无新文件(集成验证写在 Task 4 已建的 `test_query_and_index_tokenization_match`);本任务是跑全量回归。

- [ ] **Step 1: 跑全量单元测试**

Run: `pytest tests/unit -v`
Expected: 全部 PASS。重点关注:
- `test_sparse_encoder.py`、`test_query_processor.py`、`test_tokenizer.py`、`test_porter_stemmer.py`、`test_settings_local_rag.py`。
- 若有其它测试因 stem 化失败(如 `test_bm25*`、`test_sparse_retriever*` 断言原词),逐一核对并更新断言。

- [ ] **Step 2: 跑集成测试(跳过 LLM 标记)**

Run: `pytest tests -m "not llm" -v`
Expected: PASS。这会覆盖 hybrid search / RRF / retrieval 等链路,确认分词改造未破坏融合检索。

- [ ] **Step 3: 类型检查与 import 检查**

Run:
```bash
mypy src
python -m compileall src
```
Expected: 无新增类型错误;compileall 通过。

- [ ] **Step 4: 记录已知断言漂移(若有)**

若 Step 1/2 有测试因 stem 化而需改断言,**在本 commit 里一并修正**,commit message 注明"分词 stem 化导致 X 处断言更新,反映 term 现以词干形态存储"。

- [ ] **Step 5: 提交回归修复(若有)**

```bash
git add -A
git commit -m "test: 适配 stem 化后的 term 断言(回归修复)"
```
若 Step 1/2 全绿无需此步,跳过。

---

## Self-Review(计划自检)

**1. Spec 覆盖**:
- P0(LLM 本地化)→ Task 5 ✅
- P1(英文分词:Porter stem + 停用词 + 两侧一致)→ Task 1/2/3/4 ✅
- P2(chunk_refiner 关 LLM 层)→ Task 5 ✅
- P1 副作用"重新 ingest 全库"→ 属运行时操作,非代码;已在 Task 5 Step 5 注明配置层脱离云端,但**未含重新 ingest 步骤**——这是故意的:重新 ingest 依赖用户的真实 PDF 数据,属部署动作而非代码任务。计划末尾的"实施后手动步骤"会提示。
- P3(章节级溯源 section_type/bbox、rerank)→ spec 明确为可选 YAGNI,**本计划不含**,正确。

**2. Placeholder 扫描**:无 TBD/TODO;每个 Step 都有完整代码或精确命令 ✅。

**3. 类型一致性**:
- `tokenize` 签名在 Task 2 定义,Task 3/4 调用参数一致(`lowercase`/`min_term_length`/`stem_english`/`stopwords`)✅。
- `stem(word)->str` Task 1 定义,Task 2 调用一致 ✅。
- `ENGLISH_STOPWORDS` 在 query_processor 定义、Task 3 复用 import,路径一致 ✅。

**4. 潜在风险点(实施者注意)**:
- Task 4 Step 4:`test_simple_english_query` 等旧断言很可能因 stem 化失败,这是**预期的、必须更新的**(不是 bug)。已在 Step 4/5 显式说明。
- Task 3 Step 4:同理,sparse 侧旧断言可能漂移。
- `jieba` import 在 query_processor 改造后是否仍被引用——Task 4 Step 3 已提示"移除前先 grep"。

---

## 实施后手动步骤(非本计划代码任务)

1. `ollama pull granite4.1:8b`(若未拉取)。
2. **重新 ingest 全库**:分词逻辑变了,旧 BM25 索引失效。`python scripts/ingest.py --path <文献目录> --collection <名> --force`(`--force` 绕过 SHA256 跳过)。
3. 抽样验证溯源:查询一条,确认返回 citation 带 `source` + `page`。
4. (可选)跑评估对比分词改造前后:`python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json --json`。
