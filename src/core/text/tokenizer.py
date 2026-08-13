"""Shared tokenizer for BM25 index-side and query-side consistency.

The single source of truth for how text becomes BM25 terms. Both
``SparseEncoder`` (ingestion) and ``QueryProcessor`` (query) MUST call
``tokenize`` so the terms stored in the inverted index match the terms
looked up at query time — otherwise BM25 silently never recalls.

Language handling:
- Chinese: jieba segmentation, length-filtered, NO stemming.
- English (ASCII words): lowercase → drop pure-punctuation → stopword
  filter (pre- AND post-stem) → Porter stem → length filter.
- Technical tokens with symbols (C++, K-Means, R-CNN): kept whole as
  passthrough terms so they remain searchable (code-review V2 fix).
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
# Technical identifier: ASCII letter run joined by at least one technical
# symbol (+, -, #, .). Matches C++, K-Means, R-CNN, b-tree, ASP.NET. The
# lookahead requires a symbol so pure-letter words (cat, optimization) do
# NOT match — those must still go through the stem path. Kept whole
# (passthrough, lowercased) so symbol-bearing terms stay searchable;
# splitting on non-word chars would shatter "C++" into a dropped "c".
_TECH_TOKEN_RE = re.compile(r"^(?=[a-zA-Z]*[+\-#.])[a-zA-Z]+(?:[+\-#.][a-zA-Z]*)*$")
# Split jieba tokens further on whitespace/punctuation so "foo,bar" → foo, bar.
_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)


def _build_stemmed_stopwords(stopwords: FrozenSet[str]) -> FrozenSet[str]:
    """Expand a stopword set with each ASCII stopword's Porter stem.

    Pre-stem filtering alone misses inflections whose stem differs from the
    surface form: "because"→"becaus", "these"→"thes", and critically
    "use"→"us" (used/using/uses all collapse to "us", a near-zero-IDF garbage
    term). By adding every ASCII stopword's stem to the set, the post-stem
    check catches them. Non-ASCII (Chinese) stopwords are passed through
    unchanged (stem() is a Latin algorithm).
    """
    expanded: set[str] = set(stopwords)
    for w in stopwords:
        if _ASCII_WORD_RE.match(w):
            expanded.add(stem(w))
    return frozenset(expanded)


def tokenize(
    text: str,
    *,
    lowercase: bool = True,
    min_term_length: int = 2,
    stem_english: bool = True,
    stopwords: FrozenSet[str] = frozenset(),
    dedupe: bool = True,
) -> List[str]:
    """Tokenize text into final BM25 terms.

    Args:
        text: Input text (Chinese, English, or mixed).
        lowercase: Lowercase all output terms (default True).
        min_term_length: Drop terms shorter than this (default 2).
        stem_english: Apply Porter stemmer to ASCII-letter words (default True).
        stopwords: Stopword set (surface forms); matched pre-stem AND against
            each ASCII stopword's Porter stem post-stem, so inflections like
            used/using/uses→"us" are also filtered.
        dedupe: Deduplicate terms while preserving order (default True).
            Set False on the index side (SparseEncoder) so repeated terms
            produce accurate BM25 term-frequency counts.

    Returns:
        Ordered list of final terms (post stemming/stopword/length filtering).
    """
    if not text or not text.strip():
        return []

    # Pre-compute once: surface stopwords + their Porter stems (V4 fix).
    effective_stopwords = _build_stemmed_stopwords(stopwords) if stopwords else frozenset()

    terms: List[str] = []
    seen_lower: set[str] = set()  # dedup while preserving order (read only if dedupe)

    for raw in jieba.lcut(text):
        raw_stripped = raw.strip()
        if not raw_stripped:
            continue

        # Technical token with symbols (C++, K-Means) → keep whole (V2 fix).
        tok = raw_stripped.lower() if lowercase else raw_stripped
        if _TECH_TOKEN_RE.match(tok):
            if tok in effective_stopwords:
                continue
            if len(tok) < min_term_length:
                continue
            if dedupe:
                if tok in seen_lower:
                    continue
                seen_lower.add(tok)
            terms.append(tok)
            continue

        # Otherwise split on non-word chars so "foo,bar" → foo, bar.
        for piece in _SPLIT_RE.split(raw_stripped):
            piece = piece.strip()
            if not piece:
                continue

            tok = piece.lower() if lowercase else piece

            # Pure punctuation / whitespace after split → skip.
            # (Technical symbol-tokens like C++ were already captured whole
            # by _TECH_TOKEN_RE above, so split pieces are word-char material.)
            if not re.search(r"\w", tok, re.UNICODE):
                continue

            # Stopword (check pre-stem form)
            if tok in effective_stopwords:
                continue

            # English ASCII word → stem
            if _ASCII_WORD_RE.match(tok):
                final = stem(tok) if stem_english else tok
                if final in effective_stopwords:
                    continue
            else:
                # Chinese / numbers / mixed alphanumeric → passthrough
                final = tok

            if len(final) < min_term_length:
                continue
            if dedupe:
                if final in seen_lower:
                    continue
                seen_lower.add(final)
            terms.append(final)

    return terms
