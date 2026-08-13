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
