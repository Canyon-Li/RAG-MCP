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
                if _measure(stem) > 1 and stem.endswith(("s", "t")):
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
