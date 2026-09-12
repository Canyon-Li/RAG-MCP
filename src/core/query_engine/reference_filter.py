"""References-section detection for query-side candidate filtering (T19).

Cross-encoder rerank promotes "万金油" chunks — reference-list entries pack the
highest author/title keyword density in the corpus, so concept-word queries
(``gate set``, ``quantum gates``) pull other papers' References pages into the
top-k and squeeze out the chunks that actually cover the reference statements
(T15 attribution: q7 −25pt, intruder ``d28d9bf1_0066``).

Detection is **entry-line density**: a chunk is a references section when at
least half of its non-empty lines look like bibliography entries, where an
entry line must co-exhibit ≥2 of these features:

1. Author-initial pattern ``Cid, C., Murphy, S.,`` (surname + initials)
2. A year ``(1996)`` / bare ``2005``
3. Leading bracket number ``[12]`` (counts double — strong signal)
4. Venue words ``In: / Springer / IEEE / Proceedings / ePrint / pp.``

Requiring feature *co-occurrence* per line plus a ≥50% density threshold keeps
precision: body text with inline ``[n]`` citations, tables with year columns,
and conclusions full of venue words all stay below the bar. Thresholds were
validated by a full-corpus survey (hit set = exactly the papers' references
tails, zero content chunks flagged) — survey numbers live in DEV_CHANGELOG
D-034, not here.

Explicitly out of scope (T19 round 1): abstract / background-popularization
sections — no reliable text features, and abstracts often ARE the answer.
"""

from __future__ import annotations

import re
from typing import Optional

# "Cid, C., Murphy, S., Robshaw, M.:" — surname, comma, 1–3 space-separated
# initials each dotted. Tolerates unicode apostrophes in surnames (O'Neil).
_AUTHOR_INITALS = re.compile(r"[A-Z][A-Za-z'’\-]+,\s*(?:[A-Z]\.\s*){1,3}")

# Terminal "(2005)" / "(2020a)" or any bare 19xx/20xx year.
_YEAR = re.compile(r"\((?:19|20)\d{2}[a-z]?\)\s*$|\b(?:19|20)\d{2}\b")

# Leading "[12]" — numbered reference style (strongest single signal).
_BRACKET_NUM = re.compile(r"^\[\d{1,3}\]")

# Bibliography venue vocabulary — in body text these appear spread over whole
# sentences; on an entry line they cluster with a year or author pattern.
_VENUE = re.compile(
    r"\bIn:|\bSpringer\b|\bIEEE\b|\bACM\b|\bePrint\b|\bCryptology\b"
    r"|\bProceedings\b|\bLNCS\b|\bvol\.|\bpp\.\s*\d"
)

# A chunk whose non-empty lines are ≥ this fraction entry-like is a references
# section. 0.5 absorbs the "References" header line and stray OCR artifacts
# while never being reached by mixed content chunks (surveyed corpus max for
# non-references chunks: well under 0.4).
_DENSITY_THRESHOLD = 0.5

# Below this many non-empty lines the density signal is too weak to trust.
_MIN_LINES = 3


def _entry_like(line: str) -> bool:
    """Whether a single line looks like a bibliography entry.

    An entry needs ≥2 co-occurring features (bracket number alone counts as
    2): one feature in isolation matches too much body text — e.g. a year
    appears in any experimental-results sentence, a venue word in any
    related-work sentence.
    """
    signals = 0
    if _AUTHOR_INITALS.search(line):
        signals += 1
    if _YEAR.search(line):
        signals += 1
    if _BRACKET_NUM.match(line.strip()):
        signals += 2
    if _VENUE.search(line):
        signals += 1
    return signals >= 2


def detect_reference_section(text: Optional[str]) -> bool:
    """Return True if *text* is predominantly a bibliography/references chunk.

    Args:
        text: Chunk body. ``None`` / empty are never references.

    Example:
        >>> detect_reference_section(
        ...     "Grover, L.K.: A fast quantum mechanical algorithm (1996)\\n"
        ...     "Shor, P.W.: Algorithms for quantum computation (1994)\\n"
        ...     "Simon, D.: On the power of quantum computation. In: FOCS (1994)"
        ... )
        True
    """
    if not text:
        return False

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < _MIN_LINES:
        return False

    entry_lines = sum(1 for ln in lines if _entry_like(ln))
    return entry_lines / len(lines) >= _DENSITY_THRESHOLD
