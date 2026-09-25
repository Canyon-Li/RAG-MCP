"""Deterministic building blocks for the v5.0 exam synthesis (T21).

Everything here is pure logic — no ragas, no LLM, no network — so the
exam pipeline's decisions (what feeds the generator, what the final
file looks like, which QC verdicts are scriptable) are unit-testable.

The LLM-calling orchestration lives in ``scripts/synthesize_testset.py``;
this module holds:

- feed construction: docling sections → synthesis feed text, honouring
  the T23 known boundary (10 figure pages stay out of the exam because
  their text layer is not in the library either);
- the final exam file: a fixed-field-order indented JSON array
  (``tests/fixtures/golden_test_set_v5.json``);
- reference-context → library-chunk id matching (decision B, T21);
- QC pre-filters (hallucination containment, duplicate pairs, dense
  reference flag) and the rerun quota formula from the ticket;
- survivor selection (type-proportional, deterministic) and the
  human-review markdown renderer.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

# ragas 0.4.3 official default synthesizers (no "reasoning" class exists
# in 0.4.3 — that was the legacy evolution taxonomy). Display labels are
# used in the review markdown only; the exam file carries no type field.
TYPE_LABELS: dict[str, str] = {
    "single_hop_specific_query_synthesizer": "单跳细节",
    "multi_hop_abstract_query_synthesizer": "多跳抽象",
    "multi_hop_specific_query_synthesizer": "多跳对比",
}

# T23 known boundary: these pages' annotation-layer text never entered
# the library (docling figure route, structural and accepted). The exam
# feed excludes them so the exam stays same-source with the library.
# Provenance: T23 Resolution gate output (snapshot t23_gate_final.json).
FIGURE_PAGES: dict[str, list[int]] = {
    "Novel-quantum-circuit-implementation-of-Advanced-Encryption-Standard-with-low-costs": [16],
    "Optimized-quantum-implementation-of-AES": [6, 11, 30, 31, 32],
    "Optimizing-the-Depth-of-Quantum-Implementations-of-Linear-Layers": [19],
    "Quantum-circuit-implementations-of-SM4-block-cipher-optimizing-the-number-of-qubits": [6, 7, 12],
}

# Final exam field order — fixed by the T21 ticket (2026-09-13/14).
# The evaluation half (retrieved_contexts / response) is filled by the
# system at eval time (T20) and is never written here.
FINAL_FIELD_ORDER: list[str] = [
    "user_input",
    "reference",
    "reference_contexts",
    "reference_context_ids",
    "persona_name",
    "query_style",
    "query_length",
]

# v4.0 post-mortem: references covering too many points don't fit in a
# top-5 retrieval window and tank context_recall for structural reasons
# (T01 attribution). ~150 words is where a reference starts covering
# multiple independent facts; the number is recorded per-run in the QC
# ledger, not silently applied.
DENSE_REFERENCE_MAX_WORDS = 150

# Token-overlap ratio below which a reference context counts as "not
# found in this library chunk" (match fallback, see match_context_ids).
_CONTEXT_TOKEN_OVERLAP_MIN = 0.85


# ── feed construction ─────────────────────────────────────────────


def build_feed_text(
    sections: Sequence[dict[str, Any]],
    excluded_pages: Sequence[int],
) -> tuple[str, dict[str, Any]]:
    """Render docling sections into synthesis feed text.

    Drops figure sections (``[IMAGE: id]`` markers carry no retrievable
    text — the library keeps them as markers, the feed should not offer
    them as question material) and any section on an excluded figure
    page (defensive: docling's figure route normally emits no text there
    at all; if that ever changes we still drop it to stay same-source
    with the library, and report it).

    Returns:
        (feed_text, report) — report counts kept/dropped sections and
        lists excluded pages that actually carried text (expected empty;
        non-empty means docling behaviour changed and must be recorded).
    """
    excluded = set(excluded_pages)
    parts: list[str] = []
    kept = figure_dropped = excluded_dropped = 0
    excluded_with_text: set[int] = set()

    for sec in sections:
        stype = sec.get("type", "text")
        text = (sec.get("text") or "").strip()
        page = sec.get("page")

        if stype == "figure":
            # Markers only — image content is not exam material.
            figure_dropped += 1
            continue
        if not text:
            continue
        if page in excluded:
            excluded_dropped += 1
            excluded_with_text.add(page)
            continue
        parts.append(text)
        kept += 1

    report = {
        "total_sections": len(sections),
        "kept_sections": kept,
        "dropped_figure_sections": figure_dropped,
        "excluded_page_sections": excluded_dropped,
        "excluded_pages_with_text": sorted(excluded_with_text),
    }
    return "\n\n".join(parts), report


# ── final exam file ───────────────────────────────────────────────


def finalise_entry(
    sample: dict[str, Any],
    context_ids: list[str | None] | None = None,
) -> dict[str, Any]:
    """Build one final-file object with the fixed field order.

    Missing synthesis fields become ``null``; the evaluation half
    (``retrieved_contexts`` / ``response``) and any extra keys (e.g.
    ``synthesizer_name`` from the pool) never leak into the entry.
    """
    entry: dict[str, Any] = {}
    for field in FINAL_FIELD_ORDER:
        if field == "reference_context_ids" and context_ids is not None:
            entry[field] = context_ids
        else:
            value = sample.get(field)
            entry[field] = value if value is not None else None
    return entry


def write_final_json(entries: Sequence[dict[str, Any]], path: str | Path) -> None:
    """Write the exam as an indented UTF-8 JSON array (no jsonl).

    Stdlib ``json.dump`` preserves dict insertion order, which is how the
    fixed field order from the ticket is honoured; ragas 0.4.3 has
    ``to_jsonl`` but no indented ``to_json``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(entries), f, ensure_ascii=False, indent=2)
        f.write("\n")


# ── reference-context matching (decision B) ───────────────────────


def normalise_text(text: str) -> str:
    """Lowercase and collapse all whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def match_context_ids(
    contexts: Sequence[str],
    chunks: Sequence[tuple[str, str]],
    min_recall: float = _CONTEXT_TOKEN_OVERLAP_MIN,
) -> list[str | None]:
    """Map each reference context to a library chunk id (or None).

    Strategy, deterministic:
    1. containment — the normalised context appears inside the chunk's
       normalised text (contexts are verbatim spans of the same docling
       feed the library was chunked from, so most hits land here); ties
       (overlapping chunks) resolve to the tightest containing chunk.
    2. fallback — token recall against each chunk; best chunk wins if it
       covers ≥ ``min_recall`` of the context's tokens. The default suits
       same-granularity text; when contexts are coarser than chunks (T21:
       ragas KG nodes ~600 words vs 1000-char library chunks, ~1:6), pass
       a lower bar (finalize uses 0.4 = dominant-chunk anchor).
    3. None otherwise — recorded, never guessed.

    Best-effort by design: downstream consumers (T22 evaluation) read
    ``reference`` only; the ids serve QC ③ and future IR debugging.
    """
    normalised_chunks = [(cid, normalise_text(text)) for cid, text in chunks]
    result: list[str | None] = []

    for context in contexts:
        needle = normalise_text(context)
        containing = [
            (cid, ntext) for cid, ntext in normalised_chunks if needle in ntext
        ]
        if containing:
            # Tightest containing chunk = least padding around the span.
            containing.sort(key=lambda item: len(item[1]))
            result.append(containing[0][0])
            continue

        context_tokens = set(_tokens(context))
        if not context_tokens:
            result.append(None)
            continue
        best_id: str | None = None
        best_ratio = 0.0
        for cid, ntext in normalised_chunks:
            chunk_tokens = set(_tokens(ntext))
            ratio = len(context_tokens & chunk_tokens) / len(context_tokens)
            if ratio > best_ratio:
                best_ratio = ratio
                best_id = cid
        result.append(best_id if best_ratio >= min_recall else None)

    return result


# ── QC pre-filters ────────────────────────────────────────────────


def contexts_in_feed(
    contexts: Sequence[str], feed_text: str,
) -> list[bool]:
    """Per-context verdict for QC ③ (hallucination pre-filter).

    True = the normalised context appears verbatim in the synthesis
    feed, i.e. it is corpus-grounded. False routes the question to the
    human review queue (agent reads it before any verdict).
    """
    haystack = normalise_text(feed_text)
    return [bool(c) and normalise_text(c) in haystack for c in contexts]


def find_duplicate_pairs(
    texts: Sequence[str],
    embed_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    threshold: float = 0.85,
) -> list[tuple[int, int, float]]:
    """Find candidate duplicate questions by embedding cosine similarity.

    ``embed_fn`` maps texts to vectors (local nomic in production, a fake
    in tests). Pairs at or above ``threshold`` are *candidates* — the
    "substantively duplicate" verdict stays human (the scale is not
    outsourced to a threshold, same principle as the LLM judge).
    """
    vectors = [list(v) for v in embed_fn(texts)]
    pairs: list[tuple[int, int, float]] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            vi, vj = vectors[i], vectors[j]
            dot = sum(a * b for a, b in zip(vi, vj))
            ni = math.sqrt(sum(a * a for a in vi))
            nj = math.sqrt(sum(b * b for b in vj))
            if ni == 0 or nj == 0:
                continue
            score = dot / (ni * nj)
            if score >= threshold:
                pairs.append((i, j, round(score, 4)))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs


def flag_dense_reference(
    reference: str, max_words: int = DENSE_REFERENCE_MAX_WORDS,
) -> bool:
    """QC ⑤ proxy: does the reference cover too much for a top-5 window?

    Word count is a proxy, not the verdict — flagged references go to
    human review for the repair queue (regenerate, keep the type mix).
    """
    return len(reference.split()) > max_words


# ── rerun quota (ticket formula) ──────────────────────────────────


def rerun_quota(gap: int, retained: int, arrived: int) -> int:
    """How many questions the next rerun should synthesise.

    Ticket formula (T21): quota = ceil(gap ÷ first-round retention rate)
    + 2 margin. Retention rate = retained / arrived from round 1.
    """
    if gap <= 0:
        raise ValueError(f"gap must be positive, got {gap}")
    if arrived <= 0:
        raise ValueError(f"arrived must be positive, got {arrived}")
    if retained <= 0:
        raise ValueError(f"retained must be positive, got {retained}")
    return math.ceil(gap * arrived / retained) + 2


# ── final selection ───────────────────────────────────────────────


def select_final(survivors: Sequence[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Pick the final n survivors, preserving the type mix (T21).

    Quotas per synthesizer type via largest remainder over the survivor
    distribution; a thin type contributes what it has and the shortfall
    backfills from types with surplus, in first-seen pool order. Within
    a type, survivors are taken in pool order — fully deterministic,
    recorded in the QC ledger.
    """
    if len(survivors) <= n:
        return list(survivors)

    groups: dict[str, list[dict[str, Any]]] = {}
    for s in survivors:
        groups.setdefault(s.get("synthesizer_name", "unknown"), []).append(s)

    total = len(survivors)
    # Largest-remainder quotas, ties broken by first-seen group order.
    exact = {name: n * len(members) / total for name, members in groups.items()}
    quotas = {name: int(v) for name, v in exact.items()}
    remaining = n - sum(quotas.values())
    by_fraction = sorted(
        groups, key=lambda name: (-(exact[name] - quotas[name])),
    )
    for name in by_fraction:
        if remaining <= 0:
            break
        if quotas[name] < len(groups[name]):
            quotas[name] += 1
            remaining -= 1

    # Clamp to availability; redistribute the shortfall the same way.
    for _ in range(len(groups)):
        shortfall = n - sum(
            min(quotas[name], len(groups[name])) for name in groups
        )
        if shortfall <= 0:
            break
        for name in by_fraction:
            avail = len(groups[name])
            if quotas[name] < avail:
                quotas[name] += 1
                shortfall -= 1
                if shortfall <= 0:
                    break

    selected: list[dict[str, Any]] = []
    for name, members in groups.items():
        selected.extend(members[: min(quotas[name], len(members))])
    return selected[:n]


# ── review markdown ───────────────────────────────────────────────


_EXCERPT_MAX_CHARS = 200


def render_review_md(
    summary: dict[str, Any],
    entries: Sequence[dict[str, Any]],
    removed: Sequence[dict[str, Any]],
) -> str:
    """Render the fixed-structure human review file (ticket template).

    Top summary → one section per finalised question (``qNN【题型】``,
    question, reference, ≤200-char supporting excerpt, QC verdict) →
    removed questions with reasons at the end.
    """
    lines: list[str] = []
    lines.append("# T21 测试集 v5.0 终审材料")
    lines.append("")
    lines.append(
        f"合成 {summary.get('synthesised', '?')} / "
        f"剔除 {summary.get('removed', '?')} / "
        f"修复 {summary.get('repaired', '?')} / "
        f"定稿 {summary.get('finalised', '?')}"
    )
    lines.append("")
    lines.append("## 题型分布")
    for label, count in summary.get("type_distribution", {}).items():
        lines.append(f"- {label}: {count}")
    lines.append("")
    lines.append("## Persona 一览")
    for persona in summary.get("personas", []):
        lines.append(f"- {persona}")
    lines.append("")

    for entry in entries:
        label = TYPE_LABELS.get(
            entry.get("synthesizer_name", ""),
            entry.get("synthesizer_name", "未知题型"),
        )
        lines.append(f"## {entry['qid']}【{label}】")
        lines.append("")
        lines.append(f"**问题**：{entry.get('user_input', '')}")
        lines.append("")
        lines.append(f"**参考答案**：{entry.get('reference', '')}")
        lines.append("")
        excerpt = entry.get("excerpt") or ""
        if len(excerpt) > _EXCERPT_MAX_CHARS:
            excerpt = excerpt[:_EXCERPT_MAX_CHARS] + "…"
        lines.append(f"**依据 chunk 摘录**：{excerpt}")
        lines.append("")
        lines.append(f"**质检结论**：{entry.get('verdict', '')}")
        lines.append("")

    if removed:
        lines.append("## 已剔除（附理由）")
        lines.append("")
        for item in removed:
            lines.append(
                f"- **{item['qid']}** {item.get('user_input', '')} — "
                f"{item.get('reason', '')}"
            )
        lines.append("")

    return "\n".join(lines)
