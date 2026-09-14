"""Unit tests for the T21 exam-synthesis builder (testset_builder).

Covers the deterministic halves of the v5.0 exam pipeline:
feed construction from docling sections, the fixed-field-order final
JSON, reference-context → library-chunk id matching, the rerun quota
formula, QC pre-filter helpers, survivor selection, and the review
markdown renderer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.observability.evaluation.testset_builder import (
    TYPE_LABELS,
    build_feed_text,
    contexts_in_feed,
    finalise_entry,
    find_duplicate_pairs,
    flag_dense_reference,
    match_context_ids,
    normalise_text,
    render_review_md,
    rerun_quota,
    select_final,
    write_final_json,
)

# ── Tests: feed construction ──────────────────────────────────────


class TestBuildFeedText:
    def test_keeps_text_and_table_drops_figure(self) -> None:
        sections = [
            {"type": "title", "text": "Title", "page": 1},
            {"type": "text", "text": "Body paragraph.", "page": 1},
            {"type": "table", "text": "| a | b |", "page": 2},
            {"type": "figure", "text": "[IMAGE: img_001]", "page": 2},
            {"type": "figure_caption", "text": "Fig. 1: circuit", "page": 2},
        ]

        text, report = build_feed_text(sections, excluded_pages=[])

        assert "Body paragraph." in text
        assert "| a | b |" in text
        assert "Fig. 1: circuit" in text
        assert "IMAGE" not in text
        assert report["dropped_figure_sections"] == 1

    def test_excluded_page_sections_dropped(self) -> None:
        # Defensive: docling's figure route normally emits NO text for the
        # 10 known figure pages — if that ever changes, the feed must drop
        # it to stay same-source with the library (T23 known boundary).
        sections = [
            {"type": "text", "text": "keep", "page": 1},
            {"type": "text", "text": "annotation layer text", "page": 6},
        ]

        text, report = build_feed_text(sections, excluded_pages=[6])

        assert "keep" in text
        assert "annotation" not in text
        assert report["excluded_page_sections"] == 1
        assert report["excluded_pages_with_text"] == [6]

    def test_blank_sections_skipped(self) -> None:
        sections = [
            {"type": "text", "text": "  ", "page": 1},
            {"type": "text", "text": "", "page": 1},
        ]

        text, report = build_feed_text(sections, excluded_pages=[])

        assert text == ""
        assert report["kept_sections"] == 0


# ── Tests: final JSON format ──────────────────────────────────────


class TestFinaliseEntry:
    def test_fixed_field_order(self) -> None:
        sample = {
            "persona_name": "p",
            "user_input": "q",
            "query_length": "Medium",
            "query_style": "Formal",
            "reference": "r",
            "reference_contexts": ["c1", "c2"],
            "synthesizer_name": "single_hop_specific_query_synthesizer",
        }

        entry = finalise_entry(sample, context_ids=["chunk_a", None])

        assert list(entry.keys()) == [
            "user_input",
            "reference",
            "reference_contexts",
            "reference_context_ids",
            "persona_name",
            "query_style",
            "query_length",
        ]

    def test_missing_fields_become_null(self) -> None:
        entry = finalise_entry({"user_input": "q", "reference": "r"})

        assert entry["reference_contexts"] is None
        assert entry["reference_context_ids"] is None
        assert entry["persona_name"] is None
        assert entry["query_style"] is None
        assert entry["query_length"] is None

    def test_eval_half_never_leaks(self) -> None:
        sample = {
            "user_input": "q",
            "reference": "r",
            "retrieved_contexts": ["x"],
            "response": "generated answer",
        }

        entry = finalise_entry(sample)

        assert "retrieved_contexts" not in entry
        assert "response" not in entry


class TestWriteFinalJson:
    def test_indented_utf8_array(self, tmp_path: Path) -> None:
        entries = [
            finalise_entry({"user_input": "量子?", "reference": "答案"}),
        ]
        out = tmp_path / "golden_test_set_v5.json"

        write_final_json(entries, out)

        raw = out.read_text(encoding="utf-8")
        assert raw.startswith("[\n  {")
        assert "量子" in raw  # ensure_ascii=False
        assert json.loads(raw) == [
            {
                "user_input": "量子?",
                "reference": "答案",
                "reference_contexts": None,
                "reference_context_ids": None,
                "persona_name": None,
                "query_style": None,
                "query_length": None,
            },
        ]


# ── Tests: reference-context matching (decision B) ────────────────


class TestMatchContextIds:
    def test_containment_picks_tightest_chunk(self) -> None:
        chunks = [
            ("c1", "alpha beta gamma delta epsilon"),
            ("c2", "gamma delta"),
        ]

        ids = match_context_ids(["gamma delta"], chunks)

        assert ids == ["c2"]  # smallest containing chunk, deterministic

    def test_normalisation_ignores_whitespace(self) -> None:
        chunks = [("c1", "word1   word2\n\nword3")]

        ids = match_context_ids(["word1 word2 word3"], chunks)

        assert ids == ["c1"]

    def test_token_overlap_fallback(self) -> None:
        # Context spanning a chunk boundary: no single chunk contains it,
        # but c1 covers 9/10 of its tokens (≥0.85).
        chunks = [
            ("c1", "the depth of quantum circuits for AES is low"),
            ("c2", "something else entirely"),
        ]

        ids = match_context_ids(
            ["the depth of quantum circuits for AES is low here"],
            chunks,
        )

        assert ids == ["c1"]

    def test_no_match_returns_none(self) -> None:
        chunks = [("c1", "unrelated text")]

        ids = match_context_ids(["completely different tokens"], chunks)

        assert ids == [None]


# ── Tests: QC helpers ─────────────────────────────────────────────


class TestNormaliseText:
    def test_collapses_whitespace_and_case(self) -> None:
        assert normalise_text("  Foo   Bar\n") == "foo bar"


class TestContextsInFeed:
    def test_per_context_verdict(self) -> None:
        feed = "the quick brown fox jumps over the lazy dog"
        verdicts = contexts_in_feed(
            ["quick brown fox", "green dragon"], feed,
        )
        assert verdicts == [True, False]


class TestFindDuplicatePairs:
    def test_identical_texts_flagged(self) -> None:
        # Fake embedder: identity vectors for equal texts, orthogonal
        # otherwise — the helper only consumes embed_fn(texts) -> vectors.
        def embed_fn(texts):
            return [[1.0, 0.0] if "aes" in t else [0.0, 1.0] for t in texts]

        pairs = find_duplicate_pairs(
            ["depth of aes circuits", "circuits of aes depth", "sm4 sbox"],
            embed_fn,
            threshold=0.9,
        )

        assert [(i, j) for i, j, _ in pairs] == [(0, 1)]

    def test_below_threshold_not_flagged(self) -> None:
        def embed_fn(texts):
            return [[1.0, 0.0], [0.9, 0.436]]  # cos ≈ 0.9? -> <0.95

        pairs = find_duplicate_pairs(["a", "b"], embed_fn, threshold=0.95)
        assert pairs == []


class TestFlagDenseReference:
    def test_short_reference_ok(self) -> None:
        assert not flag_dense_reference("AES uses 128-bit keys.")

    def test_long_reference_flagged(self) -> None:
        reference = " ".join(f"fact{i}" for i in range(200))
        assert flag_dense_reference(reference)


# ── Tests: rerun quota formula ────────────────────────────────────


class TestRerunQuota:
    def test_formula(self) -> None:
        # gap 5, retention 28/35 = 0.8 → ceil(5/0.8) = 7 → +2 margin = 9
        assert rerun_quota(gap=5, retained=28, arrived=35) == 9

    def test_exact_division(self) -> None:
        assert rerun_quota(gap=4, retained=30, arrived=30) == 6

    def test_zero_gap_raises(self) -> None:
        with pytest.raises(ValueError, match="gap"):
            rerun_quota(gap=0, retained=30, arrived=35)

    def test_nothing_arrived_raises(self) -> None:
        with pytest.raises(ValueError, match="arrived"):
            rerun_quota(gap=5, retained=0, arrived=0)


# ── Tests: final selection ────────────────────────────────────────


def _survivor(q: str, synth: str) -> dict:
    return {"user_input": q, "reference": "r", "synthesizer_name": synth}


SH = "single_hop_specific_query_synthesizer"
MA = "multi_hop_abstract_query_synthesizer"
MS = "multi_hop_specific_query_synthesizer"


class TestSelectFinal:
    def test_proportional_quota(self) -> None:
        survivors = (
            [_survivor(f"sh{i}", SH) for i in range(10)]
            + [_survivor(f"ma{i}", MA) for i in range(10)]
            + [_survivor(f"ms{i}", MS) for i in range(10)]
        )

        selected = select_final(survivors, n=23)

        counts = {
            s: sum(1 for e in selected if e["synthesizer_name"] == s)
            for s in (SH, MA, MS)
        }
        assert counts == {SH: 8, MA: 8, MS: 7}
        assert len(selected) == 23

    def test_thin_type_backfills_from_others(self) -> None:
        survivors = (
            [_survivor(f"sh{i}", SH) for i in range(12)]
            + [_survivor(f"ma{i}", MA) for i in range(2)]
            + [_survivor(f"ms{i}", MS) for i in range(12)]
        )

        selected = select_final(survivors, n=23)

        counts = {
            s: sum(1 for e in selected if e["synthesizer_name"] == s)
            for s in (SH, MA, MS)
        }
        # MA only has 2 survivors; its exact quota (1.77 → floor 1 + 1
        # remainder) already fits — the shortfall lands on the others.
        assert counts == {SH: 11, MA: 2, MS: 10}
        assert len(selected) == 23

    def test_deterministic_pool_order(self) -> None:
        survivors = [_survivor(f"q{i}", SH) for i in range(5)]

        selected = select_final(survivors, n=3)

        assert [e["user_input"] for e in selected] == ["q0", "q1", "q2"]

    def test_fewer_survivors_than_requested(self) -> None:
        survivors = [_survivor("q0", SH), _survivor("q1", MA)]

        selected = select_final(survivors, n=23)

        assert len(selected) == 2


# ── Tests: review markdown ────────────────────────────────────────


class TestRenderReviewMd:
    def test_fixed_structure(self) -> None:
        summary = {
            "synthesised": 35,
            "removed": 2,
            "repaired": 1,
            "finalised": 23,
            "type_distribution": {"单跳细节": 8, "多跳抽象": 8, "多跳对比": 7},
            "personas": ["Alice", "Bob"],
        }
        entries = [
            {
                "qid": "q01",
                "synthesizer_name": SH,
                "user_input": "What is the depth?",
                "reference": "The depth is 100.",
                "excerpt": "the depth of the circuit is 100 gates",
                "verdict": "通过",
            },
            {
                "qid": "q02",
                "synthesizer_name": MA,
                "user_input": "Compare two papers.",
                "reference": "Paper A uses fewer qubits.",
                "excerpt": "x" * 300,
                "verdict": "修复过：reference 覆盖点过多，重生成",
            },
        ]
        removed = [
            {
                "qid": "q03",
                "user_input": "Bad question",
                "reason": "③ reference 含语料外事实",
            },
        ]

        md = render_review_md(summary, entries, removed)

        assert "合成 35 / 剔除 2 / 修复 1 / 定稿 23" in md
        assert "q01【单跳细节】" in md
        assert "q02【多跳抽象】" in md
        assert "What is the depth?" in md
        assert md.index("q01") < md.index("q02")
        assert "x" * 300 not in md  # excerpt truncated
        assert "修复过：reference 覆盖点过多" in md
        assert "已剔除" in md
        assert md.index("q02") < md.index("已剔除")  # removed at the end
        assert "Bad question" in md
        assert "③" in md

    def test_type_labels_cover_official_synthesizers(self) -> None:
        assert set(TYPE_LABELS) == {SH, MA, MS}
