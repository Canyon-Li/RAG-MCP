"""Evaluation Panel page – view evaluation reports (read-only).

Evaluations are RUN via ``python scripts/evaluate.py`` (RAGAS + LLM judge
belong in the CLI, not in a Streamlit process). This page only READS the
resulting reports: v2 standalone JSON (``logs/eval_*.json``) and CLI
history (``logs/eval_history.jsonl``).

Layout:
1. Run selector (newest first) + report metadata
2. Five aggregate metric cards (with one-line "what it measures")
3. Per-batch detail table, sortable, low context_recall highlighted
4. History comparison (multi-select → side-by-side table)
"""

from __future__ import annotations

import logging
from typing import Any, List

import pandas as pd
import streamlit as st

from src.observability.dashboard.services.evaluation_report_service import (
    HISTORY_RUN_ID,
    METRIC_SPECS,
    EvaluationReportService,
)

logger = logging.getLogger(__name__)

LOW_CONTEXT_RECALL_THRESHOLD = 0.4


def render() -> None:
    """Render the Evaluation Panel page."""
    st.header("📏 Evaluation Panel")
    st.markdown(
        "评估在 CLI 运行(`python scripts/evaluate.py`),本页只读展示报告。"
        "五指标 = v1 四指标 + **context_recall**(v2 新增)。"
    )

    svc = EvaluationReportService()
    runs = svc.list_runs()

    if not runs:
        st.info(
            "暂无评估报告。先运行:\n\n"
            "```\n"
            "python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json --collection <name>\n"
            "```\n"
            "报告会写入 `logs/eval_*.json`(或 `--output` 指定路径)。"
        )
        return

    # ── Run selector ──────────────────────────────────────────────────
    labels: List[str] = []
    snapshots = {}
    for run_id in runs:
        if run_id == HISTORY_RUN_ID:
            labels.append("📜 CLI 运行历史(eval_history.jsonl)")
            continue
        try:
            snap = svc.load_run(run_id)
            snapshots[run_id] = snap
            labels.append(f"{snap.label}  ·  {run_id}")
        except (ValueError, KeyError) as exc:
            labels.append(f"⚠️ {run_id} ({exc})")

    col_sel, col_meta = st.columns([2, 3])
    with col_sel:
        choice = st.selectbox("报告", options=labels, index=0, key="eval_run_select")
    chosen_id = runs[labels.index(choice)] if labels else None

    if chosen_id == HISTORY_RUN_ID:
        _render_history(svc)
        return

    snap = snapshots.get(chosen_id)
    if snap is None:
        st.error(f"无法加载报告 `{chosen_id}`。")
        return

    with col_meta:
        st.caption(
            f"Run: `{snap.run_id}`  ·  日期: `{snap.date}`  ·  "
            f"问题数: `{snap.n_qa if snap.n_qa is not None else '—'}`"
        )

    # ── Aggregate metric cards ────────────────────────────────────────
    st.subheader("聚合指标(加权)")
    card_cols = st.columns(len(METRIC_SPECS))
    for col, spec in zip(card_cols, METRIC_SPECS):
        with col:
            value = snap.aggregate.get(spec.key)
            st.metric(spec.label, f"{value:.3f}" if value is not None else "—")
            st.caption(spec.description)

    # ── Batch detail table ────────────────────────────────────────────
    st.subheader("分批明细")
    if not snap.batches:
        st.info("该报告没有 batches 明细(旧格式或空跑)。")
    else:
        df = pd.DataFrame(
            [{**{f"Batch {i + 1}": i + 1}, **b} for i, b in enumerate(snap.batches)]
        )
        df = df.rename(columns={s.key: s.label for s in METRIC_SPECS})
        st.dataframe(df, use_container_width=True, hide_index=True)
        low = [
            f"Batch {i + 1}"
            for i, b in enumerate(snap.batches)
            if b.get("context_recall") is not None
            and b["context_recall"] < LOW_CONTEXT_RECALL_THRESHOLD
        ]
        if low:
            st.warning(
                f"🔴 context_recall < {LOW_CONTEXT_RECALL_THRESHOLD} 的批次: "
                + ", ".join(low)
                + "(漏检重灾区,优先排查这批查询)"
            )

    # ── History comparison ────────────────────────────────────────────
    st.divider()
    _render_history(svc)


def _render_history(svc: EvaluationReportService) -> None:
    """Lightweight multi-run comparison from eval_history.jsonl."""
    st.subheader("历史对比(CLI 运行记录)")
    entries = svc.load_history_entries()
    if not entries:
        st.info("没有历史记录(`logs/eval_history.jsonl` 为空或不存在)。")
        return

    options = [snap.label for snap in entries]
    chosen = st.multiselect(
        "选择要对比的运行(按时间倒序)",
        options=options,
        default=options[:1],
        key="eval_history_multiselect",
    )
    if not chosen:
        return

    chosen_set = set(chosen)
    rows = []
    for snap in reversed(entries):        # 时间正序展示
        if snap.label in chosen_set:
            rows.append({"run": snap.label, **snap.aggregate})
    if not rows:
        return
    df = pd.DataFrame(rows).rename(columns={s.key: s.label for s in METRIC_SPECS})
    st.dataframe(df, use_container_width=True, hide_index=True)
