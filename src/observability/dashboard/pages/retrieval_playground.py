"""Retrieval Playground page – live retrieval inspection.

Enter a query → immediately see the fused top-k chunks plus a side-by-side
dense / sparse / fusion column comparison, with intersection highlighting
(🟢 = recalled by BOTH paths, ⚪ = single-path) and stage timings. No LLM
generation step — this is a retrieval-effectiveness workbench.

Registered in app.py navigation between Data Browser and Ingestion Manager.
"""

from __future__ import annotations

import logging

import streamlit as st

from src.observability.dashboard.components.chunk_list import render_chunk_list
from src.observability.dashboard.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


def render() -> None:
    """Render the Retrieval Playground page."""
    st.header("🔬 Retrieval Playground")
    st.markdown(
        "输入一条查询,立即查看 **Dense / Sparse** 两路检索结果及 **RRF 融合** 结果。"
        "🟢 = 两路都召回(交集),⚪ = 仅单路召回。"
    )

    svc = RetrievalService()

    # ── Input area ────────────────────────────────────────────────────
    collections = svc.list_collections()
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        query = st.text_input(
            "Query",
            value="",
            key="pg_query",
            placeholder="e.g. How many qubits does the AES-128 circuit use?",
        )
    with col2:
        collection = st.selectbox(
            "Collection",
            options=collections or ["default"],
            index=0,
            key="pg_collection",
            help="目标 Chroma collection(BM25 索引按 collection 隔离)。",
        )
    with col3:
        top_k = st.number_input(
            "Top-K",
            min_value=1,
            max_value=50,
            value=10,
            key="pg_top_k",
        )

    search_clicked = st.button(
        "🔍 检索",
        type="primary",
        key="pg_search_btn",
        disabled=not query.strip(),
    )

    if not collections:
        st.warning(
            "**未发现任何 collection。**先到 Ingestion Manager 页摄取文档,或运行 "
            "`python scripts/ingest.py --path <docs> --collection <name>`。"
        )

    if not search_clicked or not query.strip():
        st.info("输入查询并点击「检索」查看三路对比结果。")
        return

    # ── Execute retrieval ─────────────────────────────────────────────
    try:
        result = svc.search(query=query.strip(), top_k=int(top_k), collection=collection)
    except ValueError as exc:
        st.error(f"❌ 无效查询: {exc}")
        return
    except Exception as exc:
        st.error(f"❌ 检索失败: {exc}")
        logger.exception("Playground search failed")
        return

    # ── Status strip ──────────────────────────────────────────────────
    if result.used_fallback:
        st.warning("⚠️ 单路降级运行(used_fallback=True)。")
    if result.dense_error:
        st.error(f"Dense 路错误: {result.dense_error}")
    if result.sparse_error:
        st.error(f"Sparse 路错误: {result.sparse_error}")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("总耗时", f"{result.timings.get('total', 0):.0f} ms")
    with m2:
        st.metric("Dense 结果数", len(result.dense))
    with m3:
        st.metric("Sparse 结果数", len(result.sparse))
    with m4:
        st.metric("交集 chunk 数", len(result.intersection))

    st.caption(
        "分词结果(与索引侧同源,英文 Porter stem + 中文分词):"
        + " · ".join(f"`{kw}`" for kw in result.keywords)
        if result.keywords else "分词结果:(空)"
    )

    st.divider()

    # ── Three-column comparison ───────────────────────────────────────
    st.subheader("三路对比")
    c_dense, c_sparse, c_fused = st.columns(3)
    with c_dense:
        st.markdown("#### 🟦 Dense (cosine)")
        render_chunk_list(result.dense, prefix="pg_dense", highlight=result.intersection)
    with c_sparse:
        st.markdown("#### 🟨 Sparse (BM25)")
        render_chunk_list(result.sparse, prefix="pg_sparse", highlight=result.intersection)
    with c_fused:
        st.markdown("#### 🟪 Fusion (RRF)")
        render_chunk_list(result.fused, prefix="pg_fused", highlight=result.intersection)
