#!/usr/bin/env python
"""Streamlit chat page for the research-report knowledge service (D-032).

Talks to the FastAPI service (src.api.app) — the page is deliberately a thin
client so the service contract is what's being exercised:

    uvicorn src.api.app:app --port 8300     # terminal 1
    streamlit run scripts/chat_app.py      # terminal 2
"""

import os
import sys
from pathlib import Path

import requests
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

API = os.environ.get("RAG_API_BASE", "http://127.0.0.1:8300")

st.set_page_config(page_title="研报知识库", page_icon="📊", layout="wide")
st.title("📊 券商研报知识库")
st.caption("公有库 + 个人库 · 多租户 RAG 问答 · 引用可回链页码（仅供内部参考，不构成投资建议）")

if "user" not in st.session_state:
    st.session_state.user = None
if "history" not in st.session_state:
    st.session_state.history = []


def api_get(path: str):
    r = requests.get(f"{API}{path}", headers={"X-User-Id": st.session_state.user or ""}, timeout=30)
    r.raise_for_status()
    return r.json()


def api_post(path: str, json=None, files=None):
    r = requests.post(
        f"{API}{path}",
        headers={"X-User-Id": st.session_state.user or ""},
        json=json,
        files=files,
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


# ── sidebar: login + libraries + upload ──────────────────────────────
with st.sidebar:
    try:
        users = api_get("/api/users")["users"]
    except Exception:
        st.error(f"无法连接 API（{API}）。请先启动：`uvicorn src.api.app:app --port 8300`")
        st.stop()

    options = {f"{u['name']}（{u['id']}）": u["id"] for u in users}
    choice = st.selectbox("以谁的身份登录", list(options))
    st.session_state.user = options[choice]

    st.divider()
    st.subheader("我的知识库")
    try:
        libs = api_get("/api/libraries")["libraries"]
        for lib in libs:
            tag = "🌐 公有库" if lib["owner_type"] == "public" else "👤 个人库"
            st.write(f"{tag} **{lib['display_name']}**（{lib['id']}）")
    except Exception as e:
        st.warning(f"库列表加载失败：{e}")

    st.divider()
    st.subheader("上传到我的个人库")
    uploaded = st.file_uploader("选择 PDF", type=["pdf"])
    if uploaded is not None and st.button("开始摄取"):
        with st.spinner("摄取中…解析/图表描述/嵌入（大文件约需数分钟）"):
            try:
                task = api_post("/api/ingest", files={"file": (uploaded.name, uploaded.getvalue())})
                st.success(f"已入队任务 {task['task_id']}（{task['library']}）")
            except Exception as e:
                st.error(f"上传失败：{e}")

# ── chat ─────────────────────────────────────────────────────────────
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"📎 引用来源（{len(msg['citations'])} 条）"):
                for i, c in enumerate(msg["citations"], start=1):
                    lib_tag = "🌐" if c.get("library_type") == "public" else "👤"
                    page = f"第 {c['page']} 页" if c.get("page") else ""
                    st.markdown(
                        f"**[{i}]** {lib_tag} `{c['source']}` {page} "
                        f"（score={c['score']}）"
                    )
                    st.caption(c["snippet"][:150] + "…")

if prompt := st.chat_input("例如：润泽科技的目标价和评级是多少？"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("检索 + 生成中…"):
            try:
                resp = api_post("/api/query", json={"query": prompt, "top_k": 8})
            except Exception as e:
                resp = {"answer": None, "results": [], "error": str(e)}

        citations = resp.get("results", [])
        if resp.get("error"):
            st.error(f"查询失败：{resp['error']}")
        answer = resp.get("answer")
        if answer:
            st.markdown(answer)
            st.caption(f"生成模型：{resp.get('model')} ｜ 检索命中 {len(citations)} 条")
        elif citations:
            st.markdown("（生成未启用或失败，以下为检索命中摘要）")
            for i, c in enumerate(citations[:5], start=1):
                st.markdown(f"**[{i}]** `{c['source']}` — {c['snippet'][:120]}…")
        else:
            st.warning("未检索到相关内容，试试换一种问法或先上传文档。")

    st.session_state.history.append(
        {"role": "assistant", "content": answer or "（见检索结果）", "citations": citations}
    )
