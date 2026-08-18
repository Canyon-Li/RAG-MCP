# Retrieval Playground & Evaluation Panel v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Streamlit 面板升级为"检索效果检视工作台"——新增 Retrieval Playground 页(当场查询→三列分路对比)并重写 Evaluation Panel 对齐 v2 五指标。

**Architecture:** 新增两个 service(`RetrievalService` 唯一检索入口/A3 扩展缝;`EvaluationReportService` 只读报告解析)、一个共享 chunk 渲染组件(从 query_traces 抽取)、一个新页面 + 重写一个页面。全部走现有 Streamlit pages/services 分层,零新依赖。

**Tech Stack:** Python 3.10+/Streamlit 1.56/pytest(含 streamlit.testing.v1 AppTest)

**Spec:** `docs/superpowers/specs/2026-08-19-retrieval-playground-design.md`

## Global Constraints

- 零新 pip 依赖(纯 Streamlit 现有能力)
- 运行环境是 **conda env langchain-test**(不是 .venv);测试命令直接 `pytest`/`python`
- 测试需要 `from src.…` 可导入:conftest 已把 repo root 插入 `sys.path`,新测试文件放 `tests/unit/` 自动满足
- `ruff check .` 与 `mypy src` 必须干净
- Playground 的检索调用 **trace 不落盘**(不传 trace 参数给 HybridSearch)
- 评估**不在面板内运行**——面板只读 `logs/` 下的报告文件
- Windows 控制台:测试/脚本若打印非 ASCII,模块内设 `sys.stdout.reconfigure(encoding="utf-8")`(本计划所有 pytest 文件无 print,不受影响)
- commit 信息用中文,格式 `feat(dashboard): …` / `refactor(dashboard): …` / `test(dashboard): …`

## 关键既有代码事实(实现者必读)

1. **`HybridSearch.search(query, top_k, filters, trace, return_details)`**(src/core/query_engine/hybrid_search.py:217):`return_details=True` 返回 `HybridSearchResult`(dataclass,字段:`results: List[RetrievalResult]`, `dense_results`, `sparse_results`, `dense_error: Optional[str]`, `sparse_error: Optional[str]`, `used_fallback: bool`, `processed_query: Optional[ProcessedQuery]`)。
2. **`RetrievalResult`**(src/core/types.py:323):字段 `chunk_id: str`, `score: float`, `text: str`, `metadata: Dict[str, Any]`(含 `source_path`/`title`/`chunk_index`)。
3. **`ProcessedQuery`**(src/core/types.py):字段 `original_query: str`, `keywords: List[str]`, `filters: Dict`。
4. **装配检索组件的参考实现**是 `QueryKnowledgeHubTool._ensure_initialized`(src/mcp_server/tools/query_knowledge_hub.py:150-227)——`RetrievalService` 照抄其装配方式:
   ```python
   from src.core.query_engine.query_processor import QueryProcessor
   from src.core.query_engine.hybrid_search import create_hybrid_search
   from src.core.query_engine.dense_retriever import create_dense_retriever
   from src.core.query_engine.sparse_retriever import create_sparse_retriever
   from src.ingestion.storage.bm25_indexer import BM25Indexer
   from src.libs.embedding.embedding_factory import EmbeddingFactory
   from src.libs.vector_store.vector_store_factory import VectorStoreFactory

   embedding_client = EmbeddingFactory.create(settings)           # 可跨 collection 复用
   vector_store = VectorStoreFactory.create(settings, collection_name=collection)
   dense = create_dense_retriever(settings=settings, embedding_client=embedding_client, vector_store=vector_store)
   bm25 = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
   sparse = create_sparse_retriever(settings=settings, bm25_indexer=bm25, vector_store=vector_store)
   sparse.default_collection = collection                         # ← 必须设,否则 BM25 查错目录
   hybrid = create_hybrid_search(settings=settings, query_processor=QueryProcessor(), dense_retriever=dense, sparse_retriever=sparse)
   ```
5. **列出 collection** 参考 `Overview._safe_collection_stats`(src/observability/dashboard/pages/overview.py:18-43):`chromadb.PersistentClient(path=…, settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True))` → `client.list_collections()`,每个 `col.name`。
6. **v2 基线报告** `logs/eval_v2_baseline_deepseek.json`:`{run, date, n_qa, batches: List[Dict[str,float]], weighted_aggregate: Dict[str,float]}`;v1/CLI 历史报告 `logs/eval_history.jsonl` 每行 `{timestamp, evaluator_name, query_count, total_elapsed_ms, aggregate_metrics: {五指标}, query_results: [{query, metrics, retrieved_chunk_ids, generated_answer, elapsed_ms}], test_set_path}`。注意:现存的 eval_history.jsonl 16 条全部是**五指标完整**的 v2 格式,"v1 缺 context_recall"只是理论兼容分支。
7. **golden set 按 expected_sources 分组**得到 11 个唯一 source 组(基线报告 12 batches 与之对应);batch 无名字段,展示用 `Batch 1..N` 序号。
8. **e2e 冒烟基建** tests/e2e/test_dashboard_smoke.py:模式是 `AppTest.from_function(page_script, default_timeout=10)` + `patch("...DataService", return_value=mock_svc)` + `assert not at.exception`;`_mock_settings()` helper 已存在可直接 import。
9. **旧 evaluation_panel 的测试** tests/unit/test_evaluation_panel.py 测的是 `_save_to_history/_load_history`——重写面板后这些函数被删,该测试文件整体替换。

---

### Task 1: 共享 chunk 渲染组件(搬家 + highlight 参数)

**Files:**
- Create: `src/observability/dashboard/components/__init__.py`
- Create: `src/observability/dashboard/components/chunk_list.py`
- Modify: `src/observability/dashboard/pages/query_traces.py:554-592`(删除 `_render_chunk_list`,改 import)
- Test: `tests/unit/test_chunk_list_component.py`

**Interfaces:**
- Consumes: 无(首个任务)
- Produces: `render_chunk_list(chunks: List[Dict[str, Any]], *, prefix: str = "chunk", highlight: Optional[set] = None) -> None` — chunks 是 `{"chunk_id": str, "score": float, "text": str, "source": str, "title": str}` 的列表;`highlight` 是 chunk_id 集合,命中的 header 前缀 🟢,未命中且 highlight 非 None 时 ⚪(highlight=None 时维持原 🟢🟡🔴 分数色)。query_traces 内部调用点签名不变(位置参数 prefix 在旧代码里是关键字)。

- [ ] **Step 1: 写失败测试**

```python
"""Unit tests for the shared chunk_list rendering component."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest


def _chunk(chunk_id: str, score: float, text: str = "body", source: str = "a.pdf") -> Dict[str, Any]:
    return {"chunk_id": chunk_id, "score": score, "text": text, "source": source, "title": "T"}


class TestRenderChunkList:
    """AppTest 冒烟 + highlight 逻辑(通过 expander 标题断言)。

    AppTest 的 expander 元素在不同 streamlit 版本上暴露 label/title 属性
    的方式不同——统一用 ``_expander_titles`` 收集,兼容两者。
    """

    def _run(self, chunks: List[Dict[str, Any]], highlight: Optional[set] = None) -> Any:
        from streamlit.testing.v1 import AppTest

        # NOTE: AppTest.from_function 只提取函数源码独立执行、不捕获闭包——变量必须经
        # args=/kwargs= 传入（streamlit >=1.52 实测,否则 NameError）。与 smoke test 模式一致。
        def page_script(chunks, highlight):
            from src.observability.dashboard.components.chunk_list import render_chunk_list
            render_chunk_list(chunks, prefix="t", highlight=highlight)

        at = AppTest.from_function(
            page_script,
            default_timeout=10,
            args=(chunks,),
            kwargs={"highlight": highlight},
        )
        at.run()
        return at

    @staticmethod
    def _expander_titles(at: Any) -> List[str]:
        titles = []
        for e in at.expander:
            for attr in ("label", "title"):
                val = getattr(e, attr, None)
                if val is not None:
                    titles.append(str(val))
                    break
        return titles

    def test_renders_without_exception(self) -> None:
        at = self._run([_chunk("c1", 0.9)])
        assert not at.exception

    def test_empty_list_no_exception(self) -> None:
        at = self._run([])
        assert not at.exception

    def test_highlight_hit_and_miss(self) -> None:
        at = self._run([_chunk("c1", 0.9), _chunk("c2", 0.1)], highlight={"c1"})
        assert not at.exception
        titles = self._expander_titles(at)
        assert any("🟢" in t for t in titles), f"hit chunk 应有 🟢 前缀: {titles}"
        assert any("⚪" in t for t in titles), f"miss chunk 应有 ⚪ 前缀: {titles}"

    def test_highlight_none_keeps_score_colours(self) -> None:
        at = self._run([_chunk("c1", 0.9), _chunk("c2", 0.1)], highlight=None)
        assert not at.exception
        titles = self._expander_titles(at)
        assert any("🟢" in t for t in titles)   # score>=0.8
        assert any("🔴" in t for t in titles)   # score<0.5
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/unit/test_chunk_list_component.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.observability.dashboard.components'`

- [ ] **Step 3: 实现组件**

`src/observability/dashboard/components/__init__.py`:
```python
"""Shared UI components for dashboard pages."""

from src.observability.dashboard.components.chunk_list import render_chunk_list

__all__ = ["render_chunk_list"]
```

`src/observability/dashboard/components/chunk_list.py`:
```python
"""Shared chunk-list rendering component.

Extracted from ``query_traces._render_chunk_list`` so Playground and
Query Traces render chunks identically. ``highlight`` (a set of chunk_ids)
switches the header indicator from score-colours to intersection marking:
🟢 = in the highlight set, ⚪ = not. ``highlight=None`` keeps the original
score-based colouring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st


def render_chunk_list(
    chunks: List[Dict[str, Any]],
    *,
    prefix: str = "chunk",
    highlight: Optional[set] = None,
) -> None:
    """Render a list of chunk dicts as expandable cards.

    Args:
        chunks: List of ``{chunk_id, score, text, source, title}`` dicts.
        prefix: Unique Streamlit widget-key prefix per call site.
        highlight: Optional set of chunk_ids. When not None, header shows
            🟢 for members and ⚪ for non-members (intersection marking);
            when None, header shows score-based colours (🟢>=0.8 🟡>=0.5 🔴).
    """
    for ci, chunk in enumerate(chunks):
        score = chunk.get("score", 0)
        text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id", "")
        source = chunk.get("source", "")
        title = chunk.get("title", "")

        if highlight is not None:
            indicator = "🟢" if chunk_id in highlight else "⚪"
        elif score >= 0.8:
            indicator = "🟢"
        elif score >= 0.5:
            indicator = "🟡"
        else:
            indicator = "🔴"

        header = f"{indicator} **#{ci + 1}** — Score: `{score:.4f}`"
        if title:
            header += f" — {title}"

        with st.expander(header, expanded=False):
            cols = st.columns([2, 3])
            with cols[0]:
                st.caption(f"Chunk ID: `{chunk_id}`")
            with cols[1]:
                if source:
                    st.caption(f"Source: `{source}`")
            if text:
                st.text_area(
                    f"{prefix}_{ci}",
                    value=text,
                    height=max(80, min(len(text) // 2, 400)),
                    disabled=True,
                    label_visibility="collapsed",
                )
            else:
                st.caption("_No text available_")
```

- [ ] **Step 4: query_traces 改 import(行为不变的搬家)**

在 `src/observability/dashboard/pages/query_traces.py`:
1. 顶部 import 区加:
```python
from src.observability.dashboard.components.chunk_list import render_chunk_list
```
2. 删除整个 `_render_chunk_list` 函数(约 554-592 行)。
3. 文件内 3 处调用 `_render_chunk_list(chunks, prefix=…)` 改为 `render_chunk_list(chunks, prefix=…)`(用 grep `_render_chunk_list` 找全)。

- [ ] **Step 5: 运行测试确认通过(含回归)**

Run: `pytest tests/unit/test_chunk_list_component.py tests/e2e/test_dashboard_smoke.py -v`
Expected: 全部 PASS(query_traces 冒烟回归确认搬家无行为变化)

- [ ] **Step 6: Lint + 提交**

```bash
ruff check src/observability/dashboard
mypy src
git add src/observability/dashboard/components tests/unit/test_chunk_list_component.py src/observability/dashboard/pages/query_traces.py
git commit -m "refactor(dashboard): 抽取共享 chunk 渲染组件,支持交集高亮"
```

---

### Task 2: RetrievalService(唯一检索入口)

**Files:**
- Create: `src/observability/dashboard/services/retrieval_service.py`
- Test: `tests/unit/test_retrieval_service.py`

**Interfaces:**
- Consumes: Task 1 无依赖;用 Global Constraints 事实 4 的工厂函数。
- Produces:
  - `@dataclass PlaygroundResult`:字段 `fused: List[Dict]`, `dense: List[Dict]`, `sparse: List[Dict]`, `dense_error: Optional[str]`, `sparse_error: Optional[str]`, `used_fallback: bool`, `keywords: List[str]`, `intersection: set`(dense∩sparse 的 chunk_id), `timings: Dict[str, float]`(键 `dense`/`sparse`/`fusion`/`total`,毫秒)。chunk dict 形如 `{"chunk_id": str, "score": float(保留4位), "text": str, "source": str, "title": str}`。
  - `class RetrievalService`:`__init__(self, settings: Any = None)`;`list_collections() -> List[str]`;`search(query: str, top_k: int = 10, collection: str = "default") -> PlaygroundResult`;`_get_hybrid(collection: str)`(模块级 `@st.cache_resource` 包装,见 Step 3)。search 对空 query 抛 `ValueError`。

- [ ] **Step 1: 写失败测试**

```python
"""Unit tests for RetrievalService (Playground's retrieval facade)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _rr(chunk_id: str, score: float) -> Any:
    """Build a RetrievalResult-like object (duck-typed)."""
    from src.core.types import RetrievalResult
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=f"text-{chunk_id}",
        metadata={"source_path": "a.pdf", "title": "T"},
    )


def _mock_hybrid() -> MagicMock:
    """HybridSearch stand-in returning a HybridSearchResult-like object."""
    from src.core.query_engine.hybrid_search import HybridSearchResult
    h = MagicMock()
    h.search.return_value = HybridSearchResult(
        results=[_rr("f1", 0.9), _rr("d1", 0.8)],
        dense_results=[_rr("d1", 0.8), _rr("both", 0.7)],
        sparse_results=[_rr("both", 0.7), _rr("s1", 0.6)],
        dense_error=None,
        sparse_error=None,
        used_fallback=False,
        processed_query=MagicMock(keywords=["quantum", "aes"]),
    )
    return h


class TestRetrievalService:
    @patch("src.observability.dashboard.services.retrieval_service._get_hybrid")
    def test_search_converts_details(self, mock_get: MagicMock) -> None:
        from src.observability.dashboard.services.retrieval_service import RetrievalService

        mock_get.return_value = _mock_hybrid()
        svc = RetrievalService()
        result = svc.search("quantum AES", top_k=5, collection="papers")

        assert [c["chunk_id"] for c in result.fused] == ["f1", "d1"]
        assert [c["chunk_id"] for c in result.dense] == ["d1", "both"]
        assert result.intersection == {"both"}          # dense∩sparse
        assert result.keywords == ["quantum", "aes"]
        assert result.used_fallback is False
        assert result.dense_error is None
        # chunk dict shape
        assert result.fused[0]["source"] == "a.pdf"
        assert result.fused[0]["title"] == "T"
        assert isinstance(result.timings["total"], float)

    @patch("src.observability.dashboard.services.retrieval_service._get_hybrid")
    def test_search_empty_query_raises(self, mock_get: MagicMock) -> None:
        from src.observability.dashboard.services.retrieval_service import RetrievalService

        svc = RetrievalService()
        with pytest.raises(ValueError):
            svc.search("   ", top_k=5)

    @patch("src.observability.dashboard.services.retrieval_service._get_hybrid")
    def test_search_passes_params(self, mock_get: MagicMock) -> None:
        from src.observability.dashboard.services.retrieval_service import RetrievalService

        h = _mock_hybrid()
        mock_get.return_value = h
        svc = RetrievalService()
        svc.search("q", top_k=7, collection="papers")
        h.search.assert_called_once_with(query="q", top_k=7, return_details=True)

    @patch("src.observability.dashboard.services.retrieval_service.chromadb")
    def test_list_collections(self, mock_chroma: MagicMock) -> None:
        from src.observability.dashboard.services.retrieval_service import RetrievalService

        col = MagicMock()
        col.name = "papers"
        client = mock_chroma.PersistentClient.return_value
        client.list_collections.return_value = [col]
        svc = RetrievalService()
        assert svc.list_collections() == ["papers"]

    def test_list_collections_failure_returns_empty(self) -> None:
        from src.observability.dashboard.services.retrieval_service import RetrievalService

        svc = RetrievalService()
        # chromadb 未初始化/目录缺失 → 空列表而非异常
        with patch("src.observability.dashboard.services.retrieval_service.chromadb",
                   side_effect=Exception("boom")):
            assert svc.list_collections() == []
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/unit/test_retrieval_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.observability.dashboard.services.retrieval_service'`

- [ ] **Step 3: 实现 RetrievalService**

```python
"""RetrievalService – retrieval facade for the Retrieval Playground page.

The single retrieval entry point for the Playground (A3 extension seam):
future knobs (rerank toggle, per-path top_k, …) land here, pages only
render. Wraps ``HybridSearch.search(return_details=True)`` and converts
``HybridSearchResult`` into a UI-friendly ``PlaygroundResult``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class PlaygroundResult:
    """UI-friendly view of one hybrid search execution.

    Attributes:
        fused / dense / sparse: chunk dicts ``{chunk_id, score, text,
            source, title}`` in rank order (score rounded to 4 decimals).
        dense_error / sparse_error: error text per path, None on success.
        used_fallback: whether one path failed and the other was used raw.
        keywords: tokenizer output from ProcessedQuery (index-side aligned).
        intersection: chunk_ids present in BOTH dense and sparse results.
        timings: elapsed ms per stage (dense / sparse / fusion / total).
    """

    fused: List[Dict[str, Any]] = field(default_factory=list)
    dense: List[Dict[str, Any]] = field(default_factory=list)
    sparse: List[Dict[str, Any]] = field(default_factory=list)
    dense_error: Optional[str] = None
    sparse_error: Optional[str] = None
    used_fallback: bool = False
    keywords: List[str] = field(default_factory=list)
    intersection: set = field(default_factory=set)
    timings: Dict[str, float] = field(default_factory=dict)


def _to_chunk_dicts(results: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """Convert RetrievalResult objects to plain dicts for rendering."""
    if not results:
        return []
    return [
        {
            "chunk_id": r.chunk_id,
            "score": round(r.score, 4),
            "text": r.text or "",
            "source": r.metadata.get("source_path", r.metadata.get("source", "")),
            "title": r.metadata.get("title", ""),
        }
        for r in results
    ]


@st.cache_resource
def _get_hybrid(collection: str) -> Any:
    """Build a HybridSearch for ``collection``; cached until collection changes.

    Assembly mirrors ``QueryKnowledgeHubTool._ensure_initialized``
    (src/mcp_server/tools/query_knowledge_hub.py:150).
    """
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.hybrid_search import create_hybrid_search
    from src.core.query_engine.dense_retriever import create_dense_retriever
    from src.core.query_engine.sparse_retriever import create_sparse_retriever
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory
    from src.core.settings import load_settings, resolve_path

    settings = load_settings()
    embedding_client = EmbeddingFactory.create(settings)
    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    dense = create_dense_retriever(
        settings=settings, embedding_client=embedding_client, vector_store=vector_store,
    )
    bm25 = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
    sparse = create_sparse_retriever(
        settings=settings, bm25_indexer=bm25, vector_store=vector_store,
    )
    sparse.default_collection = collection  # BM25 must read this collection's dir
    return create_hybrid_search(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=dense,
        sparse_retriever=sparse,
    )


def _clear_hybrid_cache() -> None:
    """Test hook: drop cached HybridSearch instances."""
    _get_hybrid.clear()


class RetrievalService:
    """Facade the Playground page calls; holds no state besides settings."""

    def __init__(self, settings: Any = None) -> None:
        self._settings = settings

    def list_collections(self) -> List[str]:
        """List Chroma collection names; empty list on any failure."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            from src.core.settings import load_settings, resolve_path

            settings = self._settings or load_settings()
            persist_dir = str(resolve_path(settings.vector_store.persist_directory))
            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            names = []
            for col in client.list_collections():
                names.append(col.name if hasattr(col, "name") else str(col))
            return sorted(names)
        except Exception:
            return []

    def search(
        self,
        query: str,
        top_k: int = 10,
        collection: str = "default",
    ) -> PlaygroundResult:
        """Run one hybrid search and return the UI-friendly result.

        Raises:
            ValueError: If query is empty/whitespace.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

        t_total = time.monotonic()
        hybrid = _get_hybrid(collection)
        details = hybrid.search(query=query, top_k=top_k, return_details=True)

        dense_ids = {r.chunk_id for r in (details.dense_results or [])}
        sparse_ids = {r.chunk_id for r in (details.sparse_results or [])}
        timings = {"total": (time.monotonic() - t_total) * 1000.0}

        return PlaygroundResult(
            fused=_to_chunk_dicts(details.results),
            dense=_to_chunk_dicts(details.dense_results),
            sparse=_to_chunk_dicts(details.sparse_results),
            dense_error=details.dense_error,
            sparse_error=details.sparse_error,
            used_fallback=details.used_fallback,
            keywords=list(details.processed_query.keywords) if details.processed_query else [],
            intersection=dense_ids & sparse_ids,
            timings=timings,
        )
```

注意:`timings` 只填 `total`(dense/sparse/fusion 分段耗时需要传 trace 收集,与"trace 不落盘"冲突——但 `TraceContext` 可仅用于计时不 flush;实现者可选用:构造 `TraceContext()` 传给 search,读 `elapsed_ms()` 填 dense/sparse,不调 collector 即不落盘。若这条路径卡壳,退化为只显示 total 也可接受,**但必须保证无 trace 文件写入副作用**)。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/test_retrieval_service.py -v`
Expected: 5 个测试全 PASS

- [ ] **Step 5: Lint + 提交**

```bash
ruff check src/observability/dashboard
mypy src
git add src/observability/dashboard/services/retrieval_service.py tests/unit/test_retrieval_service.py
git commit -m "feat(dashboard): RetrievalService——Playground 唯一检索入口(A3 扩展缝)"
```

---

### Task 3: EvaluationReportService(报告解析)

**Files:**
- Create: `src/observability/dashboard/services/evaluation_report_service.py`
- Test: `tests/unit/test_evaluation_report_service.py`

**Interfaces:**
- Consumes: 无前置任务依赖。
- Produces:
  - `METRIC_SPECS: List[MetricSpec]`,其中 `@dataclass(frozen=True) MetricSpec` 字段 `key: str, label: str, description: str`——五指标固定顺序:`source_recall_at_k`→`source_recall@5`→"top-5 里有没有命中正确论文",`source_precision_at_k`→`source_precision@5`→"top-5 里命中正确论文的比例",`context_relevance`→`context_relevance`→"检索回的上下文和问题相关吗",`context_precision`→`context_precision`→"相关 chunk 是否排在前面",`context_recall`→`context_recall`→"该召回的信息,检索结果覆盖了吗"。
  - `@dataclass EvalRunSnapshot`:字段 `run_id: str`(v2 用 `run` 字段/否则文件名), `label: str`(显示名,含日期), `date: str`, `n_qa: Optional[int]`, `batches: List[Dict[str, Optional[float]]]`(每行含五指标,缺失键填 None), `aggregate: Dict[str, Optional[float]]`(五指标,缺失填 None)。
  - `class EvaluationReportService`:`__init__(self, logs_dir: Optional[Path] = None)`(None→`resolve_path("logs")`);`list_runs() -> List[str]`(v2 `eval_*.json` 按修改时间倒序 + `eval_history.jsonl` 若存在排最后);`load_run(run_id: str) -> EvalRunSnapshot`(`run_id` 为 json 文件名或 `"__history__"`);`load_history_entries() -> List[EvalRunSnapshot]`(jsonl 每行一个,label 为 timestamp)。`load_run` 对未知 id 抛 `KeyError`,对损坏 JSON 抛 `ValueError`。

- [ ] **Step 1: 写失败测试**

```python
"""Unit tests for EvaluationReportService (read-only report parsing)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


def _write_v2(logs: Path) -> None:
    logs.joinpath("eval_v2_baseline.json").write_text(json.dumps({
        "run": "v2-baseline",
        "date": "2026-08-16",
        "n_qa": 23,
        "batches": [
            {"context_precision": 0.75, "context_recall": 0.67, "context_relevance": 1.0,
             "source_precision_at_k": 0.7, "source_recall_at_k": 1.0},
        ],
        "weighted_aggregate": {
            "context_precision": 0.65, "context_recall": 0.49, "context_relevance": 0.87,
            "source_precision_at_k": 0.71, "source_recall_at_k": 0.96,
        },
    }, ensure_ascii=False), encoding="utf-8")


def _write_v1(logs: Path) -> None:
    """v1 report: aggregate only, missing context_recall key."""
    logs.joinpath("eval_v1_old.json").write_text(json.dumps({
        "run": "v1-old",
        "date": "2026-07-01",
        "n_qa": 10,
        "batches": [],
        "weighted_aggregate": {
            "context_precision": 0.6, "context_relevance": 0.9,
            "source_precision_at_k": 0.7, "source_recall_at_k": 1.0,
        },
    }, ensure_ascii=False), encoding="utf-8")


class TestListRuns:
    def test_empty_dir(self, tmp_path: Path) -> None:
        from src.observability.dashboard.services.evaluation_report_service import (
            EvaluationReportService,
        )
        svc = EvaluationReportService(logs_dir=tmp_path)
        assert svc.list_runs() == []

    def test_v2_files_sorted_latest_first(self, tmp_path: Path) -> None:
        from src.observability.dashboard.services.evaluation_report_service import (
            EvaluationReportService,
        )
        old = tmp_path / "eval_a.json"
        new = tmp_path / "eval_b.json"
        old.write_text("{}", encoding="utf-8")
        new.write_text("{}", encoding="utf-8")
        import os
        os.utime(old, (1_000_000, 1_000_000))
        os.utime(new, (2_000_000, 2_000_000))
        svc = EvaluationReportService(logs_dir=tmp_path)
        assert svc.list_runs() == ["eval_b.json", "eval_a.json"]

    def test_history_appended_last(self, tmp_path: Path) -> None:
        from src.observability.dashboard.services.evaluation_report_service import (
            EvaluationReportService,
        )
        (tmp_path / "eval_x.json").write_text("{}", encoding="utf-8")
        (tmp_path / "eval_history.jsonl").write_text("\n", encoding="utf-8")
        svc = EvaluationReportService(logs_dir=tmp_path)
        assert svc.list_runs()[-1] == "__history__"


class TestLoadRun:
    def test_v2_full(self, tmp_path: Path) -> None:
        from src.observability.dashboard.services.evaluation_report_service import (
            EvaluationReportService,
        )
        _write_v2(tmp_path)
        svc = EvaluationReportService(logs_dir=tmp_path)
        snap = svc.load_run("eval_v2_baseline.json")
        assert snap.run_id == "v2-baseline"
        assert snap.n_qa == 23
        assert len(snap.batches) == 1
        assert snap.aggregate["context_recall"] == pytest.approx(0.49)
        # 五指标键全在
        for key in ("source_recall_at_k", "source_precision_at_k",
                    "context_relevance", "context_precision", "context_recall"):
            assert key in snap.aggregate

    def test_v1_missing_metric_is_none(self, tmp_path: Path) -> None:
        from src.observability.dashboard.services.evaluation_report_service import (
            EvaluationReportService,
        )
        _write_v1(tmp_path)
        svc = EvaluationReportService(logs_dir=tmp_path)
        snap = svc.load_run("eval_v1_old.json")
        assert snap.aggregate["context_recall"] is None
        assert snap.aggregate["context_precision"] == pytest.approx(0.6)

    def test_broken_json_raises_valueerror(self, tmp_path: Path) -> None:
        from src.observability.dashboard.services.evaluation_report_service import (
            EvaluationReportService,
        )
        (tmp_path / "eval_bad.json").write_text("{not json", encoding="utf-8")
        svc = EvaluationReportService(logs_dir=tmp_path)
        with pytest.raises(ValueError):
            svc.load_run("eval_bad.json")

    def test_unknown_run_raises_keyerror(self, tmp_path: Path) -> None:
        from src.observability.dashboard.services.evaluation_report_service import (
            EvaluationReportService,
        )
        svc = EvaluationReportService(logs_dir=tmp_path)
        with pytest.raises(KeyError):
            svc.load_run("nope.json")


class TestHistoryEntries:
    def test_history_line_to_snapshot(self, tmp_path: Path) -> None:
        from src.observability.dashboard.services.evaluation_report_service import (
            EvaluationReportService,
        )
        entry = {
            "timestamp": "2026-08-16 20:40:44",
            "aggregate_metrics": {
                "context_precision": 0.5, "context_recall": 0.25, "context_relevance": 1.0,
                "source_precision_at_k": 1.0, "source_recall_at_k": 1.0,
            },
        }
        (tmp_path / "eval_history.jsonl").write_text(
            json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        svc = EvaluationReportService(logs_dir=tmp_path)
        runs = svc.load_history_entries()
        assert len(runs) == 1
        assert runs[0].aggregate["context_recall"] == pytest.approx(0.25)
        assert runs[0].batches == []          # history 行无 batches
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/unit/test_evaluation_report_service.py -v`
Expected: FAIL — `ModuleNotFoundError: … evaluation_report_service`

- [ ] **Step 3: 实现 EvaluationReportService**

```python
"""EvaluationReportService – read-only parsing of eval report files.

Reads v2 standalone reports (``logs/eval_*.json``: run/date/n_qa/batches/
weighted_aggregate) and CLI history (``logs/eval_history.jsonl``, one
report dict per line). Produces ``EvalRunSnapshot`` dataclasses; pages do
zero file IO. v1 reports missing ``context_recall`` surface as None.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import resolve_path

logger = logging.getLogger(__name__)

HISTORY_RUN_ID = "__history__"
METRIC_KEYS = (
    "source_recall_at_k",
    "source_precision_at_k",
    "context_relevance",
    "context_precision",
    "context_recall",
)


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    description: str


METRIC_SPECS: List[MetricSpec] = [
    MetricSpec("source_recall_at_k", "source_recall@5", "top-5 里有没有命中正确论文"),
    MetricSpec("source_precision_at_k", "source_precision@5", "top-5 里命中正确论文的比例"),
    MetricSpec("context_relevance", "context_relevance", "检索回的上下文和问题相关吗"),
    MetricSpec("context_precision", "context_precision", "相关 chunk 是否排在前面"),
    MetricSpec("context_recall", "context_recall", "该召回的信息,检索结果覆盖了吗"),
]


@dataclass
class EvalRunSnapshot:
    run_id: str
    label: str
    date: str = ""
    n_qa: Optional[int] = None
    batches: List[Dict[str, Optional[float]]] = field(default_factory=list)
    aggregate: Dict[str, Optional[float]] = field(default_factory=dict)


def _fill_metrics(raw: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Map a metrics dict onto the five fixed keys; missing → None."""
    raw = raw or {}
    return {k: (float(raw[k]) if raw.get(k) is not None else None) for k in METRIC_KEYS}


class EvaluationReportService:
    """Read-only accessor for evaluation reports under ``logs/``."""

    def __init__(self, logs_dir: Optional[Path] = None) -> None:
        self._logs_dir = Path(logs_dir) if logs_dir else resolve_path("logs")

    def list_runs(self) -> List[str]:
        """Standalone reports newest-first; history file (if any) last."""
        runs: List[str] = [
            p.name for p in sorted(
                self._logs_dir.glob("eval_*.json"), key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        ]
        if (self._logs_dir / "eval_history.jsonl").exists():
            runs.append(HISTORY_RUN_ID)
        return runs

    def load_run(self, run_id: str) -> EvalRunSnapshot:
        """Load one standalone report by filename.

        Raises:
            KeyError: Unknown run_id.
            ValueError: File is not valid JSON.
        """
        path = self._logs_dir / run_id
        if run_id == HISTORY_RUN_ID or not path.exists():
            raise KeyError(f"Unknown run: {run_id}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupted report {run_id}: {exc}") from exc

        return EvalRunSnapshot(
            run_id=str(data.get("run", run_id)),
            label=f"{data.get('run', run_id)} · {data.get('date', '')}",
            date=str(data.get("date", "")),
            n_qa=data.get("n_qa"),
            batches=[_fill_metrics(b) for b in data.get("batches", [])],
            aggregate=_fill_metrics(data.get("weighted_aggregate")),
        )

    def load_history_entries(self) -> List[EvalRunSnapshot]:
        """One snapshot per jsonl line; empty list when file absent."""
        path = self._logs_dir / "eval_history.jsonl"
        if not path.exists():
            return []
        snapshots: List[EvalRunSnapshot] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping corrupted history line: %.60s", line)
                continue
            snapshots.append(EvalRunSnapshot(
                run_id=str(data.get("timestamp", "history")),
                label=str(data.get("timestamp", "history")),
                date=str(data.get("timestamp", ""))[:10],
                n_qa=data.get("query_count"),
                batches=[_fill_metrics(qr.get("metrics")) for qr in data.get("query_results", [])],
                aggregate=_fill_metrics(data.get("aggregate_metrics")),
            ))
        return snapshots
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/test_evaluation_report_service.py -v`
Expected: 全 PASS

- [ ] **Step 5: Lint + 提交**

```bash
ruff check src/observability/dashboard
mypy src
git add src/observability/dashboard/services/evaluation_report_service.py tests/unit/test_evaluation_report_service.py
git commit -m "feat(dashboard): EvaluationReportService——v2/v1 报告只读解析"
```

---

### Task 4: Retrieval Playground 页面

**Files:**
- Create: `src/observability/dashboard/pages/retrieval_playground.py`
- Modify: `src/observability/dashboard/app.py:49-56`(导航注册)
- Test: `tests/e2e/test_dashboard_smoke.py`(追加一个测试)

**Interfaces:**
- Consumes: Task 1 的 `render_chunk_list(chunks, *, prefix, highlight)`;Task 2 的 `RetrievalService` / `PlaygroundResult`(字段见 Task 2 Produces)。
- Produces: `render() -> None`(页面入口,与其它页面同构);导航顺序 Overview → Data Browser → Retrieval Playground → Ingestion Manager → Ingestion Traces → Query Traces → Evaluation Panel。

- [ ] **Step 1: 写失败测试(追加到 tests/e2e/test_dashboard_smoke.py)**

```python
    # ------------------------------------------------------------------
    # 7. Retrieval Playground page
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_retrieval_playground_page_renders(self) -> None:
        """Playground renders with mocked service (empty result is OK)."""
        from streamlit.testing.v1 import AppTest

        mock_svc = MagicMock()
        mock_svc.list_collections.return_value = ["papers"]

        def page_script():
            from src.observability.dashboard.pages.retrieval_playground import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)

        with patch(
            "src.observability.dashboard.pages.retrieval_playground.RetrievalService",
            return_value=mock_svc,
        ):
            at.run()

        assert not at.exception, f"Playground raised: {at.exception}"
        text = _collect_text(at)
        assert "playground" in text.lower() or "retrieval" in text.lower()
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/e2e/test_dashboard_smoke.py::TestDashboardSmoke::test_retrieval_playground_page_renders -v`
Expected: FAIL — `ModuleNotFoundError: … retrieval_playground`

- [ ] **Step 3: 实现页面**

`src/observability/dashboard/pages/retrieval_playground.py`:
```python
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
        "输入一条查询,立即查看 **Dense / Sparse / RRF Fusion** 三路检索结果对比。"
        "🟢 = 两路都召回(交集),⚪ = 仅单路召回——一眼看出 RRF 是否救回了某路漏掉的 chunk。"
        "不做 LLM 生成,专注检索效果检视。"
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
        st.warning(f"⚠️ 单路降级运行(used_fallback=True)。")
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
```

- [ ] **Step 4: 注册导航(app.py)**

修改 `src/observability/dashboard/app.py`——`_page_overview` 定义后、`_page_ingestion_manager` 前加:

```python
def _page_retrieval_playground() -> None:
    from src.observability.dashboard.pages.retrieval_playground import render
    render()
```

`pages` 列表改为(Playground 插在 Data Browser 之后):

```python
pages = [
    st.Page(_page_overview, title="Overview", icon="📊", default=True),
    st.Page(_page_data_browser, title="Data Browser", icon="🔍"),
    st.Page(_page_retrieval_playground, title="Retrieval Playground", icon="🔬"),
    st.Page(_page_ingestion_manager, title="Ingestion Manager", icon="📥"),
    st.Page(_page_ingestion_traces, title="Ingestion Traces", icon="🔬"),
    st.Page(_page_query_traces, title="Query Traces", icon="🔎"),
    st.Page(_page_evaluation_panel, title="Evaluation Panel", icon="📏"),
]
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/e2e/test_dashboard_smoke.py tests/unit/test_retrieval_service.py -v`
Expected: 全 PASS

- [ ] **Step 6: 手动验收(spec 验收标准 1/2)**

```bash
python scripts/start_dashboard.py
```
浏览器打开 Playground 页,用 golden set 第一条查询(`source:New-record-in-the-number-of-qubits-for-a-quantum-implementation-of-AES.pdf How do the AES qubit counts…`,或去掉 source 前缀直接查关键词)验证:<10s 出三列对比、chunk 可展开看全文、换 top-k=20 重跑即时刷新。完成后 Ctrl+C 关掉。

- [ ] **Step 7: Lint + 提交**

```bash
ruff check src/observability/dashboard
mypy src
git add src/observability/dashboard/pages/retrieval_playground.py src/observability/dashboard/app.py tests/e2e/test_dashboard_smoke.py
git commit -m "feat(dashboard): Retrieval Playground 页——三路对比+交集高亮+分词展示"
```

---

### Task 5: Evaluation Panel 重写

**Files:**
- Rewrite: `src/observability/dashboard/pages/evaluation_panel.py`(整文件替换,454 行 → 约 200 行)
- Replace: `tests/unit/test_evaluation_panel.py`(整文件替换——旧测试测的 `_save_to_history/_load_history` 已删除)
- Modify: `tests/e2e/test_dashboard_smoke.py`(Evaluation Panel 冒烟测试加 service mock)

**Interfaces:**
- Consumes: Task 3 的 `EvaluationReportService` / `EvalRunSnapshot` / `METRIC_SPECS` / `HISTORY_RUN_ID`。
- Produces: `render() -> None`(签名不变,app.py 无需改动)。

- [ ] **Step 1: 替换单元测试**

`tests/unit/test_evaluation_panel.py` 整文件替换为:

```python
"""Unit tests for the rewritten Evaluation Panel (report viewer)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


class TestEvaluationPanelRendering:
    """AppTest-level checks with a mocked report service."""

    @pytest.fixture
    def mock_svc(self) -> MagicMock:
        from src.observability.dashboard.services.evaluation_report_service import EvalRunSnapshot

        svc = MagicMock()
        svc.list_runs.return_value = ["eval_v2_baseline.json"]
        svc.load_run.return_value = EvalRunSnapshot(
            run_id="v2-baseline",
            label="v2-baseline · 2026-08-16",
            date="2026-08-16",
            n_qa=23,
            batches=[
                {"source_recall_at_k": 1.0, "source_precision_at_k": 0.7,
                 "context_relevance": 1.0, "context_precision": 0.75, "context_recall": 0.67},
            ],
            aggregate={
                "source_recall_at_k": 0.96, "source_precision_at_k": 0.71,
                "context_relevance": 0.87, "context_precision": 0.65, "context_recall": 0.49,
            },
        )
        svc.load_history_entries.return_value = []
        return svc

    def _render(self, mock_svc: MagicMock) -> Any:
        from streamlit.testing.v1 import AppTest

        def page_script():
            from src.observability.dashboard.pages.evaluation_panel import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)
        with patch(
            "src.observability.dashboard.pages.evaluation_panel.EvaluationReportService",
            return_value=mock_svc,
        ):
            at.run()
        return at

    def test_renders_five_metric_cards(self, mock_svc: MagicMock) -> None:
        at = self._render(mock_svc)
        assert not at.exception
        # 五个指标卡都是 st.metric → 数量 >= 5,且文字包含 context_recall
        metrics_text = "\n".join(str(getattr(m, "label", "") or getattr(m, "value", ""))
                                  for m in at.metric)
        assert "context_recall" in metrics_text or len(at.metric) >= 5

    def test_renders_batch_table(self, mock_svc: MagicMock) -> None:
        at = self._render(mock_svc)
        assert not at.exception
        assert len(at.dataframe) >= 1

    def test_no_runs_shows_guidance(self) -> None:
        from streamlit.testing.v1 import AppTest

        svc = MagicMock()
        svc.list_runs.return_value = []

        def page_script():
            from src.observability.dashboard.pages.evaluation_panel import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)
        with patch(
            "src.observability.dashboard.pages.evaluation_panel.EvaluationReportService",
            return_value=svc,
        ):
            at.run()
        assert not at.exception
        infos = "\n".join(str(getattr(i, "value", "")) for i in at.info)
        assert "evaluate.py" in infos or "评估" in infos

    def test_broken_report_shows_error(self) -> None:
        from streamlit.testing.v1 import AppTest

        svc = MagicMock()
        svc.list_runs.return_value = ["eval_bad.json"]
        svc.load_run.side_effect = ValueError("Corrupted report")

        def page_script():
            from src.observability.dashboard.pages.evaluation_panel import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)
        with patch(
            "src.observability.dashboard.pages.evaluation_panel.EvaluationReportService",
            return_value=svc,
        ):
            at.run()
        assert not at.exception
        errors = "\n".join(str(getattr(e, "value", "")) for e in at.error)
        assert errors  # 有 error banner 且页面未崩
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/unit/test_evaluation_panel.py -v`
Expected: FAIL — 旧模块没有 `EvaluationReportService` 引用/新测试 import 路径不存在(`patch` 目标找不到)

- [ ] **Step 3: 重写页面**

`src/observability/dashboard/pages/evaluation_panel.py` 整文件替换为:

```python
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
```

依赖说明:`pandas` 已是 streamlit 依赖,无需新增。

- [ ] **Step 4: 更新 e2e 冒烟(test_dashboard_smoke.py 的 Evaluation Panel 测试)**

将现有 `test_evaluation_panel_page_renders` 整个方法替换为:

```python
    @pytest.mark.e2e
    def test_evaluation_panel_page_renders(self) -> None:
        """Evaluation Panel loads (no-report guidance is OK)."""
        from streamlit.testing.v1 import AppTest

        mock_svc = MagicMock()
        mock_svc.list_runs.return_value = []

        def page_script():
            from src.observability.dashboard.pages.evaluation_panel import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)

        with patch(
            "src.observability.dashboard.pages.evaluation_panel.EvaluationReportService",
            return_value=mock_svc,
        ):
            at.run()

        assert not at.exception, (
            f"Evaluation Panel page raised an exception: {at.exception}"
        )
        text = _collect_text(at)
        assert "evaluation" in text.lower() or "panel" in text.lower()
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/unit/test_evaluation_panel.py tests/e2e/test_dashboard_smoke.py tests/unit/test_evaluation_report_service.py -v`
Expected: 全 PASS

- [ ] **Step 6: 手动验收(spec 验收标准 3/4)**

```bash
python scripts/start_dashboard.py
```
Evaluation Panel 页:选 `eval_v2_baseline_deepseek.json` → 五指标卡 + 12 行 batches 表 + 低 context_recall 批次警告;v1 兼容(旧 jsonl 历史条目缺指标显示 "—")。Ctrl+C 关掉。

- [ ] **Step 7: Lint + 提交**

```bash
ruff check src/observability/dashboard
mypy src
git add src/observability/dashboard/pages/evaluation_panel.py tests/unit/test_evaluation_panel.py tests/e2e/test_dashboard_smoke.py
git commit -m "feat(dashboard): Evaluation Panel 重写——v2 五指标只读报告视图"
```

---

### Task 6: 全量回归 + 收尾

**Files:**
- Modify: `README.md`(功能清单处提一句 Playground,如 README 有 dashboard 功能列表)
- 无新测试

**Interfaces:**
- Consumes: Task 1-5 全部产出。
- Produces: 可交付分支。

- [ ] **Step 1: 全量测试**

```bash
pytest -m "not llm" -q
```
Expected: 全 PASS(与 main 基线对比无新增失败;`tests/unit/test_evaluation_panel.py` 已在 Task 5 替换)

- [ ] **Step 2: Lint + 类型 + 编译三连**

```bash
ruff check .
mypy src
python -m compileall src -q
```
Expected: 三条全干净

- [ ] **Step 3: README 增补**

在 README.md 的 dashboard/管理界面功能列表处(搜 `Streamlit` 或 `dashboard` 定位)追加一行,与现有条目格式一致:

```markdown
- **Retrieval Playground** — 当场查询并三路对比 Dense / Sparse / RRF 结果(交集高亮),检索效果检视工作台
```

同时把 Evaluation Panel 的描述(若有)更新为"v2 五指标报告只读视图"。**注意:README 是中文,保持风格一致;只加/改这两处,不动其他内容。**

- [ ] **Step 4: 提交**

```bash
git add README.md
git commit -m "docs(readme): dashboard 新增 Retrieval Playground,评估面板对齐 v2"
```

- [ ] **Step 5: 全部验收标准逐条确认(对照 spec 第 7 节)**

1. ✅ Playground 真实查询 <10s 出三列对比(Task 4 Step 6 已验)
2. ✅ 换 top-k=20 重跑即时刷新(Task 4 Step 6 已验)
3. ✅ Evaluation Panel 渲染 v2 基线报告(Task 5 Step 6 已验)
4. ✅ v1 旧报告不报错(Task 5 Step 6 已验)
5. ✅ 全测 + ruff + mypy 干净(本任务 Step 1/2)

**完成后停下:按用户规则,向用户展示 PR 描述(标题/动机/变更/测试证据),确认后才 push 分支并开 PR。不直接 push。**
