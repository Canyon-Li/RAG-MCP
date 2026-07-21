# tags 元数据过滤修复（G4）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让用户在 query 里写 `tag:azure 架构图` 能正确过滤，只返回带匹配标签的 chunk（修复 G4 三层故障）。

**Architecture:** tags 是 list 语义，Chroma 标量 metadata 处理不了。方案集中在 query 层（`hybrid_search.py`）：① `search()` 传 retrievers 前从 filters 剥离 tags（避免 Chroma where 杀零），tags 只走 post-fusion；② `_matches_filters` tags 分支改读 Chroma 落盘后的逗号字符串。存储层零改动。

**Tech Stack:** Python 3.10+，pytest（markers: unit/integration），conda env `langchain-test`。

## Global Constraints

- **运行环境**：所有 Python/pytest 命令走 conda env `langchain-test`，用 `conda run -n langchain-test python -m pytest ...`。**不要**建 `.venv（memory: runtime-env-conda）。
- **存储层禁区**：`src/libs/vector_store/chroma_store.py` **不得修改**——`_sanitize_metadata` 逗号拼接行为保留（向后兼容已摄取数据）。tags 存储表示保持逗号字符串。
- **MCP schema 禁区**：`src/mcp_server/tools/query_knowledge_hub.py` 不得修改（不加显式 filters 参数）。
- **范围**：只改 `src/core/query_engine/hybrid_search.py` 一个文件的两处。不修 G5/G6/G7。
- **TDD 纪律**：每个任务先写失败测试（RED）→ 跑确认失败 → 最小实现 → 跑确认通过（GREEN）→ commit。
- **测试落点**：两个任务的新测试都加到 `tests/integration/test_hybrid_search.py` 的 `TestHybridSearchFilters` 类（fixtures `query_processor`/`rrf_fusion` 和 Mock retriever 已在那里定义；spec §7.1 提到 "tests/unit 或新建"，本计划选择沿用集成测试文件以复用现有 fixtures——这是计划级细化）。

---

### Task 1: 修 `_matches_filters` tags 分支读逗号字符串（post-fusion 读端）

**Files:**
- Modify: `src/core/query_engine/hybrid_search.py:730-736`（tags 分支）
- Test: `tests/integration/test_hybrid_search.py`（`TestHybridSearchFilters` 类内新增测试类）

**Interfaces:**
- Consumes: 无（首个任务）
- Produces: `_matches_filters` 的 tags 分支能正确读取 Chroma 落盘后的逗号字符串形态，Task 2 的端到端测试依赖此修复生效。

- [ ] **Step 1: 写失败测试——在 `TestHybridSearchFilters` 类后面新增测试类**

在 `tests/integration/test_hybrid_search.py` 文件末尾（`TestHybridSearchFilters` 类之后）追加：

```python
class TestTagsFilterReadCommaString:
    """G4: _matches_filters tags 分支必须正确读 Chroma 落盘后的逗号字符串。

    Chroma 的 _sanitize_metadata 把 tags list 逗号拼接成字符串（保留不改），
    所以 post-fusion 过滤必须 split 逗号字符串再求交集。
    """

    def _make_hybrid(self, query_processor, rrf_fusion):
        """构造一个最小 HybridSearch 实例（mock retriever，供直接调 _matches_filters）。"""
        return HybridSearch(
            query_processor=query_processor,
            dense_retriever=MockDenseRetriever(results=[]),
            sparse_retriever=MockSparseRetriever(results=[]),
            fusion=rrf_fusion,
        )

    def test_tags_comma_string_match(self, query_processor, rrf_fusion):
        """逗号字符串形态，命中。"""
        hybrid = self._make_hybrid(query_processor, rrf_fusion)
        assert hybrid._matches_filters(
            {"tags": "azure,cloud"}, {"tags": ["azure"]}
        ) is True

    def test_tags_comma_string_no_match(self, query_processor, rrf_fusion):
        """逗号字符串形态，不命中。"""
        hybrid = self._make_hybrid(query_processor, rrf_fusion)
        assert hybrid._matches_filters(
            {"tags": "azure,cloud"}, {"tags": ["aws"]}
        ) is False

    def test_tags_empty_string(self, query_processor, rrf_fusion):
        """空字符串（chunk 无标签）→ 不命中。"""
        hybrid = self._make_hybrid(query_processor, rrf_fusion)
        assert hybrid._matches_filters(
            {"tags": ""}, {"tags": ["azure"]}
        ) is False

    def test_tags_strips_whitespace(self, query_processor, rrf_fusion):
        """标签带空格 → strip 后命中。"""
        hybrid = self._make_hybrid(query_processor, rrf_fusion)
        assert hybrid._matches_filters(
            {"tags": " azure , cloud "}, {"tags": ["cloud"]}
        ) is True

    def test_tags_list_form_defensive(self, query_processor, rrf_fusion):
        """防御性兼容 list 形态（未过 sanitize 的场景，如测试直构）。"""
        hybrid = self._make_hybrid(query_processor, rrf_fusion)
        assert hybrid._matches_filters(
            {"tags": ["azure", "cloud"]}, {"tags": ["azure"]}
        ) is True
```

- [ ] **Step 2: 跑测试确认失败（RED）**

Run:
```bash
conda run -n langchain-test python -m pytest tests/integration/test_hybrid_search.py::TestTagsFilterReadCommaString -v
```
Expected: **FAIL**。重点看 `test_tags_comma_string_match` 和 `test_tags_strips_whitespace`：
- 当前代码 `set("azure,cloud")` 得字符级集合 `{'a','z','u',...}`，与 `set(["azure"])`（含字符串元素）求交为空 → `not ∅` → 返回 `False` → 断言 `is True` 失败。
- `test_tags_comma_string_no_match` / `test_tags_empty_string` 碰巧返回 `False`（与期望一致），属 GREEN，但保留作回归保护。
- `test_tags_list_form_defensive`：list 形态当前代码能正确处理，GREEN。

> 至少 2 个用例 FAIL 即确认 RED（读端语义错误已暴露）。

- [ ] **Step 3: 修 `_matches_filters` tags 分支**

修改 `src/core/query_engine/hybrid_search.py`，定位 tags 分支（约 730-736 行），替换为：

```python
            elif key == "tags":
                # tags 经 Chroma _sanitize_metadata 落盘后是逗号字符串；
                # 防御性兼容 list 形态（未过 sanitize 的场景，如单元测试直构）。
                meta_tags = metadata.get("tags", "")
                if isinstance(meta_tags, str):
                    meta_tags = [t.strip() for t in meta_tags.split(",") if t.strip()]
                elif not isinstance(meta_tags, list):
                    meta_tags = []
                if not isinstance(value, list):
                    value = [value]
                if not set(meta_tags) & set(value):
                    return False
```

改动要点（对照原代码）：
- `metadata.get("tags", [])` → `metadata.get("tags", "")`（默认空字符串，与 Chroma 落盘形态一致）。
- 新增 `isinstance(meta_tags, str)` 分支：`split(",")` + `strip()` + 过滤空段。
- 新增 `elif not isinstance(meta_tags, list)` 防御：非 str 非 list → 空列表。
- `value` 规范化和 set 求交逻辑保留不变。

- [ ] **Step 4: 跑测试确认通过（GREEN）**

Run:
```bash
conda run -n langchain-test python -m pytest tests/integration/test_hybrid_search.py::TestTagsFilterReadCommaString -v
```
Expected: **5 passed**。

- [ ] **Step 5: 跑回归确认未破坏现有过滤测试**

Run:
```bash
conda run -n langchain-test python -m pytest tests/integration/test_hybrid_search.py::TestHybridSearchFilters -v
```
Expected: 现有过滤测试全 PASS（collection/doc_type 过滤不受影响）。

- [ ] **Step 6: Commit**

```bash
git add src/core/query_engine/hybrid_search.py tests/integration/test_hybrid_search.py
git commit -m "fix(query): _matches_filters tags 分支读逗号字符串 (G4 post-fusion)

Chroma 落盘后 tags 是逗号字符串，原代码 set(字符串) 按字符级拆导致
语义错误。改为 split(',') + strip() 再求交集，防御性兼容 list 形态。"
```

---

### Task 2: `search()` 剥离 tags 走 post-fusion（pre-fusion 杀零修复）

**Files:**
- Modify: `src/core/query_engine/hybrid_search.py:252-260`（`search()` 方法内，传 `_run_retrievals` 前）
- Test: `tests/integration/test_hybrid_search.py`（`TestTagsFilterReadCommaString` 类后新增）

**Interfaces:**
- Consumes: Task 1 修好的 `_matches_filters` tags 读端（post-fusion 过滤要靠它正确读逗号字符串）。
- Produces: `search()` 的 retrievers 不再收到 tags filter，tags 只在 post-fusion 生效。完整 `tag:xxx` 内联语法端到端可用。

**前置确认**：Task 1 已合并/通过，`_matches_filters` 能正确读逗号字符串。

- [ ] **Step 1: 写失败测试——端到端 `tag:xxx` 过滤**

在 `tests/integration/test_hybrid_search.py` 的 `TestTagsFilterReadCommaString` 类之后追加新测试类：

```python
class TestTagsFilterPreFusionStrip:
    """G4: tag:xxx 查询必须把 tags 从 retrieval filters 剥离（只走 post-fusion）。

    否则 tags 进 Chroma where → list 被当 $in → 匹配逗号字符串失败 → dense 杀零。
    """

    def test_tag_query_strips_pre_fusion_and_filters_post_fusion(
        self, query_processor, rrf_fusion,
    ):
        """tag:azure 查询：retrievers 不收到 tags，且 post-fusion 过滤到 azure 标签。"""
        # 模拟 Chroma 落盘后的逗号字符串 tags
        results_with_tags = [
            RetrievalResult(
                chunk_id="a", score=0.9, text="Azure 架构说明",
                metadata={"tags": "azure,cloud"},
            ),
            RetrievalResult(
                chunk_id="b", score=0.85, text="AWS 架构说明",
                metadata={"tags": "aws"},
            ),
            RetrievalResult(
                chunk_id="c", score=0.8, text="通用说明",
                metadata={"tags": ""},
            ),
        ]
        dense = MockDenseRetriever(results=results_with_tags)
        sparse = MockSparseRetriever(results=results_with_tags)
        hybrid = HybridSearch(
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=rrf_fusion,
        )

        results = hybrid.search("tag:azure 架构", top_k=10)

        # (a) tags 被剥离，不进 retrieval filters（pre-fusion 杀零避免）
        assert "tags" not in (dense.last_filters or {}), \
            "tags 不应进入 retrieval filters，否则 Chroma where 杀零"
        # (b) post-fusion 过滤：只返回带 azure 标签的 chunk
        assert [r.chunk_id for r in results] == ["a"], \
            "post-fusion 应过滤到 azure 标签的 chunk"
```

- [ ] **Step 2: 跑测试确认失败（RED）**

Run:
```bash
conda run -n langchain-test python -m pytest tests/integration/test_hybrid_search.py::TestTagsFilterPreFusionStrip -v
```
Expected: **FAIL**，断言 (a) 失败：
- 当前代码 `search()` 把 `merged_filters`（含 `tags`）原样传给 `_run_retrievals` → `dense.last_filters == {"tags": ["azure"]}` → `"tags" not in {...}` 为 False → 断言失败。
- Task 1 已修好 post-fusion 读端，但断言 (a) 先失败，测试整体 RED。

> 如果断言 (b) 先失败（返回结果不是 `["a"]`），说明 Task 1 未生效——回到 Task 1 确认。

- [ ] **Step 3: 在 `search()` 加 tags 剥离**

修改 `src/core/query_engine/hybrid_search.py` 的 `search()` 方法。定位到 `merged_filters = self._merge_filters(...)` 之后、`self._run_retrievals(...)` 调用之前（约 252-260 行）。当前代码：

```python
        # Merge explicit filters with query-extracted filters
        merged_filters = self._merge_filters(processed_query.filters, filters)
        
        # Step 2: Run retrievals
        dense_results, sparse_results, dense_error, sparse_error = self._run_retrievals(
            processed_query=processed_query,
            filters=merged_filters,
            trace=trace,
        )
```

改为（新增 `retrieval_filters` 剥离 tags）：

```python
        # Merge explicit filters with query-extracted filters
        merged_filters = self._merge_filters(processed_query.filters, filters)
        
        # tags 是 list 语义，Chroma where 处理不了（list 被当 $in，匹配逗号字符串
        # 必然失败 → 杀零）。tags 只走 post-fusion（Step 5 用 merged_filters）。
        retrieval_filters = {
            k: v for k, v in merged_filters.items() if k != "tags"
        }
        
        # Step 2: Run retrievals
        dense_results, sparse_results, dense_error, sparse_error = self._run_retrievals(
            processed_query=processed_query,
            filters=retrieval_filters,
            trace=trace,
        )
```

改动要点：
- 新增 3 行 `retrieval_filters` 字典推导（剥离 `tags` key）。
- `_run_retrievals` 的 `filters=merged_filters` → `filters=retrieval_filters`。
- post-fusion（约 293 行 `if merged_filters and self.config.metadata_filter_post:`）**不动**，继续用 `merged_filters`（含 tags）。

- [ ] **Step 4: 跑测试确认通过（GREEN）**

Run:
```bash
conda run -n langchain-test python -m pytest tests/integration/test_hybrid_search.py::TestTagsFilterPreFusionStrip -v
```
Expected: **1 passed**。断言 (a) tags 被剥离 + 断言 (b) post-fusion 过滤到 `["a"]` 都通过。

- [ ] **Step 5: 跑全量 hybrid_search 回归**

Run:
```bash
conda run -n langchain-test python -m pytest tests/integration/test_hybrid_search.py -v
```
Expected: 全部 PASS（含 Task 1 的 5 用例 + Task 2 的 1 用例 + 现有所有测试）。重点确认：
- `TestHybridSearchFilters::test_explicit_filters_passed_to_retrievers` 仍 PASS（标量 filter collection 仍正常传给 retriever，不受 tags 剥离影响）。
- `TestHybridSearchFilters::test_post_fusion_metadata_filter` 仍 PASS。

- [ ] **Step 6: 跑 chroma_store roundtrip + query_processor 回归**

确认存储层和解析层无回归（这两处本任务没改，但 tags 相关断言要稳）：

Run:
```bash
conda run -n langchain-test python -m pytest tests/integration/test_chroma_store_roundtrip.py tests/unit/test_query_processor.py -v
```
Expected: 全 PASS。重点确认：
- `test_chroma_store_roundtrip.py` 里 `tags == 'tag1,tag2,tag3'` 断言仍通过（存储行为未变）。
- `test_query_processor.py` 里 `tag:xxx` 解析用例仍通过（解析层未动）。

- [ ] **Step 7: Commit**

```bash
git add src/core/query_engine/hybrid_search.py tests/integration/test_hybrid_search.py
git commit -m "fix(query): search() 剥离 tags 走 post-fusion (G4 pre-fusion 杀零)

tags 是 list 语义，Chroma where 把 list 当 \$in 匹配逗号字符串必失败 →
dense 杀零。search() 传 retrievers 前剥离 tags，tags 只在 post-fusion 生效。
标量 filter（collection/doc_type/source_path）不受影响。"
```

---

## Self-Review

（计划作者自查，执行前完成）

**1. Spec coverage：**
- §4.1（search 剥离 tags）→ Task 2 Step 3 ✓
- §4.2（_matches_filters 读逗号字符串）→ Task 1 Step 3 ✓
- §6 边界处理（空字符串/空格/list 形态/多标签）→ Task 1 的 5 个用例覆盖前 3 项；多标签 OR 语义由现有 set 求交逻辑保证（未改）✓
- §7.1 单元测试 → Task 1 ✓
- §7.2 集成测试 → Task 2（用 mock retriever 替代真 Chroma 摄取，计划级细化，断言 (a) 证剥离、断言 (b) 证 post-fusion 过滤）✓
- §7.3 现有测试验证 → Task 1 Step 5 + Task 2 Step 5/6 ✓

**2. Placeholder scan：** 无 TODO/TBD；所有代码块完整；所有命令含 expected output ✓

**3. Type consistency：**
- `_matches_filters(metadata: Dict, filters: Dict) -> bool` —— Task 1 测试调用与签名一致 ✓
- `retrieval_filters` 仅在 Task 2 Step 3 `search()` 内引入并消费，无跨任务类型依赖 ✓
- `RetrievalResult(chunk_id, score, text, metadata)` —— Task 2 测试构造与现有 fixture 形态一致 ✓
- Mock retriever `last_filters` 属性 —— Task 2 断言与 MockDenseRetriever:51 定义一致 ✓
