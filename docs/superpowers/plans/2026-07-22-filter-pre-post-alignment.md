# Filter pre/post 双层语义对齐 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修 N1（`collection` filter 下推 Chroma `where` 杀零）+ N3（`source_path` Dense exact / post partial 不一致），让 metadata filter 在 pre（Dense `where`）+ post（`_matches_filters`）两层语义对齐。

**Architecture:** 扩展 `hybrid_search.search()` 的 pre-fusion 剥离集（从仅 `tags` 扩到 `{tags, collection, source_path}`），并把 `_matches_filters` 的 `collection` 分支从"比较 metadata"改成 `continue`（物理隔离已保证，放行）。`doc_type` / generic 仍走 Dense pre-fusion exact 下推 + post 兜底，语义已一致不动。所有改动集中在 `hybrid_search.py` 一个文件 + 测试文件。

**Tech Stack:** Python 3.10 / pytest / `src.core.query_engine.hybrid_search` / Mock retrievers（项目既有测试范式）

**Spec:** [docs/superpowers/specs/2026-07-22-filter-pre-post-alignment-design.md](../specs/2026-07-22-filter-pre-post-alignment-design.md)

## Global Constraints

- **环境**：测试在 conda env `langchain-test`。Bash 未激活该环境时，所有命令前缀为：
  `CONDA_NO_PLUGINS=true conda run -n langchain-test --no-capture-output <cmd>`
  （`conda run` 默认开启插件会崩；`CONDA_NO_PLUGINS=true` 是既有绕过方式。）
- **分支**：已切到 `fix/filter-pre-post-alignment`。
- **不改 QueryProcessor / ChromaStore / test_query_processor.py**（N4 拆单独议题，spec §7）。
- **注释中文 OK**（项目既有风格）；commit 用 conventional commits 中文（参考 `fix(query): ...`）。
- **测试位置**：追加到 `tests/integration/test_hybrid_search.py`（与 G4 的 `TestTagsFilter*` 同类，复用 `MockDenseRetriever` / `MockSparseRetriever` / fixtures）。

---

## File Structure

| 文件 | 职责 | 本次动作 |
|---|---|---|
| `src/core/query_engine/hybrid_search.py` | 查询编排 + filter 分流 + post-fusion 匹配 | 加 `POST_ONLY_FILTERS` 常量；`search()` 用它剥离；`_matches_filters` collection → `continue`；`HybridSearchConfig` 注释更新 |
| `tests/integration/test_hybrid_search.py` | HybridSearch 测试（Mock retriever） | 加 2 个 N1/N3 测试类 + 2 个锁定测试类；更新 2 个被语义变化破坏的现有测试 |

不新建文件。

---

## Task 1: pre-fusion 剥离集扩展 + collection 放行（N1 + N3 剥离）

**Files:**
- Modify: `src/core/query_engine/hybrid_search.py`（模块级常量 + `search()` :258-262 + `_matches_filters` :728-735 + `HybridSearchConfig` docstring :71-74）
- Modify: `tests/integration/test_hybrid_search.py`（追加 `TestCollectionFilterN1`；更新 `test_explicit_filters_passed_to_retrievers` :477-498、`test_post_fusion_metadata_filter` :524-549）

**Interfaces:**
- Produces（模块级常量，供测试与后续维护引用）：
  ```python
  POST_ONLY_FILTERS: set[str]  # {"tags", "collection", "source_path"}
  ```

- [ ] **Step 1：写 N1 新测试类（先失败）**

  追加到 `tests/integration/test_hybrid_search.py` 末尾（`TestTagsFilterPreFusionStrip` 之后）：

  ```python
  # =============================================================================
  # N1: collection filter 必须从 pre-fusion 剥离 + post-fusion 放行
  # =============================================================================

  class TestCollectionFilterN1:
      """N1: chunk metadata 不携带 collection（物理隔离维度），下推 Chroma where
      会杀零。collection 靠独立 Chroma collection + BM25 index 物理隔离保证，
      filter 层必须剥离 + 放行。
      """

      def test_collection_stripped_from_retrieval_filters(
          self, query_processor, rrf_fusion,
          sample_dense_results, sample_sparse_results,
      ):
          """collection:xxx 查询 → dense 不收到 collection（剥离），sparse 收到（选 index）。"""
          dense = MockDenseRetriever(results=sample_dense_results)
          sparse = MockSparseRetriever(results=sample_sparse_results)
          hybrid = HybridSearch(
              query_processor=query_processor,
              dense_retriever=dense, sparse_retriever=sparse,
              fusion=rrf_fusion,
          )
          hybrid.search("collection:api-docs Azure", top_k=5)
          # collection 被剥离出 retrieval filters（否则 Chroma where 杀零）
          assert "collection" not in (dense.last_filters or {}), \
              "collection 不应进入 retrieval filters（metadata 无此字段，下推杀零）"
          # sparse 仍用 collection 选 BM25 index（物理隔离维度）
          assert sparse.last_collection == "api-docs"

      def test_source_path_stripped_from_retrieval_filters(
          self, query_processor, rrf_fusion,
          sample_dense_results, sample_sparse_results,
      ):
          """N3: source:xxx 查询 → dense 不收到 source_path（剥离，避免 exact 杀零）。"""
          dense = MockDenseRetriever(results=sample_dense_results)
          sparse = MockSparseRetriever(results=sample_sparse_results)
          hybrid = HybridSearch(
              query_processor=query_processor,
              dense_retriever=dense, sparse_retriever=sparse,
              fusion=rrf_fusion,
          )
          hybrid.search("source:azure Azure", top_k=5)
          assert "source_path" not in (dense.last_filters or {}), \
              "source_path 不应进入 retrieval filters（用 partial，Chroma where exact 会杀零）"

      def test_collection_passthrough_when_metadata_absent(
          self, query_processor, rrf_fusion,
      ):
          """N1 核心回归：chunk metadata 无 collection 字段时，collection filter 不杀零。

          真实场景 metadata 不写 collection；修复前 post-fusion 读
          metadata['collection'] 得 None != 'api-docs' → 全排除 → 杀零。
          修复后 collection 放行（continue），结果正常返回。
          """
          results_no_collection = [
              RetrievalResult(
                  chunk_id="a", score=0.9, text="Azure 配置",
                  metadata={"source_path": "docs/azure.pdf"},  # 无 collection
              ),
              RetrievalResult(
                  chunk_id="b", score=0.85, text="OpenAI 指南",
                  metadata={"source_path": "docs/openai.pdf"},  # 无 collection
              ),
          ]
          dense = MockDenseRetriever(results=results_no_collection)
          sparse = MockSparseRetriever(results=results_no_collection)
          hybrid = HybridSearch(
              query_processor=query_processor,
              dense_retriever=dense, sparse_retriever=sparse,
              fusion=rrf_fusion,
          )
          results = hybrid.search(
              "Azure", top_k=10, filters={"collection": "api-docs"}
          )
          assert len(results) > 0, "collection filter 不应在 metadata 无该字段时杀零"
  ```

- [ ] **Step 2：跑新测试，确认失败**

  Run: `CONDA_NO_PLUGINS=true conda run -n langchain-test --no-capture-output pytest tests/integration/test_hybrid_search.py::TestCollectionFilterN1 -v`

  Expected: 3 个测试 FAIL。
  - `test_collection_stripped_from_retrieval_filters`：`assert "collection" not in ...` 失败（现状 collection 仍进 dense filters）。
  - `test_source_path_stripped_from_retrieval_filters`：同理 source_path 仍进。
  - `test_collection_passthrough_when_metadata_absent`：`len(results) > 0` 失败（现状 post-fusion 全排除 → 杀零）。

- [ ] **Step 3：更新被破坏的现有测试 `test_explicit_filters_passed_to_retrievers`**

  定位 `tests/integration/test_hybrid_search.py` 的 `TestHybridSearchFilters.test_explicit_filters_passed_to_retrievers`（约 :477-498）。该测试原断言 `dense.last_filters == {"collection": "api-docs"}`，与 N1 新语义（collection 剥离）冲突。

  将其断言部分替换为：

  ```python
          hybrid.search("Azure", top_k=5, filters={"collection": "api-docs"})

          # N1: collection 从 retrieval filters 剥离（metadata 不携带，下推杀零），
          # 但 sparse 仍用 collection 选 BM25 index。
          assert "collection" not in (dense.last_filters or {})
          assert sparse.last_collection == "api-docs"
  ```

  （删除原 `assert dense.last_filters == {"collection": "api-docs"}` 行。）

- [ ] **Step 4：替换 `test_post_fusion_metadata_filter` 为 doc_type 版本**

  定位 `TestHybridSearchFilters.test_post_fusion_metadata_filter`（约 :524-549）。原测试用 collection 做 post-filter，但 collection 现改为放行（N1），原断言（所有结果 collection==api-docs）不再成立。

  将整个方法替换为（用 `doc_type` —— chunk 必有的 metadata 字段，真正的 metadata filter）：

  ```python
      def test_post_fusion_metadata_filter_doc_type(
          self, query_processor, rrf_fusion,
      ):
          """post-fusion 按 doc_type 过滤（doc_type 是 chunk 必有 metadata 字段）。

          原 test_post_fusion_metadata_filter 用 collection，但 collection 已改
          为放行（N1），故改用 doc_type 验证 post-fusion 过滤仍有效。
          """
          results = [
              RetrievalResult(
                  chunk_id="a", score=0.9, text="PDF 文档内容",
                  metadata={"source_path": "a.pdf", "doc_type": "pdf"},
              ),
              RetrievalResult(
                  chunk_id="b", score=0.85, text="Word 文档内容",
                  metadata={"source_path": "b.docx", "doc_type": "docx"},
              ),
          ]
          dense = MockDenseRetriever(results=results)
          sparse = MockSparseRetriever(results=results)

          config = HybridSearchConfig(metadata_filter_post=True)
          hybrid = HybridSearch(
              query_processor=query_processor,
              dense_retriever=dense, sparse_retriever=sparse,
              fusion=rrf_fusion, config=config,
          )

          out = hybrid.search("文档", top_k=10, filters={"doc_type": "pdf"})

          assert [r.chunk_id for r in out] == ["a"], \
              "post-fusion 应过滤到 doc_type=pdf 的 chunk"
  ```

- [ ] **Step 5：跑被更新的现有测试，确认失败（实现尚未改）**

  Run: `CONDA_NO_PLUGINS=true conda run -n langchain-test --no-capture-output pytest tests/integration/test_hybrid_search.py::TestHybridSearchFilters -v`

  Expected: `test_explicit_filters_passed_to_retrievers` 和 `test_post_fusion_metadata_filter_doc_type` 都 FAIL（实现尚未改：collection 仍下推、doc_type 过滤行为尚未对齐——后者实际现状可能通过，但前两个 N1 测试必失败）。

  > 注：`test_post_fusion_metadata_filter_doc_type` 现状可能已通过（doc_type 本就 post exact），它主要用于替换原 collection 测试、锁定 doc_type 行为。

- [ ] **Step 6：实现 — 加 `POST_ONLY_FILTERS` 常量**

  在 `src/core/query_engine/hybrid_search.py` 的模块级（`HybridSearchConfig` 类定义**之前**，`logger = logging.getLogger(...)` 之后）添加：

  ```python
  # Metadata filter keys that must NOT be pushed down to Chroma ``where``
  # (pre-fusion). Each is handled only in post-fusion ``_matches_filters``:
  # - "tags":        list semantics; Chroma stores a comma-joined string,
  #                   ``where`` can't do intersection (D-021).
  # - "collection":  not a chunk metadata field (physical isolation via
  #                   separate Chroma collection + BM25 index); pushing down
  #                   kills all results (N1).
  # - "source_path": partial semantics; Chroma ``where`` defaults to exact (N3).
  POST_ONLY_FILTERS = {"tags", "collection", "source_path"}
  ```

- [ ] **Step 7：实现 — `search()` 用常量剥离**

  定位 `HybridSearch.search()` 内约 :258-262 的：

  ```python
          # tags 是 list 语义，Chroma where 处理不了（list 被当 $in，匹配逗号字符串
          # 必然失败 → 杀零）。tags 只走 post-fusion（Step 5 用 merged_filters）。
          retrieval_filters = {
              k: v for k, v in merged_filters.items() if k != "tags"
          }
  ```

  替换为：

  ```python
          # 这些 key 的语义 Chroma where 表达不了，只走 post-fusion（Step 5）：
          # 见 POST_ONLY_FILTERS 模块级注释。
          retrieval_filters = {
              k: v for k, v in merged_filters.items() if k not in POST_ONLY_FILTERS
          }
  ```

- [ ] **Step 8：实现 — `_matches_filters` collection 分支改 `continue`**

  定位 `HybridSearch._matches_filters()` 内约 :728-735 的 collection 分支：

  ```python
              if key == "collection":
                  # Collection might be in different metadata keys
                  meta_collection = (
                      metadata.get("collection")
                      or metadata.get("source_collection")
                  )
                  if meta_collection != value:
                      return False
  ```

  替换为：

  ```python
              if key == "collection":
                  # 物理隔离已保证（独立 Chroma collection + BM25 index），metadata
                  # 不携带 collection；放行，不检查。
                  # ⚠️ 必须用 continue，不能用 return True —— 后者会跳过后续
                  # doc_type / source_path 等其他 key 的检查，引入新 bug。
                  continue
  ```

- [ ] **Step 9：实现 — 更新 `HybridSearchConfig.metadata_filter_post` docstring**

  定位约 :71-74 的：

  ```python
          metadata_filter_post: Apply metadata filters after fusion (fallback).
              Note: tags filtering is post-fusion-only (Chroma can't do list-semantics
              on its comma-joined string), so tags silently no-op if this is False.
              Scalar filters (collection/doc_type/source_path) are unaffected.
  ```

  替换为：

  ```python
          metadata_filter_post: Apply metadata filters after fusion (fallback).
              Note: tags / collection / source_path filtering is post-fusion-only
              (see POST_ONLY_FILTERS), so these silently no-op if this is False.
              doc_type and generic (custom-field) filters still apply via Dense
              pre-fusion pushdown regardless of this flag.
  ```

- [ ] **Step 10：跑 N1 新测试 + 被更新的现有测试，确认全绿**

  Run: `CONDA_NO_PLUGINS=true conda run -n langchain-test --no-capture-output pytest tests/integration/test_hybrid_search.py::TestCollectionFilterN1 tests/integration/test_hybrid_search.py::TestHybridSearchFilters -v`

  Expected: 全部 PASS。

- [ ] **Step 11：跑整个 hybrid_search 测试文件，确认无回归**

  Run: `CONDA_NO_PLUGINS=true conda run -n langchain-test --no-capture-output pytest tests/integration/test_hybrid_search.py -v`

  Expected: 全部 PASS（包括既有 G4 `TestTagsFilter*`、basic / degradation / edge cases 等）。

- [ ] **Step 12：commit**

  ```bash
  git -C "d:/Desktop/Code/RAG-MCP/MODULAR-RAG-MCP-SERVER" add src/core/query_engine/hybrid_search.py tests/integration/test_hybrid_search.py
  git -C "d:/Desktop/Code/RAG-MCP/MODULAR-RAG-MCP-SERVER" commit -m "fix(query): collection/source_path 走 post-fusion，修 N1 杀零 + N3 不一致" -m "- POST_ONLY_FILTERS = {tags, collection, source_path}，search() 剥离后只下推 doc_type/generic
  - _matches_filters collection 分支改 continue（物理隔离已保证，放行不检查）
  - HybridSearchConfig 注释更新
  - 更新 test_explicit_filters_passed_to_retrievers / test_post_fusion_metadata_filter 以匹配新语义
  - 新增 TestCollectionFilterN1（剥离 + 放行不杀零回归）"
  ```

---

## Task 2: source_path partial + generic 缺字段排除的锁定测试（N3-post + ②A）

**Files:**
- Modify: `tests/integration/test_hybrid_search.py`（追加 `TestSourcePathPartialN3`、`TestGenericMissingFieldExclude`）

**说明**：这两个语义现状已正确（source_path post partial [:751] + generic else [:757] 的 `metadata.get(key) != value` 缺字段即排除），本 Task 只加 characterization 测试锁定行为、防回归，**不改产品代码**。

**Interfaces:**
- Consumes: Task 1 产出的 `POST_ONLY_FILTERS`（已含 `source_path`，确保 Dense 不下推）。

- [ ] **Step 1：追加 `TestSourcePathPartialN3`**

  在 `tests/integration/test_hybrid_search.py` 末尾（`TestCollectionFilterN1` 之后）追加：

  ```python
  # =============================================================================
  # N3: source_path 用 partial 语义（post-fusion 子串匹配）
  # =============================================================================

  class TestSourcePathPartialN3:
      """N3: source_path 两层一致 —— pre-fusion 剥离（Dense 不下推 exact）+
      post-fusion partial 子串匹配。锁定 partial 行为，防回归为 exact。
      """

      def test_source_path_partial_match(self, query_processor, rrf_fusion):
          """post-fusion source_path partial：子串命中。"""
          results = [
              RetrievalResult(
                  chunk_id="a", score=0.9, text="Azure 配置",
                  metadata={"source_path": "docs/azure-setup.pdf"},
              ),
              RetrievalResult(
                  chunk_id="b", score=0.85, text="其他",
                  metadata={"source_path": "docs/other.pdf"},
              ),
          ]
          dense = MockDenseRetriever(results=results)
          sparse = MockSparseRetriever(results=results)
          hybrid = HybridSearch(
              query_processor=query_processor,
              dense_retriever=dense, sparse_retriever=sparse,
              fusion=rrf_fusion,
          )
          out = hybrid.search(
              "Azure", top_k=10, filters={"source_path": "azure"}
          )
          assert [r.chunk_id for r in out] == ["a"], \
              "partial：只命中 source_path 含 'azure' 子串的 chunk"

      def test_source_path_partial_no_match(self, query_processor, rrf_fusion):
          """post-fusion source_path partial：子串不命中 → 排除。"""
          results = [
              RetrievalResult(
                  chunk_id="a", score=0.9, text="Azure",
                  metadata={"source_path": "docs/azure.pdf"},
              ),
          ]
          dense = MockDenseRetriever(results=results)
          sparse = MockSparseRetriever(results=results)
          hybrid = HybridSearch(
              query_processor=query_processor,
              dense_retriever=dense, sparse_retriever=sparse,
              fusion=rrf_fusion,
          )
          out = hybrid.search(
              "Azure", top_k=10, filters={"source_path": "nonexistent"}
          )
          assert out == [], "子串不命中应排除全部"
  ```

- [ ] **Step 2：追加 `TestGenericMissingFieldExclude`**

  紧接其后追加：

  ```python
  # =============================================================================
  # ②A: generic filter（custom_field:xxx）对缺字段的 chunk 排除
  # =============================================================================

  class TestGenericMissingFieldExclude:
      """②A: generic filter 的正确语义 —— chunk 缺该字段时排除（与 tags、
      Chroma where 天然行为一致）。_matches_filters else 分支
      ``metadata.get(key) != value``，缺字段时 None != value → 排除。
      锁定行为，防回归。
      """

      def _make_hybrid(self, query_processor, rrf_fusion):
          return HybridSearch(
              query_processor=query_processor,
              dense_retriever=MockDenseRetriever(results=[]),
              sparse_retriever=MockSparseRetriever(results=[]),
              fusion=rrf_fusion,
          )

      def test_generic_present_match(self, query_processor, rrf_fusion):
          """chunk 有字段且值匹配 → 保留。"""
          hybrid = self._make_hybrid(query_processor, rrf_fusion)
          assert hybrid._matches_filters(
              {"author": "张三"}, {"author": "张三"}
          ) is True

      def test_generic_present_mismatch(self, query_processor, rrf_fusion):
          """chunk 有字段但值不匹配 → 排除。"""
          hybrid = self._make_hybrid(query_processor, rrf_fusion)
          assert hybrid._matches_filters(
              {"author": "李四"}, {"author": "张三"}
          ) is False

      def test_generic_missing_field_excluded(self, query_processor, rrf_fusion):
          """chunk 缺该字段 → 排除（filter 正确语义，非放行）。"""
          hybrid = self._make_hybrid(query_processor, rrf_fusion)
          assert hybrid._matches_filters(
              {"other_field": "x"}, {"author": "张三"}
          ) is False
  ```

- [ ] **Step 3：跑两个新测试类，确认全绿（锁定行为）**

  Run: `CONDA_NO_PLUGINS=true conda run -n langchain-test --no-capture-output pytest tests/integration/test_hybrid_search.py::TestSourcePathPartialN3 tests/integration/test_hybrid_search.py::TestGenericMissingFieldExclude -v`

  Expected: 全部 PASS（现状语义已正确，本 Task 为锁定）。

- [ ] **Step 4：跑整个测试文件再次确认无回归**

  Run: `CONDA_NO_PLUGINS=true conda run -n langchain-test --no-capture-output pytest tests/integration/test_hybrid_search.py -v`

  Expected: 全部 PASS。

- [ ] **Step 5：commit**

  ```bash
  git -C "d:/Desktop/Code/RAG-MCP/MODULAR-RAG-MCP-SERVER" add tests/integration/test_hybrid_search.py
  git -C "d:/Desktop/Code/RAG-MCP/MODULAR-RAG-MCP-SERVER" commit -m "test(query): 锁定 source_path partial 与 generic 缺字段排除语义" -m "- TestSourcePathPartialN3：post-fusion partial 子串命中/不命中
  - TestGenericMissingFieldExclude：有/无/缺字段的排除语义（②A）
  - 均为 characterization 测试，现状已正确，防回归"
  ```

---

## 收尾（Task 2 之后，可选但建议）

- [ ] **更新 `PDF处理链路分析.md`**：在 §5 追加 **G8 — collection filter 杀零** 条目（与 D-021/G4 对照），标注已修。
- [ ] **更新 `DEV_CHANGELOG.md`**：加 D-022（或下一个编号）记录本次决策（pre+post 双层语义对齐、N4 拆出）。
- [ ] **端到端验证**（spec §8）：对已 ingest 的 collection 跑 `python scripts/query.py --query "xxx collection:<name>" --collection <name>`，确认有结果（此前返回空）。

---

## Self-Review（写计划后自检）

**1. Spec 覆盖**：
- spec §4.1 改动1（POST_ONLY_FILTERS）→ Task 1 Step 6-7 ✓
- spec §4.2 改动2（collection continue）→ Task 1 Step 8 ✓
- spec §4.3 改动3（确认其余分支无需改）→ Task 2（锁定测试覆盖 source_path/generic）✓
- spec §4.4 改动4（注释更新）→ Task 1 Step 9 ✓
- spec §6 测试（N1/N3/generic/混合）→ Task 1 + Task 2 ✓（混合 filter 由 Task 1 的 `test_collection_stripped...` + sparse collection 覆盖；doc_type 由 Task 1 Step 4 覆盖）
- spec §7 不做（不动 QueryProcessor/ChromaStore/不引入 ES）→ Global Constraints + 计划范围明确 ✓

**2. 占位符扫描**：无 TBD/TODO；每步含完整代码或完整命令 + 预期 ✓。

**3. 类型/命名一致**：`POST_ONLY_FILTERS`（Task 1 定义）在 Task 2 "Consumes" 引用一致；`MockDenseRetriever.last_filters` / `MockSparseRetriever.last_collection` 均为既有 fixture 字段（已核对 [:51](../../../tests/integration/test_hybrid_search.py#L51) / [:86](../../../tests/integration/test_hybrid_search.py#L86)）；`_matches_filters(metadata, filters) -> bool` 签名与 G4 测试用法一致 ✓。

**4. 关键风险已覆盖**：现有 `test_explicit_filters_passed_to_retrievers` / `test_post_fusion_metadata_filter` 会被 N1 语义变化破坏 → Task 1 Step 3-4 显式更新 ✓。
