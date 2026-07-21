# tags 元数据过滤修复设计（G4）

- **日期**：2026-07-21
- **分支**：`fix/tags-filter`
- **状态**：待评审
- **相关分析**：[PDF处理链路分析.md](../../../PDF处理链路分析.md) §5 G4

## 1. 背景与动机

`PDF处理链路分析.md` §5 G4 指出：`tags` 元数据过滤实际不工作。核对当前代码后，故障比最初分析的"一层"更深——**三层叠加**，且 tags 过滤会**主动杀零**检索结果：

- **故障①（存储）**：`chroma_store._sanitize_metadata`（[chroma_store.py:398-400](src/libs/vector_store/chroma_store.py#L398-L400)）把 tags 列表逗号拼接成字符串 `"tag1,tag2"`。这是 Chroma 标量约束的必然结果，**本次保留不改**。
- **故障②（pre-fusion）**：`hybrid_search.search` 把含 tags 的 `merged_filters` 整体传给 `_run_retrievals`（[hybrid_search.py:256-260](src/core/query_engine/hybrid_search.py#L256-L260)）→ dense_retriever → `chroma.query(where={"tags": ["azure"]})`。Chroma 把 list 值当 `$in`，检查存储的 tags 字符串是否 ∈ `["azure"]` → 存储值是 `"tag1,tag2"` → **永不命中 → dense 返回空**。
- **故障③（post-fusion）**：`_matches_filters` 的 tags 分支（[hybrid_search.py:730-736](src/core/query_engine/hybrid_search.py#L730-L736)）对存储的字符串做 `set(meta_tags)` → 按字符级拆分 → 语义错误。

**入口已通**：`QueryProcessor._extract_filters`（[query_processor.py:195-199](src/core/query_engine/query_processor.py#L195-L199)）已能从 `"tag:azure 架构图"` 解析出 `filters={"tags": ["azure"]}`。所以用户现在就能写内联 `tag:` 语法，只是写完之后过滤全失败。

**根因**：tags 是 list 语义，而 Chroma metadata 只接受标量。tags 过滤天然只能在 Python 内存（post-fusion）里做，不能下推到 Chroma where。当前代码既没在 pre-fusion 剥离 tags（导致杀零），又没在 post-fusion 正确读逗号字符串。

## 2. 目标与非目标

### 目标

1. 用户在 query 里写 `tag:azure 架构图`（单标签或多标签）能正确过滤，只返回带匹配标签的 chunk。
2. tags 过滤不再误杀 dense/sparse 检索结果（pre-fusion 路径不受 tags 影响）。

### 非目标（明确排除）

- **不改 MCP 工具 schema**：不加 `filters` / `tags` 显式参数。内联 `tag:` 语法是唯一入口（范围决策 A）。
- **不改 tags 存储表示**：sanitizer 逗号拼接行为保留，对已摄取数据向后兼容，无需重新摄取（存储决策 A）。
- **不做 pre-fusion tags 过滤**：Chroma 对字符串字段做不了 "list contains"，tags 下推到 Chroma 无意义。
- **不修 G5/G6/G7**：留后续。

## 3. 设计决策（含理由）

| 决策 | 选择 | 备选 | 理由 |
|---|---|---|---|
| 范围 | 仅修内联 `tag:` 语法 | 加 MCP 显式参数 / 含 pre-fusion | 内联入口已存在，修通即兑现"标签过滤可用"；MCP 参数是锦上添花，YAGNI |
| tags 存储表示 | 保留逗号拼接 | JSON 数组字符串 | tags 是关键词不含逗号；向后兼容已摄取数据，免迁移 |
| 剥离注入点 | `hybrid_search.search()` 传 retrievers 前 | chroma_store `_build_where_clause` 跳过 | "tags 是 post-fusion 专属"语义集中在 query 层一处；存储层保持通用 |

## 4. 改动清单（1 个文件，2 处改动）

全部改动集中在 `src/core/query_engine/hybrid_search.py`。**存储层 `chroma_store.py` 零改动**。

### 4.1 `search()` 剥离 tags（修故障②）

[hybrid_search.py:252-260](src/core/query_engine/hybrid_search.py#L252-L260) 附近，传给 `_run_retrievals` 前摘掉 tags：

```python
# Merge explicit filters with query-extracted filters
merged_filters = self._merge_filters(processed_query.filters, filters)

# tags 是 list 语义，Chroma where 处理不了 → 只走 post-fusion
retrieval_filters = {
    k: v for k, v in merged_filters.items() if k != "tags"
}

# Step 2: Run retrievals
dense_results, sparse_results, dense_error, sparse_error = self._run_retrievals(
    processed_query=processed_query,
    filters=retrieval_filters,   # 原: merged_filters
    trace=trace,
)
```

post-fusion（[hybrid_search.py:293-294](src/core/query_engine/hybrid_search.py#L293-L294)）继续用 `merged_filters`（含 tags），不动。

### 4.2 `_matches_filters` tags 分支改读逗号字符串（修故障③）

[hybrid_search.py:730-736](src/core/query_engine/hybrid_search.py#L730-L736)：

```python
elif key == "tags":
    meta_tags = metadata.get("tags", "")
    # Chroma 落盘后是逗号字符串；防御性兼容 list（未 sanitize 的情况）
    if isinstance(meta_tags, str):
        meta_tags = [t.strip() for t in meta_tags.split(",") if t.strip()]
    elif not isinstance(meta_tags, list):
        meta_tags = []
    if not isinstance(value, list):
        value = [value]
    if not set(meta_tags) & set(value):
        return False
```

**设计要点**：
- `isinstance(meta_tags, str)` 分支处理 Chroma 落盘后的逗号字符串（生产路径）。
- `isinstance(..., list)` 分支防御性兼容未过 sanitizer 的场景（如单元测试直接构造 metadata）。
- `strip()` 清理标签前后空格；`if t.strip()` 过滤空段（连续逗号 / 空字符串）。

## 5. 修复后数据流

```
用户输入: "tag:azure 架构图"

QueryProcessor._extract_filters
  → filters={"tags": ["azure"]}, 剩余关键词=["架构", "图"]

hybrid_search.search:
  merged_filters = {"tags": ["azure"]}
  retrieval_filters = {} (tags 被剥离)          ← 4.1 新增
  ├─ dense: chroma.query(where=None)            ← 不再被 tags 杀零，正常检索
  ├─ sparse: BM25("架构 图")                    ← 正常
  └─ fusion → N 条候选
  post-fusion _matches_filters (metadata_filter_post=True):
     meta_tags = "azure,cloud,架构".split(",") = ["azure","cloud","架构"]   ← 4.2 修好
     set(["azure","cloud","架构"]) & set(["azure"]) = {"azure"} ✓ 保留
     不含 azure 的 chunk → 交集空 → 剔除
  → 最终只返回带 azure 标签的 chunk
```

## 6. 边界处理

| 场景 | meta tags | 行为 |
|---|---|---|
| chunk 无 tags 字段 | `metadata.get("tags","")` → `""` | split 得 `[]` → 交集空 → 剔除（正确：没标签不命中 tag 过滤） |
| 空字符串 | `""` | `[]` → 剔除 |
| 标签带空格 | `" azure , cloud "` | `strip()` → `["azure","cloud"]` |
| list 形态（未 sanitize） | `["azure","cloud"]` | 防御走 list 分支 |
| 多标签过滤 | filter `["azure","aws"]` | 交集非空即保留（OR 语义，与现有设计一致） |

## 7. 测试策略（TDD）

QueryProcessor 侧不用测（`tag:` 解析已工作，现有 `test_query_processor.py` 覆盖）。bug 只在过滤执行端。

### 7.1 新增单元测试：`_matches_filters` tags 分支

位置：`tests/unit/test_hybrid_search.py` 追加，或新建 `tests/unit/test_matches_filters_tags.py`。

| 用例 | meta tags | filter value | 期望 |
|---|---|---|---|
| 逗号字符串命中 | `"azure,cloud"` | `["azure"]` | True |
| 逗号字符串不命中 | `"azure,cloud"` | `["aws"]` | False |
| 空字符串 | `""` | `["azure"]` | False |
| 空格清理 | `" azure , cloud "` | `["cloud"]` | True |
| list 形态防御 | `["azure","cloud"]` | `["azure"]` | True |

**RED**：当前代码对 `"azure,cloud"` 做 `set(meta_tags)` 得到**字符级**集合 `{'a','z','u','r','e',...}`，而 `set(["azure"])` 是含一个字符串元素的集合，两者求交为**空**（字符串 `"azure"` ≠ 任何单字符）→ `not ∅` → 返回 False。故"命中"用例（期望 True）当前返回 False → 测试失败；"空格清理"同理 RED。"不命中"用例碰巧返回 False（正确），属 GREEN 但仍保留作回归保护。**GREEN**：4.2 改 `split(",")` 后读端语义正确，全绿。

### 7.2 新增集成测试：端到端 `tag:xxx` 过滤

位置：`tests/integration/test_tags_filter.py`（新建）。

- 摄取 3 个 chunk 进测试 collection，**手动设 tags**（不依赖 enricher 关键词抽取，保证确定性）：chunk A `tags=["azure"]`、chunk B `tags=["aws"]`、chunk C `tags=[]`。
- 查 `"tag:azure 架构"` → **只返回 A**。
- 此测试同时验证：①pre-fusion 剥离生效（否则 dense 被 Chroma where 杀零，全空）；②post-fusion 读端修好（B 被过滤）。

**RED**：当前代码下 dense 因 `where={"tags":["azure"]}` 匹配字符串失败 → 返回空 → 测试失败。**GREEN**：4.1 剥离 + 4.2 读端修好 → A 正确返回。

### 7.3 现有测试验证（不改）

- [test_chroma_store_roundtrip.py:424](tests/integration/test_chroma_store_roundtrip.py#L424) 断言 `tags=='tag1,tag2,tag3'` → 存储行为没变，**继续通过**。
- [test_hybrid_search.py:524](tests/integration/test_hybrid_search.py#L524) `test_post_fusion_metadata_filter`（collection 过滤）→ tags 剥离不影响 collection，**继续通过**。
- `pytest tests/unit/test_query_processor.py` 确认 QueryProcessor 解析无回归。

## 8. 改动面总结

- **生产代码**：1 文件（`hybrid_search.py`）2 处改动，存储层零改动。
- **测试**：1 个单元测试组（5 用例）+ 1 个集成测试文件（端到端）。
- **风险**：低。tags 剥离只影响含 tags 的 filter，标量 filter（collection/doc_type/source_path）路径不变。
