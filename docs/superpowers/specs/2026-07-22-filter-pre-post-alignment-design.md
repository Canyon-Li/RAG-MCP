# Filter pre/post 双层语义对齐设计（修 N1 + N3）

> 日期：2026-07-22
> 范围：`src/core/query_engine/hybrid_search.py` 层 metadata filter 的杀零修复
> 相关：[PDF处理链路分析.md](../../../PDF处理链路分析.md) G1–G7 / DEV_CHANGELOG D-021（tags post-fusion-only）

---

## 1. 背景与问题

### 1.1 N1 — `collection` filter 杀零

`QueryProcessor` 支持 `collection:xxx` 内联语法（[query_processor.py:189-190](../../../src/core/query_engine/query_processor.py#L189-L190)），解析后进入 `merged_filters`。`hybrid_search` 当前**仅剥离 tags**（[hybrid_search.py:260-262](../../../src/core/query_engine/hybrid_search.py#L260-L262)），`collection` 被下推到 `ChromaStore.query(where={"collection":"xxx"})`。

但 chunk metadata **设计上没有 collection 字段**（它是物理隔离维度：parser / chunker / enricher / upserter 都不写），Chroma `where` 对缺失 key 判不匹配 → Dense 返回空。post-fusion `_matches_filters` 的 collection 分支同样读不到 → 把 Sparse 来源也过滤光。

**净效果**：query 里出现 `collection:xxx` → 整个查询返回空。

实测确认（chromadb in-memory，单条记录无 collection 字段）：
```
collection:hub (key absent) -> ids=[[]]   # 杀零
```

### 1.2 N3 — `source_path` 过滤语义不一致

- post-fusion `_matches_filters` 对 `source_path` 用 **partial**（`value not in source`，[hybrid_search.py:751-755](../../../src/core/query_engine/hybrid_search.py#L751-L755)）。
- Dense pre-fusion 下推 Chroma `where` 用 **exact**（[chroma_store.py:431](../../../src/libs/vector_store/chroma_store.py#L431)）。

用户写 `source:report`：Dense exact 不匹配杀零，post partial 能命中。同一条结果在两层语义不一致。

### 1.3 N4（本次不修，拆单独议题）

`QueryProcessor` 的正则 `(\w+):([^\s]+)`（[query_processor.py:77](../../../src/core/query_engine/query_processor.py#L77)）会把自然语言里的 `word:value`（如 `Azure:服务端`）误解析成 generic filter，下推后杀零。根在 `QueryProcessor`，且 generic filter 是被 `test_generic_filter`（[test_query_processor.py:185-191](../../../tests/unit/test_query_processor.py#L185-L191)）锁定的 feature（支持自定义 metadata 过滤），不能简单丢弃。需单独设计 filter 解析策略。**本次不动。**

---

## 2. 架构认知（决策依据）

### 2.1 post-fusion 层不可去除

BM25（本项目 JSON 倒排索引实现，[bm25_indexer.py](../../../src/ingestion/storage/bm25_indexer.py)）只存词项、不存 metadata，先天无法 pre-filter。只要 Sparse 路径参与融合，post-fusion 兜底就**必须存在**——不是设计选择，是轻量 local-first 实现的固有取舍。

（换 Elasticsearch / Lucene 能让 Sparse 也 pre-filter，但引入重型依赖，与项目零依赖风格冲突；post 兜底是务实的补偿。不在本次范围。）

### 2.2 谓词下推原则 vs 存储层能力

能 pre-fusion 下推的应下推（保召回 quota、减后续计算），但受限于 Chroma `where` 的表达能力：
- **能下推**：标量 exact（`doc_type`、generic exact）。
- **不能下推**：list（`tags`）、partial（`source_path`）、不存在字段（`collection`）。

### 2.3 两层语义必须一致

同一 filter key 在 pre（Dense `where`）和 post（`_matches_filters`）两层的匹配语义必须相同，否则一条结果在一层通过、另一层被干掉（N3 的根源）。

---

## 3. 决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| ① `source_path` 语义 | **partial** | 长路径 exact 几乎无法命中；post 层本就在，partial 零增量成本 |
| ② generic 缺字段 | **排除** | filter 正确本义；与 tags、Chroma `where` 天然行为一致 |
| ③ N4 是否纳入 | **不纳入** | 根在 QueryProcessor（另一层），动 `test_generic_filter` 风险不同，拆单独 spec |

---

## 4. 设计（改动集中在 [hybrid_search.py](../../../src/core/query_engine/hybrid_search.py)）

### 4.1 改动 1：扩展 pre-fusion 剥离集（N1 + N3）

新增模块级常量，[hybrid_search.py:260-262](../../../src/core/query_engine/hybrid_search.py#L260-L262) 改为剥离 `{tags, collection, source_path}`：

```python
# post-only：Chroma where 表达不了正确语义，只能融合后兜底
# - tags:        list 语义，标量化后 where 做不了交集（D-021）
# - collection:  metadata 不携带（物理隔离维度），下推必杀零（N1）
# - source_path: 用 partial 语义，where 默认只 exact（N3）
POST_ONLY_FILTERS = {"tags", "collection", "source_path"}

retrieval_filters = {
    k: v for k, v in merged_filters.items() if k not in POST_ONLY_FILTERS
}
```

效果：传给 Dense 的 `retrieval_filters` 只剩 `{doc_type, generic}`，Chroma `where` exact 可表达，pre 下推正确。

### 4.2 改动 2：post-fusion collection 放行（N1）

[`_matches_filters`](../../../src/core/query_engine/hybrid_search.py#L713-L761) 的 collection 分支改为 `continue`：

```python
if key == "collection":
    # 物理隔离已保证（独立 Chroma collection + BM25 index），
    # metadata 不携带 collection；放行，不检查。
    # ⚠️ 必须用 continue，不能用 return True —— 后者会跳过
    # 后续 doc_type / source_path 等其他 key 的检查，引入新 bug。
    continue
```

### 4.3 改动 3：确认其余分支无需改

- `source_path`（[hybrid_search.py:751-755](../../../src/core/query_engine/hybrid_search.py#L751-L755)）：已 partial ✓
- `doc_type`（[hybrid_search.py:736-738](../../../src/core/query_engine/hybrid_search.py#L736-L738)）：已 exact ✓
- `tags`（[hybrid_search.py:739-750](../../../src/core/query_engine/hybrid_search.py#L739-L750)）：已 list 交集（D-021）✓
- generic `else`（[hybrid_search.py:757-759](../../../src/core/query_engine/hybrid_search.py#L757-L759)）：`metadata.get(key) != value`，缺字段时 `None != value` → 排除 ✓（决策②A 现状已正确，**无需改**）

### 4.4 改动 4：注释更新

[`HybridSearchConfig.metadata_filter_post`](../../../src/core/query_engine/hybrid_search.py#L71-L74) 的 docstring 更新：post-only 集 = `{tags, collection, source_path}` 及各自原因；`metadata_filter_post=False` 时这三类静默 no-op，`doc_type` / generic 仍走 Dense pre 下推。

---

## 5. 双层过滤最终行为

| filter | Dense pre (`where`) | post-fusion (`_matches_filters`) | 一致 |
|---|---|---|---|
| `collection` | 剥离 | `continue` 放行 | ✓（物理隔离保证）|
| `source_path` | 剥离 | partial | ✓（两层都 partial）|
| `tags` | 剥离 | list 交集 | ✓（D-021）|
| `doc_type` | exact 下推 | exact 兜底 | ✓（覆盖 Sparse 来源）|
| generic | exact 下推 | exact 缺字段排除 | ✓（覆盖 Sparse 来源）|

每个 key 两层语义一致 → N3 根治；collection 不再下推 → N1 根治。

---

## 6. 测试

新增 `tests/unit/test_hybrid_search_filters.py`（或追加到现有 hybrid_search 测试）：

1. **N1 回归**：query 带 `collection:xxx`（QueryProcessor 解析出 collection filter），断言 `search()` 返回非空（不杀零）。
2. **N3 一致**：构造 chunk `metadata.source_path` 含子串，query 带 `source:<子串>`，断言命中（Dense 不下推 + post partial）。
3. **②A generic 排除**：构造 chunk 无 `custom_field`，query 带 `custom_field:xxx`，断言该 chunk 被排除。
4. **混合 filter**：query 同时带 `collection + doc_type + source`，断言各层各 key 行为正确（collection 放行、doc_type exact、source partial）。

`test_query_processor.py` **不动**（N4 未纳入）。

---

## 7. 明确不做

- 不动 `QueryProcessor` / `test_generic_filter`（N4 拆单独议题）。
- 不动 `ChromaStore._build_where_clause`（generic / doc_type 现状正确）。
- 不引入 Elasticsearch（post 兜底是轻量 local-first 方案的合理补偿）。

---

## 8. 验证

- **单测**：上述 4 个用例全绿。
- **静态实测**（已完成）：chromadb in-memory 确认 `where` 对缺失 key 杀零，证实 N1 根因；修复后 collection 不再下推、不再杀零。
- **端到端**：`python scripts/query.py --query "xxx collection:<name>" --collection <name>` 对已 ingest 的 collection 跑一遍，确认有结果（此前会返回空）。

---

## 9. 后续（N4 单独议题）

N4 需重新设计 `QueryProcessor` 的 filter 解析策略：区分"用户想用 filter"与"自然语言冒号"。可能方向：filter key 须在已知白名单 ∪ 实际 enrich 字段集合内才被当 filter，否则当普通文本。需相应改 `test_generic_filter` 的预期。另立 spec。
