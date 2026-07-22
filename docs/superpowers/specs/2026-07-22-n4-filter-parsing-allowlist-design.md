# QueryProcessor filter 解析收口设计（修 N4）

> 日期：2026-07-22
> 范围：`src/core/query_engine/query_processor.py` filter 解析层——自然语言 `word:value` 误解析为 generic filter 的杀零根治
> 相关：[PDF处理链路分析.md](../../../PDF处理链路分析.md) N4 / [前序 spec §9](2026-07-22-filter-pre-post-alignment-design.md) / DEV_CHANGELOG D-021（tags post-fusion-only）

---

## 1. 背景与问题

### 1.1 N4 — 自然语言 `word:value` 被误解析为 generic filter，三路杀零

`QueryProcessor` 的 filter 正则 `(\w+):([^\s]+)`（[query_processor.py:77](../../../src/core/query_engine/query_processor.py#L77)）**无边界**——它盲目匹配"一串字母数字 + 半角冒号 + 一串非空白"这个形状，分不清自然语言冒号与有意 filter。`_extract_filters`（[query_processor.py:168-208](../../../src/core/query_engine/query_processor.py#L168-L208)）对匹配结果做分发：白名单 key（collection/type/source/tags）归到语义 filter，**其余一律走 `else` 当 generic filter**（[query_processor.py:200-202](../../../src/core/query_engine/query_processor.py#L200-L202)）。

触发场景（自然语言里冒号随处可见）：

| query 片段 | 命中 key | 命中 value | 人的理解 |
|---|---|---|---|
| `Azure:服务端` | `Azure` | `服务端` | 自然语言 |
| `12:30` | `12` | `30` | 时间 |
| `https://example.com` | `https` | `//example.com` | URL |
| `3:1` | `3` | `1` | 比例 |

杀零链路（三路全废）：

1. **关键词被吞**：[query_processor.py:205](../../../src/core/query_engine/query_processor.py#L205) `FILTER_PATTERN.sub("", query)` 把**所有**匹配段从 query 文本删掉再提取关键词。`如何配置 Azure:服务端` → 删掉 `Azure:服务端` → 只剩 `如何配置`，`Azure` 这个核心词丢失。
2. **Dense 杀零**：generic key 不在 `POST_ONLY_FILTERS`（N1 修复后 = `{tags, collection, source_path}`）里 → 下推 Chroma `where`。metadata 无此 key → 全不匹配 → Dense 返回空。
3. **Post-fusion 清零**：`_matches_filters` 的 generic else 分支 `metadata.get(key) != value` → `None != value` → 返回 False → 连 Sparse 找回的结果也被剔光。

净效果：用户只是问了个带冒号的问题 → 查到 0 条，且无任何提示。

### 1.2 generic filter 是"死功能"

generic filter 理论卖点是"按任意 metadata 字段过滤"。但 chunk metadata 字段是 ingestion 代码**写死的一组**（source_path / doc_type / page_num / section_type / title / summary / tags / bbox / table_html…），用户不知道这些内部字段名；即使碰巧写对（如 `title:年度报告`），post-fusion 是**精确相等**比较，手写 value 几乎不可能等于 enricher 生成的完整值。

结论：generic filter **实践里命不中任何东西，唯一效果是误触发杀零**——负资产。

### 1.3 与 N1 / N3 的关系

- **N1**（collection 杀零）、**N3**（source_path 两层语义不一致）已在 `fix/filter-pre-post-alignment` 分支修复（见前序 spec），改动在 `hybrid_search.py` 层。
- **N4** 是同一类问题在**解析层**（QueryProcessor）的根：从源头不让 generic filter 产生，下游自然不再误触发。本次改动与 N1/N3 **完全独立**（不同文件），不依赖 N1/N3 已合入。

---

## 2. 架构认知（决策依据）

### 2.1 filter key 与下游处理逻辑强绑定

每个白名单 key 在下游都绑着**专门处理逻辑**：

| key | 下游处理 |
|---|---|
| `collection` | 物理隔离（独立 Chroma collection + BM25 index），post-fusion 放行（N1）|
| `doc_type` | exact，Dense `where` 下推 + post-fusion 兜底 |
| `source_path` | partial，两层一致（N3）|
| `tags` | list 交集，post-fusion-only（D-021）|

加一个新 filter key 不只是改数据，还得在 `hybrid_search` / `ChromaStore` 加对应分支。所以白名单本质是**代码契约**，不是纯配置——这是白名单写死在代码、而非 config-driven 的核心理由。

### 2.2 现有白名单 4 类的健康度

| key | 状态 |
|---|---|
| `collection` | 物理隔离字段（metadata 不写），N1 已修（post-fusion 放行）|
| `doc_type` | ✅ 活（parser 写入，如 `pdf`/`docx`）|
| `source_path` | ✅ 活（upserter 写入）|
| `tags` | ✅ 活（enricher 写入）|

白名单本身健康，唯一坏的是 `else` generic 分支。

### 2.3 单字母别名是 N4 近亲

现有白名单含单字母别名 `c`/`s`/`t`（[query_processor.py:189-193](../../../src/core/query_engine/query_processor.py#L189-L193)）。它们太短，自然语言高频：

- `c:\Users\李生`（Windows 文件路径）→ `c` 是 collection 别名 → `filters["collection"] = "\Users\李生"`。N1 救了 collection（不杀零），但语义错乱。
- `s:\xxx` 或 `请看 s:3 段` → `s` 是 source 别名 → `filters["source_path"] = "..."` → **N3 同款杀零**（dense exact 丢贡献，`s` 没被 N1 救）。
- `t:pdf` 类似。

删掉 `else` 后这些别名仍在白名单，是 N4 的残余风险。

---

## 3. 决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| ① generic filter 去留 | **删除**（白名单化）| 死功能，唯一效果是杀零；与 audit §172、前序 spec §9 一致 |
| ② 白名单来源 | **静态写死在代码** | key 与下游逻辑强绑定，config-driven 无法"配置即生效"；QueryProcessor 现状不接 settings（stopwords 等也都在代码里），单独为 filter_keys 接 settings 破坏其独立性 |
| ③ 白名单范围 | **现有 4 类** | 都是活字段或已处理（见 §2.2）；扩展到 title/summary（文本，该用搜索非 `:` 过滤）/ page_num（布局字段用户不会写）无收益，YAGNI |
| ④ 单字母别名 `c`/`s`/`t` | **删除** | N4 近亲，自然语言高频（盘符/比例/缩写）；多字母别名 `col`/`type`/`src` 够便捷；现有测试未覆盖单字母别名，删除不破坏测试 |

---

## 4. 设计（改动集中在 [query_processor.py](../../../src/core/query_engine/query_processor.py)）

### 4.1 改动 1：白名单收口 + 删除 generic `else`

`_extract_filters` 语义翻转：

| 匹配类型 | 当 filter？ | 从 query 文本删？ |
|---|---|---|
| 白名单 key（collection/doc_type/source_path/tags）| ✅ 收 | ✅ 删（不当关键词）|
| 未识别 key | ❌ 不收 | ❌ **保留**（当普通查询文本，参与分词）|

实现把"无差别 `findall` + 整体 `sub` 删除"改成逐个 `finditer`，白名单匹配收 filter 并跳过、未识别匹配原样回填：

```python
def _extract_filters(self, query: str) -> tuple[Dict[str, Any], str]:
    if not self.config.enable_filter_parsing:
        return {}, query

    filters: Dict[str, Any] = {}
    kept: list[str] = []
    last_end = 0
    for m in FILTER_PATTERN.finditer(query):
        key, value = m.group(1), m.group(2)
        key_lower = key.lower()
        kept.append(query[last_end:m.start()])          # 匹配前的文本段

        if key_lower in ("collection", "col"):          # 删了 "c"
            filters["collection"] = value               # 收 filter，不回填（删除）
        elif key_lower in ("type", "doc_type"):         # 删了 "t"
            filters["doc_type"] = value
        elif key_lower in ("source", "src"):            # 删了 "s"
            filters["source_path"] = value
        elif key_lower in ("tag", "tags"):
            filters.setdefault("tags", []).extend(value.split(","))
        else:
            kept.append(m.group(0))                     # 未识别：原样保留为查询文本
        last_end = m.end()
    kept.append(query[last_end:])

    query_without_filters = " ".join("".join(kept).split())
    return filters, query_without_filters
```

> 伪代码示意语义，不锁具体写法（如改用 `re.sub(func)` 等价）。关键契约：**只有白名单 key 进 filters 且从文本删除；未识别匹配当文本保留**。

### 4.2 改动 2：删除单字母别名

- `collection` 别名：`("collection", "col", "c")` → `("collection", "col")`
- `doc_type` 别名：`("type", "doc_type", "t")` → `("type", "doc_type")`
- `source_path` 别名：`("source", "src", "s")` → `("source", "src")`
- `tags` 别名：`("tag", "tags")` 不变。

效果：`c:xxx`/`s:xxx`/`t:xxx` 不再被当 filter，走未识别路径保留为查询文本。

### 4.3 正则本身不改

`FILTER_PATTERN`（[query_processor.py:77](../../../src/core/query_engine/query_processor.py#L77)）保持 `(\w+):([^\s]+)`。匹配形状的"无边界"问题由"白名单收口 + 未识别保留文本"化解——正则照常匹配，只是处理逻辑不再无差别收 filter / 删文本。无需收紧正则（收紧会引入新的边界判定复杂度，且无法根治歧义）。

---

## 5. 解析行为最终对照

| query 中的串 | 改前 | 改后 |
|---|---|---|
| `Azure:服务端` | `filter azure=服务端`（杀零 + 关键词被删）| 文本保留，`Azure`/`服务端` 进关键词 |
| `12:30` | `filter 12=30`（杀零）| 文本保留 |
| `https://example.com` | `filter https=//example.com`（杀零）| 文本保留 |
| `3:1` | `filter 3=1`（杀零）| 文本保留 |
| `c:\Users\test` | `filter collection=\Users\test`（语义错乱）| 文本保留（`c` 不再是别名）|
| `s:\xxx` | `filter source_path=\xxx`（N3 同款杀零）| 文本保留 |
| `collection:docs` | `filter collection=docs` | `filter collection=docs`（不变）|
| `type:pdf` | `filter doc_type=pdf` | `filter doc_type=pdf`（不变）|
| `source:report.md` | `filter source_path=report.md` | `filter source_path=report.md`（不变）|
| `col:docs` | `filter collection=docs` | `filter collection=docs`（多字母别名保留）|

---

## 6. 测试（`tests/unit/test_query_processor.py`）

### 6.1 改 `test_generic_filter`

[test_query_processor.py:185-191](../../../tests/unit/test_query_processor.py#L185-L191) 现状断言 `custom_field:custom_value` 产生 generic filter。改为断言收口语义：

- `filters` 不含 `custom_field`（为空或无此 key）
- `"search"` 仍在 keywords
- `custom_field:custom_value` 未从 `original_query` / 关键词路径丢失（关键词集非空）

> 关键词断言写宽松（如 `custom` 或 `value` 在关键词，或 keywords 非空且含 `search`），不锁死 jieba 对下划线/冒号的具体切分。

### 6.2 新增用例（锁防回归）

1. **自然语言冒号不当 filter**：`会议 12:30 下午`、`Azure:服务端 配置`、`官网 https://example.com` → `filters` 为空，核心词（`会议`/`Azure`/`官网`）在关键词。
2. **Windows 路径不当 filter**：`路径 c:\Users\test 文档` → `filters` 为空（`c` 已不是别名）。
3. **单字母别名失效**：`c:docs`、`s:readme`、`t:pdf` → `filters` 为空，文本保留。
4. **多字母别名仍工作**：`col:docs Azure` → `filters["collection"]=="docs"`；`src:x.md` → `filters["source_path"]=="x.md"`。
5. **白名单回归**：`collection:docs type:pdf source:r.md tag:a,b` → 各 filter 正确收（保护现有 4 类不回归）。
6. **enable_filter_parsing=False 不变**：`collection:docs Azure` → `filters` 空、文本当关键词（总开关语义保持）。

### 6.3 现有测试影响评估

- `test_collection_filter` / `test_collection_short_syntax`（用 `col:`）/ `test_type_filter` / `test_source_filter` / `test_tag_filter` / `test_multiple_filters`：全绿（白名单行为不变，`col` 保留）。
- `test_disable_filter_parsing`：全绿（总开关不受影响）。
- 无现有测试使用单字母别名 `c:`/`s:`/`t:`，删除别名不破坏任何现有测试。

---

## 7. 明确不做

- **不动 `hybrid_search.py` / `chroma_store.py`**：N4 收口后，`_matches_filters` 的 generic else 分支与 `ChromaStore` 的 generic `where` 不再有 generic filter 触发，成为**死代码**。本次**不清理**（防御性保留，避免扩大改动面；标注为已知遗留，未来统一清理）。
- **不扩展白名单**到 title/summary/page_num/section_type（YAGNI）。
- **不动 `enable_filter_parsing=False` 总开关**语义。
- **不收紧 `FILTER_PATTERN` 正则**（见 §4.3）。
- **不接 settings / 不 config-driven**（见 §3 ②）。

---

## 8. 验证

- **单测**：`tests/unit/test_query_processor.py` 全绿（改 1 + 新增 ~6，现有用例无回归）。
- **端到端 dogfood**：对已 ingest 的 collection 跑
  `python scripts/query.py --query "Azure:服务端 配置" --verbose`
  确认 `ProcessedQuery.filters` 为空、关键词含 `Azure`/`服务端`、FUSION 返回非空（此前杀零）。
- **静态确认**：generic 不再进入 `merged_filters`，下游 `where` / `_matches_filters` 的 generic 路径不再被触发。

> 端到端焦点是"自然语言冒号不杀零"，不依赖 N1/N3 是否已合入（N4 独立分支基于 main）。

---

## 9. 后续

- **死代码清理**：`_matches_filters` generic else 分支、`ChromaStore` generic `where` 可在未来统一清理（需确认无其他路径产生 generic filter）。
- **白名单扩展**：若未来确需按 `title`/`section_type` 等过滤，扩白名单 key + 在下游加对应分支 + 加测试。
