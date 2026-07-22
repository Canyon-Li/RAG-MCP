# N4 QueryProcessor filter 解析收口 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 收口 `QueryProcessor._extract_filters`，未识别的 `word:value` 不再当 generic filter，从源头根治 N4（自然语言冒号 → 三路杀零）。

**Architecture:** 把"无差别 `findall` + 整体 `sub` 删除"重构为 `finditer` 逐个处理：白名单 key（collection/col、type/doc_type、source/src、tag/tags）收 filter 并从 query 文本删除；未识别 key 原样保留为查询文本参与分词。同时删除单字母别名 `c`/`s`/`t`（N4 近亲）。正则本身不动。

**Tech Stack:** Python 3、jieba、pytest；conda env `langchain-test`。

**Spec:** [docs/superpowers/specs/2026-07-22-n4-filter-parsing-allowlist-design.md](../specs/2026-07-22-n4-filter-parsing-allowlist-design.md)

## Global Constraints

- **环境**：conda env `langchain-test`。测试必须用 `python -m pytest`（`conda run -n langchain-test pytest` 会调 base 环境的 pytest，缺 jieba）。
- **pytest 命令**（Git Bash；PowerShell 需 `$env:CONDA_NO_PLUGINS='true'` 前缀替代 `CONDA_NO_PLUGINS=true `）：
  ```
  CONDA_NO_PLUGINS=true conda run -n langchain-test --no-capture-output python -m pytest <path> -v
  ```
- **注释中文 OK**（项目既有风格）；commit 用 conventional commits 中文（参考 `fix(query): ...`）。
- **分支**：当前 `fix/filter-pre-post-alignment`，直接在此分支 commit（N4 与 N1/N3 同主题、零冲突）。
- **不纳入 commit**：工作区遗留的 `M .gitignore`（+1 行 `.claude/settings.local.json`）与本次无关，`git add` 时只 add 本任务的两个文件。
- **不动**：`hybrid_search.py` / `chroma_store.py` / `settings.yaml` / `FILTER_PATTERN` 正则（见 spec §4.3、§7）。
- **下游死代码**：N4 收口后 `_matches_filters` 的 generic else 分支与 `ChromaStore` generic `where` 不再被触发，本次不清理（spec §7）。

## File Structure

- **Modify** `src/core/query_engine/query_processor.py`：`_extract_filters`（:168-208）整体重构 + if-elif 白名单 tuple（:189-193）删单字母别名。
- **Modify** `tests/unit/test_query_processor.py`：改 `test_generic_filter`（:185-191）+ 新增 `TestFilterAllowlistN4` 类（置于 `TestFilterParsing` 类之后）。

---

## Task 1: N4 filter 解析收口（删 generic else + 未识别保留文本 + 删单字母别名）

**Files:**
- Modify: `src/core/query_engine/query_processor.py:168-208`（`_extract_filters`）与 `:189-193`（白名单 tuple）
- Test: `tests/unit/test_query_processor.py:185-191`（改 `test_generic_filter`）+ `:202` 之后（新增 `TestFilterAllowlistN4`）

**Interfaces:**
- Consumes: 无（叶子任务；`FILTER_PATTERN` 常量 :77 不变，`QueryProcessorConfig.enable_filter_parsing` 不变）
- Produces: `QueryProcessor.process()` 返回的 `ProcessedQuery.filters` 只含白名单 key；未识别 `word:value` 回到 `keywords` 路径。下游 `hybrid_search` 无需任何配合改动。

- [ ] **Step 1: 改 `test_generic_filter` 预期（N4 收口语义）**

替换 `tests/unit/test_query_processor.py:185-191` 整个方法为：

```python
    def test_generic_filter(self):
        """N4: 未识别 key:value 不当 filter，当普通查询文本回到 keywords。"""
        processor = QueryProcessor()
        result = processor.process("custom_field:custom_value search")

        # 未识别 key 不再产生 generic filter
        assert "custom_field" not in result.filters
        assert result.filters == {}
        # search 仍是关键词
        assert "search" in result.keywords
```

- [ ] **Step 2: 新增 `TestFilterAllowlistN4` 测试类**

在 `TestFilterParsing` 类之后（`test_disable_filter_parsing` 方法结束后、`class TestEdgeCases` 之前）插入新类：

```python
class TestFilterAllowlistN4:
    """N4: filter 解析收口——自然语言 word:value 不再被误解析为 filter。"""

    def test_natural_language_colon_not_filter(self):
        """自然语言冒号（时间 / URL / 中文）不当 filter。"""
        processor = QueryProcessor()

        r = processor.process("会议 12:30 下午")
        assert r.filters == {}
        assert "会议" in r.keywords

        r = processor.process("官网 https://example.com 地址")
        assert r.filters == {}

        r = processor.process("Azure:服务端 配置")
        assert r.filters == {}

    def test_windows_path_not_filter(self):
        """Windows 路径 c:\\... 不当 filter（c 已不是别名）。"""
        processor = QueryProcessor()
        r = processor.process(r"路径 c:\Users\test 文档")
        assert r.filters == {}

    def test_single_letter_alias_disabled(self):
        """单字母别名 c/s/t 已删除，不再当 filter。"""
        processor = QueryProcessor()
        for q in ("c:docs", "s:readme", "t:pdf"):
            r = processor.process(q)
            assert r.filters == {}, f"{q} 不应产生 filter"

    def test_multiletter_alias_still_works(self):
        """多字母别名 col/src 仍工作。"""
        processor = QueryProcessor()

        r = processor.process("col:docs Azure")
        assert r.filters.get("collection") == "docs"
        assert "Azure" in r.keywords

        r = processor.process("src:readme.md content")
        assert r.filters.get("source_path") == "readme.md"

    def test_whitelist_filters_still_parsed(self):
        """白名单 4 类回归保护（全名语法）。"""
        processor = QueryProcessor()
        r = processor.process("collection:docs type:pdf source:r.md tag:a,b")
        assert r.filters.get("collection") == "docs"
        assert r.filters.get("doc_type") == "pdf"
        assert r.filters.get("source_path") == "r.md"
        assert "a" in r.filters["tags"]
        assert "b" in r.filters["tags"]
```

- [ ] **Step 3: 跑新测试，确认按预期失败**

Run:
```
CONDA_NO_PLUGINS=true conda run -n langchain-test --no-capture-output python -m pytest "tests/unit/test_query_processor.py::TestFilterParsing::test_generic_filter" "tests/unit/test_query_processor.py::TestFilterAllowlistN4" -v
```
Expected（实现未改，4 FAIL + 2 PASS）：
- `test_generic_filter` FAIL（当前 else 把 `custom_field` 收进 filters）
- `TestFilterAllowlistN4::test_natural_language_colon_not_filter` FAIL（`12:30`→`filters["12"]="30"`）
- `TestFilterAllowlistN4::test_windows_path_not_filter` FAIL（`c:` 是 collection 别名）
- `TestFilterAllowlistN4::test_single_letter_alias_disabled` FAIL（`c/s/t` 是别名）
- `TestFilterAllowlistN4::test_multiletter_alias_still_works` PASS（`col/src` 现状就支持）
- `TestFilterAllowlistN4::test_whitelist_filters_still_parsed` PASS（白名单现状正确）

- [ ] **Step 4: 重构 `_extract_filters`（删 generic else + 未识别保留文本 + 删单字母别名）**

替换 `src/core/query_engine/query_processor.py:168-208` 整个方法为：

```python
    def _extract_filters(self, query: str) -> tuple[Dict[str, Any], str]:
        """Extract filter syntax from query.

        只有白名单 key（collection/col、type/doc_type、source/src、tag/tags）
        产生 filter 并从 query 文本删除；未识别的 word:value 当普通查询文本
        原样保留（参与分词），不再走 generic filter 分支（N4 收口）。

        Args:
            query: Normalized query string

        Returns:
            Tuple of (filters dict, query without filter syntax)
        """
        if not self.config.enable_filter_parsing:
            return {}, query

        filters: Dict[str, Any] = {}
        kept: List[str] = []
        last_end = 0

        for m in FILTER_PATTERN.finditer(query):
            key, value = m.group(1), m.group(2)
            key_lower = key.lower()
            # 匹配段之前的文本原样保留
            kept.append(query[last_end:m.start()])

            if key_lower in ("collection", "col"):
                filters["collection"] = value
            elif key_lower in ("type", "doc_type"):
                filters["doc_type"] = value
            elif key_lower in ("source", "src"):
                filters["source_path"] = value
            elif key_lower in ("tag", "tags"):
                filters.setdefault("tags", []).extend(value.split(","))
            else:
                # 未识别 key：N4 收口，不当 filter，原样保留为查询文本
                kept.append(m.group(0))
            last_end = m.end()
        kept.append(query[last_end:])

        query_without_filters = " ".join("".join(kept).split())

        return filters, query_without_filters
```

改动要点（implementer 自检）：
1. `("collection", "col", "c")` → `("collection", "col")`，`("type", "doc_type", "t")` → `("type", "doc_type")`，`("source", "src", "s")` → `("source", "src")`（删 `c`/`t`/`s`）。
2. 删除原 `else: filters[key] = value` generic 分支，改为 `else: kept.append(m.group(0))`（未识别匹配原样回填）。
3. 从 `FILTER_PATTERN.findall` + `FILTER_PATTERN.sub("")` 改为 `finditer` 逐段拼接：白名单命中时不回填匹配段（= 删除），未识别时回填（= 保留）。
4. `List` 已在文件 :18 import，无需新增 import。

- [ ] **Step 5: 跑 Step 3 的测试，确认全部通过**

Run:
```
CONDA_NO_PLUGINS=true conda run -n langchain-test --no-capture-output python -m pytest "tests/unit/test_query_processor.py::TestFilterParsing::test_generic_filter" "tests/unit/test_query_processor.py::TestFilterAllowlistN4" -v
```
Expected: 6 passed。

- [ ] **Step 6: 全文件回归**

Run:
```
CONDA_NO_PLUGINS=true conda run -n langchain-test --no-capture-output python -m pytest tests/unit/test_query_processor.py -v
```
Expected: 全部 passed（原有用例 0 回归；`test_collection_short_syntax` 用 `col:` 不受别名删除影响；`test_disable_filter_parsing` 总开关语义不变）。

- [ ] **Step 7: Commit**

只 add 本任务两个文件（**不** add `.gitignore`）：
```bash
git add src/core/query_engine/query_processor.py tests/unit/test_query_processor.py
git commit -m "fix(query): N4 filter 解析收口，未识别 word:value 不再当 generic filter"
```

---

## 验证（spec §8）

- **单测**：Task 1 Step 6 全文件绿。
- **端到端 dogfood**（实现后可选）：对已 ingest 的 collection 跑
  ```
  conda run -n langchain-test --no-capture-output python scripts/query.py --query "Azure:服务端 配置" --verbose
  ```
  确认 `ProcessedQuery.filters` 为空、关键词含 `Azure`/`服务端`、FUSION 返回非空（此前杀零）。

## 明确不做（spec §7）

- 不动 `hybrid_search.py` / `chroma_store.py`（generic 死代码防御性保留）。
- 不扩展白名单（title/summary/page_num 等，YAGNI）。
- 不改 `FILTER_PATTERN` 正则。
- 不接 settings / 不 config-driven。
