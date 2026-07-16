# PDF 图片检索闭环修复设计

- **日期**：2026-07-16
- **分支**：`feat/docling-vlm`
- **状态**：待评审
- **相关分析**：[PDF处理链路分析.md](../../../PDF处理链路分析.md) §5 G1/G2/G3

## 1. 背景与动机

`PDF处理链路分析.md` 的核对发现：PDF 摄取侧写入了图片结构化元数据，但查询侧取不出来，多模态闭环断裂。具体三个问题：

- **G1（断链）**：图片在默认 MCP 查询路径下取不出。三层路径解析全失败——Chroma 里的 `metadata.images`（dict 列表）被 `_sanitize_metadata` 标量化损坏；`MultimodalAssembler` 未注入 `ImageStorage`；约定路径兜底又缺 `{doc_hash}` 目录段。
- **G2（潜在）**：caption schema 不匹配。摄取侧 `ImageCaptioner` 把 caption 存成 list of dict，查询侧 `extract_image_refs` 期望 dict；即便类型对上，该 list 同样会被 Chroma 标量化。当前 `vision_llm.enabled=false`，暂未触发。
- **G3（功能受损）**：文本/表格 chunk 的页码丢失。section 带了 `page`，但 chunker 的 `_split_text_like`/`_split_table_segment` 没透传，只有携带图片的 chunk 才有 `page_num`，导致多数 citation 无页码。

**根因**：Chroma 只接受标量 metadata，而代码把 `images`/`image_captions`/`tags`/`bbox` 等复合结构直接塞了进去，落盘即有损；查询端又默认假设能拿到原始结构。

## 2. 目标与非目标

### 目标

1. **G1**：MCP 查询能正确返回命中的 `ImageContent`（base64 图片）。
2. **G2**：caption 的 schema 与读写链路对齐，结构化 caption 集中存储，`vision_llm` 开启后自动生效。
3. **G3**：所有结构化 chunk（文本/表格/列表/公式）携带 `page_num`，citation 能带页码。

### 非目标（明确排除）

- G4 tags 过滤、G5 bbox 字符串化、G6 PdfTextParser 重复 helper、G7 占位符位置——留后续。
- 本次不实际启用 vision LLM 生成 caption（provider 留开关，链路用 mock 验证）。
- 不改 `scripts/query.py`（它只 print 文本，本就不碰图片）。
- 不做老数据迁移（数据库已清空）。

## 3. 设计决策（含理由）

| 决策 | 选择 | 备选 | 理由 |
|---|---|---|---|
| 范围 | G1+G2+G3 | 仅 G1 / 全部 G1–G7 | 一组连贯的、查询返回中用户可见的问题 |
| 图片数据源策略 | **彻底解耦**：图片结构化数据集中到 ImageStorage SQLite，Chroma 只存文本+向量 | 保留 Chroma 字段打补丁 | 从根因消除有损标量化，G2 被结构性消掉 |
| caption 程度 | schema/链路做对，`vision_llm` 默认关、留开关、可单测 | 顺手打通 vision LLM | 聚焦"存取闭环正确"，vision LLM 启用作为独立后续任务 |
| SQLite 组织 | `image_index` 加 `caption` 列，一表拿全 | 新建独立 `image_caption` 表 | 最简单、读写各一次查询；"重跑 caption"需求目前不存在（YAGNI） |

## 4. 架构与数据流

**核心思路**：图片的结构化数据（路径/页码/caption）全部以 `image_id` 为主键存在 ImageStorage SQLite；Chroma 只存文本和向量；chunk 正文里的 `[IMAGE: id]` 占位符是查询端反查图片的唯一线索。

### 改动后摄取流

```
parse → document.metadata["images"]（中间态）
  → [新增] 立即注册图片到 image_index（file_path/page/doc_hash）
  → chunker：chunk.text 保留 [IMAGE:id] 占位符；不再把 images/image_captions 写进 chunk.metadata
              section 的 page 透传为 chunk 的 page_num（G3）
  → ImageCaptioner：从占位符解析 id → ImageStorage 查 file_path → vision LLM 生成 caption
                    → set_caption 写 SQLite；同时把 caption 缝进 chunk.text 参与检索
                    → 不再写 chunk.metadata["image_captions"]
  → dense/BM25 编码 → Chroma（metadata 已全标量，无类型损失）
```

### 改动后查询流

```
query → Dense+Sparse+RRF → RetrievalResult（text 含 [IMAGE:id]）
  → MultimodalAssembler（注入 ImageStorage）
      → 从 text 解析 [IMAGE:id]
      → ImageStorage.get_image_meta(id) 一次拿 {file_path, page_num, caption}
      → load_image base64 → ImageContent（caption 命中则附文本块）
```

## 5. 详细改动

### 5.1 ImageStorage（[src/ingestion/storage/image_storage.py](src/ingestion/storage/image_storage.py)）

**Schema**：`image_index` 表新增 `caption TEXT` 列（默认 NULL）：
```sql
image_id TEXT PRIMARY KEY, file_path TEXT NOT NULL, collection TEXT,
doc_hash TEXT, page_num INTEGER, caption TEXT, created_at TEXT NOT NULL
```

**Schema 升级**：`_ensure_database()` 用 `PRAGMA table_info` 检测 `caption` 列是否存在，缺则 `ALTER TABLE image_index ADD COLUMN caption TEXT`。

**新增方法**：
- `set_caption(image_id, caption, model=None)`——幂等 `UPDATE image_index SET caption=? WHERE image_id=?`；行不存在则 no-op + warn。
- `get_image_meta(image_id) -> dict | None`——返回 `{image_id, file_path, collection, doc_hash, page_num, caption}`，查不到返回 None。

### 5.2 顺序依赖（关键）

把图片注册从 stage 6c **提前到 stage 2 之后**（`parser.parse()` 完成后、`chunker.split_document()` 之前）。理由：`ImageCaptioner` 在 stage 4c 调 `set_caption`（UPDATE），必须保证图片行已入库，否则 caption 丢失。注册所需字段（`file_path/collection/doc_hash/page_num`）在 parse 后即齐备。

### 5.3 DocumentChunker（[src/ingestion/chunking/document_chunker.py](src/ingestion/chunking/document_chunker.py)）

- **`_inherit_metadata` 瘦身**（[L424-464](src/ingestion/chunking/document_chunker.py#L424-L464)）：删除 `image_refs` 解析、`chunk_images` 构建、`images` 写入、含图才设的 `page_num`。简化为：继承文档元数据 + `chunk_index` + `source_ref`。
- **G3 页码透传**：`_split_segment` 四个分支（text/table/list/equation）返回 dict 统一加 `"page": seg.get("page")`；`_split_by_sections`（[L163-173](src/ingestion/chunking/document_chunker.py#L163-L173)）组装时：
  ```python
  if rc.get("page") is not None:
      chunk_metadata["page_num"] = rc["page"]
  ```
- 纯文本路径（`_split_plain`，无 section）不带 `page_num`——符合预期。

### 5.4 ImageCaptioner（[src/ingestion/transform/image_captioner.py](src/ingestion/transform/image_captioner.py)）

- 构造加 `image_storage: Optional[ImageStorage] = None`（pipeline 注入）。
- `image_lookup` 不再从 `chunk.metadata["images"]` 读；改为对占位符解析出的每个 id 调 `image_storage.get_image_meta(id)` 取 `file_path`。
- 生成 caption 后：**不再写 `chunk.metadata["image_captions"]`**；改调 `image_storage.set_caption(id, caption)`。
- **保留**把 caption 缝进 `chunk.text`（`[IMAGE: id]\n(Description: …)`）——让 caption 参与 dense/BM25 检索。
- `vision_llm` 关闭时仍 no-op，caption 列保持 NULL。

### 5.5 IngestionPipeline（[src/ingestion/pipeline.py](src/ingestion/pipeline.py)）

- `__init__` 构造 `ImageCaptioner` 时传入 `self.image_storage`（[L164](src/ingestion/pipeline.py#L164)）。
- 把 stage 6c 的图片注册循环（[L474-484](src/ingestion/pipeline.py#L474-L484)）移到 stage 2 之后执行；6c 原位置删除。

### 5.6 MultimodalAssembler（[src/core/response/multimodal_assembler.py](src/core/response/multimodal_assembler.py)）

- `extract_image_refs`（[L161-212](src/core/response/multimodal_assembler.py#L161-L212)）重写为单一数据源：从 `result.text` 解析 `[IMAGE:id]` → `image_storage.get_image_meta(id)` 构建 `ImageReference(file_path, page, caption)`。删除对 `metadata.images`/`metadata.image_captions` 的读取（G2 schema 不匹配随之消失）。
- `resolve_image_path`（[L214-251](src/core/response/multimodal_assembler.py#L214-L251)）简化为两步：`ref.file_path` 直查 + ImageStorage 回查；**删除**缺 `{doc_hash}` 段的无效约定路径兜底。

### 5.7 ResponseBuilder（[src/core/response/response_builder.py](src/core/response/response_builder.py)）

- 构造加 `image_storage: Optional[Any] = None`。
- `multimodal_assembler` property（[L146-152](src/core/response/response_builder.py#L146-L152)）创建 `MultimodalAssembler(image_storage=self._image_storage)`。

### 5.8 query_knowledge_hub（[src/mcp_server/tools/query_knowledge_hub.py](src/mcp_server/tools/query_knowledge_hub.py)）

- 工具初始化时建跨 collection 的单例 `ImageStorage`（参数与 pipeline 一致：`data/db/image_index.db` + `data/images/`）。
- `self._response_builder = response_builder or ResponseBuilder(image_storage=self._image_storage)`（[L127](src/mcp_server/tools/query_knowledge_hub.py#L127)）。

## 6. 降级（沿用项目"优雅降级"规则）

- 图片文件缺失 / SQLite 查不到 → `load_image` 返回 None → 跳过该图，不报错。
- `vision_llm` 关 → caption 列 NULL → `ref.caption=None` → 不输出 caption 文本块。

## 7. 测试策略（pytest，沿用现有 markers）

**单元（`unit`）：**

1. `test_image_storage_caption.py`——`set_caption`/`get_image_meta` 读写、幂等、查空返回 None；schema 升级（无 `caption` 列的老库上 `_ensure_database` 自动加列）。
2. `test_document_chunker_page.py`（G3）——带 `page` 的 sections 切出的 chunk `page_num` 正确；表格 chunk 带页码；`chunk.metadata` 不含 `images/image_captions/image_refs`。
3. `test_multimodal_assembler_sqlite.py`（G1）——注入带临时图片的 ImageStorage，`result.text` 含 `[IMAGE: id]` → `assemble` 产出非空 base64 `ImageContent`；caption 命中文本块；文件缺失时跳过不报错。
4. `test_image_captioner_sqlite.py`（G2）——mock vision LLM，注入已注册图片的 ImageStorage；transform 后 `get_image_meta(id).caption` == 生成值，`chunk.metadata` 无 `image_captions`，`chunk.text` 含 `(Description: …)`。

**集成（`integration`，可选）：** 带内嵌图的小 PDF + `parser=pdf_table`（不依赖 docling 模型权重），跑完 ingestion → query → 断言返回含 `ImageContent`（vision_llm 关，caption 空）。

## 8. 文件改动清单

| 文件 | 改动 |
|---|---|
| `src/ingestion/storage/image_storage.py` | +`caption` 列与 schema 升级；+`set_caption`/`get_image_meta` |
| `src/ingestion/pipeline.py` | 注入 ImageStorage 给 ImageCaptioner；图片注册前移到 stage 2 后 |
| `src/ingestion/chunking/document_chunker.py` | `_inherit_metadata` 瘦身；section `page` 透传（G3） |
| `src/ingestion/transform/image_captioner.py` | 改走 ImageStorage 查 path、写 caption |
| `src/core/response/multimodal_assembler.py` | `extract_image_refs`/`resolve_image_path` 重写 |
| `src/core/response/response_builder.py` | 透传 `image_storage` |
| `src/mcp_server/tools/query_knowledge_hub.py` | 注入 `ImageStorage` |
| `tests/unit/test_image_storage_caption.py` 等 4 个 | 新增 |

## 9. 成功标准

- 单元测试 1–4 全绿。
- 集成测试（若编写）：带图 PDF 摄取后，MCP 查询命中含图 chunk 时返回非空 `ImageContent`。
- 现有 `pytest tests/unit` 不回归。
- `vision_llm` 关时全链路无报错、caption 为空；开启 + mock 时 caption 端到端可见。
