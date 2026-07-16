# PDF 图片检索闭环修复 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 MCP 查询能正确返回命中的图片（G1）、对齐 caption 的存取 schema（G2）、让所有结构化 chunk 带页码（G3）。

**Architecture:** 图片结构化数据（路径/页码/caption）全部以 `image_id` 为主键集中到 ImageStorage SQLite（`image_index` 表加 `caption` 列）；Chroma 只存文本和向量；chunk 正文里的 `[IMAGE: id]` 占位符是查询端反查图片的唯一线索。摄取端 caption 写 SQLite，查询端从占位符解析 id 后一次查全。

**Tech Stack:** Python 3（conda env `langchain-test`）、SQLite、pytest、ChromaDB、MCP SDK。

**Spec:** [docs/superpowers/specs/2026-07-16-pdf-image-retrieval-design.md](../specs/2026-07-16-pdf-image-retrieval-design.md)

## Global Constraints

- 运行环境用 conda env `langchain-test`，**不建 `.venv`**。
- 测试 import 统一用 `src.*`（`conftest.py` 已把 repo root 插 `sys.path`）。
- pytest markers：`unit` / `integration` / `e2e` / `llm` / `slow`。本计划的单元测试用 `unit`，接线测试用 `integration`。
- 路径一律用 `resolve_path()`（锚 `REPO_ROOT`，CWD 无关），不要写死相对路径。
- 遵循项目"优雅降级"规则：图片文件缺失 / 查不到 → 跳过，不抛异常。
- `vision_llm` 保持默认关闭；caption 生成路径用 mock 验证，不在本计划实际跑 vision LLM。
- commit message 用中文 conventional commit（`feat:` / `fix:` / `refactor:` / `test:` / `docs:`）。

---

## File Structure

| 文件 | 责任 | 本计划改动 |
|---|---|---|
| `src/ingestion/storage/image_storage.py` | 图片 SQLite 索引 | +`caption` 列与 schema 升级；+`set_caption` / `get_image_meta` |
| `src/ingestion/chunking/document_chunker.py` | section-aware 切分 | `_inherit_metadata` 瘦身（去图片字段）；section `page` 透传（G3） |
| `src/core/response/multimodal_assembler.py` | 查询端图片拼装 | `extract_image_refs` / `resolve_image_path` 重写，走 ImageStorage |
| `src/core/response/response_builder.py` | MCP 响应构建 | 构造透传 `image_storage` 给 assembler |
| `src/ingestion/transform/image_captioner.py` | 图片 caption 生成 | 改走 ImageStorage 查路径、写 caption |
| `src/ingestion/pipeline.py` | 摄取流水线 | 注入 ImageStorage 给 ImageCaptioner；图片注册前移到 stage 2.5 |
| `src/mcp_server/tools/query_knowledge_hub.py` | MCP 查询工具 | 注入 ImageStorage 给 ResponseBuilder |

测试文件（全部新建于 `tests/unit/` 与 `tests/integration/`）：
`test_image_storage_caption.py`、`test_document_chunker_page.py`、`test_multimodal_assembler_sqlite.py`、`test_response_builder_image_storage.py`、`test_image_captioner_sqlite.py`、`test_pipeline_image_wiring.py`。

---

## Task 1: ImageStorage — caption 列 + 读写方法

**Files:**
- Modify: `src/ingestion/storage/image_storage.py`
- Test: `tests/unit/test_image_storage_caption.py`

**Interfaces:**
- Produces:
  - `ImageStorage.set_caption(image_id: str, caption: str, model: Optional[str] = None) -> None`
  - `ImageStorage.get_image_meta(image_id: str) -> Optional[dict]`（返回 `{image_id, file_path, collection, doc_hash, page_num, caption, created_at}`，查不到返回 `None`）
- 后续 Task 3 / Task 5 依赖这两个方法。

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_image_storage_caption.py`：

```python
"""ImageStorage: caption 读写 + schema 升级（G2 地基）。"""
from pathlib import Path
import sqlite3

import pytest

from src.ingestion.storage.image_storage import ImageStorage


@pytest.fixture
def storage(tmp_path):
    return ImageStorage(
        db_path=str(tmp_path / "image_index.db"),
        images_root=str(tmp_path / "images"),
    )


def _register(storage, image_id="img1", page=1):
    img_file = Path(storage.images_root) / f"{image_id}.png"
    img_file.parent.mkdir(parents=True, exist_ok=True)
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n")
    storage.register_image(
        image_id=image_id, file_path=img_file,
        collection="c", doc_hash="h", page_num=page,
    )


def test_set_and_get_caption(storage):
    _register(storage)
    storage.set_caption("img1", "a RAG diagram")
    meta = storage.get_image_meta("img1")
    assert meta is not None
    assert meta["caption"] == "a RAG diagram"
    assert meta["image_id"] == "img1"
    assert meta["page_num"] == 1


def test_set_caption_idempotent(storage):
    _register(storage)
    storage.set_caption("img1", "first")
    storage.set_caption("img1", "second")
    assert storage.get_image_meta("img1")["caption"] == "second"


def test_set_caption_missing_image_is_noop(storage):
    storage.set_caption("ghost", "x")  # 未注册 → UPDATE 命中 0 行，不报错
    assert storage.get_image_meta("ghost") is None


def test_get_image_meta_missing_returns_none(storage):
    assert storage.get_image_meta("nope") is None


def test_get_image_meta_has_all_fields(storage):
    _register(storage, image_id="img2", page=3)
    meta = storage.get_image_meta("img2")
    assert set(meta.keys()) >= {"image_id", "file_path", "collection",
                                "doc_hash", "page_num", "caption"}
    assert meta["caption"] is None  # 未 set_caption


def test_schema_upgrade_adds_caption_column(tmp_path):
    """无 caption 列的老库打开后应自动加列。"""
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE image_index (
        image_id TEXT PRIMARY KEY, file_path TEXT NOT NULL,
        collection TEXT, doc_hash TEXT, page_num INTEGER,
        created_at TEXT NOT NULL
    )""")
    conn.commit()
    conn.close()

    storage = ImageStorage(db_path=str(db_path), images_root=str(tmp_path / "images"))
    _register(storage, image_id="img9")
    storage.set_caption("img9", "after upgrade")
    assert storage.get_image_meta("img9")["caption"] == "after upgrade"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/unit/test_image_storage_caption.py -v`
Expected: FAIL — `AttributeError: 'ImageStorage' object has no attribute 'set_caption'`（以及 `get_image_meta`）。

- [ ] **Step 3: 实现 — schema 升级 + 新增列**

在 `src/ingestion/storage/image_storage.py` 的 `_ensure_database` 方法里，把 `CREATE TABLE` 语句加上 `caption TEXT` 列（新库直接有），并在建表后调用一个新的升级检查。定位现有 `CREATE TABLE IF NOT EXISTS image_index (...)` 块（约 [image_storage.py:112-121](src/ingestion/storage/image_storage.py#L112-L121)），改为：

```python
            conn.execute("""
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    collection TEXT,
                    doc_hash TEXT,
                    page_num INTEGER,
                    caption TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # 老库（无 caption 列）自动升级
            existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(image_index)")}
            if "caption" not in existing_cols:
                conn.execute("ALTER TABLE image_index ADD COLUMN caption TEXT")
```

- [ ] **Step 4: 实现 — `set_caption` 与 `get_image_meta` 方法**

在 `ImageStorage` 类里、`get_image_path` 方法之后（约 [image_storage.py:325](src/ingestion/storage/image_storage.py#L325) 之后）新增两个方法：

```python
    def set_caption(
        self,
        image_id: str,
        caption: str,
        model: Optional[str] = None,
    ) -> None:
        """为已注册的图片写入 caption（幂等 UPDATE）。

        若 image_id 未注册，UPDATE 命中 0 行，记一条 warning 后 no-op，
        不抛异常（优雅降级）。

        Args:
            image_id: 已注册的图片 id。
            caption: caption 文本。
            model: 可选，生成 caption 的模型名（预留，当前不持久化）。
        """
        if not image_id or not image_id.strip():
            raise ValueError("image_id cannot be empty")

        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute(
                "UPDATE image_index SET caption=? WHERE image_id=?",
                (caption, image_id),
            )
            conn.commit()
            if cur.rowcount == 0:
                logger.warning(
                    f"set_caption: no row for image_id='{image_id}' "
                    "(not registered yet?)"
                )
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to set caption for {image_id}: {e}") from e
        finally:
            conn.close()

    def get_image_meta(self, image_id: str) -> Optional[Dict[str, Any]]:
        """一次取回图片全部字段；查不到返回 None。

        Returns:
            {image_id, file_path, collection, doc_hash, page_num, caption,
             created_at} 或 None。
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                "SELECT * FROM image_index WHERE image_id=?", (image_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
```

- [ ] **Step 5: 跑测试确认通过**

Run: `pytest tests/unit/test_image_storage_caption.py -v`
Expected: 6 passed。

- [ ] **Step 6: 提交**

```bash
git add src/ingestion/storage/image_storage.py tests/unit/test_image_storage_caption.py
git commit -m "feat(storage): ImageStorage 支持 caption 列与读写 (G2 地基)"
```

---

## Task 2: DocumentChunker — section page 透传 + metadata 瘦身（G3）

**Files:**
- Modify: `src/ingestion/chunking/document_chunker.py`
- Test: `tests/unit/test_document_chunker_page.py`

**Interfaces:**
- Consumes: 现有 `Document.metadata["sections"]`（每个 section 带 `page`）。
- Produces: 每个 section-based chunk 的 `metadata["page_num"]` 来自 section；`chunk.metadata` 不再含 `images` / `image_captions` / `image_refs`。

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_document_chunker_page.py`：

```python
"""DocumentChunker: section page 透传 + metadata 瘦身（G3）。"""
import pytest

from src.core.settings import load_settings
from src.core.types import Document
from src.ingestion.chunking.document_chunker import DocumentChunker


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture
def chunker(settings):
    return DocumentChunker(settings)


def _doc_with_sections():
    sections = [
        {"type": "title", "text": "标题", "page": 1, "bbox": None, "images": [], "html": None},
        {"type": "text", "text": "正文片段" * 200, "page": 1, "bbox": None, "images": [], "html": None},
        {"type": "table", "text": "A: 1\nB: 2", "page": 2, "bbox": None, "images": [], "html": "<table></table>"},
        {"type": "list", "text": "- 项一\n- 项二", "page": 3, "bbox": None, "images": [], "html": None},
    ]
    return Document(
        id="doc_test",
        text="标题\n\n" + "正文片段" * 200 + "\n\nA: 1\nB: 2\n\n- 项一\n- 项二",
        metadata={"source_path": "test.pdf", "doc_type": "pdf",
                  "doc_hash": "abc", "sections": sections},
    )


def test_text_chunk_carries_page(chunker):
    chunks = chunker.split_document(_doc_with_sections())
    text_chunks = [c for c in chunks if c.metadata.get("section_type") == "text"]
    assert text_chunks, "应至少有一个 text chunk"
    assert all(c.metadata.get("page_num") == 1 for c in text_chunks)


def test_table_chunk_carries_page(chunker):
    chunks = chunker.split_document(_doc_with_sections())
    table_chunks = [c for c in chunks if c.metadata.get("section_type") == "table"]
    assert table_chunks, "应有一个 table chunk"
    assert table_chunks[0].metadata.get("page_num") == 2


def test_list_chunk_carries_page(chunker):
    chunks = chunker.split_document(_doc_with_sections())
    list_chunks = [c for c in chunks if c.metadata.get("section_type") == "list"]
    assert list_chunks, "应有一个 list chunk"
    assert list_chunks[0].metadata.get("page_num") == 3


def test_chunk_metadata_no_image_fields(chunker):
    chunks = chunker.split_document(_doc_with_sections())
    assert chunks
    for c in chunks:
        assert "images" not in c.metadata
        assert "image_captions" not in c.metadata
        assert "image_refs" not in c.metadata
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/unit/test_document_chunker_page.py -v`
Expected: FAIL — `test_*_carries_page` 断言失败（当前 chunk 无 `page_num`）。

- [ ] **Step 3: 实现 — `_split_segment` 各分支透传 `page`**

在 `src/ingestion/chunking/document_chunker.py` 的 `_split_segment`（约 [document_chunker.py:259-278](src/ingestion/chunking/document_chunker.py#L259-L278)），让每个返回的 raw dict 带 `"page": seg.get("page")`。整体替换为：

```python
    def _split_segment(self, seg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Pass 2: split one merged segment into raw chunk dicts."""
        stype = seg["type"]
        page = seg.get("page")

        if stype == "table":
            return self._split_table_segment(seg)
        if stype == "list":
            return self._split_list_segment(seg)
        if stype == "equation":
            text = seg["text"]
            if text and text.strip():
                return [{
                    "text": text,
                    "section_type": "equation",
                    "table_html": None,
                    "bbox": seg.get("bbox"),
                    "page": page,
                }]
            return []
        # text-like (text / figure / figure_caption / unknown merged into text)
        return self._split_text_like(seg["text"], "text", seg.get("bbox"), page)
```

- [ ] **Step 4: 实现 — `_split_text_like` 接收并写出 `page`**

把 `_split_text_like`（约 [document_chunker.py:280-293](src/ingestion/chunking/document_chunker.py#L280-L293)）改为：

```python
    def _split_text_like(
        self, text: str, section_type: str, bbox: Any, page: Any = None
    ) -> List[Dict[str, Any]]:
        """Recursive-split a text-like segment via the configured splitter."""
        if not text or not text.strip():
            return []
        fragments = self._splitter.split_text(text)
        if not fragments:
            fragments = [text]
        return [
            {"text": f, "section_type": section_type, "table_html": None,
             "bbox": bbox, "page": page}
            for f in fragments
            if f and f.strip()
        ]
```

- [ ] **Step 5: 实现 — table / list 分支的返回 dict 加 `page`**

在 `_split_table_segment`（约 [document_chunker.py:295-337](src/ingestion/chunking/document_chunker.py#L295-L337)）中，给每个返回 dict 加 `"page": bbox` 同级。即把所有形如：

```python
            return [{
                "text": plain,
                "section_type": "table",
                "table_html": html,
                "bbox": bbox,
            }]
```

改为追加一行 `"page": seg.get("page"),`。**三处**都要改：整表返回处、超长切分的首个/其余 chunk 返回处、以及末尾 `results or [{...}]` 兜底处。示例（整表返回）：

```python
        if len(plain) <= 2 * self._chunk_size:
            return [{
                "text": plain,
                "section_type": "table",
                "table_html": html,
                "bbox": bbox,
                "page": seg.get("page"),
            }]
```

同理在 `_split_list_segment`（约 [document_chunker.py:361-379](src/ingestion/chunking/document_chunker.py#L361-L379)）的两个返回 dict 加 `"page": seg.get("page")`。

- [ ] **Step 6: 实现 — `_split_by_sections` 写出 `page_num`**

在 `_split_by_sections`（约 [document_chunker.py:163-174](src/ingestion/chunking/document_chunker.py#L163-L174)）组装 chunk_metadata 处，追加 page 写入：

```python
        chunks: List[Chunk] = []
        for index, rc in enumerate(raw):
            text = rc["text"]
            chunk_id = self._generate_chunk_id(document.id, index, text)
            chunk_metadata = self._inherit_metadata(document, index, text)
            chunk_metadata["section_type"] = rc["section_type"]
            if rc.get("bbox"):
                chunk_metadata["bbox"] = rc["bbox"]
            if rc.get("table_html"):
                chunk_metadata["table_html"] = rc["table_html"]
            if rc.get("page") is not None:
                chunk_metadata["page_num"] = rc["page"]
            chunks.append(Chunk(id=chunk_id, text=text, metadata=chunk_metadata))
        return chunks
```

- [ ] **Step 7: 实现 — `_inherit_metadata` 瘦身（删除图片字段）**

把 `_inherit_metadata`（约 [document_chunker.py:424-464](src/ingestion/chunking/document_chunker.py#L424-L464)）整体替换为：

```python
    def _inherit_metadata(
        self, document: Document, chunk_index: int, chunk_text: str = ""
    ) -> dict:
        """Inherit metadata from document and add chunk-level fields.

        图片相关的 images / image_captions / image_refs 不再写入 chunk
        metadata（它们进 Chroma 会被标量化损坏；图片结构化数据统一由
        ImageStorage 承载，查询端通过正文 [IMAGE: id] 占位符反查）。
        page_num 由 section-aware 路径在 _split_by_sections 中按 section
        page 透传，不再在此处从图片元数据推断。
        """
        chunk_metadata = document.metadata.copy()
        chunk_metadata.pop("images", None)
        chunk_metadata.pop("sections", None)  # parser intermediate
        chunk_metadata["chunk_index"] = chunk_index
        chunk_metadata["source_ref"] = document.id
        return chunk_metadata
```

- [ ] **Step 8: 跑测试确认通过**

Run: `pytest tests/unit/test_document_chunker_page.py -v`
Expected: 4 passed。

- [ ] **Step 9: 跑现有 chunker 测试确认无回归**

Run: `pytest tests/unit -k "chunk" -v`
Expected: 全绿。

- [ ] **Step 10: 提交**

```bash
git add src/ingestion/chunking/document_chunker.py tests/unit/test_document_chunker_page.py
git commit -m "feat(chunker): section page 透传到 chunk + metadata 瘦身 (G3)"
```

---

## Task 3: MultimodalAssembler — 走 ImageStorage 还原图片（G1）

**Files:**
- Modify: `src/core/response/multimodal_assembler.py`
- Test: `tests/unit/test_multimodal_assembler_sqlite.py`

**Interfaces:**
- Consumes: Task 1 的 `ImageStorage.get_image_meta(image_id) -> dict|None`。
- Produces: `MultimodalAssembler(image_storage=...)`；`assemble(results, collection)` 产出含 `types.ImageContent` 的 block 列表。

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_multimodal_assembler_sqlite.py`：

```python
"""MultimodalAssembler: 正文占位符 + ImageStorage 还原图片（G1）。"""
import pytest
from mcp import types

from src.core.response.multimodal_assembler import MultimodalAssembler
from src.core.types import RetrievalResult
from src.ingestion.storage.image_storage import ImageStorage


@pytest.fixture
def storage_with_image(tmp_path):
    storage = ImageStorage(
        db_path=str(tmp_path / "idx.db"),
        images_root=str(tmp_path / "images"),
    )
    img = tmp_path / "images" / "img1.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    storage.register_image(
        image_id="img1", file_path=img, collection="c",
        doc_hash="h", page_num=2,
    )
    storage.set_caption("img1", "架构图")
    return storage


def _result(text):
    return RetrievalResult(
        chunk_id="c1", score=0.9, text=text, metadata={"source_path": "x.pdf"}
    )


def test_assemble_produces_image_content(storage_with_image):
    asm = MultimodalAssembler(image_storage=storage_with_image)
    blocks = asm.assemble([_result("正文 [IMAGE: img1] 结束")], collection="c")
    image_blocks = [b for b in blocks if isinstance(b, types.ImageContent)]
    assert len(image_blocks) == 1
    assert image_blocks[0].mimeType == "image/png"
    assert len(image_blocks[0].data) > 0


def test_caption_emitted_as_text_block(storage_with_image):
    asm = MultimodalAssembler(image_storage=storage_with_image)
    blocks = asm.assemble([_result("[IMAGE: img1]")], collection="c")
    text_blocks = [b for b in blocks if isinstance(b, types.TextContent)]
    assert any("架构图" in b.text for b in text_blocks)


def test_missing_image_skipped(storage_with_image):
    asm = MultimodalAssembler(image_storage=storage_with_image)
    blocks = asm.assemble([_result("[IMAGE: ghost]")], collection="c")
    assert not any(isinstance(b, types.ImageContent) for b in blocks)


def test_no_storage_returns_no_images():
    asm = MultimodalAssembler(image_storage=None)
    blocks = asm.assemble([_result("[IMAGE: img1]")], collection="c")
    assert not any(isinstance(b, types.ImageContent) for b in blocks)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/unit/test_multimodal_assembler_sqlite.py -v`
Expected: FAIL — 当前 `extract_image_refs` 读 `metadata.images`（list），测试给的 metadata 没有 images，回退到占位符解析但 `file_path=None`、`resolve_image_path` 又查不到 → 不产出 ImageContent。

- [ ] **Step 3: 实现 — 重写 `extract_image_refs`**

替换 `src/core/response/multimodal_assembler.py` 的 `extract_image_refs`（约 [multimodal_assembler.py:161-212](src/core/response/multimodal_assembler.py#L161-L212)）：

```python
    def extract_image_refs(
        self,
        result: RetrievalResult,
    ) -> List[ImageReference]:
        """Extract image references from a retrieval result.

        单一数据源：从 result.text 解析 [IMAGE: id] 占位符，再用
        ImageStorage.get_image_meta 一次取回 {file_path, page_num, caption}。
        不再读取 metadata.images / metadata.image_captions（它们不再写入
        chunk metadata）。
        """
        refs: List[ImageReference] = []
        if not result.text:
            return refs

        seen: set = set()
        for image_id in IMAGE_PLACEHOLDER_PATTERN.findall(result.text):
            image_id = image_id.strip()
            if not image_id or image_id in seen:
                continue
            if len(refs) >= self.max_images_per_result:
                break
            seen.add(image_id)

            meta = None
            if self._image_storage is not None:
                try:
                    meta = self._image_storage.get_image_meta(image_id)
                except Exception as e:
                    logger.warning(f"get_image_meta failed for {image_id}: {e}")
                    meta = None

            if meta:
                refs.append(ImageReference(
                    image_id=image_id,
                    file_path=meta.get("file_path"),
                    page=meta.get("page_num"),
                    caption=meta.get("caption"),
                ))
            else:
                # 查不到也保留 ref，交给 resolve_image_path 兜底
                refs.append(ImageReference(image_id=image_id))
        return refs
```

- [ ] **Step 4: 实现 — 简化 `resolve_image_path`**

替换 `resolve_image_path`（约 [multimodal_assembler.py:214-251](src/core/response/multimodal_assembler.py#L214-L251)），删掉缺 `{doc_hash}` 段的无效约定路径兜底：

```python
    def resolve_image_path(
        self,
        ref: ImageReference,
        collection: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve the filesystem path for an image reference.

        两步：(1) 用 ref.file_path（来自 ImageStorage，绝对路径）直查；
        (2) ImageStorage 回查兜底。约定路径兜底已删除（实际存放路径含
        {doc_hash} 段，约定路径无法命中）。
        """
        if ref.file_path and Path(ref.file_path).exists():
            return str(Path(ref.file_path).resolve())

        if self._image_storage is not None:
            try:
                path = self._image_storage.get_image_path(ref.image_id)
                if path and Path(path).exists():
                    return path
            except Exception as e:
                logger.warning(f"ImageStorage lookup failed for {ref.image_id}: {e}")

        return None
```

- [ ] **Step 5: 跑测试确认通过**

Run: `pytest tests/unit/test_multimodal_assembler_sqlite.py -v`
Expected: 4 passed。

- [ ] **Step 6: 提交**

```bash
git add src/core/response/multimodal_assembler.py tests/unit/test_multimodal_assembler_sqlite.py
git commit -m "fix(response): MultimodalAssembler 走 ImageStorage 还原图片 (G1)"
```

---

## Task 4: ResponseBuilder — 透传 image_storage

**Files:**
- Modify: `src/core/response/response_builder.py`
- Test: `tests/unit/test_response_builder_image_storage.py`

**Interfaces:**
- Consumes: Task 3 的 `MultimodalAssembler(image_storage=...)`。
- Produces: `ResponseBuilder(image_storage=...)`；其 `multimodal_assembler` 持有同一 `image_storage`。

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_response_builder_image_storage.py`：

```python
"""ResponseBuilder: 把 image_storage 透传给 multimodal_assembler。"""
from src.core.response.response_builder import ResponseBuilder


class _FakeStorage:
    def get_image_meta(self, image_id):
        return None

    def get_image_path(self, image_id):
        return None


def test_image_storage_reaches_assembler():
    storage = _FakeStorage()
    rb = ResponseBuilder(image_storage=storage)
    asm = rb.multimodal_assembler
    assert asm._image_storage is storage


def test_default_no_storage():
    rb = ResponseBuilder()
    asm = rb.multimodal_assembler
    assert asm._image_storage is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/unit/test_response_builder_image_storage.py -v`
Expected: FAIL — `ResponseBuilder.__init__()` 不接受 `image_storage` 参数 → `TypeError`。

- [ ] **Step 3: 实现 — 构造加参数**

修改 `ResponseBuilder.__init__`（约 [response_builder.py:119-144](src/core/response/response_builder.py#L119-L144)），加 `image_storage` 参数并存为字段：

```python
    def __init__(
        self,
        citation_generator: Optional[CitationGenerator] = None,
        multimodal_assembler: Optional["MultimodalAssembler"] = None,
        max_results_in_content: int = 5,
        snippet_max_length: int = 300,
        enable_multimodal: bool = True,
        image_storage: Optional[Any] = None,
    ) -> None:
        self.citation_generator = citation_generator or CitationGenerator()
        self.max_results_in_content = max_results_in_content
        self.snippet_max_length = snippet_max_length
        self.enable_multimodal = enable_multimodal
        self._image_storage = image_storage
        self._multimodal_assembler = multimodal_assembler
```

- [ ] **Step 4: 实现 — property 注入**

修改 `multimodal_assembler` property（约 [response_builder.py:146-152](src/core/response/response_builder.py#L146-L152)）：

```python
    @property
    def multimodal_assembler(self) -> "MultimodalAssembler":
        """Get or create MultimodalAssembler instance."""
        if self._multimodal_assembler is None:
            from src.core.response.multimodal_assembler import MultimodalAssembler
            self._multimodal_assembler = MultimodalAssembler(
                image_storage=self._image_storage
            )
        return self._multimodal_assembler
```

- [ ] **Step 5: 跑测试确认通过**

Run: `pytest tests/unit/test_response_builder_image_storage.py -v`
Expected: 2 passed。

- [ ] **Step 6: 提交**

```bash
git add src/core/response/response_builder.py tests/unit/test_response_builder_image_storage.py
git commit -m "feat(response): ResponseBuilder 透传 image_storage 给 assembler"
```

---

## Task 5: ImageCaptioner — 改走 ImageStorage（G2）

**Files:**
- Modify: `src/ingestion/transform/image_captioner.py`
- Test: `tests/unit/test_image_captioner_sqlite.py`

**Interfaces:**
- Consumes: Task 1 的 `ImageStorage.get_image_meta`（查 file_path）与 `set_caption`（写 caption）。
- Produces: `ImageCaptioner(settings, llm=..., image_storage=...)`；caption 写入 ImageStorage，不再写 `chunk.metadata["image_captions"]`，但仍缝进 `chunk.text` 参与检索。
- 注：本 transform 现在依赖 `image_storage`（由 pipeline 注入）；未注入时不生成 caption（拿不到图片路径）。

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_image_captioner_sqlite.py`：

```python
"""ImageCaptioner: caption 写 ImageStorage，不再写 chunk metadata（G2）。"""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.types import Chunk
from src.ingestion.storage.image_storage import ImageStorage
from src.ingestion.transform.image_captioner import ImageCaptioner


@dataclass
class _FakeVisionResponse:
    content: str


class _FakeVisionLLM:
    def __init__(self, caption_text):
        self._caption = caption_text
        self.calls = 0

    def chat_with_image(self, text, image, trace=None):
        self.calls += 1
        return _FakeVisionResponse(content=self._caption)


def _settings_with_vision_enabled():
    return SimpleNamespace(vision_llm=SimpleNamespace(enabled=True))


@pytest.fixture
def storage(tmp_path):
    return ImageStorage(
        db_path=str(tmp_path / "idx.db"),
        images_root=str(tmp_path / "images"),
    )


def _chunk_with_image(storage, image_id="img1"):
    img = Path(storage.images_root) / f"{image_id}.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    storage.register_image(
        image_id=image_id, file_path=img, collection="c",
        doc_hash="h", page_num=1,
    )
    return Chunk(
        id="chunk1",
        text=f"正文 [IMAGE: {image_id}] 结束",
        metadata={"source_path": "test.pdf", "chunk_index": 0},
    )


def test_caption_written_to_storage(storage):
    chunk = _chunk_with_image(storage, "img1")
    captioner = ImageCaptioner(
        settings=_settings_with_vision_enabled(),
        llm=_FakeVisionLLM("图说：架构图"),
        image_storage=storage,
    )
    [out] = captioner.transform([chunk])

    assert storage.get_image_meta("img1")["caption"] == "图说：架构图"
    assert "image_captions" not in out.metadata
    assert "(Description: 图说：架构图)" in out.text


def test_no_storage_no_captioning(storage):
    """未注入 image_storage：拿不到图片路径，不生成 caption，不崩。"""
    chunk = _chunk_with_image(storage, "img2")
    fake_llm = _FakeVisionLLM("x")
    captioner = ImageCaptioner(
        settings=_settings_with_vision_enabled(),
        llm=fake_llm,
        image_storage=None,
    )
    [out] = captioner.transform([chunk])
    assert "(Description:" not in out.text
    assert fake_llm.calls == 0


def test_text_without_placeholder_unchanged(storage):
    chunk = Chunk(id="c0", text="普通文本，无图",
                  metadata={"source_path": "t.pdf", "chunk_index": 0})
    captioner = ImageCaptioner(
        settings=_settings_with_vision_enabled(),
        llm=_FakeVisionLLM("不应该被调用"),
        image_storage=storage,
    )
    [out] = captioner.transform([chunk])
    assert out.text == "普通文本，无图"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/unit/test_image_captioner_sqlite.py -v`
Expected: FAIL — `ImageCaptioner.__init__()` 不接受 `image_storage` → `TypeError`。

- [ ] **Step 3: 实现 — 构造加 `image_storage` 参数**

修改 `ImageCaptioner.__init__`（约 [image_captioner.py:46-68](src/ingestion/transform/image_captioner.py#L46-L68)），加参数并存字段。在现有签名里加 `image_storage`，并在方法体赋值：

```python
    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseVisionLLM] = None,
        image_storage: Optional[Any] = None,
    ):
        self.settings = settings
        self.image_storage = image_storage
        self.llm = None
        self._caption_cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()

        if self.settings.vision_llm and self.settings.vision_llm.enabled:
             try:
                 self.llm = llm or LLMFactory.create_vision_llm(settings)
             except Exception as e:
                 logger.error(f"Failed to initialize Vision LLM: {e}")
        else:
             logger.warning("Vision LLM is disabled or not configured. ImageCaptioner will skip processing.")

        self.prompt = self._load_prompt()
```

- [ ] **Step 4: 实现 — 重写 `transform`，走 ImageStorage**

替换 `transform`（约 [image_captioner.py:138-223](src/ingestion/transform/image_captioner.py#L138-L223)）：

```python
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """生成 caption：从正文占位符解析 id → ImageStorage 查路径 → vision LLM
        生成 → set_caption 写库 + 缝进正文参与检索。

        不再写 chunk.metadata["image_captions"]。未注入 image_storage 时
        拿不到图片路径，直接跳过（不生成 caption）。
        """
        if not self.llm:
            return chunks

        with self._cache_lock:
            self._caption_cache.clear()

        # 收集需要 caption 的 image_id → file_path
        images_to_caption: Dict[str, str] = {}
        for chunk in chunks:
            for img_id in self._find_referenced_image_ids(chunk.text):
                img_id = img_id.strip()
                if img_id in images_to_caption:
                    continue
                file_path = self._resolve_image_path(img_id)
                if file_path:
                    images_to_caption[img_id] = file_path

        if images_to_caption:
            self._generate_captions_parallel(images_to_caption, trace)

        # 写回：缝进正文 + set_caption
        total_captions_added = 0
        for chunk in chunks:
            referenced_ids = self._find_referenced_image_ids(chunk.text)
            if not referenced_ids:
                continue
            new_text = chunk.text
            for img_id in referenced_ids:
                img_id_s = img_id.strip()
                with self._cache_lock:
                    caption = self._caption_cache.get(img_id_s)
                if not caption:
                    continue
                new_text = new_text.replace(
                    f"[IMAGE: {img_id}]",
                    f"[IMAGE: {img_id}]\n(Description: {caption})",
                )
                total_captions_added += 1
                if self.image_storage is not None:
                    self.image_storage.set_caption(img_id_s, caption)
            chunk.text = new_text

        with self._cache_lock:
            api_calls = len(self._caption_cache)
        logger.info(f"Added {total_captions_added} captions, API calls: {api_calls}")
        return chunks

    def _resolve_image_path(self, image_id: str) -> Optional[str]:
        """通过 ImageStorage 查 image_id 的文件路径。"""
        if self.image_storage is None:
            return None
        try:
            meta = self.image_storage.get_image_meta(image_id)
        except Exception as e:
            logger.warning(f"get_image_meta failed for {image_id}: {e}")
            return None
        if meta and meta.get("file_path") and Path(meta["file_path"]).exists():
            return meta["file_path"]
        return None
```

- [ ] **Step 5: 实现 — 补 `Path` import**

在 `src/ingestion/transform/image_captioner.py` 顶部 import 区（约 [image_captioner.py:13](src/ingestion/transform/image_captioner.py#L13) 附近），确认有 `from pathlib import Path`。若没有则加上。

- [ ] **Step 6: 跑测试确认通过**

Run: `pytest tests/unit/test_image_captioner_sqlite.py -v`
Expected: 3 passed。

- [ ] **Step 7: 提交**

```bash
git add src/ingestion/transform/image_captioner.py tests/unit/test_image_captioner_sqlite.py
git commit -m "refactor(captioner): caption 改走 ImageStorage 读写 (G2)"
```

---

## Task 6: 接线 — pipeline 注入 + 图片注册前移 + 查询工具注入

**Files:**
- Modify: `src/ingestion/pipeline.py`
- Modify: `src/mcp_server/tools/query_knowledge_hub.py`
- Test: `tests/integration/test_pipeline_image_wiring.py`

**Interfaces:**
- Consumes: Task 1 的 `ImageStorage`；Task 5 的 `ImageCaptioner(image_storage=...)`；Task 4 的 `ResponseBuilder(image_storage=...)`。
- Produces: 端到端打通——摄取时图片先入索引再生成 caption；查询工具的 ResponseBuilder 持有 ImageStorage。

**关键顺序**：图片注册必须从 stage 6c 前移到 stage 2.5（parse 之后、chunking 之前），否则 ImageCaptioner（stage 4c）的 `set_caption`（UPDATE）命中 0 行、caption 丢失。

- [ ] **Step 1: 写接线测试**

创建 `tests/integration/test_pipeline_image_wiring.py`：

```python
"""接线：pipeline 给 ImageCaptioner 注入 image_storage；
查询工具的 ResponseBuilder 持有 image_storage。"""
import pytest

from src.core.response.response_builder import ResponseBuilder
from src.ingestion.storage.image_storage import ImageStorage


@pytest.mark.integration
def test_pipeline_injects_image_storage_to_captioner():
    from src.core.settings import load_settings
    from src.ingestion.pipeline import IngestionPipeline

    settings = load_settings()
    pipeline = IngestionPipeline(settings, collection="test_wiring")
    try:
        assert pipeline.image_captioner is not None
        assert pipeline.image_captioner.image_storage is pipeline.image_storage
    finally:
        pipeline.close()


def test_query_tool_response_builder_has_image_storage():
    from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool

    tool = QueryKnowledgeHubTool()
    assert isinstance(tool._image_storage, ImageStorage)
    asm = tool._response_builder.multimodal_assembler
    assert asm._image_storage is tool._image_storage
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/integration/test_pipeline_image_wiring.py -v`
Expected: FAIL — `pipeline.image_captioner.image_storage` 不存在 / `tool._image_storage` 不存在。

- [ ] **Step 3: 实现 — pipeline 注入 image_storage 给 ImageCaptioner**

修改 `src/ingestion/pipeline.py` 的 `__init__`（约 [pipeline.py:164](src/ingestion/pipeline.py#L164)）：

```python
        self.image_captioner = ImageCaptioner(settings, image_storage=self.image_storage)
        has_vision = self.image_captioner.llm is not None
        logger.info(f"  ✓ ImageCaptioner initialized (vision_enabled={has_vision})")
```

（`self.image_storage` 在该类里已于 stage 6 初始化——见 [pipeline.py:191-195](src/ingestion/pipeline.py#L191-L195)；需把 `self.image_storage` 的初始化**上移到 ImageCaptioner 初始化之前**。）

具体：把现有这一段（约 [pipeline.py:191-195](src/ingestion/pipeline.py#L191-L195)）：

```python
        self.image_storage = ImageStorage(
            db_path=str(resolve_path("data/db/image_index.db")),
            images_root=str(resolve_path("data/images"))
        )
        logger.info("  ✓ ImageStorage initialized")
```

移到 **Stage 4: Transforms 之前**（即 `self.chunk_refiner = ChunkRefiner(settings)` 那一行之前），保证 `ImageCaptioner` 构造时 `self.image_storage` 已就绪。

- [ ] **Step 4: 实现 — 图片注册从 stage 6c 前移到 stage 2.5**

在 `pipeline.py` 的 `run` 方法里，找到 stage 2（Document Loading）trace 记录之后、stage 3（Chunking）`logger.info("\n✂️  Stage 3: Document Chunking")` 之前（约 [pipeline.py:296-300](src/ingestion/pipeline.py#L296-L300) 之间），插入新的 stage 2.5：

```python
            # ─────────────────────────────────────────────────────────────
            # Stage 2.5: Register images (moved from 6c)
            # 必须在 ImageCaptioner (stage 4c) 之前完成，否则 set_caption
            # 命中 0 行、caption 丢失。
            # ─────────────────────────────────────────────────────────────
            logger.info("\n🖼️  Stage 2.5: Image Storage Index")
            images = document.metadata.get("images", [])
            for img in images:
                img_path = Path(img["path"])
                if img_path.exists():
                    self.image_storage.register_image(
                        image_id=img["id"],
                        file_path=img_path,
                        collection=self.collection,
                        doc_hash=file_hash,
                        page_num=img.get("page", 0),
                    )
            logger.info(f"  Indexed {len(images)} images")
```

然后**删除**原 stage 6c 的图片注册块（约 [pipeline.py:471-485](src/ingestion/pipeline.py#L471-L485)，即 `# 6c: Register images ...` 到 `logger.info(f"      Indexed {len(images)} images")`）。注意：stage 6 末尾 trace 里 `image_storage_details`（约 [pipeline.py:505-513](src/ingestion/pipeline.py#L505-L513)）用的是 `images` 变量构造详情，不调用 register，可原地保留——但在删除 6c 后 `images` 变量未在 stage 6 定义。把 trace 里 `image_storage_details` 用的 `images` 改为 `document.metadata.get("images", [])`：

```python
                image_storage_details = [
                    {"image_id": img["id"], "file_path": str(img["path"]),
                     "page": img.get("page", 0), "doc_hash": file_hash}
                    for img in document.metadata.get("images", [])
                ]
```

并把 `stages["storage"]["images_indexed"]` 的值改为 `len(document.metadata.get("images", []))`。

- [ ] **Step 5: 实现 — 查询工具注入 ImageStorage**

修改 `src/mcp_server/tools/query_knowledge_hub.py` 的 `QueryKnowledgeHubTool.__init__`（约 [query_knowledge_hub.py:105-131](src/mcp_server/tools/query_knowledge_hub.py#L105-L131)）。在 import 区加：

```python
from src.core.settings import resolve_path
from src.ingestion.storage.image_storage import ImageStorage
```

在 `__init__` 里，`self._response_builder` 赋值之前建 image_storage 并注入：

```python
        self._settings = settings
        self.config = config or QueryKnowledgeHubConfig()
        self._hybrid_search = hybrid_search
        self._reranker = reranker
        self._embedding_client = None

        # 跨 collection 单例 ImageStorage（与 pipeline 同库），供
        # ResponseBuilder → MultimodalAssembler 反查图片。
        self._image_storage = ImageStorage(
            db_path=str(resolve_path("data/db/image_index.db")),
            images_root=str(resolve_path("data/images")),
        )
        self._response_builder = response_builder or ResponseBuilder(
            image_storage=self._image_storage
        )

        self._initialized = False
        self._current_collection: Optional[str] = None
```

- [ ] **Step 6: 跑测试确认通过**

Run: `pytest tests/integration/test_pipeline_image_wiring.py -v`
Expected: 2 passed。

- [ ] **Step 7: 全量回归（单元层）**

Run: `pytest tests/unit -v`
Expected: 全绿（含本计划新增的 5 个单元测试文件）。

- [ ] **Step 8: 编译 / import 自检**

Run:
```bash
python -m compileall src
ruff check .
```
Expected: 无错误。

- [ ] **Step 9: 提交**

```bash
git add src/ingestion/pipeline.py src/mcp_server/tools/query_knowledge_hub.py tests/integration/test_pipeline_image_wiring.py
git commit -m "feat(wiring): pipeline 注入 image_storage + 图片注册前移; 查询工具注入"
```

---

## 完成标准（对应 spec §9）

- Task 1–5 的单元测试 + Task 6 接线测试全绿。
- `pytest tests/unit` 不回归。
- `python -m compileall src` 与 `ruff check .` 通过。
- 手动验证（可选，需带内嵌图的 PDF）：`python scripts/ingest.py --path <带图PDF> --collection demo` 后，`python -m src.mcp_server.server` 起服务，查询命中含图 chunk 时返回非空 `ImageContent`（vision_llm 关闭时 caption 为空，属预期）。

## Self-Review 记录

- **Spec 覆盖**：G1→Task 3+Task 6（查询工具注入）；G2→Task 1+Task 5（schema + 读写对齐）；G3→Task 2。✅
- **占位符**：无 TBD/TODO；每个实现步骤均含完整代码。✅
- **类型一致**：`get_image_meta` / `set_caption` / `image_storage` 在 Task 1 定义、Task 3/5/6 消费，签名一致；`_split_text_like` 新增 `page` 参数在 Task 2 定义与调用处一致。✅
- **顺序依赖**：Task 6 Step 3-4 明确 `self.image_storage` 上移 + 图片注册前移，避免 caption 丢失。✅
