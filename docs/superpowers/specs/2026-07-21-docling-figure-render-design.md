# Design: Docling Figure Region Rendering (矢量图支持)

**Date**: 2026-07-21
**Branch**: `feat/docling-figure-render` (from `main`)
**Status**: Approved, ready for implementation planning

## Context

当前 `docling` parser 的图片抽取依赖 PyMuPDF 的 `page.get_images(full=True)`，该 API 只能检测 PDF 内嵌的**光栅图对象**（PNG/JPEG）。论文中常见的 matplotlib 图表、流程图、模型架构图等**矢量图形**由绘图指令构成，不是图对象，完全无法捕获。这些图不进 `image_index`、不走 `ImageCaptioner`、查询端拿不到。

Docling 的版面分析其实已经识别出每个 Figure 的位置（`page + bbox`），但 `_item_to_section()` 遇到 Figure 类型直接 `return None` 跳过（`docling_parser.py:221-224`），bbox 信息被浪费。

**目标**：让论文里的矢量图表也能走通"抽取 → caption → 检索 → 回显"完整链路。

## Approach

**放弃 `get_images()`，改用 `page.get_pixmap(clip=bbox)` 渲染 Docling 识别到的每个 Figure 区域。**

`get_pixmap(clip=bbox)` 把 PDF 页面上指定矩形区域渲染成像素图片，无论该区域是光栅图、矢量图还是文字，一律转成 PNG。这样矢量图和光栅图统一处理，代码也大幅简化。

被舍弃的方案（不采用）：
- **混合保留**（保留 `get_images()` + 仅对未覆盖区域渲染）—— 去重判断复杂，收益不大
- **全配置化**（`image_strategy` 开关）—— 增加配置面和测试面，YAGNI

## Scope

**只改 `src/libs/parser/docling_parser.py`。**

`pdf_text` / `pdf_table` 没有版面分析能力，不知道 Figure 的 bbox，无法采用此方案，保持现状（继续用 `get_images()`）。

## Changes

### 移除

| 位置 | 内容 |
|------|------|
| `_extract_page_images()` (line 287-344) | PyMuPDF `get_images()` 整套逻辑，删掉 |
| `_extract_with_docling()` (line 171-188) | PyMuPDF 抽图循环 |
| `_item_to_section()` (line 221-224) | `stype == "figure"` 跳过逻辑 —— 现在主动处理 |

### 新增

**类常量**：
```python
RENDER_DPI = 200          # 渲染分辨率（平衡清晰度与存储）
MIN_FIGURE_SIZE = 100     # bbox 宽/高下限（PDF points，≈1.4 cm），过滤 logo/装饰
```

**新方法 `_render_figure_region()`**：
```python
def _render_figure_region(
    self, fitz_doc, item, doc_hash: str, fig_index: int
) -> Optional[Dict[str, Any]]:
    """渲染 Docling Figure item 的 bbox 区域为 PNG。

    Returns: 标准 image dict（与原 _extract_page_images 输出格式一致），
             尺寸过小/缺 bbox 时返回 None。
    """
```

逻辑：
1. 从 `item.prov` 取 `page_no` 和 `bbox`（通过现有 `_provenance()` 静态方法）
2. 无 bbox 或无 page → 返回 None + debug 日志
3. 计算宽高，任一 < `MIN_FIGURE_SIZE` → 返回 None + debug 日志
4. `fitz.Rect(l, t, r, b)` 构造裁剪矩形
5. `page.get_pixmap(clip=rect, dpi=RENDER_DPI)` 渲染
6. 保存 PNG 到 `data/images/{doc_hash}/{image_id}.png`
7. 返回标准 image dict（`id` / `path` / `page` / `text_offset` / `text_length` / `position`），格式与原 `_extract_page_images` 输出完全一致

### 改造 `_extract_with_docling()`

```python
converter = self._build_converter()
result = converter.convert(str(path))
ddoc = result.document

sections: List[Dict[str, Any]] = []
images: List[Dict[str, Any]] = []

# 打开一次 fitz_doc，所有 Figure 渲染共用
fitz_doc = fitz.open(path) if (self.extract_images and PYMUPDF_AVAILABLE) else None
fig_counter = 0

try:
    for item, _level in ddoc.iterate_items():
        # 优先处理 Figure：渲染 bbox 成图
        if fitz_doc is not None and self._is_figure_item(item):
            rendered = self._render_figure_region(fitz_doc, item, doc_hash, fig_counter)
            if rendered is not None:
                fig_counter += 1
                images.append(rendered)
                sections.append({
                    "type": "figure",
                    "text": f"[IMAGE: {rendered['id']}]",
                    "page": rendered["page"],
                    "bbox": None,
                    "images": [rendered],
                    "html": None,
                })
                continue  # 不再走 _item_to_section

        sec = self._item_to_section(item, ddoc)
        if sec:
            sections.append(sec)
finally:
    if fitz_doc is not None:
        fitz_doc.close()

return sections, images
```

**辅助方法 `_is_figure_item(item)`**：判断 item 标签是否为 FIGURE/PICTURE（复用现有 `_item_label()` + `_LABEL_MAP`）。

### `_item_to_section()` 清理

原来 Figure 跳过的分支可以删除（已经在上层拦截），但保留一个防御性 `return None` 避免 Figure 漏网。

## Configuration

**不加新配置项。** 现有 `ingestion.parser.extract_images` 继续当总开关：
- `true` → 走新的 bbox 渲染逻辑
- `false` → `fitz_doc` 不打开，不渲染任何图

DPI 和最小尺寸作为类常量硬编码。后续如真有调优需要再加 settings.yaml 配置。

## Data Flow（改后完整链路）

```
摄入：
  PDF → Docling 版面分析
    → 遍历 items
      → Figure item（带 page, bbox）
        → 尺寸 ≥ 100 pt → get_pixmap(clip=bbox, dpi=200) → PNG
        → 尺寸 < 100 pt → 跳过（过滤 logo/装饰）
      → TABLE → Docling TableFormer GFM（保持现有）
      → TEXT → 正常提取（保持现有）
    → [IMAGE: id] 占位符进 section
  ↓
  Stage 2.5: ImageStorage.register_image()
  ↓
  Stage 4c: ImageCaptioner
    → Ollama Vision LLM (llava-phi3:3.8b) 看图 → 生成 caption
    → caption 缝进 chunk body
  ↓
  Storage: chunk 进 Chroma + BM25（文本含 caption + [IMAGE: id]）

查询（MCP）：
  用户问 "介绍下论文里的模型架构"
    → Hybrid Search 命中含 "架构" + caption 文字的 chunk
    → MultimodalAssembler 解析 [IMAGE: id]
      → ImageStorage 查路径 → 读文件 → Base64
    → MCP 返回 TextContent + ImageContent
```

## Testing

`tests/unit/test_docling_parser.py` 加用例（沿用现有 mock 模式：mock `DocumentConverter` + mock PyMuPDF `get_pixmap`）：

1. **`test_figure_with_bbox_rendered`** — Figure item 有合法 bbox → 调用 `get_pixmap` 一次、生成 `[IMAGE: id]` section、image dict 格式正确
2. **`test_figure_too_small_skipped`** — bbox 宽或高 < 100pt → 不渲染、不生成 section
3. **`test_figure_without_bbox_skipped`** — `prov.bbox = None` → 跳过、不报错
4. **`test_extract_images_false_no_render`** — `extract_images=False` → `fitz.open` 不被调用、无图
5. **`test_non_figure_items_unaffected`** — text/table item 走原 `_item_to_section` 逻辑，行为不变
6. **`test_pymupdf_unavailable_no_render`** — `PYMUPDF_AVAILABLE=False` → 优雅跳过（与原行为一致）

## Risks & Notes

- **坐标系统**：Docling 的 `prov.bbox` 用 `l/t/r/b`，需确认与 PyMuPDF `fitz.Rect(x0, y0, x1, y1)` 的坐标系一致（PDF points、top-left origin）。实现时用一个真实 PDF 验证渲染出来的区域正确。
- **DPI 损失**：原本 `get_images()` 抽内嵌图是无损的，改渲染后是有损的。200 DPI 对论文场景足够（论文 PDF 本身就是渲染产物）。
- **Docling 漏检**：如果 Docling 没把某个图识别成 Figure，依然抓不到。这是版面分析模型的固有限制，不在本次范围。
- **图片 id 序号**：用 `fig_counter` 全局递增（不再按页内 img_index），保证 id 唯一。

## Files

- 改：`src/libs/parser/docling_parser.py`
- 改：`tests/unit/test_docling_parser.py`
- 讨论 doc（已存在）：`docs/设计/parser修改-论文矢量图处理.md`
