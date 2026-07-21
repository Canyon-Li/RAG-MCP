# Docling Figure Region Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `docling_parser`'s `page.get_images()` raster-only extraction with `page.get_pixmap(clip=bbox)` region rendering so vector graphics (matplotlib charts, architecture diagrams) in PDFs flow through the caption → retrieve → display pipeline.

**Architecture:** Iterate Docling's layout items; for each Figure item, render its bbox region to PNG via PyMuPDF. This captures both raster and vector content in one pass. The emitted image dict is shape-compatible with the old output, so downstream stages (ImageStorage, ImageCaptioner, MultimodalAssembler) need no changes.

**Tech Stack:** Python 3.10+, PyMuPDF (`fitz`), IBM Docling, pytest with `unittest.mock`.

## Global Constraints

- **DPI = 200** — hardcoded class constant `RENDER_DPI = 200` (balance clarity vs storage).
- **MIN_FIGURE_SIZE = 100** — hardcoded class constant, in PDF points (≈1.4 cm); figures smaller than this in either dimension are skipped (filters logos/decorative bands).
- **Only modify `src/libs/parser/docling_parser.py` and `tests/unit/test_docling_parser.py`.** `pdf_text` / `pdf_table` parsers stay unchanged (no layout analysis, can't use bbox rendering).
- **No new config items.** Existing `ingestion.parser.extract_images` remains the on/off switch: `false` → no rendering at all.
- **Image dict contract** (must match existing shape exactly): `{"id", "path", "page", "text_offset", "text_length", "position": {"width", "height", "page", "index"}}`.
- **Code style:** match existing file — `from __future__ import annotations`, type hints, `logger = logging.getLogger(__name__)`, 4-space indent, section dividers with `# ---` blocks.
- **Conda env `langchain-test`** is the runtime — run all commands there.
- **Commit messages** follow repo convention: `type(scope): description` (Chinese allowed in description).

**Reference spec:** `docs/superpowers/specs/2026-07-21-docling-figure-render-design.md`

---

### Task 1: `_render_figure_region()` + class constants (TDD)

**Files:**
- Modify: `src/libs/parser/docling_parser.py` (add class constants on `DoclingParser`, add `_render_figure_region` and `_is_figure_item` methods)
- Test: `tests/unit/test_docling_parser.py` (append new tests)

**Interfaces:**
- Consumes: `_provenance(item)` (existing static method at `docling_parser.py:264-280`), `_generate_image_id(doc_hash, page, sequence)` (inherited from `BaseParser`), `_item_label(item)` (existing static method at `docling_parser.py:238-243`), module-level `_LABEL_MAP`
- Produces:
  - `DoclingParser._render_figure_region(fitz_doc, item, doc_hash, fig_index) -> Optional[dict]` — renders one Figure item's bbox to PNG, returns standard image dict or None
  - `DoclingParser._is_figure_item(item) -> bool` — True if `_LABEL_MAP.get(_item_label(item)) == "figure"`
  - `DoclingParser.RENDER_DPI = 200`, `DoclingParser.MIN_FIGURE_SIZE = 100` — class constants

- [ ] **Step 1: Create feature branch from main**

Run:
```bash
git checkout main
git checkout -b feat/docling-figure-render
```
Expected: branch switches to `feat/docling-figure-render`.

- [ ] **Step 2: Write three failing tests for `_render_figure_region`**

Append to `tests/unit/test_docling_parser.py`:

```python
def test_render_figure_region_renders_valid_bbox(settings, tmp_path):
    """Figure with a valid, large-enough bbox is rendered to a PNG."""
    parser = DoclingParser(
        settings, collection="t",
        image_storage_dir=str(tmp_path / "images"), extract_images=True,
    )
    # Mock pixmap + page + fitz_doc
    pix = MagicMock()
    pix.width = 1000
    pix.height = 700
    page = MagicMock()
    page.get_pixmap = MagicMock(return_value=pix)
    fitz_doc = MagicMock()
    fitz_doc.__getitem__.return_value = page

    item = _make_item("PICTURE", page=2, bbox=(50, 50, 550, 400))  # 500x350 pt

    with patch("src.libs.parser.docling_parser.fitz.Rect"):
        img = parser._render_figure_region(fitz_doc, item, "abcd1234ef", 0)

    assert img is not None
    assert img["id"] == "abcd1234_2_1"  # doc_hash[:8]_page_seq
    assert img["page"] == 2
    assert img["text_length"] == len("[IMAGE: abcd1234_2_1]")
    assert img["position"] == {"width": 1000, "height": 700, "page": 2, "index": 0}
    page.get_pixmap.assert_called_once()
    # dpi=200 passed; clip= a fitz.Rect mock (patched)
    _name, kwargs = page.get_pixmap.call_args
    assert kwargs["dpi"] == 200
    pix.save.assert_called_once()


def test_render_figure_region_skips_small_bbox(settings, tmp_path):
    """Figure smaller than MIN_FIGURE_SIZE in either dimension is skipped."""
    parser = DoclingParser(
        settings, collection="t",
        image_storage_dir=str(tmp_path / "images"), extract_images=True,
    )
    fitz_doc = MagicMock()
    item = _make_item("PICTURE", page=1, bbox=(10, 10, 50, 50))  # 40x40 pt < 100

    img = parser._render_figure_region(fitz_doc, item, "abcd1234ef", 0)

    assert img is None
    fitz_doc.__getitem__.assert_not_called()  # page never accessed


def test_render_figure_region_skips_missing_bbox(settings, tmp_path):
    """Figure with no bbox provenance is skipped (Docling sometimes omits it)."""
    parser = DoclingParser(
        settings, collection="t",
        image_storage_dir=str(tmp_path / "images"), extract_images=True,
    )
    fitz_doc = MagicMock()
    item = _make_item("PICTURE", page=1)  # no bbox → prov.bbox = None

    img = parser._render_figure_region(fitz_doc, item, "abcd1234ef", 0)

    assert img is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_docling_parser.py::test_render_figure_region_renders_valid_bbox tests/unit/test_docling_parser.py::test_render_figure_region_skips_small_bbox tests/unit/test_docling_parser.py::test_render_figure_region_skips_missing_bbox -v`
Expected: FAIL with `AttributeError: 'DoclingParser' object has no attribute '_render_figure_region'`.

- [ ] **Step 4: Add class constants and implement `_render_figure_region` + `_is_figure_item`**

In `src/libs/parser/docling_parser.py`, add two class constants inside `class DoclingParser` (right after the docstring, before `def __init__`). Find the line:
```python
    """PDF parser using IBM Docling (DocLayNet + TableFormer).
    ...
    docling failure.
    """
```
Add after the closing `"""`:

```python
    # Region rendering for figure extraction (replaces get_images).
    # get_pixmap(clip=bbox) captures both raster and vector content.
    RENDER_DPI = 200  # Resolution for bbox rendering (clarity vs storage tradeoff)
    MIN_FIGURE_SIZE = 100  # Skip figures < 100pt in either dim (filters logos/decorations)
```

Then add the two new methods. Place them in the "Docling extraction" section, right BEFORE `def _extract_with_docling` (before line 157). Add this block:

```python
    def _is_figure_item(self, item: Any) -> bool:
        """Return True if the Docling item is a figure (PICTURE/FIGURE label)."""
        return _LABEL_MAP.get(self._item_label(item)) == "figure"

    def _render_figure_region(
        self,
        fitz_doc: Any,
        item: Any,
        doc_hash: str,
        fig_index: int,
    ) -> Optional[Dict[str, Any]]:
        """Render a Docling Figure item's bbox region to a PNG via PyMuPDF.

        Captures BOTH raster and vector graphics by rendering the page area
        to pixels — unlike the old ``get_images()`` which only found embedded
        raster objects. Returns None when the figure should be skipped (too
        small, missing bbox, or render failure).

        Args:
            fitz_doc: Open ``fitz.Document`` shared across all figures in a parse.
            item: Docling item with label PICTURE/FIGURE and ``prov.bbox``.
            doc_hash: Document hash (for image sub-directory + id).
            fig_index: Global figure counter (for unique image id).

        Returns:
            Standard image dict (same shape as old ``_extract_page_images``
            output), or None if the figure is skipped.
        """
        page_num, bbox = self._provenance(item)
        if not bbox:
            logger.debug(f"Skip figure on page {page_num}: no bbox provenance")
            return None

        width = bbox["x1"] - bbox["x0"]
        height = bbox["bottom"] - bbox["top"]
        if width < self.MIN_FIGURE_SIZE or height < self.MIN_FIGURE_SIZE:
            logger.debug(
                f"Skip small figure on page {page_num}: {width:.0f}x{height:.0f}pt"
            )
            return None

        try:
            page = fitz_doc[page_num - 1]
            rect = fitz.Rect(bbox["x0"], bbox["top"], bbox["x1"], bbox["bottom"])
            pix = page.get_pixmap(clip=rect, dpi=self.RENDER_DPI)

            image_id = self._generate_image_id(doc_hash, page_num, fig_index + 1)
            image_dir = self.image_storage_dir / doc_hash
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"{image_id}.png"
            pix.save(image_path)

            try:
                stored_path = image_path.relative_to(Path.cwd())
            except ValueError:
                stored_path = image_path.absolute()

            return {
                "id": image_id,
                "path": str(stored_path),
                "page": page_num,
                "text_offset": 0,
                "text_length": len(f"[IMAGE: {image_id}]"),
                "position": {
                    "width": pix.width,
                    "height": pix.height,
                    "page": page_num,
                    "index": fig_index,
                },
            }
        except Exception as e:
            logger.warning(f"Failed to render figure on page {page_num}: {e}")
            return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_docling_parser.py::test_render_figure_region_renders_valid_bbox tests/unit/test_docling_parser.py::test_render_figure_region_skips_small_bbox tests/unit/test_docling_parser.py::test_render_figure_region_skips_missing_bbox -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/libs/parser/docling_parser.py tests/unit/test_docling_parser.py
git commit -m "feat(parser): docling _render_figure_region + 类常量 (矢量图渲染准备)"
```

---

### Task 2: Wire into `_extract_with_docling`, remove old `_extract_page_images`

**Files:**
- Modify: `src/libs/parser/docling_parser.py` (rewrite `_extract_with_docling`, delete `_extract_page_images`, update `_item_to_section` comment, remove unused imports)
- Test: `tests/unit/test_docling_parser.py` (add integration-style test, update outdated comments)

**Interfaces:**
- Consumes: `_render_figure_region` and `_is_figure_item` from Task 1
- Produces: A docling parser where Figure items become rendered PNGs + `[IMAGE: id]` figure sections; old raster-only `_extract_page_images` method is gone.

- [ ] **Step 1: Write failing integration test for figure rendering through `parse()`**

Append to `tests/unit/test_docling_parser.py`:

```python
def test_parse_renders_figure_section_from_picture_item(settings, fake_pdf, tmp_path):
    """A PICTURE item with a bbox is rendered → emits a figure section + image dict."""
    items = [
        _make_item("TEXT", text="正文段落", page=1),
        _make_item("PICTURE", page=2, bbox=(50, 50, 550, 400)),  # 500x350 pt
        _make_item("FIGURE", page=3, bbox=(20, 20, 40, 40)),     # 20x20 pt → skipped
    ]
    converter = _make_converter(items)

    pix = MagicMock()
    pix.width = 1000
    pix.height = 700
    page = MagicMock()
    page.get_pixmap = MagicMock(return_value=pix)
    fitz_doc = MagicMock()
    fitz_doc.__getitem__.return_value = page

    with patch("src.libs.parser.docling_parser.DocumentConverter", return_value=converter), \
         patch("src.libs.parser.docling_parser.fitz.open", return_value=fitz_doc), \
         patch("src.libs.parser.docling_parser.fitz.Rect"), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", True):
        parser = DoclingParser(
            settings, collection="t",
            image_storage_dir=str(tmp_path / "images"), extract_images=True,
        )
        doc = parser.parse(fake_pdf)

    sections = doc.metadata["sections"]
    fig_sections = [s for s in sections if s["type"] == "figure"]
    assert len(fig_sections) == 1                       # only the 500x350 figure
    assert fig_sections[0]["text"].startswith("[IMAGE:")
    assert fig_sections[0]["page"] == 2
    images = doc.metadata.get("images", [])
    assert len(images) == 1
    assert images[0]["page"] == 2
    # The small FIGURE (20x20) was NOT rendered
    assert all(img["page"] != 3 for img in images)


def test_parse_no_rendering_when_extract_images_false(settings, fake_pdf, tmp_path):
    """extract_images=False → fitz.open never called, no figure sections."""
    items = [
        _make_item("PICTURE", page=2, bbox=(50, 50, 550, 400)),
    ]
    converter = _make_converter(items)

    with patch("src.libs.parser.docling_parser.DocumentConverter", return_value=converter), \
         patch("src.libs.parser.docling_parser.fitz.open") as mock_open:
        parser = DoclingParser(
            settings, collection="t",
            image_storage_dir=str(tmp_path / "images"), extract_images=False,
        )
        doc = parser.parse(fake_pdf)

    mock_open.assert_not_called()
    sections = doc.metadata["sections"]
    assert all(s["type"] != "figure" for s in sections)
    assert doc.metadata.get("images", []) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_docling_parser.py::test_parse_renders_figure_section_from_picture_item tests/unit/test_docling_parser.py::test_parse_no_rendering_when_extract_images_false -v`
Expected: FAIL — `test_parse_renders_figure_section_from_picture_item` fails because the old code calls `get_images()` (returns nothing on the mock) and skips the PICTURE in `_item_to_section`, so no figure section is emitted. `test_parse_no_rendering_when_extract_images_false` may already pass (old code also skips rendering when `extract_images=False`); if it passes, that's fine — it guards against regression.

- [ ] **Step 3: Rewrite `_extract_with_docling`**

In `src/libs/parser/docling_parser.py`, replace the entire `_extract_with_docling` method (currently lines 157-189) with:

```python
    def _extract_with_docling(
        self, path: Path, doc_hash: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract typed sections and figures via Docling.

        Figures are rendered from each Docling Figure item's bbox via PyMuPDF
        (``get_pixmap(clip=bbox)``) — this captures both raster and vector
        graphics in one pass, replacing the old ``get_images()`` raster-only
        extraction.
        """
        converter = self._build_converter()
        result = converter.convert(str(path))
        ddoc = result.document

        sections: List[Dict[str, Any]] = []
        images: List[Dict[str, Any]] = []

        # Open the fitz document once and share it across all figure renders.
        fitz_doc = (
            fitz.open(path) if (self.extract_images and PYMUPDF_AVAILABLE) else None
        )
        fig_counter = 0
        try:
            for item, _level in ddoc.iterate_items():
                if fitz_doc is not None and self._is_figure_item(item):
                    rendered = self._render_figure_region(
                        fitz_doc, item, doc_hash, fig_counter
                    )
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
                        continue
                sec = self._item_to_section(item, ddoc)
                if sec:
                    sections.append(sec)
        finally:
            if fitz_doc is not None:
                fitz_doc.close()

        return sections, images
```

- [ ] **Step 4: Update the figure-skip comment in `_item_to_section`**

In `src/libs/parser/docling_parser.py`, find the block inside `_item_to_section` (currently lines 221-224):

```python
        stype = _LABEL_MAP.get(label, "text")
        if stype == "figure":
            # Pictures are extracted via PyMuPDF above; skip here to avoid dupes.
            return None
```

Replace the comment line with:

```python
        stype = _LABEL_MAP.get(label, "text")
        if stype == "figure":
            # Figures are rendered in _extract_with_docling above. Reaching here
            # means rendering was skipped (too small / no bbox / images disabled)
            # — drop the item rather than emit an empty figure section.
            return None
```

- [ ] **Step 5: Delete `_extract_page_images` and clean unused imports**

In `src/libs/parser/docling_parser.py`:

1. Delete the entire `_extract_page_images` method (currently lines 287-344), including its section-divider comment block immediately above it (lines 282-285). Stop deleting at the next section divider (`# ---` for "Fallback + helpers").

2. Remove the now-unused imports. Find lines 22-25:
```python
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
```
Remove `import io` (only used by the deleted method). The edited block becomes:
```python
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
```

3. Find line 41 (after the PyMuPDF try/except):
```python
from PIL import Image
```
Delete this line entirely (only used by the deleted method's `Image.open`).

4. Update the module docstring (lines 14-18) — change the last sentence from:
```
extract_images, **kwargs)``. Falls back to ``PdfTextParser`` (MarkItDown) when
docling raises (degradation chain: docling → pdf_text). Image extraction via
PyMuPDF (same image-dict format as PdfTableParser).
```
to:
```
extract_images, **kwargs)``. Falls back to ``PdfTextParser`` (MarkItDown) when
docling raises (degradation chain: docling → pdf_text). Figure regions are
rendered to PNG via PyMuPDF ``get_pixmap(clip=bbox)`` (captures both raster
and vector graphics); the emitted image dicts match PdfTableParser's contract.
```

5. Update the class docstring's image sentence (around line 74-76) from:
```
    (display) fields. Images are extracted via PyMuPDF (Docling picture pixel
    extraction is heavier and not needed here). Falls back to PdfTextParser on
    docling failure.
```
to:
```
    (display) fields. Figure regions are rendered to PNG via PyMuPDF
    ``get_pixmap(clip=bbox)`` (captures raster + vector). Falls back to
    PdfTextParser on docling failure.
```

- [ ] **Step 6: Update outdated comments in existing tests**

In `tests/unit/test_docling_parser.py`, `test_parse_emits_typed_sections` (around lines 68-93):

- Line 74, change:
  ```python
        _make_item("PICTURE", page=3),  # skipped — PyMuPDF handles images
  ```
  to:
  ```python
        _make_item("PICTURE", page=3),  # not rendered (extract_images=False below)
  ```

- Lines 92-93, change:
  ```python
    # PICTURE not emitted as a figure section (images come via PyMuPDF).
    assert "figure" not in types
  ```
  to:
  ```python
    # PICTURE not rendered (extract_images=False → fitz_doc not opened).
    assert "figure" not in types
  ```

- [ ] **Step 7: Run the full docling parser test suite**

Run: `pytest tests/unit/test_docling_parser.py -v`
Expected: ALL tests pass — the 6 new tests from Tasks 1-2 plus the 6 existing tests (`test_docling_parser_is_base_parser`, `test_parse_emits_typed_sections`, `test_table_section_carries_gfm_in_both_fields`, `test_bbox_provenance_mapped`, `test_parse_falls_back_to_pdf_text_on_error`, `test_parse_falls_back_when_no_sections`).

- [ ] **Step 8: Lint and import-check**

Run:
```bash
ruff check src/libs/parser/docling_parser.py tests/unit/test_docling_parser.py
python -m compileall src/libs/parser/docling_parser.py
```
Expected: ruff clean (no unused imports — `io` and `PIL.Image` were removed), compileall succeeds.

- [ ] **Step 9: Commit**

```bash
git add src/libs/parser/docling_parser.py tests/unit/test_docling_parser.py
git commit -m "feat(parser): docling 走 bbox 渲染替代 get_images (矢量图支持)"
```

---

## Verification (End-to-End on Real PDF)

After Task 2, verify the full pipeline with a real PDF containing vector graphics.

- [ ] **V1: Pick a test PDF with known vector graphics**

Locate a paper PDF with charts/diagrams (e.g. matplotlib output). If the repo has fixtures, check `tests/fixtures/`. Otherwise use any local paper PDF. Note its path.

- [ ] **V2: Ingest and check image count**

Run (in conda env `langchain-test`):
```bash
python scripts/ingest.py --path "<your-paper.pdf>" --collection test_vector --force
```
Expected: log line `Images extracted: N` where N > 0 for a paper that previously yielded 0. Check `data/images/test_vector/<doc_hash>/` has PNG files.

- [ ] **V3: Spot-check a rendered figure**

Open one of the rendered PNGs — confirm it shows the actual figure region (not blank, not the whole page, not garbled). If regions are off (wrong area, flipped, empty), the coordinate mapping between Docling bbox and PyMuPDF Rect needs adjustment in `_render_figure_region` — verify `fitz.Rect(x0, top, x1, bottom)` matches Docling's `l/t/r/b` ordering.

- [ ] **V4: Verify caption + retrieval**

If `vision_llm.enabled: true` (Ollama llava-phi3 running), the ingestion log should show ImageCaptioner generating captions. Then query:
```bash
python scripts/query.py --query "<something about a figure>" --collection test_vector --verbose
```
Expected: response includes image content (Base64) for hit chunks whose figures matched.

- [ ] **V5: Run full unit test suite (regression check)**

Run: `pytest tests/unit -v`
Expected: all unit tests pass, no regressions in other parsers or downstream stages.

---

## Self-Review Notes

**Spec coverage:**
- ✅ Drop `get_images()`, use `get_pixmap(clip=bbox)` → Task 2 Step 3
- ✅ DPI=200 hardcoded → Task 1 Step 4 (`RENDER_DPI`)
- ✅ MIN_FIGURE_SIZE=100 filter → Task 1 Step 4 + tested in Task 1 Step 2 and Task 2 Step 1
- ✅ Skip figures without bbox → Task 1 Step 2/4
- ✅ `extract_images` remains the switch → Task 2 Step 3 (`fitz_doc = ... if self.extract_images and PYMUPDF_AVAILABLE`)
- ✅ Only `docling_parser.py` changes → all file paths confirm
- ✅ Image dict shape preserved → Task 1 Step 4 returns identical key set
- ✅ Tests mock Docling + PyMuPDF → Task 1 Step 2, Task 2 Step 1

**Type consistency:** `_render_figure_region(fitz_doc, item, doc_hash, fig_index)` signature identical in Task 1 (defined) and Task 2 (called). `_is_figure_item(item)` same. Image dict keys match the old `_extract_page_images` output and the contract in Global Constraints.
