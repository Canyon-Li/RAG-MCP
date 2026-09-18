"""Unit tests for DoclingParser.

Mock Docling's DocumentConverter / DoclingDocument so tests don't depend on the
heavy docling model stack. Covers: typed-section emission, table GFM in both
text+html, picture-skip, label mapping, bbox, and pdf_text fallback.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.libs.parser.base_parser import BaseParser
from src.libs.parser.docling_parser import DoclingParser
from src.libs.parser.pdf_text_parser import PdfTextParser
from src.core.types import Document


def _make_item(label: str, text=None, page=1, table_md=None, bbox=None):
    """Build a mock Docling item with label/text/prov (and table export)."""
    item = MagicMock()
    lbl = MagicMock()
    lbl.name = label
    item.label = lbl
    item.text = text
    prov = MagicMock()
    prov.page_no = page
    if bbox:
        prov.bbox = MagicMock(l=bbox[0], t=bbox[1], r=bbox[2], b=bbox[3])
    else:
        prov.bbox = None
    item.prov = [prov]
    if table_md is not None:
        item.export_to_markdown = MagicMock(return_value=table_md)
    return item


def _make_converter(items):
    """Mock DocumentConverter whose convert().document.iterate_items yields items."""
    ddoc = MagicMock()
    ddoc.iterate_items = MagicMock(return_value=[(it, 0) for it in items])
    result = MagicMock()
    result.document = ddoc
    converter = MagicMock()
    converter.convert.return_value = result
    return converter


@pytest.fixture
def settings():
    return MagicMock()


@pytest.fixture
def fake_pdf(tmp_path):
    p = tmp_path / "fake.pdf"
    p.write_bytes(b"%PDF-1.4 fake content")
    return p


def test_docling_parser_is_base_parser(settings):
    p = DoclingParser(
        settings, collection="t",
        image_storage_dir="data/images/t", extract_images=False,
    )
    assert isinstance(p, BaseParser)


def test_parse_emits_typed_sections(settings, fake_pdf):
    items = [
        _make_item("SECTION_HEADER", text="第一章 架构", page=1),
        _make_item("TEXT", text="正文段落内容", page=1, bbox=(10, 20, 30, 40)),
        _make_item("TABLE", page=2, table_md="| 列1 | 列2 |\n|---|---|\n| a | b |"),
        _make_item("CAPTION", text="表 1：示例", page=2),
        _make_item("PICTURE", page=3),  # not rendered (extract_images=False below)
    ]
    converter = _make_converter(items)

    with patch("src.libs.parser.docling_parser.DocumentConverter", return_value=converter), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False):
        parser = DoclingParser(
            settings, collection="t",
            image_storage_dir="data/images/t", extract_images=False,
        )
        doc = parser.parse(fake_pdf)

    sections = doc.metadata["sections"]
    types = [s["type"] for s in sections]
    assert "title" in types
    assert "text" in types
    assert "table" in types
    assert "figure_caption" in types
    # PICTURE not rendered (extract_images=False → fitz_doc not opened).
    assert "figure" not in types


def test_table_section_carries_gfm_in_both_fields(settings, fake_pdf):
    gfm = "| 列1 | 列2 |\n|---|---|\n| a | b |"
    items = [_make_item("TABLE", page=2, table_md=gfm)]
    converter = _make_converter(items)

    with patch("src.libs.parser.docling_parser.DocumentConverter", return_value=converter), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False):
        parser = DoclingParser(
            settings, collection="t",
            image_storage_dir="data/images/t", extract_images=False,
        )
        doc = parser.parse(fake_pdf)

    table_sec = doc.metadata["sections"][0]
    assert table_sec["type"] == "table"
    assert table_sec["text"] == gfm
    assert table_sec["html"] == gfm  # Docling yields GFM directly (not HTML)
    assert table_sec["page"] == 2


def test_bbox_provenance_mapped(settings, fake_pdf):
    items = [_make_item("TEXT", text="hi", page=5, bbox=(1.0, 2.0, 3.0, 4.0))]
    converter = _make_converter(items)

    with patch("src.libs.parser.docling_parser.DocumentConverter", return_value=converter), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False):
        parser = DoclingParser(
            settings, collection="t",
            image_storage_dir="data/images/t", extract_images=False,
        )
        doc = parser.parse(fake_pdf)

    sec = doc.metadata["sections"][0]
    assert sec["bbox"] == {"x0": 1.0, "top": 2.0, "x1": 3.0, "bottom": 4.0}
    assert sec["page"] == 5


def test_parse_falls_back_to_pdf_text_on_error(settings, fake_pdf):
    fake_doc = Document(
        id="doc_fallback",
        text="fallback text",
        metadata={"source_path": str(fake_pdf), "doc_type": "pdf"},
    )
    converter = MagicMock()
    converter.convert.side_effect = RuntimeError("docling boom")

    with patch("src.libs.parser.docling_parser.DocumentConverter", return_value=converter), \
         patch.object(PdfTextParser, "parse", return_value=fake_doc) as mock_parse:
        parser = DoclingParser(
            settings, collection="t",
            image_storage_dir="data/images/t", extract_images=False,
        )
        doc = parser.parse(fake_pdf)

    assert mock_parse.called
    assert doc.text == "fallback text"
    assert doc.metadata.get("degraded") is True


def test_parse_falls_back_when_no_sections(settings, fake_pdf):
    # docling succeeds but yields zero usable items → fallback
    converter = _make_converter([])

    fake_doc = Document(
        id="doc_fallback",
        text="fallback",
        metadata={"source_path": str(fake_pdf), "doc_type": "pdf"},
    )
    with patch("src.libs.parser.docling_parser.DocumentConverter", return_value=converter), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False), \
         patch.object(PdfTextParser, "parse", return_value=fake_doc):
        parser = DoclingParser(
            settings, collection="t",
            image_storage_dir="data/images/t", extract_images=False,
        )
        doc = parser.parse(fake_pdf)

    assert doc.metadata.get("degraded") is True


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


def test_render_figure_region_flips_y_axis_for_docling_bbox(settings, tmp_path):
    """Docling's prov.bbox uses PDF bottom-left origin (t > b); the size filter
    must take abs() and the PyMuPDF Rect must flip y via page height.

    Real values from a quantum-circuit paper (page 3, A4): l=191.6 t=339.5
    r=404.4 b=232.5 — height (b-t) is -107, so without abs() this is silently
    filtered as "too small", and without the y-flip the rendered region is wrong.
    """
    parser = DoclingParser(
        settings, collection="t",
        image_storage_dir=str(tmp_path / "images"), extract_images=True,
    )
    pix = MagicMock()
    pix.width = 1000
    pix.height = 700
    page = MagicMock()
    page.get_pixmap = MagicMock(return_value=pix)
    page.rect = MagicMock(height=841.9)  # A4 page height in points
    fitz_doc = MagicMock()
    fitz_doc.__getitem__.return_value = page

    # Docling bottom-left origin: t=339.5 > b=232.5 (visual height = t-b = 107pt)
    item = _make_item("PICTURE", page=3, bbox=(191.6, 339.5, 404.4, 232.5))

    with patch("src.libs.parser.docling_parser.fitz.Rect") as mock_rect:
        img = parser._render_figure_region(fitz_doc, item, "abcd1234ef", 0)

    # NOT filtered despite negative (b - t)
    assert img is not None
    # y-axis flipped via page height: Rect(l, page_h - t, r, page_h - b)
    mock_rect.assert_called_once_with(191.6, 841.9 - 339.5, 404.4, 841.9 - 232.5)
    page.get_pixmap.assert_called_once()


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
    """extract_images=False → no figure rendering, no figure sections.

    NOTE: _page_count may still open the PDF via fitz (page-batched conversion
    needs the page count regardless of images) — that read-only probe is not
    image extraction, so this test mocks it out and asserts no figure output.
    """
    items = [
        _make_item("TEXT", text="正文段落", page=1),  # non-figure so sections non-empty
        _make_item("PICTURE", page=2, bbox=(50, 50, 550, 400)),
    ]
    converter = _make_converter(items)

    with patch("src.libs.parser.docling_parser.DocumentConverter", return_value=converter), \
         patch.object(DoclingParser, "_page_count", return_value=None):
        parser = DoclingParser(
            settings, collection="t",
            image_storage_dir=str(tmp_path / "images"), extract_images=False,
        )
        doc = parser.parse(fake_pdf)

    sections = doc.metadata["sections"]
    assert all(s["type"] != "figure" for s in sections)
    assert doc.metadata.get("images", []) == []


def test_parse_no_rendering_when_pymupdf_unavailable(settings, fake_pdf, tmp_path):
    """PYMUPDF_AVAILABLE=False + extract_images=True → fitz.open not called,
    PICTURE falls through (no figure section, no images)."""
    items = [
        _make_item("TEXT", text="正文段落", page=1),
        _make_item("PICTURE", page=2, bbox=(50, 50, 550, 400)),
    ]
    converter = _make_converter(items)

    with patch("src.libs.parser.docling_parser.DocumentConverter", return_value=converter), \
         patch("src.libs.parser.docling_parser.fitz.open") as mock_open, \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False):
        parser = DoclingParser(
            settings, collection="t",
            image_storage_dir=str(tmp_path / "images"), extract_images=True,
        )
        doc = parser.parse(fake_pdf)

    mock_open.assert_not_called()
    sections = doc.metadata["sections"]
    assert all(s["type"] != "figure" for s in sections)
    assert doc.metadata.get("images", []) == []


def test_render_figure_region_returns_none_on_render_failure(settings, tmp_path):
    """If page.get_pixmap raises, _render_figure_region catches and returns None."""
    parser = DoclingParser(
        settings, collection="t",
        image_storage_dir=str(tmp_path / "images"), extract_images=True,
    )
    page = MagicMock()
    page.get_pixmap = MagicMock(side_effect=RuntimeError("render boom"))
    fitz_doc = MagicMock()
    fitz_doc.__getitem__.return_value = page
    item = _make_item("PICTURE", page=1, bbox=(50, 50, 550, 400))  # 500x350 pt

    with patch("src.libs.parser.docling_parser.fitz.Rect"):
        img = parser._render_figure_region(fitz_doc, item, "abcd1234ef", 0)

    assert img is None


# ── Page-batched conversion (docling memory accumulation workaround) ──────


def _pdf_page_count(path) -> int:
    """Total pages of a test PDF (fitz-free: count '/Type /Page' markers)."""
    import re
    data = Path(path).read_bytes()
    return len(re.findall(rb"/Type\s*/Page[^s]", data)) or 1


def _make_batch_converter(items_by_batch):
    """DocumentConverter factory mock: each call returns a NEW converter that
    yields the items of the next batch (len(items_by_batch) batches total).
    Records (converter, page_range kwargs) per convert() call for assertions."""
    calls = {"n": 0, "log": []}

    class _Factory:
        def __call__(self, **_ctor_kwargs):
            # Accepts ctor kwargs (real _build_converter passes format_options,
            # ticket 03) and drops them — this mock counts constructions.
            idx = calls["n"]
            calls["n"] += 1
            conv = _make_converter(items_by_batch[idx])
            # track this converter's convert calls in the shared log
            orig_convert = conv.convert

            def _convert(*a, **kw):
                calls["log"].append({"batch": idx, "kwargs": kw})
                return orig_convert(*a, **kw)

            conv.convert = _convert
            return conv

    factory = _Factory()
    return factory, calls


def test_batching_disabled_for_short_docs(settings, fake_pdf, tmp_path):
    """Docs with fewer pages than page_batch_size convert in ONE call
    (single converter, no page_range restriction) — existing behaviour."""
    # 12-page PDF, batch size 8 → needs batching; use > to keep this test
    # about the SHORT case: build a 3-page PDF by patching _page_count.
    items = [_make_item("TEXT", text="all pages", page=2)]
    converter = _make_converter(items)

    with patch("src.libs.parser.docling_parser.DocumentConverter", return_value=converter), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False), \
         patch.object(DoclingParser, "_page_count", return_value=3):
        parser = DoclingParser(
            settings, collection="t",
            image_storage_dir=str(tmp_path / "images"), extract_images=False,
        )
        doc = parser.parse(fake_pdf)

    assert converter.convert.call_count == 1
    # no page_range restriction on the single call
    assert converter.convert.call_args[1].get("page_range", None) is None
    assert len(doc.metadata["sections"]) == 1


def test_long_doc_converted_in_batches_with_fresh_converters(
    settings, fake_pdf, tmp_path
):
    """Pages > page_batch_size: convert() called once per page batch, each
    with its 1-based inclusive page_range, and a NEW converter per batch."""
    # 10-page doc, batch size 8 → batches (1,8) and (9,10)
    batch_items = [
        [_make_item("TEXT", text=f"page {p}", page=p) for p in (1, 3, 8)],
        [_make_item("TEXT", text=f"page {p}", page=p) for p in (9, 10)],
    ]
    factory, calls = _make_batch_converter(batch_items)

    with patch("src.libs.parser.docling_parser.DocumentConverter", factory), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False), \
         patch.object(DoclingParser, "_page_count", return_value=10):
        parser = DoclingParser(
            settings, collection="t",
            image_storage_dir=str(tmp_path / "images"), extract_images=False,
        )
        doc = parser.parse(fake_pdf)

    assert calls["n"] == 2, "one fresh DocumentConverter per batch"
    # each convert() call got its 1-based inclusive page_range, in order
    ranges = [entry["kwargs"]["page_range"] for entry in calls["log"]]
    assert ranges == [(1, 8), (9, 10)]
    texts = [s["text"] for s in doc.metadata["sections"]]
    assert texts == ["page 1", "page 3", "page 8", "page 9", "page 10"]


def test_partial_batch_failure_keeps_earlier_batches(
    settings, fake_pdf, tmp_path
):
    """If a LATER batch's convert() raises, sections from earlier batches
    survive — no whole-document fallback when some content was extracted."""
    batch_items = [
        [_make_item("TEXT", text="page 1", page=1)],
    ]
    factory, _ = _make_batch_converter(batch_items)
    boom = MagicMock()
    boom.convert.side_effect = RuntimeError("bad_alloc on batch 2")

    converters = iter([factory(), boom])

    with patch(
        "src.libs.parser.docling_parser.DocumentConverter",
        side_effect=lambda **_kw: next(converters),
    ), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False), \
         patch.object(DoclingParser, "_page_count", return_value=10):
        parser = DoclingParser(
            settings, collection="t",
            image_storage_dir=str(tmp_path / "images"), extract_images=False,
        )
        doc = parser.parse(fake_pdf)

    texts = [s["text"] for s in doc.metadata["sections"]]
    assert texts == ["page 1"]
    assert doc.metadata.get("degraded") is not True  # not a pdf_text fallback


def test_page_batch_size_is_configurable(settings, fake_pdf, tmp_path):
    """page_batch_size kwarg changes the batching cadence."""
    batch_items = [
        [_make_item("TEXT", text="page 1", page=1)],
        [_make_item("TEXT", text="page 2", page=2)],
    ]
    factory, calls = _make_batch_converter(batch_items)

    with patch("src.libs.parser.docling_parser.DocumentConverter", factory), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False), \
         patch.object(DoclingParser, "_page_count", return_value=2):
        parser = DoclingParser(
            settings, collection="t",
            image_storage_dir=str(tmp_path / "images"), extract_images=False,
            page_batch_size=1,
        )
        doc = parser.parse(fake_pdf)

    assert calls["n"] == 2, "batch size 1 over a 2-page doc → 2 converters"
    texts = [s["text"] for s in doc.metadata["sections"]]
    assert texts == ["page 1", "page 2"]


# ====================================================================
# Ticket 04 (D-037): expose the raw DoclingDocuments for HybridChunker
# ====================================================================

def test_parse_exposes_docling_documents_for_hybrid_chunker(settings, fake_pdf):
    """The chunker adapter needs the DoclingDocument objects (not just the
    collapsed sections) — parse() attaches them under metadata so the same
    object stream feeds fresh parses and cache replays alike."""
    items = [
        _make_item("SECTION_HEADER", text="第一章", page=1),
        _make_item("TEXT", text="正文", page=1),
    ]
    converter = _make_converter(items)

    with patch("src.libs.parser.docling_parser.DocumentConverter", return_value=converter), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False):
        parser = DoclingParser(
            settings, collection="t",
            image_storage_dir="data/images/t", extract_images=False,
        )
        doc = parser.parse(fake_pdf)

    ddocs = doc.metadata.get("docling_documents")
    assert ddocs, "metadata['docling_documents'] must carry the parsed DoclingDocuments"
    # Same object(s) the converter produced — identity, not a re-serialization.
    assert ddocs[0] is converter.convert.return_value.document


def test_fallback_document_has_no_docling_documents(settings, fake_pdf):
    """Degraded (pdf_text) documents have no docling objects — the chunker
    must fall back to the original splitter for them (ticket 04)."""
    items: list = []
    converter = _make_converter(items)

    with patch("src.libs.parser.docling_parser.DocumentConverter", return_value=converter), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False):
        parser = DoclingParser(
            settings, collection="t",
            image_storage_dir="data/images/t", extract_images=False,
        )
        doc = parser.parse(fake_pdf)

    assert doc.metadata.get("degraded") is True
    assert "docling_documents" not in doc.metadata


def test_rendered_image_carries_prov_bbox(settings, tmp_path):
    """Ticket 04: the hybrid chunker matches chunk picture-items to rendered
    images by (page, bbox) — the rendered image dict must carry the Docling
    prov bbox (PDF points, bottom-left origin, same shape as section bboxes)."""
    parser = DoclingParser(
        settings, collection="t",
        image_storage_dir=str(tmp_path / "images"), extract_images=True,
    )
    pix = MagicMock()
    pix.width = 1000
    pix.height = 700
    page = MagicMock()
    page.get_pixmap = MagicMock(return_value=pix)
    page.rect = MagicMock(height=841.9)
    fitz_doc = MagicMock()
    fitz_doc.__getitem__.return_value = page

    item = _make_item("PICTURE", page=2, bbox=(50.0, 400.0, 550.0, 50.0))

    with patch("src.libs.parser.docling_parser.fitz.Rect"):
        img = parser._render_figure_region(fitz_doc, item, "abcd1234ef", 0)

    assert img is not None
    assert img["bbox"] == {"x0": 50.0, "top": 400.0, "x1": 550.0, "bottom": 50.0}
