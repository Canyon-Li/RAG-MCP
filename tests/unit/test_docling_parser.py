"""Unit tests for DoclingParser.

Mock Docling's DocumentConverter / DoclingDocument so tests don't depend on the
heavy docling model stack. Covers: typed-section emission, table GFM in both
text+html, picture-skip, label mapping, bbox, and pdf_text fallback.
"""

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
    """extract_images=False → fitz.open never called, no figure sections."""
    items = [
        _make_item("TEXT", text="正文段落", page=1),  # non-figure so sections non-empty
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
