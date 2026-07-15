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
        _make_item("PICTURE", page=3),  # skipped — PyMuPDF handles images
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
    # PICTURE not emitted as a figure section (images come via PyMuPDF).
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
