"""Ticket 03 – OCR 探测路由 against REAL docling (slow).

Complements the mocked unit tests (test_docling_ocr_routing.py): here the
full conversion runs for real. The scanned fixture is built at runtime —
a text page rasterized to pixels and pasted as a full-page image (zero text
layer), exactly the scan signature the probe targets — and docling's OCR
engine (RapidOCR via OcrAutoOptions on this machine) must read the pixels
back into text. The digital fixture (simple.pdf) must route OCR OFF and
still parse.

Marked slow: docling layout/tableformer model load is minutes-order cold.
No LLM/embedding services are touched (parser seam only).
"""

from pathlib import Path

import fitz
import pytest

from src.libs.parser import docling_parser
from src.libs.parser.docling_parser import DoclingParser

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(
        not docling_parser.DOCLING_AVAILABLE, reason="docling not installed"
    ),
]

DIGITAL_FIXTURE = Path("tests/fixtures/sample_documents/simple.pdf")

SCAN_LINES = (
    "OCR PROBE routing test",
    "Unique token ZQX seven",
    "Line three docling OCR",
)


def _scanned_pdf(tmp_path: Path) -> Path:
    """扫描型夹具：真文字页光栅化后整页贴图 —— 文字只在像素里。

    Pure fitz (no PIL): insert_text → get_pixmap → insert_image.
    """
    tmp = fitz.open()
    tp = tmp.new_page(width=595, height=842)
    for i, line in enumerate(SCAN_LINES):
        tp.insert_text((72, 150 + 150 * i), line, fontsize=36)
    pix = tp.get_pixmap(dpi=150)
    pdf = fitz.open()
    page = pdf.new_page(width=595, height=842)
    page.insert_image(page.rect, pixmap=pix)
    p = tmp_path / "scanned.pdf"
    pdf.save(str(p))
    return p


def _parse(pdf_path: Path, tmp_path: Path, **parser_kwargs):
    parser = DoclingParser(
        None, collection="t",
        image_storage_dir=str(tmp_path / "images"),
        extract_images=False, **parser_kwargs,
    )
    return parser, parser.parse(pdf_path)


def test_real_scanned_pdf_routes_ocr_on_and_extracts_text(tmp_path):
    """无文本层夹具 → do_ocr=True 且 OCR 真的把像素读回了文本（工单验收）。"""
    scanned = _scanned_pdf(tmp_path)

    parser, doc = _parse(scanned, tmp_path)  # default ocr_mode=auto

    assert parser.last_ocr_do_ocr is True, "scanned file must route OCR on"
    text = doc.text.upper()
    assert "ZQX" in text, f"OCR must extract the unique token, got: {doc.text!r}"
    assert any(line.upper() in text for line in SCAN_LINES)
    assert doc.metadata["sections"]


def test_real_digital_pdf_routes_ocr_off_and_parses(tmp_path):
    """原生数字 PDF（simple.pdf 有文本层）→ do_ocr=False，解析照常。"""
    assert DIGITAL_FIXTURE.exists(), f"fixture missing: {DIGITAL_FIXTURE}"

    parser, doc = _parse(DIGITAL_FIXTURE, tmp_path)

    assert parser.last_ocr_do_ocr is False, "digital file must route OCR off"
    assert doc.metadata["sections"], "OCR-off parse must still yield sections"
    assert "sample pdf" in doc.text.lower()


def test_real_digital_pdf_forced_always_still_parses(tmp_path):
    """ocr_mode=always（docling 旧行为）在数字 PDF 上照常出文本。"""
    assert DIGITAL_FIXTURE.exists(), f"fixture missing: {DIGITAL_FIXTURE}"

    parser, doc = _parse(DIGITAL_FIXTURE, tmp_path, ocr_mode="always")

    assert parser.last_ocr_do_ocr is True
    assert doc.metadata["sections"]
