"""DoclingParser OCR 探测路由 tests (ticket 03).

Seam: ``DoclingParser.parse`` → ``_decide_ocr`` → ``_build_converter(do_ocr)``.
The converter is a counting probe (no docling models), but the probe records
the REAL ``format_options`` the parser builds — so these tests pin the full
wiring: probe decision → ``PdfPipelineOptions.do_ocr`` → DocumentConverter.

Fixtures are REAL fitz PDFs (text layer present / absent / corrupt), so the
probe itself runs for real. Routing acceptance (ticket 03):
- 原生数字 PDF → converter 收到 ``do_ocr=False``
- 扫描型（无文本层）→ ``do_ocr=True`` 且 parse 正常产出
- 探测失败（文件读不出）→ 保守降级 OCR 开、不阻断摄入、日志可查
- 配置三态 auto / always / never 生效
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest
from docling.datamodel.base_models import InputFormat

from src.libs.parser.docling_parser import DoclingParser

SPECS = [
    {"label": "TEXT", "text": "正文段落内容", "page": 1,
     "table_md": None, "bbox": (10, 20, 30, 40)},
]


# ---------------------------------------------------------------------------
# Fixtures: real fitz PDFs
# ---------------------------------------------------------------------------


def _digital_pdf(tmp_path: Path, pages: int = 2) -> Path:
    """原生数字 PDF：每页都有真实文本层（无图片）。"""
    p = tmp_path / "digital.pdf"
    pdf = fitz.open()
    for i in range(pages):
        page = pdf.new_page()
        for j in range(8):
            page.insert_text(
                (72, 100 + 30 * j),
                f"Page {i + 1} line {j + 1}: hybrid retrieval architecture notes",
                fontsize=11,
            )
    pdf.save(str(p))
    return p


def _scanned_pdf(tmp_path: Path) -> Path:
    """扫描型 PDF：整页贴一张光栅图、零文本层（ticket 03 的探测目标）。"""
    p = tmp_path / "scanned.pdf"
    # 临时页写真实文字 → 光栅化 → 新页整页贴图（文字只存在于像素里）
    tmp = fitz.open()
    tp = tmp.new_page(width=595, height=842)
    tp.insert_text((72, 150), "rasterized content page", fontsize=36)
    pix = tp.get_pixmap(dpi=150)
    pdf = fitz.open()
    page = pdf.new_page(width=595, height=842)
    page.insert_image(page.rect, pixmap=pix)
    pdf.save(str(p))
    return p


def _cover_image_pdf(tmp_path: Path) -> Path:
    """混合型：p1 = 无文本整页大图（封面），p2 = 真实文本层。"""
    p = tmp_path / "cover_image.pdf"
    tmp = fitz.open()
    tp = tmp.new_page(width=595, height=842)
    tp.insert_text((72, 400), "cover art", fontsize=48)
    pix = tp.get_pixmap(dpi=96)
    pdf = fitz.open()
    p1 = pdf.new_page(width=595, height=842)
    p1.insert_image(p1.rect, pixmap=pix)
    p2 = pdf.new_page(width=595, height=842)
    for j in range(8):
        p2.insert_text((72, 100 + 30 * j), f"digital body line {j + 1}", fontsize=11)
    pdf.save(str(p))
    return p


# ---------------------------------------------------------------------------
# Converter probe: records the real format_options each construction got
# ---------------------------------------------------------------------------


class _OcrProbe:
    """DocumentConverter stand-in — records ctor kwargs, returns spec docs."""

    def __init__(self, specs_by_batch):
        self.specs_by_batch = specs_by_batch
        self.ctor_kwargs: list[dict] = []
        self.converted = 0

    def factory(self):
        probe = self

        class _Converter:
            def __init__(self, **kwargs):
                probe.ctor_kwargs.append(kwargs)

            def convert(self, *args, **kwargs):
                probe.converted += 1
                idx = min(probe.converted - 1, len(probe.specs_by_batch) - 1)
                result = MagicMock()
                result.document = _make_doc(probe.specs_by_batch[idx])
                return result

        return _Converter


def _make_doc(specs):
    ddoc = MagicMock()
    items = []
    for s in specs:
        item = MagicMock()
        lbl = MagicMock()
        lbl.name = s["label"]
        item.label = lbl
        item.text = s["text"]
        prov = MagicMock()
        prov.page_no = s["page"]
        prov.bbox = None
        item.prov = [prov]
        items.append((item, 0))
    ddoc.iterate_items = MagicMock(return_value=items)
    return ddoc


def _do_ocr_sent(probe: _OcrProbe) -> bool:
    """The do_ocr the real PdfPipelineOptions carried into the converter."""
    assert probe.ctor_kwargs, "no DocumentConverter was constructed"
    fmt_options = probe.ctor_kwargs[0]["format_options"]
    return fmt_options[InputFormat.PDF].pipeline_options.do_ocr


def _parse(pdf_path: Path, tmp_path: Path, probe: _OcrProbe, **parser_kwargs):
    with patch("src.libs.parser.docling_parser.DocumentConverter", probe.factory()):
        parser = DoclingParser(
            MagicMock(), collection="t",
            image_storage_dir=str(tmp_path / "images"),
            extract_images=False, **parser_kwargs,
        )
        doc = parser.parse(pdf_path)
    return parser, doc


# ---------------------------------------------------------------------------
# Acceptance: auto routing (probe the text layer per file)
# ---------------------------------------------------------------------------


def test_digital_pdf_auto_routes_do_ocr_false(tmp_path):
    """原生数字 PDF（有文本层）→ converter 收到 do_ocr=False（提速、零损失）。"""
    pdf = _digital_pdf(tmp_path)
    probe = _OcrProbe([SPECS])
    parser, doc = _parse(pdf, tmp_path, probe)

    assert _do_ocr_sent(probe) is False
    assert parser.last_ocr_do_ocr is False
    assert doc.metadata["sections"], "digital parse must still produce sections"


def test_scanned_pdf_auto_routes_do_ocr_true(tmp_path):
    """扫描型（前几页文本量≈零 + 整页大图）→ 该文件 do_ocr=True。"""
    pdf = _scanned_pdf(tmp_path)
    probe = _OcrProbe([SPECS])
    parser, doc = _parse(pdf, tmp_path, probe)

    assert _do_ocr_sent(probe) is True
    assert parser.last_ocr_do_ocr is True
    assert doc.metadata["sections"], "routed parse must still produce sections"


def test_cover_image_with_text_pages_routes_do_ocr_off(tmp_path):
    """p1 无文本整页大图 + p2 有文本 → 判定有文本层（任一探测页有足量文本），
    不因封面图误开 OCR。"""
    pdf = _cover_image_pdf(tmp_path)
    probe = _OcrProbe([SPECS])
    parser, _ = _parse(pdf, tmp_path, probe)

    assert _do_ocr_sent(probe) is False


def test_probe_failure_degrades_conservatively(tmp_path, caplog):
    """探测本身失败（页数/文件读不出）→ 保守开 OCR、不阻断摄入、日志可查。"""
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4 not a real pdf")
    probe = _OcrProbe([SPECS])

    with caplog.at_level(logging.WARNING, logger="src.libs.parser.docling_parser"):
        parser, doc = _parse(corrupt, tmp_path, probe)

    assert parser.last_ocr_do_ocr is True, "probe failure must degrade to OCR on"
    assert _do_ocr_sent(probe) is True
    assert doc.metadata["sections"], "probe failure must not block ingestion"
    assert any("OCR probe failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Acceptance: three-state config (auto / always / never)
# ---------------------------------------------------------------------------


def test_ocr_mode_always_forces_ocr_on_digital(tmp_path):
    pdf = _digital_pdf(tmp_path)
    probe = _OcrProbe([SPECS])
    parser, _ = _parse(pdf, tmp_path, probe, ocr_mode="always")

    assert _do_ocr_sent(probe) is True
    assert parser.last_ocr_do_ocr is True


def test_ocr_mode_never_forces_ocr_off_scanned(tmp_path):
    pdf = _scanned_pdf(tmp_path)
    probe = _OcrProbe([SPECS])
    parser, _ = _parse(pdf, tmp_path, probe, ocr_mode="never")

    assert _do_ocr_sent(probe) is False
    assert parser.last_ocr_do_ocr is False


def test_invalid_ocr_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="ocr_mode"):
        DoclingParser(
            MagicMock(), collection="t",
            image_storage_dir=str(tmp_path / "images"),
            extract_images=False, ocr_mode="sometimes",
        )


# ---------------------------------------------------------------------------
# Cache hygiene: OCR policy participates in the parse-cache version stamp
# ---------------------------------------------------------------------------


def test_version_stamp_changes_with_ocr_mode(tmp_path):
    """切换 OCR 策略必须使旧缓存失效（同文件解析产物可能不同）。"""
    common = dict(
        collection="t",
        image_storage_dir=str(tmp_path / "images"),
        extract_images=False,
    )
    stamps = {
        DoclingParser(MagicMock(), ocr_mode=mode, **common)._version_stamp()
        for mode in ("auto", "always", "never")
    }
    assert len(stamps) == 3


def test_parser_valid_ocr_modes_match_settings(tmp_path):
    """交叉钉住：DoclingParser.VALID_OCR_MODES ≡ settings.PARSER_OCR_MODES。

    两个字面量各自独立定义（libs 不在 import 期依赖 core，见 factory 注释），
    没有这条测试它们可以悄悄漂移——settings 放行的值 parser 拒绝（或反之）。
    """
    from src.core.settings import PARSER_OCR_MODES

    assert DoclingParser.VALID_OCR_MODES == PARSER_OCR_MODES


def test_version_stamp_tracks_probe_thresholds(tmp_path):
    """探测阈值改动必须换戳（D-036 第一类成分：阈值变了同一文件的 JSON 就变）。"""
    parser = DoclingParser(
        MagicMock(), collection="t",
        image_storage_dir=str(tmp_path / "images"),
        extract_images=False,
    )
    base = parser._version_stamp()
    parser.OCR_PROBE_PAGES = 9
    parser.OCR_TEXT_MIN_CHARS = 99
    assert parser._version_stamp() != base
