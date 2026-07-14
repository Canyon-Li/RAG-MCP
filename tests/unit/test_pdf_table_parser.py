"""Unit tests for PdfTableParser (K4′).

Validates:
- table extraction via pdfplumber → sections (HTML + cleaned plain text)
- text outside table bands → text sections; words inside table bands excluded
- "no tables" → plain-text sections, NOT degraded (§7.7)
- pdfplumber exception → fallback to pdf_text + degraded=true (§7.7)
- _table_to_html_and_plain serialization (§15.1)
- pdf_table provider registered with .pdf extension

pdfplumber is mocked so tests don't depend on real PDF fixtures with tables.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.types import Document
from src.libs.parser.pdf_table_parser import PdfTableParser


def _fake_pdf_path(tmp_path) -> Path:
    p = tmp_path / "test.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return p


def _make_page(tables, words):
    """Build a fake pdfplumber page with the given tables and words."""
    page = MagicMock()
    fake_tables = []
    for bbox, data in tables:
        t = MagicMock()
        t.bbox = bbox
        t.extract.return_value = data
        fake_tables.append(t)
    page.find_tables.return_value = fake_tables
    page.extract_words.return_value = words
    return page


class TestTableSerialization:
    """_table_to_html_and_plain (pure, no mocking)."""

    def test_html_and_plain_from_rows(self):
        data = [["A", "B"], ["1", "2"], ["3", "4"]]
        html, plain = PdfTableParser._table_to_html_and_plain(data)
        assert html == (
            "<table><tr><td>A</td><td>B</td></tr>"
            "<tr><td>1</td><td>2</td></tr>"
            "<tr><td>3</td><td>4</td></tr></table>"
        )
        assert "A: 1 | B: 2" in plain
        assert "A: 3 | B: 4" in plain

    def test_none_cells_become_empty(self):
        html, plain = PdfTableParser._table_to_html_and_plain([["A", "B"], [None, "2"]])
        assert "<td></td><td>2</td>" in html
        assert "A:  | B: 2" in plain

    def test_empty_table(self):
        html, plain = PdfTableParser._table_to_html_and_plain([])
        assert html == ""
        assert plain == ""


class TestExtraction:
    """_extract_with_pdfplumber / parse with mocked pdfplumber."""

    @patch("src.libs.parser.pdf_table_parser.pdfplumber")
    def test_table_extracted_to_sections(self, mock_pdfplumber, tmp_path):
        page = _make_page(
            tables=[((0, 100, 600, 200), [["A", "B"], ["1", "2"]])],
            words=[{"text": "Intro", "x0": 0, "x1": 30, "top": 50, "bottom": 60}],
        )
        mock_pdfplumber.open.return_value.__enter__.return_value.pages = [page]

        parser = PdfTableParser(image_storage_dir=str(tmp_path / "imgs"), extract_images=False)
        sections, _ = parser._extract_with_pdfplumber(_fake_pdf_path(tmp_path), "hash")

        table_sections = [s for s in sections if s["type"] == "table"]
        assert len(table_sections) == 1
        assert table_sections[0]["html"].startswith("<table>")
        assert "A: 1 | B: 2" in table_sections[0]["text"]

    @patch("src.libs.parser.pdf_table_parser.pdfplumber")
    def test_text_outside_table_band_becomes_text_section(self, mock_pdfplumber, tmp_path):
        page = _make_page(
            tables=[((0, 100, 600, 200), [["A", "B"], ["1", "2"]])],
            words=[{"text": "Intro", "x0": 0, "x1": 30, "top": 50, "bottom": 60}],
        )
        mock_pdfplumber.open.return_value.__enter__.return_value.pages = [page]

        parser = PdfTableParser(image_storage_dir=str(tmp_path / "imgs"), extract_images=False)
        sections, _ = parser._extract_with_pdfplumber(_fake_pdf_path(tmp_path), "hash")

        text_sections = [s for s in sections if s["type"] == "text"]
        assert len(text_sections) >= 1
        assert "Intro" in text_sections[0]["text"]

    @patch("src.libs.parser.pdf_table_parser.pdfplumber")
    def test_word_inside_table_band_excluded_from_text(self, mock_pdfplumber, tmp_path):
        # word at top=150 (inside table band 100-200) → excluded from text
        page = _make_page(
            tables=[((0, 100, 600, 200), [["A", "B"], ["1", "2"]])],
            words=[{"text": "inside", "x0": 0, "x1": 30, "top": 150, "bottom": 160}],
        )
        mock_pdfplumber.open.return_value.__enter__.return_value.pages = [page]

        parser = PdfTableParser(image_storage_dir=str(tmp_path / "imgs"), extract_images=False)
        sections, _ = parser._extract_with_pdfplumber(_fake_pdf_path(tmp_path), "hash")

        text_sections = [s for s in sections if s["type"] == "text"]
        assert len(text_sections) == 0  # the only word was inside the table band

    @patch("src.libs.parser.pdf_table_parser.pdfplumber")
    def test_no_tables_produces_text_sections_not_degraded(self, mock_pdfplumber, tmp_path):
        page = _make_page(
            tables=[],
            words=[{"text": "plain", "x0": 0, "x1": 20, "top": 50, "bottom": 60}],
        )
        mock_pdfplumber.open.return_value.__enter__.return_value.pages = [page]

        parser = PdfTableParser(image_storage_dir=str(tmp_path / "imgs"), extract_images=False)
        doc = parser.parse(_fake_pdf_path(tmp_path))

        assert "degraded" not in doc.metadata  # NOT degraded (§7.7)
        assert doc.metadata["sections"]
        assert all(s["type"] == "text" for s in doc.metadata["sections"])


class TestFallback:
    """pdfplumber failure → pdf_text + degraded (§7.7)."""

    @patch("src.libs.parser.pdf_table_parser.pdfplumber")
    def test_pdfplumber_exception_falls_back(self, mock_pdfplumber, tmp_path):
        mock_pdfplumber.open.side_effect = RuntimeError("boom")

        parser = PdfTableParser(image_storage_dir=str(tmp_path / "imgs"), extract_images=False)
        # mock the pdf_text fallback so the test doesn't need MarkItDown
        fallback_doc = Document(id="doc_fb", text="fallback", metadata={"source_path": "x"})
        parser._pdf_text = MagicMock()
        parser._pdf_text.parse.return_value = fallback_doc

        doc = parser.parse(_fake_pdf_path(tmp_path))

        assert doc.metadata.get("degraded") is True
        parser._pdf_text.parse.assert_called_once()


class TestProviderRegistration:
    """pdf_table provider registered with .pdf extension."""

    def test_pdf_table_extension(self):
        from src.libs.parser.parser_factory import ParserFactory
        assert ParserFactory.get_supported_extensions("pdf_table") == [".pdf"]

    def test_pdf_table_registered(self):
        from src.libs.parser.parser_factory import ParserFactory
        assert "pdf_table" in ParserFactory.list_providers()
