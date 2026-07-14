"""Unit tests for PDF Text Parser contract and behavior.

Tests verify:
- BaseParser abstract interface
- PdfTextParser initialization and configuration
- Helper methods (hash computation, title extraction, etc.)
- Error handling for invalid inputs
- Core PDF conversion functionality using real test files

K1 (DEV_SPEC phase K): renamed from test_loader_pdf_contract.py. PdfLoader was
renamed to PdfTextParser; load() → parse(). Constructor contract unified but
all params have defaults so existing call styles still work.

Note: Additional integration tests are in tests/integration/test_pdf_text_parser_integration.py
"""

from pathlib import Path

import pytest

from src.core.types import Document
from src.libs.parser.base_parser import BaseParser
from src.libs.parser.pdf_text_parser import PdfTextParser


# Test fixtures paths
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sample_documents"
SIMPLE_PDF = FIXTURES_DIR / "simple.pdf"
IMAGES_PDF = FIXTURES_DIR / "with_images.pdf"


class TestBaseParser:
    """Tests for BaseParser abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """BaseParser cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseParser()

    def test_validate_file_existing_file(self, tmp_path):
        """_validate_file returns Path for existing files."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy content")

        # Access static method via subclass
        validated = PdfTextParser._validate_file(test_file)
        assert validated.exists()
        assert validated.is_file()

    def test_validate_file_nonexistent(self):
        """_validate_file raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            PdfTextParser._validate_file("nonexistent_file.pdf")

    def test_validate_file_directory(self, tmp_path):
        """_validate_file raises ValueError for directories."""
        with pytest.raises(ValueError, match="not a file"):
            PdfTextParser._validate_file(tmp_path)


class TestPdfTextParserInitialization:
    """Tests for PdfTextParser initialization."""

    def test_default_initialization(self):
        """PdfTextParser can be initialized with defaults."""
        parser = PdfTextParser()
        assert parser.extract_images is True
        assert parser.image_storage_dir == Path("data/images")

    def test_custom_initialization(self):
        """PdfTextParser respects custom configuration."""
        parser = PdfTextParser(
            extract_images=False,
            image_storage_dir="custom/path"
        )
        assert parser.extract_images is False
        assert parser.image_storage_dir == Path("custom/path")

    def test_markitdown_available(self):
        """PdfTextParser requires MarkItDown to be available."""
        parser = PdfTextParser()
        assert parser._markitdown is not None


class TestPdfTextParserValidation:
    """Tests for input validation."""

    def test_parse_requires_pdf_extension(self, tmp_path):
        """parse() raises ValueError for non-PDF files."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a pdf")

        parser = PdfTextParser()
        with pytest.raises(ValueError, match="not a PDF"):
            parser.parse(txt_file)

    def test_parse_nonexistent_file(self):
        """parse() raises FileNotFoundError for missing files."""
        parser = PdfTextParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("nonexistent.pdf")


class TestPdfTextParserHelperMethods:
    """Tests for helper methods."""

    def test_compute_file_hash_consistency(self, tmp_path):
        """_compute_file_hash returns consistent hash for same content."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"consistent content")

        parser = PdfTextParser()
        hash1 = parser._compute_file_hash(test_file)
        hash2 = parser._compute_file_hash(test_file)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_compute_file_hash_differs_for_different_content(self, tmp_path):
        """_compute_file_hash returns different hashes for different content."""
        file1 = tmp_path / "file1.pdf"
        file2 = tmp_path / "file2.pdf"
        file1.write_bytes(b"content 1")
        file2.write_bytes(b"content 2")

        parser = PdfTextParser()
        hash1 = parser._compute_file_hash(file1)
        hash2 = parser._compute_file_hash(file2)

        assert hash1 != hash2

    def test_extract_title_from_markdown_heading(self):
        """_extract_title finds first Markdown heading."""
        parser = PdfTextParser()

        text = "# Main Title\n\nParagraph text\n\n## Subtitle"
        title = parser._extract_title(text)
        assert title == "Main Title"

    def test_extract_title_from_first_line(self):
        """_extract_title uses first non-empty line as fallback."""
        parser = PdfTextParser()

        text = "First Line Title\n\nParagraph text"
        title = parser._extract_title(text)
        assert title == "First Line Title"

    def test_extract_title_handles_empty_text(self):
        """_extract_title returns None for empty text."""
        parser = PdfTextParser()

        text = ""
        title = parser._extract_title(text)
        assert title is None

    def test_generate_image_id_format(self):
        """_generate_image_id creates consistent ID format."""
        image_id = PdfTextParser._generate_image_id("abc123def456", 2, 0)
        assert image_id == "abc123de_2_0"


class TestPdfConversionCore:
    """Tests for core PDF conversion functionality using real PDF files."""

    def test_convert_simple_pdf_to_text(self):
        """Convert simple PDF to text - verifies core Markdown conversion."""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        parser = PdfTextParser()
        doc = parser.parse(SIMPLE_PDF)

        # Verify Document structure
        assert isinstance(doc, Document)
        assert doc.id.startswith("doc_")

        # Verify text content is extracted
        assert len(doc.text) > 0
        assert isinstance(doc.text, str)

        # Verify expected content is present (from our generated PDF)
        text_lower = doc.text.lower()
        assert "sample" in text_lower or "document" in text_lower
        assert "test" in text_lower or "pdf" in text_lower

        # Verify metadata
        assert doc.metadata["source_path"] == str(SIMPLE_PDF)
        assert doc.metadata["doc_type"] == "pdf"
        assert "doc_hash" in doc.metadata

    def test_extract_title_from_pdf(self):
        """Verify title extraction from real PDF."""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        parser = PdfTextParser()
        doc = parser.parse(SIMPLE_PDF)

        # Should extract title (either from heading or first line)
        assert "title" in doc.metadata
        assert doc.metadata["title"] is not None
        assert len(doc.metadata["title"]) > 0

        # Title should contain relevant keywords
        title_lower = doc.metadata["title"].lower()
        assert "sample" in title_lower or "document" in title_lower

    def test_pdf_with_images_structure(self):
        """Verify PDF with images is processed correctly with image extraction."""
        if not IMAGES_PDF.exists():
            pytest.skip(f"Test fixture not found: {IMAGES_PDF}")

        parser = PdfTextParser(extract_images=True)
        doc = parser.parse(IMAGES_PDF)

        # Verify basic structure
        assert isinstance(doc, Document)
        assert len(doc.text) > 0

        # Verify images were extracted
        assert "images" in doc.metadata
        assert isinstance(doc.metadata["images"], list)

        if len(doc.metadata["images"]) > 0:
            # Verify image metadata structure
            for img in doc.metadata["images"]:
                assert "id" in img
                assert "path" in img
                assert "page" in img
                assert "text_offset" in img
                assert "text_length" in img
                assert "position" in img

                # Verify image file exists
                img_path = Path(img["path"])
                assert img_path.exists(), f"Image file should exist: {img_path}"

                # Verify placeholder exists in text
                placeholder = f"[IMAGE: {img['id']}]"
                assert placeholder in doc.text, f"Placeholder {placeholder} should be in text"

    def test_image_extraction_disabled(self):
        """Verify image extraction can be disabled."""
        if not IMAGES_PDF.exists():
            pytest.skip(f"Test fixture not found: {IMAGES_PDF}")

        parser = PdfTextParser(extract_images=False)
        doc = parser.parse(IMAGES_PDF)

        # Should still extract text
        assert len(doc.text) > 0

        # Should not have images metadata
        assert "images" not in doc.metadata or doc.metadata.get("images") == []

    def test_document_hash_consistency(self):
        """Verify same PDF produces same document hash."""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        parser = PdfTextParser()

        # Parse same file twice
        doc1 = parser.parse(SIMPLE_PDF)
        doc2 = parser.parse(SIMPLE_PDF)

        # Should produce identical hashes (for idempotency)
        assert doc1.metadata["doc_hash"] == doc2.metadata["doc_hash"]
        assert doc1.id == doc2.id

    def test_document_serialization(self):
        """Verify parsed document can be serialized."""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        parser = PdfTextParser()
        doc = parser.parse(SIMPLE_PDF)

        # Serialize to dict
        doc_dict = doc.to_dict()
        assert isinstance(doc_dict, dict)
        assert "id" in doc_dict
        assert "text" in doc_dict
        assert "metadata" in doc_dict

        # Verify metadata is complete
        assert "source_path" in doc_dict["metadata"]
        assert "doc_type" in doc_dict["metadata"]
        assert doc_dict["metadata"]["doc_type"] == "pdf"

        # Verify can recreate from dict
        doc_recreated = Document.from_dict(doc_dict)
        assert doc_recreated.id == doc.id
        assert doc_recreated.text == doc.text
        assert doc_recreated.metadata == doc.metadata
