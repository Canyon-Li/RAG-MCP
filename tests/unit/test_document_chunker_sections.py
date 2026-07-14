"""Unit tests for DocumentChunker K2 section-aware splitting.

Validates the section-aware path (pdf改进计划.md §15.2 / §15.1 / §17.2):
- tables kept whole with chunk.text = cleaned plain text + metadata.table_html
- oversized tables split at row boundaries (table_html only on the first chunk)
- titles merge into the following text run
- headers/footers dropped
- lists split at item boundaries when oversized
- equations kept whole
- figures/captions merge into adjacent text
- Documents without sections (or all-filtered sections) fall back to the plain path
- Document.metadata["sections"] is not leaked into chunk metadata

The plain-path regression is covered by test_document_chunker.py (unchanged).
"""

import pytest
from unittest.mock import Mock

from src.core.types import Document, Section
from src.core.settings import Settings
from src.ingestion.chunking import DocumentChunker
from src.libs.splitter.base_splitter import BaseSplitter


class FakeSplitter(BaseSplitter):
    """Splits on double newlines (paragraph boundaries) — like the real test file."""

    def __init__(self, **kwargs):
        pass

    def split_text(self, text: str) -> list:
        return [p.strip() for p in text.split("\n\n") if p.strip()]


@pytest.fixture
def chunker(monkeypatch):
    settings = Mock(spec=Settings)
    settings.splitter = Mock(provider="fake", chunk_size=100, overlap=0)
    settings.ingestion = Mock(chunk_size=100)
    from src.libs.splitter import splitter_factory
    monkeypatch.setattr(splitter_factory.SplitterFactory, "create", lambda s: FakeSplitter())
    return DocumentChunker(settings)


def _doc(sections, text="placeholder", doc_id="doc_test"):
    """Build a Document whose metadata carries typed sections."""
    return Document(
        id=doc_id,
        text=text,
        metadata={
            "source_path": "test.pdf",
            "sections": [s.to_dict() if isinstance(s, Section) else s for s in sections],
        },
    )


# =============================================================================
# Routing: plain fallback vs section path
# =============================================================================

class TestSectionRouting:
    def test_no_sections_uses_plain_path(self, chunker):
        """Document without sections → plain path, no section_type set."""
        doc = Document(id="d1", text="Para one.\n\nPara two.", metadata={"source_path": "t.pdf"})
        chunks = chunker.split_document(doc)
        assert len(chunks) == 2
        for c in chunks:
            assert "section_type" not in c.metadata

    def test_all_filtered_sections_fall_back_to_plain(self, chunker):
        """Sections all header/footer → fall back to plain path using document.text."""
        doc = _doc([Section(type="header", text="page header")], text="Real content here.")
        chunks = chunker.split_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Real content here."
        assert "section_type" not in chunks[0].metadata


# =============================================================================
# Title merges into following text (§15.2)
# =============================================================================

class TestTitleMerge:
    def test_title_merges_into_following_text(self, chunker):
        doc = _doc([
            Section(type="title", text="My Title"),
            Section(type="text", text="Body paragraph one.\n\nBody paragraph two."),
        ])
        chunks = chunker.split_document(doc)
        # merged text = "My Title\n\nBody paragraph one.\n\nBody paragraph two."
        # FakeSplitter → 3 fragments
        assert len(chunks) == 3
        assert chunks[0].text == "My Title"
        assert chunks[1].text == "Body paragraph one."
        assert chunks[2].text == "Body paragraph two."
        for c in chunks:
            assert c.metadata["section_type"] == "text"


# =============================================================================
# Tables (§15.1 embed/display split, §17.2 oversized)
# =============================================================================

class TestTableHandling:
    def test_table_carries_parser_plain_text(self, chunker):
        """K2b: Chunker carries the Parser's cleaned plain text verbatim (no re-cleaning)."""
        plain = "A: 1 | B: 2\nA: 3 | B: 4"  # Parser-cleaned "列名: 值" format
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
        doc = _doc([Section(type="table", text=plain, html=html)])
        chunks = chunker.split_document(doc)

        assert len(chunks) == 1
        c = chunks[0]
        assert c.metadata["section_type"] == "table"
        # chunk.text is the Parser's plain text, carried verbatim (NOT re-cleaned)
        assert c.text == plain
        # original HTML preserved for display
        assert c.metadata["table_html"] == html

    def test_table_chunker_does_not_reclean_html_tags(self, chunker):
        """K2b: even if section.text contains stray HTML tags, Chunker carries it
        as-is — proves Chunker is not cleaning (that's the Parser's job)."""
        raw = "<p>not cleaned by parser</p>"
        doc = _doc([Section(type="table", text=raw, html="<table></table>")])
        chunks = chunker.split_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == raw  # tags preserved → Chunker did not clean

    def test_oversized_table_splits_at_rows_html_on_first_only(self, chunker):
        """K2b: oversized table (plain text > 2×chunk_size) splits at \\n rows;
        table_html only on first chunk; row texts reassemble to original."""
        # chunk_size=100 → 2× = 200. Build plain text > 200 chars.
        plain = "\n".join(f"col1: value {i} A | col2: value {i} B" for i in range(30))
        html = "<table>(omitted)</table>"
        doc = _doc([Section(type="table", text=plain, html=html)])
        chunks = chunker.split_document(doc)

        assert len(chunks) > 1
        # table_html only on the first chunk
        assert chunks[0].metadata["table_html"] == html
        for c in chunks[1:]:
            assert c.metadata.get("table_html") is None
        # all chunks typed as table
        for c in chunks:
            assert c.metadata["section_type"] == "table"
        # reassembling row texts reconstructs the original plain text
        assert "\n".join(c.text for c in chunks) == plain


# =============================================================================
# Lists (§15.2 item-boundary split)
# =============================================================================

class TestListSplit:
    def test_list_kept_whole_when_small(self, chunker):
        text = "- item one\n- item two\n- item three"
        doc = _doc([Section(type="list", text=text)])
        chunks = chunker.split_document(doc)
        assert len(chunks) == 1
        assert chunks[0].metadata["section_type"] == "list"
        assert chunks[0].text == text

    def test_list_splits_at_items_when_oversized(self, chunker):
        items = "\n".join(f"- item number {i} with some text here" for i in range(20))
        doc = _doc([Section(type="list", text=items)])
        chunks = chunker.split_document(doc)
        assert len(chunks) > 1
        for c in chunks:
            assert c.metadata["section_type"] == "list"


# =============================================================================
# Headers / footers dropped (§15.2)
# =============================================================================

class TestHeaderFooterDrop:
    def test_headers_and_footers_dropped(self, chunker):
        doc = _doc([
            Section(type="header", text="PAGE HEADER"),
            Section(type="text", text="Real body content."),
            Section(type="footer", text="page 1"),
        ])
        chunks = chunker.split_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Real body content."
        assert "PAGE HEADER" not in chunks[0].text
        assert "page 1" not in chunks[0].text


# =============================================================================
# Equation kept whole (§15.2)
# =============================================================================

class TestEquation:
    def test_equation_kept_whole(self, chunker):
        doc = _doc([
            Section(type="equation", text="E = mc^2"),
            Section(type="text", text="Some explanation."),
        ])
        chunks = chunker.split_document(doc)
        types = [c.metadata["section_type"] for c in chunks]
        assert "equation" in types
        eq = [c for c in chunks if c.metadata["section_type"] == "equation"][0]
        assert eq.text == "E = mc^2"


# =============================================================================
# Figure + caption merge into text (§15.2)
# =============================================================================

class TestFigureCaption:
    def test_figure_and_caption_merge_into_text(self, chunker):
        doc = _doc([
            Section(type="text", text="Intro paragraph."),
            Section(type="figure", text="[IMAGE: img_001]"),
            Section(type="figure_caption", text="Figure 1: a diagram."),
        ])
        chunks = chunker.split_document(doc)
        # all merged into one text buffer → 3 fragments (split on \n\n)
        assert len(chunks) == 3
        fig_chunks = [c for c in chunks if "[IMAGE:" in c.text]
        assert len(fig_chunks) == 1
        assert "img_001" in fig_chunks[0].metadata["image_refs"]


# =============================================================================
# Chunk metadata hygiene
# =============================================================================

class TestChunkMetadata:
    def test_section_chunks_have_index_source_ref_and_type(self, chunker):
        html = "<table><tr><td>x</td></tr></table>"
        doc = _doc([
            Section(type="text", text="Para one.\n\nPara two."),
            Section(type="table", text=html, html=html),
        ], doc_id="doc_x")
        chunks = chunker.split_document(doc)
        assert len(chunks) >= 3
        for i, c in enumerate(chunks):
            assert c.metadata["chunk_index"] == i
            assert c.metadata["source_ref"] == "doc_x"
            assert c.metadata["source_path"] == "test.pdf"
            assert "section_type" in c.metadata

    def test_sections_not_leaked_into_chunk_metadata(self, chunker):
        doc = _doc([Section(type="text", text="body content")])
        chunks = chunker.split_document(doc)
        assert len(chunks) == 1
        assert "sections" not in chunks[0].metadata
