"""DocumentChunker: section page 透传 + metadata 瘦身（G3）。"""
import pytest

from src.core.settings import load_settings
from src.core.types import Document
from src.ingestion.chunking.document_chunker import DocumentChunker


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture
def chunker(settings):
    return DocumentChunker(settings)


def _doc_with_sections():
    sections = [
        {"type": "title", "text": "标题", "page": 1, "bbox": None, "images": [], "html": None},
        {"type": "text", "text": "正文片段" * 200, "page": 1, "bbox": None, "images": [], "html": None},
        {"type": "table", "text": "A: 1\nB: 2", "page": 2, "bbox": None, "images": [], "html": "<table></table>"},
        {"type": "list", "text": "- 项一\n- 项二", "page": 3, "bbox": None, "images": [], "html": None},
    ]
    return Document(
        id="doc_test",
        text="标题\n\n" + "正文片段" * 200 + "\n\nA: 1\nB: 2\n\n- 项一\n- 项二",
        metadata={"source_path": "test.pdf", "doc_type": "pdf",
                  "doc_hash": "abc", "sections": sections},
    )


def test_text_chunk_carries_page(chunker):
    chunks = chunker.split_document(_doc_with_sections())
    text_chunks = [c for c in chunks if c.metadata.get("section_type") == "text"]
    assert text_chunks, "应至少有一个 text chunk"
    assert all(c.metadata.get("page_num") == 1 for c in text_chunks)


def test_table_chunk_carries_page(chunker):
    chunks = chunker.split_document(_doc_with_sections())
    table_chunks = [c for c in chunks if c.metadata.get("section_type") == "table"]
    assert table_chunks, "应有一个 table chunk"
    assert table_chunks[0].metadata.get("page_num") == 2


def test_list_chunk_carries_page(chunker):
    chunks = chunker.split_document(_doc_with_sections())
    list_chunks = [c for c in chunks if c.metadata.get("section_type") == "list"]
    assert list_chunks, "应有一个 list chunk"
    assert list_chunks[0].metadata.get("page_num") == 3


def test_chunk_metadata_no_image_fields(chunker):
    chunks = chunker.split_document(_doc_with_sections())
    assert chunks
    for c in chunks:
        assert "images" not in c.metadata
        assert "image_captions" not in c.metadata
        assert "image_refs" not in c.metadata
