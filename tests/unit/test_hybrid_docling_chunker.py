"""Unit tests for the HybridChunker adapter (ticket 04 / D-037).

Level A runs the real docling HybridChunker over synthetic DoclingDocuments
(needs the local nomic tokenizer dir, matching the shipped config); Level B
tests the pure helpers without running any chunker. The adapter seam is
``split_document`` — everything asserted here is external behaviour: chunk
text shape, metadata fields, ids, and the fallback to the original paths.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core.settings import ChunkerSettings, Settings
from src.core.types import Document
from src.ingestion.chunking.hybrid_docling_chunker import HybridDoclingChunker
from src.libs.splitter.base_splitter import BaseSplitter

# The whole module builds DoclingDocuments — skip cleanly (not error) when
# docling-core is absent.
try:
    from docling_core.types.doc import (
        BoundingBox,
        DocItemLabel,
        DoclingDocument,
        PageItem,
        ProvenanceItem,
        Size,
        TableCell,
        TableData,
    )

    DOCLING_CORE_AVAILABLE = True
except ImportError:
    DOCLING_CORE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DOCLING_CORE_AVAILABLE, reason="docling-core not installed"
)

TOKENIZER_DIR = "D:/models/nomic-embed-text"
needs_tokenizer = pytest.mark.skipif(
    not Path(TOKENIZER_DIR, "tokenizer.json").exists(),
    reason="local nomic tokenizer dir not present (config: tokenizer_path)",
)


class FakeSplitter(BaseSplitter):
    """Same fake as test_document_chunker — isolates the inherited paths."""

    def __init__(self, chunk_size: int = 100, overlap: int = 0, **kwargs):
        pass

    def split_text(self, text: str) -> list[str]:
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]


@pytest.fixture
def hybrid_settings(monkeypatch):
    """Mock settings wired for the hybrid path (max_tokens high → few merges)."""
    from src.libs.splitter import splitter_factory

    monkeypatch.setattr(
        splitter_factory.SplitterFactory,
        "create",
        staticmethod(lambda settings: FakeSplitter()),
    )
    settings = Mock(spec=Settings)
    settings.splitter = Mock(provider="fake", chunk_size=100, overlap=0)
    settings.ingestion = Mock()
    settings.ingestion.chunk_size = 1000
    settings.ingestion.chunker = ChunkerSettings(
        provider="hybrid_docling",
        max_tokens=2000,  # generous: keeps the synthetic doc in one chunk
        merge_peers=True,
        tokenizer_path=TOKENIZER_DIR,
    )
    return settings


def _prov(page, l, t, r, b):
    return ProvenanceItem(
        page_no=page, bbox=BoundingBox(l=l, t=t, r=r, b=b), charspan=(0, 0)
    )


def _cell(row, col, text, header=False):
    return TableCell(
        row_id=row,
        col_id=col,
        start_row_offset_idx=row,
        end_row_offset_idx=row + 1,
        start_col_offset_idx=col,
        end_col_offset_idx=col + 1,
        text=text,
        col_header=header,
    )


def make_ddoc(with_table=False, with_picture=False):
    """Synthetic DoclingDocument: header + prose (+ table) (+ picture)."""
    pages = {n: PageItem(page_no=n, size=Size(width=595, height=842)) for n in (1, 2, 3)}
    doc = DoclingDocument(name="synthetic", pages=pages)
    doc.add_text(
        label=DocItemLabel.SECTION_HEADER, text="Chapter One",
        prov=_prov(1, 0, 100, 500, 120),
    )
    doc.add_text(
        label=DocItemLabel.PARAGRAPH, text="Body text about the method.",
        prov=_prov(1, 0, 130, 500, 200),
    )
    if with_table:
        doc.add_table(
            data=TableData(
                num_rows=2,
                num_cols=2,
                table_cells=[
                    _cell(0, 0, "h1", header=True), _cell(0, 1, "h2", header=True),
                    _cell(1, 0, "a"), _cell(1, 1, "b"),
                ],
            ),
            prov=_prov(2, 0, 50, 400, 250),
        )
    if with_picture:
        doc.add_picture(prov=_prov(3, 50, 400, 550, 50))
    return doc


def make_document(ddocs, images=None, doc_id="doc_abc123"):
    """Document shaped exactly like DoclingParser's output for the chunker."""
    return Document(
        id=doc_id,
        text="flat fallback text",
        metadata={
            "source_path": "docs/synthetic.pdf",
            "doc_hash": "abc123",
            "sections": [{"type": "text", "text": "flat"}],
            "docling_documents": ddocs,
            "images": images or [],
        },
    )


# =====================================================================
# Level A — real HybridChunker over synthetic documents
# =====================================================================

@needs_tokenizer
class TestHybridSplitExternalBehaviour:
    """External behaviour through the split_document seam."""

    @pytest.fixture(autouse=True)
    def _chunker(self, hybrid_settings):
        self.chunker = HybridDoclingChunker(hybrid_settings)

    def test_chunk_text_carries_heading_path_prefix(self):
        """contextualize() 生效: body chunks start with their heading path."""
        doc = make_document([make_ddoc()])
        chunks = self.chunker.split_document(doc)

        assert chunks, "expected at least one chunk"
        assert self.chunker.last_method == "hybrid_docling"
        body = next(c for c in chunks if "Body text" in c.text)
        # Heading prefix precedes the body text (contextualize, spike 01).
        assert body.text.index("Chapter One") < body.text.index("Body text")
        # Structured headings metadata (for section filtering).
        assert body.metadata["headings"] == ["Chapter One"]

    def test_chunk_ids_follow_existing_scheme_and_are_stable(self):
        """ids keep {doc_id}_{index:04d}_{hash8}; re-splitting is identical."""
        doc = make_document([make_ddoc(with_table=True)])
        first = self.chunker.split_document(doc)
        second = self.chunker.split_document(doc)

        for i, c in enumerate(first):
            prefix, idx, digest = c.id.rsplit("_", 2)
            assert prefix == doc.id
            assert idx == f"{i:04d}"
            assert len(digest) == 8
        assert [c.id for c in first] == [c.id for c in second]
        assert [c.text for c in first] == [c.text for c in second]

    def test_mixed_chunk_flags_coexist(self):
        """TEXT+TABLE in one chunk → contains_text AND contains_table both set."""
        doc = make_document([make_ddoc(with_table=True)])
        chunks = self.chunker.split_document(doc)

        mixed = [c for c in chunks if c.metadata["contains_table"]]
        assert mixed, "table content expected somewhere"
        # With max_tokens=2000 the prose and table merge into one chunk —
        # the mixed-flag scenario the ticket calls out.
        m = mixed[0]
        assert m.metadata["contains_text"] is True
        assert m.metadata["contains_table"] is True
        assert m.metadata["section_type"] == "table"  # compat single value
        assert m.metadata["contains_figure"] is False
        assert m.metadata["contains_formula"] is False
        assert m.metadata["table_html"], "table GFM must ride in metadata"

    def test_formula_section_type_matches_legacy_vocabulary(self):
        """Compat single value keeps the legacy path's word: 'equation'."""
        pages = {1: PageItem(page_no=1, size=Size(width=595, height=842))}
        ddoc = DoclingDocument(name="formula", pages=pages)
        ddoc.add_text(
            label=DocItemLabel.FORMULA,
            text="E = mc^2",
            prov=_prov(1, 0, 100, 500, 200),
        )
        doc = make_document([ddoc])
        chunks = self.chunker.split_document(doc)

        f = next(c for c in chunks if c.metadata["contains_formula"])
        assert f.metadata["section_type"] == "equation"  # legacy vocabulary
        assert f.metadata["contains_formula"] is True
        assert f.metadata["contains_text"] is False

    def test_image_placeholder_injected_by_prov(self):
        """Rendered figures inject [IMAGE: id] into the nearest chunk.

        docling's chunker drops picture items (no image ref), so injection
        lands on the spatially nearest chunk — same page wins.
        """
        ddoc = make_ddoc(with_picture=True)
        img = {
            "id": "abc123_3_1",
            "path": "data/images/abc123/abc123_3_1.png",
            "page": 3,
            "bbox": {"x0": 50.0, "top": 400.0, "x1": 550.0, "bottom": 50.0},
        }
        doc = make_document([ddoc], images=[img])
        chunks = self.chunker.split_document(doc)

        hosting = [c for c in chunks if "[IMAGE: abc123_3_1]" in c.text]
        assert len(hosting) == 1, "placeholder must appear exactly once"
        assert hosting[0].metadata["contains_figure"] is True
        # Injection happened before id generation (id hashes final text).
        import hashlib

        expected_hash = hashlib.sha256(
            hosting[0].text.encode("utf-8")
        ).hexdigest()[:8]
        assert hosting[0].id.endswith(expected_hash)

    def test_page_and_bbox_metadata(self):
        # Table-only document: the chunk's provenance comes from the TABLE
        # item itself (page 2), not mixed with a page-1 paragraph.
        pages = {2: PageItem(page_no=2, size=Size(width=595, height=842))}
        ddoc = DoclingDocument(name="table-only", pages=pages)
        ddoc.add_table(
            data=TableData(
                num_rows=2,
                num_cols=2,
                table_cells=[
                    _cell(0, 0, "h1", header=True), _cell(0, 1, "h2", header=True),
                    _cell(1, 0, "a"), _cell(1, 1, "b"),
                ],
            ),
            prov=_prov(2, 0, 50, 400, 250),
        )
        doc = make_document([ddoc])
        chunks = self.chunker.split_document(doc)

        table_chunk = next(c for c in chunks if c.metadata["contains_table"])
        assert table_chunk.metadata["page_num"] == 2
        assert isinstance(table_chunk.metadata["bbox"], dict)

    def test_fallback_without_docling_documents(self):
        """No docling_documents (non-docling / degraded) → original paths."""
        doc = Document(
            id="doc_plain",
            text="First paragraph.\n\nSecond paragraph.",
            metadata={"source_path": "docs/plain.pdf"},
        )
        chunks = self.chunker.split_document(doc)

        assert [c.text for c in chunks] == ["First paragraph.", "Second paragraph."]
        assert self.chunker.last_method == "recursive"
        assert all("contains_table" not in c.metadata for c in chunks)

    def test_degrades_when_chunker_build_fails(self, hybrid_settings):
        """Broken tokenizer config degrades to the recursive path, not a crash."""
        hybrid_settings.ingestion.chunker = ChunkerSettings(
            provider="hybrid_docling",
            tokenizer_path="D:/models/does-not-exist",
        )
        chunker = HybridDoclingChunker(hybrid_settings)
        doc = make_document([make_ddoc()])

        chunks = chunker.split_document(doc)

        # Degraded: falls back to the inherited plain path (flat text).
        assert chunks and chunker.last_method == "recursive"
        assert "docling_documents" not in chunks[0].metadata


# =====================================================================
# Level B — pure helper logic (no chunker run)
# =====================================================================

class TestNearestChunkSelection:
    """_nearest_chunk_index: same page wins, then page distance, then space."""

    def _raw(self, page, bbox):
        return {"page": page, "bbox": bbox}

    def test_same_page_beats_other_page(self):
        raw = [
            self._raw(1, {"x0": 0, "top": 0, "x1": 10, "bottom": 10}),
            self._raw(3, {"x0": 40, "top": 380, "x1": 460, "bottom": 380}),
        ]
        img = {"page": 3, "bbox": {"x0": 50, "top": 400, "x1": 550, "bottom": 50}}
        idx = HybridDoclingChunker._nearest_chunk_index(raw, img)
        assert idx == 1  # page 3 chunk wins despite page-1 chunk being spatially nearer

    def test_tie_keeps_document_order(self):
        raw = [
            self._raw(1, {"x0": 0, "top": 0, "x1": 10, "bottom": 10}),
            self._raw(1, {"x0": 0, "top": 0, "x1": 10, "bottom": 10}),
        ]
        img = {"page": 1, "bbox": {"x0": 0, "top": 0, "x1": 5, "bottom": 5}}
        assert HybridDoclingChunker._nearest_chunk_index(raw, img) == 0

    def test_exact_prov_key_match_wins(self):
        """Same (page, bbox) as a rendered figure → that chunk hosts it."""
        chunker = HybridDoclingChunker.__new__(HybridDoclingChunker)
        bbox = {"x0": 50.0, "top": 400.0, "x1": 550.0, "bottom": 50.0}
        raw = [
            {"page": 1, "bbox": None, "text": "prose"},
            {"page": 3, "bbox": bbox, "text": "caption"},
        ]
        img = {"id": "fig_1", "page": 3, "bbox": dict(bbox)}
        chunker._inject_image_placeholders(raw, [img])

        assert "[IMAGE: fig_1]" in raw[1]["text"]
        assert "[IMAGE: fig_1]" not in raw[0]["text"]
        assert raw[1]["contains_figure"] is True

    def test_no_chunks_no_crash(self):
        chunker = HybridDoclingChunker.__new__(HybridDoclingChunker)
        chunker._inject_image_placeholders(
            [], [{"id": "x", "page": 1, "bbox": {"x0": 0, "top": 0, "x1": 1, "bottom": 1}}]
        )
