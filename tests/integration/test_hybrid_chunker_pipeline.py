"""Main-seam integration tests for the hybrid chunker (ticket 04 / D-037).

``IngestionPipeline.run(file) → stored results``, read back from where they
actually land. The parser is faked into the *replay* shape (a Document
carrying ``docling_documents`` — exactly what DoclingParser produces for
fresh parses and cache replays alike); the HybridChunker, Chroma, BM25 and
trace are real. Embedding is a deterministic fake so nothing touches a
model server; vision is disabled. What this proves end-to-end:

- chunks stored in Chroma carry the contextualize() heading prefix;
- table chunks carry metadata.table_html (display criterion) and the
  contains_table flag (filter criterion) — and the flag is queryable via
  a Chroma where clause;
- [IMAGE: id] placeholders survive into stored chunk text;
- the trace split stage records method=hybrid_docling;
- a re-run produces identical chunk ids (idempotent re-ingest, D-031).

Needs the local nomic tokenizer dir (same as the shipped config).
"""

import base64
import dataclasses
import shutil
import uuid
from pathlib import Path
from typing import List

import pytest

from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Document
from src.ingestion.pipeline import IngestionPipeline
from src.libs.embedding.base_embedding import BaseEmbedding

# The fake parser builds real DoclingDocuments — skip cleanly when
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

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DOCLING_CORE_AVAILABLE, reason="docling-core not installed"),
]

# 1x1 transparent PNG — a real file ImageStorage can register.
_PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

COLLECTION = "test_hybrid04"


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


class FakeDoclingParser:
    """Parser emitting the DoclingParser output shape (docling_documents).

    All hashes/ids are FIXED: the idempotency test re-runs the pipeline and
    must get identical chunk ids (ids hash the final chunk text).
    """

    IMAGE_ID = "t04fixed_img_0001"
    DOC_HASH = "t04fixed_doc_hash"

    def __init__(self, image_path: Path):
        self.image_path = image_path

    def parse(self, file_path) -> Document:
        pages = {
            n: PageItem(page_no=n, size=Size(width=595, height=842))
            for n in (1, 2, 3)
        }
        ddoc = DoclingDocument(name="fake", pages=pages)
        ddoc.add_text(
            label=DocItemLabel.SECTION_HEADER, text="Chapter One",
            prov=_prov(1, 0, 100, 500, 120),
        )
        ddoc.add_text(
            label=DocItemLabel.PARAGRAPH,
            text="The quantum circuit implementation uses layered gates.",
            prov=_prov(1, 0, 130, 500, 200),
        )
        ddoc.add_table(
            data=TableData(
                num_rows=2,
                num_cols=2,
                table_cells=[
                    _cell(0, 0, "Scheme", header=True),
                    _cell(0, 1, "Qubits", header=True),
                    _cell(1, 0, "AES-128"),
                    _cell(1, 1, "184"),
                ],
            ),
            prov=_prov(2, 0, 50, 400, 250),
        )
        ddoc.add_picture(prov=_prov(3, 50, 400, 550, 50))

        self.image_path.write_bytes(_PNG_1PX)
        images = [{
            "id": self.IMAGE_ID,
            "path": str(self.image_path),
            "page": 3,
            "bbox": {"x0": 50.0, "top": 400.0, "x1": 550.0, "bottom": 50.0},
            "position": {"width": 1, "height": 1, "page": 3, "index": 0},
        }]
        return Document(
            id=f"doc_{self.DOC_HASH[:16]}",
            text="Chapter One\n\nflat fallback",
            metadata={
                "source_path": file_path if isinstance(file_path, str) else str(file_path),
                "doc_type": "pdf",
                "doc_hash": self.DOC_HASH,
                "sections": [{"type": "text", "text": "flat"}],
                "docling_documents": [ddoc],
                "images": images,
            },
        )


class FakeEmbedding(BaseEmbedding):
    """Deterministic offline embedding (same idea as test_dense_encoder)."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def embed(self, texts: List[str], trace=None, **kwargs) -> List[List[float]]:
        self.validate_texts(texts)
        return [
            [len(t) / 1000.0 + i * 0.001 for i in range(self.dimension)]
            for t in texts
        ]

    def get_dimension(self) -> int:
        return self.dimension


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Real settings, redirected: chroma → tmp, vision off, parsers/embeddings faked."""
    from src.libs.parser.parser_factory import ParserFactory
    from src.libs.embedding.embedding_factory import EmbeddingFactory

    s = load_settings()
    s = dataclasses.replace(
        s,
        vector_store=dataclasses.replace(
            s.vector_store,
            persist_directory=str(tmp_path / "chroma"),
            collection_name=COLLECTION,
        ),
        vision_llm=(
            dataclasses.replace(s.vision_llm, enabled=False)
            if s.vision_llm else None
        ),
    )

    fake_parser = FakeDoclingParser(tmp_path / "fig_t04.png")
    monkeypatch.setattr(ParserFactory, "create", classmethod(lambda cls, *a, **k: fake_parser))
    monkeypatch.setattr(
        EmbeddingFactory, "create", classmethod(lambda cls, *a, **k: FakeEmbedding())
    )
    return s


@pytest.fixture(autouse=True)
def _cleanup_bm25():
    """The pipeline hard-codes the repo-relative BM25 dir per collection."""
    bm25_dir = Path(__file__).resolve().parents[2] / "data" / "db" / "bm25" / COLLECTION
    yield
    shutil.rmtree(bm25_dir, ignore_errors=True)


@pytest.fixture
def pdf_path(tmp_path) -> str:
    p = tmp_path / "fake_doc.pdf"
    p.write_bytes(b"%PDF-1.4 fake hybrid chunker pipeline test")
    return str(p)


class TestHybridChunkerPipeline:
    def test_run_stores_hybrid_chunks(self, settings, pdf_path):
        """Full pipeline run: split method, stored shape, trace stage."""
        pipeline = IngestionPipeline(settings, collection=COLLECTION, force=True)
        trace = TraceContext(trace_type="ingestion")
        try:
            result = pipeline.run(pdf_path, trace=trace)
        finally:
            pipeline.close()

        assert result.success, f"pipeline failed: {result.error}"
        assert result.chunk_count > 0

        # Trace: split stage reports the hybrid method (dashboard zero-change).
        split = next(s for s in trace.stages if s["stage"] == "split")
        assert split["data"]["method"] == "hybrid_docling"
        assert split["data"]["chunk_count"] == result.chunk_count

        # Read back from Chroma through the store seam — filtering by the
        # contains_table flag (the filtering surface the ticket mandates).
        store = pipeline.vector_upserter.vector_store
        table_hits = store.query(
            vector=[0.5] * 768, top_k=50, filters={"contains_table": True}
        )
        assert table_hits, "flag-filtered table chunks must be retrievable"
        table_meta = [h["metadata"] for h in table_hits]
        assert any(m.get("table_html") for m in table_meta)
        assert "Scheme" in table_meta[0]["table_html"]

        all_hits = store.query(vector=[0.5] * 768, top_k=50)
        assert all_hits, "stored chunks must be queryable from Chroma"
        by_id = {h["id"]: h["metadata"] for h in all_hits}

        # Heading prefix landed in stored chunk text (contextualize).
        stored_texts = [h["metadata"].get("text", "") for h in all_hits]
        assert any("Chapter One" in t and "quantum circuit" in t for t in stored_texts)

        # Parser intermediates never leak into stored metadata.
        for m in by_id.values():
            assert "docling_documents" not in m
            assert "sections" not in m

    def test_image_placeholder_lands_in_stored_text(self, settings, pdf_path):
        pipeline = IngestionPipeline(settings, collection=COLLECTION, force=True)
        try:
            result = pipeline.run(pdf_path)
        finally:
            pipeline.close()

        assert result.success
        store = pipeline.vector_upserter.vector_store
        hits = store.query(vector=[0.5] * 768, top_k=50)
        assert any(
            "[IMAGE:" in h["metadata"].get("text", "") for h in hits
        ), "[IMAGE: id] placeholder must survive into stored chunk text"

    def test_rerun_yields_identical_chunk_ids(self, settings, pdf_path):
        """Idempotent re-ingest (D-031 shape): same content → same ids."""
        first = IngestionPipeline(settings, collection=COLLECTION, force=True)
        try:
            r1 = first.run(pdf_path)
        finally:
            first.close()
        second = IngestionPipeline(settings, collection=COLLECTION, force=True)
        try:
            r2 = second.run(pdf_path)
        finally:
            second.close()

        assert r1.success and r2.success
        assert sorted(r1.vector_ids) == sorted(r2.vector_ids)
