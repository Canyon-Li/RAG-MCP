"""T17 – DocumentManager.delete_document must purge BM25 postings.

Regression test: ``delete_document`` used to pass *source_hash* (a file
*content* hash) to ``BM25Indexer.remove_document``, whose match rule is
``chunk_id.startswith(doc_id)`` with chunk ids prefixed by
``sha256(source_path)[:8]``.  A content hash never prefix-matches a
path hash, so explicit deletion never removed BM25 postings — the
dashboard's "delete document" left dead chunks retrievable on the
sparse path (same family as the T16 re-ingest bug, delete path).

Seam under test: ``DocumentManager.delete_document`` — a real
BM25Indexer (index artifact on disk, fresh instance to mimic a new
process) against the artifact a fake pipeline produced; ChromaDB /
ImageStorage / FileIntegrity are fakes (their keying is *content*-hash
based and unchanged by this fix).  Expected chunk ids come from
``PipelineResult.vector_ids`` (public interface output), never
recomputed here.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

from src.core.types import Chunk, Document
from src.ingestion.document_manager import DocumentManager
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

COLLECTION = "t17_delete"
DOC_A = "docs/quantum_paper.pdf"
DOC_B = "docs/nn_survey.pdf"

DOC_A_TEXTS = [
    "Quantum key distribution relies on entanglement and no-cloning.",
    "BB84 protocol uses photon polarization states to detect eavesdropping.",
]
DOC_B_TEXTS = [
    "Gradient descent optimizes neural network weights iteratively.",
    "Backpropagation chains derivatives through network layers.",
]


class _MemoryStore:
    """Stand-in vector store: accept upserts, persist nothing."""

    def upsert(self, records, trace=None):
        return None


def _ingest(bm25_dir: Path, source_path: str, texts) -> list:
    """Run one fake pipeline ingest (T16 pattern); return vector ids."""
    fp = type("FP", (), {"collection": COLLECTION, "force": True})()

    fp.integrity_checker = MagicMock()
    fp.integrity_checker.compute_sha256.return_value = "hash_" + source_path
    fp.integrity_checker.should_skip.return_value = False

    doc_text = "\n\n".join(texts)
    fp.parser = MagicMock()
    fp.parser.parse.return_value = Document(
        id="doc_x", text=doc_text,
        metadata={"source_path": source_path, "images": []},
    )

    chunks = [
        Chunk(
            id=f"c{i}",
            text=text,
            # source_path flows verbatim: parser sets it from str(file_path),
            # _inherit_metadata copies document.metadata unchanged — so the
            # delete-side prefix must be computed from this exact string.
            metadata={"source_path": source_path, "chunk_index": i},
        )
        for i, text in enumerate(texts)
    ]
    fp.chunker = MagicMock()
    fp.chunker.split_document.return_value = chunks
    fp.chunk_refiner = MagicMock()
    fp.chunk_refiner.transform.return_value = chunks
    fp.metadata_enricher = MagicMock()
    fp.metadata_enricher.transform.return_value = chunks
    fp.image_captioner = MagicMock()
    fp.image_captioner.transform.return_value = chunks
    fp.table_summarizer = MagicMock()
    fp.table_summarizer.transform.return_value = chunks

    def _stat(text):
        tf = {}
        tokens = [t.strip(".,").lower() for t in text.split()]
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        return {"term_frequencies": tf, "doc_length": len(tokens)}

    batch_result = MagicMock()
    batch_result.dense_vectors = [[0.1, 0.2]] * len(chunks)
    batch_result.sparse_stats = [_stat(t) for t in texts]
    fp.batch_processor = MagicMock()
    fp.batch_processor.process.return_value = batch_result

    # Real VectorUpserter against an in-memory store, so chunk_ids use
    # the production scheme without touching any real vector DB.
    original_create = VectorStoreFactory.create
    VectorStoreFactory.create = staticmethod(lambda settings, **kw: _MemoryStore())
    try:
        fp.vector_upserter = VectorUpserter(settings=None)
    finally:
        VectorStoreFactory.create = original_create

    fp.bm25_indexer = BM25Indexer(index_dir=str(bm25_dir))
    fp.image_storage = MagicMock()

    result = IngestionPipeline.run(fp, source_path)
    assert result.success, result.error
    return result.vector_ids


def _load_index(bm25_dir: Path):
    with open(bm25_dir / f"{COLLECTION}_bm25.json", encoding="utf-8") as f:
        return json.load(f)


def _unique_chunk_ids(index_data):
    return {
        p["chunk_id"]
        for term_data in index_data["index"].values()
        for p in term_data["postings"]
    }


class TestDocumentManagerDeleteBM25:
    """Deleting one document must purge its BM25 rows and spare others."""

    def test_delete_document_purges_bm25_postings(self, tmp_path):
        bm25_dir = tmp_path / "bm25"

        # ── Ingest two documents into the same collection ───────────
        ids_a = _ingest(bm25_dir, DOC_A, DOC_A_TEXTS)
        ids_b = _ingest(bm25_dir, DOC_B, DOC_B_TEXTS)

        data = _load_index(bm25_dir)
        unique = _unique_chunk_ids(data)
        assert unique == set(ids_a) | set(ids_b), "ingest sanity: both docs in"
        assert data["metadata"]["num_docs"] == len(unique)

        # ── Delete doc A via DocumentManager ────────────────────────
        # content hash of A — what Chroma/ImageStorage/integrity key on
        # and what the bug passed to BM25.  64 hex chars, like the real
        # FileIntegrity.compute_sha256 output.
        content_hash_a = "ab12cd34" * 8

        chroma = MagicMock()
        chroma.delete_by_metadata.return_value = len(DOC_A_TEXTS)
        bm25 = BM25Indexer(index_dir=str(bm25_dir))  # fresh → disk path
        images = MagicMock()
        images.list_images.return_value = []
        integrity = MagicMock()
        integrity.compute_sha256.return_value = content_hash_a
        integrity.remove_record.return_value = True

        manager = DocumentManager(chroma, bm25, images, integrity)
        result = manager.delete_document(DOC_A, collection=COLLECTION)

        # All four stores attempted, no partial-failure errors.
        assert result.success, result.errors
        assert result.errors == []

        # BM25: doc A fully purged (the T17 fix — used to be a no-op) ...
        assert result.bm25_removed is True
        data = _load_index(bm25_dir)
        remaining = _unique_chunk_ids(data)
        assert not (set(ids_a) & remaining), "doc A postings survived delete"
        # ... while doc B is untouched (no over-deletion by prefix).
        assert remaining == set(ids_b), "doc B postings must survive intact"
        assert data["metadata"]["num_docs"] == len(remaining)

        # Regression: the other stores still key on the *content* hash.
        chroma.delete_by_metadata.assert_called_once_with(
            {"doc_hash": content_hash_a}
        )
        images.list_images.assert_called_once_with(doc_hash=content_hash_a)
        integrity.remove_record.assert_called_once_with(content_hash_a)
        assert result.chunks_deleted == len(DOC_A_TEXTS)
        assert result.integrity_removed is True
