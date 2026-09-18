"""T16 – BM25 re-ingest idempotency at the pipeline wiring seam.

Regression test: pipeline stage 6b used to pass ``document.id`` (a
*content* hash, ``doc_...``) as the BM25 ``doc_id``, while chunk_ids are
prefixed by ``sha256(source_path)[:8]``.  ``remove_document``'s
``startswith`` never matched, so re-ingesting a file never deleted its
old BM25 postings — num_docs accumulated phantoms and unchanged chunks
got double-counted postings across re-ingests (see T13 / T16).

Seam under test: ``IngestionPipeline.run`` — the real VectorUpserter
(chunk-ID scheme) and real BM25Indexer (index artifact on disk); parser /
chunker / transforms / encoders are faked.  Expected chunk ids come from
``result.vector_ids`` (public interface output), never recomputed here.
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

from src.core.types import Chunk, Document
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

SOURCE_PATH = "docs/quantum_paper.pdf"
# The parser receives str(Path(file_path)); chunk metadata mirrors that.
CHUNK_SOURCE_PATH = str(Path(SOURCE_PATH))
COLLECTION = "t16_reingest"

# Two versions of the same document (same source path).  v2 drifts:
# chunk 0 is rewritten, chunk 2 is dropped, chunk 3 is reworded and
# re-indexed at position 2.  Only chunk 1 keeps text *and* index, so
# exactly one chunk_id carries over.
V1_TEXTS = [
    "Quantum key distribution relies on entanglement and no-cloning.",
    "BB84 protocol uses photon polarization states to detect eavesdropping.",
    "Device independent security builds on Bell inequality violations.",
    "Decoherence limits the qubit lifetime in noisy channels.",
]
V2_TEXTS = [
    "Quantum key distribution secures communication via fundamental physics.",
    "BB84 protocol uses photon polarization states to detect eavesdropping.",
    "Decoherence limits the qubit lifetime in noisy quantum channels.",
]


class _MemoryStore:
    """Stand-in vector store: accept upserts, persist nothing."""

    def upsert(self, records, trace=None):
        return None


def _make_fake_pipeline(bm25_dir: Path, chunk_texts):
    """Build a fake IngestionPipeline (test_pipeline_progress pattern).

    Real components: VectorUpserter (ID scheme) + BM25Indexer (artifact).
    Everything else is a MagicMock.  Returns (fp, fake_parser_doc_id).
    """
    fp = type("FP", (), {"collection": COLLECTION, "force": True})()

    fp.integrity_checker = MagicMock()
    fp.integrity_checker.compute_sha256.return_value = "hash123"
    fp.integrity_checker.should_skip.return_value = False

    # docling-style document id: content hash — the value the bug passed
    # as BM25 doc_id, which can never prefix-match a path-hash chunk_id.
    doc_text = "\n\n".join(chunk_texts)
    fp.parser = MagicMock()
    fp.parser.parse.return_value = Document(
        id="doc_" + hashlib.sha256(doc_text.encode("utf-8")).hexdigest()[:16],
        text=doc_text,
        metadata={"source_path": SOURCE_PATH, "images": []},
    )

    chunks = [
        Chunk(
            id=f"c{i}",
            text=text,
            metadata={"source_path": CHUNK_SOURCE_PATH, "chunk_index": i},
        )
        for i, text in enumerate(chunk_texts)
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
        tokens = [t.strip(".,").lower() for t in text.split()]
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        return {"term_frequencies": tf, "doc_length": len(tokens)}

    batch_result = MagicMock()
    batch_result.dense_vectors = [[0.1, 0.2]] * len(chunks)
    batch_result.sparse_stats = [_stat(t) for t in chunk_texts]
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
    return fp


def _load_index(bm25_dir: Path):
    with open(bm25_dir / f"{COLLECTION}_bm25.json", encoding="utf-8") as f:
        return json.load(f)


def _unique_chunk_ids(index_data):
    return {
        p["chunk_id"]
        for term_data in index_data["index"].values()
        for p in term_data["postings"]
    }


def _assert_no_duplicate_postings(index_data):
    for term, term_data in index_data["index"].items():
        ids = [p["chunk_id"] for p in term_data["postings"]]
        assert len(ids) == len(set(ids)), f"duplicate posting for term {term!r}"


def _assert_index_clean(index_data, expected_ids, stage):
    """Index holds exactly *expected_ids*, once each, num_docs aligned."""
    unique = _unique_chunk_ids(index_data)
    assert unique == set(expected_ids), f"{stage}: BM25 chunk set mismatch"
    assert index_data["metadata"]["num_docs"] == len(unique), (
        f"{stage}: num_docs {index_data['metadata']['num_docs']} "
        f"!= unique {len(unique)}"
    )
    _assert_no_duplicate_postings(index_data)


class TestPipelineBM25Reingest:
    """Re-ingesting the same path must replace, not accumulate, BM25 rows."""

    def test_reingest_same_path_replaces_bm25_postings(self, tmp_path):
        bm25_dir = tmp_path / "bm25"

        # ── First ingest (document v1) ────────────────────────────────
        fp1 = _make_fake_pipeline(bm25_dir, V1_TEXTS)
        result1 = IngestionPipeline.run(fp1, SOURCE_PATH)
        assert result1.success, result1.error
        assert len(result1.vector_ids) == len(V1_TEXTS)
        _assert_index_clean(_load_index(bm25_dir), result1.vector_ids, "v1")

        # ── Re-ingest same path, drifted content (document v2) ────────
        # Fresh pipeline + indexer: a re-ingest is a new process.
        fp2 = _make_fake_pipeline(bm25_dir, V2_TEXTS)
        result2 = IngestionPipeline.run(fp2, SOURCE_PATH)
        assert result2.success, result2.error

        data2 = _load_index(bm25_dir)
        _assert_index_clean(data2, result2.vector_ids, "v2")

        # Old-version chunks (drifted or dropped in v2) must be gone.
        retired = set(result1.vector_ids) - set(result2.vector_ids)
        assert retired, "scenario sanity: drift must retire some chunk ids"
        assert not retired & _unique_chunk_ids(data2), (
            "stale v1 postings survived re-ingest"
        )

        # The unchanged chunk (same text, same index) keeps its id —
        # content-hash stability is the one carry-over we expect.
        kept = set(result1.vector_ids) & set(result2.vector_ids)
        assert len(kept) == 1, f"expected exactly one carried-over id, got {kept}"
