"""D-036 / ticket 02 – parse-cache replay at the pipeline wiring seam.

Regression seam (mirrors test_pipeline_bm25_reingest.py): a fake
``IngestionPipeline`` object with REAL DoclingParser (DocumentConverter
probed, DoclingDocument JSON roundtrip through real files), REAL
DocumentChunker, REAL VectorUpserter (chunk-ID scheme) and REAL BM25Indexer.

Acceptance under test: two consecutive ``--force`` ingests of the same file —
the second must make ZERO DocumentConverter calls (replay from cache) while
persisting byte-identical chunk ids/texts (chunk-id scheme + BM25 prefix
contract unchanged), with no duplicate BM25 postings.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import load_settings
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.libs.parser.docling_parser import DoclingParser
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

COLLECTION = "t_d036_replay"

SPECS = [
    {"label": "SECTION_HEADER", "text": "第一章 系统架构", "page": 1,
     "table_md": None, "bbox": None},
    {"label": "TEXT",
     "text": "混合检索由稠密向量与 BM25 稀疏检索两路组成，经 RRF 融合后输出候选。",
     "page": 1, "table_md": None, "bbox": (10, 20, 30, 40)},
    {"label": "TEXT",
     "text": "重排阶段使用交叉编码器对候选重打分，截断后取 top_k。",
     "page": 1, "table_md": None, "bbox": None},
    {"label": "TABLE", "text": None, "page": 2,
     "table_md": "| 组件 | 延迟 |\n|---|---|\n| dense | 12ms |\n| sparse | 3ms |",
     "bbox": None},
]


# ---------------------------------------------------------------------------
# Docling mocks: counting converter + real-file JSON roundtrip
# (same shapes as tests/unit/test_docling_parse_cache.py)
# ---------------------------------------------------------------------------


def _spec_to_item(spec):
    from unittest.mock import MagicMock
    item = MagicMock()
    lbl = MagicMock()
    lbl.name = spec["label"]
    item.label = lbl
    item.text = spec["text"]
    prov = MagicMock()
    prov.page_no = spec["page"]
    bbox = spec["bbox"]
    if bbox:
        prov.bbox = MagicMock(l=bbox[0], t=bbox[1], r=bbox[2], b=bbox[3])
    else:
        prov.bbox = None
    item.prov = [prov]
    if spec["table_md"] is not None:
        item.export_to_markdown = MagicMock(return_value=spec["table_md"])
    return item


def _make_doc(specs):
    ddoc = MagicMock()
    ddoc.iterate_items = MagicMock(
        return_value=[(_spec_to_item(s), 0) for s in specs])

    def _save(filename):
        Path(filename).write_text(json.dumps({"specs": specs}), encoding="utf-8")

    ddoc.save_as_json = MagicMock(side_effect=_save)
    return ddoc


def _load_from_json(filename):
    data = json.loads(Path(filename).read_text(encoding="utf-8"))
    ddoc = MagicMock()
    ddoc.iterate_items = MagicMock(
        return_value=[(_spec_to_item(s), 0) for s in data["specs"]])
    return ddoc


class _ConverterProbe:
    """Counts DocumentConverter constructions and convert() calls."""

    def __init__(self, specs):
        self.specs = specs
        self.constructed = 0
        self.converted = 0

    def factory(self):
        probe = self

        class _Converter:
            def __init__(self, **_kwargs):
                # Accepts (and drops) ctor kwargs — real _build_converter
                # passes format_options=… (ticket 03 OCR wiring).
                probe.constructed += 1

            def convert(self, *args, **kwargs):
                probe.converted += 1
                result = MagicMock()
                result.document = _make_doc(probe.specs)
                return result

        return _Converter


class _RecordingStore:
    """Stand-in vector store: records every upserted record."""

    def __init__(self):
        self.records = []

    def upsert(self, records, trace=None):
        self.records.extend(records)
        return None


def _make_pipeline(bm25_dir: Path, cache_dir: Path, pdf_path: Path,
                   store: _RecordingStore, settings):
    """Fake IngestionPipeline (bm25_reingest pattern) with REAL parser /
    chunker / upserter / BM25 components wired for this seam."""
    fp = type("FP", (), {"collection": COLLECTION, "force": True})()

    fp.integrity_checker = MagicMock()
    fp.integrity_checker.compute_sha256.return_value = "unused_by_parser"
    fp.integrity_checker.should_skip.return_value = False

    fp.parser = DoclingParser(
        settings, collection=COLLECTION,
        image_storage_dir=str(bm25_dir / "images"),
        extract_images=False, parse_cache_dir=str(cache_dir),
    )
    fp.chunker = DocumentChunker(settings)

    fp.chunk_refiner = MagicMock()
    fp.chunk_refiner.transform.side_effect = lambda chunks, trace=None: chunks
    fp.metadata_enricher = MagicMock()
    fp.metadata_enricher.transform.side_effect = lambda chunks, trace=None: chunks
    fp.image_captioner = MagicMock()
    fp.image_captioner.transform.side_effect = lambda chunks, trace=None: chunks
    fp.table_summarizer = MagicMock()
    fp.table_summarizer.transform.side_effect = lambda chunks, trace=None: chunks

    def _stat(text):
        tokens = [t.strip(".,").lower() for t in text.split()]
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        return {"term_frequencies": tf, "doc_length": len(tokens)}

    fp.batch_processor = MagicMock()

    def _process(chunks, trace=None):
        result = MagicMock()
        result.dense_vectors = [[0.1, 0.2]] * len(chunks)
        result.sparse_stats = [_stat(c.text) for c in chunks]
        return result

    fp.batch_processor.process.side_effect = _process

    original_create = VectorStoreFactory.create
    VectorStoreFactory.create = staticmethod(lambda settings, **kw: store)
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


def _store_texts(store: _RecordingStore):
    """Persisted chunk content: {chunk_id: text} from the upsert records."""
    return {r["id"]: r["metadata"]["text"] for r in store.records}


def _load_stage_payload(trace):
    """Payload of the 'load' stage record_stage call (None if absent)."""
    for call in trace.record_stage.call_args_list:
        if call.args[0] == "load":
            return call.args[1]
    return None


@pytest.fixture
def repo_settings():
    return load_settings("config/settings.yaml")


@pytest.fixture
def fake_pdf(tmp_path):
    p = tmp_path / "paper.pdf"
    p.write_bytes(b"%PDF-1.4 d036 replay seam fixture")
    return p


def test_force_reingest_replays_without_converter_and_persists_identical_chunks(
    tmp_path, repo_settings, fake_pdf
):
    cache_dir = tmp_path / "parsed"
    bm25_dir = tmp_path / "bm25"

    # ── Ingest #1 (force): real parse, cache written ────────────────────
    probe1 = _ConverterProbe(SPECS)
    store1 = _RecordingStore()
    fp1 = _make_pipeline(bm25_dir, cache_dir, fake_pdf, store1, repo_settings)
    trace1 = MagicMock()
    with patch("src.libs.parser.docling_parser.DocumentConverter",
               probe1.factory()), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False):
        result1 = IngestionPipeline.run(fp1, str(fake_pdf), trace=trace1)

    assert result1.success, result1.error
    assert result1.chunk_count > 0
    assert probe1.converted == 1  # first ingest really parsed
    assert any(cache_dir.rglob("manifest.json")), "first ingest must write cache"
    payload1 = _load_stage_payload(trace1)
    assert payload1 is not None
    assert "parse_cache_replayed" not in payload1  # real parse, unflagged

    ids1 = result1.vector_ids
    texts1 = _store_texts(store1)
    assert len(texts1) == len(ids1)

    # ── Ingest #2 (force, fresh process): replay, zero converter calls ──
    probe2 = _ConverterProbe(SPECS)
    store2 = _RecordingStore()
    fp2 = _make_pipeline(bm25_dir, cache_dir, fake_pdf, store2, repo_settings)
    trace2 = MagicMock()
    with patch("src.libs.parser.docling_parser.DocumentConverter",
               probe2.factory()), \
         patch("src.libs.parser.docling_parser.DoclingDocument") as mock_dd, \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False):
        mock_dd.load_from_json = MagicMock(side_effect=_load_from_json)
        result2 = IngestionPipeline.run(fp2, str(fake_pdf), trace=trace2)

    assert result2.success, result2.error
    assert probe2.constructed == 0, "second --force must not build a converter"
    assert probe2.converted == 0, "second --force must not call convert()"
    mock_dd.load_from_json.assert_called_once()
    # Trace visibility (D-036): replay is distinguishable in traces.jsonl.
    assert _load_stage_payload(trace2).get("parse_cache_replayed") is True

    # Chunk-id scheme + content identical to the first ingest (D-031 shape).
    assert result2.vector_ids == ids1
    assert _store_texts(store2) == texts1

    # BM25 prefix contract: index holds exactly the current chunk set, once
    # each — no phantoms or duplicates from the re-ingest.
    index = _load_index(bm25_dir)
    assert _unique_chunk_ids(index) == set(ids1)
    assert index["metadata"]["num_docs"] == len(set(ids1))
    for term, term_data in index["index"].items():
        ids = [p["chunk_id"] for p in term_data["postings"]]
        assert len(ids) == len(set(ids)), f"duplicate posting for {term!r}"

    # Sanity: ids follow the production scheme (path-hash prefix + index +
    # content hash), i.e. the contract was not silently changed.
    prefix = VectorUpserter.chunk_id_prefix(str(Path(str(fake_pdf))))
    for i, cid in enumerate(ids1):
        parts = cid.split("_")
        assert parts[0] == prefix
        assert parts[1] == f"{i:04d}"
        assert len(parts[2]) == 8
