"""Ticket 05: table summaries through the pipeline seam.

Runs the real ``IngestionPipeline.run`` over a fake component set with a REAL
``TableSummarizer`` (mock LLM) wired in — the spec's main-seam coverage for
the summary prefix: what stage 4d produces is what encoding/storage receive.
Degradation case per the ticket: mock LLM failure → the table chunk falls
back to its no-summary form and the pipeline still succeeds.
"""

from unittest.mock import MagicMock

from src.core.settings import LLMSettings, Settings, TableSummarizerSettings
from src.core.types import Chunk, Document
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.transform.table_summarizer import (
    SUMMARY_MARKER,
    TableSummarizer,
)
from src.libs.llm.base_llm import ChatResponse


class FakeLLM:
    """Stand-in for BaseLLM: records calls, optional failure mode."""

    def __init__(self, content: str = "Latency figures for the evaluated systems.", fail: bool = False):
        self.content = content
        self.fail = fail

    def chat(self, messages, trace=None, **kwargs):
        if self.fail:
            raise RuntimeError("LLM unavailable")
        return ChatResponse(content=self.content, model="fake-model", usage=None)


def _make_settings() -> Settings:
    settings = MagicMock(spec=Settings)
    settings.llm = LLMSettings(
        provider="deepseek", model="deepseek-flash",
        temperature=0.0, max_tokens=8192,
    )
    settings.ingestion = MagicMock()
    settings.ingestion.table_summarizer = TableSummarizerSettings(
        enabled=True, provider="deepseek", model="deepseek-flash",
        max_summary_tokens=120,
    )
    return settings


def _make_fake_pipeline(table_summarizer: TableSummarizer):
    """Fake IngestionPipeline: every component mocked except 4d, which is real."""
    table = Chunk(
        id="c0",
        text="System A 12ms\nSystem B 30ms",
        metadata={
            "section_type": "table",
            "table_html": "| System | Latency |\n|---|---|\n| A | 12ms |",
            "source_path": "test.pdf",
        },
        source_ref="test.pdf",
    )
    prose = Chunk(
        id="c1",
        text="Plain prose about the experimental setup.",
        metadata={"section_type": "text", "source_path": "test.pdf"},
        source_ref="test.pdf",
    )
    chunks = [table, prose]

    class FP:
        collection = "test"
        force = False

    fp = FP()
    fp.integrity_checker = MagicMock()
    fp.integrity_checker.compute_sha256.return_value = "hash123"
    fp.integrity_checker.should_skip.return_value = False

    fp.parser = MagicMock()
    fp.parser.parse.return_value = Document(
        id="doc1", text="Hello world. " * 50,
        metadata={"source_path": "test.pdf", "images": []},
    )

    fp.chunker = MagicMock()
    fp.chunker.split_document.return_value = chunks

    fp.chunk_refiner = MagicMock()
    fp.chunk_refiner.transform.return_value = chunks
    fp.metadata_enricher = MagicMock()
    fp.metadata_enricher.transform.return_value = chunks
    fp.image_captioner = MagicMock()
    fp.image_captioner.transform.return_value = chunks
    fp.table_summarizer = table_summarizer

    batch_result = MagicMock()
    batch_result.dense_vectors = [[0.1, 0.2]] * 2
    batch_result.sparse_stats = [{"doc_id": f"c{i}"} for i in range(2)]
    fp.batch_processor = MagicMock()
    fp.batch_processor.process.return_value = batch_result

    fp.vector_upserter = MagicMock()
    fp.vector_upserter.upsert.return_value = ["v0", "v1"]
    fp.bm25_indexer = MagicMock()
    fp.image_storage = MagicMock()
    return fp, chunks


class TestPipelineTableSummary:

    def test_summary_prefix_survives_to_stage_output(self, tmp_path):
        ts = TableSummarizer(
            _make_settings(), llm=FakeLLM(), cache_path=tmp_path / "cache.db"
        )
        fp, chunks = _make_fake_pipeline(ts)

        result = IngestionPipeline.run(fp, "test.pdf")

        assert result.success
        # What the embedder/storer receive: prefixed table chunk, untouched prose.
        assert chunks[0].text.startswith(SUMMARY_MARKER)
        assert "System A 12ms" in chunks[0].text
        assert chunks[0].metadata["table_summarized_by"] == "llm"
        assert chunks[1].text == "Plain prose about the experimental setup."
        assert result.stages["transform"]["table_summarizer"]["summarized_chunks"] == 1

    def test_llm_failure_pipeline_still_succeeds(self, tmp_path):
        ts = TableSummarizer(
            _make_settings(), llm=FakeLLM(fail=True), cache_path=tmp_path / "cache.db"
        )
        fp, chunks = _make_fake_pipeline(ts)

        result = IngestionPipeline.run(fp, "test.pdf")

        assert result.success  # D-005: degradation never blocks ingestion
        assert chunks[0].text == "System A 12ms\nSystem B 30ms"  # no-summary form
        assert "table_summarized_by" not in chunks[0].metadata
        assert result.stages["transform"]["table_summarizer"]["summarized_chunks"] == 0
