"""Ticket 05: table-chunk summary transform — mock-LLM seam tests.

Precedent: the MetadataEnricher mock-injection pattern (the transform takes
an optional ``llm``; tests pass a fake and the factory is never called).
Covers the ticket checklist: summary prefix on table chunks, cache hits
(no repeat LLM calls), failure degradation, provider-config derivation, and
zero impact on non-table chunks.
"""

from unittest.mock import Mock

from src.core.settings import LLMSettings, Settings, TableSummarizerSettings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.transform.table_summarizer import (
    SUMMARY_MARKER,
    TableSummarizer,
    resolve_llm_config,
)
from src.libs.llm.base_llm import ChatResponse


class FakeLLM:
    """Stand-in for BaseLLM: records calls, optional failure mode."""

    def __init__(self, content: str = "Latency figures for the evaluated systems.", fail: bool = False):
        self.content = content
        self.fail = fail
        self.calls = []  # list of (messages, kwargs)

    def chat(self, messages, trace=None, **kwargs):
        self.calls.append((messages, kwargs))
        if self.fail:
            raise RuntimeError("LLM unavailable")
        return ChatResponse(content=self.content, model="fake-model", usage=None)


def make_settings(cfg):
    settings = Mock(spec=Settings)
    settings.llm = LLMSettings(
        provider="deepseek",
        model="deepseek-chat",
        temperature=0.0,
        max_tokens=8192,
        base_url="https://api.deepseek.com",
    )
    settings.ingestion = Mock()
    settings.ingestion.table_summarizer = cfg
    return settings


ENABLED_CFG = TableSummarizerSettings(
    enabled=True, provider="deepseek", model="deepseek-flash", max_summary_tokens=120
)


def table_chunk(cid="t1", text="System A 12ms\nSystem B 30ms", table_html="| System | Latency |\n|---|---|\n| A | 12ms |"):
    return Chunk(
        id=cid,
        text=text,
        metadata={"section_type": "table", "table_html": table_html, "source_path": "paper.pdf"},
        source_ref="paper.pdf",
    )


def text_chunk(cid="c1"):
    return Chunk(
        id=cid,
        text="Plain prose about the experimental setup.",
        metadata={"section_type": "text", "source_path": "paper.pdf"},
        source_ref="paper.pdf",
    )


# ---------------------------------------------------------------------------
# Summary stitching
# ---------------------------------------------------------------------------

def test_table_chunk_gets_summary_prepended(tmp_path):
    llm = FakeLLM(content="Latency of A is 12 ms and B is 30 ms.")
    ts = TableSummarizer(make_settings(ENABLED_CFG), llm=llm, cache_path=tmp_path / "cache.db")

    out = ts.transform([table_chunk()])

    assert out[0].text.startswith(SUMMARY_MARKER)
    assert "Latency of A is 12 ms and B is 30 ms." in out[0].text
    # Original chunk body survives after the summary (same-chunk evidence rule)
    assert "System A 12ms" in out[0].text
    assert out[0].metadata["table_summarized_by"] == "llm"


def test_non_table_chunks_untouched(tmp_path):
    llm = FakeLLM()
    ts = TableSummarizer(make_settings(ENABLED_CFG), llm=llm, cache_path=tmp_path / "cache.db")
    chunk = text_chunk()
    before = chunk.text

    out = ts.transform([chunk])

    assert out[0].text == before
    assert "table_summarized_by" not in out[0].metadata
    assert llm.calls == []


def test_prompt_contains_chunk_text_and_table_gfm(tmp_path):
    llm = FakeLLM()
    ts = TableSummarizer(make_settings(ENABLED_CFG), llm=llm, cache_path=tmp_path / "cache.db")

    ts.transform([table_chunk()])

    prompt = llm.calls[0][0][0].content
    assert "System A 12ms" in prompt  # chunk text
    assert "| System | Latency |" in prompt  # full-table GFM
    assert "{table_gfm}" not in prompt and "{chunk_text}" not in prompt


def test_call_passes_token_budget(tmp_path):
    llm = FakeLLM()
    ts = TableSummarizer(make_settings(ENABLED_CFG), llm=llm, cache_path=tmp_path / "cache.db")

    ts.transform([table_chunk()])

    kwargs = llm.calls[0][1]
    assert kwargs.get("max_tokens") == 120
    assert kwargs.get("temperature") == 0.0


def test_prompt_budget_follows_config(tmp_path):
    """The prompt's token instruction must track max_summary_tokens, not a
    hardcoded number (a 60-token config must not ask the model for 120)."""
    cfg = TableSummarizerSettings(
        enabled=True, provider="deepseek", model="deepseek-flash", max_summary_tokens=60
    )
    llm = FakeLLM()
    ts = TableSummarizer(make_settings(cfg), llm=llm, cache_path=tmp_path / "cache.db")

    ts.transform([table_chunk()])

    assert "at most 60 tokens" in llm.calls[0][0][0].content
    assert llm.calls[0][1]["max_tokens"] == 60


def test_no_table_html_falls_back_to_chunk_text(tmp_path):
    llm = FakeLLM()
    ts = TableSummarizer(make_settings(ENABLED_CFG), llm=llm, cache_path=tmp_path / "cache.db")
    chunk = Chunk(
        id="t2",
        text="| Model | F1 |\n| m1 | 0.9 |",
        metadata={"section_type": "table", "source_path": "paper.pdf"},
        source_ref="paper.pdf",
    )

    out = ts.transform([chunk])

    assert out[0].text.startswith(SUMMARY_MARKER)
    assert "| Model | F1 |" in llm.calls[0][0][0].content


def test_already_summarized_chunk_not_double_prefixed(tmp_path):
    llm = FakeLLM()
    ts = TableSummarizer(make_settings(ENABLED_CFG), llm=llm, cache_path=tmp_path / "cache.db")
    pre = f"{SUMMARY_MARKER} old summary)\n\nrow data"
    chunk = table_chunk(text=pre, table_html="| a | b |")

    out = ts.transform([chunk])

    assert out[0].text == pre  # unchanged, no LLM call
    assert llm.calls == []


# ---------------------------------------------------------------------------
# Cache (ticket: same table must not trigger repeat LLM calls)
# ---------------------------------------------------------------------------

def test_second_transform_hits_cache(tmp_path):
    llm = FakeLLM()
    ts = TableSummarizer(make_settings(ENABLED_CFG), llm=llm, cache_path=tmp_path / "cache.db")

    ts.transform([table_chunk()])
    ts.transform([table_chunk()])  # fresh run, same content

    assert len(llm.calls) == 1


def test_cache_persists_across_instances(tmp_path):
    """--force re-ingest is a new process: the cache must live on disk."""
    ts1 = TableSummarizer(make_settings(ENABLED_CFG), llm=FakeLLM(), cache_path=tmp_path / "cache.db")
    ts1.transform([table_chunk()])

    llm2 = FakeLLM()
    ts2 = TableSummarizer(make_settings(ENABLED_CFG), llm=llm2, cache_path=tmp_path / "cache.db")
    out = ts2.transform([table_chunk()])

    assert llm2.calls == []
    assert out[0].text.startswith(SUMMARY_MARKER)


def test_cache_init_failure_degrades_to_no_cache(tmp_path):
    """D-005: a broken cache DB must not take the pipeline down — summarize
    without persistence instead (an existing directory can't be a SQLite file)."""
    bad_path = tmp_path / "not_a_db"
    bad_path.mkdir()
    llm = FakeLLM()
    ts = TableSummarizer(make_settings(ENABLED_CFG), llm=llm, cache_path=bad_path)

    out = ts.transform([table_chunk()])

    assert ts._cache is None
    assert len(llm.calls) == 1  # still summarized, just not cached
    assert out[0].text.startswith(SUMMARY_MARKER)


def test_identical_chunks_share_one_call(tmp_path):
    llm = FakeLLM()
    ts = TableSummarizer(make_settings(ENABLED_CFG), llm=llm, cache_path=tmp_path / "cache.db")
    chunks = [table_chunk(cid="t1"), table_chunk(cid="t2")]  # same content, two chunks

    out = ts.transform(chunks)

    assert len(llm.calls) == 1
    assert all(c.text.startswith(SUMMARY_MARKER) for c in out)


# ---------------------------------------------------------------------------
# Degradation (D-005: summarizer must never block ingestion)
# ---------------------------------------------------------------------------

def test_llm_failure_chunk_falls_back_to_no_summary(tmp_path):
    llm = FakeLLM(fail=True)
    ts = TableSummarizer(make_settings(ENABLED_CFG), llm=llm, cache_path=tmp_path / "cache.db")
    table = table_chunk()
    before = table.text

    out = ts.transform([table, text_chunk()])

    assert len(out) == 2  # pipeline gets all chunks back
    assert out[0].text == before  # table chunk: no-summary form
    assert "table_summarized_by" not in out[0].metadata


def test_empty_llm_response_treated_as_failure(tmp_path):
    llm = FakeLLM(content="   ")
    ts = TableSummarizer(make_settings(ENABLED_CFG), llm=llm, cache_path=tmp_path / "cache.db")
    table = table_chunk()

    out = ts.transform([table])

    assert out[0].text == table.text


def test_absent_config_block_is_noop(tmp_path):
    ts = TableSummarizer(make_settings(None), llm=FakeLLM(), cache_path=tmp_path / "cache.db")
    chunk = table_chunk()

    out = ts.transform([chunk])

    assert ts.llm is None
    assert out[0].text == chunk.text


def test_enabled_false_is_noop(tmp_path):
    cfg = TableSummarizerSettings(enabled=False, provider="deepseek", model="deepseek-flash")
    ts = TableSummarizer(make_settings(cfg), llm=FakeLLM(), cache_path=tmp_path / "cache.db")
    chunk = table_chunk()

    out = ts.transform([chunk])

    assert ts.llm is None
    assert out[0].text == chunk.text


# ---------------------------------------------------------------------------
# Provider config (independent block, switchable to a local model)
# ---------------------------------------------------------------------------

def test_resolve_llm_same_provider_inherits_endpoint():
    base = LLMSettings(
        provider="deepseek", model="deepseek-chat", temperature=0.0, max_tokens=8192,
        base_url="https://api.deepseek.com", api_key="key-1",
    )
    cfg = TableSummarizerSettings(provider="deepseek", model="deepseek-flash")

    out = resolve_llm_config(base, cfg)

    assert (out.provider, out.model) == ("deepseek", "deepseek-flash")
    assert out.base_url == "https://api.deepseek.com"
    assert out.api_key == "key-1"


def test_resolve_llm_different_provider_drops_endpoint():
    """deepseek base_url/api_key are wrong for ollama — provider defaults apply."""
    base = LLMSettings(
        provider="deepseek", model="deepseek-chat", temperature=0.0, max_tokens=8192,
        base_url="https://api.deepseek.com", api_key="key-1",
    )
    cfg = TableSummarizerSettings(provider="ollama", model="granite4.1:8b")

    out = resolve_llm_config(base, cfg)

    assert (out.provider, out.model) == ("ollama", "granite4.1:8b")
    assert out.base_url is None  # OllamaLLM falls back to its own default
    assert out.api_key is None


def test_resolve_llm_explicit_overrides_win():
    base = LLMSettings(
        provider="deepseek", model="deepseek-chat", temperature=0.0, max_tokens=8192,
        base_url="https://api.deepseek.com",
    )
    cfg = TableSummarizerSettings(
        provider="ollama", model="granite4.1:8b",
        base_url="http://localhost:11434", api_key="explicit",
    )

    out = resolve_llm_config(base, cfg)

    assert out.base_url == "http://localhost:11434"
    assert out.api_key == "explicit"


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

def test_trace_records_counts(tmp_path):
    llm = FakeLLM()
    ts = TableSummarizer(make_settings(ENABLED_CFG), llm=llm, cache_path=tmp_path / "cache.db")
    trace = TraceContext(trace_id="ts_test")

    ts.transform([table_chunk(), text_chunk()], trace=trace)
    ts.transform([table_chunk()], trace=trace)  # cache hit run

    first, second = (e["data"] for e in trace.stages if e["stage"] == "table_summarizer")
    assert first["table_chunks"] == 1 and first["llm_calls"] == 1 and first["cache_hits"] == 0
    assert second["table_chunks"] == 1 and second["llm_calls"] == 0 and second["cache_hits"] == 1
    assert second["summarized"] == 1
