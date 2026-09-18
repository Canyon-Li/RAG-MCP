"""Table-chunk summary transform (ticket 05 / D-024 追记).

``section_type=table`` chunks get an LLM paraphrase (≤ ``max_summary_tokens``,
prompt- and API-capped) prepended to the chunk text, so semantic table
queries ("which system has the lowest latency") hit the summary instead of
relying on the number rows' vectors. The summary only restates facts that
are explicitly present in the chunk / table (prompt constraint); the original
text stays in the same chunk, so a wrong paraphrase can only mis-rank, never
replace the evidence (spec user story 11).

Design points:
- **Independent provider/model** (``ingestion.table_summarizer``): the
  summarizer may point at deepseek-flash while the main ``llm`` block goes
  anywhere else, and can be switched back to a local model (ollama granite)
  — resolution happens in :func:`resolve_llm_config` on top of the existing
  LLMFactory, zero new factory code.
- **Persistent content-hash cache** (SQLite under ``data/db/``): the same
  table re-ingested — including a fresh ``--force`` process — never re-pays
  the LLM call. The key hashes the prompt too, so editing the prompt file
  invalidates stale summaries (D-036 version-stamp spirit).
- **Graceful degradation (D-005)**: LLM init failure disables the transform;
  a per-chunk call failure leaves that chunk in its no-summary form. The
  summarizer must never be the stage that blocks ingestion.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.core.settings import (
    LLMSettings,
    Settings,
    TableSummarizerSettings,
    resolve_path,
)
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_llm import BaseLLM, Message
from src.libs.llm.llm_factory import LLMFactory
from src.observability.logger import get_logger

logger = get_logger(__name__)

# Detection marker for the summary prefix (stitching + re-run guard).
SUMMARY_MARKER = "(Table Summary:"

# Same LLM-call budget as MetadataEnricher; table chunks are typically a
# large fraction of a document, so parallel calls keep ingest time sane.
DEFAULT_MAX_WORKERS = 5

# Gitignored under data/db/ next to the other SQLite stores.
DEFAULT_CACHE_PATH = "data/db/table_summary_cache.db"


def resolve_llm_config(
    base: LLMSettings, cfg: TableSummarizerSettings
) -> LLMSettings:
    """Merge the summarizer block into the main llm block.

    provider/model always come from the summarizer config. base_url/api_key
    inherit from the main block only when the provider is unchanged — a
    deepseek base_url is simply wrong for ollama, so a provider switch drops
    them (the provider's own defaults apply) unless the block sets them
    explicitly.
    """
    same_provider = base.provider.lower() == cfg.provider.lower()
    base_url = cfg.base_url or (base.base_url if same_provider else None)
    api_key = cfg.api_key or (base.api_key if same_provider else None)
    return replace(
        base, provider=cfg.provider, model=cfg.model,
        base_url=base_url, api_key=api_key,
    )


class TableSummaryCache:
    """Persistent cache_key → summary store (SQLite, main-thread use only).

    All reads/writes happen on the transform thread; only the LLM calls run
    in worker threads, so no cross-thread SQLite access.
    """

    def __init__(self, db_path: str | Path):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS table_summaries ("
            " cache_key TEXT PRIMARY KEY,"
            " summary TEXT NOT NULL,"
            " created_at TEXT NOT NULL DEFAULT (datetime('now')))"
        )
        self._conn.commit()

    def get(self, cache_key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT summary FROM table_summaries WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        return row[0] if row else None

    def put_many(self, items: Dict[str, str]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO table_summaries (cache_key, summary) "
            "VALUES (?, ?)",
            list(items.items()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class TableSummarizer(BaseTransform):
    """Prepends an LLM paraphrase to ``section_type=table`` chunk text.

    Prompt input = chunk text + the whole table's GFM (``metadata.table_html``
    — the adapter carries it separately from the chunk text; chunks without
    it fall back to their own text as the table content).
    """

    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        cache_path: Optional[str] = None,
    ):
        """Args:
            settings: Application settings.
            llm: Optional LLM instance (for testing; auto-created if None).
            cache_path: Optional cache DB path (default data/db/table_summary_cache.db).
        """
        self.settings = settings
        ingestion = getattr(settings, "ingestion", None)
        cfg = getattr(ingestion, "table_summarizer", None) if ingestion is not None else None
        self.config: Optional[TableSummarizerSettings] = cfg
        self.enabled = bool(cfg is not None and cfg.enabled)
        self.max_summary_tokens = cfg.max_summary_tokens if cfg is not None else 120
        # Narrowed copy for the enabled path (mypy: cfg is not None there).
        active_cfg = cfg if self.enabled else None

        self.llm: Optional[BaseLLM] = None
        self._cache: Optional[TableSummaryCache] = None
        self._prompt: str = ""  # empty = prompt file missing → inactive
        self._prompt_hash = ""

        if active_cfg is None:
            logger.info("TableSummarizer disabled (config off) — table chunks pass through")
            return

        try:
            self._cache = TableSummaryCache(
                cache_path or str(resolve_path(DEFAULT_CACHE_PATH))
            )
        except Exception as e:
            # D-005: a broken cache must not take the pipeline down —
            # summarize without persistence instead.
            logger.error(f"Table summary cache unavailable ({e}); continuing without cache")

        if llm is not None:
            self.llm = llm
        else:
            try:
                derived = replace(
                    settings, llm=resolve_llm_config(settings.llm, active_cfg)
                )
                self.llm = LLMFactory.create(derived)
            except Exception as e:
                # D-005: ingestion proceeds without table summaries.
                logger.error(
                    f"Failed to initialize table-summarizer LLM: {e}; "
                    f"table chunks will pass through without summaries"
                )

        if not self._load_prompt():
            logger.error(
                "config/prompts/table_summary.txt not found; table chunks "
                "will pass through without summaries"
            )

    # ------------------------------------------------------------------
    # Seam
    # ------------------------------------------------------------------

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None,
    ) -> List[Chunk]:
        """Prepend summaries to table chunks in place; never raises (D-005)."""
        if not chunks:
            return chunks
        if not self.enabled or self.llm is None or not self._prompt:
            return chunks

        t0 = time.monotonic()
        targets = [
            (i, c) for i, c in enumerate(chunks)
            if (c.metadata or {}).get("section_type") == "table"
            and c.text and c.text.strip()
            # Re-run guard: don't double-prefix already-summarized chunks.
            and not c.text.lstrip().startswith(SUMMARY_MARKER)
        ]

        llm_calls = 0
        cache_hits = 0
        failures = 0
        summarized = 0

        if targets:
            # Group by cache key: identical (table, chunk) content shares one
            # LLM call; the stitch phase fans the result back out.
            entries: Dict[str, List[Tuple[int, str, str]]] = {}
            for idx, chunk in targets:
                gfm = (chunk.metadata or {}).get("table_html") or chunk.text
                entries.setdefault(self._cache_key(gfm, chunk.text), []).append(
                    (idx, gfm, chunk.text)
                )

            results: Dict[str, str] = {}
            for key in entries:
                hit = self._cache_get(key)
                if hit is not None:
                    results[key] = hit
                    cache_hits += 1

            misses = [key for key in entries if key not in results]
            if misses:
                max_workers = min(DEFAULT_MAX_WORKERS, len(misses))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(self._summarize, entries[key][0][1], entries[key][0][2]): key
                        for key in misses
                    }
                    for future in as_completed(futures):
                        llm_calls += 1
                        try:
                            summary = future.result()
                        except Exception as e:
                            logger.warning(f"Table summary LLM call failed: {e}")
                            summary = None
                        if summary:
                            results[futures[future]] = summary
                        else:
                            failures += 1
                fresh = {key: results[key] for key in misses if key in results}
                if fresh:
                    self._cache_put(fresh)

            for key, group in entries.items():
                summary = results.get(key)
                if not summary:
                    continue
                prefix = f"{SUMMARY_MARKER} {summary})\n\n"
                for idx, _, _ in group:
                    chunks[idx].text = prefix + chunks[idx].text
                    if chunks[idx].metadata is None:
                        chunks[idx].metadata = {}
                    chunks[idx].metadata["table_summarized_by"] = "llm"
                    summarized += 1

        if trace is not None:
            trace.record_stage("table_summarizer", {
                "method": "llm_prefix",
                "table_chunks": len(targets),
                "summarized": summarized,
                "cache_hits": cache_hits,
                "llm_calls": llm_calls,
                "failures": failures,
            }, elapsed_ms=(time.monotonic() - t0) * 1000.0)

        logger.info(
            f"Table summaries: {summarized}/{len(targets)} table chunks "
            f"(LLM calls: {llm_calls}, cache hits: {cache_hits}, failures: {failures})"
        )
        return chunks

    def close(self) -> None:
        """Release the cache DB handle (pipeline close)."""
        if self._cache is not None:
            self._cache.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _summarize(self, table_gfm: str, chunk_text: str) -> Optional[str]:
        """One LLM call; returns the cleaned summary or None on empty output."""
        prompt = (
            self._prompt
            .replace("{max_tokens}", str(self.max_summary_tokens))
            .replace("{table_gfm}", table_gfm)
            .replace("{chunk_text}", chunk_text)
        )
        if self.llm is None:  # transform() already guards; narrowing only
            return None
        response = self.llm.chat(
            [Message(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=self.max_summary_tokens,
        )
        content = (getattr(response, "content", None) or "").strip()
        return content or None

    def _cache_get(self, cache_key: str) -> Optional[str]:
        """Cache read; None on miss, no-cache, or read failure (D-005)."""
        if self._cache is None:
            return None
        try:
            return self._cache.get(cache_key)
        except Exception as e:
            logger.warning(f"Table summary cache read failed ({e}); treating as miss")
            return None

    def _cache_put(self, items: Dict[str, str]) -> None:
        """Cache write; failure loses persistence but not the summary (D-005)."""
        if self._cache is None:
            return
        try:
            self._cache.put_many(items)
        except Exception as e:
            logger.warning(f"Table summary cache write failed ({e}); summaries not persisted")

    def _cache_key(self, table_gfm: str, chunk_text: str) -> str:
        """sha256(prompt, table GFM, chunk text) — spec: (表格内容哈希+块内容)."""
        digest = hashlib.sha256()
        for part in (self._prompt_hash, table_gfm, chunk_text):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def _load_prompt(self) -> bool:
        """Load the prompt template; False (inactive) when the file is absent."""
        prompt_path = resolve_path("config/prompts/table_summary.txt")
        if not prompt_path.exists():
            return False
        self._prompt = prompt_path.read_text(encoding="utf-8").strip()
        # Budget folded in: a different max_summary_tokens must invalidate
        # cached summaries just like a prompt edit does.
        material = f"{self._prompt}\x00{self.max_summary_tokens}"
        self._prompt_hash = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
        return True
