"""Unit tests for HybridSearch references-section filtering (T19).

Mount contract: when ``retrieval.filter_references`` is on, references chunks
detected in the *fused* candidate pool are removed BEFORE the top-k
truncation, so deeper candidates backfill (后续候选顶上). Rationale: the
cross-encoder re-scores every candidate independently, so demotion-to-tail
pre-rerank is nullified — only removal from the pool survives rerank.

Both consumer paths (EvalRunner / query_knowledge_hub MCP tool) call
``HybridSearch.search``, so this single mount point covers them without
touching either caller.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.query_engine.hybrid_search import (
    HybridSearch,
    HybridSearchConfig,
)
from src.core.query_engine.fusion import RRFFusion
from src.core.query_engine.query_processor import QueryProcessor
from src.core.settings import RetrievalSettings, Settings
from src.core.trace import TraceContext
from src.core.types import RetrievalResult

# Abridged from real corpus chunk d28d9bf1_0066 (q7's rerank intruder).
REFS_TEXT = (
    "Cid, C., Murphy, S., Robshaw, M.: Small scale variants of the aes. "
    "In: International Conference on Fast Software Encryption (2005)\n\n"
    "Grassl, M., Langenberg, B., Roetteler, M., Steinwandt, R.: Applying "
    "grover's algorithm to aes. In: Springer International Publishing (2015)\n\n"
    "Grover, L.K.: A fast quantum mechanical algorithm for database search (1996)\n\n"
    "Daemen, J., Rijmen, V.: The Design of Rijndael: AES (2002)"
)

# Abridged from real corpus chunk 0fd74b7e_0060 (q22's squeezed-out victim).
CONTENT_TEXT = (
    "One can easily check that both circuits listed in Table 4 perform the "
    "same function. However, Circuit 2 costs one more CNOT gate than "
    "Circuit 1. Besides, the Toffoli depth of Circuit 2 is two, while the "
    "Toffoli depth of Circuit 1 is one.\n\n"
    "Observation 1 Given a quantum circuit with Toffoli gates involved, the "
    "Toffoli depth and the CNOT gate consumption may be affected by the "
    "specific arrangement of CNOT gates.\n\n"
    "Example 2 For a quantum circuit denoted by Circuit 3 in Table 5, a is "
    "not the operand of the second operation."
)


class StubRetriever:
    """Duck-typed retriever returning a fixed ranking."""

    def __init__(self, results: List[RetrievalResult]):
        self.results = results

    def retrieve(
        self,
        query: str = "",
        keywords: Optional[List[str]] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        return self.results[:top_k]


def _make_results(n: int, refs_at: tuple[int, ...] = ()) -> List[RetrievalResult]:
    """n results ranked best-first; positions in refs_at get references text."""
    return [
        RetrievalResult(
            chunk_id=f"c_{i:02d}",
            score=1.0 - i * 0.01,
            text=REFS_TEXT if i in refs_at else f"{CONTENT_TEXT}",
            metadata={"source_path": f"doc_{i % 3}.pdf"},
        )
        for i in range(n)
    ]


def _hybrid(
    dense: List[RetrievalResult],
    sparse: List[RetrievalResult],
    config: HybridSearchConfig,
) -> HybridSearch:
    return HybridSearch(
        query_processor=QueryProcessor(),
        dense_retriever=StubRetriever(dense),
        sparse_retriever=StubRetriever(sparse),
        fusion=RRFFusion(k=60),
        config=config,
    )


class TestReferenceFilterWiring:
    def test_filter_on_removes_refs_and_backfills_beyond_topk(self) -> None:
        """References chunk fused at #1 is removed; rank-(top_k+1) candidate
        backfills — the pool must fuse fully BEFORE truncating."""
        # RRF over identical orderings = that order. 8 candidates, refs at 0
        # and 4; top_k=3 → without filter: c_00,c_01,c_02; with filter: the
        # pool is fused fully (8), refs removed, then truncated.
        results = _make_results(8, refs_at=(0, 4))
        hybrid = _hybrid(
            results,
            results,
            HybridSearchConfig(
                dense_top_k=8, sparse_top_k=8, filter_references=True,
                parallel_retrieval=False,
            ),
        )

        got = hybrid.search("quantum gates depth", top_k=3)

        ids = [r.chunk_id for r in got]
        assert "c_00" not in ids and "c_04" not in ids  # refs removed
        assert ids == ["c_01", "c_02", "c_03"]  # backfill from beyond top_k

    def test_filter_off_by_default_bit_identical(self) -> None:
        """Default config: refs chunk stays, ordering unchanged (zero-diff
        control arm for the T19 A/B)."""
        results = _make_results(8, refs_at=(0, 4))
        cfg_off = HybridSearchConfig(dense_top_k=8, sparse_top_k=8, parallel_retrieval=False)

        on_list = _hybrid(results, results, cfg_off).search("quantum gates depth", top_k=3)
        assert [r.chunk_id for r in on_list] == ["c_00", "c_01", "c_02"]

    def test_filter_records_trace_stage(self) -> None:
        results = _make_results(6, refs_at=(0,))
        hybrid = _hybrid(
            results,
            results,
            HybridSearchConfig(
                dense_top_k=6, sparse_top_k=6, filter_references=True,
                parallel_retrieval=False,
            ),
        )
        trace = TraceContext(trace_type="query")

        hybrid.search("quantum gates depth", top_k=3, trace=trace)

        stage_names = [s.get("stage") for s in trace.stages]
        assert "reference_filter" in stage_names
        stage = next(s for s in trace.stages if s.get("stage") == "reference_filter")
        assert stage["data"]["removed_chunk_ids"] == ["c_00"]
        assert stage["data"]["kept_count"] == 5

    def test_filter_applies_on_single_path_fallback(self) -> None:
        """Dense-only fallback (sparse has no keywords): refs still removed —
        list shortens, no backfill possible (pool exhausted)."""
        results = _make_results(4, refs_at=(1,))

        class NoKeywordsProcessor(QueryProcessor):
            def process(self, query: str):
                from src.core.types import ProcessedQuery
                return ProcessedQuery(original_query=query, keywords=[], filters={})

        hybrid = HybridSearch(
            query_processor=NoKeywordsProcessor(),
            dense_retriever=StubRetriever(results),
            sparse_retriever=StubRetriever(results),
            fusion=RRFFusion(k=60),
            config=HybridSearchConfig(
                dense_top_k=4, sparse_top_k=4, filter_references=True,
                parallel_retrieval=False,
            ),
        )

        got = hybrid.search("quantum gates depth", top_k=4)
        assert [r.chunk_id for r in got] == ["c_00", "c_02", "c_03"]

    def test_all_refs_pool_returns_empty_not_raise(self) -> None:
        """Pathological pool that is 100% references: filter yields an empty
        list (nothing to backfill from) rather than raising."""
        results = _make_results(5, refs_at=(0, 1, 2, 3, 4))
        hybrid = _hybrid(
            results,
            results,
            HybridSearchConfig(
                dense_top_k=5, sparse_top_k=5, filter_references=True,
                parallel_retrieval=False,
            ),
        )
        got = hybrid.search("quantum gates depth", top_k=3)
        assert got == []

    def test_full_pool_fusion_prefix_slice_invariant(self) -> None:
        """Equivalence guard for the control arm: with the filter OFF,
        fusing the full pool then slicing must be element-wise identical to
        fusing with top_k directly (RRF is a total order — score desc,
        chunk_id tiebreak — so the prefix is well-defined). The zero-diff
        A/B claim rests on this invariant."""
        results = _make_results(10, refs_at=())  # no refs: pure RRF ordering
        lists = [results, list(reversed(results))]
        truncated = RRFFusion(k=60).fuse(ranking_lists=lists, top_k=4)
        full = RRFFusion(k=60).fuse(ranking_lists=lists, top_k=None)

        assert [r.chunk_id for r in truncated] == [
            r.chunk_id for r in full[:4]
        ]


class TestConfigExtraction:
    def _settings(self, filter_references: Optional[bool]) -> Settings:
        base = Settings.__new__(Settings)  # only retrieval is read
        kw: Dict[str, Any] = dict(
            dense_top_k=20, sparse_top_k=10, fusion_top_k=10, rrf_k=60,
        )
        if filter_references is not None:
            kw["filter_references"] = filter_references
        # Settings is frozen; bypass to inject just the retrieval field
        object.__setattr__(base, "retrieval", RetrievalSettings(**kw))
        return base

    def test_extract_reads_flag_from_settings(self) -> None:
        hybrid = HybridSearch(settings=self._settings(True))
        assert hybrid.config.filter_references is True

    def test_extract_defaults_off_when_absent(self) -> None:
        hybrid = HybridSearch(settings=self._settings(None))
        assert hybrid.config.filter_references is False
