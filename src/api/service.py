"""RAG service layer for the multi-tenant web API (D-032).

Sits between FastAPI and the query engine. Responsibilities:
- resolve the user's visible libraries server-side (store.resolve_visible_libraries)
- retrieve from every visible library (per-collection component cache reusing
  the exact HybridSearch stack the CLI uses) and fuse across libraries with
  outer RRF over per-library rankings
- dedup by chunk content hash — the same report copied into both the public
  and a personal library surfaces once, public copy wins
- optional rerank + LLM generation with [n] citations aligned to the result
  order returned to the client

Ingestion runs the standard IngestionPipeline against the target library's
collection, so D-031 version management (fingerprint / supersede / caches)
applies to web uploads unchanged.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import load_settings, resolve_path
from src.core.trace import TraceContext, TraceCollector
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.hybrid_search import create_hybrid_search
from src.core.query_engine.dense_retriever import create_dense_retriever
from src.core.query_engine.sparse_retriever import create_sparse_retriever
from src.core.query_engine.reranker import create_core_reranker
from src.core.query_engine.generator import create_core_generator
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.observability.logger import get_logger
from src.api import store

logger = get_logger(__name__)

RRF_K = 60  # same constant as the intra-library fusion


@dataclass
class _ScoredChunk:
    """A retrieval result tagged with its originating library."""

    result: Any                      # RetrievalResult
    library: Dict[str, Any]          # library row (id/owner_type/.../collection)
    rrf_score: float = 0.0


class RagService:
    """Multi-library query + ingest service (one instance per process)."""

    def __init__(self, settings: Optional[Any] = None):
        self.settings = settings or load_settings(
            str(resolve_path("config/settings.yaml"))
        )
        store.init_store()
        # Per-collection query component cache. Heavy objects (embedding
        # client, vector store handle) are built once and reused.
        self._components: Dict[str, Dict[str, Any]] = {}
        self._comp_lock = threading.Lock()
        # Shared reranker / generator are collection-agnostic.
        self._reranker = None
        self._generator = None
        # Ingest task registry: task_id -> {status, stage, error, file, library}
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._task_lock = threading.Lock()

    # ── component wiring (mirrors scripts/query.py::_build_components) ──

    def _query_stack(self, collection: str) -> Any:
        with self._comp_lock:
            stack = self._components.get(collection)
            if stack is not None:
                return stack["hybrid_search"]

            settings = self.settings
            vector_store = VectorStoreFactory.create(
                settings, collection_name=collection
            )
            embedding_client = EmbeddingFactory.create(settings)
            dense = create_dense_retriever(
                settings=settings,
                embedding_client=embedding_client,
                vector_store=vector_store,
            )
            bm25 = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
            sparse = create_sparse_retriever(
                settings=settings, bm25_indexer=bm25, vector_store=vector_store
            )
            sparse.default_collection = collection
            hybrid = create_hybrid_search(
                settings=settings,
                query_processor=QueryProcessor(),
                dense_retriever=dense,
                sparse_retriever=sparse,
            )
            self._components[collection] = {"hybrid_search": hybrid}
            return hybrid

    def _get_reranker(self):
        if self._reranker is None:
            self._reranker = create_core_reranker(settings=self.settings)
        return self._reranker

    def _get_generator(self):
        if self._generator is None:
            self._generator = create_core_generator(settings=self.settings)
        return self._generator

    # ── query ──────────────────────────────────────────────────────────

    def answer(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        libraries: Optional[List[str]] = None,
        use_rerank: bool = True,
        use_generate: bool = True,
    ) -> Dict[str, Any]:
        visible = store.resolve_visible_libraries(user_id, requested=libraries)
        if not visible:
            return {
                "answer": None,
                "results": [],
                "libraries": [],
                "error": "no visible libraries for this request",
            }

        trace = TraceContext(trace_type="query")
        trace.metadata["query"] = query[:200]
        trace.metadata["user_id"] = user_id
        trace.metadata["libraries"] = [lib["id"] for lib in visible]

        # 1) retrieve per library (skip broken libraries — graceful degradation)
        per_lib: List[List[Any]] = []
        degraded: List[str] = []
        for lib in visible:
            try:
                hybrid = self._query_stack(lib["collection"])
                per_lib.append(hybrid.search(query=query, top_k=top_k, filters=None, trace=trace))
            except Exception as e:
                logger.warning(f"library {lib['id']} retrieval failed: {e}")
                degraded.append(lib["id"])

        # 2) outer RRF fusion over per-library rankings
        scored: List[_ScoredChunk] = []
        for lib, results in zip(visible, per_lib):
            for rank, result in enumerate(results or [], start=1):
                scored.append(
                    _ScoredChunk(
                        result=result,
                        library=lib,
                        rrf_score=1.0 / (RRF_K + rank),
                    )
                )
        scored.sort(key=lambda s: s.rrf_score, reverse=True)

        # 3) cross-library dedup by chunk content hash (last id segment).
        #    Same report copied into public + personal libraries surfaces
        #    once; the public copy wins over a personal copy.
        seen: Dict[str, _ScoredChunk] = {}
        for item in scored:
            content_hash = item.result.chunk_id.rsplit("_", 1)[-1]
            existing = seen.get(content_hash)
            if existing is None:
                seen[content_hash] = item
            elif (
                item.library["owner_type"] == "public"
                and existing.library["owner_type"] != "public"
            ):
                item.rrf_score = max(item.rrf_score, existing.rrf_score)
                seen[content_hash] = item

        fused = sorted(seen.values(), key=lambda s: s.rrf_score, reverse=True)[
            :top_k
        ]
        # (result, library) pairs stay joined through rerank so the [n]
        # citation order the generator uses == the order returned to client.
        final_pairs: List[tuple] = [(item.result, item.library) for item in fused]
        results = [pair[0] for pair in final_pairs]

        # 4) optional rerank + generation (same objects/flow as the CLI)
        model: Optional[str] = None
        answer: Optional[str] = None
        if results:
            if use_rerank:
                reranker = self._get_reranker()
                if reranker.is_enabled:
                    try:
                        reranked = reranker.rerank(
                            query=query, results=results, top_k=top_k, trace=trace
                        ).results
                        lib_by_chunk = {item.result.chunk_id: item.library for item in fused}
                        fallback_lib = fused[0].library
                        final_pairs = [
                            (r, lib_by_chunk.get(r.chunk_id, fallback_lib))
                            for r in reranked
                        ]
                        results = [pair[0] for pair in final_pairs]
                    except Exception as e:
                        logger.warning(f"rerank failed, keeping RRF order: {e}")
            if use_generate:
                generator = self._get_generator()
                if generator.is_enabled:
                    try:
                        generation = generator.generate(
                            query=query, results=results, trace=trace
                        )
                        answer = generation.answer
                        model = generation.model
                    except Exception as e:
                        logger.warning(f"generation failed: {e}")

        TraceCollector().collect(trace)

        return {
            "answer": answer,
            "model": model,
            "results": [self._format_result(pair) for pair in final_pairs],
            "libraries": [
                {k: lib[k] for k in ("id", "owner_type", "display_name")}
                for lib in visible
            ],
            "degraded_libraries": degraded,
        }

    @staticmethod
    def _format_result(pair: tuple) -> Dict[str, Any]:
        result, library = pair
        meta = result.metadata or {}
        source = str(meta.get("source_path", ""))
        return {
            "chunk_id": result.chunk_id,
            "score": round(result.score, 4),
            "library": library["id"],
            "library_type": library["owner_type"],
            "source": source.rsplit("\\", 1)[-1].rsplit("/", 1)[-1],
            "page": meta.get("page_num"),
            "chunk_index": meta.get("chunk_index"),
            "snippet": (result.text or "").strip()[:300],
        }

    # ── libraries / documents ─────────────────────────────────────────

    def list_libraries(self, user_id: str) -> List[Dict[str, Any]]:
        return store.resolve_visible_libraries(user_id)

    def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        from src.libs.loader.file_integrity import SQLiteIntegrityChecker

        checker = SQLiteIntegrityChecker(
            str(resolve_path("data/db/ingestion_history.db"))
        )
        try:
            docs = []
            for lib in store.resolve_visible_libraries(user_id):
                for row in checker.list_processed(collection=lib["collection"]):
                    docs.append(
                        {
                            "library": lib["id"],
                            "file": Path(row["file_path"]).name,
                            "business_key": row.get("business_key"),
                            "processed_at": row.get("processed_at"),
                        }
                    )
            return docs
        finally:
            checker.close()

    # ── ingest (async task) ───────────────────────────────────────────

    def start_ingest(
        self,
        user_id: str,
        file_path: str,
        library_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        user = store.get_user(user_id)
        if user is None:
            return {"error": f"unknown user {user_id}"}

        if library_id:
            lib = next(
                (l for l in store.resolve_visible_libraries(user_id, requested=[library_id])),
                None,
            )
            if lib is None:
                return {"error": f"library {library_id} not visible/allowed"}
            if lib["owner_type"] == "public" and user["role"] != "admin":
                return {"error": "public library ingest requires admin role"}
        else:
            lib = store.ensure_personal_library(user_id)

        task_id = uuid.uuid4().hex[:12]
        with self._task_lock:
            self._tasks[task_id] = {
                "status": "pending",
                "stage": "queued",
                "file": Path(file_path).name,
                "library": lib["id"],
                "error": None,
            }
        return {"task_id": task_id, "library": lib["id"]}

    def run_ingest_task(self, task_id: str, user_id: str, file_path: str, library_id: str) -> None:
        """Executed in a worker thread by the API layer (FastAPI BackgroundTasks)."""
        task = self._tasks.get(task_id)
        lib = next(
            (l for l in store.resolve_visible_libraries(user_id) if l["id"] == library_id),
            None,
        )
        if task is None or lib is None:
            return
        task.update(status="running", stage="init")

        def _progress(stage_name: str, step: int, total: int) -> None:
            task.update(stage=stage_name, progress=f"{step}/{total}")

        try:
            pipeline = IngestionPipeline(
                self.settings, collection=lib["collection"]
            )
            result = pipeline.run(file_path, on_progress=_progress)
            if result.success:
                task.update(status="success", stage="done", chunks=result.chunk_count)
            else:
                task.update(status="failed", error="pipeline returned failure")
        except Exception as e:
            task.update(status="failed", error=str(e))

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._task_lock:
            task = self._tasks.get(task_id)
            return dict(task) if task else None
