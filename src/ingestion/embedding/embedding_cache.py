"""Persistent embedding cache keyed by (model, content hash) — D-031.

Chunk texts are content-stable across re-ingestions (deterministic chunking
of identical content), but the old pipeline re-embedded EVERY chunk on every
re-ingest. With this cache, a chunk whose text hash is already stored for the
active model skips the embedding call entirely — the bulk of the "incremental
ingest" speedup for re-distributed (watermarked) copies and version updates.

SQLite backend, same access pattern as ingestion_history (short-lived
connections, WAL). Vector payloads stored as JSON arrays.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


class SQLiteEmbeddingCache:
    """Persistent (model, content_hash) → vector cache."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    model TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (model, content_hash)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def get_many(
        self, model: str, hashes: Sequence[str]
    ) -> Dict[str, List[float]]:
        """Look up cached vectors. Returns {content_hash: vector}."""
        if not hashes:
            return {}
        conn = sqlite3.connect(self.db_path)
        try:
            found: Dict[str, List[float]] = {}
            # chunked IN-clauses (SQLite parameter limit safety)
            for lo in range(0, len(hashes), 500):
                batch = list(hashes[lo : lo + 500])
                placeholders = ",".join("?" * len(batch))
                cursor = conn.execute(
                    "SELECT content_hash, vector FROM embedding_cache "
                    f"WHERE model = ? AND content_hash IN ({placeholders})",
                    [model, *batch],
                )
                for content_hash, vector_json in cursor:
                    found[content_hash] = json.loads(vector_json)
            return found
        finally:
            conn.close()

    def put_many(
        self, model: str, items: Iterable[tuple[str, List[float]]]
    ) -> None:
        """Store vectors. items: iterable of (content_hash, vector)."""
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (model, h, len(v), json.dumps(v), now) for h, v in items if v
        ]
        if not rows:
            return
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache "
                "(model, content_hash, dim, vector, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def stats(self, model: str) -> Dict[str, Any]:
        """Row count / dimension summary for a model (ops helper)."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(dim), 0) FROM embedding_cache "
                "WHERE model = ?",
                (model,),
            )
            count, dim = cursor.fetchone()
            return {"model": model, "entries": count, "dim": dim}
        finally:
            conn.close()
