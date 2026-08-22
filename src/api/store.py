"""Users & libraries store for the multi-tenant web service (D-032).

Library model (enterprise-rag-design §4, MVP mapping): each library maps to
one existing storage "collection" (Chroma collection + BM25 index), so the
whole retrieval stack is reused unchanged. `libraries.collection` decouples
the logical library id from the physical collection name, which keeps the
future Milvus/partition migration a one-column change.

Permission rule (design §4.2): the SERVER resolves a user's visible library
set — every public library + the user's own personal library. Clients can
narrow, never broaden: any requested library is intersected with this set.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import resolve_path

DB_PATH = str(resolve_path("data/db/rag_service.db"))


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_store(db_path: str = DB_PATH, seed: bool = True) -> None:
    """Create schema (idempotent) and optionally seed demo users/libraries."""
    conn = _connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'biz'   -- 'biz' | 'admin'
            );
            CREATE TABLE IF NOT EXISTS libraries (
                id TEXT PRIMARY KEY,               -- logical library id
                owner_type TEXT NOT NULL,          -- 'public' | 'personal' | 'team'
                owner_id TEXT,                     -- user id for personal libs
                display_name TEXT NOT NULL,
                collection TEXT NOT NULL           -- physical Chroma/BM25 collection
            );
            """
        )
        if seed:
            conn.executescript(
                """
                INSERT OR IGNORE INTO users (id, name, role) VALUES
                    ('u001', '业务员·张三', 'biz'),
                    ('admin001', '知识库管理员', 'admin');
                INSERT OR IGNORE INTO libraries
                    (id, owner_type, owner_id, display_name, collection) VALUES
                    ('public', 'public', NULL, '公司公有库', 'zh_eval_bge_m3'),
                    ('personal_u001', 'personal', 'u001',
                     '张三的个人库', 'personal_u001');
                """
            )
        conn.commit()
    finally:
        conn.close()


def get_user(user_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT id, name, role FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_users(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    try:
        return [dict(r) for r in conn.execute(
            "SELECT id, name, role FROM users ORDER BY id"
        ).fetchall()]
    finally:
        conn.close()


def personal_library_id(user_id: str) -> str:
    return f"personal_{user_id}"


def resolve_visible_libraries(
    user_id: str,
    requested: Optional[List[str]] = None,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    """Server-side ACL: public libs + own personal lib (∩ requested, if given).

    Returns list of {id, owner_type, display_name, collection} rows.
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, owner_type, display_name, collection FROM libraries "
            "WHERE owner_type = 'public' OR (owner_type = 'personal' AND owner_id = ?)"
            " ORDER BY owner_type DESC, id",
            (user_id,),
        ).fetchall()
    finally:
        conn.close()

    visible = [dict(r) for r in rows]
    if requested:
        allowed = {lib["id"] for lib in visible}
        visible = [lib for lib in visible if lib["id"] in set(requested)]
    return visible


def ensure_personal_library(user_id: str, db_path: str = DB_PATH) -> Dict[str, Any]:
    """Create the user's personal library row if missing; return it."""
    lib_id = personal_library_id(user_id)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT id, owner_type, display_name, collection FROM libraries WHERE id = ?",
            (lib_id,),
        ).fetchone()
        if row:
            return dict(row)
        conn.execute(
            "INSERT INTO libraries (id, owner_type, owner_id, display_name, collection) "
            "VALUES (?, 'personal', ?, ?, ?)",
            (lib_id, user_id, f"{user_id} 的个人库", lib_id),
        )
        conn.commit()
        return {
            "id": lib_id,
            "owner_type": "personal",
            "display_name": f"{user_id} 的个人库",
            "collection": lib_id,
        }
    finally:
        conn.close()
