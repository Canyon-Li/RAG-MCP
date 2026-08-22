"""File integrity checker for incremental ingestion.

This module provides SHA256-based file integrity tracking to enable incremental
ingestion. Files that have been successfully processed can be skipped on
subsequent ingestion runs.

D-031 extensions for report version management:
- Records are scoped by ``(file_hash, collection)`` — a file ingested for one
  collection no longer blocks ingesting the same file into another collection.
- Records optionally carry a ``business_key`` (report fingerprint) so a
  re-distributed copy of the SAME logical report (different bytes, e.g.
  platform watermark) can be recognised and the old version superseded.

Design Principles:
- Idempotent: Multiple ingestion runs of the same file are safe
- Persistent: SQLite-backed storage survives process restarts
- Concurrent: WAL mode enables concurrent read/write operations
- Graceful: Failed ingestions are tracked but don't block retries
"""

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class FileIntegrityChecker(ABC):
    """Abstract base class for file integrity checking.

    Implementations track which files have been successfully processed
    to enable incremental ingestion.
    """

    @abstractmethod
    def compute_sha256(self, file_path: str) -> str:
        """Compute SHA256 hash of file.

        Args:
            file_path: Path to the file to hash.

        Returns:
            Hexadecimal SHA256 hash string (64 characters).

        Raises:
            FileNotFoundError: If file does not exist.
            IOError: If path is not a file or cannot be read.
        """
        pass

    @abstractmethod
    def should_skip(self, file_hash: str, collection: Optional[str] = None) -> bool:
        """Check if file should be skipped based on hash.

        Args:
            file_hash: SHA256 hash of the file.
            collection: Optional collection scope. When given, only a
                *success* record in THIS collection skips the file (D-031:
                a file ingested for collection A must not block ingesting
                it into collection B). When *None*, legacy behaviour —
                any success record skips.

        Returns:
            True if file has been successfully processed before, False otherwise.
        """
        pass

    @abstractmethod
    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        collection: Optional[str] = None,
        business_key: Optional[str] = None,
    ) -> None:
        """Mark file as successfully processed.

        Uses upsert semantics — idempotent per (file_hash, collection).

        Args:
            file_hash: SHA256 hash of the file.
            file_path: Original file path (for tracking).
            collection: Optional collection/namespace identifier.
            business_key: Optional report business fingerprint (D-031) —
                identifies the same logical report across re-distributed
                copies with different file bytes.

        Raises:
            RuntimeError: If database operation fails.
        """
        pass

    @abstractmethod
    def mark_failed(
        self,
        file_hash: str,
        file_path: str,
        error_msg: str,
        collection: Optional[str] = None,
    ) -> None:
        """Mark file processing as failed.

        Failed files are tracked but not skipped on subsequent runs,
        allowing retries.

        Args:
            file_hash: SHA256 hash of the file.
            file_path: Original file path.
            error_msg: Error message describing the failure.
            collection: Optional collection scope.

        Raises:
            RuntimeError: If database operation fails.
        """
        pass

    @abstractmethod
    def find_by_business_key(
        self,
        business_key: str,
        collection: str,
        exclude_file_hash: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find active (status='success') records sharing a business key.

        Used to detect a new version of the same logical report arriving
        with different file bytes (e.g. a watermarked re-distribution).

        Args:
            business_key: Report business fingerprint (D-031).
            collection: Collection scope to search.
            exclude_file_hash: Optionally exclude one file hash (typically
                the file currently being processed).

        Returns:
            List of record dicts (file_hash, file_path, collection,
            business_key, processed_at, updated_at).
        """
        pass

    @abstractmethod
    def mark_superseded(self, file_hash: str, collection: Optional[str] = None) -> bool:
        """Mark a record as superseded by a newer version of the report.

        Superseded records no longer cause should_skip and disappear from
        list_processed; their chunks are expected to have been cleaned up by
        the caller (pipeline version-management stage).

        Args:
            file_hash: SHA256 hash identifying the record.
            collection: Optional collection scope.

        Returns:
            True if a record was updated, False if not found.
        """
        pass

    @abstractmethod
    def remove_record(self, file_hash: str, collection: Optional[str] = None) -> bool:
        """Remove ingestion record(s) by file hash.

        Args:
            file_hash: SHA256 hash identifying the record.
            collection: Optional collection scope; when *None* records in
                all collections are removed.

        Returns:
            True if at least one record was deleted, False if not found.
        """
        pass

    @abstractmethod
    def list_processed(
        self, collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List successfully processed files (excludes superseded).

        Args:
            collection: Optional collection filter.  When *None* all
                successful records are returned.

        Returns:
            List of dicts with keys: file_hash, file_path, collection,
            business_key, processed_at, updated_at.
        """
        pass


_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS ingestion_history (
    file_hash TEXT NOT NULL,
    file_path TEXT NOT NULL,
    status TEXT NOT NULL,
    collection TEXT NOT NULL DEFAULT '',
    business_key TEXT,
    error_msg TEXT,
    processed_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (file_hash, collection)
)
"""


def _collection_key(collection: Optional[str]) -> str:
    """Normalise collection for the composite PK (NULL → '')."""
    return collection if collection else ""


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """SQLite-backed file integrity checker.

    Stores ingestion history in a SQLite database with WAL mode for
    concurrent access.

    Database Schema (v2, D-031):
        ingestion_history (
            file_hash TEXT,           -- together with collection: PK
            file_path TEXT NOT NULL,
            status TEXT NOT NULL,     -- 'success' | 'failed' | 'superseded'
            collection TEXT NOT NULL DEFAULT '',
            business_key TEXT,        -- report fingerprint (nullable)
            error_msg TEXT,
            processed_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (file_hash, collection)
        )

    v1 → v2 migration is automatic on first open: v1 keyed records by
    file_hash alone and had no business_key column.

    Args:
        db_path: Path to SQLite database file (will be created if needed).

    Raises:
        sqlite3.DatabaseError: If database file is corrupted.
    """

    def __init__(self, db_path: str):
        """Initialize checker and create database if needed.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._conn = None
        self._ensure_database()

    def close(self) -> None:
        """Close database connection if open."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        """Cleanup: close connection on deletion."""
        self.close()

    def _ensure_database(self) -> None:
        """Create database file and schema if they don't exist; migrate v1."""
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_SCHEMA_V2)

            table_info = list(conn.execute("PRAGMA table_info(ingestion_history)"))
            columns = {row[1] for row in table_info}
            pk_columns = [row[1] for row in table_info if row[5]]

            needs_migration = (
                "business_key" not in columns or pk_columns == ["file_hash"]
            )
            if needs_migration:
                conn.executescript(
                    """
                    CREATE TABLE ingestion_history_migrated (
                        file_hash TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        status TEXT NOT NULL,
                        collection TEXT NOT NULL DEFAULT '',
                        business_key TEXT,
                        error_msg TEXT,
                        processed_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (file_hash, collection)
                    );
                    INSERT OR IGNORE INTO ingestion_history_migrated
                        (file_hash, file_path, status, collection, business_key,
                         error_msg, processed_at, updated_at)
                    SELECT file_hash, file_path, status, COALESCE(collection, ''),
                           NULL, error_msg, processed_at, updated_at
                    FROM ingestion_history;
                    DROP TABLE ingestion_history;
                    ALTER TABLE ingestion_history_migrated RENAME TO ingestion_history;
                    """
                )

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_status "
                "ON ingestion_history(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_business_key "
                "ON ingestion_history(business_key, collection)"
            )
            conn.commit()
        finally:
            conn.close()

    def compute_sha256(self, file_path: str) -> str:
        """Compute SHA256 hash of file using chunked reading.

        Uses 64KB chunks to handle large files without loading entire
        file into memory.

        Args:
            file_path: Path to the file to hash.

        Returns:
            Hexadecimal SHA256 hash string (64 characters).

        Raises:
            FileNotFoundError: If file does not exist.
            IOError: If path is not a file or cannot be read.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise IOError(f"Path is not a file: {file_path}")

        sha256_hash = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(chunk)
        except Exception as e:
            raise IOError(f"Failed to read file {file_path}: {e}")

        return sha256_hash.hexdigest()

    def should_skip(self, file_hash: str, collection: Optional[str] = None) -> bool:
        """Check if file should be skipped.

        Only records with status='success' are skipped. Failed files can be
        retried; superseded records are never "current" for their file bytes.

        Args:
            file_hash: SHA256 hash of the file.
            collection: Optional collection scope (see class docs for the
                D-031 scoping rule).

        Returns:
            True if file has status='success' (in scope), False otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            if collection is not None:
                cursor = conn.execute(
                    "SELECT status FROM ingestion_history "
                    "WHERE file_hash = ? AND collection = ?",
                    (file_hash, _collection_key(collection)),
                )
            else:
                cursor = conn.execute(
                    "SELECT status FROM ingestion_history WHERE file_hash = ?",
                    (file_hash,),
                )
            row = cursor.fetchone()
            return bool(row) and row[0] == "success"
        finally:
            conn.close()

    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        collection: Optional[str] = None,
        business_key: Optional[str] = None,
    ) -> None:
        """Mark file as successfully processed (upsert by hash+collection).

        Args:
            file_hash: SHA256 hash of the file.
            file_path: Original file path (for tracking).
            collection: Optional collection/namespace identifier.
            business_key: Optional report fingerprint (D-031).

        Raises:
            RuntimeError: If database operation fails.
        """
        now = datetime.now(timezone.utc).isoformat()
        ck = _collection_key(collection)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO ingestion_history
                    (file_hash, file_path, status, collection, business_key,
                     error_msg, processed_at, updated_at)
                VALUES (?, ?, 'success', ?, ?, NULL, ?, ?)
                ON CONFLICT(file_hash, collection) DO UPDATE SET
                    file_path = excluded.file_path,
                    status = 'success',
                    business_key = excluded.business_key,
                    error_msg = NULL,
                    updated_at = excluded.updated_at
                """,
                (file_hash, file_path, ck, business_key, now, now),
            )
            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to mark success for {file_path}: {e}")
        finally:
            conn.close()

    def mark_failed(
        self,
        file_hash: str,
        file_path: str,
        error_msg: str,
        collection: Optional[str] = None,
    ) -> None:
        """Mark file processing as failed.

        Failed files are not skipped, allowing retries.

        Args:
            file_hash: SHA256 hash of the file.
            file_path: Original file path.
            error_msg: Error message describing the failure.
            collection: Optional collection scope.

        Raises:
            RuntimeError: If database operation fails.
        """
        now = datetime.now(timezone.utc).isoformat()
        ck = _collection_key(collection)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO ingestion_history
                    (file_hash, file_path, status, collection, business_key,
                     error_msg, processed_at, updated_at)
                VALUES (?, ?, 'failed', ?, NULL, ?, ?, ?)
                ON CONFLICT(file_hash, collection) DO UPDATE SET
                    file_path = excluded.file_path,
                    status = 'failed',
                    error_msg = excluded.error_msg,
                    updated_at = excluded.updated_at
                """,
                (file_hash, file_path, ck, error_msg, now, now),
            )
            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to mark failure for {file_path}: {e}")
        finally:
            conn.close()

    def find_by_business_key(
        self,
        business_key: str,
        collection: str,
        exclude_file_hash: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find active records sharing a business key in a collection.

        Args:
            business_key: Report fingerprint (D-031).
            collection: Collection scope.
            exclude_file_hash: File hash to exclude (the incoming file).

        Returns:
            List of record dicts for status='success' matches.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            query = (
                "SELECT file_hash, file_path, collection, business_key, "
                "processed_at, updated_at FROM ingestion_history "
                "WHERE business_key = ? AND collection = ? AND status = 'success'"
            )
            params: list = [business_key, _collection_key(collection)]
            if exclude_file_hash is not None:
                query += " AND file_hash != ?"
                params.append(exclude_file_hash)
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def mark_superseded(self, file_hash: str, collection: Optional[str] = None) -> bool:
        """Mark a record superseded (status='superseded').

        Args:
            file_hash: SHA256 hash identifying the record.
            collection: Optional collection scope.

        Returns:
            True if a record was updated, False otherwise.
        """
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self.db_path)
        try:
            if collection is not None:
                cursor = conn.execute(
                    "UPDATE ingestion_history SET status = 'superseded', "
                    "updated_at = ? WHERE file_hash = ? AND collection = ?",
                    (now, file_hash, _collection_key(collection)),
                )
            else:
                cursor = conn.execute(
                    "UPDATE ingestion_history SET status = 'superseded', "
                    "updated_at = ? WHERE file_hash = ?",
                    (now, file_hash),
                )
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to mark superseded {file_hash}: {e}")
        finally:
            conn.close()

    def remove_record(self, file_hash: str, collection: Optional[str] = None) -> bool:
        """Remove ingestion record(s) by file hash.

        Args:
            file_hash: SHA256 hash identifying the record.
            collection: Optional collection scope; *None* removes records
                across all collections.

        Returns:
            True if at least one record was deleted, False if not found.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            if collection is not None:
                cursor = conn.execute(
                    "DELETE FROM ingestion_history "
                    "WHERE file_hash = ? AND collection = ?",
                    (file_hash, _collection_key(collection)),
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM ingestion_history WHERE file_hash = ?",
                    (file_hash,),
                )
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to remove record {file_hash}: {e}")
        finally:
            conn.close()

    def list_processed(
        self, collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List successfully processed files (excludes superseded).

        Args:
            collection: Optional collection filter.

        Returns:
            List of dicts with keys: file_hash, file_path, collection,
            business_key, processed_at, updated_at.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            query = (
                "SELECT file_hash, file_path, collection, business_key, "
                "processed_at, updated_at FROM ingestion_history "
                "WHERE status = 'success'"
            )
            params: list = []
            if collection is not None:
                query += " AND collection = ?"
                params.append(_collection_key(collection))
            query += " ORDER BY processed_at ASC"

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
