"""D-031 tests: collection scoping, business_key version management, migration."""

import sqlite3
import sys
import tempfile
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.libs.loader.file_integrity import SQLiteIntegrityChecker


@pytest.fixture
def db_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(pathlib.Path(tmpdir) / "test.db")


@pytest.fixture
def checker(db_path):
    return SQLiteIntegrityChecker(db_path=db_path)


class TestCollectionScoping:
    def test_ingesting_for_collection_a_does_not_block_collection_b(self, checker):
        """The exact bug hit in the wild (D-030 实锤): 研报 already ingested
        for knowledge_hub was skipped entirely when ingesting zh_eval."""
        checker.mark_success("h" * 64, "a.pdf", collection="A")
        assert checker.should_skip("h" * 64, collection="A") is True
        assert checker.should_skip("h" * 64, collection="B") is False

    def test_legacy_unscoped_should_skip_still_works(self, checker):
        checker.mark_success("h" * 64, "a.pdf", collection="A")
        assert checker.should_skip("h" * 64) is True

    def test_same_file_two_collections_two_records(self, checker):
        checker.mark_success("h" * 64, "a.pdf", collection="A")
        checker.mark_success("h" * 64, "a.pdf", collection="B")
        rows = checker.list_processed()
        assert len(rows) == 2


class TestBusinessKeyVersioning:
    def test_find_by_business_key(self, checker):
        checker.mark_success(
            "a" * 64, "原版.pdf", collection="zh",
            business_key="bk123",
        )
        hits = checker.find_by_business_key("bk123", "zh", exclude_file_hash="b" * 64)
        assert len(hits) == 1
        assert hits[0]["file_path"] == "原版.pdf"
        # excluding itself → empty
        assert checker.find_by_business_key("bk123", "zh", exclude_file_hash="a" * 64) == []

    def test_business_key_scoped_to_collection(self, checker):
        checker.mark_success("a" * 64, "a.pdf", collection="lib1", business_key="bk")
        assert checker.find_by_business_key("bk", "lib2") == []

    def test_mark_superseded_stops_skip_and_hides_from_list(self, checker):
        checker.mark_success("a" * 64, "old.pdf", collection="zh", business_key="bk")
        assert checker.mark_superseded("a" * 64, collection="zh") is True
        assert checker.should_skip("a" * 64, collection="zh") is False
        assert checker.list_processed(collection="zh") == []
        # superseded is not "active" for business-key lookup either
        assert checker.find_by_business_key("bk", "zh") == []


class TestV1Migration:
    def test_v1_database_migrated_on_open(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = str(pathlib.Path(tmpdir) / "v1.db")
            conn = sqlite3.connect(db)
            conn.execute(
                """CREATE TABLE ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    collection TEXT,
                    error_msg TEXT,
                    processed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                "INSERT INTO ingestion_history VALUES (?,?,?,?,?,?,?)",
                ("f" * 64, "old.pdf", "success", None, None, "2026-01-01", "2026-01-01"),
            )
            conn.commit()
            conn.close()

            checker = SQLiteIntegrityChecker(db_path=db)

            # v1 row preserved, still skips (legacy unscoped)
            assert checker.should_skip("f" * 64) is True
            # business_key column exists and is queryable
            checker.mark_success("f" * 64, "old.pdf", collection="c", business_key="k")
            assert checker.find_by_business_key("k", "c")
