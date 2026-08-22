"""Unit tests for the multi-tenant web service layer (D-032).

Focus: the security- and correctness-critical logic that must hold regardless
of the retrieval backend —
- server-side visible-library resolution (public + own personal; requested
  libs can only narrow)
- public-library write requires admin
- cross-library RRF fusion + content-hash dedup (public wins over personal)
- citation order survives (library tags follow results through formatting)

Retrieval components are monkeypatched with deterministic fakes so the tests
run without Ollama/Chroma.
"""

import sys
import tempfile
import pathlib
from types import SimpleNamespace

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.api import store
from src.api.service import RagService, _ScoredChunk


@pytest.fixture()
def temp_store(tmp_path):
    db = str(tmp_path / "svc.db")
    store.init_store(db_path=db, seed=True)
    # extra: a second user with no personal library seeded
    import sqlite3

    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO users VALUES ('u002','业务员·李四','biz')")
    conn.commit()
    conn.close()
    return db


class TestVisibleLibraries:
    def test_user_sees_public_plus_own_personal(self, temp_store):
        libs = store.resolve_visible_libraries("u001", db_path=temp_store)
        ids = [l["id"] for l in libs]
        assert "public" in ids and "personal_u001" in ids

    def test_cannot_see_others_personal(self, temp_store):
        ids = [l["id"] for l in store.resolve_visible_libraries("u002", db_path=temp_store)]
        assert "personal_u001" not in ids

    def test_requested_can_only_narrow(self, temp_store):
        libs = store.resolve_visible_libraries(
            "u001", requested=["personal_u001", "someone_elses"], db_path=temp_store
        )
        assert [l["id"] for l in libs] == ["personal_u001"]


class TestIngestPermissions:
    def _patched_service(self, temp_store, monkeypatch):
        # bind store helpers to the temp db WITHOUT self-recursive lambdas
        orig_visible = store.resolve_visible_libraries
        orig_user = store.get_user
        orig_personal = store.ensure_personal_library
        monkeypatch.setattr(
            store, "resolve_visible_libraries",
            lambda uid, requested=None, db_path=None: orig_visible(uid, requested, db_path=temp_store),
        )
        monkeypatch.setattr(
            store, "get_user",
            lambda uid, db_path=None: orig_user(uid, db_path=temp_store),
        )
        monkeypatch.setattr(
            store, "ensure_personal_library",
            lambda uid, db_path=None: orig_personal(uid, db_path=temp_store),
        )
        svc = RagService.__new__(RagService)  # skip component init
        svc._tasks = {}
        svc._task_lock = __import__("threading").Lock()
        return svc

    def test_biz_user_cannot_write_public(self, temp_store, monkeypatch):
        svc = self._patched_service(temp_store, monkeypatch)
        out = svc.start_ingest("u001", "x.pdf", library_id="public")
        assert "error" in out and "admin" in out["error"]

    def test_admin_can_write_public(self, temp_store, monkeypatch):
        svc = self._patched_service(temp_store, monkeypatch)
        out = svc.start_ingest("admin001", "x.pdf", library_id="public")
        assert out.get("library") == "public"


def _fake_result(chunk_id: str, score: float = 0.9, text: str = "chunk"):
    return SimpleNamespace(
        chunk_id=chunk_id,
        score=score,
        text=text,
        metadata={"source_path": f"D:/x/{chunk_id.split('_')[0]}_doc.pdf", "page_num": 3, "chunk_index": 1},
    )


PUBLIC_LIB = {"id": "public", "owner_type": "public", "display_name": "公司公有库", "collection": "c_pub"}
PERSONAL_LIB = {"id": "personal_u001", "owner_type": "personal", "display_name": "张三个人库", "collection": "c_me"}


class TestCrossLibraryFusion:
    def _patch_stack(self, monkeypatch, per_collection_results):
        def fake_stack(collection):
            class FakeHybrid:
                def search(self, query, top_k, filters=None, trace=None):
                    class R(list):
                        pass

                    return per_collection_results[collection]

            return FakeHybrid()

        monkeypatch.setattr(store, "resolve_visible_libraries", lambda uid, requested=None, db_path=None: [PUBLIC_LIB, PERSONAL_LIB])
        svc = RagService.__new__(RagService)
        svc.settings = None
        svc._components = {}
        svc._comp_lock = __import__("threading").Lock()
        svc._reranker = SimpleNamespace(is_enabled=False)
        svc._generator = SimpleNamespace(is_enabled=False)
        svc._tasks = {}
        svc._task_lock = __import__("threading").Lock()
        monkeypatch.setattr(svc, "_query_stack", fake_stack)
        return svc

    def test_dedup_public_wins_over_personal_copy(self, monkeypatch):
        # same content hash in both libraries (different source prefixes)
        svc = self._patch_stack(
            monkeypatch,
            {
                "c_pub": [_fake_result("aaaa0000_0001_deadbeef")],
                "c_me": [_fake_result("bbbb1111_0001_deadbeef"), _fake_result("bbbb1111_0002_feedface")],
            },
        )
        out = svc.answer("u001", "q")
        sources = [r["chunk_id"] for r in out["results"]]
        assert len(out["results"]) == 2  # dedup: 3 raw → 2 unique
        assert "aaaa0000_0001_deadbeef" in sources  # public copy kept
        assert "bbbb1111_0001_deadbeef" not in sources  # personal duplicate dropped
        assert out["results"][0]["library"] == "public"  # higher RRF rank

    def test_library_tag_follows_each_result(self, monkeypatch):
        svc = self._patch_stack(
            monkeypatch,
            {
                "c_pub": [_fake_result("aaaa0000_0001_deadbeef")],
                "c_me": [_fake_result("bbbb1111_0002_feedface")],
            },
        )
        out = svc.answer("u001", "q")
        by_lib = {r["library"]: r for r in out["results"]}
        assert by_lib["public"]["library_type"] == "public"
        assert by_lib["personal_u001"]["library_type"] == "personal"
        assert by_lib["public"]["page"] == 3

    def test_broken_library_degrades_gracefully(self, monkeypatch):
        def fake_stack(collection):
            if collection == "c_pub":
                raise RuntimeError("chroma down")

            class FakeHybrid:
                def search(self, query, top_k, filters=None, trace=None):
                    return [_fake_result("bbbb1111_0002_feedface")]

            return FakeHybrid()

        monkeypatch.setattr(store, "resolve_visible_libraries", lambda uid, requested=None, db_path=None: [PUBLIC_LIB, PERSONAL_LIB])
        svc = RagService.__new__(RagService)
        svc.settings = None
        svc._components = {}
        svc._comp_lock = __import__("threading").Lock()
        svc._reranker = SimpleNamespace(is_enabled=False)
        svc._generator = SimpleNamespace(is_enabled=False)
        svc._tasks = {}
        svc._task_lock = __import__("threading").Lock()
        monkeypatch.setattr(svc, "_query_stack", fake_stack)
        out = svc.answer("u001", "q")
        assert out["degraded_libraries"] == ["public"]
        assert len(out["results"]) == 1
