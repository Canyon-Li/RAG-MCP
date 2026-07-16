"""ImageStorage: caption 读写 + schema 升级（G2 地基）。"""
from pathlib import Path
import sqlite3

import pytest

from src.ingestion.storage.image_storage import ImageStorage


@pytest.fixture
def storage(tmp_path):
    return ImageStorage(
        db_path=str(tmp_path / "image_index.db"),
        images_root=str(tmp_path / "images"),
    )


def _register(storage, image_id="img1", page=1):
    img_file = Path(storage.images_root) / f"{image_id}.png"
    img_file.parent.mkdir(parents=True, exist_ok=True)
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n")
    storage.register_image(
        image_id=image_id, file_path=img_file,
        collection="c", doc_hash="h", page_num=page,
    )


def test_set_and_get_caption(storage):
    _register(storage)
    storage.set_caption("img1", "a RAG diagram")
    meta = storage.get_image_meta("img1")
    assert meta is not None
    assert meta["caption"] == "a RAG diagram"
    assert meta["image_id"] == "img1"
    assert meta["page_num"] == 1


def test_set_caption_idempotent(storage):
    _register(storage)
    storage.set_caption("img1", "first")
    storage.set_caption("img1", "second")
    assert storage.get_image_meta("img1")["caption"] == "second"


def test_set_caption_missing_image_is_noop(storage):
    storage.set_caption("ghost", "x")  # 未注册 → UPDATE 命中 0 行，不报错
    assert storage.get_image_meta("ghost") is None


def test_get_image_meta_missing_returns_none(storage):
    assert storage.get_image_meta("nope") is None


def test_get_image_meta_has_all_fields(storage):
    _register(storage, image_id="img2", page=3)
    meta = storage.get_image_meta("img2")
    assert set(meta.keys()) >= {"image_id", "file_path", "collection",
                                "doc_hash", "page_num", "caption"}
    assert meta["caption"] is None  # 未 set_caption


def test_schema_upgrade_adds_caption_column(tmp_path):
    """无 caption 列的老库打开后应自动加列。"""
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE image_index (
        image_id TEXT PRIMARY KEY, file_path TEXT NOT NULL,
        collection TEXT, doc_hash TEXT, page_num INTEGER,
        created_at TEXT NOT NULL
    )""")
    conn.commit()
    conn.close()

    storage = ImageStorage(db_path=str(db_path), images_root=str(tmp_path / "images"))
    _register(storage, image_id="img9")
    storage.set_caption("img9", "after upgrade")
    assert storage.get_image_meta("img9")["caption"] == "after upgrade"
