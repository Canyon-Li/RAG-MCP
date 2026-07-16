"""ImageCaptioner: caption 写 ImageStorage，不再写 chunk metadata（G2）。"""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.types import Chunk
from src.ingestion.storage.image_storage import ImageStorage
from src.ingestion.transform.image_captioner import ImageCaptioner


@dataclass
class _FakeVisionResponse:
    content: str


class _FakeVisionLLM:
    def __init__(self, caption_text):
        self._caption = caption_text
        self.calls = 0

    def chat_with_image(self, text, image, trace=None):
        self.calls += 1
        return _FakeVisionResponse(content=self._caption)


def _settings_with_vision_enabled():
    return SimpleNamespace(vision_llm=SimpleNamespace(enabled=True))


@pytest.fixture
def storage(tmp_path):
    return ImageStorage(
        db_path=str(tmp_path / "idx.db"),
        images_root=str(tmp_path / "images"),
    )


def _chunk_with_image(storage, image_id="img1"):
    img = Path(storage.images_root) / f"{image_id}.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    storage.register_image(
        image_id=image_id, file_path=img, collection="c",
        doc_hash="h", page_num=1,
    )
    return Chunk(
        id="chunk1",
        text=f"正文 [IMAGE: {image_id}] 结束",
        metadata={"source_path": "test.pdf", "chunk_index": 0},
    )


def test_caption_written_to_storage(storage):
    chunk = _chunk_with_image(storage, "img1")
    captioner = ImageCaptioner(
        settings=_settings_with_vision_enabled(),
        llm=_FakeVisionLLM("图说：架构图"),
        image_storage=storage,
    )
    [out] = captioner.transform([chunk])

    assert storage.get_image_meta("img1")["caption"] == "图说：架构图"
    assert "image_captions" not in out.metadata
    assert "(Description: 图说：架构图)" in out.text


def test_no_storage_no_captioning(storage):
    """未注入 image_storage：拿不到图片路径，不生成 caption，不崩。"""
    chunk = _chunk_with_image(storage, "img2")
    fake_llm = _FakeVisionLLM("x")
    captioner = ImageCaptioner(
        settings=_settings_with_vision_enabled(),
        llm=fake_llm,
        image_storage=None,
    )
    [out] = captioner.transform([chunk])
    assert "(Description:" not in out.text
    assert fake_llm.calls == 0


def test_text_without_placeholder_unchanged(storage):
    chunk = Chunk(id="c0", text="普通文本，无图",
                  metadata={"source_path": "t.pdf", "chunk_index": 0})
    captioner = ImageCaptioner(
        settings=_settings_with_vision_enabled(),
        llm=_FakeVisionLLM("不应该被调用"),
        image_storage=storage,
    )
    [out] = captioner.transform([chunk])
    assert out.text == "普通文本，无图"
