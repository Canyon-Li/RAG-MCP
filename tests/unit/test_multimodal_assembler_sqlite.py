"""MultimodalAssembler: 正文占位符 + ImageStorage 还原图片（G1）。"""
import pytest
from mcp import types

from src.core.response.multimodal_assembler import MultimodalAssembler
from src.core.types import RetrievalResult
from src.ingestion.storage.image_storage import ImageStorage


@pytest.fixture
def storage_with_image(tmp_path):
    storage = ImageStorage(
        db_path=str(tmp_path / "idx.db"),
        images_root=str(tmp_path / "images"),
    )
    img = tmp_path / "images" / "img1.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    storage.register_image(
        image_id="img1", file_path=img, collection="c",
        doc_hash="h", page_num=2,
    )
    storage.set_caption("img1", "架构图")
    return storage


def _result(text):
    return RetrievalResult(
        chunk_id="c1", score=0.9, text=text, metadata={"source_path": "x.pdf"}
    )


def test_assemble_produces_image_content(storage_with_image):
    asm = MultimodalAssembler(image_storage=storage_with_image)
    blocks = asm.assemble([_result("正文 [IMAGE: img1] 结束")], collection="c")
    image_blocks = [b for b in blocks if isinstance(b, types.ImageContent)]
    assert len(image_blocks) == 1
    assert image_blocks[0].mimeType == "image/png"
    assert len(image_blocks[0].data) > 0


def test_caption_emitted_as_text_block(storage_with_image):
    asm = MultimodalAssembler(image_storage=storage_with_image)
    blocks = asm.assemble([_result("[IMAGE: img1]")], collection="c")
    text_blocks = [b for b in blocks if isinstance(b, types.TextContent)]
    assert any("架构图" in b.text for b in text_blocks)


def test_missing_image_skipped(storage_with_image):
    asm = MultimodalAssembler(image_storage=storage_with_image)
    blocks = asm.assemble([_result("[IMAGE: ghost]")], collection="c")
    assert not any(isinstance(b, types.ImageContent) for b in blocks)


def test_no_storage_returns_no_images():
    asm = MultimodalAssembler(image_storage=None)
    blocks = asm.assemble([_result("[IMAGE: img1]")], collection="c")
    assert not any(isinstance(b, types.ImageContent) for b in blocks)
