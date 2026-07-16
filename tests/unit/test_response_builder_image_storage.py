"""ResponseBuilder: 把 image_storage 透传给 multimodal_assembler。"""
from src.core.response.response_builder import ResponseBuilder


class _FakeStorage:
    def get_image_meta(self, image_id):
        return None

    def get_image_path(self, image_id):
        return None


def test_image_storage_reaches_assembler():
    storage = _FakeStorage()
    rb = ResponseBuilder(image_storage=storage)
    asm = rb.multimodal_assembler
    assert asm._image_storage is storage


def test_default_no_storage():
    rb = ResponseBuilder()
    asm = rb.multimodal_assembler
    assert asm._image_storage is None
