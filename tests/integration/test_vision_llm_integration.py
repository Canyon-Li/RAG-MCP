import pytest
from pathlib import Path
from src.core.settings import load_settings
from src.core.types import Chunk
from src.ingestion.storage.image_storage import ImageStorage
from src.ingestion.transform.image_captioner import ImageCaptioner

@pytest.mark.integration
def test_image_captioner_azure_integration():
    """Integration test for ImageCaptioner using real Azure OpenAI Vision LLM.

    Requires valid credentials in config/settings.yaml (configured by user).
    """
    # 1. Load Settings
    settings = load_settings("config/settings.yaml")

    # Skip if vision not enabled or provider not configured
    if not settings.vision_llm or not settings.vision_llm.enabled:
        pytest.skip("Vision LLM not enabled in settings")

    if settings.vision_llm.provider != "azure":
        pytest.skip("Test specific for Azure provider (as requested)")

    # 2. Check Test Image
    image_path = Path("tests/fixtures/sample_documents/test_vision_llm.jpg")
    if not image_path.exists():
        pytest.fail(f"Test image not found at {image_path}")

    # 3. Set up ImageStorage and register the test image
    image_storage = ImageStorage()
    image_storage.register_image(
        image_id="img_001",
        file_path=str(image_path),
        collection="test",
        doc_hash="integration_test",
        page_num=1,
    )

    # 4. Create Sample Chunk (no images metadata needed — path resolved via Storage)
    chunk = Chunk(
        id="chunk_test_001",
        text="Here is an image: [IMAGE: img_001]",
        metadata={
            "source_path": str(image_path),
        }
    )

    # 5. Initialize ImageCaptioner with image_storage injected
    captioner = ImageCaptioner(settings=settings, image_storage=image_storage)

    # 6. Run Transform
    # This calls the real API
    processed_chunks = captioner.transform([chunk])

    # 7. Verify Results
    assert len(processed_chunks) == 1
    processed_chunk = processed_chunks[0]

    # Check text modification
    print(f"\nOriginal Text: 'Here is an image: [IMAGE: img_001]'")
    print(f"New Text: '{processed_chunk.text}'")

    assert "[IMAGE: img_001]" in processed_chunk.text
    assert "(Description:" in processed_chunk.text

    # Caption is now persisted in ImageStorage, not chunk metadata
    meta = image_storage.get_image_meta("img_001")
    assert meta is not None
    caption = meta["caption"]
    print(f"Generated Caption: {caption}")
    assert caption is not None and len(caption) > 10
