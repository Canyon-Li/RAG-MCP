"""接线：pipeline 给 ImageCaptioner 注入 image_storage；
查询工具的 ResponseBuilder 持有 image_storage。"""
import pytest

from src.core.response.response_builder import ResponseBuilder
from src.ingestion.storage.image_storage import ImageStorage


@pytest.mark.integration
def test_pipeline_injects_image_storage_to_captioner():
    from src.core.settings import load_settings
    from src.ingestion.pipeline import IngestionPipeline

    settings = load_settings()
    pipeline = IngestionPipeline(settings, collection="test_wiring")
    try:
        assert pipeline.image_captioner is not None
        assert pipeline.image_captioner.image_storage is pipeline.image_storage
    finally:
        pipeline.close()


def test_query_tool_response_builder_has_image_storage():
    from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool

    tool = QueryKnowledgeHubTool()
    assert isinstance(tool._image_storage, ImageStorage)
    asm = tool._response_builder.multimodal_assembler
    assert asm._image_storage is tool._image_storage
