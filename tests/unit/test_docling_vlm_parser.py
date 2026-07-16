"""Unit tests for DoclingVlmParser (C1: local transformers Granite-Docling).

Covers the C1 specifics (model cache pinning, VlmPipeline converter build) and
that section mapping is inherited from DoclingParser unchanged. Real docling
imports run where they don't trigger model downloads.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.libs.parser.docling_parser import DoclingParser
from src.libs.parser.docling_vlm_parser import DoclingVlmParser, DEFAULT_MODEL_CACHE


def _make_item(label, text=None, page=1, table_md=None):
    item = MagicMock()
    lbl = MagicMock()
    lbl.name = label
    item.label = lbl
    item.text = text
    prov = MagicMock()
    prov.page_no = page
    prov.bbox = None
    item.prov = [prov]
    if table_md is not None:
        item.export_to_markdown = MagicMock(return_value=table_md)
    return item


def _make_converter(items):
    ddoc = MagicMock()
    ddoc.iterate_items = MagicMock(return_value=[(it, 0) for it in items])
    result = MagicMock()
    result.document = ddoc
    converter = MagicMock()
    converter.convert.return_value = result
    return converter


@pytest.fixture
def settings():
    return MagicMock()


@pytest.fixture
def fake_pdf(tmp_path):
    p = tmp_path / "fake.pdf"
    p.write_bytes(b"%PDF-1.4 fake content")
    return p


def test_is_docling_parser_subclass(settings):
    p = DoclingVlmParser(
        settings, collection="t",
        image_storage_dir="data/images/t", extract_images=False,
        model_cache_dir="data/_vlmcache_test",
    )
    assert isinstance(p, DoclingParser)


def test_default_model_cache(settings):
    p = DoclingVlmParser(
        settings, collection="t",
        image_storage_dir="data/images/t", extract_images=False,
        model_cache_dir="data/_vlmcache_test",
    )
    assert DEFAULT_MODEL_CACHE == r"D:\Transformers\Model"


def test_model_cache_override(settings, monkeypatch):
    monkeypatch.delenv("DOCLING_VLM_MODEL_DIR", raising=False)
    p = DoclingVlmParser(
        settings, collection="t",
        image_storage_dir="data/images/t", extract_images=False,
        model_cache_dir="/tmp/custom_cache",
    )
    assert p._model_cache_dir == "/tmp/custom_cache"


def test_ensure_local_model_cache_sets_env(settings, tmp_path, monkeypatch):
    cache = tmp_path / "models"
    p = DoclingVlmParser(
        settings, collection="t",
        image_storage_dir="data/images/t", extract_images=False,
        model_cache_dir=str(cache),
    )
    # snapshot env so the test doesn't leak HF_HOME globally
    monkeypatch.setenv("HF_HOME", os.environ.get("HF_HOME", ""))
    monkeypatch.setenv("HF_HUB_CACHE", os.environ.get("HF_HUB_CACHE", ""))
    p._ensure_local_model_cache()
    assert os.environ["HF_HOME"] == str(cache)
    assert os.environ["HF_HUB_CACHE"] == str(cache)
    assert cache.exists()


def test_build_converter_does_not_raise(settings):
    """_build_converter wires VlmPipeline + local Granite-Docling (no download)."""
    p = DoclingVlmParser(
        settings, collection="t",
        image_storage_dir="data/images/t", extract_images=False,
        model_cache_dir="data/_vlmcache_test",
    )
    converter = p._build_converter()  # constructs config object, no model load
    assert converter is not None


def test_parse_inherits_section_mapping(settings, fake_pdf):
    """Section mapping flows through from DoclingParser (table GFM in text+html)."""
    items = [
        _make_item("SECTION_HEADER", text="标题", page=1),
        _make_item("TABLE", page=2, table_md="| a | b |\n|---|---|\n| 1 | 2 |"),
    ]
    converter = _make_converter(items)
    with patch.object(DoclingVlmParser, "_build_converter", return_value=converter), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False):
        parser = DoclingVlmParser(
            settings, collection="t",
            image_storage_dir="data/images/t", extract_images=False,
            model_cache_dir="data/_vlmcache_test",
        )
        doc = parser.parse(fake_pdf)

    types = [s["type"] for s in doc.metadata["sections"]]
    assert "title" in types
    assert "table" in types
    table_sec = next(s for s in doc.metadata["sections"] if s["type"] == "table")
    assert table_sec["html"] == table_sec["text"]  # GFM in both, inherited
