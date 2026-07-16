"""Docling VLM Parser — VlmPipeline with local Granite-Docling (方案 C1).

A thin subclass of ``DoclingParser`` that swaps the pipeline: each page is
rendered to an image and recognized by a **local** Granite-Docling-258M VLM
(via transformers, CPU), whose DocTags output Docling parses into a
``DoclingDocument``. No inference-time network call — model weights are cached
on disk.

Why local and not Ollama: granite-docling is a completion-style model and
Ollama's chat template mangles its output (only ``<loc_>`` tokens, no
``<doctag>`` structure → empty DoclingDocument). docling's local transformers
mode drives the model with its native tokenizer prompt and works correctly.
See 表格选型.md §方案 C1/C2.

Only ``_build_converter`` is overridden (hook on DoclingParser).

Constructor contract (shared): ``__init__(settings, collection,
image_storage_dir, extract_images, **kwargs)``. HuggingFace cache is pinned to
``D:\\Transformers\\Model`` (override via ``kwargs['model_cache_dir']`` or
``$DOCLING_VLM_MODEL_DIR``) to keep an ASCII-path cache.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.libs.parser.docling_parser import DoclingParser

logger = logging.getLogger(__name__)

DEFAULT_MODEL_CACHE = r"D:\Transformers\Model"


class DoclingVlmParser(DoclingParser):
    """PDF parser using Docling's ``VlmPipeline`` + local Granite-Docling-258M.

    Unlike ``DoclingParser`` (DocLayNet + TableFormer, native models), this
    delegates per-page recognition to a local vision-language model. Stronger
    on scanned/image-like tables and charts; weaker on cross-page tables (VLM
    is page-scoped). See 表格选型.md §方案 C1.
    """

    def __init__(
        self,
        settings: Any = None,
        collection: str = "default",
        image_storage_dir: str | Path = "data/images",
        extract_images: bool = True,
        **kwargs: Any,
    ):
        """Initialize DoclingVlmParser.

        Args:
            settings: Application settings.
            collection: Collection name scoping the image storage directory.
            image_storage_dir: Base dir for extracted images (Factory-resolved).
            extract_images: Whether to extract embedded images via PyMuPDF.
            **kwargs: ``model_cache_dir`` (str) overrides the HF cache location.
        """
        super().__init__(
            settings=settings,
            collection=collection,
            image_storage_dir=image_storage_dir,
            extract_images=extract_images,
            **kwargs,
        )
        self._model_cache_dir = kwargs.get(
            "model_cache_dir",
            os.environ.get("DOCLING_VLM_MODEL_DIR", DEFAULT_MODEL_CACHE),
        )

    def _build_converter(self) -> Any:
        """Build a DocumentConverter backed by VlmPipeline + local Granite-Docling."""
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import (
            DocumentConverter,
            PdfFormatOption,
        )
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        from docling.pipeline.vlm_pipeline import VlmPipeline

        # Pin HF cache to an ASCII path before any model download happens.
        self._ensure_local_model_cache()

        # Default VlmPipelineOptions uses Granite-Docling-258M via local
        # transformers with ResponseFormat.DOCTAGS — exactly what we want.
        pipeline_options = VlmPipelineOptions()
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                ),
            },
        )

    def _ensure_local_model_cache(self) -> None:
        """Pin the HuggingFace cache to ``self._model_cache_dir`` (ASCII path).

        Keeps model weights under ``D:\\Transformers\\Model`` instead of the
        default ``~/.cache/huggingface`` (which on this machine is under a
        non-ASCII user dir). Must run before any model is loaded.
        """
        cache = self._model_cache_dir
        try:
            os.makedirs(cache, exist_ok=True)
        except OSError as e:
            logger.warning(f"Could not create model cache dir {cache}: {e}")
        os.environ["HF_HOME"] = cache
        os.environ["HF_HUB_CACHE"] = cache
        logger.debug(f"HF model cache pinned to {cache}")
