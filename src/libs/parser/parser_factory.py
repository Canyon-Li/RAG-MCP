"""Factory for creating document Parser instances.

This module implements the Factory Pattern to instantiate the appropriate
document parser based on configuration, enabling configuration-driven selection
of different file formats (PDF, DOCX, ...) without code changes — consistent
with the other pluggable backends in ``src/libs/`` (LLM/Embedding/Splitter/
VectorStore/Reranker/Evaluator).

K1 (DEV_SPEC phase K): renamed from LoaderFactory → ParserFactory. The
constructor contract is unified: providers receive
``(settings, collection, image_storage_dir, extract_images=...)`` so the factory
builds any parser the same way. Backwards-compatible: when ``ingestion.parser`` is
absent it falls back to the legacy ``ingestion.loader`` block, then to the
``pdf`` provider default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.parser.base_parser import BaseParser

if TYPE_CHECKING:
    from src.core.settings import Settings


class ParserFactory:
    """Factory for creating document Parser instances.

    Reads the parser provider from ``settings.ingestion.parser`` and
    instantiates the corresponding ``BaseParser`` implementation. The image
    storage directory is derived from the target collection:
    ``data/images/{collection}/``.

    Design Principles Applied:
    - Factory Pattern: Centralizes parser creation logic.
    - Config-Driven: Provider selection based on settings.yaml.
    - Fail-Fast: Raises clear errors for unknown providers.
    - Backwards-Compatible: Defaults to the ``pdf`` provider (with image
      extraction enabled) when ``ingestion.parser`` is absent; also accepts
      the legacy ``ingestion.loader`` block.
    """

    # Registry of supported providers
    _PROVIDERS: dict[str, type[BaseParser]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseParser]) -> None:
        """Register a new document parser implementation.

        Args:
            name: The provider identifier (e.g. 'pdf', 'docx').
            provider_class: The BaseParser subclass implementing the provider.

        Raises:
            ValueError: If provider_class doesn't inherit from BaseParser.
        """
        if not issubclass(provider_class, BaseParser):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseParser"
            )
        cls._PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create(cls, settings: "Settings", collection: str = "default") -> BaseParser:
        """Create a Parser instance based on configuration.

        Args:
            settings: The application settings (reads ``ingestion.parser``).
            collection: Collection name, used to scope the image storage dir
                to ``data/images/{collection}/``.

        Returns:
            An instance of the configured Parser provider.

        Raises:
            ValueError: If the configured provider is not supported.
            RuntimeError: If the provider fails to instantiate.
        """
        ingestion = getattr(settings, "ingestion", None)
        parser_cfg = getattr(ingestion, "parser", None)
        # Legacy fallback: accept ingestion.loader (deprecated, pre-K1 configs).
        legacy_loader_cfg = getattr(ingestion, "loader", None)

        if parser_cfg is not None:
            try:
                provider_name = parser_cfg.provider.lower()
            except AttributeError as e:
                raise ValueError(
                    "Missing required configuration: settings.ingestion.parser.provider. "
                    "Please ensure 'ingestion.parser.provider' is specified in settings.yaml"
                ) from e
            extract_images = parser_cfg.extract_images
        elif legacy_loader_cfg is not None:
            provider_name = legacy_loader_cfg.provider.lower()
            extract_images = legacy_loader_cfg.extract_images
        else:
            provider_name = "pdf"
            extract_images = True

        # Look up provider class in registry
        provider_class = cls._PROVIDERS.get(provider_name)

        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Parser provider: '{provider_name}'. "
                f"Available providers: {available}"
            )

        # Instantiate the parser. Providers follow the shared constructor
        # contract __init__(settings, collection, image_storage_dir, **kwargs);
        # extract_images is forwarded via kwargs.
        # resolve_path is imported lazily to keep libs free of a runtime
        # core dependency at import time (mirrors the other factories).
        from src.core.settings import resolve_path

        image_storage_dir = str(resolve_path(f"data/images/{collection}"))
        try:
            return provider_class(
                settings=settings,
                collection=collection,
                image_storage_dir=image_storage_dir,
                extract_images=extract_images,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Parser provider '{provider_name}': {e}"
            ) from e

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names (sorted)."""
        return sorted(cls._PROVIDERS.keys())

    # Provider -> file extensions it accepts. Kept in sync with the providers
    # registered in _register_builtin_providers.
    _PROVIDER_EXTENSIONS: dict[str, list[str]] = {
        "pdf": [".pdf"],
        "pdf_text": [".pdf"],  # K3: alias of pdf
        "pdf_table": [".pdf"],  # K4′: native table extraction (pdfplumber)
        "docling": [".pdf"],  # Docling (DocLayNet + TableFormer) layout + table
        "docling_vlm": [".pdf"],  # Docling VlmPipeline + Ollama Granite-Docling (C2)
        "docx": [".docx"],
    }

    @classmethod
    def get_supported_extensions(cls, provider: str) -> list[str]:
        """Return the file extensions a parser provider accepts.

        Used by the CLI (ingest.py) so file discovery matches the configured
        parser — e.g. provider=docx discovers .docx, not .pdf.

        Args:
            provider: Provider name (e.g. 'pdf', 'docx').

        Returns:
            List of lowercase dot-prefixed extensions. Defaults to ``['.pdf']``
            for unknown providers.
        """
        return cls._PROVIDER_EXTENSIONS.get(provider.lower(), [".pdf"])


# Auto-register providers on module import
def _register_builtin_providers() -> None:
    """Register built-in document parsers with the factory."""
    try:
        from src.libs.parser.pdf_text_parser import PdfTextParser

        ParserFactory.register_provider("pdf", PdfTextParser)
        # K3: pdf_text is an explicit alias for pdf (both → PdfTextParser),
        # the text-only tier of the PDF degradation chain (pdf_deep → pdf_text).
        ParserFactory.register_provider("pdf_text", PdfTextParser)
    except ImportError:
        pass  # PdfTextParser (or its deps) not available

    try:
        from src.libs.parser.word_parser import WordParser

        ParserFactory.register_provider("docx", WordParser)
    except ImportError:
        pass  # WordParser (or its deps) not available

    try:
        from src.libs.parser.pdf_table_parser import PdfTableParser

        ParserFactory.register_provider("pdf_table", PdfTableParser)
    except ImportError:
        pass  # PdfTableParser (or pdfplumber) not available

    try:
        from src.libs.parser.docling_parser import DoclingParser

        ParserFactory.register_provider("docling", DoclingParser)
    except ImportError:
        pass  # DoclingParser (or docling) not available

    try:
        from src.libs.parser.docling_vlm_parser import DoclingVlmParser

        ParserFactory.register_provider("docling_vlm", DoclingVlmParser)
    except ImportError:
        pass  # DoclingVlmParser (or docling) not available


# Register providers when module is imported
_register_builtin_providers()
