"""Factory for creating document Loader instances.

This module implements the Factory Pattern to instantiate the appropriate
document loader based on configuration, enabling configuration-driven selection
of different file formats (PDF, DOCX, ...) without code changes — consistent
with the other pluggable backends in ``src/libs/`` (LLM/Embedding/Splitter/
VectorStore/Reranker/Evaluator).

J1: previously the IngestionPipeline hardcoded ``PdfLoader``; this factory
makes the loader pluggable so that "add a new format = implement BaseLoader +
register + edit settings.yaml" holds for the loader layer too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.loader.base_loader import BaseLoader

if TYPE_CHECKING:
    from src.core.settings import Settings


class LoaderFactory:
    """Factory for creating document Loader instances.

    Reads the loader provider from ``settings.ingestion.loader`` and
    instantiates the corresponding ``BaseLoader`` implementation. The image
    storage directory is derived from the target collection:
    ``data/images/{collection}/``.

    Design Principles Applied:
    - Factory Pattern: Centralizes loader creation logic.
    - Config-Driven: Provider selection based on settings.yaml.
    - Fail-Fast: Raises clear errors for unknown providers.
    - Backwards-Compatible: Defaults to the ``pdf`` provider (with image
      extraction enabled) when ``ingestion.loader`` is absent.
    """

    # Registry of supported providers
    _PROVIDERS: dict[str, type[BaseLoader]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseLoader]) -> None:
        """Register a new document loader implementation.

        Args:
            name: The provider identifier (e.g., 'pdf', 'docx').
            provider_class: The BaseLoader subclass implementing the provider.

        Raises:
            ValueError: If provider_class doesn't inherit from BaseLoader.
        """
        if not issubclass(provider_class, BaseLoader):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseLoader"
            )
        cls._PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create(cls, settings: "Settings", collection: str = "default") -> BaseLoader:
        """Create a Loader instance based on configuration.

        Args:
            settings: The application settings (reads ``ingestion.loader``).
            collection: Collection name, used to scope the image storage dir
                to ``data/images/{collection}/``.

        Returns:
            An instance of the configured Loader provider.

        Raises:
            ValueError: If the configured provider is not supported.
            RuntimeError: If the provider fails to instantiate.

        Example:
            >>> settings = load_settings('config/settings.yaml')
            >>> loader = LoaderFactory.create(settings, collection='contracts')
            >>> document = loader.load('documents/report.pdf')
        """
        # Resolve provider name + image flag, defaulting to pdf when unset
        # (backwards compatibility for configs predating the loader block).
        loader_cfg = getattr(getattr(settings, "ingestion", None), "loader", None)
        if loader_cfg is not None:
            try:
                provider_name = loader_cfg.provider.lower()
            except AttributeError as e:
                raise ValueError(
                    "Missing required configuration: settings.ingestion.loader.provider. "
                    "Please ensure 'ingestion.loader.provider' is specified in settings.yaml"
                ) from e
            extract_images = loader_cfg.extract_images
        else:
            provider_name = "pdf"
            extract_images = True

        # Look up provider class in registry
        provider_class = cls._PROVIDERS.get(provider_name)

        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Loader provider: '{provider_name}'. "
                f"Available providers: {available}"
            )

        # Instantiate the loader. Loaders follow the shared constructor contract
        # __init__(extract_images: bool, image_storage_dir: str | Path).
        # resolve_path is imported lazily to keep libs free of a runtime
        # core dependency at import time (mirrors the other factories).
        from src.core.settings import resolve_path

        image_storage_dir = str(resolve_path(f"data/images/{collection}"))
        try:
            return provider_class(
                extract_images=extract_images,
                image_storage_dir=image_storage_dir,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Loader provider '{provider_name}': {e}"
            ) from e

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names.

        Returns:
            Sorted list of available provider identifiers.
        """
        return sorted(cls._PROVIDERS.keys())


# Auto-register providers on module import
def _register_builtin_providers() -> None:
    """Register built-in document loaders with the factory."""
    try:
        from src.libs.loader.pdf_loader import PdfLoader

        LoaderFactory.register_provider("pdf", PdfLoader)
    except ImportError:
        pass  # PdfLoader (or its deps) not available


# Register providers when module is imported
_register_builtin_providers()
