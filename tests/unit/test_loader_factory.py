"""Unit tests for Loader Factory and BaseLoader contract.

Test Coverage (mirrors tests/unit/test_embedding_factory.py):
- Factory pattern: provider registration, creation, and routing
- Configuration-driven instantiation (ingestion.loader)
- Backwards-compatible default to 'pdf' when loader config is absent
- Error handling for unknown/missing providers and instantiation failures
- Validation logic in BaseLoader (_validate_file)

Registry state is saved/restored around each test so the shared class-level
``_PROVIDERS`` dict is not polluted for downstream integration tests that rely
on the built-in 'pdf' provider being registered.
"""

from pathlib import Path
from typing import Any, List, Optional, Union
from unittest.mock import MagicMock

import pytest

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.loader_factory import LoaderFactory


class FakeLoader(BaseLoader):
    """Fake loader provider for testing.

    Records the constructor arguments so tests can assert factory wiring
    without touching real parsing libraries (MarkItDown / PyMuPDF).
    """

    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: Any = "data/images",
        **kwargs: Any,
    ) -> None:
        self.extract_images = extract_images
        self.image_storage_dir = image_storage_dir
        self.init_kwargs = kwargs

    def load(self, file_path: Union[str, Path]) -> Document:  # type: ignore[override]
        return Document(
            id="doc_fake",
            text="fake content",
            metadata={"source_path": str(file_path)},
        )


class TestBaseLoaderContract:
    """Tests for BaseLoader abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """BaseLoader cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseLoader()

    def test_validate_file_existing_file(self, tmp_path):
        """_validate_file returns Path for existing files."""
        test_file = tmp_path / "test.docx"
        test_file.write_text("dummy")

        validated = FakeLoader._validate_file(test_file)
        assert validated.exists()
        assert validated.is_file()

    def test_validate_file_nonexistent(self):
        """_validate_file raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            FakeLoader._validate_file("nonexistent_file.docx")

    def test_validate_file_directory(self, tmp_path):
        """_validate_file raises ValueError for directories."""
        with pytest.raises(ValueError, match="not a file"):
            FakeLoader._validate_file(tmp_path)


class TestLoaderFactoryRegistration:
    """Tests for LoaderFactory.register_provider / list_providers."""

    def setup_method(self):
        """Snapshot the registry; run each test against a clean copy."""
        self._saved = LoaderFactory._PROVIDERS.copy()
        LoaderFactory._PROVIDERS.clear()

    def teardown_method(self):
        """Restore the registry so downstream tests still see built-ins."""
        LoaderFactory._PROVIDERS.clear()
        LoaderFactory._PROVIDERS.update(self._saved)

    def test_register_provider_success(self):
        """Registering a valid provider should succeed."""
        LoaderFactory.register_provider("fake", FakeLoader)
        assert "fake" in LoaderFactory._PROVIDERS
        assert LoaderFactory._PROVIDERS["fake"] == FakeLoader

    def test_register_provider_case_insensitive(self):
        """Provider names should be normalized to lowercase."""
        LoaderFactory.register_provider("DOCX", FakeLoader)
        assert "docx" in LoaderFactory._PROVIDERS

    def test_register_provider_invalid_class(self):
        """Registering a non-BaseLoader class should raise ValueError."""

        class NotALoader:
            pass

        with pytest.raises(ValueError, match="must inherit from BaseLoader"):
            LoaderFactory.register_provider("invalid", NotALoader)  # type: ignore

    def test_list_providers_empty(self):
        """list_providers should return [] when registry is empty."""
        assert LoaderFactory.list_providers() == []

    def test_list_providers_sorted(self):
        """list_providers should return sorted provider names."""
        LoaderFactory.register_provider("zebra", FakeLoader)
        LoaderFactory.register_provider("alpha", FakeLoader)
        LoaderFactory.register_provider("beta", FakeLoader)

        assert LoaderFactory.list_providers() == ["alpha", "beta", "zebra"]


class TestLoaderFactoryCreate:
    """Tests for LoaderFactory.create routing & instantiation."""

    def setup_method(self):
        self._saved = LoaderFactory._PROVIDERS.copy()
        LoaderFactory._PROVIDERS.clear()

    def teardown_method(self):
        LoaderFactory._PROVIDERS.clear()
        LoaderFactory._PROVIDERS.update(self._saved)

    def _settings(self, provider: str, extract_images: bool = True) -> MagicMock:
        """Build a mock Settings with ingestion.loader configured."""
        settings = MagicMock()
        settings.ingestion.loader.provider = provider
        settings.ingestion.loader.extract_images = extract_images
        return settings

    def test_create_success(self):
        """Creating a registered provider should return its instance."""
        LoaderFactory.register_provider("fake", FakeLoader)

        loader = LoaderFactory.create(self._settings("fake"), collection="contracts")

        assert isinstance(loader, FakeLoader)
        assert loader.extract_images is True
        # image_storage_dir is derived from collection
        assert "contracts" in str(loader.image_storage_dir)

    def test_create_case_insensitive(self):
        """Provider lookup should be case-insensitive."""
        LoaderFactory.register_provider("fake", FakeLoader)

        loader = LoaderFactory.create(self._settings("FAKE"), collection="c")
        assert isinstance(loader, FakeLoader)

    def test_create_passes_extract_images_flag(self):
        """Factory should forward the extract_images flag to the loader."""
        LoaderFactory.register_provider("fake", FakeLoader)

        loader = LoaderFactory.create(self._settings("fake", extract_images=False), collection="c")
        assert loader.extract_images is False

    def test_create_unknown_provider(self):
        """An unregistered provider should raise a clear ValueError."""
        LoaderFactory.register_provider("fake", FakeLoader)

        with pytest.raises(ValueError) as exc_info:
            LoaderFactory.create(self._settings("unknown"), collection="c")

        msg = str(exc_info.value)
        assert "Unsupported Loader provider: 'unknown'" in msg
        assert "Available providers:" in msg

    def test_create_no_providers_registered(self):
        """Empty registry should surface 'none' in the available list."""
        with pytest.raises(ValueError) as exc_info:
            LoaderFactory.create(self._settings("pdf"), collection="c")

        assert "Unsupported Loader provider: 'pdf'" in str(exc_info.value)
        assert "Available providers: none" in str(exc_info.value)

    def test_create_provider_instantiation_failure(self):
        """Provider constructor errors should be wrapped in RuntimeError."""

        class BrokenLoader(BaseLoader):
            def __init__(self, **kwargs: Any) -> None:
                raise ValueError("Intentional init error")

            def load(self, file_path: Union[str, Path]) -> Document:  # type: ignore[override]
                return Document(id="x", text="", metadata={"source_path": str(file_path)})

        LoaderFactory.register_provider("broken", BrokenLoader)

        with pytest.raises(RuntimeError) as exc_info:
            LoaderFactory.create(self._settings("broken"), collection="c")

        msg = str(exc_info.value)
        assert "Failed to instantiate Loader provider 'broken'" in msg
        assert "Intentional init error" in msg

    def test_create_defaults_to_pdf_when_loader_config_absent(self):
        """Missing ingestion.loader should default to the 'pdf' provider."""
        LoaderFactory.register_provider("pdf", FakeLoader)  # stand-in for PdfLoader

        settings = MagicMock()
        settings.ingestion.loader = None  # no loader block in config

        loader = LoaderFactory.create(settings, collection="c")
        assert isinstance(loader, FakeLoader)
        assert loader.extract_images is True  # default

    def test_create_defaults_to_pdf_when_ingestion_absent(self):
        """Missing ingestion entirely should also default to 'pdf'."""
        LoaderFactory.register_provider("pdf", FakeLoader)

        settings = MagicMock()
        settings.ingestion = None

        loader = LoaderFactory.create(settings, collection="c")
        assert isinstance(loader, FakeLoader)
