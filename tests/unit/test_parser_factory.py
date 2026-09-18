"""Unit tests for Parser Factory and BaseParser contract.

Test Coverage (mirrors tests/unit/test_embedding_factory.py):
- Factory pattern: provider registration, creation, and routing
- Configuration-driven instantiation (ingestion.parser)
- Backwards-compatible default to 'pdf' when parser config is absent
- K1 legacy fallback: ingestion.loader (deprecated) still routes correctly
- Error handling for unknown/missing providers and instantiation failures
- Validation logic in BaseParser (_validate_file)

K1 (DEV_SPEC phase K): renamed from test_loader_factory.py. Constructor
contract unified to (settings, collection, image_storage_dir, extract_images=...).

Registry state is saved/restored around each test so the shared class-level
``_PROVIDERS`` dict is not polluted for downstream integration tests that rely
on the built-in 'pdf' provider being registered.
"""

from pathlib import Path
from typing import Any, Union
from unittest.mock import MagicMock

import pytest

from src.core.settings import ParseCacheSettings, REPO_ROOT
from src.core.types import Document
from src.libs.parser.base_parser import BaseParser
from src.libs.parser.parser_factory import ParserFactory


class FakeParser(BaseParser):
    """Fake parser provider for testing.

    Records the constructor arguments so tests can assert factory wiring
    without touching real parsing libraries (MarkItDown / PyMuPDF).
    """

    def __init__(
        self,
        settings: Any = None,
        collection: str = "default",
        image_storage_dir: Any = "data/images",
        extract_images: bool = True,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.collection = collection
        self.extract_images = extract_images
        self.image_storage_dir = image_storage_dir
        self.init_kwargs = kwargs

    def parse(self, file_path: Union[str, Path]) -> Document:  # type: ignore[override]
        return Document(
            id="doc_fake",
            text="fake content",
            metadata={"source_path": str(file_path)},
        )


class TestBaseParserContract:
    """Tests for BaseParser abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """BaseParser cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseParser()

    def test_validate_file_existing_file(self, tmp_path):
        """_validate_file returns Path for existing files."""
        test_file = tmp_path / "test.docx"
        test_file.write_text("dummy")

        validated = FakeParser._validate_file(test_file)
        assert validated.exists()
        assert validated.is_file()

    def test_validate_file_nonexistent(self):
        """_validate_file raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            FakeParser._validate_file("nonexistent_file.docx")

    def test_validate_file_directory(self, tmp_path):
        """_validate_file raises ValueError for directories."""
        with pytest.raises(ValueError, match="not a file"):
            FakeParser._validate_file(tmp_path)


class TestParserFactoryRegistration:
    """Tests for ParserFactory.register_provider / list_providers."""

    def setup_method(self):
        """Snapshot the registry; run each test against a clean copy."""
        self._saved = ParserFactory._PROVIDERS.copy()
        ParserFactory._PROVIDERS.clear()

    def teardown_method(self):
        """Restore the registry so downstream tests still see built-ins."""
        ParserFactory._PROVIDERS.clear()
        ParserFactory._PROVIDERS.update(self._saved)

    def test_register_provider_success(self):
        """Registering a valid provider should succeed."""
        ParserFactory.register_provider("fake", FakeParser)
        assert "fake" in ParserFactory._PROVIDERS
        assert ParserFactory._PROVIDERS["fake"] == FakeParser

    def test_register_provider_case_insensitive(self):
        """Provider names should be normalized to lowercase."""
        ParserFactory.register_provider("DOCX", FakeParser)
        assert "docx" in ParserFactory._PROVIDERS

    def test_register_provider_invalid_class(self):
        """Registering a non-BaseParser class should raise ValueError."""

        class NotAParser:
            pass

        with pytest.raises(ValueError, match="must inherit from BaseParser"):
            ParserFactory.register_provider("invalid", NotAParser)  # type: ignore

    def test_list_providers_empty(self):
        """list_providers should return [] when registry is empty."""
        assert ParserFactory.list_providers() == []

    def test_list_providers_sorted(self):
        """list_providers should return sorted provider names."""
        ParserFactory.register_provider("zebra", FakeParser)
        ParserFactory.register_provider("alpha", FakeParser)
        ParserFactory.register_provider("beta", FakeParser)

        assert ParserFactory.list_providers() == ["alpha", "beta", "zebra"]


class TestParserFactoryCreate:
    """Tests for ParserFactory.create routing & instantiation."""

    def setup_method(self):
        self._saved = ParserFactory._PROVIDERS.copy()
        ParserFactory._PROVIDERS.clear()

    def teardown_method(self):
        ParserFactory._PROVIDERS.clear()
        ParserFactory._PROVIDERS.update(self._saved)

    def _settings(self, provider: str, extract_images: bool = True) -> MagicMock:
        """Build a mock Settings with ingestion.parser configured."""
        settings = MagicMock()
        settings.ingestion.parser.provider = provider
        settings.ingestion.parser.extract_images = extract_images
        settings.ingestion.loader = None  # no legacy block → forces parser path
        return settings

    def test_create_success(self):
        """Creating a registered provider should return its instance."""
        ParserFactory.register_provider("fake", FakeParser)

        parser = ParserFactory.create(self._settings("fake"), collection="contracts")

        assert isinstance(parser, FakeParser)
        assert parser.extract_images is True
        # image_storage_dir is derived from collection
        assert "contracts" in str(parser.image_storage_dir)

    def test_create_case_insensitive(self):
        """Provider lookup should be case-insensitive."""
        ParserFactory.register_provider("fake", FakeParser)

        parser = ParserFactory.create(self._settings("FAKE"), collection="c")
        assert isinstance(parser, FakeParser)

    def test_create_passes_extract_images_flag(self):
        """Factory should forward the extract_images flag to the parser."""
        ParserFactory.register_provider("fake", FakeParser)

        parser = ParserFactory.create(self._settings("fake", extract_images=False), collection="c")
        assert parser.extract_images is False

    def test_create_passes_page_batch_size(self):
        """Factory should forward parser.page_batch_size to the provider (T23).

        0 = single unrestricted docling conversion — the T23 corpus-integrity
        fix for batched-conversion content loss.
        """
        ParserFactory.register_provider("fake", FakeParser)

        settings = self._settings("fake")
        settings.ingestion.parser.page_batch_size = 0
        parser = ParserFactory.create(settings, collection="c")
        assert parser.init_kwargs.get("page_batch_size") == 0

    def test_create_omits_page_batch_size_when_unset(self):
        """Unconfigured page_batch_size must not leak into provider kwargs."""
        ParserFactory.register_provider("fake", FakeParser)

        settings = self._settings("fake")
        del settings.ingestion.parser.page_batch_size
        parser = ParserFactory.create(settings, collection="c")
        assert "page_batch_size" not in parser.init_kwargs

    def test_create_passes_parse_cache_dir_when_enabled(self):
        """Factory forwards the repo-root-resolved cache dir when the D-036
        parse cache is configured and enabled."""
        ParserFactory.register_provider("fake", FakeParser)
        settings = self._settings("fake")
        settings.ingestion.parser.parse_cache = ParseCacheSettings(
            enabled=True, dir="data/parsed"
        )

        parser = ParserFactory.create(settings, collection="c")

        # Relative dir is anchored to REPO_ROOT, CWD-independent (resolve_path).
        assert parser.init_kwargs["parse_cache_dir"] == str(
            (REPO_ROOT / "data" / "parsed").resolve()
        )

    def test_create_omits_parse_cache_when_disabled(self):
        """enabled=false must not leak the kwarg — providers stay cache-free."""
        ParserFactory.register_provider("fake", FakeParser)
        settings = self._settings("fake")
        settings.ingestion.parser.parse_cache = ParseCacheSettings(
            enabled=False, dir="data/parsed"
        )

        parser = ParserFactory.create(settings, collection="c")
        assert "parse_cache_dir" not in parser.init_kwargs

    def test_create_omits_parse_cache_when_unset(self):
        """Unconfigured parse_cache (pre-D-036 configs and mocks) → no kwarg."""
        ParserFactory.register_provider("fake", FakeParser)

        settings = self._settings("fake")
        del settings.ingestion.parser.parse_cache
        parser = ParserFactory.create(settings, collection="c")
        assert "parse_cache_dir" not in parser.init_kwargs

    def test_create_unknown_provider(self):
        """An unregistered provider should raise a clear ValueError."""
        ParserFactory.register_provider("fake", FakeParser)

        with pytest.raises(ValueError) as exc_info:
            ParserFactory.create(self._settings("unknown"), collection="c")

        msg = str(exc_info.value)
        assert "Unsupported Parser provider: 'unknown'" in msg
        assert "Available providers:" in msg

    def test_create_no_providers_registered(self):
        """Empty registry should surface 'none' in the available list."""
        with pytest.raises(ValueError) as exc_info:
            ParserFactory.create(self._settings("pdf"), collection="c")

        assert "Unsupported Parser provider: 'pdf'" in str(exc_info.value)
        assert "Available providers: none" in str(exc_info.value)

    def test_create_provider_instantiation_failure(self):
        """Provider constructor errors should be wrapped in RuntimeError."""

        class BrokenParser(BaseParser):
            def __init__(self, **kwargs: Any) -> None:
                raise ValueError("Intentional init error")

            def parse(self, file_path: Union[str, Path]) -> Document:  # type: ignore[override]
                return Document(id="x", text="", metadata={"source_path": str(file_path)})

        ParserFactory.register_provider("broken", BrokenParser)

        with pytest.raises(RuntimeError) as exc_info:
            ParserFactory.create(self._settings("broken"), collection="c")

        msg = str(exc_info.value)
        assert "Failed to instantiate Parser provider 'broken'" in msg
        assert "Intentional init error" in msg

    def test_create_defaults_to_pdf_when_parser_config_absent(self):
        """Missing ingestion.parser should default to the 'pdf' provider."""
        ParserFactory.register_provider("pdf", FakeParser)  # stand-in for PdfTextParser

        settings = MagicMock()
        settings.ingestion.parser = None  # no parser block in config
        settings.ingestion.loader = None  # no legacy loader block either

        parser = ParserFactory.create(settings, collection="c")
        assert isinstance(parser, FakeParser)
        assert parser.extract_images is True  # default

    def test_create_defaults_to_pdf_when_ingestion_absent(self):
        """Missing ingestion entirely should also default to 'pdf'."""
        ParserFactory.register_provider("pdf", FakeParser)

        settings = MagicMock()
        settings.ingestion = None

        parser = ParserFactory.create(settings, collection="c")
        assert isinstance(parser, FakeParser)

    def test_create_legacy_loader_fallback(self):
        """K1 backwards-compat: a legacy ingestion.loader block still routes."""
        ParserFactory.register_provider("fake", FakeParser)

        settings = MagicMock()
        settings.ingestion.parser = None  # new block absent
        settings.ingestion.loader.provider = "fake"  # legacy block present
        settings.ingestion.loader.extract_images = False

        parser = ParserFactory.create(settings, collection="c")
        assert isinstance(parser, FakeParser)
        assert parser.extract_images is False  # read from legacy loader block


class TestBuiltinAliases:
    """K3: pdf_text is an explicit alias for pdf (both → PdfTextParser).

    These tests re-register builtins in setup_method because other test classes
    clear the shared ``_PROVIDERS`` registry in their own setup/teardown.
    """

    def setup_method(self):
        from src.libs.parser.parser_factory import _register_builtin_providers
        _register_builtin_providers()

    def test_pdf_and_pdf_text_both_registered(self):
        from src.libs.parser.pdf_text_parser import PdfTextParser
        assert "pdf" in ParserFactory._PROVIDERS
        assert "pdf_text" in ParserFactory._PROVIDERS
        # alias points at the same class object
        assert ParserFactory._PROVIDERS["pdf_text"] is ParserFactory._PROVIDERS["pdf"]
        assert ParserFactory._PROVIDERS["pdf_text"] is PdfTextParser

    def test_pdf_text_extensions(self):
        assert ParserFactory.get_supported_extensions("pdf_text") == [".pdf"]
        assert ParserFactory.get_supported_extensions("pdf") == [".pdf"]

    def test_create_with_pdf_text_routes_to_pdf_text_parser(self):
        from src.libs.parser.pdf_text_parser import PdfTextParser
        settings = MagicMock()
        settings.ingestion.parser.provider = "pdf_text"
        settings.ingestion.parser.extract_images = True
        settings.ingestion.loader = None
        parser = ParserFactory.create(settings, collection="c")
        assert isinstance(parser, PdfTextParser)
