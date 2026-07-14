"""Abstract base class for document parsers.

This module defines the pluggable interface for document parsers,
enabling seamless parsing of different document formats (PDF, DOCX, ...)
with unified output structure.

Design Principles:
- Single Responsibility: Parsers only handle format unification + structure extraction
- Type Safety: Return standardized Document type from core.types
- No Splitting: Parsers don't chunk documents, only parse and normalize
- Graceful Degradation: Failures in optional features (e.g. image extraction) shouldn't block text parsing

K1 (DEV_SPEC phase K): renamed from BaseLoader → BaseParser as part of the
parser-centric architecture (RAGFlow deepdoc 移植). The constructor contract is
unified to ``__init__(self, settings, collection, image_storage_dir, **kwargs)``
so ``ParserFactory`` can build any provider uniformly (aligns with the
LLM/Embedding factory pattern). See pdf改进计划.md §15.4.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.core.types import Document


class BaseParser(ABC):
    """Abstract base class for document parsers.

    All parsers must implement the parse() method to parse a file
    and return a standardized Document object with:
    - text: Normalized content (preferably Markdown format)
    - metadata: At minimum must contain 'source_path'

    Constructor contract (shared across providers so ParserFactory can build
    any parser uniformly)::

        __init__(self, settings, collection: str, image_storage_dir: str | Path, **kwargs)

    ``extract_images`` is forwarded by the factory via ``**kwargs``; provider-specific
    config (e.g. ``pdf_deep.sidecar_url``) is read from ``settings`` inside the provider.

    Parsers should handle:
    - Format-specific parsing logic
    - Metadata extraction (title, page count, etc.)
    - Structure normalization (to Markdown when possible)
    - Optional: Image extraction and placeholder insertion
    """

    @abstractmethod
    def parse(self, file_path: str | Path) -> Document:
        """Parse a document file.

        Args:
            file_path: Path to the document file to parse.

        Returns:
            Document object with parsed content and metadata.
            metadata MUST contain at least 'source_path'.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is invalid or unsupported.
            RuntimeError: If parsing fails critically.

        Example:
            >>> parser = PdfTextParser(settings, collection="contracts", image_storage_dir="data/images/contracts")
            >>> doc = parser.parse("data/documents/report.pdf")
            >>> assert "source_path" in doc.metadata
            >>> assert doc.text  # Non-empty text
        """
        pass

    @staticmethod
    def _validate_file(file_path: str | Path) -> Path:
        """Validate that file exists and is readable.

        Args:
            file_path: Path to validate.

        Returns:
            Resolved Path object.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the path is not a file.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return path

    # ------------------------------------------------------------------
    # Shared helpers (new parsers — PdfTextParser/WordParser/... — reuse
    # these instead of copy-pasting).
    # ------------------------------------------------------------------

    def _compute_file_hash(self, file_path: str | Path) -> str:
        """Compute SHA256 hash of file content (64-char hex string).

        Used for a stable doc_id and the per-document image sub-directory.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract a title from the first Markdown H1, else the first non-empty line."""
        lines = text.split("\n")
        for line in lines[:20]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        for line in lines[:10]:
            line = line.strip()
            if line:
                return line
        return None

    @staticmethod
    def _generate_image_id(doc_hash: str, page: int, sequence: int) -> str:
        """Generate a deterministic image id: ``{doc_hash[:8]}_{page}_{sequence}``."""
        return f"{doc_hash[:8]}_{page}_{sequence}"
