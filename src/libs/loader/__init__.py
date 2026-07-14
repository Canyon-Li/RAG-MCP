"""Loader Module — file integrity utilities only.

Document parsing has moved to ``src/libs/parser/`` (K1, parser-centric
architecture; see pdf改进计划.md). This package now retains only the
file-integrity helpers (SHA256-based ingestion skip), which are file IO
utilities rather than document-format parsers.
"""

from src.libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker

__all__ = [
    "FileIntegrityChecker",
    "SQLiteIntegrityChecker",
]
