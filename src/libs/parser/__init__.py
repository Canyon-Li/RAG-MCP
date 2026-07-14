"""Parser Module.

Document parser components (K1, renamed from the loader package):
- Base parser class (BaseParser)
- PDF text parser (PdfTextParser, MarkItDown-based text tier)
- Word parser (WordParser, DOCX)
- Parser factory (ParserFactory)

Note: ``src/libs/loader/`` still hosts ``file_integrity`` (file IO utilities,
not document parsing). The parser/ package is the pluggable document-parsing
layer; see pdf改进计划.md for the architecture decision.
"""

from src.libs.parser.base_parser import BaseParser
from src.libs.parser.pdf_text_parser import PdfTextParser
from src.libs.parser.word_parser import WordParser
from src.libs.parser.parser_factory import ParserFactory

__all__ = [
    "BaseParser",
    "PdfTextParser",
    "WordParser",
    "ParserFactory",
]
