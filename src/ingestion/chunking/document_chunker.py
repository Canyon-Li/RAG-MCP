"""Document chunking module - adapts libs.splitter for business layer.

This module serves as the adapter layer between libs.splitter (pure text splitting)
and Ingestion Pipeline (business object transformation). It transforms Document
objects into Chunk objects with proper ID generation, metadata inheritance, and
traceability.

Core Value-Add (vs libs.splitter):
1. Chunk ID Generation: Deterministic and unique IDs for each chunk
2. Metadata Inheritance: Propagates Document metadata to all chunks
3. chunk_index: Records sequential position within document
4. source_ref: Establishes parent-child traceability
5. Type Conversion: str → Chunk object (core.types contract)

K2 (DEV_SPEC phase K): section-aware splitting. When a layout-aware parser
(e.g. pdf_deep via the RAGFlow deepdoc sidecar) populates
``Document.metadata["sections"]``, the chunker respects type boundaries —
tables are kept whole, titles merge into the following text, headers/footers
are dropped, oversized tables split at row boundaries, lists at item boundaries.
Table HTML is separated: ``chunk.text`` holds cleaned plain text for embedding
while ``metadata.table_html`` keeps the original HTML for display (table_html
only on the first chunk when split).
Documents without sections fall through to the original plain-text path,
byte-for-byte unchanged.

Design Principles:
- Adapter Pattern: Bridges text splitter tool with business objects
- Config-Driven: Uses SplitterFactory for configuration-based strategy selection
- Deterministic: Same Document produces same Chunk IDs on repeat splits
- Type-Safe: Enforces core.types.Chunk contract
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.types import Chunk, Document
from src.libs.splitter.splitter_factory import SplitterFactory

if TYPE_CHECKING:
    from src.core.settings import Settings


class DocumentChunker:
    """Converts Documents into Chunks with business-level enrichment.

    This class wraps a text splitter (from libs) and adds business logic:
    - Generates stable chunk IDs
    - Inherits and extends metadata
    - Maintains document traceability
    - K2: respects typed section boundaries when Document.metadata["sections"]
      is present

    Attributes:
        _splitter: The underlying text splitter from libs layer
        _settings: Configuration settings for chunking behavior
        _chunk_size: chunk-size threshold for section-aware splitting (tables/lists)
    """

    def __init__(self, settings: "Settings"):
        """Initialize DocumentChunker with configuration.

        Args:
            settings: Configuration settings containing splitter configuration.

        Raises:
            ValueError: If splitter configuration is invalid or provider unknown
        """
        self._settings = settings
        self._splitter = SplitterFactory.create(settings)
        # K2: chunk-size threshold for oversized table/list splitting.
        ingestion = getattr(settings, "ingestion", None)
        self._chunk_size = getattr(ingestion, "chunk_size", None) or 1000

    def split_document(self, document: Document) -> List[Chunk]:
        """Split a Document into Chunks with full business enrichment.

        K2: when ``document.metadata["sections"]`` is present (layout-aware
        parsing), split by section type boundaries;
        otherwise use the original plain-text path, byte-for-byte unchanged.

        Args:
            document: Source document to split into chunks

        Returns:
            List of Chunk objects with deterministic IDs, inherited metadata,
            chunk_index, source_ref, and (K2) optional section_type/bbox/table_html.

        Raises:
            ValueError: If document has no text or invalid structure
        """
        if not document.text or not document.text.strip():
            raise ValueError(f"Document {document.id} has no text content to split")

        # K2: section-aware path when the parser supplied typed sections.
        sections = document.metadata.get("sections")
        if sections:
            chunks = self._split_by_sections(document, sections)
            if chunks:
                return chunks
            # sections produced nothing usable (e.g. all headers/footers) → fall back

        # Plain-text path (pre-K2 behavior, byte-for-byte identical).
        return self._split_plain(document)

    # ==================================================================
    # Plain-text path (pre-K2, byte-for-byte stable)
    # ==================================================================

    def _split_plain(self, document: Document) -> List[Chunk]:
        """Original plain-text splitting path.

        Used when ``Document.metadata["sections"]`` is absent or produced no
        output. Preserved verbatim from pre-K2 for regression stability.
        """
        text_fragments = self._splitter.split_text(document.text)

        if not text_fragments:
            raise ValueError(
                f"Splitter returned no chunks for document {document.id}. "
                f"Text length: {len(document.text)}"
            )

        chunks: List[Chunk] = []
        for index, text in enumerate(text_fragments):
            chunk_id = self._generate_chunk_id(document.id, index, text)
            chunk_metadata = self._inherit_metadata(document, index, text)
            chunk = Chunk(id=chunk_id, text=text, metadata=chunk_metadata)
            chunks.append(chunk)
        return chunks

    # ==================================================================
    # K2: section-aware splitting
    # ==================================================================

    def _split_by_sections(
        self, document: Document, sections: List[Dict[str, Any]]
    ) -> List[Chunk]:
        """Split a Document using its typed sections.

        Two passes:
        1. ``_merge_sections_to_segments`` — collapse the section stream into
           typed segments (titles merge into following text, figures/captions
           into adjacent text, tables/lists/equations are hard boundaries,
           headers/footers dropped).
        2. ``_split_segment`` — split each segment by type (text → recursive
           split; table → whole, oversized → row boundaries; list → item
           boundaries; equation → whole).

        Returns [] (caller reverts to plain path) if nothing usable survives.
        """
        segments = self._merge_sections_to_segments(sections)

        raw: List[Dict[str, Any]] = []
        for seg in segments:
            raw.extend(self._split_segment(seg))

        if not raw:
            return []

        chunks: List[Chunk] = []
        for index, rc in enumerate(raw):
            text = rc["text"]
            chunk_id = self._generate_chunk_id(document.id, index, text)
            chunk_metadata = self._inherit_metadata(document, index, text)
            chunk_metadata["section_type"] = rc["section_type"]
            if rc.get("bbox"):
                chunk_metadata["bbox"] = rc["bbox"]
            if rc.get("table_html"):
                chunk_metadata["table_html"] = rc["table_html"]
            if rc.get("page") is not None:
                chunk_metadata["page_num"] = rc["page"]
            chunks.append(Chunk(id=chunk_id, text=text, metadata=chunk_metadata))
        return chunks

    def _merge_sections_to_segments(
        self, sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Pass 1: collapse the section stream into typed segments.

        - header/footer → dropped
        - title → flushes the text buffer, then starts a new one (merges into
          the following text/figure/caption run, per §15.2 "并入后邻 text")
        - text/figure/figure_caption → accumulated into the current text buffer
        - table/list/equation → flush the text buffer, emit as a hard segment
        - unknown type → treated as text
        """
        segments: List[Dict[str, Any]] = []
        text_buffer: List[str] = []
        buf_page: Optional[int] = None

        def flush_text() -> None:
            nonlocal text_buffer, buf_page
            if text_buffer:
                combined = "\n\n".join(t for t in text_buffer if t and t.strip())
                if combined:
                    segments.append({
                        "type": "text",
                        "text": combined,
                        "table_html": None,
                        "bbox": None,
                        "page": buf_page,
                    })
            text_buffer = []
            buf_page = None

        for sec in sections:
            stype = (sec.get("type") or "text").strip()
            stext = sec.get("text") or ""
            spage = sec.get("page")

            if stype in ("header", "footer"):
                continue
            elif stype == "title":
                # Title merges into the following text run → flush old buffer first.
                flush_text()
                text_buffer.append(stext)
                buf_page = spage
            elif stype in ("text", "figure", "figure_caption"):
                text_buffer.append(stext)
                if buf_page is None:
                    buf_page = spage
            elif stype == "table":
                flush_text()
                segments.append({
                    "type": "table",
                    "text": stext,
                    "table_html": sec.get("html") or stext,
                    "bbox": sec.get("bbox"),
                    "page": spage,
                })
            elif stype == "list":
                flush_text()
                segments.append({
                    "type": "list",
                    "text": stext,
                    "table_html": None,
                    "bbox": sec.get("bbox"),
                    "page": spage,
                })
            elif stype == "equation":
                flush_text()
                segments.append({
                    "type": "equation",
                    "text": stext,
                    "table_html": None,
                    "bbox": sec.get("bbox"),
                    "page": spage,
                })
            else:
                # Unknown type → treat as text (§15.2 default).
                text_buffer.append(stext)
                if buf_page is None:
                    buf_page = spage

        flush_text()
        return segments

    def _split_segment(self, seg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Pass 2: split one merged segment into raw chunk dicts."""
        stype = seg["type"]
        page = seg.get("page")

        if stype == "table":
            return self._split_table_segment(seg)
        if stype == "list":
            return self._split_list_segment(seg)
        if stype == "equation":
            text = seg["text"]
            if text and text.strip():
                return [{
                    "text": text,
                    "section_type": "equation",
                    "table_html": None,
                    "bbox": seg.get("bbox"),
                    "page": page,
                }]
            return []
        # text-like (text / figure / figure_caption / unknown merged into text)
        return self._split_text_like(seg["text"], "text", seg.get("bbox"), page)

    def _split_text_like(
        self, text: str, section_type: str, bbox: Any, page: Any = None
    ) -> List[Dict[str, Any]]:
        """Recursive-split a text-like segment via the configured splitter."""
        if not text or not text.strip():
            return []
        fragments = self._splitter.split_text(text)
        if not fragments:
            fragments = [text]
        return [
            {"text": f, "section_type": section_type, "table_html": None,
             "bbox": bbox, "page": page}
            for f in fragments
            if f and f.strip()
        ]

    def _split_table_segment(self, seg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a table segment: whole if ≤ 2×chunk_size, else plain-text rows.

        K2b (§7.5/§14.1): the Chunker only *carries* the Parser's already-cleaned
        plain text (``seg["text"]``) — it does **not** re-clean HTML. Oversized
        tables split at plain-text row boundaries (``\\n``); ``table_html`` is
        attached only to the first chunk (§17.2).
        """
        plain = seg["text"]  # Parser-supplied cleaned text (e.g. "列名: 值" rows)
        html = seg.get("table_html")
        bbox = seg.get("bbox")

        if not plain or not plain.strip():
            return []

        # Whole table fits → keep whole.
        if len(plain) <= 2 * self._chunk_size:
            return [{
                "text": plain,
                "section_type": "table",
                "table_html": html,
                "bbox": bbox,
                "page": seg.get("page"),
            }]

        # Oversized → split the plain text at \n row boundaries;
        # table_html only on the first chunk.
        row_chunks = self._split_plain_rows(plain, self._chunk_size) or [plain]
        results: List[Dict[str, Any]] = []
        for i, chunk_text in enumerate(row_chunks):
            if not chunk_text.strip():
                continue
            results.append({
                "text": chunk_text,
                "section_type": "table",
                "table_html": html if i == 0 else None,
                "bbox": bbox,
                "page": seg.get("page"),
            })
        return results or [{
            "text": plain,
            "section_type": "table",
            "table_html": html,
            "bbox": bbox,
            "page": seg.get("page"),
        }]

    def _split_plain_rows(self, text: str, max_size: int) -> List[str]:
        """Split plain text at ``\\n`` row boundaries, each piece ≈ ≤ max_size.

        Used for oversized tables (K2b): rows are already separated by ``\\n``
        in the Parser's cleaned text, so we pack them greedily up to max_size.
        """
        lines = text.split("\n")
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for line in lines:
            line_len = len(line) + 1  # +1 for the newline join
            if current and current_len + line_len > max_size:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            current.append(line)
            current_len += line_len
        if current:
            chunks.append("\n".join(current))
        return chunks

    def _split_list_segment(self, seg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a list segment: whole if ≤ chunk_size, else item boundaries."""
        text = seg["text"]
        bbox = seg.get("bbox")
        if not text or not text.strip():
            return []
        if len(text) <= self._chunk_size:
            return [{
                "text": text,
                "section_type": "list",
                "table_html": None,
                "bbox": bbox,
                "page": seg.get("page"),
            }]
        items = self._split_list_by_items(text, self._chunk_size) or [text]
        return [
            {"text": c, "section_type": "list", "table_html": None, "bbox": bbox,
             "page": seg.get("page")}
            for c in items
            if c.strip()
        ]

    # --- list helper (compiled pattern at class level) ---
    # K2b: HTML table helpers (_html_table_to_text / _split_html_table_rows) and
    # their regexes (_TAG_RE / _ROW_RE / _CELL_RE) removed — table cleaning is
    # the Parser's job now (§7.5/§14.1); the Chunker only carries cleaned text.

    _LIST_ITEM_RE = re.compile(r"(?m)^[ \t]*(?:[-*+]|\d+\.)[ \t]+")

    def _split_list_by_items(self, text: str, max_size: int) -> List[str]:
        """Split a list at item boundaries (bullet ``- * +`` or numbered ``1.``)."""
        starts = [m.start() for m in self._LIST_ITEM_RE.finditer(text)]
        if not starts:
            return [text] if text.strip() else []
        starts.append(len(text))
        head = text[: starts[0]]
        items: List[str] = []
        for i in range(len(starts) - 1):
            piece = text[starts[i]: starts[i + 1]]
            if i == 0 and head.strip():
                piece = head + piece
            items.append(piece)
        chunks: List[str] = []
        current = ""
        for item in items:
            if current and len(current) + len(item) > max_size:
                chunks.append(current)
                current = ""
            current += item
        if current.strip():
            chunks.append(current)
        return chunks

    # ==================================================================
    # Pre-K2 helpers (unchanged)
    # ==================================================================

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """Generate unique and deterministic chunk ID.

        ID format: ``{doc_id}_{index:04d}_{content_hash8}``.
        """
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{content_hash}"

    def _inherit_metadata(
        self, document: Document, chunk_index: int, chunk_text: str = ""
    ) -> dict:
        """Inherit metadata from document and add chunk-level fields.

        图片相关的 images / image_captions / image_refs 不再写入 chunk
        metadata（它们进 Chroma 会被标量化损坏；图片结构化数据统一由
        ImageStorage 承载，查询端通过正文 [IMAGE: id] 占位符反查）。
        page_num 由 section-aware 路径在 _split_by_sections 中按 section
        page 透传，不再在此处从图片元数据推断。
        """
        chunk_metadata = document.metadata.copy()
        chunk_metadata.pop("images", None)
        chunk_metadata.pop("sections", None)  # parser intermediate
        chunk_metadata["chunk_index"] = chunk_index
        chunk_metadata["source_ref"] = document.id
        return chunk_metadata
