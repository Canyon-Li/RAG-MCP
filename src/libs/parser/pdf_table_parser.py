"""PDF Table Parser — native table extraction via pdfplumber (K4′).

Text-tier PDF parsing + table extraction using pdfplumber (pure algorithm, no
models / no sidecar). For each page, text words (with bbox) and tables (with
bbox) are emitted as sections ordered by vertical position, so tables land in
their correct position in the text flow (pdf改进计划.md §7.4). Tables are
serialized to both HTML (metadata.table_html, display) and cleaned plain text
(chunk.text, embed) — §15.1. Falls back to PdfTextParser (MarkItDown) when
pdfplumber raises (§7.7). "No tables" is NOT degradation — the parser still
produces plain-text sections via pdfplumber.extract_words.

Constructor contract (shared): __init__(settings, collection, image_storage_dir,
extract_images, **kwargs).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from PIL import Image

from src.core.types import Document
from src.libs.parser.base_parser import BaseParser
from src.libs.parser.pdf_text_parser import PdfTextParser

logger = logging.getLogger(__name__)


class PdfTableParser(BaseParser):
    """PDF parser with native table extraction (pdfplumber, pure algorithm).

    Per page: extract tables (``find_tables``, with bbox) and text words
    (``extract_words``), filter words outside table regions, cluster words into
    lines, and emit sections ordered by vertical position (top). Tables →
    (cleaned plain text, HTML); text → text sections. Falls back to
    PdfTextParser on pdfplumber failure.
    """

    def __init__(
        self,
        settings: Any = None,
        collection: str = "default",
        image_storage_dir: str | Path = "data/images",
        extract_images: bool = True,
        **kwargs: Any,
    ):
        """Initialize PdfTableParser.

        Args:
            settings: Application settings.
            collection: Collection name scoping the image storage directory.
            image_storage_dir: Base dir for extracted images (Factory-resolved).
            extract_images: Whether to extract embedded images via PyMuPDF.

        Raises:
            ImportError: If pdfplumber is not installed.
        """
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError(
                "pdfplumber is required for PdfTableParser. "
                "Install with: pip install pdfplumber"
            )
        self.settings = settings
        self.collection = collection
        self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)
        # Composition: pdf_text for fallback (§7.7).
        self._pdf_text = PdfTextParser(
            settings=settings,
            collection=collection,
            image_storage_dir=image_storage_dir,
            extract_images=extract_images,
        )

    def parse(self, file_path: str | Path) -> Document:
        """Parse a PDF into a Document carrying typed sections (text + table)."""
        path = self._validate_file(file_path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {path}")

        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"

        try:
            sections, images = self._extract_with_pdfplumber(path, doc_hash)
        except Exception as e:
            logger.warning(
                f"pdfplumber failed for {path}, falling back to pdf_text: {e}"
            )
            return self._fallback_to_pdf_text(path)

        if not sections:
            logger.warning(
                f"pdfplumber yielded no sections for {path}, falling back to pdf_text"
            )
            return self._fallback_to_pdf_text(path)

        # Document.text is a flat rendering of sections (Chunker uses sections).
        full_text = self._sections_to_text(sections)
        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "sections": sections,
        }
        if images:
            metadata["images"] = images
        title = self._extract_title(full_text)
        if title:
            metadata["title"] = title
        return Document(id=doc_id, text=full_text, metadata=metadata)

    # ------------------------------------------------------------------
    # pdfplumber extraction
    # ------------------------------------------------------------------

    def _extract_with_pdfplumber(
        self, path: Path, doc_hash: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract sections (text + table) and images across all pages."""
        sections: List[Dict[str, Any]] = []
        images: List[Dict[str, Any]] = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_sections = self._extract_page_sections(page, page_num)
                sections.extend(page_sections)
                if self.extract_images and PYMUPDF_AVAILABLE:
                    page_images = self._extract_page_images(path, page_num, doc_hash)
                    for img in page_images:
                        images.append(img)
                        # figure section so the [IMAGE:id] placeholder flows
                        # through the chunker (merged into adjacent text).
                        sections.append({
                            "type": "figure",
                            "text": f"[IMAGE: {img['id']}]",
                            "page": page_num,
                            "bbox": None,
                            "images": [img],
                            "html": None,
                        })
        return sections, images

    def _extract_page_sections(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract text/table sections from one page, ordered by top position."""
        tables = page.find_tables()
        table_regions: List[Tuple[float, float, List[List[Any]]]] = []
        for t in tables:
            data = t.extract()  # list[list[str|None]]
            bbox = t.bbox  # (x0, top, x1, bottom)
            if bbox and data:
                table_regions.append((float(bbox[1]), float(bbox[3]), data))

        # Text words outside table regions.
        try:
            words = page.extract_words(use_text_flow=True)
        except Exception:
            words = []
        words = [w for w in words if not self._word_in_any_table(w, table_regions)]

        lines = self._cluster_words_to_lines(words)  # [(top, bottom, text)]

        # Interleave lines + tables by top.
        items: List[Tuple[float, str, Any]] = []
        for top, _bot, text in lines:
            items.append((top, "text", text))
        for ttop, tbot, data in table_regions:
            items.append((ttop, "table", data))
        items.sort(key=lambda x: x[0])

        sections: List[Dict[str, Any]] = []
        text_buf: List[str] = []
        for _top, kind, content in items:
            if kind == "text":
                if content and content.strip():
                    text_buf.append(content)
            else:  # table
                if text_buf:
                    sections.append(self._make_text_section(text_buf, page_num))
                    text_buf = []
                html, plain = self._table_to_html_and_plain(content)
                if plain.strip() or html:
                    sections.append({
                        "type": "table",
                        "text": plain,
                        "page": page_num,
                        "bbox": None,
                        "images": [],
                        "html": html,
                    })
        if text_buf:
            sections.append(self._make_text_section(text_buf, page_num))
        return sections

    @staticmethod
    def _word_in_any_table(w: Dict[str, Any], table_regions) -> bool:
        """True if a word's vertical center falls inside any table band."""
        wtop = w.get("top", 0)
        for ttop, tbot, _ in table_regions:
            if ttop <= wtop <= tbot:
                return True
        return False

    @staticmethod
    def _cluster_words_to_lines(words: List[Dict[str, Any]]) -> List[Tuple[float, float, str]]:
        """Cluster words into lines by top position (tolerance 3px)."""
        if not words:
            return []
        words = sorted(words, key=lambda w: (w.get("top", 0), w.get("x0", 0)))
        lines: List[Tuple[float, float, str]] = []
        cur = [words[0]]
        for w in words[1:]:
            if abs(w.get("top", 0) - cur[0].get("top", 0)) <= 3:
                cur.append(w)
            else:
                lines.append(PdfTableParser._line_to_text(cur))
                cur = [w]
        lines.append(PdfTableParser._line_to_text(cur))
        return lines

    @staticmethod
    def _line_to_text(line_words: List[Dict[str, Any]]) -> Tuple[float, float, str]:
        line_words = sorted(line_words, key=lambda w: w.get("x0", 0))
        text = " ".join(str(w.get("text", "")) for w in line_words)
        top = min(w.get("top", 0) for w in line_words)
        bottom = max(w.get("bottom", 0) for w in line_words)
        return (top, bottom, text)

    @staticmethod
    def _make_text_section(text_lines: List[str], page: int) -> Dict[str, Any]:
        return {
            "type": "text",
            "text": "\n".join(t for t in text_lines if t and t.strip()),
            "page": page,
            "bbox": None,
            "images": [],
            "html": None,
        }

    @staticmethod
    def _table_to_html_and_plain(data: List[List[Any]]) -> Tuple[str, str]:
        """Serialize a table (list[list]) to (HTML, plain text '列名: 值').

        First row is treated as the header / column names.
        """
        if not data:
            return "", ""
        rows = [[("" if c is None else str(c)).strip() for c in row] for row in data]
        # HTML
        html_parts = ["<table>"]
        for row in rows:
            html_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
        html_parts.append("</table>")
        html = "".join(html_parts)
        # Plain text: header (row 0) as column names → "列名: 值 | 列名2: 值2"
        plain_lines: List[str] = []
        if rows:
            header = rows[0]
            for row in rows[1:]:
                pairs = []
                for i, cell in enumerate(row):
                    col = header[i] if i < len(header) else f"col{i}"
                    pairs.append(f"{col}: {cell}")
                plain_lines.append(" | ".join(pairs))
        plain = "\n".join(plain_lines)
        return html, plain

    # ------------------------------------------------------------------
    # Images (PyMuPDF) — emit figure sections
    # ------------------------------------------------------------------

    def _extract_page_images(
        self, pdf_path: Path, page_num: int, doc_hash: str
    ) -> List[Dict[str, Any]]:
        """Extract embedded images from a page via PyMuPDF (best-effort)."""
        if not self.extract_images or not PYMUPDF_AVAILABLE:
            return []
        images: List[Dict[str, Any]] = []
        try:
            image_dir = self.image_storage_dir / doc_hash
            image_dir.mkdir(parents=True, exist_ok=True)
            doc = fitz.open(pdf_path)
            try:
                if page_num - 1 >= len(doc):
                    return []
                page = doc[page_num - 1]
                for img_index, img_info in enumerate(page.get_images(full=True)):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        image_id = self._generate_image_id(doc_hash, page_num, img_index + 1)
                        image_filename = f"{image_id}.{image_ext}"
                        image_path = image_dir / image_filename
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)
                        try:
                            img = Image.open(io.BytesIO(image_bytes))
                            width, height = img.size
                        except Exception:
                            width, height = 0, 0
                        try:
                            stored_path = image_path.relative_to(Path.cwd())
                        except ValueError:
                            stored_path = image_path.absolute()
                        images.append({
                            "id": image_id,
                            "path": str(stored_path),
                            "page": page_num,
                            "text_offset": 0,
                            "text_length": len(f"[IMAGE: {image_id}]"),
                            "position": {
                                "width": width,
                                "height": height,
                                "page": page_num,
                                "index": img_index,
                            },
                        })
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract image {img_index} from page {page_num}: {e}"
                        )
                        continue
            finally:
                doc.close()
        except Exception as e:
            logger.warning(f"Image extraction failed for {pdf_path} page {page_num}: {e}")
        return images

    # ------------------------------------------------------------------
    # Fallback + helpers
    # ------------------------------------------------------------------

    def _fallback_to_pdf_text(self, path: Path) -> Document:
        """Delegate to PdfTextParser (MarkItDown) and flag degraded (§7.7)."""
        doc = self._pdf_text.parse(path)
        doc.metadata["degraded"] = True
        return doc

    def _sections_to_text(self, sections: List[Dict[str, Any]]) -> str:
        """Render sections to flat text (Document.text fallback; Chunker uses sections)."""
        parts: List[str] = []
        for s in sections:
            stype = s.get("type", "text")
            if stype == "table":
                parts.append(s.get("text") or s.get("html") or "")
            else:
                parts.append(s.get("text") or "")
        return "\n\n".join(p for p in parts if p and p.strip())
