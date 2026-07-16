"""Docling Parser — IBM Docling layout + table extraction.

Docling (DocLayNet layout detection + TableFormer table-structure recognition)
parses the PDF into typed items (title / text / table / picture / caption) and
returns tables directly as GFM Markdown. This parser maps Docling's item stream
into the same typed-section contract as ``PdfTableParser`` so the
``DocumentChunker`` section-aware splitter reuses unchanged (pdf改进计划.md §7.3,
§15.2).

Table representation differs from PdfTableParser: Docling yields GFM Markdown
(not HTML), so both ``section.text`` and ``section.html`` carry the GFM form.
``ResponseBuilder._render_table_snippet`` detects GFM-vs-HTML and renders GFM
as-is (§15.1: plain text for embed, structured form for display).

Constructor contract (shared): ``__init__(settings, collection, image_storage_dir,
extract_images, **kwargs)``. Falls back to ``PdfTextParser`` (MarkItDown) when
docling raises (degradation chain: docling → pdf_text). Image extraction via
PyMuPDF (same image-dict format as PdfTableParser).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from docling.document_converter import DocumentConverter

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

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


# Docling item label (DocItemLabel.name) → project section type understood by
# the DocumentChunker's _merge_sections_to_segments.
_LABEL_MAP: Dict[str, str] = {
    "TITLE": "title",
    "SECTION_HEADER": "title",
    "TEXT": "text",
    "PARAGRAPH": "text",
    "TABLE": "table",
    "PICTURE": "figure",
    "FIGURE": "figure",
    "CAPTION": "figure_caption",
    "FOOTNOTE": "text",
    "LIST_ITEM": "text",
    "PAGE_HEADER": "header",
    "PAGE_FOOTER": "footer",
}


class DoclingParser(BaseParser):
    """PDF parser using IBM Docling (DocLayNet + TableFormer).

    Per document: run ``DocumentConverter``, walk ``DoclingDocument.iterate_items``
    in reading order, and emit typed sections (title/text/table/figure/figure_caption).
    Tables keep Docling's GFM Markdown in both ``text`` (embed/BM25) and ``html``
    (display) fields. Images are extracted via PyMuPDF (Docling picture pixel
    extraction is heavier and not needed here). Falls back to PdfTextParser on
    docling failure.
    """

    def __init__(
        self,
        settings: Any = None,
        collection: str = "default",
        image_storage_dir: str | Path = "data/images",
        extract_images: bool = True,
        **kwargs: Any,
    ):
        """Initialize DoclingParser.

        Args:
            settings: Application settings.
            collection: Collection name scoping the image storage directory.
            image_storage_dir: Base dir for extracted images (Factory-resolved).
            extract_images: Whether to extract embedded images via PyMuPDF.

        Raises:
            ImportError: If docling is not installed.
        """
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "docling is required for DoclingParser. "
                "Install with: pip install docling"
            )
        self.settings = settings
        self.collection = collection
        self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)
        # Composition: pdf_text for fallback (degradation chain §7.7).
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
            sections, images = self._extract_with_docling(path, doc_hash)
        except Exception as e:
            logger.warning(
                f"docling failed for {path}, falling back to pdf_text: {e}"
            )
            return self._fallback_to_pdf_text(path)

        if not sections:
            logger.warning(
                f"docling yielded no sections for {path}, falling back to pdf_text"
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
    # Docling extraction
    # ------------------------------------------------------------------

    def _extract_with_docling(
        self, path: Path, doc_hash: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract typed sections and images via Docling (+ PyMuPDF for images)."""
        converter = self._build_converter()
        result = converter.convert(str(path))
        ddoc = result.document

        sections: List[Dict[str, Any]] = []
        for item, _level in ddoc.iterate_items():
            sec = self._item_to_section(item, ddoc)
            if sec:
                sections.append(sec)

        # Images via PyMuPDF (per-page, same dict format as PdfTableParser).
        images: List[Dict[str, Any]] = []
        if self.extract_images and PYMUPDF_AVAILABLE:
            with fitz.open(path) as doc:
                n_pages = len(doc)
            for page_num in range(1, n_pages + 1):
                for img in self._extract_page_images(path, page_num, doc_hash):
                    images.append(img)
                    # Figure section so the [IMAGE:id] placeholder flows through
                    # the chunker (merged into adjacent text).
                    sections.append({
                        "type": "figure",
                        "text": f"[IMAGE: {img['id']}]",
                        "page": page_num,
                        "bbox": None,
                        "images": [img],
                        "html": None,
                    })
        return sections, images

    def _build_converter(self) -> Any:
        """Build the DocumentConverter (default: StandardPdfPipeline).

        Hook for subclasses to swap the pipeline — e.g. ``DoclingVlmParser``
        overrides this to use ``VlmPipeline`` backed by a remote VLM.
        """
        return DocumentConverter()

    def _item_to_section(
        self, item: Any, ddoc: Any
    ) -> Optional[Dict[str, Any]]:
        """Map one Docling item to a typed section dict, or None to skip."""
        label = self._item_label(item)
        page, bbox = self._provenance(item)

        if label == "TABLE":
            md = self._table_markdown(item, ddoc)
            if not md or not md.strip():
                return None
            # Both text (embed) and html (display) carry the GFM form — Docling
            # yields Markdown directly, unlike pdfplumber's HTML.
            return {
                "type": "table",
                "text": md,
                "page": page,
                "bbox": bbox,
                "images": [],
                "html": md,
            }

        stype = _LABEL_MAP.get(label, "text")
        if stype == "figure":
            # Pictures are extracted via PyMuPDF above; skip here to avoid dupes.
            return None

        text = getattr(item, "text", None) or ""
        if not text.strip():
            return None
        return {
            "type": stype,
            "text": text,
            "page": page,
            "bbox": bbox,
            "images": [],
            "html": None,
        }

    @staticmethod
    def _item_label(item: Any) -> str:
        lbl = getattr(item, "label", None)
        if lbl is None:
            return ""
        return lbl.name if hasattr(lbl, "name") else str(lbl)

    @staticmethod
    def _table_markdown(table_item: Any, ddoc: Any) -> str:
        """Export a Docling TableItem to GFM Markdown.

        Prefers the ``doc=`` kwarg form (current API); falls back to the
        parameterless form for older versions.
        """
        for kwargs in ({"doc": ddoc}, {}):
            try:
                out = table_item.export_to_markdown(**kwargs)
            except TypeError:
                continue  # signature mismatch — try next form
            except Exception:
                continue
            if out and out.strip():
                return out
        return ""

    @staticmethod
    def _provenance(item: Any) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Extract (page, bbox-dict) from an item's first provenance entry."""
        prov = getattr(item, "prov", None)
        if not prov:
            return 1, None
        p = prov[0]
        page = getattr(p, "page_no", None) or 1
        bbox_obj = getattr(p, "bbox", None)
        bbox: Optional[Dict[str, Any]] = None
        if bbox_obj is not None:
            bbox = {
                "x0": getattr(bbox_obj, "l", None),
                "top": getattr(bbox_obj, "t", None),
                "x1": getattr(bbox_obj, "r", None),
                "bottom": getattr(bbox_obj, "b", None),
            }
        return page, bbox

    # ------------------------------------------------------------------
    # Images (PyMuPDF) — mirrors PdfTableParser._extract_page_images so the
    # emitted image dicts match the Document.images contract exactly.
    # ------------------------------------------------------------------

    def _extract_page_images(
        self, pdf_path: Path, page_num: int, doc_hash: str
    ) -> List[Dict[str, Any]]:
        """Extract embedded images from one page via PyMuPDF (best-effort)."""
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
        """Delegate to PdfTextParser (MarkItDown) and flag degraded."""
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
