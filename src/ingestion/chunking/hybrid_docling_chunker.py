"""HybridChunker adapter for docling-parsed documents (ticket 04 / D-037).

Docling's HybridChunker does layout-aware splitting with a native token
budget: heading paths are folded into each chunk via ``contextualize()``,
oversized tables split with repeated headers, undersized peers merge.
This adapter bridges its output into the project's ``split_document`` seam
(``Document → List[Chunk]``) without touching any downstream stage:

- chunk text = ``chunker.contextualize(chunk)`` (title path prefix included);
- ``metadata.table_html`` = the table item's own GFM Markdown export —
  carried separately so display never depends on the chunk text form;
- ``[IMAGE: id]`` placeholders matched by provenance (page + bbox) —
  ImageCaptioner and the multimodal path stay unchanged;
- chunk ids follow the existing ``{doc_id}_{index:04d}_{hash8}`` scheme, so
  BM25 prefixes and idempotent re-ingest keep working (D-031);
- chunk metadata carries structured ``headings`` (docling meta.headings)
  plus a type-flag group (``contains_text/table/figure/formula``) where
  mixed chunks keep several flags at once; the single-value
  ``section_type`` stays as a compatibility field only — filtering goes
  through the flags, display goes through ``table_html`` presence.

Documents without ``metadata["docling_documents"]`` (non-docling parsers,
the pdf_text degradation chain) fall through to the original
``DocumentChunker`` paths unchanged. A HybridChunker construction failure
(e.g. missing tokenizer dir) also degrades to the original splitter —
splitting must not be the stage that blocks ingestion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.core.types import Chunk, Document
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = get_logger(__name__)

# Docling labels that are NOT plain text for the contains_* flag group.
# Anything else (TEXT/PARAGRAPH/TITLE/SECTION_HEADER/CAPTION/LIST_ITEM/
# FOOTNOTE/… and unknown future labels) counts as text — same default-to-text
# semantics as DoclingParser._LABEL_MAP.
_NON_TEXT_LABELS = frozenset({"TABLE", "PICTURE", "FIGURE", "FORMULA"})
_FIGURE_LABELS = frozenset({"PICTURE", "FIGURE"})


class HybridDoclingChunker(DocumentChunker):
    """``split_document`` implemented over docling's HybridChunker.

    Selection happens in ``create_document_chunker`` (config-driven,
    ``ingestion.chunker.provider``); non-docling documents degrade to the
    inherited recursive/section paths byte-for-byte.
    """

    def __init__(self, settings: "Settings"):
        super().__init__(settings)
        # Lazy: building the HybridChunker loads the tokenizer from the local
        # model dir (heavy-ish); None = not built yet, False = build failed
        # (degraded — don't retry per document).
        self._hybrid_chunker: Any = None

    # ------------------------------------------------------------------
    # Seam
    # ------------------------------------------------------------------

    def split_document(self, document: Document) -> List[Chunk]:
        """Split via HybridChunker when DoclingDocuments are available.

        Falls back to the inherited paths (unchanged behaviour) when the
        document did not come from docling, or when the hybrid chunker
        could not be built / produced nothing usable.
        """
        ddocs = document.metadata.get("docling_documents")
        if ddocs:
            chunks = self._split_hybrid(document, ddocs)
            if chunks:
                self.last_method = "hybrid_docling"
                return chunks
            logger.warning(
                f"HybridChunker produced no chunks for {document.id}; "
                f"falling back to the recursive path"
            )
        self.last_method = "recursive"
        return super().split_document(document)

    # ------------------------------------------------------------------
    # Hybrid path
    # ------------------------------------------------------------------

    def _split_hybrid(
        self, document: Document, ddocs: List[Any]
    ) -> List[Chunk]:
        """Map DoclingDocuments → DocChunks → Chunks (sequential ids)."""
        chunker = self._get_hybrid_chunker()
        if chunker is None:
            return []

        raw: List[Dict[str, Any]] = []
        for ddoc in ddocs:
            # docling chunks carry DocItem views (label/prov survive, but
            # TableItem.export_to_markdown does not) — index the real table
            # items by self_ref to recover them for the GFM export.
            tables_by_ref = {
                t.self_ref: t for t in (getattr(ddoc, "tables", None) or [])
            }
            for dc in chunker.chunk(ddoc):
                rc = self._raw_from_doc_chunk(dc, chunker, ddoc, tables_by_ref)
                if rc is not None:
                    raw.append(rc)
        raw = [rc for rc in raw if rc["text"].strip()]
        if not raw:
            return []

        # Placeholders before id generation: chunk ids hash the final text.
        self._inject_image_placeholders(raw, document.metadata.get("images") or [])

        chunks: List[Chunk] = []
        for index, rc in enumerate(raw):
            text = rc["text"]
            chunk_id = self._generate_chunk_id(document.id, index, text)
            md = self._inherit_metadata(document, index, text)
            md["section_type"] = rc["section_type"]
            md["contains_text"] = rc["contains_text"]
            md["contains_table"] = rc["contains_table"]
            md["contains_figure"] = rc["contains_figure"]
            md["contains_formula"] = rc["contains_formula"]
            if rc["headings"]:
                md["headings"] = rc["headings"]
            if rc["table_html"]:
                md["table_html"] = rc["table_html"]
            if rc["page"] is not None:
                md["page_num"] = rc["page"]
            if rc["bbox"]:
                md["bbox"] = rc["bbox"]
            chunks.append(Chunk(id=chunk_id, text=text, metadata=md))
        return chunks

    def _raw_from_doc_chunk(
        self,
        dc: Any,
        chunker: Any,
        ddoc: Any,
        tables_by_ref: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """One docling DocChunk → raw chunk dict (text/flags/structure)."""
        doc_items = list(getattr(dc.meta, "doc_items", None) or [])
        labels = [self._label_name(i) for i in doc_items]
        label_set = set(labels)

        contains_table = "TABLE" in label_set
        contains_figure = bool(_FIGURE_LABELS & label_set)
        contains_formula = "FORMULA" in label_set
        # Mixed chunks (e.g. TEXT+CAPTION+TABLE) keep every flag at once —
        # "search tables only" and "search prose only" both hit them.
        # The empty-doc_items case is unreachable (pydantic min_length=1)
        # but falls back to text so the flag and section_type agree.
        contains_text = any(l not in _NON_TEXT_LABELS for l in labels) or not label_set

        page, bbox = self._first_provenance(doc_items)

        return {
            # contextualize() = serialized headings path + body (spike 01:
            # it is a chunker method, not a chunk method).
            "text": chunker.contextualize(chunk=dc),
            "headings": list(getattr(dc.meta, "headings", None) or []),
            # Compatibility single value only — filtering goes through the
            # contains_* flags. Vocabulary matches the legacy section path
            # (table / equation / figure / text) so old consumers see one
            # consistent set of values across paths.
            "section_type": (
                "table" if contains_table
                else "equation" if contains_formula
                else "figure" if contains_figure
                else "text"
            ),
            "contains_text": contains_text,
            "contains_table": contains_table,
            "contains_figure": contains_figure,
            "contains_formula": contains_formula,
            "table_html": self._table_gfm(doc_items, ddoc, tables_by_ref),
            "page": page,
            "bbox": bbox,
        }

    # ------------------------------------------------------------------
    # Table / picture helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _label_name(item: Any) -> str:
        """Docling item → label name string ('' when absent)."""
        lbl = getattr(item, "label", None)
        if lbl is None:
            return ""
        return lbl.name if hasattr(lbl, "name") else str(lbl)

    @staticmethod
    def _table_gfm(
        doc_items: List[Any], ddoc: Any, tables_by_ref: Dict[str, Any]
    ) -> Optional[str]:
        """First table in the chunk → its own GFM Markdown export.

        Chunk doc_items are plain DocItem views — resolve back to the real
        TableItem via self_ref (future-proof: if a view already carries the
        export, use it directly). Carried separately from the chunk text:
        display (ResponseBuilder) renders from ``table_html``; a chunk that
        is a slice of an oversized table still shows the full table.
        """
        for item in doc_items:
            if HybridDoclingChunker._label_name(item) != "TABLE":
                continue
            table = tables_by_ref.get(getattr(item, "self_ref", None))
            if table is None and hasattr(item, "export_to_markdown"):
                table = item  # already a full TableItem
            if table is None:
                continue
            for kwargs in ({"doc": ddoc}, {}):
                try:
                    out = table.export_to_markdown(**kwargs)
                except TypeError:
                    continue  # signature mismatch — try next form
                except Exception:
                    continue
                if out and out.strip():
                    return out
        return None

    @staticmethod
    def _first_provenance(doc_items: List[Any]) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        """(page, bbox) of the first item that has provenance (chunk-level)."""
        for item in doc_items:
            prov = getattr(item, "prov", None)
            if not prov:
                continue
            p = prov[0]
            page = getattr(p, "page_no", None)
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
        return None, None

    # ------------------------------------------------------------------
    # Image placeholder injection (prov-matched)
    # ------------------------------------------------------------------

    def _inject_image_placeholders(
        self, raw: List[Dict[str, Any]], images: List[Dict[str, Any]]
    ) -> None:
        """Append ``[IMAGE: id]`` to the chunk owning each rendered figure.

        Two-step match (ticket 04, "按 prov(page+bbox) 匹配"):
        1. exact (page, bbox) — a chunk whose items sit exactly where the
           figure was rendered (picture items rarely survive into docling
           chunks, but when they do this is authoritative);
        2. spatial nearest chunk — same page preferred, then nearest page,
           then shortest bbox-center distance; ties keep document order.
        Injecting here (not in the parser) keeps ImageCaptioner and the
        whole multimodal path untouched: they only ever look for the
        placeholder inside chunk text.
        """
        if not images:
            return
        by_key: Dict[Tuple, int] = {}
        for i, rc in enumerate(raw):
            key = self._prov_key(rc["page"], rc["bbox"])
            if key is not None and key not in by_key:
                by_key[key] = i

        for img in images:
            img_id = img.get("id")
            if not img_id:
                continue
            key = self._prov_key(img.get("page"), img.get("bbox"))
            target = by_key.get(key) if key is not None else None
            if target is None:
                target = self._nearest_chunk_index(raw, img)
            if target is None:
                logger.warning(
                    f"No chunk to host image {img_id}; placeholder skipped"
                )
                continue
            placeholder = f"[IMAGE: {img_id}]"
            if placeholder not in raw[target]["text"]:
                raw[target]["text"] += f"\n{placeholder}"
            # The hosting chunk now carries figure content even though the
            # picture item itself produced no chunk (contains_* is the
            # filtering surface; section_type stays content-dominant).
            raw[target]["contains_figure"] = True

    @staticmethod
    def _prov_key(
        page: Optional[int], bbox: Optional[Dict[str, Any]]
    ) -> Optional[Tuple]:
        """(page, x0, top, x1, bottom) identity key, None when incomplete."""
        if page is None or not bbox:
            return None
        try:
            return (
                page,
                bbox.get("x0"), bbox.get("top"),
                bbox.get("x1"), bbox.get("bottom"),
            )
        except AttributeError:
            return None

    @staticmethod
    def _nearest_chunk_index(
        raw: List[Dict[str, Any]], img: Dict[str, Any]
    ) -> Optional[int]:
        """Index of the chunk spatially closest to the image.

        Sort key = (page distance, squared bbox-center distance) so same-page
        chunks always win; first-best wins ties → deterministic.
        """
        img_page = img.get("page") or 1
        img_bbox = img.get("bbox") or {}
        icx = ((img_bbox.get("x0") or 0) + (img_bbox.get("x1") or 0)) / 2
        icy = ((img_bbox.get("top") or 0) + (img_bbox.get("bottom") or 0)) / 2

        best_i: Optional[int] = None
        best_key: Optional[Tuple[int, float]] = None
        for i, rc in enumerate(raw):
            page = rc["page"] or 1
            bbox = rc["bbox"] or {}
            ccx = ((bbox.get("x0") or 0) + (bbox.get("x1") or 0)) / 2
            ccy = ((bbox.get("top") or 0) + (bbox.get("bottom") or 0)) / 2
            key = (
                abs(page - img_page),
                (ccx - icx) ** 2 + (ccy - icy) ** 2,
            )
            if best_key is None or key < best_key:
                best_i, best_key = i, key
        return best_i

    # ------------------------------------------------------------------
    # Lazy chunker construction
    # ------------------------------------------------------------------

    def _get_hybrid_chunker(self) -> Any:
        """Build (once) or return the degraded sentinel.

        The tokenizer comes from the configured local model dir — no HF
        network access anywhere on this path (ticket 04; the network is
        blocked on this machine). A build failure logs once and degrades
        every split to the inherited recursive path.
        """
        if self._hybrid_chunker is None:
            try:
                from docling.chunking import HybridChunker
                from docling_core.transforms.chunker.tokenizer.huggingface import (
                    HuggingFaceTokenizer,
                )

                ingestion = self._settings.ingestion
                cfg = ingestion.chunker if ingestion is not None else None
                tokenizer_path = cfg.tokenizer_path if cfg is not None else None
                if cfg is None or not tokenizer_path:
                    raise ValueError(
                        "ingestion.chunker is not configured for "
                        "hybrid_docling (missing tokenizer_path)"
                    )
                tokenizer = HuggingFaceTokenizer.from_pretrained(
                    model_name=tokenizer_path,
                    max_tokens=cfg.max_tokens,
                    # Hard no-network guarantee (ticket 04): a local dir loads
                    # offline anyway; a mistaken repo id fails here instead of
                    # reaching for the (blocked) HF hub.
                    local_files_only=True,
                )
                self._hybrid_chunker = HybridChunker(
                    tokenizer=tokenizer, merge_peers=cfg.merge_peers
                )
                logger.info(
                    f"HybridChunker ready (tokenizer={tokenizer_path}, "
                    f"max_tokens={cfg.max_tokens}, merge_peers={cfg.merge_peers})"
                )
            except Exception as e:
                logger.error(
                    f"HybridChunker construction failed ({e}); degrading to "
                    f"the recursive splitter for this session"
                )
                self._hybrid_chunker = False
        return self._hybrid_chunker or None
