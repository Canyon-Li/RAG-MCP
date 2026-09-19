"""Docling Parser — IBM Docling layout + table extraction.

Docling (DocLayNet layout detection + TableFormer table-structure recognition)
parses the PDF into typed items (title / text / table / picture / caption) and
returns tables directly as GFM Markdown. This parser maps Docling's item stream
into the same typed-section contract as ``PdfTableParser`` so the
``DocumentChunker`` section-aware splitter reuses unchanged.

Table representation differs from PdfTableParser: Docling yields GFM Markdown
(not HTML), so both ``section.text`` and ``section.html`` carry the GFM form.
``ResponseBuilder._render_table_snippet`` detects GFM-vs-HTML and renders GFM
as-is (§15.1: plain text for embed, structured form for display).

Parse cache (D-036): a successful complete parse saves its DoclingDocuments
as lossless JSON (``ParseCache``); a later parse of the same content (i.e. a
``--force`` re-ingest — the pipeline's SHA256 skip handles the rest) replays
from JSON instead of re-paying the docling CPU conversion. Replay walks the
identical item-mapping code over the identical item stream, so sections are
equivalent by construction. The cache carries a parser version stamp; a
mapping upgrade invalidates old entries automatically.

Constructor contract (shared): ``__init__(settings, collection, image_storage_dir,
extract_images, **kwargs)``. Falls back to ``PdfTextParser`` (MarkItDown) when
docling raises (degradation chain: docling → pdf_text). Figure regions are
rendered to PNG via PyMuPDF ``get_pixmap(clip=bbox)`` (captures both raster
and vector graphics); the emitted image dicts match PdfTableParser's contract.

OCR routing (ticket 03 / D-038): docling's default runs OCR on every file.
Here the converter's ``do_ocr`` is decided per file — ``ocr_mode=auto``
probes the first pages' text layer via PyMuPDF: digital PDFs (text present)
parse with OCR off (faster, zero quality loss), no-text-layer files (scans)
turn OCR on for that file only, so a scanned document can never silently
index empty. Probe failure degrades conservatively to OCR on. ``always`` /
``never`` force the flag and skip the probe.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import DoclingDocument

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from src.core.types import Document
from src.libs.parser.base_parser import BaseParser
from src.libs.parser.parse_cache import ParseCache
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
    # Ticket 06 / D-039: enriched formula items carry LaTeX in text → they
    # flow as text sections; empty ones stay dropped by the text.strip() gate
    # (pre-enrichment behaviour unchanged).
    "FORMULA": "text",
    "CAPTION": "figure_caption",
    "FOOTNOTE": "text",
    "LIST_ITEM": "text",
    "PAGE_HEADER": "header",
    "PAGE_FOOTER": "footer",
}


def _docling_package_version() -> str:
    """Installed docling version ('unknown' when metadata lookup fails).

    Part of the parse-cache version stamp — parse output follows the docling
    version, so an upgrade must invalidate old cache entries (D-036/D-037).
    """
    try:
        return importlib.metadata.version("docling")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class DoclingParser(BaseParser):
    """PDF parser using IBM Docling (DocLayNet + TableFormer).

    Per document: run ``DocumentConverter``, walk ``DoclingDocument.iterate_items``
    in reading order, and emit typed sections (title/text/table/figure/figure_caption).
    Tables keep Docling's GFM Markdown in both ``text`` (embed/BM25) and ``html``
    (display) fields. Figure regions are rendered to PNG via PyMuPDF
    ``get_pixmap(clip=bbox)`` (captures raster + vector). Falls back to
    PdfTextParser on docling failure.

    With a parse cache configured (``parse_cache_dir``), complete successful
    parses persist their DoclingDocuments as lossless JSON and later parses
    of identical content replay from them — zero DocumentConverter calls
    (D-036). Fallback (degraded) results are never cached.
    """

    # Region rendering for figure extraction (replaces get_images).
    # get_pixmap(clip=bbox) captures both raster and vector content.
    RENDER_DPI = 200  # Resolution for bbox rendering (clarity vs storage tradeoff)
    # Filters logos / page-header bands / decorative elements Docling may label
    # as PICTURE. 50pt ≈ 1.76 cm — low enough to keep small paper figures (a
    # 99pt-tall circuit diagram is real), high enough to drop true micro-noise.
    MIN_FIGURE_SIZE = 50
    # Page-batched conversion (16 GB RAM workaround). docling's preprocess
    # stage fails with std::bad_alloc after ~9 pages per DocumentConverter
    # (ONNX/arena memory accumulates per converter and is never returned),
    # silently skipping every page past the threshold. A FRESH converter per
    # batch resets that accumulation. 8 < 9 keeps every batch under the
    # observed failure threshold with a one-page safety margin.
    DEFAULT_PAGE_BATCH_SIZE = 8
    # OCR routing (ticket 03 / D-038). Valid values for ocr_mode — duplicated
    # from settings.PARSER_OCR_MODES (libs must not import core at module
    # import time; both literals are pinned by unit tests).
    VALID_OCR_MODES = frozenset({"auto", "always", "never"})
    # auto-mode probe: pages to inspect for a text layer.
    OCR_PROBE_PAGES = 3
    # Text-layer threshold, TOTAL-based: stripped chars summed over the
    # probed pages < pages × this value ⇒ no text layer ⇒ OCR on. Under a
    # total a single text page can carry the file (e.g. full-page-image
    # cover + digital body ⇒ treated as having a text layer).
    OCR_TEXT_MIN_CHARS = 25
    # Corroboration signal for the probe log: an image covering ≥ this
    # fraction of the page (the classic scan signature). Log-only — see
    # _probe_needs_ocr for why it does not gate the decision.
    OCR_DOMINANT_IMAGE_RATIO = 0.7

    def __init__(
        self,
        settings: Any = None,
        collection: str = "default",
        image_storage_dir: str | Path = "data/images",
        extract_images: bool = True,
        ocr_mode: str = "auto",
        page_batch_size: int | None = None,
        parse_cache_dir: str | Path | None = None,
        formula_enrichment: bool = False,
        **kwargs: Any,
    ):
        """Initialize DoclingParser.

        Args:
            settings: Application settings.
            collection: Collection name scoping the image storage directory.
            image_storage_dir: Base dir for extracted images (Factory-resolved).
            extract_images: Whether to extract embedded images via PyMuPDF.
            ocr_mode: OCR strategy (ticket 03): ``auto`` probes the text
                layer per file (digital → OCR off, scan → OCR on);
                ``always`` / ``never`` force the converter flag.
            page_batch_size: Pages per docling conversion batch. None →
                DEFAULT_PAGE_BATCH_SIZE (8). 1 disables batching value but not
                the code path; values <= 0 mean "never batch" (single convert).
            parse_cache_dir: Root dir for the D-036 parse cache
                (Factory-resolved when ``ingestion.parser.parse_cache.enabled``
                is true). None keeps the cache off — pre-D-036 behaviour.
            formula_enrichment: Transcribe empty FORMULA items to LaTeX via
                the Vision LLM (ticket 06 / D-039). AND-gated with the
                ``vision_llm`` block; failures degrade to empty items (D-005).

        Raises:
            ImportError: If docling is not installed.
            ValueError: If ocr_mode is not auto/always/never.
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
        self.ocr_mode = str(ocr_mode).lower()
        if self.ocr_mode not in self.VALID_OCR_MODES:
            raise ValueError(
                f"ocr_mode must be one of {sorted(self.VALID_OCR_MODES)}, "
                f"got {ocr_mode!r}"
            )
        # Ticket 06 / D-039: formula enrichment switch. The transcriber is
        # built lazily on first fresh parse (see _transcribe_formulas) —
        # constructing a parser must stay free of Vision-LLM client setup.
        self.formula_enrichment = bool(formula_enrichment)
        self._formula_transcriber: Any = None
        self.page_batch_size = (
            page_batch_size
            if page_batch_size is not None
            else self.DEFAULT_PAGE_BATCH_SIZE
        )
        # D-036 replay cache: absent dir = cache off.
        self._parse_cache = (
            ParseCache(cache_dir=parse_cache_dir, version_stamp=self._version_stamp())
            if parse_cache_dir is not None
            else None
        )
        # Observability side-channel for the pipeline's load-stage trace:
        # True when the last parse() replayed from cache (zero converter
        # calls). Kept off Document.metadata so it never leaks into chunk
        # metadata / stored payloads.
        self.last_parse_replayed = False
        # Side-channel for the OCR routing decision of the last parse():
        # the do_ocr the converter was built with (ticket 03). None when no
        # conversion ran (cache replay) — the cached output already embeds
        # the decision made under the same version stamp.
        self.last_ocr_do_ocr: bool | None = None
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

        self.last_parse_replayed = False
        self.last_ocr_do_ocr = None
        replayed = self._try_replay(path, doc_hash)
        if replayed is not None:
            sections, images, documents = replayed
            self.last_parse_replayed = True
        else:
            try:
                do_ocr = self._decide_ocr(path)
                self.last_ocr_do_ocr = do_ocr
                documents, complete = self._convert_batches(path, do_ocr)
                # BEFORE _walk_documents (enriched LaTeX must reach sections)
                # and before the cache save (LaTeX freezes into the replay
                # JSON — replay never re-transcribes, D-036). Replay path
                # above never gets here.
                self._transcribe_formulas(path, documents)
                sections, images = self._walk_documents(path, doc_hash, documents)
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

            # Only complete, non-empty parses are cacheable — a partial
            # (batch-failed) or degraded result must not be frozen as the
            # file's replay source (D-036). save() is best-effort itself.
            if self._parse_cache is not None and complete:
                self._parse_cache.save(doc_hash, documents)

        # Document.text is a flat rendering of sections (Chunker uses sections).
        full_text = self._sections_to_text(sections)
        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "sections": sections,
            # Ticket 04 (D-037): raw DoclingDocuments for the HybridChunker
            # adapter — the section collapse above loses the doc structure
            # (headings / doc_items / tables) the chunker needs. Fresh parses
            # and cache replays share this exit, so both feed the same
            # objects. Degraded (pdf_text) documents never carry the key —
            # the chunker falls back to the original splitter for them.
            # Popped from chunk metadata in DocumentChunker._inherit_metadata.
            "docling_documents": documents,
        }
        if images:
            metadata["images"] = images
        title = self._extract_title(full_text)
        if title:
            metadata["title"] = title
        return Document(id=doc_id, text=full_text, metadata=metadata)

    def _transcribe_formulas(self, path: Path, documents: List[Any]) -> None:
        """Enrich empty FORMULA items to LaTeX (ticket 06 / D-039), best-effort.

        Lazy construction (first fresh parse): building the transcriber wires
        a Vision-LLM client, which parser construction must not pay when the
        switch is off or the parse replays from cache. The transcriber itself
        degrades per item; this wrapper additionally guards against
        construction failures so enrichment can never break a parse (D-005).
        """
        if not self.formula_enrichment:
            return
        try:
            if self._formula_transcriber is None:
                from src.libs.parser.formula_transcriber import FormulaTranscriber

                self._formula_transcriber = FormulaTranscriber(self.settings)
            self._formula_transcriber.transcribe(path, documents)
        except Exception as e:
            logger.warning(f"formula enrichment skipped for {path}: {e}")

    def _version_stamp(self) -> str:
        """Stamp of the parse mapping that writes/reads the cache (D-036).

        Everything in the payload changes what a replay must reproduce;
        changing any of it bumps the stamp and auto-invalidates old cache
        entries. Extends the D-036 suggestion (hash of ``_LABEL_MAP`` +
        enrichment flags) with the concrete knobs that affect the saved
        docling JSON or the replayed sections: figure rendering
        (DPI / min size), image extraction, batch layout, and the docling
        package version itself — parse output follows it, which is exactly
        why replay exists (D-037).
        """
        payload = {
            "parser": type(self).__name__,
            "label_map": _LABEL_MAP,
            "extract_images": self.extract_images,
            "render_dpi": self.RENDER_DPI,
            "min_figure_size": self.MIN_FIGURE_SIZE,
            "page_batch_size": self.page_batch_size,
            "ocr_mode": self.ocr_mode,
            # Probe thresholds steer the do_ocr that actually entered the
            # converter — retuning them changes the saved JSON, so they are
            # D-036 first-class stamp members (same rule as ocr_mode).
            "ocr_probe_pages": self.OCR_PROBE_PAGES,
            "ocr_text_min_chars": self.OCR_TEXT_MIN_CHARS,
            # Ticket 06 / D-039: enrichment changes the saved JSON content
            # (FORMULA texts gain LaTeX) — flipping the switch must
            # invalidate old cache entries. EFFECTIVE state, not the raw
            # switch: the transcriber is AND-gated with vision_llm.enabled
            # (FormulaTranscriber), so formula_enrichment=true +
            # vision_llm off must stamp as "no enrichment" — otherwise empty
            # formulas freeze into the cache under an enrichment-on stamp and
            # replay stale after vision_llm is enabled.
            "formula_enrichment": self.formula_enrichment
            and self._vision_gate_on(),
            "docling_version": _docling_package_version(),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def _vision_gate_on(self) -> bool:
        """True when the vision_llm block is enabled (mirrors the
        FormulaTranscriber AND-gate; getattr-tolerant for mocked settings)."""
        vision_cfg = getattr(self.settings, "vision_llm", None)
        return bool(getattr(vision_cfg, "enabled", False))

    def _try_replay(
        self, path: Path, doc_hash: str
    ) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Any]]]:
        """Replay a cached parse (D-036); None → run the real docling parse.

        Returns ``(sections, images, documents)`` — the replayed
        DoclingDocuments ride along for the hybrid chunker (ticket 04),
        exactly like the fresh-parse path. Any unusable cache (miss, stamp
        mismatch, corrupt/incomplete entry, or a replay that yields no
        sections) returns None so ``parse`` falls through to a fresh
        conversion — the cache is advisory and must never break ingestion.
        Cache hits only ever happen for content that was parsed before under
        the same stamp, i.e. ``--force`` re-ingests.
        """
        if self._parse_cache is None:
            return None
        try:
            paths = self._parse_cache.lookup(doc_hash)
            if not paths:
                return None
            documents = [
                DoclingDocument.load_from_json(filename=str(p)) for p in paths
            ]
            sections, images = self._walk_documents(path, doc_hash, documents)
        except Exception as e:
            logger.warning(
                f"parse cache replay failed for {doc_hash[:8]}, re-parsing: {e}"
            )
            return None
        if not sections:
            logger.warning(
                f"parse cache for {doc_hash[:8]} yielded no sections, re-parsing"
            )
            return None
        logger.info(
            f"parse cache hit for {doc_hash[:8]}: replayed {len(documents)} "
            f"batch JSON(s) (DocumentConverter skipped)"
        )
        return sections, images, documents

    # ------------------------------------------------------------------
    # Docling extraction
    # ------------------------------------------------------------------

    def _is_figure_item(self, item: Any) -> bool:
        """Return True if the Docling item is a figure (PICTURE/FIGURE label)."""
        return _LABEL_MAP.get(self._item_label(item)) == "figure"

    def _render_figure_region(
        self,
        fitz_doc: Any,
        item: Any,
        doc_hash: str,
        fig_index: int,
    ) -> Optional[Dict[str, Any]]:
        """Render a Docling Figure item's bbox region to a PNG via PyMuPDF.

        Captures BOTH raster and vector graphics by rendering the page area
        to pixels — unlike the old ``get_images()`` which only found embedded
        raster objects. Returns None when the figure should be skipped (too
        small, missing bbox, or render failure).

        Args:
            fitz_doc: Open ``fitz.Document`` shared across all figures in a parse.
            item: Docling item with label PICTURE/FIGURE and ``prov.bbox``.
            doc_hash: Document hash (for image sub-directory + id).
            fig_index: Global figure counter (for unique image id).

        Returns:
            Standard image dict (same shape as old ``_extract_page_images``
            output), or None if the figure is skipped.
        """
        page_num, bbox = self._provenance(item)
        if not bbox:
            logger.debug(f"Skip figure on page {page_num}: no bbox provenance")
            return None

        width = bbox["x1"] - bbox["x0"]
        # Docling's prov.bbox uses PDF bottom-left origin (t > b), so
        # (bottom - top) is negative — take abs() or every real figure gets
        # silently filtered as "too small".
        height = abs(bbox["bottom"] - bbox["top"])
        if width < self.MIN_FIGURE_SIZE or height < self.MIN_FIGURE_SIZE:
            logger.debug(
                f"Skip small figure on page {page_num}: {width:.0f}x{height:.0f}pt"
            )
            return None

        try:
            page = fitz_doc[page_num - 1]
            # Flip y-axis: Docling is bottom-left origin, PyMuPDF Rect is
            # top-left origin. PyMuPDF_y = page_height - Docling_y.
            page_height = page.rect.height
            rect = fitz.Rect(
                bbox["x0"],
                page_height - bbox["top"],
                bbox["x1"],
                page_height - bbox["bottom"],
            )
            pix = page.get_pixmap(clip=rect, dpi=self.RENDER_DPI)

            image_id = self._generate_image_id(doc_hash, page_num, fig_index + 1)
            image_dir = self.image_storage_dir / doc_hash
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"{image_id}.png"
            pix.save(image_path)

            try:
                stored_path = image_path.relative_to(Path.cwd())
            except ValueError:
                stored_path = image_path.absolute()

            return {
                "id": image_id,
                "path": str(stored_path),
                "page": page_num,
                # Ticket 04 (D-037): the hybrid chunker matches chunk picture
                # items back to rendered images by (page, bbox) to inject
                # ``[IMAGE: id]`` placeholders — carry the prov bbox along.
                "bbox": bbox,
                "text_offset": 0,
                "text_length": len(f"[IMAGE: {image_id}]"),
                "position": {
                    "width": pix.width,
                    "height": pix.height,
                    "page": page_num,
                    "index": fig_index,
                },
            }
        except Exception as e:
            logger.warning(f"Failed to render figure on page {page_num}: {e}")
            return None

    def _convert_batches(self, path: Path, do_ocr: bool = True) -> Tuple[List[Any], bool]:
        """Convert the PDF batch-by-batch → (documents, all_batches_ok).

        ``do_ocr`` (ticket 03 / D-038) is the per-file OCR decision threaded
        into every batch's converter — all batches of one file must agree.

        Long PDFs are converted page-batch by page-batch, each batch with a
        FRESH DocumentConverter. Docling's preprocess accumulates native
        memory per converter and dies with std::bad_alloc after ~9 pages
        (observed on 16 GB machines), silently skipping the rest — a fresh
        converter per batch resets the accumulation. See DEFAULT_PAGE_BATCH_SIZE.

        A batch-level failure (e.g. bad_alloc deeper than the threshold) keeps
        documents already converted from earlier batches rather than losing
        the whole document; ``all_batches_ok=False`` marks the result partial
        (and thus uncacheable). If ALL batches fail, documents stays empty and
        parse() falls back to pdf_text as before.
        """
        documents: List[Any] = []
        complete = True
        for s, e in self._page_batches(path):
            converter = self._build_converter(do_ocr)
            try:
                result = (
                    converter.convert(str(path), page_range=(s, e))
                    if e is not None
                    else converter.convert(str(path))
                )
            except Exception as exc:
                complete = False
                logger.warning(
                    f"docling batch pages {s}-{e or 'end'} failed for "
                    f"{path.name}: {exc}"
                )
                continue
            documents.append(result.document)
        return documents, complete

    def _walk_documents(
        self, path: Path, doc_hash: str, documents: List[Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Map converted DoclingDocuments into (sections, images), in order.

        Shared by the fresh-parse and cache-replay paths — replay walks the
        identical mapping code over the identical item stream, which is what
        makes cached sections equivalent to a real parse (D-036). Figures are
        rendered from each Docling Figure item's bbox via PyMuPDF
        (``get_pixmap(clip=bbox)``) — raster + vector in one pass — and
        re-rendered on replay with the same deterministic ids (rendering never
        touches DocumentConverter, so it stays cheap).
        """
        sections: List[Dict[str, Any]] = []
        images: List[Dict[str, Any]] = []

        # Open the fitz document once and share it across all figure renders.
        fitz_doc = (
            fitz.open(path) if (self.extract_images and PYMUPDF_AVAILABLE) else None
        )
        fig_counter = 0
        try:
            for ddoc in documents:
                fig_counter = self._consume_items(
                    ddoc, fitz_doc, doc_hash, sections, images, fig_counter
                )
        finally:
            if fitz_doc is not None:
                fitz_doc.close()

        return sections, images

    def _consume_items(
        self,
        ddoc: Any,
        fitz_doc: Any,
        doc_hash: str,
        sections: List[Dict[str, Any]],
        images: List[Dict[str, Any]],
        fig_counter: int,
    ) -> int:
        """Walk one DoclingDocument's items into sections/images (in place).

        Returns the updated global figure counter — image ids depend on
        figure order across the whole document, not per batch.
        """
        for item, _level in ddoc.iterate_items():
            if fitz_doc is not None and self._is_figure_item(item):
                rendered = self._render_figure_region(
                    fitz_doc, item, doc_hash, fig_counter
                )
                if rendered is not None:
                    fig_counter += 1
                    images.append(rendered)
                    sections.append({
                        "type": "figure",
                        "text": f"[IMAGE: {rendered['id']}]",
                        "page": rendered["page"],
                        "bbox": None,
                        "images": [rendered],
                        "html": None,
                    })
                    continue
            sec = self._item_to_section(item, ddoc)
            if sec:
                sections.append(sec)
        return fig_counter

    def _page_count(self, path: Path) -> Optional[int]:
        """Total page count of the PDF, or None when it can't be read.

        Respects PYMUPDF_AVAILABLE (tests patch it False) and swallows fitz
        errors — an unreadable page count must never fail the parse, it just
        disables batching for that file.
        """
        if not PYMUPDF_AVAILABLE:
            return None
        try:
            with fitz.open(path) as d:
                n = d.page_count
                # fitz returns int; anything else (a mocked fitz in tests, a
                # weird PDF) just disables batching rather than crashing.
                return n if isinstance(n, int) else None
        except Exception as e:
            logger.warning(f"Could not read page count of {path}: {e}")
            return None

    def _page_batches(self, path: Path) -> List[Tuple[int, Optional[int]]]:
        """Yield (start, end) 1-based inclusive page ranges to convert.

        Returns [(1, None)] (a single unrestricted convert — the original
        behaviour) when batching doesn't apply: page count unknown, batch
        size <= 0, or the document fits in one batch.
        """
        n = self._page_count(path)
        size = self.page_batch_size
        if n is None or size is None or size <= 0 or n <= size:
            return [(1, None)]
        return [
            (start, min(start + size - 1, n))
            for start in range(1, n + 1, size)
        ]

    def _build_converter(self, do_ocr: bool = True) -> Any:
        """Build the DocumentConverter (default: StandardPdfPipeline).

        ``do_ocr`` selects docling's OCR stage (ticket 03 / D-038): digital
        PDFs (probed text layer) parse with OCR off — faster, zero quality
        loss — while no-text-layer files keep OCR on. Default True keeps
        subclass overrides signature-compatible even when they ignore it.

        Hook for subclasses to swap the pipeline — e.g. ``DoclingVlmParser``
        overrides this to use ``VlmPipeline`` backed by a remote VLM.
        """
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = do_ocr
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
            },
        )

    # ------------------------------------------------------------------
    # OCR routing (ticket 03 / D-038)
    # ------------------------------------------------------------------

    def _decide_ocr(self, path: Path) -> bool:
        """Resolve the converter's do_ocr for *path* under the configured mode.

        always/never short-circuit; auto probes the text layer. A failed
        probe degrades conservatively to OCR on — the pre-D-038 docling
        default — because the reverse (OCR off for an unreadable file)
        risks a silently empty index (ticket 03: 探测失败不阻断摄入).
        """
        if self.ocr_mode == "always":
            logger.info(f"ocr decision {path.name}: do_ocr=True (mode=always)")
            return True
        if self.ocr_mode == "never":
            logger.info(f"ocr decision {path.name}: do_ocr=False (mode=never)")
            return False
        needs = self._probe_needs_ocr(path)
        if needs is None:
            logger.warning(
                f"OCR probe failed for {path}, degrading to do_ocr=True "
                f"(mode=auto) — conservative default, ingestion continues"
            )
            return True
        return needs

    def _probe_needs_ocr(self, path: Path) -> Optional[bool]:
        """Probe the first pages' text layer → True when OCR is needed.

        Returns None when the probe itself fails (unreadable PDF, PyMuPDF
        missing) — the caller degrades conservatively. Reads only the first
        OCR_PROBE_PAGES pages: cheap, and scan-vs-digital is a per-document
        property.

        Decision signal is TEXT VOLUME ONLY — the stripped chars summed
        over the probed pages must fall below pages × OCR_TEXT_MIN_CHARS
        for OCR to engage (total-based: one text-bearing page carries the
        file).
        The full-page-image scan signature is computed for the log line but
        deliberately does NOT gate the decision: a textless page without a
        dominant image (vector-drawn glyphs) needs OCR just as badly, and
        the failure asymmetry is stark — a false-positive OCR run costs
        time, a false-negative silently empties the index.
        """
        if not PYMUPDF_AVAILABLE:
            return None
        try:
            with fitz.open(path) as d:
                pages = min(d.page_count, self.OCR_PROBE_PAGES)
                if pages <= 0:
                    return None
                chars = sum(
                    len(d[i].get_text().strip()) for i in range(pages)
                )
                big_image_pages = sum(
                    1
                    for i in range(pages)
                    if self._has_dominant_image(d[i])
                )
        except Exception as e:
            logger.warning(f"OCR probe failed for {path}: {e}")
            return None
        needs = chars < pages * self.OCR_TEXT_MIN_CHARS
        logger.info(
            f"OCR probe {path.name}: {chars} text chars over {pages} probed "
            f"page(s), {big_image_pages} full-page-image page(s) → "
            f"needs_ocr={needs}"
        )
        # bool() to shed fitz's Any (page_count is untyped) — keeps mypy honest.
        return bool(needs)

    @classmethod
    def _has_dominant_image(cls, page: Any) -> bool:
        """True when one page image covers ≥ OCR_DOMINANT_IMAGE_RATIO of it.

        Log corroboration only (see _probe_needs_ocr). Any lookup error
        simply reports "no dominant image" — it must never fail the probe.
        """
        try:
            page_area = abs(page.rect)
            for info in page.get_image_info():
                bbox = info.get("bbox")
                if bbox and abs(fitz.Rect(bbox)) >= (
                    page_area * cls.OCR_DOMINANT_IMAGE_RATIO
                ):
                    return True
        except Exception:
            return False
        return False

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
            # Figures are rendered in _extract_with_docling above. Reaching here
            # means rendering was skipped (too small / no bbox / images disabled)
            # — drop the item rather than emit an empty figure section.
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
