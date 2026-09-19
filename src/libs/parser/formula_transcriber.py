"""Formula enrichment — transcribe docling FORMULA items to LaTeX (ticket 06 / D-039).

docling's layout model detects formula regions for free but (with the VLM
enrichment stage off) emits them as empty ``FORMULA`` items — boxed, no
content. This module fills the boxes: each empty item's bbox region is
rendered to PNG and transcribed to LaTeX by a Vision LLM (production:
ollama qwen2.5vl-3b on GPU, via the same ``create_vision_llm`` chain
ImageCaptioner uses; prompt lives in ``config/prompts/formula_transcription.txt``).

Why self-built instead of docling's ``do_formula_enrichment`` (D-039): the
built-in stage needs a VLM through docling's transformers engine — its
default model (CodeFormulaV2) is unreachable with HF blocked, and the only
locally-available preset (granite-docling-258M) measured **165 s/page on
CPU torch** with visible quality defects; qwen2.5vl over ollama measured
**~3.6 s/formula, ~7 s/page** with cleaner output (spike 02/04, 2026-09-19).

Placement contract (DoclingParser): run on fresh parses BEFORE the parse
cache save, so LaTeX is frozen into the lossless JSON and ``--force``
re-ingests replay it for free (D-036) — the replay path never calls this.
Degradation rule (D-005): a failing VLM leaves the item empty and never
breaks the parse; enrichment is best-effort per item.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.libs.llm.base_vision_llm import ImageInput

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class FormulaTranscriber:
    """Fill empty ``FORMULA`` DoclingDocument items with VLM-transcribed LaTeX.

    Idempotent by construction: only items whose text is still empty are
    processed, so re-running over partially enriched documents skips them.
    """

    # Same rendering resolution as DoclingParser.RENDER_DPI figure regions.
    RENDER_DPI = 200
    # Margin around the detected bbox, matching docling's own enrichment
    # stage expansion_factor (0.18) — keeps cut-off sub/superscripts.
    EXPANSION = 0.18

    def __init__(self, settings: Any, llm: Any = None, **kwargs: Any):
        """Initialize the transcriber.

        Args:
            settings: Application settings (reads ``vision_llm``).
            llm: Optional pre-initialized Vision LLM (mock-injection seam,
                ImageCaptioner precedent). None → built via
                ``LLMFactory.create_vision_llm``.
            **kwargs: Tolerates extra parser kwargs (ignored).
        """
        self.settings = settings
        vision_cfg = getattr(settings, "vision_llm", None)
        # AND-gate with the vision_llm block: the transcription model IS the
        # vision model — one kill-switch (vision_llm.enabled) covers both
        # captioning and formula transcription.
        self._configured = bool(getattr(vision_cfg, "enabled", False))
        if llm is not None:
            self.llm = llm
            self._configured = True  # explicit injection overrides the gate
        elif self._configured:
            from src.libs.llm.llm_factory import LLMFactory

            self.llm = LLMFactory.create_vision_llm(settings)
        else:
            self.llm = None
            logger.info(
                "FormulaTranscriber: vision_llm disabled — formula items stay empty"
            )
        self.prompt = self._load_prompt()

    @property
    def enabled(self) -> bool:
        """True when a Vision LLM is wired and rendering is possible."""
        return self.llm is not None and self._configured and PYMUPDF_AVAILABLE

    def _load_prompt(self) -> str:
        """Load the transcription prompt (config/prompts/, project convention)."""
        from src.core.settings import resolve_path

        prompt_path = resolve_path("config/prompts/formula_transcription.txt")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        logger.warning(
            f"formula transcription prompt missing: {prompt_path}; using fallback"
        )
        return (
            "The image is a mathematical formula cropped from a scientific "
            "paper. Transcribe it as LaTeX source code. Output ONLY the LaTeX."
        )

    def transcribe(self, pdf_path: str | Path, documents: list[Any]) -> int:
        """Transcribe every empty FORMULA item across the documents, in place.

        Args:
            pdf_path: Source PDF (regions are rendered from its pages).
            documents: DoclingDocuments from a fresh parse (all page batches).

        Returns:
            Number of items successfully enriched (0 when disabled/unusable).
        """
        if not documents:
            return 0
        if not self.enabled:
            return 0
        enriched = 0
        try:
            with fitz.open(str(pdf_path)) as fitz_doc:
                for ddoc in documents:
                    for item, _level in ddoc.iterate_items():
                        if self._label(item) != "FORMULA":
                            continue
                        if (getattr(item, "text", None) or "").strip():
                            continue  # already enriched (idempotency)
                        if self._transcribe_item(fitz_doc, item):
                            enriched += 1
        except Exception as e:
            # Never break a parse for enrichment (D-005).
            logger.warning(f"formula transcription aborted for {pdf_path}: {e}")
        if enriched:
            logger.info(f"formula transcription enriched {enriched} item(s)")
        return enriched

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _transcribe_item(self, fitz_doc: Any, item: Any) -> bool:
        """Render + transcribe one item; True when its text was filled."""
        page_no, bbox = self._provenance(item)
        if bbox is None:
            return False
        png = self._render_region(fitz_doc, page_no, bbox)
        if png is None:
            return False
        try:
            response = self.llm.chat_with_image(
                text=self.prompt, image=ImageInput(data=png, mime_type="image/png")
            )
            latex = self._clean_latex(getattr(response, "content", "") or "")
        except Exception as e:
            logger.warning(f"formula transcription failed (page {page_no}): {e}")
            return False
        if not latex:
            return False
        item.text = latex
        return True

    def _render_region(
        self, fitz_doc: Any, page_no: int | None, bbox: dict
    ) -> bytes | None:
        """Render a bbox region to PNG bytes, or None on failure.

        Docling's PDF pipeline emits bottom-left-origin bboxes ((l, b, r,
        t)); PyMuPDF is top-left — flip y and expand by EXPANSION on every
        side. A TOPLEFT-origin bbox (programmatic sources) is used as-is.
        """
        if not page_no:
            return None
        try:
            page = fitz_doc[page_no - 1]
            left, bottom, right, top = bbox["l"], bbox["b"], bbox["r"], bbox["t"]
            w, h = right - left, abs(top - bottom)
            dx, dy = w * self.EXPANSION, h * self.EXPANSION
            if "TOPLEFT" in str(bbox.get("coord_origin", "")).upper():
                y0, y1 = top, bottom
            else:
                page_height = page.rect.height
                y0 = page_height - top
                y1 = page_height - bottom
            rect = fitz.Rect(left - dx, y0 - dy, right + dx, y1 + dy)
            png: bytes = page.get_pixmap(
                clip=rect, dpi=self.RENDER_DPI
            ).tobytes("png")
            return png
        except Exception as e:
            logger.warning(f"formula region render failed (page {page_no}): {e}")
            return None

    @staticmethod
    def _clean_latex(text: str) -> str:
        """Strip wrappers VLMs love to add; empty result means 'unusable'."""
        out = (text or "").strip()
        if out.startswith("```"):
            # ```latex\n...\n``` → ...
            body = out.strip("`")
            out = body[6:] if body.lower().startswith("latex") else body
            out = out.strip()
        # Longest delimiters first: "$$x$$" must not be peeled as "$" "$"
        # leaving "$x$" behind.
        for a, b in (("$$", "$$"), ("\\[", "\\]"), ("$", "$")):
            if out.startswith(a) and out.endswith(b) and len(out) >= 2 * len(a):
                out = out[len(a): -len(b)].strip()
        return out

    @staticmethod
    def _label(item: Any) -> str:
        lbl = getattr(item, "label", None)
        if lbl is None:
            return ""
        return lbl.name if hasattr(lbl, "name") else str(lbl)

    @staticmethod
    def _provenance(item: Any) -> tuple[int | None, dict[str, Any] | None]:
        """(page_no, bbox-dict) from the first prov entry.

        A missing page_no returns (None, None) → skip: rendering a prov-less
        bbox against page 1 would produce confidently wrong LaTeX (unlike
        the section walker, which can default page to 1 harmlessly).
        """
        prov = getattr(item, "prov", None)
        if not prov:
            return None, None
        p = prov[0]
        # Guard against the doubly-nested prov shape docling-core's
        # programmatic API can emit ([[ProvenanceItem]]); real parses are flat.
        if isinstance(p, (list, tuple)):
            p = p[0] if p else None
        if p is None:
            return None, None
        page_no = getattr(p, "page_no", None)
        if not page_no:
            return None, None
        b = getattr(p, "bbox", None)
        if b is None:
            return page_no, None
        return page_no, {
            "l": getattr(b, "l", None),
            "b": getattr(b, "b", None),
            "r": getattr(b, "r", None),
            "t": getattr(b, "t", None),
            # Carried so the renderer can honor TOPLEFT prov if it ever
            # appears (the PDF pipeline always emits BOTTOMLEFT).
            "coord_origin": str(getattr(b, "coord_origin", "") or ""),
        }
