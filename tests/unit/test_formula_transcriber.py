"""FormulaTranscriber tests (ticket 06 / D-039 — 自研公式补全路线).

Seam: ``FormulaTranscriber.transcribe(pdf, documents)`` — docling's layout
model already emits empty ``FORMULA`` items with provenance; the transcriber
renders each item's bbox region, asks a Vision LLM (ollama qwen2.5vl on GPU
in production) for LaTeX, and fills ``item.text`` in place. Failure leaves
the item empty (D-005 degradation — enrichment must never break a parse).

The Vision LLM is mocked (mock-injection precedent: ImageCaptioner); fitz
rendering is patched (real rendering runs in the slow integration test).
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.libs.llm.base_llm import ChatResponse
from src.libs.parser.formula_transcriber import FormulaTranscriber


class MockVisionLLM:
    """Deterministic stand-in for the Vision LLM (ImageCaptioner precedent)."""

    def __init__(self, replies=None, fail_first=0):
        self.replies = list(replies or [])
        self.fail_first = fail_first  # raise for the first N calls
        self.calls: list[str] = []  # prompt text per call

    def chat_with_image(self, text, image, **kwargs):
        if self.fail_first > 0:
            self.fail_first -= 1
            raise RuntimeError("vision llm down")
        self.calls.append(text)
        content = self.replies.pop(0) if self.replies else "x^{2}"
        return ChatResponse(content=content, model="mock")


def _formula_item(text="", page=2, bbox=(10, 20, 200, 60)):
    """A docling-like FORMULA item (bottom-left-origin prov, like the parser sees)."""
    item = MagicMock()
    lbl = MagicMock()
    lbl.name = "FORMULA"
    item.label = lbl
    item.text = text
    prov = MagicMock()
    prov.page_no = page
    if bbox is None:
        prov.bbox = None
    else:
        b = MagicMock()
        b.l, b.b, b.r, b.t = bbox  # docling origin: (l, b, r, t)
        prov.bbox = b
    item.prov = [prov]
    return item


def _text_item(text="正文段落"):
    item = MagicMock()
    lbl = MagicMock()
    lbl.name = "TEXT"
    item.label = lbl
    item.text = text
    item.prov = [MagicMock()]
    return item


def _doc(items):
    d = MagicMock()
    d.iterate_items = MagicMock(return_value=[(i, 0) for i in items])
    return d


def _make(pdf: Path = Path("dummy.pdf"), **llm_kwargs):
    settings = MagicMock()
    settings.vision_llm = SimpleNamespace(enabled=True, model="qwen2.5vl-3b")
    t = FormulaTranscriber(settings=settings, llm=MockVisionLLM(**llm_kwargs))
    return t, t.llm


def _run(t, docs, pdf=Path("dummy.pdf")):
    with patch("src.libs.parser.formula_transcriber.PYMUPDF_AVAILABLE", True), patch(
        "src.libs.parser.formula_transcriber.fitz"
    ) as mfitz:
        page = mfitz.open.return_value.__enter__.return_value.__getitem__.return_value
        page.rect.height = 800.0
        page.get_pixmap.return_value.tobytes.return_value = b"png-bytes"
        n = t.transcribe(pdf, docs)
    return n


# ---------------------------------------------------------------------------
# Construction / gating
# ---------------------------------------------------------------------------


def test_prompt_loaded_from_config():
    """The transcription prompt lives in config/prompts/ (project convention)."""
    t, _ = _make()
    assert "LaTeX" in t.prompt


def test_disabled_when_vision_llm_off(tmp_path):
    """vision_llm disabled in settings → transcriber is a no-op (gated AND).

    No llm injected here: explicit injection overrides the gate (parser unit
    tests rely on that), so the gate itself is only observable without it.
    """
    settings = MagicMock()
    settings.vision_llm = SimpleNamespace(enabled=False)
    t = FormulaTranscriber(settings=settings)
    assert t.enabled is False
    assert t.llm is None
    doc = _doc([_formula_item()])
    assert t.transcribe(tmp_path / "x.pdf", [doc]) == 0
    assert doc.iterate_items()[0][0].text == ""


def test_disabled_without_pymupdf():
    """No fitz → cannot render regions → no-op instead of crash."""
    t, llm = _make()
    doc = _doc([_formula_item()])
    with patch("src.libs.parser.formula_transcriber.PYMUPDF_AVAILABLE", False):
        assert t.transcribe(Path("x.pdf"), [doc]) == 0
    assert llm.calls == []


# ---------------------------------------------------------------------------
# Transcription behaviour
# ---------------------------------------------------------------------------


def test_fills_empty_formula_items_only():
    """Empty FORMULA items get LaTeX; text items and already-filled formulas untouched."""
    filled = _formula_item(text="already^{done}")
    items = [_formula_item(), _text_item(), filled]
    t, llm = _make(replies=["x^{2} + 1"])
    n = _run(t, [_doc(items)])

    assert n == 1
    assert llm.calls and "LaTeX" in llm.calls[0], "prompt must come from config"
    assert items[0].text == "x^{2} + 1"
    assert items[1].text == "正文段落"
    assert filled.text == "already^{done}"


def test_across_multiple_documents():
    """Page-batched parses hand over several DoclingDocuments — all covered."""
    items1, items2 = [_formula_item(page=2)], [_formula_item(page=3)]
    t, llm = _make(replies=["a", "b"])
    n = _run(t, [_doc(items1), _doc(items2)])

    assert n == 2
    assert items1[0].text == "a" and items2[0].text == "b"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("$$x^{2}$$", "x^{2}"),
        ("\\[x^{2}\\]", "x^{2}"),
        ("$x^{2}$", "x^{2}"),
        ("```latex\nx^{2}\n```", "x^{2}"),
        ("  \n x^{2} \n ", "x^{2}"),
    ],
)
def test_clean_latex_strips_wrappers(raw, expected):
    assert FormulaTranscriber._clean_latex(raw) == expected


def test_clean_empty_is_failure(tmp_path):
    """Empty/unusable output (e.g. '$$ $$') leaves the item empty, counts as not enriched."""
    t, _ = _make(replies=["$$ $$"])
    items = [_formula_item()]
    assert _run(t, [_doc(items)]) == 0
    assert items[0].text == ""


def test_llm_failure_degrades_and_continues(caplog):
    """One failing call must not abort the batch (D-005): log + leave empty + keep going."""
    items = [_formula_item(page=2), _formula_item(page=3)]
    t, _ = _make(replies=["b"], fail_first=1)
    with caplog.at_level(logging.WARNING, logger="src.libs.parser.formula_transcriber"):
        n = _run(t, [_doc(items)])

    assert n == 1
    assert items[0].text == "", "failed item stays empty"
    assert items[1].text == "b"
    assert any("formula transcription failed" in r.message for r in caplog.records)


def test_missing_bbox_or_provenance_skipped():
    """FORMULA items without usable provenance can't be rendered — skipped silently."""
    no_bbox = _formula_item(bbox=None)
    no_prov = _formula_item()
    no_prov.prov = []
    no_page = _formula_item()
    no_page.prov[0].page_no = None  # prov exists but points nowhere
    t, llm = _make(replies=["x"])
    n = _run(t, [_doc([no_bbox, no_prov, no_page])])

    assert n == 0
    assert llm.calls == []
    assert no_bbox.text == "" and no_prov.text == "" and no_page.text == ""
