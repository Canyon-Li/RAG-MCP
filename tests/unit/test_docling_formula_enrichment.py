"""DoclingParser ↔ FormulaTranscriber wiring tests (ticket 06 / D-039).

Seam: ``DoclingParser.parse`` — fresh parses run the transcriber over the
converted documents BEFORE ``_walk_documents`` (so enriched LaTeX reaches
sections / Document.text) and before the parse-cache save (so LaTeX is
frozen into the lossless JSON, D-036 — replay never re-transcribes).
Replay, stamp, and degradation are pinned here; the transcriber itself has
its own unit tests, real-render+real-ollama runs in the slow integration.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
)

from src.libs.parser.docling_parser import DoclingParser

pytest.importorskip("docling_core.types.doc", reason="docling-core not installed")

SPECS = [
    # label, text, page, bbox (docling bottom-left origin)
    {"label": DocItemLabel.TEXT, "text": "The S-box maps bytes as follows:",
     "page": 1, "bbox": (50, 700, 500, 720)},
    {"label": DocItemLabel.FORMULA, "text": "", "page": 1,
     "bbox": (50, 620, 400, 700)},
]


def _make_doc(specs) -> DoclingDocument:
    """Programmatic DoclingDocument (no converter run).

    Real documents, not mocks: the cache-save/replay leg must serialize for
    actual (``save_as_json``/``load_from_json``), which MagicMock cannot.
    """
    d = DoclingDocument(name="t")
    for s in specs:
        l, b, r, t = s["bbox"]
        prov = ProvenanceItem(
            page_no=s["page"],
            bbox=BoundingBox(l=l, t=t, r=r, b=b, coord_origin=CoordOrigin.BOTTOMLEFT),
            charspan=(0, 0),
        )
        # NOTE: docling-core's add_* methods take a BARE ProvenanceItem —
        # wrapping it in a list yields a doubly-nested prov that
        # load_from_json then rejects (API quirk, verified empirically).
        d.add_text(label=s["label"], text=s["text"], prov=prov)
    return d


class _ConverterProbe:
    """DocumentConverter stand-in returning spec documents (OCR-test pattern)."""

    instances = 0

    def __init__(self):
        type(self).instances += 1
        self.kwargs = None

    def convert(self, *args, **kwargs):
        result = MagicMock()
        result.document = _make_doc(SPECS)
        return result

    @classmethod
    def factory(cls):
        def _f(**kwargs):
            p = cls()
            p.kwargs = kwargs
            return p
        return _f


def _pdf(tmp_path: Path) -> Path:
    p = tmp_path / "doc.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 100), "formula bearing page", fontsize=11)
    pdf.save(str(p))
    return p


def _formula_items(doc):
    return [
        it for it, _ in doc.metadata["docling_documents"][0].iterate_items()
        if it.label.name == "FORMULA"
    ]


# ---------------------------------------------------------------------------
# Fresh parse: transcriber runs before section walk and cache save
# ---------------------------------------------------------------------------


def test_fresh_parse_enriches_formula_items(tmp_path):
    """formula_enrichment=True → empty FORMULA items get LaTeX and it reaches
    Document.text / sections (before _walk_documents collects them)."""
    calls = {"transcribe": 0}

    def _fake_transcribe(self, pdf_path, documents):
        calls["transcribe"] += 1
        assert isinstance(pdf_path, (str, Path))
        for ddoc in documents:
            for item, _ in ddoc.iterate_items():
                if item.label.name == "FORMULA" and not item.text.strip():
                    item.text = "S ( a ) = A a ^ { - 1 } \\oplus c"
        return 1

    with patch(
        "src.libs.parser.docling_parser.DocumentConverter", _ConverterProbe.factory()
    ), patch(
        "src.libs.llm.llm_factory.LLMFactory.create_vision_llm",
        return_value=MagicMock(),
    ), patch(
        "src.libs.parser.formula_transcriber.FormulaTranscriber.transcribe",
        _fake_transcribe,
    ):
        parser = DoclingParser(
            MagicMock(), collection="t",
            image_storage_dir=str(tmp_path / "images"),
            extract_images=False, formula_enrichment=True,
        )
        doc = parser.parse(_pdf(tmp_path))

    assert calls["transcribe"] == 1
    formulas = _formula_items(doc)
    assert len(formulas) == 1
    assert "\\oplus" in formulas[0].text
    assert "\\oplus" in doc.text, "enriched LaTeX must reach the flat text"
    assert any("\\oplus" in (s.get("text") or "") for s in doc.metadata["sections"])


def test_enrichment_failure_never_breaks_parse(tmp_path):
    """Transcriber crashing must not fail the parse (D-005)."""

    def _boom(self, pdf_path, documents):
        raise RuntimeError("vision llm exploded")

    with patch(
        "src.libs.parser.docling_parser.DocumentConverter", _ConverterProbe.factory()
    ), patch(
        "src.libs.parser.formula_transcriber.FormulaTranscriber.transcribe", _boom
    ):
        parser = DoclingParser(
            MagicMock(), collection="t",
            image_storage_dir=str(tmp_path / "images"),
            extract_images=False, formula_enrichment=True,
        )
        doc = parser.parse(_pdf(tmp_path))

    assert doc.metadata["sections"], "parse survives transcription failure"
    assert _formula_items(doc)[0].text == ""


def test_off_by_default_no_transcription(tmp_path):
    """formula_enrichment absent → transcriber never constructed."""
    with patch(
        "src.libs.parser.docling_parser.DocumentConverter", _ConverterProbe.factory()
    ), patch(
        "src.libs.parser.formula_transcriber.FormulaTranscriber"
    ) as mock_cls:
        parser = DoclingParser(
            MagicMock(), collection="t",
            image_storage_dir=str(tmp_path / "images"),
            extract_images=False,
        )
        parser.parse(_pdf(tmp_path))
        mock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Cache interplay: replay never re-transcribes; stamp tracks the switch
# ---------------------------------------------------------------------------


def test_replay_does_not_retranscribe(tmp_path):
    """Cached parse replays the frozen LaTeX with zero transcriber calls (D-036)."""
    calls = {"transcribe": 0}

    def _fake_transcribe(self, pdf_path, documents):
        calls["transcribe"] += 1
        for ddoc in documents:
            for item, _ in ddoc.iterate_items():
                if item.label.name == "FORMULA" and not item.text.strip():
                    item.text = "x ^ { 2 }"
        return 1

    cache = tmp_path / "parsed"
    common = dict(
        collection="t",
        image_storage_dir=str(tmp_path / "images"),
        extract_images=False,
        formula_enrichment=True,
        parse_cache_dir=str(cache),
    )
    # One PDF for both parses — fitz embeds creation timestamps, so a
    # re-generated file hashes differently and would miss the cache.
    pdf = _pdf(tmp_path)
    with patch(
        "src.libs.parser.docling_parser.DocumentConverter", _ConverterProbe.factory()
    ), patch(
        "src.libs.llm.llm_factory.LLMFactory.create_vision_llm",
        return_value=MagicMock(),
    ), patch(
        "src.libs.parser.formula_transcriber.FormulaTranscriber.transcribe",
        _fake_transcribe,
    ):
        d1 = DoclingParser(MagicMock(), **common).parse(pdf)
        assert calls["transcribe"] == 1
        d2 = DoclingParser(MagicMock(), **common).parse(pdf)

    assert calls["transcribe"] == 1, "replay must not re-transcribe"
    assert _formula_items(d2)[0].text == "x ^ { 2 }", "LaTeX replayed from cache"
    assert d2.metadata["sections"] == d1.metadata["sections"]


def test_version_stamp_changes_with_formula_enrichment(tmp_path):
    """The switch changes the saved JSON → must change the stamp (D-036)."""
    common = dict(
        collection="t",
        image_storage_dir=str(tmp_path / "images"),
        extract_images=False,
    )
    s_off = DoclingParser(MagicMock(), **common)._version_stamp()
    s_on = DoclingParser(MagicMock(), formula_enrichment=True, **common)._version_stamp()
    assert s_off != s_on


def test_version_stamp_uses_effective_vision_gate(tmp_path):
    """Review finding: the stamp must record the EFFECTIVE gate, not the raw
    switch — formula_enrichment on + vision_llm off enriches nothing, and
    stamping that as "on" would freeze empty formulas into the cache and
    replay them stale after vision_llm is later enabled."""
    from types import SimpleNamespace

    common = dict(
        collection="t",
        image_storage_dir=str(tmp_path / "images"),
        extract_images=False,
    )
    settings = MagicMock()
    settings.vision_llm = SimpleNamespace(enabled=False)
    s_all_off = DoclingParser(settings, **common)._version_stamp()
    s_gate_off = DoclingParser(
        settings, formula_enrichment=True, **common
    )._version_stamp()
    assert s_gate_off == s_all_off, "gated-off enrichment must stamp as off"

    settings.vision_llm = SimpleNamespace(enabled=True)
    s_on = DoclingParser(
        settings, formula_enrichment=True, **common
    )._version_stamp()
    assert s_on != s_all_off
