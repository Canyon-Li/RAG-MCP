"""DoclingParser parse-cache replay tests (D-036, ticket 02).

Seam: ``DoclingParser.parse`` — DocumentConverter is mocked (counting probe),
but the cache roundtrip goes through REAL files on disk: mock documents
serialize their item specs to JSON on save_as_json, and the patched
``DoclingDocument.load_from_json`` rebuilds mock items from whatever the file
actually contains. Docling's own JSON fidelity was proven by spike 01
(.scratch/ingestion-revamp); these tests pin the parser wiring: second
--force-style parse → zero DocumentConverter calls, byte-equivalent sections,
stamp tampering → real re-parse, cache off → pre-D-036 behaviour.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.libs.parser.docling_parser import DoclingParser

# ---------------------------------------------------------------------------
# Mock plumbing (same shapes as test_docling_parser.py, plus a real-file
# save/load roundtrip so cache entries on disk are genuine JSON)
# ---------------------------------------------------------------------------


def _spec_item(label, text=None, page=1, table_md=None, bbox=None):
    """Plain-data item spec — serializable, rebuilds a mock Docling item."""
    return {"label": label, "text": text, "page": page,
            "table_md": table_md, "bbox": bbox}


def _spec_to_item(spec):
    item = MagicMock()
    lbl = MagicMock()
    lbl.name = spec["label"]
    item.label = lbl
    item.text = spec["text"]
    prov = MagicMock()
    prov.page_no = spec["page"]
    bbox = spec["bbox"]
    if bbox:
        prov.bbox = MagicMock(l=bbox[0], t=bbox[1], r=bbox[2], b=bbox[3])
    else:
        prov.bbox = None
    item.prov = [prov]
    if spec["table_md"] is not None:
        item.export_to_markdown = MagicMock(return_value=spec["table_md"])
    return item


def _make_doc(specs):
    """Mock DoclingDocument: iterate_items yields spec items; save_as_json
    writes the specs to the requested file as real JSON."""
    ddoc = MagicMock()
    ddoc.iterate_items = MagicMock(
        return_value=[(_spec_to_item(s), 0) for s in specs]
    )

    def _save(filename):
        Path(filename).write_text(
            json.dumps({"specs": specs}), encoding="utf-8")

    ddoc.save_as_json = MagicMock(side_effect=_save)
    return ddoc


def _patched_load_from_json(filename):
    """DoclingDocument.load_from_json stand-in: rebuild a mock doc from the
    JSON actually on disk (raises on corrupt files, like the real one)."""
    data = json.loads(Path(filename).read_text(encoding="utf-8"))
    ddoc = MagicMock()
    ddoc.iterate_items = MagicMock(
        return_value=[(_spec_to_item(s), 0) for s in data["specs"]]
    )
    return ddoc


class _ConverterProbe:
    """DocumentConverter stand-in that counts constructions and converts."""

    def __init__(self, specs_by_batch):
        self.specs_by_batch = specs_by_batch
        self.constructed = 0
        self.converted = 0

    def factory(self):
        probe = self

        class _Converter:
            def __init__(self):
                probe.constructed += 1

            def convert(self, *args, **kwargs):
                probe.converted += 1
                idx = min(probe.converted - 1, len(probe.specs_by_batch) - 1)
                result = MagicMock()
                result.document = _make_doc(probe.specs_by_batch[idx])
                return result

        return _Converter


SPECS = [
    {"label": "SECTION_HEADER", "text": "第一章 架构", "page": 1,
     "table_md": None, "bbox": None},
    {"label": "TEXT", "text": "正文段落内容", "page": 1,
     "table_md": None, "bbox": (10, 20, 30, 40)},
    {"label": "TABLE", "text": None, "page": 2,
     "table_md": "| 列1 | 列2 |\n|---|---|\n| a | b |", "bbox": None},
]


@pytest.fixture
def settings():
    return MagicMock()


@pytest.fixture
def fake_pdf(tmp_path):
    p = tmp_path / "fake.pdf"
    p.write_bytes(b"%PDF-1.4 fake content")
    return p


def _sections_of(doc):
    return doc.metadata["sections"]


def _first_parse(fake_pdf, tmp_path, probe, **parser_kwargs):
    """Fresh parse with a counting converter probe; cache under tmp_path."""
    with patch("src.libs.parser.docling_parser.DocumentConverter", probe.factory()), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False):
        parser = DoclingParser(
            MagicMock(), collection="t",
            image_storage_dir=str(tmp_path / "images"),
            extract_images=False, **parser_kwargs,
        )
        doc = parser.parse(fake_pdf)
    return parser, doc


# ---------------------------------------------------------------------------
# Acceptance: consecutive --force re-ingests → second parse replays
# ---------------------------------------------------------------------------


def test_second_parse_replays_with_zero_converter_calls(settings, fake_pdf, tmp_path):
    """Same file parsed twice with the cache on: the second parse makes ZERO
    DocumentConverter constructions/calls and yields identical sections."""
    cache_dir = tmp_path / "parsed"
    probe = _ConverterProbe([SPECS])

    _, doc1 = _first_parse(
        fake_pdf, tmp_path, probe, parse_cache_dir=str(cache_dir))
    assert probe.constructed == 1 and probe.converted == 1  # real parse happened

    # Second parse = a new process: fresh probe whose converter explodes if
    # touched — replay must never construct it.
    probe2 = _ConverterProbe([SPECS])
    with patch("src.libs.parser.docling_parser.DocumentConverter",
               probe2.factory()), \
         patch("src.libs.parser.docling_parser.DoclingDocument") as mock_dd, \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False):
        mock_dd.load_from_json = MagicMock(side_effect=_patched_load_from_json)
        parser2 = DoclingParser(
            settings, collection="t",
            image_storage_dir=str(tmp_path / "images"),
            extract_images=False, parse_cache_dir=str(cache_dir),
        )
        doc2 = parser2.parse(fake_pdf)

    assert probe2.constructed == 0, "replay must not build a DocumentConverter"
    assert probe2.converted == 0
    mock_dd.load_from_json.assert_called_once()
    assert parser2.last_parse_replayed is True  # trace side-channel (D-036)

    # Chunk-input equivalence: sections identical to the real parse.
    assert _sections_of(doc2) == _sections_of(doc1)
    assert doc2.text == doc1.text
    assert doc2.id == doc1.id  # doc_{sha[:16]} — same file hash


def test_cache_entry_layout_on_disk(settings, fake_pdf, tmp_path):
    """First parse writes manifest + batch JSON under {cache_dir}/{sha256}."""
    cache_dir = tmp_path / "parsed"
    probe = _ConverterProbe([SPECS])
    _first_parse(fake_pdf, tmp_path, probe, parse_cache_dir=str(cache_dir))

    import hashlib
    sha = hashlib.sha256(fake_pdf.read_bytes()).hexdigest()
    entry = cache_dir / sha
    assert (entry / "manifest.json").is_file()
    assert (entry / "batch_001.json").is_file()
    manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["batches"] == ["batch_001.json"]
    # Saved batch JSON really carries the lossless payload (spec roundtrip;
    # compare against the JSON-serialized form — tuples become lists in JSON).
    payload = json.loads((entry / "batch_001.json").read_text(encoding="utf-8"))
    assert payload["specs"] == json.loads(json.dumps(SPECS))


# ---------------------------------------------------------------------------
# Acceptance: version-stamp mismatch → automatic re-parse
# ---------------------------------------------------------------------------


def test_tampered_stamp_triggers_real_reparse(settings, fake_pdf, tmp_path):
    cache_dir = tmp_path / "parsed"
    probe = _ConverterProbe([SPECS])
    _first_parse(fake_pdf, tmp_path, probe, parse_cache_dir=str(cache_dir))

    # Tamper the manifest stamp — simulates "mapping logic was upgraded".
    import hashlib
    sha = hashlib.sha256(fake_pdf.read_bytes()).hexdigest()
    manifest_path = cache_dir / sha / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version_stamp"] = "deadbeefdeadbeef"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    probe2 = _ConverterProbe([SPECS])
    with patch("src.libs.parser.docling_parser.DocumentConverter",
               probe2.factory()), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False):
        parser2 = DoclingParser(
            settings, collection="t",
            image_storage_dir=str(tmp_path / "images"),
            extract_images=False, parse_cache_dir=str(cache_dir),
        )
        doc2 = parser2.parse(fake_pdf)

    assert probe2.converted == 1, "stale stamp must trigger a real re-parse"

    # And the re-parse refreshes the entry with the current stamp.
    refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert refreshed["version_stamp"] != "deadbeefdeadbeef"
    with patch("src.libs.parser.docling_parser.DocumentConverter",
               _ConverterProbe([SPECS]).factory()), \
         patch("src.libs.parser.docling_parser.DoclingDocument") as mock_dd, \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False):
        mock_dd.load_from_json = MagicMock(side_effect=_patched_load_from_json)
        parser3 = DoclingParser(
            settings, collection="t",
            image_storage_dir=str(tmp_path / "images"),
            extract_images=False, parse_cache_dir=str(cache_dir),
        )
        doc3 = parser3.parse(fake_pdf)
    assert _sections_of(doc3) == _sections_of(doc2)


def test_corrupt_batch_json_degrades_to_real_parse(settings, fake_pdf, tmp_path):
    """A torn batch file (manifest fine, stamp fine) must not break ingest —
    load fails → fall through to a real parse."""
    cache_dir = tmp_path / "parsed"
    probe = _ConverterProbe([SPECS])
    _, doc1 = _first_parse(
        fake_pdf, tmp_path, probe, parse_cache_dir=str(cache_dir))

    import hashlib
    sha = hashlib.sha256(fake_pdf.read_bytes()).hexdigest()
    (cache_dir / sha / "batch_001.json").write_text("{torn", encoding="utf-8")

    probe2 = _ConverterProbe([SPECS])
    with patch("src.libs.parser.docling_parser.DocumentConverter",
               probe2.factory()), \
         patch("src.libs.parser.docling_parser.DoclingDocument") as mock_dd, \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False):
        mock_dd.load_from_json = MagicMock(side_effect=_patched_load_from_json)
        parser2 = DoclingParser(
            settings, collection="t",
            image_storage_dir=str(tmp_path / "images"),
            extract_images=False, parse_cache_dir=str(cache_dir),
        )
        doc2 = parser2.parse(fake_pdf)

    assert probe2.converted == 1, "corrupt cache must fall back to real parse"
    assert _sections_of(doc2) == _sections_of(doc1)


# ---------------------------------------------------------------------------
# Acceptance: cache off → pre-D-036 behaviour
# ---------------------------------------------------------------------------


def test_cache_off_parses_for_real_every_time(settings, fake_pdf, tmp_path):
    """No parse_cache_dir kwarg (factory omits it when disabled/unset):
    every parse converts for real, nothing is written anywhere."""
    probe = _ConverterProbe([SPECS])
    _, doc1 = _first_parse(fake_pdf, tmp_path, probe)

    probe2 = _ConverterProbe([SPECS])
    with patch("src.libs.parser.docling_parser.DocumentConverter",
               probe2.factory()), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False):
        parser2 = DoclingParser(
            settings, collection="t",
            image_storage_dir=str(tmp_path / "images"),
            extract_images=False,
        )
        doc2 = parser2.parse(fake_pdf)

    assert probe2.converted == 1
    assert parser2.last_parse_replayed is False  # real parse, no replay flag
    assert _sections_of(doc2) == _sections_of(doc1)
    assert not (tmp_path / "parsed").exists()


# ---------------------------------------------------------------------------
# Batching + figures on the replay path
# ---------------------------------------------------------------------------


def test_replay_walks_all_cached_batches_in_order(settings, fake_pdf, tmp_path):
    """A page-batched parse caches one JSON per batch; replay concatenates
    them in order without touching the converter."""
    cache_dir = tmp_path / "parsed"
    batch_specs = [
        [_spec_item("TEXT", text=f"page {p}", page=p) for p in (1, 2)],
        [_spec_item("TEXT", text=f"page {p}", page=p) for p in (3, 4)],
    ]
    probe = _ConverterProbe(batch_specs)
    with patch("src.libs.parser.docling_parser.DocumentConverter",
               probe.factory()), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False), \
         patch.object(DoclingParser, "_page_count", return_value=4):
        parser = DoclingParser(
            settings, collection="t",
            image_storage_dir=str(tmp_path / "images"),
            extract_images=False, parse_cache_dir=str(cache_dir),
            page_batch_size=2,
        )
        doc1 = parser.parse(fake_pdf)

    texts1 = [s["text"] for s in _sections_of(doc1)]
    assert texts1 == ["page 1", "page 2", "page 3", "page 4"]

    probe2 = _ConverterProbe(batch_specs)
    with patch("src.libs.parser.docling_parser.DocumentConverter",
               probe2.factory()), \
         patch("src.libs.parser.docling_parser.DoclingDocument") as mock_dd, \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False):
        mock_dd.load_from_json = MagicMock(side_effect=_patched_load_from_json)
        parser2 = DoclingParser(
            settings, collection="t",
            image_storage_dir=str(tmp_path / "images"),
            extract_images=False, parse_cache_dir=str(cache_dir),
            page_batch_size=2,
        )
        doc2 = parser2.parse(fake_pdf)

    assert probe2.converted == 0
    texts2 = [s["text"] for s in _sections_of(doc2)]
    assert texts2 == texts1


def test_replay_renders_figures_with_identical_ids(settings, fake_pdf, tmp_path):
    """Replay re-renders figure regions through PyMuPDF (no docling) and
    emits the same deterministic image ids as the first parse."""
    specs = [
        _spec_item("TEXT", text="正文段落", page=1),
        _spec_item("PICTURE", page=2, bbox=(50, 50, 550, 400)),
    ]
    cache_dir = tmp_path / "parsed"
    probe = _ConverterProbe([specs])

    pix = MagicMock()
    pix.width = 1000
    pix.height = 700
    page = MagicMock()
    page.get_pixmap = MagicMock(return_value=pix)
    fitz_doc = MagicMock()
    fitz_doc.__getitem__.return_value = page

    with patch("src.libs.parser.docling_parser.DocumentConverter",
               probe.factory()), \
         patch("src.libs.parser.docling_parser.fitz.open",
               return_value=fitz_doc), \
         patch("src.libs.parser.docling_parser.fitz.Rect"), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", True):
        parser = DoclingParser(
            settings, collection="t",
            image_storage_dir=str(tmp_path / "images"),
            extract_images=True, parse_cache_dir=str(cache_dir),
        )
        doc1 = parser.parse(fake_pdf)

    images1 = doc1.metadata.get("images", [])
    assert len(images1) == 1

    # Replay: converter must stay untouched, figures re-render, ids identical.
    probe2 = _ConverterProbe([specs])
    with patch("src.libs.parser.docling_parser.DocumentConverter",
               probe2.factory()), \
         patch("src.libs.parser.docling_parser.DoclingDocument") as mock_dd, \
         patch("src.libs.parser.docling_parser.fitz.open",
               return_value=fitz_doc), \
         patch("src.libs.parser.docling_parser.fitz.Rect"), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", True):
        mock_dd.load_from_json = MagicMock(side_effect=_patched_load_from_json)
        parser2 = DoclingParser(
            settings, collection="t",
            image_storage_dir=str(tmp_path / "images"),
            extract_images=True, parse_cache_dir=str(cache_dir),
        )
        doc2 = parser2.parse(fake_pdf)

    assert probe2.converted == 0
    page.get_pixmap.assert_called()  # replay DID re-render the region
    assert doc2.metadata.get("images", []) == images1
    fig_sections2 = [s for s in _sections_of(doc2) if s["type"] == "figure"]
    assert [s["text"] for s in fig_sections2] == [
        f"[IMAGE: {images1[0]['id']}]"
    ]


def test_partial_batch_failure_writes_no_cache(settings, fake_pdf, tmp_path):
    """A partially-failed convert (some batches raised) must not be frozen
    into the cache — only complete parses are cacheable."""
    cache_dir = tmp_path / "parsed"
    probe = _ConverterProbe([[_spec_item("TEXT", text="page 1", page=1)]])

    boom = MagicMock()
    boom.convert.side_effect = RuntimeError("bad_alloc on batch 2")
    converters = iter([probe.factory()(), boom])

    with patch("src.libs.parser.docling_parser.DocumentConverter",
               side_effect=lambda: next(converters)), \
         patch("src.libs.parser.docling_parser.PYMUPDF_AVAILABLE", False), \
         patch.object(DoclingParser, "_page_count", return_value=2):
        parser = DoclingParser(
            settings, collection="t",
            image_storage_dir=str(tmp_path / "images"),
            extract_images=False, parse_cache_dir=str(cache_dir),
            page_batch_size=1,
        )
        doc = parser.parse(fake_pdf)

    # Earlier batch survives (existing behaviour) …
    assert [s["text"] for s in _sections_of(doc)] == ["page 1"]
    assert doc.metadata.get("degraded") is not True
    # … but nothing was cached.
    assert not cache_dir.exists() or not any(cache_dir.rglob("manifest.json"))
