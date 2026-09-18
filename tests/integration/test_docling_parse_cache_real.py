"""D-036 / ticket 02 – parse-cache replay against REAL docling (slow).

Complements the mocked unit tests (test_docling_parse_cache.py): here the
full roundtrip runs for real — DocumentConverter parses simple.pdf,
``save_as_json`` writes the lossless JSON, and the second parse replays via
``DoclingDocument.load_from_json`` with ZERO converter constructions and
byte-identical sections (the production-side proof of the spike-01 chain).

Marked slow: docling model load makes this minutes-order on a cold cache.
No LLM/embedding services are touched (parser seam only).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.libs.parser import docling_parser
from src.libs.parser.docling_parser import DoclingParser

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(
        not docling_parser.DOCLING_AVAILABLE, reason="docling not installed"
    ),
]

FIXTURE = Path("tests/fixtures/sample_documents/simple.pdf")


def _parse_with_probe(cache_dir: Path, collection_dir: Path, counter: dict):
    """One full parse; DocumentConverter constructions are counted."""
    from docling.document_converter import DocumentConverter

    def _counting_factory(**kwargs):
        # Forwards ctor kwargs (format_options from ticket 03's OCR wiring)
        # so the counter wraps the real constructor faithfully.
        counter["constructed"] = counter.get("constructed", 0) + 1
        return DocumentConverter(**kwargs)

    with patch(
        "src.libs.parser.docling_parser.DocumentConverter", _counting_factory
    ):
        parser = DoclingParser(
            None, collection="t",
            image_storage_dir=str(collection_dir),
            extract_images=False, parse_cache_dir=str(cache_dir),
        )
        return parser.parse(FIXTURE)


def test_real_docling_replay_is_equivalent_and_converter_free(tmp_path):
    assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"
    cache_dir = tmp_path / "parsed"

    counter1: dict = {}
    doc1 = _parse_with_probe(cache_dir, tmp_path / "images1", counter1)
    assert counter1["constructed"] >= 1, "first parse must run docling for real"
    assert doc1.metadata["sections"], "fixture must yield sections"

    # Entry on disk: manifest + lossless docling JSON (real save_as_json).
    entries = list(cache_dir.glob("*/manifest.json"))
    assert len(entries) == 1
    batch_jsons = list(entries[0].parent.glob("batch_*.json"))
    assert len(batch_jsons) == 1

    # Second parse (fresh parser = new process): zero converters, replay only.
    counter2: dict = {}
    doc2 = _parse_with_probe(cache_dir, tmp_path / "images2", counter2)

    assert counter2.get("constructed", 0) == 0, (
        "replay must not construct a DocumentConverter"
    )
    # Byte-equivalent parse output — same sections, same flat text, same id.
    assert doc2.metadata["sections"] == doc1.metadata["sections"]
    assert doc2.text == doc1.text
    assert doc2.id == doc1.id
