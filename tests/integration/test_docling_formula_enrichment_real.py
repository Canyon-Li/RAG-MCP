"""Ticket 06 / D-039 – formula enrichment against REAL docling + REAL ollama.

Acceptance for the enrichment route (the mocked unit tests pin wiring only):
ingesting the formula-bearing fixture must leave **LaTeX in the chunk text** —
retrievable and visible to the generation side. Chain under test:

    DoclingParser(formula_enrichment=True)          real docling parse (3 pages)
      → FormulaTranscriber → ollama qwen2.5vl-3b    real VLM on GPU (~3.6 s/formula)
    → HybridDoclingChunker.split_document           real chunker + local tokenizer

Marked slow: docling layout+tableformer load makes this minutes-order on a
cold cache; formula transcription adds ~20-30 s. Skips when ollama or the
vision model is unavailable (the degradation path itself is unit-tested).
"""

import re
from pathlib import Path

import httpx
import pytest

from src.core.settings import load_settings
from src.ingestion.chunking.hybrid_docling_chunker import HybridDoclingChunker
from src.libs.parser import docling_parser
from src.libs.parser.docling_parser import DoclingParser

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(
        not docling_parser.DOCLING_AVAILABLE, reason="docling not installed"
    ),
]

FIXTURE = Path("tests/fixtures/sample_documents/formula_pages.pdf")
# LaTeX syntax markers: a backslash command, _ or ^ subscript/superscript.
_LATEX = re.compile(r"\\[a-zA-Z]+|[_^]\{|[_^][\w]")
OLLAMA = "http://localhost:11434"
VISION_MODEL = "qwen2.5vl-3b:latest"


@pytest.fixture(scope="module")
def vision_available() -> bool:
    """Ollama up AND the vision model pulled — otherwise this acceptance
    cannot run (transcriber would degrade to empty items by design)."""
    try:
        tags = httpx.get(
            f"{OLLAMA}/api/tags", timeout=5.0, trust_env=False
        ).json()
    except httpx.HTTPError:
        return False
    return any(m.get("name") == VISION_MODEL for m in tags.get("models", []))


@pytest.fixture(scope="module")
def parsed_document(tmp_path_factory, vision_available):
    """One real parse (docling + qwen2.5vl) shared by the assertions below."""
    if not vision_available:
        pytest.skip(f"ollama/{VISION_MODEL} unavailable")
    settings = load_settings()
    parser = DoclingParser(
        settings=settings,
        collection="formula_it",
        image_storage_dir=str(tmp_path_factory.mktemp("images")),
        extract_images=False,
        formula_enrichment=True,
    )
    return parser.parse(FIXTURE)


def test_formula_items_carry_latex(parsed_document):
    """Item level: enriched FORMULA items must hold LaTeX, not empty text."""
    enriched = empty = 0
    for ddoc in parsed_document.metadata["docling_documents"]:
        for item, _ in ddoc.iterate_items():
            if getattr(item.label, "name", "") != "FORMULA":
                continue
            if (getattr(item, "text", "") or "").strip():
                if _LATEX.search(item.text):
                    enriched += 1
            else:
                empty += 1
    assert enriched >= 3, (
        f"expected most of the fixture's formulas enriched, got "
        f"{enriched} enriched / {empty} empty"
    )


def test_chunk_text_contains_latex(parsed_document):
    """Acceptance: LaTeX reaches the chunk text (retrievable / generatable)."""
    chunks = HybridDoclingChunker(load_settings()).split_document(parsed_document)
    assert chunks
    with_latex = [c for c in chunks if _LATEX.search(c.text)]
    assert with_latex, "no chunk carries LaTeX — enrichment invisible downstream"
    formula_chunks = [c for c in chunks if c.metadata.get("contains_formula")]
    assert formula_chunks and all(
        _LATEX.search(c.text) for c in formula_chunks
    ), "formula-flagged chunks must carry LaTeX"


def test_enriched_latex_reaches_flat_text(parsed_document):
    """Document.text (fallback view) also carries the LaTeX."""
    assert _LATEX.search(parsed_document.text)
