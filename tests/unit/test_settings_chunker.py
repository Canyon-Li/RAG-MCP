"""Ticket 04 (D-037): ``ingestion.chunker`` config block — parsing + kill-switch.

The block selects the ingestion splitting strategy: ``hybrid_docling`` routes
docling-parsed documents through docling's HybridChunker adapter; ``recursive``
keeps the pre-D-037 RecursiveCharacterTextSplitter path (the fallback route,
switchable with one line). An absent block means the old behaviour.
"""

import re
from pathlib import Path

import pytest

from src.core.settings import SettingsError, load_settings

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "config" / "settings.yaml"

_CHUNKER_BLOCK = re.compile(r"^  chunker:\n(?:    .*\n)+", flags=re.M)


def _copy_yaml(tmp_path: Path, *, drop_block: bool = False, block: str | None = None) -> Path:
    """Copy the real settings.yaml, replacing (or dropping) the chunker block."""
    text = _SRC.read_text(encoding="utf-8")
    if drop_block:
        text = _CHUNKER_BLOCK.sub("", text)
    elif block is not None:
        text = _CHUNKER_BLOCK.sub(block, text, count=1)
    out = tmp_path / "settings.yaml"
    out.write_text(text, encoding="utf-8")
    return out


def test_chunker_block_landed():
    """Pin the shipped config: hybrid_docling, max_tokens 1000, merge_peers on."""
    s = load_settings()
    assert s.ingestion.chunker is not None
    assert s.ingestion.chunker.provider == "hybrid_docling"
    assert s.ingestion.chunker.max_tokens == 1000
    assert s.ingestion.chunker.merge_peers is True
    # Local tokenizer directory — the whole point is zero HF network access.
    assert s.ingestion.chunker.tokenizer_path


def test_chunker_absent_block_keeps_recursive(tmp_path):
    """Kill-switch base case: no block → None → pipeline uses the old path."""
    s = load_settings(_copy_yaml(tmp_path, drop_block=True))
    assert s.ingestion.chunker is None


def test_chunker_switch_back_to_recursive(tmp_path):
    """One-line switch to the fallback route parses."""
    block = (
        "  chunker:\n"
        "    provider: \"recursive\"\n"
    )
    s = load_settings(_copy_yaml(tmp_path, block=block))
    assert s.ingestion.chunker.provider == "recursive"


def test_chunker_rejects_unknown_provider(tmp_path):
    block = (
        "  chunker:\n"
        "    provider: \"semanticsplit\"\n"
    )
    with pytest.raises(SettingsError):
        load_settings(_copy_yaml(tmp_path, block=block))


def test_hybrid_requires_tokenizer_path(tmp_path):
    """hybrid_docling without a tokenizer dir would silently fall back to the
    HF-network default tokenizer — fail fast instead (no-HF rule, ticket 04)."""
    block = (
        "  chunker:\n"
        "    provider: \"hybrid_docling\"\n"
        "    max_tokens: 1000\n"
        "    merge_peers: true\n"
    )
    with pytest.raises(SettingsError):
        load_settings(_copy_yaml(tmp_path, block=block))
