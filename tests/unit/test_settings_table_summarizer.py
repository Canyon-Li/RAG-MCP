"""Ticket 05: ``ingestion.table_summarizer`` config block — parsing + kill-switch.

The block configures the table-chunk summary transform (independent
provider/model so the summarizer can point at deepseek-flash while the main
llm block goes anywhere else, or be switched back to a local model). An
absent block means the transform is off — old configs keep working.
"""

import re
from pathlib import Path

import pytest

from src.core.settings import SettingsError, load_settings

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "config" / "settings.yaml"

_BLOCK = re.compile(r"^  table_summarizer:\n(?:    .*\n)+", flags=re.M)


def _copy_yaml(tmp_path: Path, *, drop_block: bool = False, block: str | None = None) -> Path:
    """Copy the real settings.yaml, replacing (or dropping) the block."""
    text = _SRC.read_text(encoding="utf-8")
    if drop_block:
        text = _BLOCK.sub("", text)
    elif block is not None:
        text = _BLOCK.sub(block, text, count=1)
    out = tmp_path / "settings.yaml"
    out.write_text(text, encoding="utf-8")
    return out


def test_block_landed():
    """Pin the shipped config: enabled, deepseek-flash, 120-token budget."""
    ts = load_settings().ingestion.table_summarizer
    assert ts is not None
    assert ts.enabled is True
    assert ts.provider == "deepseek"
    assert ts.model == "deepseek-flash"
    assert ts.max_summary_tokens == 120


def test_absent_block_means_off(tmp_path):
    """Kill-switch base case: no block → None → transform is a no-op."""
    s = load_settings(_copy_yaml(tmp_path, drop_block=True))
    assert s.ingestion.table_summarizer is None


def test_enabled_false_parses(tmp_path):
    block = (
        "  table_summarizer:\n"
        "    enabled: false\n"
    )
    ts = load_settings(_copy_yaml(tmp_path, block=block)).ingestion.table_summarizer
    assert ts.enabled is False


def test_switch_to_local_model_parses(tmp_path):
    """One-block switch back to a local ollama model (fallback route)."""
    block = (
        "  table_summarizer:\n"
        "    enabled: true\n"
        "    provider: \"ollama\"\n"
        "    model: \"granite4.1:8b\"\n"
        "    base_url: \"http://localhost:11434\"\n"
    )
    ts = load_settings(_copy_yaml(tmp_path, block=block)).ingestion.table_summarizer
    assert ts.provider == "ollama"
    assert ts.model == "granite4.1:8b"
    assert ts.base_url == "http://localhost:11434"


def test_rejects_non_bool_enabled(tmp_path):
    block = (
        "  table_summarizer:\n"
        "    enabled: \"yes\"\n"
    )
    with pytest.raises(SettingsError):
        load_settings(_copy_yaml(tmp_path, block=block))


def test_rejects_non_int_budget(tmp_path):
    block = (
        "  table_summarizer:\n"
        "    provider: \"deepseek\"\n"
        "    max_summary_tokens: \"120\"\n"
    )
    with pytest.raises(SettingsError):
        load_settings(_copy_yaml(tmp_path, block=block))


def test_rejects_non_positive_budget(tmp_path):
    block = (
        "  table_summarizer:\n"
        "    max_summary_tokens: 0\n"
    )
    with pytest.raises(SettingsError):
        load_settings(_copy_yaml(tmp_path, block=block))
