"""Ticket 06 / D-039: ``ingestion.parser.formula_enrichment`` config switch.

Off by default (pre-ticket behaviour: empty formula items stay dropped); on
= docling parser transcribes them to LaTeX via the Vision LLM. The switch is
part of the parse-cache version stamp, so flipping it invalidates old cache
entries (pinned in test_docling_formula_enrichment.py).
"""

import re
from pathlib import Path

import pytest

from src.core.settings import SettingsError, load_settings

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "config" / "settings.yaml"

_PARSER_LINE = re.compile(r"^    formula_enrichment: .*\n", flags=re.M)


def _copy_yaml(tmp_path: Path, *, drop: bool = False, line: str | None = None) -> Path:
    text = _SRC.read_text(encoding="utf-8")
    if drop:
        text = _PARSER_LINE.sub("", text)
    elif line is not None:
        text = _PARSER_LINE.sub(line, text, count=1)
    out = tmp_path / "settings.yaml"
    out.write_text(text, encoding="utf-8")
    return out


def test_shipped_config_enables_formula_enrichment():
    """Pin the shipped config: switch on (ticket 06 acceptance)."""
    s = load_settings()
    assert s.ingestion.parser.formula_enrichment is True


def test_absent_switch_means_off(tmp_path):
    s = load_settings(_copy_yaml(tmp_path, drop=True))
    assert s.ingestion.parser.formula_enrichment is False


def test_explicit_false_parses(tmp_path):
    s = load_settings(_copy_yaml(tmp_path, line="    formula_enrichment: false\n"))
    assert s.ingestion.parser.formula_enrichment is False


def test_non_bool_rejected(tmp_path):
    with pytest.raises(SettingsError):
        load_settings(_copy_yaml(tmp_path, line="    formula_enrichment: yes-please\n"))
