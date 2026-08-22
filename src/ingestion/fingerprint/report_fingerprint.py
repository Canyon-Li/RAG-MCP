"""Report fingerprinting for version management (D-031, L1/L2 of the design).

A file-level SHA256 fails to recognise the *same* research report arriving
through different distribution channels: platforms (慧博-style) stamp a
watermark page footer and re-save the PDF, so every copy has different bytes.
The business identity of a report, however, is anchored in **legally
mandated disclosure elements** (《发布证券研究报告暂行规定》): the broker
name, the signing analysts with their 执业证书编号 (S/A + 10 digits, format
is layout-independent), the covered security code and the publication date.

L1 (element fingerprint): extract those elements with layout-agnostic regex
from the first-page text + filename, and derive a stable ``business_key``.
L2 (SimHash fallback): when core elements cannot be extracted, a 64-bit
SimHash over character shingles provides near-duplicate detection.

This module is pure (no I/O, no settings) and unit-testable in isolation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional

# --- element patterns (layout-agnostic: full-text regex, no coordinates) ---

# 执业证书编号: S1010520040001 — leading S (or A) + digits; length varies by
# broker era (10–14 digits observed, e.g. 中信 S1010520040001 = 13 digits).
CERT_ID_RE = re.compile(r"\b([SA]\d{10,14})\b")

# A-share security code with exchange suffix: 300442.SZ / 600519.SH / 830799.BJ
SECURITY_CODE_RE = re.compile(r"\b(\d{6}\.(?:SZ|SH|BJ))\b", re.IGNORECASE)

# Date as leading filename token (aggregator naming: 2023-06-12_中信证券_标题.pdf)
FILENAME_DATE_RE = re.compile(r"^(\d{4})[-._](\d{1,2})[-._](\d{1,2})")

# Dates in first-page body text: 2023.6.12 / 2023-06-12 / 2023年6月12日
TEXT_DATE_RE = re.compile(
    r"(\d{4})\s?[.\-/年]\s?(\d{1,2})\s?[.\-/月]\s?(\d{1,2})\s?日?"
)

# Broker name: 2-6 CJK chars + 证券 (中信证券/华泰证券/国泰海通证券…)
BROKER_RE = re.compile(r"([一-龥]{2,6}证券)")


@dataclass
class ReportFingerprint:
    """Extracted business identity of a research report.

    ``confidence``: "high"   — cert id(s) found (unique per analyst, the
                               strongest anchor; also captures broker+date)
                    "medium" — security code + date found (series-level id)
                    "low"    — insufficient elements → rely on SimHash (L2)
    """

    broker: Optional[str] = None
    cert_ids: List[str] = field(default_factory=list)
    security_codes: List[str] = field(default_factory=list)
    report_date: Optional[str] = None  # normalised YYYY-MM-DD
    business_key: Optional[str] = None  # sha1[:16] of joined core elements
    confidence: str = "low"
    source: str = "elements"  # "elements" | "simhash_fallback"

    def to_metadata(self) -> dict:
        """Flatten for storage in trace / ingestion history."""
        return {
            "broker": self.broker,
            "cert_ids": ",".join(self.cert_ids),
            "security_codes": ",".join(self.security_codes),
            "report_date": self.report_date,
            "business_key": self.business_key,
            "confidence": self.confidence,
        }


def _normalise_date(y: str, m: str, d: str) -> Optional[str]:
    try:
        if not (1 <= int(m) <= 12 and 1 <= int(d) <= 31):
            return None
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    except (ValueError, TypeError):
        return None


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        key = it.upper()
        if key not in seen:
            seen.add(key)
            out.append(it.upper() if any(c.isascii() for c in it) else it)
    return out


def extract_report_fingerprint(
    first_page_text: str,
    filename: str = "",
) -> ReportFingerprint:
    """Extract the business fingerprint from a report's first page + filename.

    The search corpus deliberately favours *stable anchors*: cert ids are
    matched over the whole first page (they can sit in any corner), dates
    prefer the aggregator filename prefix (unambiguous) and fall back to
    body-text candidates, the broker prefers the filename's ``_券商_``
    segment (aggregator convention) then the page text.
    """
    text = first_page_text or ""

    cert_ids = _dedupe_keep_order(CERT_ID_RE.findall(text))

    codes = _dedupe_keep_order(
        c.upper() for c in SECURITY_CODE_RE.findall(text)
    )

    # date: body text first — the first full date on page 1 is the report
    # publication date (title line per disclosure rules). The filename prefix
    # is only a FALLBACK: aggregator names usually carry the report date, but
    # a re-distributed copy may be renamed after its *download* date, which
    # would silently shift the business key.
    report_date = None
    for m in TEXT_DATE_RE.finditer(text):
        cand = _normalise_date(m.group(1), m.group(2), m.group(3))
        if cand:
            report_date = cand
            break
    if report_date is None:
        m = FILENAME_DATE_RE.match(filename or "")
        if m:
            report_date = _normalise_date(m.group(1), m.group(2), m.group(3))

    # broker: ``<date>_券商_标题`` filename segment, else page text.
    broker = None
    if filename:
        parts = filename.split("_")
        if len(parts) >= 2:
            for part in parts[1:3]:
                m = BROKER_RE.search(part)
                if m:
                    broker = m.group(1)
                    break
    if broker is None:
        m = BROKER_RE.search(text)
        if m:
            broker = m.group(1)

    fp = ReportFingerprint(
        broker=broker, cert_ids=cert_ids, security_codes=codes,
        report_date=report_date,
    )

    # business key: high confidence = cert ids (analyst-anchored identity);
    # medium = broker? + codes + date (a report series instance). Absent
    # both → None (caller falls back to SimHash).
    if cert_ids:
        core = f"{broker or ''}|{'/'.join(sorted(cert_ids))}|{report_date or ''}"
        fp.confidence = "high"
    elif codes and report_date:
        core = f"{broker or ''}|{'/'.join(sorted(codes))}|{report_date}"
        fp.confidence = "medium"
    else:
        fp.business_key = None
        fp.confidence = "low"
        return fp

    fp.business_key = hashlib.sha1(core.encode("utf-8")).hexdigest()[:16]
    return fp


# ────────────────────────── L2: SimHash near-duplicate ──────────────────────


def _shingles(text: str, size: int = 3) -> List[str]:
    """Character shingles — CJK-safe (no tokeniser dependency, deterministic)."""
    cleaned = re.sub(r"\s+", "", text)
    if len(cleaned) < size:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + size] for i in range(len(cleaned) - size + 1)]


def simhash64(text: str) -> str:
    """64-bit SimHash (char 3-gram shingles weighted by frequency) as hex."""
    v = [0] * 64
    for sh in _shingles(text):
        h = int.from_bytes(
            hashlib.sha256(sh.encode("utf-8")).digest()[:8], "big"
        )
        for b in range(64):
            v[b] += 1 if (h >> b) & 1 else -1
    fingerprint = 0
    for b in range(64):
        if v[b] > 0:
            fingerprint |= 1 << b
    return f"{fingerprint:016x}"


def hamming_distance(hex_a: str, hex_b: str) -> int:
    a, b = int(hex_a, 16), int(hex_b, 16)
    return bin(a ^ b).count("1")


def is_near_duplicate(
    simhash_a: str,
    simhash_b: str,
    threshold_bits: int = 6,
) -> bool:
    """Near-duplicate judgement. Default threshold 6/64 bits — tuned to be
    conservative: watermark/格式差异 typically ≤ 4 bits apart on report
    bodies, while different reports (even same-series weeklies) sit far apart.
    Under-detecting only stores a redundant copy; over-detecting merges
    distinct documents, which is the worse failure."""
    return hamming_distance(simhash_a, simhash_b) <= threshold_bits
