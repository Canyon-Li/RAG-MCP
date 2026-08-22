"""Report fingerprinting for version management (D-031).

L1: element fingerprint from legally-mandated disclosure elements
    (cert ids / security codes / date / broker) — layout-agnostic.
L2: SimHash near-duplicate fallback when elements can't be extracted.
"""

from src.ingestion.fingerprint.report_fingerprint import (
    ReportFingerprint,
    extract_report_fingerprint,
    hamming_distance,
    is_near_duplicate,
    simhash64,
)

__all__ = [
    "ReportFingerprint",
    "extract_report_fingerprint",
    "simhash64",
    "hamming_distance",
    "is_near_duplicate",
]
