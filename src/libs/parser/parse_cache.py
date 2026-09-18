"""Docling parse-result cache: lossless JSON replay for --force re-ingest (D-036).

Layout (one directory per source file, keyed by the file's content SHA256)::

    {cache_dir}/{sha256}/manifest.json     # version stamp + batch list
    {cache_dir}/{sha256}/batch_001.json    # docling lossless JSON, one per
    {cache_dir}/{sha256}/batch_002.json    # conversion batch (page-batched parse)

The manifest's ``version_stamp`` records which parser mapping produced the
JSON (hash of ``_LABEL_MAP`` + enrichment flags + docling version — see
``DoclingParser._version_stamp``).  A mismatch means the mapping logic was
upgraded since the entry was written, so the entry is invalidated and the
file re-parsed for real — stale caches can never silently swallow mapping
changes.

Design rules:
- The cache is advisory: any failure (save, load, corrupt entry) degrades to
  a fresh parse — it must never break ingestion (graceful degradation rule).
- The manifest is written LAST and atomically (temp-then-rename, same
  discipline as the BM25 JSON index): a torn write leaves the previous
  manifest authoritative instead of pointing at missing batch files.
- Replay only ever triggers where a cache key already exists, i.e. on
  ``--force`` re-ingest of an unchanged file (the file-level SHA256 skip in
  the pipeline handles everything else — D-006).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


class ParseCache:
    """Filesystem cache of docling lossless-JSON parse results (D-036)."""

    MANIFEST_NAME = "manifest.json"

    def __init__(self, cache_dir: str | Path, version_stamp: str) -> None:
        """
        Args:
            cache_dir: Root directory holding one sub-directory per file hash.
            version_stamp: Stamp of the parser mapping that will read this
                cache (computed by the parser, e.g. DoclingParser._version_stamp).
        """
        self.cache_dir = Path(cache_dir)
        self.version_stamp = version_stamp

    def lookup(self, file_sha256: str) -> Optional[list[Path]]:
        """Return the cached batch JSON paths for *file_sha256*, or None.

        None (a miss → "re-parse this file for real") is returned for every
        unusable case: no entry yet, version-stamp mismatch (mapping logic
        upgraded), or a corrupt/incomplete entry.
        """
        entry_dir = self.cache_dir / file_sha256
        manifest_path = entry_dir / self.MANIFEST_NAME
        if not manifest_path.is_file():
            return None  # first parse of this file — the common case, stay quiet
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning(
                f"parse cache manifest unreadable for {file_sha256[:8]} ({e}); re-parsing"
            )
            return None

        cached_stamp = manifest.get("version_stamp")
        if cached_stamp != self.version_stamp:
            logger.info(
                f"parse cache version stamp mismatch for {file_sha256[:8]}: "
                f"cached={cached_stamp} current={self.version_stamp} — re-parsing"
            )
            return None

        batch_names = manifest.get("batches") or []
        paths = [entry_dir / name for name in batch_names]
        if not paths or not all(p.is_file() for p in paths):
            logger.warning(
                f"parse cache entry incomplete for {file_sha256[:8]} "
                f"({len(batch_names)} batches listed); re-parsing"
            )
            return None
        return paths

    def save(self, file_sha256: str, documents: Sequence[Any]) -> None:
        """Persist *documents* (one ``save_as_json``-able per conversion batch).

        Best-effort: failures are logged and swallowed — a cache that can't
        be written must never fail the parse it is caching.
        """
        if not documents:
            return
        entry_dir = self.cache_dir / file_sha256
        try:
            entry_dir.mkdir(parents=True, exist_ok=True)
            batch_names: list[str] = []
            for index, ddoc in enumerate(documents, start=1):
                name = f"batch_{index:03d}.json"
                ddoc.save_as_json(filename=str(entry_dir / name))
                batch_names.append(name)
            self._write_manifest(entry_dir, batch_names)
        except Exception as e:
            logger.warning(
                f"parse cache save failed for {file_sha256[:8]} (ignored): {e}"
            )

    def _write_manifest(self, entry_dir: Path, batch_names: list[str]) -> None:
        """Write the manifest atomically, then prune files it no longer lists."""
        manifest = {
            "version_stamp": self.version_stamp,
            "batches": batch_names,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = entry_dir / self.MANIFEST_NAME
        tmp_path = entry_dir / (self.MANIFEST_NAME + ".tmp")
        tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp_path, manifest_path)
        # Deterministic names mean a re-save usually overwrites in place; only
        # a shrunk batch list (e.g. page_batch_size changed to single-batch)
        # leaves orphans behind.
        referenced = set(batch_names) | {self.MANIFEST_NAME}
        for stale in entry_dir.glob("*.json"):
            if stale.name not in referenced:
                stale.unlink(missing_ok=True)
