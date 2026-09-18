"""Unit tests for ParseCache (D-036 parse-result cache).

Covers the cache's own contract only — file layout, stamp invalidation,
corruption tolerance.  The docling roundtrip fidelity (save_as_json →
load_from_json → identical item stream) was proven by spike 01
(.scratch/ingestion-revamp); parser-level replay wiring is tested in
test_docling_parse_cache.py.
"""

import json
from pathlib import Path

from src.libs.parser.parse_cache import ParseCache

STAMP_A = "aaaa1111bbbb2222"
STAMP_B = "ffff9999eeee8888"


class _FakeDoc:
    """Stand-in for a DoclingDocument: save_as_json writes plain JSON."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.saved_to: list[str] = []

    def save_as_json(self, filename) -> None:
        Path(filename).write_text(json.dumps(self.payload), encoding="utf-8")
        self.saved_to.append(str(filename))


class _BoomDoc:
    """save_as_json always fails (simulates a torn/crashed write)."""

    def save_as_json(self, filename) -> None:
        raise RuntimeError("disk full")


def test_lookup_returns_none_when_no_entry(tmp_path):
    cache = ParseCache(cache_dir=tmp_path / "parsed", version_stamp=STAMP_A)
    assert cache.lookup("f" * 64) is None


def test_save_then_lookup_roundtrip(tmp_path):
    cache = ParseCache(cache_dir=tmp_path / "parsed", version_stamp=STAMP_A)
    sha = "a" * 64

    cache.save(sha, [_FakeDoc({"batch": 1}), _FakeDoc({"batch": 2})])

    # Layout: one directory per file hash, manifest + batch JSONs inside.
    entry = tmp_path / "parsed" / sha
    assert (entry / "manifest.json").is_file()
    assert (entry / "batch_001.json").is_file()
    assert (entry / "batch_002.json").is_file()

    paths = cache.lookup(sha)
    assert paths == [entry / "batch_001.json", entry / "batch_002.json"]
    # Manifest records the stamp that produced the entry.
    manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version_stamp"] == STAMP_A
    assert manifest["batches"] == ["batch_001.json", "batch_002.json"]


def test_lookup_stamp_mismatch_is_a_miss(tmp_path):
    """Cache written by an older parser mapping must not be replayed."""
    sha = "b" * 64
    ParseCache(cache_dir=tmp_path, version_stamp=STAMP_A).save(sha, [_FakeDoc({"x": 1})])

    fresh = ParseCache(cache_dir=tmp_path, version_stamp=STAMP_B)
    assert fresh.lookup(sha) is None


def test_resave_with_new_stamp_overwrites_entry(tmp_path):
    """After a stamp-mismatched re-parse, save() replaces the stale entry."""
    sha = "c" * 64
    ParseCache(cache_dir=tmp_path, version_stamp=STAMP_A).save(sha, [_FakeDoc({"old": 1})])
    ParseCache(cache_dir=tmp_path, version_stamp=STAMP_B).save(sha, [_FakeDoc({"new": 1})])

    paths = ParseCache(cache_dir=tmp_path, version_stamp=STAMP_B).lookup(sha)
    assert paths is not None
    assert json.loads(paths[0].read_text(encoding="utf-8")) == {"new": 1}


def test_lookup_corrupt_manifest_is_a_miss_not_a_crash(tmp_path):
    sha = "d" * 64
    entry = tmp_path / sha
    entry.mkdir(parents=True)
    (entry / "manifest.json").write_text("{not json", encoding="utf-8")

    assert ParseCache(cache_dir=tmp_path, version_stamp=STAMP_A).lookup(sha) is None


def test_lookup_missing_batch_file_is_a_miss(tmp_path):
    """Manifest lists a batch whose file is gone (torn write) → re-parse."""
    sha = "e" * 64
    cache = ParseCache(cache_dir=tmp_path, version_stamp=STAMP_A)
    cache.save(sha, [_FakeDoc({"1": 1}), _FakeDoc({"2": 2})])
    (tmp_path / sha / "batch_002.json").unlink()

    assert cache.lookup(sha) is None


def test_save_swallows_document_write_failure(tmp_path):
    """Cache is advisory: a failing save_as_json must not raise."""
    cache = ParseCache(cache_dir=tmp_path / "parsed", version_stamp=STAMP_A)
    cache.save("f" * 64, [_BoomDoc()])  # must not raise
    assert cache.lookup("f" * 64) is None


def test_save_skips_empty_documents(tmp_path):
    cache = ParseCache(cache_dir=tmp_path / "parsed", version_stamp=STAMP_A)
    cache.save("0" * 64, [])
    assert not (tmp_path / "parsed" / ("0" * 64)).exists()


def test_resave_prunes_orphaned_batches(tmp_path):
    """Shrunk batch lists (e.g. page_batch_size changed to 0) leave no orphans."""
    sha = "1" * 64
    cache = ParseCache(cache_dir=tmp_path, version_stamp=STAMP_A)
    cache.save(sha, [_FakeDoc({"a": 1}), _FakeDoc({"b": 2}), _FakeDoc({"c": 3})])
    cache.save(sha, [_FakeDoc({"a": 1})])

    entry = tmp_path / sha
    jsons = sorted(p.name for p in entry.glob("*.json"))
    assert jsons == ["batch_001.json", "manifest.json"]


def test_default_cache_dir_is_gitignored():
    """The configured default cache root lives under a gitignored path
    (ticket 02 acceptance: parse-cache directory covered by .gitignore)."""
    from src.core.settings import REPO_ROOT, ParseCacheSettings

    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    ignored_roots = {
        line.strip().rstrip("/")
        for line in gitignore.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    cache_root = Path(ParseCacheSettings().dir).parts[0]
    assert cache_root in ignored_roots, (
        f"default parse-cache dir {ParseCacheSettings().dir!r} is not under "
        f"a gitignored root"
    )
