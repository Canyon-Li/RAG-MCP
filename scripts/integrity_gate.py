#!/usr/bin/env python
"""Corpus integrity gate: page-level verification of an ingested collection.

For every PDF in the corpus directory, extracts page text with PyMuPDF
(independent of any parser) and checks feature windows against the
collection's full text in Chroma. Catches silent ingestion content loss
(missing pages/sections) before it becomes a blind spot — see
DEV_CHANGELOG D-035 and .wayfinder/tickets/T23 (docling batched-conversion
loss) for why this gate exists.

Page verdicts:
- in_library     >=1 feature window (8-word windows at 5 positions, 6-word
                 fallback) found in the same document's library text
- figure_page    page text >=30 words but <20% alphabetic words >=4 chars —
                 full-page circuit/structure figures whose text layer is
                 only headers/coordinate labels (docling renders these as
                 images; text not ingested by design)
- blank_or_image page text <30 words
- MISSING        0 hits on a normal-text page — real content loss, gate FAIL

Also checks S1-style store alignment: chroma/bm25 id sets (chunks whose BM25
tokenization is empty are exempt — symbol tables/table rules/equations),
local-prefix purity, and hallucinated captions (non-empty caption rows).

Usage:
    python scripts/integrity_gate.py                       # evaluation corpus
    python scripts/integrity_gate.py --tag post_reingest   # named snapshot
    python scripts/integrity_gate.py --collection other --corpus dir/

Exit codes: 0 = all green, 1 = any FAIL (usable as a reingest gate).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

# UTF-8 console (Windows) — same pattern as ingest.py
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import chromadb
import fitz

DEFAULT_COLLECTION = "evaluation"
DEFAULT_CORPUS = "tests/fixtures/eval_docs"
IMAGE_INDEX = _REPO_ROOT / "data/db/image_index.db"

MIN_WORDS = 30        # <30 words -> blank/image page
ALPHA_LONG_MIN = 0.20  # <20% alphabetic words >=4 chars -> figure page
WINDOW_POSITIONS = (0.10, 0.30, 0.50, 0.70, 0.90)
WINDOW_LENGTHS = (8, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify per-page corpus integrity of an ingested collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--collection", "-c", default=DEFAULT_COLLECTION,
                        help=f"Collection to verify (default: {DEFAULT_COLLECTION})")
    parser.add_argument("--corpus", default=str(_REPO_ROOT / DEFAULT_CORPUS),
                        help=f"Corpus directory of source PDFs (default: {DEFAULT_CORPUS})")
    parser.add_argument("--tag", default=None,
                        help="Snapshot name (default: timestamp YYYYmmdd_HHMMSS)")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="Snapshot output path (default: logs/integrity_gate_{tag}.json)")
    return parser.parse_args()


def normalize_words(raw: str) -> list[str]:
    """PDF/library text -> lowercase, de-hyphenated, alphanumeric words."""
    joined = re.sub(r"-\s*\n\s*", "", raw)  # thermo-\ndynamic -> thermodynamic
    lowered = joined.lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).split()


def page_windows(words: list[str]) -> list[tuple[str, ...]]:
    """Feature windows at fixed relative positions (ordered word tuples)."""
    out = []
    for pos in WINDOW_POSITIONS:
        start = min(int(len(words) * pos), max(len(words) - 8, 0))
        for n in WINDOW_LENGTHS:
            win = words[start:start + n]
            if len(win) == n:
                out.append(tuple(win))
    return out


def main() -> int:
    args = parse_args()
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    json_out = Path(args.json_out) if args.json_out else _REPO_ROOT / f"logs/integrity_gate_{tag}.json"
    corpus_dir = Path(args.corpus)
    bm25_path = _REPO_ROOT / f"data/db/bm25/{args.collection}/{args.collection}_bm25.json"
    collection = args.collection

    # Library side: chroma chunks grouped by document prefix -> normalized full text
    client = chromadb.PersistentClient(path=str(_REPO_ROOT / "data/db/chroma"))
    if collection not in [c.name for c in client.list_collections()]:
        print(f"chroma[{collection}] does not exist — ingest first, then run the gate.")
        return 1
    col = client.get_collection(collection)
    data = col.get(include=["metadatas", "documents"])
    prefix_of = lambda cid: cid.split("_")[0]
    doc_text: dict[str, str] = {}
    doc_paths: dict[str, set[str]] = {}
    for cid, meta, doc in zip(data["ids"], data["metadatas"], data["documents"]):
        pfx = prefix_of(cid)
        joined = " ".join(normalize_words(doc or ""))
        doc_text[pfx] = f"{doc_text.get(pfx, '')} {joined}".strip()
        sp = (meta or {}).get("source_path")
        if sp:
            doc_paths.setdefault(pfx, set()).add(sp)
    print(f"chroma[{collection}] = {len(data['ids'])} chunks / {len(doc_text)} docs")

    # Corpus side: per-page verdicts
    results = []
    all_green = True
    for pdf in sorted(corpus_dir.glob("*.pdf")):
        pfx = hashlib.sha256(str(pdf.resolve()).encode("utf-8")).hexdigest()[:8]
        lib = doc_text.get(pfx)
        if not lib:
            with fitz.open(str(pdf)) as fd:
                pages = fd.page_count
            print(f"\n[NOT INGESTED] {pfx}  {pdf.name}")
            results.append({"pdf": pdf.name, "prefix": pfx, "status": "not_ingested",
                            "pages": pages})
            all_green = False
            continue
        pages_report = []
        n_in = n_blank = n_fig = 0
        with fitz.open(str(pdf)) as doc:
            for pno in range(doc.page_count):
                words = normalize_words(doc[pno].get_text())
                if len(words) < MIN_WORDS:
                    n_blank += 1
                    pages_report.append({"page": pno + 1, "words": len(words),
                                         "verdict": "blank_or_image"})
                    continue
                alpha_long = [w for w in words if len(w) >= 4 and w.isalpha()]
                if len(alpha_long) / len(words) < ALPHA_LONG_MIN:
                    n_fig += 1
                    pages_report.append({"page": pno + 1, "words": len(words),
                                         "verdict": "figure_page",
                                         "alpha_long_ratio": round(len(alpha_long) / len(words), 2)})
                    continue
                wins = page_windows(words)
                hits = sum(1 for w in wins if " ".join(w) in lib)
                if hits:
                    n_in += 1
                    pages_report.append({"page": pno + 1, "words": len(words),
                                         "verdict": "in_library", "hits": hits,
                                         "windows": len(wins)})
                else:
                    pages_report.append({"page": pno + 1, "words": len(words),
                                         "verdict": "MISSING",
                                         "first_window": " ".join(wins[0]) if wins else ""})
        missing = [p for p in pages_report if p["verdict"] == "MISSING"]
        ok = not missing
        all_green &= ok
        print(f"\n[{'PASS' if ok else 'FAIL'}] {pfx}  {pdf.name}")
        print(f"  {len(pages_report)}p = in_library {n_in} + blank/image {n_blank}"
              f" + figure_page {n_fig} + MISSING {len(missing)}"
              f"  (library chunks: {len([1 for c in data['ids'] if prefix_of(c) == pfx])})")
        for p in missing:
            print(f"    MISSING p{p['page']} ({p['words']}w)  first window: {p['first_window'][:70]}")
        results.append({"pdf": pdf.name, "prefix": pfx, "status": "ok" if ok else "missing",
                        "total_pages": len(pages_report), "in_library": n_in,
                        "blank_or_image": n_blank,
                        "figure_pages": [p["page"] for p in pages_report
                                         if p["verdict"] == "figure_page"],
                        "missing_pages": [p["page"] for p in missing],
                        "pages": pages_report,
                        "source_paths": sorted(doc_paths.get(pfx, []))})

    # S1-style checks: chroma/bm25 alignment + local prefixes + captions
    s1 = {}
    if bm25_path.exists():
        idx = json.loads(bm25_path.read_text(encoding="utf-8"))
        postings: Counter[str] = Counter()
        for td in idx["index"].values():
            for p in td["postings"]:
                postings[p["chunk_id"]] += 1
        bm25_set = set(postings)
        chroma_set = set(data["ids"])
        s1["bm25_unique"] = len(bm25_set)
        s1["bm25_num_docs"] = idx["metadata"].get("num_docs")
        s1["chroma_count"] = len(chroma_set)
        # Exempt chroma-only chunks whose BM25 tokenization is empty
        # (symbol tables / table rules / pure equations — legitimately unindexable)
        from src.core.text.tokenizer import tokenize as bm25_tokenize
        doc_by_id = dict(zip(data["ids"], data["documents"]))
        wordless = {c for c in chroma_set - bm25_set
                    if not bm25_tokenize(doc_by_id.get(c) or "")}
        s1["chroma_only"] = sorted(chroma_set - bm25_set - wordless)
        s1["chroma_only_wordless"] = sorted(wordless)
        s1["bm25_only"] = sorted(bm25_set - chroma_set)
        s1["duplicate_postings"] = {c: n for c, n in postings.items() if n > 1}
        aligned = (not s1["chroma_only"] and not s1["bm25_only"]
                   and s1["bm25_num_docs"] == len(bm25_set))
        print(f"\nS1 chroma/bm25 alignment: {'PASS' if aligned else 'FAIL'}"
              f"  (chroma {len(chroma_set)} / bm25 {len(bm25_set)} / num_docs {s1['bm25_num_docs']})")
        all_green &= aligned
    else:
        print("\nS1 chroma/bm25 alignment: SKIP (bm25 json missing)")
        s1["aligned"] = None

    local_prefixes = {hashlib.sha256(str(p.resolve()).encode("utf-8")).hexdigest()[:8]
                      for p in corpus_dir.glob("*.pdf")}
    foreign = {c: sp for c, sp in
               ((cid, (m or {}).get("source_path", "")) for cid, m in
                zip(data["ids"], data["metadatas"]))
               if prefix_of(c) not in local_prefixes}
    print(f"local prefixes: {'PASS' if not foreign else 'FAIL'}  (foreign: {len(foreign)})")
    for cid, sp in list(foreign.items())[:5]:
        print(f"  {cid}  {sp}")
    all_green &= not foreign
    s1["foreign_prefix_count"] = len(foreign)

    if IMAGE_INDEX.exists():
        con = sqlite3.connect(IMAGE_INDEX)
        n_rows = con.execute(
            "SELECT COUNT(*) FROM image_index WHERE collection = ?", (collection,)).fetchone()[0]
        n_cap = con.execute(
            "SELECT COUNT(*) FROM image_index WHERE collection = ? "
            "AND caption IS NOT NULL AND caption != ''", (collection,)).fetchone()[0]
        con.close()
        print(f"captions[{collection}]: {n_rows} images / {n_cap} non-empty "
              f"({'PASS' if n_cap == 0 else 'FAIL'})")
        all_green &= n_cap == 0
        s1["image_rows"] = n_rows
        s1["captions"] = n_cap

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps({"tag": tag, "collection": collection, "gate": results,
                    "s1": s1, "all_green": all_green},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'=' * 60}\nintegrity gate: {'ALL GREEN' if all_green else 'FAIL'}"
          f"\nsnapshot -> {json_out}")
    return 0 if all_green else 1


if __name__ == "__main__":
    sys.exit(main())
