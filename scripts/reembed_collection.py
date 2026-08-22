"""Re-embed a Chroma collection under a different Ollama embedding model.

Controlled A/B for embedding-model selection (D-030): reads every chunk
(id / text / metadata) from the source collection, re-embeds the text with
the requested model, and upserts into a target collection. Chunk text and
metadata are copied verbatim, so the ONLY variable that changes between
source and target is the embedding model.

The sparse side (BM25) is embedding-independent, so the source index file is
copied under the target collection name — hybrid retrieval works immediately
against the cloned collection.

Usage:
    python scripts/reembed_collection.py \
        --source zh_eval --target zh_eval_bge_m3 \
        --model bge-m3 --dimensions 1024 [--batch 32]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.settings import load_settings, resolve_path  # noqa: E402
from src.libs.embedding.ollama_embedding import OllamaEmbedding  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone a collection under a new embedding model.")
    parser.add_argument("--source", required=True, help="Existing Chroma collection to read chunks from.")
    parser.add_argument("--target", required=True, help="Target collection name (created/replaced).")
    parser.add_argument("--model", required=True, help="Ollama embedding model tag (e.g. bge-m3).")
    parser.add_argument("--dimensions", type=int, required=True, help="Output dimension of the model.")
    parser.add_argument("--batch", type=int, default=32, help="Chunks per embed call (default 32).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(str(PROJECT_ROOT / "config" / "settings.yaml"))

    # OllamaEmbedding only reads .embedding.model / .embedding.dimensions —
    # a lightweight namespace lets us override the model without touching
    # the active settings.yaml (query side flips separately per eval run).
    embedder = OllamaEmbedding(SimpleNamespace(
        embedding=SimpleNamespace(model=args.model, dimensions=args.dimensions)
    ))

    import chromadb

    client = chromadb.PersistentClient(
        path=str(resolve_path(settings.vector_store.persist_directory))
    )
    src = client.get_collection(args.source)
    data = src.get(include=["metadatas", "documents"])
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
    if not ids:
        print(f"[abort] source collection '{args.source}' is empty")
        return 1
    print(f"[load] {args.source}: {len(ids)} chunks")

    # Target: drop any previous clone so reruns are clean (ids are identical,
    # upsert alone would also work, but a fresh collection guarantees no
    # stale leftovers from a differently-dimensioned earlier attempt).
    try:
        client.delete_collection(args.target)
        print(f"[reset] deleted previous '{args.target}'")
    except Exception:
        pass
    dst = client.get_or_create_collection(
        name=args.target, metadata={"hnsw:space": "cosine"}
    )

    started = time.perf_counter()
    for lo in range(0, len(ids), args.batch):
        hi = min(lo + args.batch, len(ids))
        batch_ids = [str(i) for i in ids[lo:hi]]
        batch_docs = [(d or " ").strip() or " " for d in docs[lo:hi]]
        batch_metas = []
        for meta, text in zip(metas[lo:hi], batch_docs):
            m = dict(meta or {})
            m["text"] = text  # keep ChromaStore.upsert shape (document = metadata.text)
            batch_metas.append(m)

        vectors = embedder.embed(batch_docs)
        if len(vectors) != len(batch_ids):
            raise RuntimeError(
                f"embedding count mismatch at rows {lo}..{hi}: "
                f"{len(vectors)} vs {len(batch_ids)}"
            )
        dst.upsert(
            ids=batch_ids, embeddings=vectors,
            metadatas=batch_metas, documents=batch_docs,
        )
        done = hi
        rate = done / max(time.perf_counter() - started, 1e-6)
        print(f"[embed] {done}/{len(ids)} chunks ({rate:.1f}/s, model={args.model})")

    # Sparse side is model-independent — copy the BM25 index under the new
    # collection name so hybrid retrieval resolves it via the standard path.
    bm25_src = PROJECT_ROOT / "data" / "db" / "bm25" / args.source / f"{args.source}_bm25.json"
    bm25_dst = PROJECT_ROOT / "data" / "db" / "bm25" / args.target / f"{args.target}_bm25.json"
    if bm25_src.exists():
        bm25_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bm25_src, bm25_dst)
        print(f"[bm25] copied index -> {bm25_dst}")
    else:
        print(f"[bm25] WARN no index at {bm25_src} (dense-only clone)")

    print(
        f"[done] '{args.target}': {dst.count()} chunks, dim={args.dimensions}, "
        f"model={args.model}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
