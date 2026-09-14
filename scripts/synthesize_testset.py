#!/usr/bin/env python
"""T21 exam v5.0 synthesis orchestrator (ragas TestsetGenerator + deepseek).

Four resumable stages, each writing its artifact under --workdir:

    feed      docling parse (unchunked full text, extract_images=False)
              → feed.json; drops figure markers and the 10 T23 figure
              pages so the exam stays same-source with the library
    kg        feed → ragas KnowledgeGraph via default_transforms →
              kg.json (the expensive one: ~3 LLM calls per chunk node).
              Crash-resilient: applied step-by-step with a checkpoint
              (kg.json + manifest kg_steps_done) after each transform,
              and the kg-stage LLM calls are DiskCache-cached (deterministic
              index build — a redo replays instead of re-paying). Use
              --rebuild to start over.
    generate  kg → batches of questions → pool.jsonl (append-only) +
              manifest.json progress; crash-safe: rerun skips finished
              batches
    finalize  pool + qc_ledger.json → tests/fixtures/golden_test_set_v5.json
              (indented array, fixed field order, reference_context_ids
              matched against the live Chroma collection) +
              .wayfinder/tmp/t21_review.md (human review material)

Model wiring reuses the judge env (generator == judge model by design,
same deepseek source, no new exposure): DEEPSEEK_API_KEY, RAGAS_JUDGE_MODEL
(default deepseek-flash), RAGAS_JUDGE_BASE_URL (default
https://api.deepseek.com), OLLAMA_BASE_URL + RAGAS_JUDGE_EMB_MODEL
(nomic-embed-text) for synthesis-side embeddings.

Throttling (T20 lesson): deepseek client connect=15s/read=300s/
max_retries=4; batches of --batch-size with --pause seconds between;
RunConfig(max_workers=--workers). All LLM calls are DiskCache-cached
(<workdir>/llm_cache): at temperature 0 reruns replay essentially the
same outputs anyway, so caching costs no diversity and buys crash
insurance; repair rounds change --seed, which changes prompts, which
misses the cache naturally.

Seeding (user-set 2026-09-14): LLM temperature 0 + API seed 42 + script
base seed 42 — the maximum-determinism configuration. Per-batch seeds
derive from the base (base*1000 + batch, manifest-recorded) so batches
stay distinct while the whole round replays. The API seed is best-effort
(probed: identical requests varied slightly); script-level seeds are the
real lever. Repair/rerun rounds pass --seed 43, 44, … by convention (user-set
2026-09-14) — regenerating with the same seed returns (essentially)
the same question.

QC ledger format (hand-maintained during T21 QC, consumed by finalize):
    [{"pool_idx": 0, "verdict": "pass"},
     {"pool_idx": 3, "verdict": "remove", "reasons": ["③ …"]},
     {"pool_idx": 7, "verdict": "repair", "reasons": ["⑤ …"],
      "replaced_by_pool_idx": 41}]
Replacement questions are appended to pool.jsonl with "replaces": 7
(e.g. via: generate --round repair --only-type <synth> --count 1).

Usage (per-type rounds — see the note below):
    python scripts/synthesize_testset.py feed
    python scripts/synthesize_testset.py kg
    python scripts/synthesize_testset.py generate --round 1 \
        --only-type single_hop_specific_query_synthesizer --count 12
    python scripts/synthesize_testset.py generate --round 1 \
        --only-type multi_hop_abstract_query_synthesizer --count 12 \
        --batch-size 4
    python scripts/synthesize_testset.py generate --round 1 \
        --only-type multi_hop_specific_query_synthesizer --count 11 \
        --batch-size 4
    python scripts/synthesize_testset.py generate --round 2 --seed 43 \
        --only-type <synth> --count 1          # repair rounds: 43, 44, …
    python scripts/synthesize_testset.py finalize

Per-type rounds (user-confirmed 2026-09-14): the single-hop synthesizer
selects material by KG node order (first n nodes — the shuffle only
picks the entity/persona/style combo within a node), so mixed small
batches would draw every batch's single-hop questions from the same
first nodes (QC-④ duplicate risk). Generating each type in its own
round fixes chunk coverage: single-hop in ONE call (12 → nodes 0-11);
multi-hop types batched — their pair selection is a global shuffle, so
per-batch derived seeds diversify the pairs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

# UTF-8 console (Windows) — same pattern as ingest.py
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # noqa: E402 — after sys.path bootstrap

from src.core.settings import load_settings  # noqa: E402
from src.observability.evaluation.ragas_compat import (  # noqa: E402
    build_deepseek_client,
    import_ragas,
    resolve_deepseek_config,
)
from src.observability.evaluation.testset_builder import (  # noqa: E402
    FIGURE_PAGES,
    TYPE_LABELS,
    build_feed_text,
    finalise_entry,
    match_context_ids,
    render_review_md,
    select_final,
    write_final_json,
)

DEFAULT_CORPUS = "tests/fixtures/eval_docs"
DEFAULT_WORKDIR = "data/testset_v5"
DEFAULT_COLLECTION = "evaluation"
DEFAULT_FINAL = "tests/fixtures/golden_test_set_v5.json"
DEFAULT_REVIEW = ".wayfinder/tmp/t21_review.md"
DEFAULT_TARGET = 35          # oversample; 23 finalised after QC (T21)
DEFAULT_BATCH_SIZE = 5
DEFAULT_PAUSE = 30           # seconds between batches (T20 throttling)
DEFAULT_WORKERS = 4
DEFAULT_SEED = 42            # round-1 base seed (user-set 2026-09-14); repair/rerun rounds MUST pass a different --seed
FINAL_COUNT = 23
_EXCERPT_CHARS = 200       # review-md excerpt length per question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesise the v5.0 exam with ragas TestsetGenerator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("stage", choices=["feed", "kg", "generate", "finalize"])
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help="Chroma collection for context-id matching")
    parser.add_argument("--config", default=str(_REPO_ROOT / "config" / "settings.yaml"))
    # kg
    parser.add_argument("--rebuild", action="store_true",
                        help="kg: rebuild even if kg.json exists")
    # generate
    parser.add_argument("--count", type=int, default=DEFAULT_TARGET,
                        help="generate: questions requested this round "
                             f"(default {DEFAULT_TARGET})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--round", type=int, default=1,
                        help="generate: round tag recorded in pool/manifest")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="generate: base seed for this round")
    parser.add_argument("--only-type", choices=sorted(TYPE_LABELS), default=None,
                        help="generate: targeted regeneration for one "
                             "synthesizer type (repair path)")
    parser.add_argument("--kg-path", default=None,
                        help="generate: alternate KG json (e.g. a node-pruned "
                             "copy for top-ups); default <workdir>/kg.json")
    # finalize
    parser.add_argument("--out", default=DEFAULT_FINAL)
    parser.add_argument("--review", default=DEFAULT_REVIEW)
    parser.add_argument("--qc-ledger", default=None,
                        help="finalize: path to qc_ledger.json "
                             "(default: <workdir>/qc_ledger.json)")
    return parser.parse_args()


# ── shared builders (judge env reuse; see module docstring) ───────

# Synthesiser sampling params (user-set 2026-09-14) — constants so the
# llm_factory call and the manifest provenance block can't drift apart.
SYNTH_MAX_TOKENS = 16384   # API acceptance probed 2026-09-14: the first KG
                           # build died at NER — IncompleteOutputException on
                           # one numeric-dense chunk at 8192 (T20 overflow
                           # family). Judge stays 8192 until T22 (gauge freeze).
SYNTH_TEMPERATURE = 0      # greedy decoding; script-level seeds are the real
                           # reproducibility lever (API seed is best-effort)
SYNTH_API_SEED = 42        # API-side seed, best-effort semantics (probed)
SYNTH_THINKING = {"type": "disabled"}  # deepseek-flash defaults thinking ON;
                           # off = cheaper/faster for extraction-shaped work
                           # (extra_body passthrough; bare kwarg TypeErrors)


def _record_synthesizer(manifest: dict[str, Any]) -> None:
    """Exam-provenance block: which model+params produced the questions.

    The model name comes from the judge env (RAGAS_JUDGE_MODEL, reused by
    design) — recording it here keeps the exam's lineage intact when the
    env changes later (T21 code-review finding).
    """
    manifest["synthesizer"] = {
        "model": resolve_deepseek_config()["model"],
        "max_tokens": SYNTH_MAX_TOKENS,
        "temperature": SYNTH_TEMPERATURE,
        "api_seed": SYNTH_API_SEED,
        "thinking": "disabled",
        "recorded": datetime.now().isoformat(),
    }


def _build_llm(cache: Any = None, usage_log: Path | None = None) -> Any:
    """Synthesiser LLM: deepseek via ragas llm_factory, T20 client params.

    ``usage_log``: when set, an httpx response hook appends one JSON line
    per real API call (model / prompt_tokens / completion_tokens) — the
    usage field is otherwise discarded when instructor parses the
    response into a Pydantic model. Cache replays send no request and log
    nothing, so the file reflects actual spend.
    """
    import httpx
    from ragas.llms import llm_factory

    http_client = None
    if usage_log is not None:
        usage_path = Path(usage_log)
        usage_path.parent.mkdir(parents=True, exist_ok=True)

        async def _log_usage(response: Any) -> None:
            # Observability only — must never break the call it observes.
            try:
                await response.aread()
                body = json.loads(response.content)
                usage = body.get("usage") or {}
                details = usage.get("completion_tokens_details") or {}
                prompt_details = usage.get("prompt_tokens_details") or {}
                with usage_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "ts": datetime.now().isoformat(),
                        "model": body.get("model"),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "reasoning_tokens": details.get("reasoning_tokens"),
                        "cached_prompt_tokens": prompt_details.get("cached_tokens"),
                    }, ensure_ascii=False) + "\n")
            except Exception:  # noqa: BLE001 — logging is best-effort
                pass

        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=15.0),
            event_hooks={"response": [_log_usage]},
        )

    # Transport params (T20 timeouts/retries rationale) live in
    # ragas_compat.build_deepseek_client, shared with the judge — one
    # home, retunes land in both.
    client = build_deepseek_client(http_client=http_client)
    return llm_factory(
        resolve_deepseek_config()["model"], client=client,
        max_tokens=SYNTH_MAX_TOKENS, cache=cache,
        temperature=SYNTH_TEMPERATURE, seed=SYNTH_API_SEED,
        extra_body={"thinking": SYNTH_THINKING},
    )


def _build_embeddings() -> Any:
    """Synthesis-side embeddings: local nomic via Ollama /v1 (no cloud emb)."""
    import httpx
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings

    env_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    base_url = env_url if env_url.endswith("/v1") else f"{env_url}/v1"
    model = os.environ.get("RAGAS_JUDGE_EMB_MODEL", "nomic-embed-text")
    # trust_env=False bypasses the system proxy for localhost (D-014).
    http_client = httpx.AsyncClient(trust_env=False, timeout=60.0)
    client = AsyncOpenAI(base_url=base_url, api_key="ollama", http_client=http_client)
    return OpenAIEmbeddings(client=client, model=model)


# ── stage: feed ───────────────────────────────────────────────────


def cmd_feed(args: argparse.Namespace) -> None:
    from src.libs.parser.docling_parser import DoclingParser

    workdir = _workdir(args)
    settings = load_settings(str(args.config))
    corpus = Path(args.corpus)
    pdfs = sorted(corpus.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs under {corpus}")

    # Parse-only construction: no image side effects; page_batch_size
    # still settings-driven (single-batch default is the D-035 fix).
    parser_cfg = getattr(settings.ingestion, "parser", None) if settings.ingestion else None
    page_batch_size = getattr(parser_cfg, "page_batch_size", None)
    parser = DoclingParser(
        settings=settings,
        collection="testset_synth",
        image_storage_dir=_REPO_ROOT / "data" / "images",
        extract_images=False,
        page_batch_size=page_batch_size,
    )

    documents: list[dict[str, Any]] = []
    for pdf in pdfs:
        print(f"[feed] parsing {pdf.name} …", flush=True)
        doc = parser.parse(str(pdf))
        sections = doc.metadata.get("sections", [])
        excluded = FIGURE_PAGES.get(pdf.stem, [])
        text, report = build_feed_text(sections, excluded)
        if report["excluded_pages_with_text"]:
            # Expected empty (docling figure route emits no text there);
            # non-empty means behaviour drifted — the defensive drop kept
            # the feed same-source, but it must be recorded.
            print(f"[feed] WARNING {pdf.stem}: excluded pages carried text: "
                  f"{report['excluded_pages_with_text']}")
        documents.append({
            "stem": pdf.stem,
            "path": str(pdf),
            "excluded_pages": excluded,
            "text": text,
            "report": report,
        })
        print(f"[feed] {pdf.stem}: {report['kept_sections']} sections kept, "
              f"{report['dropped_figure_sections']} figure markers dropped, "
              f"{len(text)} chars", flush=True)

    out = {"created": datetime.now().isoformat(), "documents": documents}
    (workdir / "feed.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    total = sum(len(d["text"]) for d in documents)
    print(f"[feed] done: {len(documents)} docs, {total} chars → {workdir / 'feed.json'}")


# ── stage: kg ─────────────────────────────────────────────────────


def cmd_kg(args: argparse.Namespace) -> None:
    import_ragas()
    from langchain_core.documents import Document as LCDocument
    from ragas.cache import DiskCacheBackend
    from ragas.run_config import RunConfig
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.transforms import apply_transforms, default_transforms
    from ragas.testset.transforms.engine import get_desc

    workdir = _workdir(args)
    kg_path = workdir / "kg.json"
    manifest = _read_manifest(args)
    steps_done = int(manifest.get("kg_steps_done", 0))

    feed = json.loads((workdir / "feed.json").read_text(encoding="utf-8"))
    docs = [
        LCDocument(page_content=d["text"], metadata={"file_name": d["stem"]})
        for d in feed["documents"]
    ]

    # LLM calls are DiskCache-cached (deterministic at temperature 0): a
    # crashed or resumed build replays completed calls instead of paying
    # again. First-build lesson (2026-09-14): one overflow crash cost a
    # full ~350-call redo because nothing was durable. The generate stage
    # shares this cache — see cmd_generate.
    cache = DiskCacheBackend(cache_dir=str(workdir / "llm_cache"))
    llm = _build_llm(cache=cache, usage_log=workdir / "llm_usage.jsonl")
    _record_synthesizer(manifest)
    emb = _build_embeddings()
    transforms = default_transforms(documents=docs, llm=llm, embedding_model=emb)

    if args.rebuild:
        kg_path.unlink(missing_ok=True)
        manifest["kg_steps_done"] = steps_done = 0
        _write_manifest(args, manifest)

    if kg_path.exists() and not args.rebuild and steps_done >= len(transforms):
        print(f"[kg] {kg_path} complete ({steps_done} steps) — skipping "
              f"(use --rebuild to redo)")
        return

    if steps_done > 0 and kg_path.exists():
        print(f"[kg] resuming after {steps_done}/{len(transforms)} steps")
        kg = KnowledgeGraph.load(str(kg_path))
    else:
        kg = KnowledgeGraph(nodes=[
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": d.page_content,
                    "document_metadata": d.metadata,
                },
            )
            for d in docs
        ])

    # Step-wise application with a checkpoint after each transform: a crash
    # loses only the in-flight step. kg.json is saved BEFORE the manifest
    # claims progress — the failure direction (manifest lags one step, so a
    # completed step re-applies over itself) is benign; the reverse would
    # skip an unfinished step.
    run_config = RunConfig(max_workers=args.workers)
    started = time.time()
    for i in range(steps_done, len(transforms)):
        print(f"[kg] step {i + 1}/{len(transforms)}: {get_desc(transforms[i])} …",
              flush=True)
        step_started = time.time()
        apply_transforms(kg, transforms[i], run_config=run_config)
        kg.save(str(kg_path))
        manifest["kg_steps_done"] = i + 1
        _write_manifest(args, manifest)
        print(f"[kg] step {i + 1} done in {time.time() - step_started:.0f}s "
              f"(checkpoint: {len(kg.nodes)} nodes)", flush=True)
    print(f"[kg] done in {time.time() - started:.0f}s: "
          f"{len(kg.nodes)} nodes, {len(kg.relationships)} relationships → {kg_path}")


# ── stage: generate ───────────────────────────────────────────────


def _pool_path(args: argparse.Namespace) -> Path:
    return _workdir(args) / "pool.jsonl"


def _read_pool(args: argparse.Namespace) -> list[dict[str, Any]]:
    pool_file = _pool_path(args)
    if not pool_file.exists():
        return []
    entries = []
    for line in pool_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _append_pool(args: argparse.Namespace, entries: list[dict[str, Any]]) -> None:
    with _pool_path(args).open("a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def _manifest_path(args: argparse.Namespace) -> Path:
    return _workdir(args) / "manifest.json"


def _read_manifest(args: argparse.Namespace) -> dict[str, Any]:
    p = _manifest_path(args)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _write_manifest(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    _manifest_path(args).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def cmd_generate(args: argparse.Namespace) -> None:
    import_ragas()
    import numpy as np
    from ragas.cache import DiskCacheBackend
    from ragas.run_config import RunConfig
    from ragas.testset import TestsetGenerator
    from ragas.testset.graph import KnowledgeGraph

    workdir = _workdir(args)
    kg_path = Path(args.kg_path) if args.kg_path else workdir / "kg.json"
    kg = KnowledgeGraph.load(str(kg_path))
    # Shares the kg cache: at temperature 0 a rerun replays (essentially)
    # the same outputs anyway, so caching costs no diversity — it buys
    # crash insurance (a redone batch replays its finished LLM calls).
    # Repair rounds change --seed → different scenario shuffles →
    # different prompts → natural cache misses.
    llm = _build_llm(
        cache=DiskCacheBackend(cache_dir=str(workdir / "llm_cache")),
        usage_log=workdir / "llm_usage.jsonl",
    )
    emb = _build_embeddings()
    generator = TestsetGenerator(llm=llm, embedding_model=emb, knowledge_graph=kg)

    if args.only_type:
        from ragas.testset.synthesizers import default_query_distribution
        available = default_query_distribution(llm, kg)
        chosen = [s for s, _ in available if s.name == args.only_type]
        if not chosen:
            raise SystemExit(f"Synthesizer {args.only_type} not available for this KG")
        # Renormalise to 1.0: keeping the default 1/3 weight made
        # generate() aim at ceil(count × 1/3) instead of count — round-1
        # lesson (asked 12 got 4, asked 2 got 1; every shortfall traced
        # to calculate_split_values, synthesizers/utils.py).
        query_distribution = [(chosen[0], 1.0)]
        target = args.count
    else:
        query_distribution = None  # official default three-way split
        target = args.count

    manifest = _read_manifest(args)
    _record_synthesizer(manifest)
    progress = manifest.setdefault("rounds", {})
    # Progress key MUST include the type: the per-type protocol runs
    # several generate invocations under one --round, and a round-only
    # key made each later invocation "resume" past batches it never ran
    # (round-1 lesson: abstract skipped batch 1, specific skipped all 3).
    round_key = f"{args.round}:{args.only_type or 'default'}"
    round_info = progress.setdefault(round_key, {
        "seed": args.seed,
        "target": target,
        "batch_size": args.batch_size,
        "only_type": args.only_type,
        "requested": 0,
        "arrived": 0,
        "batches_done": 0,
    })
    # Resume: batches already recorded for this round are skipped.
    start_batch = round_info["batches_done"]
    n_batches = math.ceil(target / args.batch_size)
    run_config = RunConfig(max_workers=args.workers)

    pool_count = len(_read_pool(args))
    print(f"[gen] round {args.round}: target {target}, batches {n_batches}, "
          f"resuming from batch {start_batch}, pool has {pool_count}", flush=True)

    for batch in range(start_batch, n_batches):
        # Distinct derived seed per (seed, round, batch) — the round tag
        # MUST mix in, else round 0 (smoke) and round 1 with the same
        # base seed collide on derived seeds and regenerate identical
        # questions. Same-seed regeneration returns essentially the same
        # question (temp 0; API seed is best-effort, script seeds are the
        # strong lever).
        batch_seed = args.seed * 100000 + args.round * 100 + batch
        random.seed(batch_seed)
        np.random.seed(batch_seed % (2**32))
        size = min(args.batch_size, target - batch * args.batch_size)
        print(f"[gen] batch {batch + 1}/{n_batches} (size {size}, seed {batch_seed})…",
              flush=True)
        started = time.time()
        try:
            testset = generator.generate(
                testset_size=size,
                query_distribution=query_distribution,
                run_config=run_config,
                raise_exceptions=False,
            )
        except Exception as exc:  # noqa: BLE001 — keep prior batches, resume later
            print(f"[gen] batch {batch + 1} failed ({type(exc).__name__}: {exc}) — "
                  f"progress kept, rerun to resume", flush=True)
            break
        entries = testset.to_list()  # dicts already carry synthesizer_name
        now = datetime.now().isoformat()
        stamped = []
        for e in entries:
            stamped.append({
                "pool_idx": pool_count,
                "round": args.round,
                "batch": batch,
                "created": now,
                "replaces": None,
                **e,
            })
            pool_count += 1
        _append_pool(args, stamped)
        round_info["requested"] += size
        round_info["arrived"] += len(stamped)
        round_info["batches_done"] = batch + 1
        _write_manifest(args, manifest)
        print(f"[gen] batch {batch + 1}: {len(stamped)}/{size} arrived "
              f"in {time.time() - started:.0f}s (total arrived this round: "
              f"{round_info['arrived']})", flush=True)
        if batch + 1 < n_batches and args.pause > 0:
            print(f"[gen] pausing {args.pause:.0f}s…", flush=True)
            time.sleep(args.pause)

    manifest["updated"] = datetime.now().isoformat()
    _write_manifest(args, manifest)
    print(f"[gen] round {args.round} complete: arrived "
          f"{round_info['arrived']}/{round_info['requested']}")


# ── stage: finalize ───────────────────────────────────────────────


def _library_chunks(collection: str) -> list[tuple[str, str]]:
    import chromadb

    client = chromadb.PersistentClient(path=str(_REPO_ROOT / "data/db/chroma"))
    col = client.get_collection(collection)
    res = col.get(include=["documents"])
    return list(zip(res["ids"], res["documents"] or []))


def cmd_finalize(args: argparse.Namespace) -> None:
    workdir = _workdir(args)
    pool = _read_pool(args)
    if not pool:
        raise SystemExit("pool.jsonl is empty — run generate first")

    ledger_path = Path(args.qc_ledger) if args.qc_ledger else workdir / "qc_ledger.json"
    if not ledger_path.exists():
        raise SystemExit(
            f"QC ledger not found: {ledger_path}\n"
            "Run the QC pass first and record pass/remove/repair verdicts."
        )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    verdicts = {item["pool_idx"]: item for item in ledger}

    # Resolve survivors: pass → itself; repair → its replacement (latest
    # wins); remove / unrecorded → excluded.
    survivors: list[dict[str, Any]] = []
    removed_records: list[dict[str, Any]] = []
    repaired = 0
    for entry in pool:
        idx = entry["pool_idx"]
        v = verdicts.get(idx)
        if v is None:
            continue
        if v["verdict"] == "pass":
            survivors.append(entry)
        elif v["verdict"] == "repair":
            replacement_idx = v.get("replaced_by_pool_idx")
            replacement = next(
                (e for e in pool if e["pool_idx"] == replacement_idx), None,
            )
            if replacement is None:
                raise SystemExit(
                    f"Ledger repair for pool_idx {idx} points to missing "
                    f"replacement {replacement_idx}"
                )
            survivors.append(replacement)
            repaired += 1
        elif v["verdict"] == "remove":
            removed_records.append({
                "qid": f"x{idx:02d}",
                "user_input": entry.get("user_input", ""),
                "reason": "；".join(v.get("reasons", [])),
            })

    selected = select_final(survivors, n=FINAL_COUNT)
    if len(selected) < FINAL_COUNT:
        print(f"[final] WARNING: only {len(selected)} survivors for "
              f"{FINAL_COUNT} slots — rerun needed before freezing")

    chunks = _library_chunks(args.collection)
    chunks_by_id = dict(chunks)
    print(f"[final] matching context ids against {len(chunks)} library chunks…")

    # Strip ragas scenario scaffolding (<1-hop>/<2-hop> prefixes) from
    # multi-hop contexts: they are synthesis markers, not corpus content —
    # and they break containment matching (QC-③ lesson from precheck).
    import re
    _hop_prefix = re.compile(r"^<\d+-hop>\s*")

    entries: list[dict[str, Any]] = []
    review_entries: list[dict[str, Any]] = []
    for i, s in enumerate(selected, start=1):
        contexts = [_hop_prefix.sub("", c).strip()
                    for c in (s.get("reference_contexts") or [])]
        ids = (
            match_context_ids(contexts, chunks, min_recall=0.4)
            if contexts else None
        )
        # Stripped contexts go into the entry too — the prefixes are
        # synthesis scaffolding, not corpus content.
        s = {**s, "reference_contexts": contexts or None}
        entry = finalise_entry(s, context_ids=ids)
        entries.append(entry)
        v = verdicts.get(s["pool_idx"], {})
        verdict = "通过" if v.get("verdict") == "pass" else (
            f"修复过：{'；'.join(v.get('reasons', []))}"
        )
        # Review excerpt: prefer the matched library chunk's opening (the
        # retrievable grounding); fall back to the context itself with
        # journal front-matter (OPEN ACCESS / EDITED BY … review blocks —
        # T23 completeness keeps them in the corpus) skipped so the
        # excerpt shows substance, not boilerplate. Display-only: the
        # exam file's contexts stay verbatim for provenance.
        _front_matter = re.compile(r"^OPEN ACCESS[\s\S]{0,500}")
        if ids and ids[0] and ids[0] in chunks_by_id:
            excerpt = chunks_by_id[ids[0]][:_EXCERPT_CHARS]
        elif contexts:
            excerpt = _front_matter.sub("", contexts[0], count=1).strip()
            excerpt = excerpt[:_EXCERPT_CHARS]
        else:
            excerpt = ""
        review_entries.append({
            "qid": f"q{i:02d}",
            "synthesizer_name": s.get("synthesizer_name", ""),
            "user_input": s.get("user_input", ""),
            "reference": s.get("reference", ""),
            "excerpt": excerpt,
            "verdict": verdict,
        })

    write_final_json(entries, args.out)
    print(f"[final] wrote {len(entries)} entries → {args.out}")

    type_distribution: dict[str, int] = {}
    personas = sorted({s.get("persona_name") for s in selected if s.get("persona_name")})
    for s in selected:
        label = TYPE_LABELS.get(s.get("synthesizer_name", ""), s.get("synthesizer_name", "?"))
        type_distribution[label] = type_distribution.get(label, 0) + 1
    summary = {
        "synthesised": len(pool),
        "removed": len(removed_records),
        "repaired": repaired,
        "finalised": len(selected),
        "type_distribution": type_distribution,
        "personas": personas,
    }
    md = render_review_md(summary, review_entries, removed_records)
    review_path = Path(args.review)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(md, encoding="utf-8")
    print(f"[final] review material → {review_path}")


# ── entry ─────────────────────────────────────────────────────────


def _workdir(args: argparse.Namespace) -> Path:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def main() -> None:
    load_dotenv()
    args = parse_args()
    handlers = {
        "feed": cmd_feed,
        "kg": cmd_kg,
        "generate": cmd_generate,
        "finalize": cmd_finalize,
    }
    handlers[args.stage](args)


if __name__ == "__main__":
    main()
