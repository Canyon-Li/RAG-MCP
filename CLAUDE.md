# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

A modular, pluggable RAG (Retrieval-Augmented Generation) server that exposes knowledge-retrieval tools over the **MCP (Model Context Protocol)** stdio transport, consumable directly by GitHub Copilot / Claude Desktop / Cursor. It is a learning-and-interview project (Chinese-language README). Design rationale lives in `DEV_CHANGELOG.md` (ADR-style records, D-001 ~ D-023); the current architecture is described in `## Architecture` below and rendered visually in the README's mermaid diagram. There is no `DEV_SPEC.md` anymore — it was an initial design draft that got out of sync and was removed.

The README is explicit that this is **not** a hardened production system: bugs are expected, and two modules (Custom Evaluator, Cross-Encoder Reranker) are scaffolded but not fully tested.

## Document scope (what lives where)

This file describes **stable architecture only** — mechanisms that hold across config changes (four-layer design, Factory+Registry pattern, pipeline flow, idempotency, trace, MCP stdio constraints). It intentionally carries **no volatile facts**; those live in their own sources of truth:

| Volatile info | Where it actually lives |
|---|---|
| Registered provider names (per layer) | [.claude/rules/extending-backends.md](.claude/rules/extending-backends.md) (dated snapshot) or run `<Factory>.list_providers()` |
| Current default provider / model | [config/settings.yaml](config/settings.yaml) |
| Capability boundaries & decision rationale | [DEV_CHANGELOG.md](DEV_CHANGELOG.md) (e.g. D-011 ~ D-023) |
| Runtime environment (conda env, proxy) | [DEV_CHANGELOG.md](DEV_CHANGELOG.md) + global memory |

When you find yourself adding a provider name, a default model, or a "currently supports X" claim to this file — **stop and redirect it to the table's target instead**. That discipline is what keeps this file from rotting as the codebase evolves.

## Commands

Install (editable, with dev tools):
```powershell
pip install -e ".[dev]"
```

Run the MCP server (stdio transport — for Copilot/Claude to spawn):
```powershell
python -m src.mcp_server.server
```
> ⚠️ `main.py` is a stub that only loads settings; it does **not** start the MCP server. The real entry point is `src/mcp_server/server.py:main`. The `mcp-server` console-script in `pyproject.toml` (`main:main`) is misleading — prefer the module form above, which is what the integration/e2e tests shell out to.

Ingest documents (format is config-driven via `ingestion.parser.provider`; registered parsers listed in [.claude/rules/extending-backends.md](.claude/rules/extending-backends.md) or via `ParserFactory.list_providers()`):
```powershell
python scripts/ingest.py --path <file-or-dir> --collection <name> [--force] [--dry-run]
```

Run a one-off hybrid query from the CLI:
```powershell
python scripts/query.py --query "..." --collection <name> [--top-k 10] [--verbose] [--no-rerank]
```

Launch the Streamlit dashboard (6-page management UI, default :8501):
```powershell
python scripts/start_dashboard.py [--port 8501]
```

Run evaluation against a golden test set:
```powershell
python scripts/evaluate.py --test-set tests/fixtures/golden_test_set_v41.json [--collection <name>] [--json]
```

Tests — `pytest` with markers defined in `pyproject.toml` (`unit`, `integration`, `e2e`, `llm`, `slow`):
```powershell
pytest                               # everything
pytest tests/unit                    # one layer only
pytest tests/unit/test_fusion_rrf.py # one file
pytest -m "not llm"                  # skip tests needing real LLM API calls
pytest -k "rrf"                      # by name substring
```

Lint / typecheck / import-check:
```powershell
ruff check .
mypy src
python -m compileall src
```

## Architecture

Four-layer design, all wired by config. The README renders the full pipeline diagram as mermaid; this section is the stable prose reference.

```
MCP Server (interface)  ─ src/mcp_server/      tools exposed over JSON-RPC stdio
Core (business logic)   ─ src/core/            query engine, response builder, trace
Storage                 ─ data/db/             Chroma + BM25 + ImageStorage SQLite + trace logs
Libs (pluggable)        ─ src/libs/            Factory pattern for every swappable backend
Ingestion pipeline      ─ src/ingestion/       load → split → transform → embed → upsert
Observability           ─ src/observability/   trace context + Streamlit dashboard + eval
```

### Two pipelines, both traced end-to-end

- **Ingestion** (`src/ingestion/pipeline.py::IngestionPipeline`): FileIntegrity (SHA256 skip) → `ParserFactory.create()` (provider from `ingestion.parser`) → Chunking (strategy from `ingestion.chunker`: docling-parsed documents go through the HybridChunker adapter — heading contextualization / table GFM in `table_html` / contains_* type flags; everything else, or a config switch-back, uses the original recursive `DocumentChunker`) → Transform (ChunkRefiner + MetadataEnricher + ImageCaptioner + TableSummarizer) → Dense+Sparse encoding → Upsert (Chroma + BM25 + ImageStorage).
- **Query** (`src/core/query_engine/`): QueryProcessor → parallel Dense (embedding cosine) + Sparse (BM25) → **RRF fusion** → optional Rerank (none / cross_encoder / llm) → ResponseBuilder (citations + multimodal assembly).

Both pipelines take an explicit `TraceContext` (`src/core/trace/`) that records each stage's `method`/`provider`/latency and flushes one JSON Lines record to `logs/traces.jsonl`. The dashboard reads that file — it has no other API. **Stage names are stable categories** (`retrieval`, `rerank`, …); the concrete method goes in a `method`/`details` field so swapping backends doesn't break dashboard rendering.

### Config-driven everything

`config/settings.yaml` is the single source of truth for **which backends are active** (LLM / embedding / vision / parser / rerank / vector store). It is parsed into frozen dataclasses in [src/core/settings.py](src/core/settings.py), which also exposes `resolve_path()` — paths are anchored to `REPO_ROOT` (computed from `__file__`, **CWD-independent**), so scripts run correctly from any working directory. Validation is fail-fast (`SettingsError`).

### Pluggable backends via Factory + Registry

Every swappable component follows the same pattern — **Base class + Factory + Registry**:
- A `Base*` abstract class: `BaseLLM`, `BaseVisionLLM`, `BaseEmbedding`, `BaseSplitter`, `BaseVectorStore`, `BaseReranker`, `BaseEvaluator`, `BaseParser` (all in `src/libs/`), plus `BaseTransform` (`src/ingestion/transform/`).
- A `*Factory` with a class-level registry (`_PROVIDERS`) populated by `register_provider()` at import time. (Transforms are the exception — wired directly in the pipeline, no factory.)
- Selection reads `settings.yaml` → **changing backends is config-only, no code edits**.

How to add a provider (new LLM / new doc format / …) and the live provider registry live in [.claude/rules/extending-backends.md](.claude/rules/extending-backends.md) — a path-scoped rule that loads only when you touch `src/libs/` or `src/ingestion/`. The `setup` skill automates the common case.

### Multimodal = Image-to-Text (no CLIP)

Images are extracted during parse, saved to `data/images/{collection}/`, and captioned by a Vision LLM during transform; the caption text is **stitched into the chunk body** so plain-text retrieval surfaces images. **Structured image data (id / path / page / caption) lives in ImageStorage SQLite (`data/db/image_index.db`), decoupled from Chroma metadata** (D-018) — hit chunks resolve images via ImageStorage (`ImageCaptioner` calls `image_storage.get_image_meta` / `set_caption`), then file read → Base64 into the MCP `content` array.

Image extraction method is **parser-specific** (capability boundary — see DEV_CHANGELOG D-018 / D-019): the `docling` parser renders bbox regions to raster (captures vector-drawn figures); other parsers use PyMuPDF `get_images()` (embedded raster only).

### Idempotency & storage layout (all gitignored under `data/` + `logs/`)

- File-level: SHA256 in SQLite at `data/db/ingestion_history.db` → unchanged files are skipped (zero-cost incremental ingest). `--force` bypasses this.
- Chunk-level: deterministic `chunk_id = {doc_id}_{index:04d}_{content_hash8}`; upserts are idempotent.
- Stores: Chroma at `data/db/chroma/` (dense + sparse vectors + payload), BM25 **JSON** index at `data/db/bm25/{collection}/{collection}_bm25.json` (`json.dump` with atomic temp-then-rename — **not** pickle), image files at `data/images/` + structured image metadata in ImageStorage SQLite at `data/db/image_index.db` (D-018 — image data is **not** stored in Chroma), docling parse cache at `data/parsed/{sha256}/` (lossless JSON replay for `--force` re-ingest, keyed by file SHA256 + parser version stamp — enable/dir live in [config/settings.yaml](config/settings.yaml), D-036), traces at `logs/traces.jsonl`.

## Conventions and gotchas

- **MCP stdio is stdout-strict.** Only JSON-RPC may go to stdout; all logs go to stderr. `server.py` redirects root handlers and **preloads heavy imports (chromadb, and the cross-encoder torch stack when `rerank.provider` is `cross_encoder`)** in the main thread to avoid import-lock deadlocks against anyio's I/O threads when tool handlers later call `asyncio.to_thread`. Heavy **model loads** (not just imports) additionally load in a background warm thread — a blocking load at startup would blow the initialize handshake timeout, and loading inside a tool-handler thread re-triggers the deadlock via transformers' lazy imports (D-033). Keep this pattern if you touch server startup.
- **Graceful degradation is a design rule.** LLM-backed transforms (`chunk_refiner`, `metadata_enricher`) fall back to rule-based logic when `use_llm: false` or the LLM call fails — they must not block the pipeline. Reranker failures fall back to RRF order; rerank and evaluation are config-gated (`rerank.enabled` / `evaluation.enabled` in [config/settings.yaml](config/settings.yaml) — current defaults live there, not here).
- **Windows console + Chinese output.** CLI scripts set `sys.stdout/stderr` to UTF-8 wrappers on `win32`; match this if adding scripts that print non-ASCII.
- **Tests insert repo root onto `sys.path`** (`conftest.py` and each script), so `from src.…` imports work without installing the package. Integration/e2e tests shell out to `python -m src.mcp_server.server` as a subprocess.
- **Adding a new document format:** subclass `BaseParser` + `ParserFactory.register_provider()`; the rest of the pipeline is format-agnostic. Details & registered parsers in [.claude/rules/extending-backends.md](.claude/rules/extending-backends.md).
- **Prompts** live as plain text in `config/prompts/` (`image_captioning.txt`, `chunk_refinement.txt`, `metadata_enrichment.txt`, `rerank.txt`, `table_summary.txt`, `formula_transcription.txt`) — edit there, not in code.
- **Eval vocabulary: use the new terms.** Frozen archives (git history, `.wayfinder/`, older DEV_CHANGELOG entries) still use old Chinese codewords (判官/噪声底/考卷/杠杆/重灌…); translate via the glossary in [CONTEXT.md](CONTEXT.md) when reading. New commits, docs, comments, and UI strings must use the canonical new terms — never copy an old codeword from history into new output.
- **Design decisions & pitfalls live in [DEV_CHANGELOG.md](DEV_CHANGELOG.md)** — check it before changing providers / parsers / runtime env to avoid repeating past traps (e.g. local-service httpx needs `trust_env=False`; use conda, not `.venv`; completion models can't be wrapped in a chat template).

## Skills (agent-driven workflow)

`.claude/skills/` now contains only `skill-creator` (a meta-skill for authoring new skills). The build-time skills that originally constructed this repo (`auto-coder`, `qa-tester`, `setup`, `package`, `resume-writer`) were retired and removed once the project reached its current shape — they were scaffolding, not part of the runtime. Do work manually rather than invoking them.

## Agent skills

### Issue tracker

Local markdown tracker: implementation tickets live under `.scratch/<feature>/issues/` (gitignored); durable specs go to `docs/superpowers/specs/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: glossary in `CONTEXT.md`; ADRs are the D-0xx entries in `DEV_CHANGELOG.md` (there is no `docs/adr/`). See `docs/agents/domain.md`.
