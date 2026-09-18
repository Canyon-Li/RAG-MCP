---
paths:
  - "src/libs/**/*.py"
  - "src/ingestion/**/*.py"
---

# Extending pluggable backends

Applies when editing `src/libs/` or `src/ingestion/`. Every swappable backend is **Base class + Factory + Registry**; adding a backend is mechanical and touches none of the pipelines or core.

## Add a provider (general procedure)

1. Subclass the relevant `Base*` (see table) in its matching directory.
2. Register it: `<Factory>.register_provider("<name>", YourClass)` in the factory's `_register_builtin_providers()` (LLM / vector store / embedding register in the package `__init__.py` instead).
3. Add config under the matching key in `config/settings.yaml` and switch `provider:` to `<name>`.

Then verify with `<Factory>.list_providers()` or by reading the registration block — **that is the source of truth, not the table below**.

## Base classes, factories, and currently-registered providers

> 🧭 **评估期聚焦提示**:评估(`scripts/evaluate.py` + `src/observability/evaluation/` + `src/core/query_engine/`)用的是已 ingest 的数据,**不重新 parse、不触碰本表任何 provider**。评估时无需关注 parser 注册表——它对评估结果是透明的。当前运行时只激活 `docling` 一条 parser 链,其余 provider 是可插拔备选,仅作切换/降级兜底用。

> ⚠️ **The "Registered providers" column below is a point-in-time snapshot (@ 2026-07-27), NOT a live query.** Authoritative source: run `<Factory>.list_providers()` (or `LLMFactory.list_vision_providers()` for Vision LLM), or read the `_register_builtin_providers()` / `_register_vision_providers()` block in each factory. Providers are added frequently — **do not trust the table without verifying against code.**

| Layer | Base class | Factory | Registered providers (snapshot @ 2026-07-27) |
|---|---|---|---|
| LLM | `BaseLLM` (`libs/llm/`) | `LLMFactory` | openai, azure, deepseek, ollama, qwen, zhipu |
| Vision LLM | `BaseVisionLLM` (`libs/llm/`) | `LLMFactory` (`create_vision_llm` / `list_vision_providers`) | azure, openai, ollama |
| Embedding | `BaseEmbedding` (`libs/embedding/`) | `EmbeddingFactory` | openai, azure, ollama, qwen, zhipu |
| Splitter | `BaseSplitter` (`libs/splitter/`) | `SplitterFactory` | recursive |
| Vector store | `BaseVectorStore` (`libs/vector_store/`) | `VectorStoreFactory` | chroma |
| Reranker | `BaseReranker` (`libs/reranker/`) | `RerankerFactory` | llm, cross_encoder |
| Evaluator | `BaseEvaluator` (`libs/evaluator/`) | `EvaluatorFactory` | custom, ragas (lazy), composite (lazy) |
| Parser | `BaseParser` (`libs/parser/`) | `ParserFactory` | pdf, pdf_text, pdf_table, docling, docling_vlm, docx |
| Transform | `BaseTransform` (`ingestion/transform/`) | none — wired directly in `IngestionPipeline` | chunk_refiner, metadata_enricher, image_captioner, table_summarizer |

Note: Vision LLM providers are registered on `LLMFactory._VISION_PROVIDERS` (a separate registry on the same factory) and instantiated via `LLMFactory.create_vision_llm(settings)` — `ImageCaptioner` calls this in its `__init__`. There is no separate VisionFactory.

## Adding a new document format (Parser)

This is the extension path most likely to be used.

1. Create `src/libs/parser/foo_parser.py` with `class FooParser(BaseParser)`.
   Constructor contract (shared with PdfTextParser / WordParser / PdfTableParser / DoclingParser so `ParserFactory.create` builds any parser uniformly): `__init__(self, settings, collection: str, image_storage_dir: str | Path, **kwargs)`.
2. In `parser_factory._register_builtin_providers()`: call `ParserFactory.register_provider("foo", FooParser)` and add `"foo": [".foo"]` to `_PROVIDER_EXTENSIONS`.
3. Set `ingestion.parser.provider: foo` in `settings.yaml`.
4. No CLI edit needed — `scripts/ingest.py` discovers files via `ParserFactory.get_supported_extensions(provider)`, and `IngestionPipeline` calls `ParserFactory.create(settings, collection)`. The rest of the pipeline (chunk / transform / embed / upsert) is format-agnostic and operates on the `Document` your parser returns.

Image extraction is optional per-parser (governed by `extract_images`); if your format has no images, accept the flag and no-op. Note that image-extraction **method** is parser-specific (see CLAUDE.md "Multimodal" section + DEV_CHANGELOG D-018/D-019) — `docling` renders bbox regions, others use PyMuPDF `get_images()`.
