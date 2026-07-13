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

| Layer | Base class | Factory | Registered providers |
|---|---|---|---|
| LLM | `BaseLLM` (`libs/llm/`) | `LLMFactory` | openai, azure, deepseek, ollama, qwen |
| Vision LLM | `BaseVisionLLM` (`libs/llm/`) | none — instantiated by `ImageCaptioner` | — |
| Embedding | `BaseEmbedding` (`libs/embedding/`) | `EmbeddingFactory` | openai, azure, ollama, qwen |
| Splitter | `BaseSplitter` (`libs/splitter/`) | `SplitterFactory` | recursive |
| Vector store | `BaseVectorStore` (`libs/vector_store/`) | `VectorStoreFactory` | chroma |
| Reranker | `BaseReranker` (`libs/reranker/`) | `RerankerFactory` | llm, cross_encoder |
| Evaluator | `BaseEvaluator` (`libs/evaluator/`) | `EvaluatorFactory` | (scaffolded — see factory) |
| Loader | `BaseLoader` (`libs/loader/`) | `LoaderFactory` | pdf, docx |
| Transform | `BaseTransform` (`ingestion/transform/`) | none — wired directly in `IngestionPipeline` | chunk_refiner, metadata_enricher, image_captioner |

> The "Registered providers" column changes often; trust `list_providers()` / the registration code over this table.

## Adding a new document format (Loader)

This is the extension path most likely to be used.

1. Create `src/libs/loader/foo_loader.py` with `class FooLoader(BaseLoader)`.
   Constructor contract (shared with PdfLoader / WordLoader so `LoaderFactory.create` builds any loader uniformly): `__init__(self, extract_images: bool, image_storage_dir: str | Path)`.
2. In `loader_factory._register_builtin_providers()`: call `LoaderFactory.register_provider("foo", FooLoader)` and add `"foo": [".foo"]` to `_PROVIDER_EXTENSIONS`.
3. Set `ingestion.loader.provider: foo` in `settings.yaml`.
4. No CLI edit needed — `scripts/ingest.py` discovers files via `LoaderFactory.get_supported_extensions(provider)`, and `IngestionPipeline` calls `LoaderFactory.create(settings, collection)`. The rest of the pipeline (chunk / transform / embed / upsert) is format-agnostic and operates on the `Document` your loader returns.

Image extraction is optional per-loader (governed by `extract_images`); if your format has no images, accept the flag and no-op.
