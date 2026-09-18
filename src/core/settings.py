"""Configuration loading and validation for the Modular RAG MCP Server."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

# ---------------------------------------------------------------------------
# Repo root & path resolution
# ---------------------------------------------------------------------------
# Anchored to this file's location: <repo>/src/core/settings.py → parents[2]
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Default absolute path to settings.yaml
DEFAULT_SETTINGS_PATH: Path = REPO_ROOT / "config" / "settings.yaml"


def resolve_path(relative: Union[str, Path]) -> Path:
    """Resolve a repo-relative path to an absolute path.

    If *relative* is already absolute it is returned as-is.  Otherwise
    it is resolved against :data:`REPO_ROOT`.

    >>> resolve_path("config/settings.yaml")  # doctest: +SKIP
    PosixPath('/home/user/Modular-RAG-MCP-Server/config/settings.yaml')
    """
    p = Path(relative)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


class SettingsError(ValueError):
    """Raised when settings validation fails."""


def _require_mapping(data: Dict[str, Any], key: str, path: str) -> Dict[str, Any]:
    value = data.get(key)
    if value is None:
        raise SettingsError(f"Missing required field: {path}.{key}")
    if not isinstance(value, dict):
        raise SettingsError(f"Expected mapping for field: {path}.{key}")
    return value


def _require_value(data: Dict[str, Any], key: str, path: str) -> Any:
    if key not in data or data.get(key) is None:
        raise SettingsError(f"Missing required field: {path}.{key}")
    return data[key]


def _require_str(data: Dict[str, Any], key: str, path: str) -> str:
    value = _require_value(data, key, path)
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"Expected non-empty string for field: {path}.{key}")
    return value


def _require_int(data: Dict[str, Any], key: str, path: str) -> int:
    value = _require_value(data, key, path)
    if not isinstance(value, int):
        raise SettingsError(f"Expected integer for field: {path}.{key}")
    return value


def _require_number(data: Dict[str, Any], key: str, path: str) -> float:
    value = _require_value(data, key, path)
    if not isinstance(value, (int, float)):
        raise SettingsError(f"Expected number for field: {path}.{key}")
    return float(value)


def _require_bool(data: Dict[str, Any], key: str, path: str) -> bool:
    value = _require_value(data, key, path)
    if not isinstance(value, bool):
        raise SettingsError(f"Expected boolean for field: {path}.{key}")
    return value


def _require_choice(
    data: Dict[str, Any], key: str, choices: frozenset, path: str
) -> str:
    """Require a string value drawn from *choices* (fail-fast, e.g. OCR mode)."""
    value = _require_value(data, key, path)
    if not isinstance(value, str) or value.lower() not in choices:
        raise SettingsError(
            f"Expected one of {sorted(choices)} for field: {path}.{key}, "
            f"got {value!r}"
        )
    return value.lower()


def _require_list(data: Dict[str, Any], key: str, path: str) -> List[Any]:
    value = _require_value(data, key, path)
    if not isinstance(value, list):
        raise SettingsError(f"Expected list for field: {path}.{key}")
    return value


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    # Azure/OpenAI-specific optional fields
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    # Ollama-specific optional fields
    base_url: Optional[str] = None


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str
    dimensions: int
    # Azure-specific optional fields
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    # Ollama-specific optional fields
    base_url: Optional[str] = None


@dataclass(frozen=True)
class VectorStoreSettings:
    provider: str
    persist_directory: str
    collection_name: str


@dataclass(frozen=True)
class RetrievalSettings:
    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    rrf_k: int
    # T19: query-side references-section filter (post-fusion, pre-truncation).
    # Optional so pre-T19 configs keep parsing (defaults to off).
    filter_references: bool = False
    # T22 D1: 重排候选池深度 = 最终 top_k × 此倍数（原先 eval/MCP/dashboard
    # 三处硬编码 ×2）。可选，缺省 2 = 与旧硬编码行为逐字节一致。
    rerank_pool_multiplier: int = 2


@dataclass(frozen=True)
class RerankSettings:
    enabled: bool
    provider: str
    model: str
    top_k: int


@dataclass(frozen=True)
class EvaluationSettings:
    enabled: bool
    provider: str
    metrics: List[str]
    backends: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservabilitySettings:
    log_level: str
    trace_enabled: bool
    trace_file: str
    structured_logging: bool


@dataclass(frozen=True)
class VisionLLMSettings:
    enabled: bool
    provider: str
    model: str
    max_image_size: int
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    base_url: Optional[str] = None


@dataclass(frozen=True)
class ParseCacheSettings:
    """Docling parse-result cache (D-036): lossless JSON replay for --force.

    A ``--force`` re-ingest of an unchanged file replays the cached docling
    JSON instead of re-paying the CPU parse. Replay only engages when a cache
    entry already exists; the file-level SHA256 skip is untouched.
    """

    enabled: bool = True
    # Repo-relative root; one sub-directory per file SHA256 (gitignored).
    dir: str = "data/parsed"


# Valid values for ingestion.parser.ocr_mode (ticket 03 / D-038). The set is
# duplicated (not imported) in DoclingParser — libs must not depend on core
# at import time (see parser_factory's lazy resolve_path import).
PARSER_OCR_MODES = frozenset({"auto", "always", "never"})


@dataclass(frozen=True)
class ParserSettings:
    """Pluggable document parser configuration (K1, renamed from LoaderSettings).

    Backwards-compatible: when ``ingestion.parser`` is absent the pipeline
    defaults to the ``pdf`` provider with image extraction enabled. The legacy
    ``ingestion.loader`` block is still accepted (deprecated).
    """
    provider: str
    extract_images: bool = True
    # OCR strategy for the docling converter (ticket 03 / D-038):
    # ``auto``    — probe the text layer per file; digital PDFs parse with
    #               OCR off (faster, zero quality loss), no-text-layer files
    #               (scans) get OCR for that file only.
    # ``always``  — force OCR on (pre-D-038 docling default behaviour).
    # ``never``   — force OCR off. The literal set is duplicated (validated)
    # in DoclingParser — libs must not import core at module import time.
    ocr_mode: str = "auto"
    # Optional override of the parser's conversion batching. ``0`` means a
    # single unrestricted convert — the T23 corpus-integrity fix: docling's
    # batched conversion (fresh converter per batch) loses sections
    # intermittently, while single-batch conversion is deterministic and
    # complete on machines with enough RAM (the batching was a 16 GB
    # workaround). ``None`` keeps the parser's built-in default.
    page_batch_size: Optional[int] = None
    # Optional parse-result cache config (D-036). ``None`` keeps the cache
    # off — pre-D-036 behaviour for configs without the block.
    parse_cache: Optional[ParseCacheSettings] = None


@dataclass(frozen=True)
class IngestionSettings:
    chunk_size: int
    chunk_overlap: int
    splitter: str
    batch_size: int
    chunk_refiner: Optional[Dict[str, Any]] = None  # 动态配置
    metadata_enricher: Optional[Dict[str, Any]] = None  # 动态配置
    parser: Optional[ParserSettings] = None  # K1: 可插拔 Parser 配置（缺省默认 pdf）


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    ingestion: Optional[IngestionSettings] = None
    vision_llm: Optional[VisionLLMSettings] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        if not isinstance(data, dict):
            raise SettingsError("Settings root must be a mapping")

        llm = _require_mapping(data, "llm", "settings")
        embedding = _require_mapping(data, "embedding", "settings")
        vector_store = _require_mapping(data, "vector_store", "settings")
        retrieval = _require_mapping(data, "retrieval", "settings")
        # T22 D1: 可选 key——缺省 2（= 旧硬编码行为）；有值则由 _require_int
        # 做类型检查（非 int 报 SettingsError）
        retrieval.setdefault("rerank_pool_multiplier", 2)
        rerank = _require_mapping(data, "rerank", "settings")
        evaluation = _require_mapping(data, "evaluation", "settings")
        observability = _require_mapping(data, "observability", "settings")

        ingestion_settings = None
        if "ingestion" in data:
            ingestion = _require_mapping(data, "ingestion", "settings")
            # K1: pluggable parser config (optional; defaults to pdf provider).
            # Accept "parser" (new) or "loader" (legacy, deprecated pre-K1 configs).
            parser_settings = None
            if "parser" in ingestion:
                parser_cfg = _require_mapping(ingestion, "parser", "ingestion")
                parse_cache_settings = None
                if "parse_cache" in parser_cfg:
                    cache_cfg = _require_mapping(
                        parser_cfg, "parse_cache", "ingestion.parser.parse_cache")
                    parse_cache_settings = ParseCacheSettings(
                        enabled=(
                            _require_bool(
                                cache_cfg, "enabled", "ingestion.parser.parse_cache")
                            if "enabled" in cache_cfg else True
                        ),
                        dir=(
                            _require_str(
                                cache_cfg, "dir", "ingestion.parser.parse_cache")
                            if "dir" in cache_cfg else "data/parsed"
                        ),
                    )
                parser_settings = ParserSettings(
                    provider=_require_str(parser_cfg, "provider", "ingestion.parser"),
                    extract_images=(
                        _require_bool(parser_cfg, "extract_images", "ingestion.parser")
                        if "extract_images" in parser_cfg else True
                    ),
                    page_batch_size=(
                        _require_int(parser_cfg, "page_batch_size", "ingestion.parser")
                        if "page_batch_size" in parser_cfg else None
                    ),
                    ocr_mode=(
                        _require_choice(
                            parser_cfg, "ocr_mode", PARSER_OCR_MODES,
                            "ingestion.parser")
                        if "ocr_mode" in parser_cfg else "auto"
                    ),
                    parse_cache=parse_cache_settings,
                )
            elif "loader" in ingestion:
                # Legacy fallback for pre-K1 configs.
                loader_cfg = _require_mapping(ingestion, "loader", "ingestion")
                parser_settings = ParserSettings(
                    provider=_require_str(loader_cfg, "provider", "ingestion.loader"),
                    extract_images=(
                        _require_bool(loader_cfg, "extract_images", "ingestion.loader")
                        if "extract_images" in loader_cfg else True
                    ),
                )

            ingestion_settings = IngestionSettings(
                chunk_size=_require_int(ingestion, "chunk_size", "ingestion"),
                chunk_overlap=_require_int(ingestion, "chunk_overlap", "ingestion"),
                splitter=_require_str(ingestion, "splitter", "ingestion"),
                batch_size=_require_int(ingestion, "batch_size", "ingestion"),
                chunk_refiner=ingestion.get("chunk_refiner"),  # 可选配置
                metadata_enricher=ingestion.get("metadata_enricher"),  # 可选配置
                parser=parser_settings,
            )

        vision_llm_settings = None
        if "vision_llm" in data:
            vision_llm = _require_mapping(data, "vision_llm", "settings")
            vision_llm_settings = VisionLLMSettings(
                enabled=_require_bool(vision_llm, "enabled", "vision_llm"),
                provider=_require_str(vision_llm, "provider", "vision_llm"),
                model=_require_str(vision_llm, "model", "vision_llm"),
                max_image_size=_require_int(vision_llm, "max_image_size", "vision_llm"),
                api_key=vision_llm.get("api_key"),
                api_version=vision_llm.get("api_version"),
                azure_endpoint=vision_llm.get("azure_endpoint"),
                deployment_name=vision_llm.get("deployment_name"),
                base_url=vision_llm.get("base_url"),
            )

        settings = cls(
            llm=LLMSettings(
                provider=_require_str(llm, "provider", "llm"),
                model=_require_str(llm, "model", "llm"),
                temperature=_require_number(llm, "temperature", "llm"),
                max_tokens=_require_int(llm, "max_tokens", "llm"),
                api_key=llm.get("api_key"),
                api_version=llm.get("api_version"),
                azure_endpoint=llm.get("azure_endpoint"),
                deployment_name=llm.get("deployment_name"),
                base_url=llm.get("base_url"),
            ),
            embedding=EmbeddingSettings(
                provider=_require_str(embedding, "provider", "embedding"),
                model=_require_str(embedding, "model", "embedding"),
                dimensions=_require_int(embedding, "dimensions", "embedding"),
                api_key=embedding.get("api_key"),
                api_version=embedding.get("api_version"),
                azure_endpoint=embedding.get("azure_endpoint"),
                deployment_name=embedding.get("deployment_name"),
                base_url=embedding.get("base_url"),
            ),
            vector_store=VectorStoreSettings(
                provider=_require_str(vector_store, "provider", "vector_store"),
                persist_directory=_require_str(vector_store, "persist_directory", "vector_store"),
                collection_name=_require_str(vector_store, "collection_name", "vector_store"),
            ),
            retrieval=RetrievalSettings(
                dense_top_k=_require_int(retrieval, "dense_top_k", "retrieval"),
                sparse_top_k=_require_int(retrieval, "sparse_top_k", "retrieval"),
                fusion_top_k=_require_int(retrieval, "fusion_top_k", "retrieval"),
                rrf_k=_require_int(retrieval, "rrf_k", "retrieval"),
                filter_references=bool(retrieval.get("filter_references", False)),
                rerank_pool_multiplier=_require_int(
                    retrieval, "rerank_pool_multiplier", "retrieval"),
            ),
            rerank=RerankSettings(
                enabled=_require_bool(rerank, "enabled", "rerank"),
                provider=_require_str(rerank, "provider", "rerank"),
                model=_require_str(rerank, "model", "rerank"),
                top_k=_require_int(rerank, "top_k", "rerank"),
            ),
            evaluation=EvaluationSettings(
                enabled=_require_bool(evaluation, "enabled", "evaluation"),
                provider=_require_str(evaluation, "provider", "evaluation"),
                metrics=[str(item) for item in _require_list(evaluation, "metrics", "evaluation")],
                backends=tuple(str(b) for b in evaluation.get("backends", [])),
            ),
            observability=ObservabilitySettings(
                log_level=_require_str(observability, "log_level", "observability"),
                trace_enabled=_require_bool(observability, "trace_enabled", "observability"),
                trace_file=_require_str(observability, "trace_file", "observability"),
                structured_logging=_require_bool(observability, "structured_logging", "observability"),
            ),
            ingestion=ingestion_settings,
            vision_llm=vision_llm_settings,
        )

        return settings


def validate_settings(settings: Settings) -> None:
    """Validate settings and raise SettingsError if invalid."""

    if not settings.llm.provider:
        raise SettingsError("Missing required field: llm.provider")
    if not settings.embedding.provider:
        raise SettingsError("Missing required field: embedding.provider")
    if not settings.vector_store.provider:
        raise SettingsError("Missing required field: vector_store.provider")
    if not settings.retrieval.rrf_k:
        raise SettingsError("Missing required field: retrieval.rrf_k")
    if settings.retrieval.rerank_pool_multiplier < 1:
        raise SettingsError(
            "Invalid value: retrieval.rerank_pool_multiplier must be >= 1")
    if not settings.rerank.provider:
        raise SettingsError("Missing required field: rerank.provider")
    if not settings.evaluation.provider:
        raise SettingsError("Missing required field: evaluation.provider")
    if not settings.observability.log_level:
        raise SettingsError("Missing required field: observability.log_level")


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from a YAML file and validate required fields.

    Args:
        path: Path to settings YAML.  Defaults to
            ``<repo>/config/settings.yaml`` (absolute, CWD-independent).
    """
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    if not settings_path.is_absolute():
        settings_path = resolve_path(settings_path)
    if not settings_path.exists():
        raise SettingsError(f"Settings file not found: {settings_path}")

    with settings_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    settings = Settings.from_dict(data or {})
    validate_settings(settings)
    return settings