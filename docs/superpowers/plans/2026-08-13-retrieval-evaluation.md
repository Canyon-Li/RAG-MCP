# 检索 + 溯源能力评估闭环 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为项目的检索 + 溯源能力搭建评估闭环 —— golden QA 集(source 级 GT)→ 评估 → 混合指标报告(Source Recall/Precision@5 + RAGAS Context Relevance/Precision),全程不生成答案。

**Architecture:** 改造 `CustomEvaluator` 增加 source 级确定性指标(集合运算);改造 `RagasEvaluator` 增加本地 Ollama judge(granite4.1:8b)并将指标替换为 context_relevance + context_precision;用 `CompositeEvaluator` 组合两者;重写 golden set 为 source 级结构;`EvalRunner` 透传 source GT。所有改动基于代码事实:查询链路无答案生成([response_builder.py](src/core/response/response_response_builder.py) / [query.py](scripts/query.py) 均无 LLM 答案合成)。

**Tech Stack:** Python 3.10+, ragas 0.4.3, ollama(granite4.1:8b, OpenAI-compatible client), pytest, ChromaDB。

## Global Constraints

- **k = 5**:Source Recall@5 / Precision@5 中的 k 固定为 5,通过 `CustomEvaluator` 构造参数 `source_top_k=5` 注入,不硬编码在计算逻辑里、不读 settings。
- **judge 模型**:`granite4.1:8b`,本地 Ollama。端点解析顺序:`explicit > OLLAMA_BASE_URL env > http://localhost:11434/v1`(复用 [ollama_vision_llm.py:86-103](src/libs/llm/ollama_vision_llm.py#L86-L103) 模式)。模型名与端点**不读 settings**(judge 与检索链路 LLM 解耦,硬配置在 evaluator 内)。
- **collection 名**:`evaluation`,ingest 与 evaluate 必须完全一致(BM25 按 collection 名分目录)。
- **RAGAS 导入需 vertexai stub**:ragas 0.4.3 在导入期引用 `langchain_community.chat_models.vertexai`,本环境未安装该包,需在导入前注入 stub 模块(参考 [agentic-rag-for-dummies evaluation.ipynb](D:/Desktop/git/repo/agentic-rag-for-dummies/notebooks/evaluation.ipynb) 的 workaround)。
- **RAGAS metric 调用**:`await scorer.ascore(**kwargs)` 返回 `MetricResult`,分数取 `.value`;`.score(**kwargs)` 是同步版本,返回同一类型。本计划用同步 `.score()` 简化(评估非热点路径)。
- **不生成答案**:`answer_generator` 保持现有 fallback(拼接 chunk 文本),**不接 LLM**。生成的"答案"不参与任何指标计算。
- **parser 保持 docling**:`config/settings.yaml` 的 `ingestion.parser.provider` 不改。
- **Windows + conda**:测试与脚本在 conda env `langchain-test` 下运行;中文输出用 UTF-8。

**参考文档**:[设计 spec](docs/superpowers/specs/2026-08-13-retrieval-evaluation-design.md)

---

## File Structure

| 文件 | 责任 | 动作 |
|---|---|---|
| [src/libs/evaluator/custom_evaluator.py](src/libs/evaluator/custom_evaluator.py) | source 级确定性指标(Source Recall/Precision@k) | 修改 |
| [src/observability/evaluation/ragas_evaluator.py](src/observability/evaluation/ragas_evaluator.py) | RAGAS 语义指标 + 本地 Ollama judge | 修改 |
| [src/observability/evaluation/eval_runner.py](src/observability/evaluation/eval_runner.py) | 编排评估,透传 source GT | 修改 |
| [tests/fixtures/golden_test_set.json](tests/fixtures/golden_test_set.json) | source 级 golden QA 集 | 重写 |
| [config/settings.yaml](config/settings.yaml) | evaluation 配置切到 composite | 修改 |
| [tests/unit/test_custom_evaluator.py](tests/unit/test_custom_evaluator.py) | source 指标单元测试 | 新增测试 |
| [tests/unit/test_ragas_evaluator.py](tests/unit/test_ragas_evaluator.py) | Ollama judge wrapper 单元测试 | 新增测试 |

---

## Task 1: CustomEvaluator 增加 source 级溯源指标

**Files:**
- Modify: `src/libs/evaluator/custom_evaluator.py`
- Test: `tests/unit/test_custom_evaluator.py`

**Interfaces:**
- Consumes: `retrieved_chunks`(每个 chunk 是 dict,带 `source` 字段,参考 [hybrid_search.py:64](src/core/query_engine/hybrid_search.py#L64) 输出);`ground_truth` 为 `dict` 含 `{"sources": [<文件名>, ...]}`
- Produces: `CustomEvaluator` 新增构造参数 `source_top_k: int = 5`;`SUPPORTED_METRICS` 增加 `"source_recall_at_k"` / `"source_precision_at_k"`;`evaluate()` 返回的 dict 新增 `source_recall_at_k` / `source_precision_at_k` 两个 key(值为 0.0–1.0 float)

**source 字段提取规则**:`_extract_sources()` 从每个 chunk 读 `source` 字段(chunk dict 里);若 chunk 是字符串或无 `source` 字段,该项记为 `""`。ground truth 的 sources 从 `ground_truth["sources"]` 读(列表)。

- [ ] **Step 1: 写失败测试 —— source 指标命中场景**

在 `tests/unit/test_custom_evaluator.py` 的 `TestCustomEvaluatorBoundary` 类**之后**新增一个测试类:

```python
class TestCustomEvaluatorSourceMetrics:
    """Tests for source-level retrieval metrics (Source Recall/Precision@k)."""

    def test_source_recall_at_5_hit(self) -> None:
        """top-5 contains the expected source → recall = 1.0."""
        evaluator = CustomEvaluator(
            metrics=["source_recall_at_k", "source_precision_at_k"],
            source_top_k=5,
        )
        retrieved = [
            {"id": "c1", "source": "paperA.pdf"},
            {"id": "c2", "source": "paperB.pdf"},
        ]
        gt = {"sources": ["paperA.pdf"]}

        metrics = evaluator.evaluate("q", retrieved, ground_truth=gt)

        assert metrics["source_recall_at_k"] == 1.0

    def test_source_recall_at_5_miss(self) -> None:
        """top-5 does not contain expected source → recall = 0.0."""
        evaluator = CustomEvaluator(
            metrics=["source_recall_at_k"], source_top_k=5,
        )
        retrieved = [
            {"id": "c1", "source": "paperB.pdf"},
            {"id": "c2", "source": "paperC.pdf"},
        ]
        gt = {"sources": ["paperA.pdf"]}

        metrics = evaluator.evaluate("q", retrieved, ground_truth=gt)

        assert metrics["source_recall_at_k"] == 0.0

    def test_source_precision_at_5(self) -> None:
        """2 retrieved, 1 matches → precision = 1/2 = 0.5."""
        evaluator = CustomEvaluator(
            metrics=["source_precision_at_k"], source_top_k=5,
        )
        retrieved = [
            {"id": "c1", "source": "paperA.pdf"},
            {"id": "c2", "source": "paperB.pdf"},
        ]
        gt = {"sources": ["paperA.pdf"]}

        metrics = evaluator.evaluate("q", retrieved, ground_truth=gt)

        assert metrics["source_precision_at_k"] == 0.5

    def test_source_top_k_truncation(self) -> None:
        """Only top-k chunks considered for precision; k=2, both match."""
        evaluator = CustomEvaluator(
            metrics=["source_precision_at_k"], source_top_k=2,
        )
        retrieved = [
            {"id": "c1", "source": "paperA.pdf"},
            {"id": "c2", "source": "paperA.pdf"},
            {"id": "c3", "source": "paperB.pdf"},  # beyond k, ignored
        ]
        gt = {"sources": ["paperA.pdf"]}

        metrics = evaluator.evaluate("q", retrieved, ground_truth=gt)

        assert metrics["source_precision_at_k"] == 1.0

    def test_source_metrics_empty_gt_returns_zero(self) -> None:
        """Empty expected_sources → recall/precision = 0.0 (not error)."""
        evaluator = CustomEvaluator(
            metrics=["source_recall_at_k", "source_precision_at_k"],
            source_top_k=5,
        )
        retrieved = [{"id": "c1", "source": "paperA.pdf"}]
        gt = {"sources": []}

        metrics = evaluator.evaluate("q", retrieved, ground_truth=gt)

        assert metrics["source_recall_at_k"] == 0.0
        assert metrics["source_precision_at_k"] == 0.0

    def test_source_top_k_default_is_5(self) -> None:
        """Constructor without source_top_k defaults to 5."""
        evaluator = CustomEvaluator(metrics=["source_recall_at_k"])
        assert evaluator.source_top_k == 5
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n langchain-test python -m pytest tests/unit/test_custom_evaluator.py::TestCustomEvaluatorSourceMetrics -v`
Expected: FAIL —— `source_recall_at_k` 不在 `SUPPORTED_METRICS`,`CustomEvaluator(metrics=["source_recall_at_k"])` 抛 `ValueError: Unsupported custom metrics`。

- [ ] **Step 3: 修改 CustomEvaluator —— 加 source 指标**

打开 [src/libs/evaluator/custom_evaluator.py](src/libs/evaluator/custom_evaluator.py),做以下修改:

**(a) 扩展 `SUPPORTED_METRICS` 和 `_SOURCE_FIELDS`**(在 `SUPPORTED_METRICS` 下方):

```python
    SUPPORTED_METRICS = {"hit_rate", "mrr", "source_recall_at_k", "source_precision_at_k"}
    _ID_FIELDS = ("id", "chunk_id", "document_id", "doc_id")
    _SOURCE_FIELDS = ("source", "source_path")
```

**(b) 修改 `__init__`** 增加 `source_top_k` 参数(替换现有 `__init__` 整个方法):

```python
    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        source_top_k: int = 5,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.kwargs = kwargs
        self.source_top_k = source_top_k

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalized = [str(metric).strip().lower() for metric in (metrics or [])]
        if not normalized:
            normalized = ["hit_rate", "mrr"]

        unsupported = [metric for metric in normalized if metric not in self.SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                "Unsupported custom metrics: "
                f"{', '.join(unsupported)}. Supported: {', '.join(sorted(self.SUPPORTED_METRICS))}"
            )

        self.metrics = normalized
```

**(c) 在 `evaluate` 方法里增加 source 指标分支**(在 `if "mrr" in self.metrics:` 之后、`return results` 之前插入):

```python
        if "source_recall_at_k" in self.metrics or "source_precision_at_k" in self.metrics:
            retrieved_sources = self._extract_sources(retrieved_chunks)
            gt_sources = self._extract_ground_truth_sources(ground_truth)
            top_k_sources = retrieved_sources[: self.source_top_k]

            if "source_recall_at_k" in self.metrics:
                results["source_recall_at_k"] = self._compute_source_recall(
                    top_k_sources, gt_sources
                )
            if "source_precision_at_k" in self.metrics:
                results["source_precision_at_k"] = self._compute_source_precision(
                    top_k_sources, gt_sources
                )
```

**(d) 增加辅助方法**(在 `_compute_mrr` 方法之后追加):

```python
    def _extract_sources(self, chunks: Iterable[Any]) -> List[str]:
        """Extract source file identifiers from retrieved chunks.

        Reads the ``source`` (or ``source_path``) field from each chunk dict.
        Missing fields yield an empty string so the slot count matches the
        retrieval order.
        """
        sources: List[str] = []
        for item in chunks:
            if isinstance(item, dict):
                value = ""
                for field in self._SOURCE_FIELDS:
                    if field in item and item[field]:
                        value = str(item[field])
                        break
                sources.append(value)
            elif hasattr(item, "source"):
                sources.append(str(getattr(item, "source")))
            else:
                sources.append("")
        return sources

    def _extract_ground_truth_sources(self, ground_truth: Optional[Any]) -> List[str]:
        """Extract expected source identifiers from ground_truth.

        Accepts a dict shaped as ``{"sources": [<file>, ...]}`` or a bare list.
        """
        if ground_truth is None:
            return []
        if isinstance(ground_truth, dict):
            sources = ground_truth.get("sources", [])
            return [str(s) for s in sources] if sources else []
        if isinstance(ground_truth, list):
            return [str(s) for s in ground_truth]
        return []

    @staticmethod
    def _compute_source_recall(
        retrieved_sources: Sequence[str],
        gt_sources: Sequence[str],
    ) -> float:
        """Source Recall@k: 1.0 if any expected source appears in top-k, else 0.0.

        Binary recall — 'did we find the right paper at all'.
        """
        if not gt_sources:
            return 0.0
        gt_set = set(gt_sources)
        return 1.0 if any(s in gt_set for s in retrieved_sources if s) else 0.0

    @staticmethod
    def _compute_source_precision(
        retrieved_sources: Sequence[str],
        gt_sources: Sequence[str],
    ) -> float:
        """Source Precision@k: fraction of top-k chunks whose source is expected."""
        if not retrieved_sources:
            return 0.0
        gt_set = set(gt_sources)
        hits = sum(1 for s in retrieved_sources if s and s in gt_set)
        return hits / len(retrieved_sources)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n langchain-test python -m pytest tests/unit/test_custom_evaluator.py -v`
Expected: 全部 PASS(包括原有的 hit_rate/mrr 测试 + 新增的 6 个 source 测试)。

- [ ] **Step 5: 提交**

```bash
git add src/libs/evaluator/custom_evaluator.py tests/unit/test_custom_evaluator.py
git commit -m "feat(eval): CustomEvaluator 增加 source 级溯源指标 Recall/Precision@k"
```

---

## Task 2: EvalRunner 透传 source ground truth

**Files:**
- Modify: `src/observability/evaluation/eval_runner.py`
- Test: `tests/unit/test_eval_runner.py`

**Interfaces:**
- Consumes: `GoldenTestCase.expected_sources`(已存在,[eval_runner.py:40](src/observability/evaluation/eval_runner.py#L40))
- Produces: `_evaluate_single` 传给 evaluator 的 `ground_truth` 从仅 `{"ids": ...}` 扩展为同时含 `{"ids": ..., "sources": [...]}`;CustomEvaluator 读取 `sources`(Task 1 已实现)

**背景**:`EvalRunner` 当前在 [eval_runner.py:288-293](src/observability/evaluation/eval_runner.py#L288-L293) 只构造 `{"ids": expected_chunk_ids}`,source 信息丢失。需同时传 `sources`。

- [ ] **Step 1: 写失败测试 —— source GT 透传**

在 `tests/unit/test_eval_runner.py` 末尾新增(若文件已有 import 块则复用;若没有 `CustomEvaluator` import 则加):

```python
from src.libs.evaluator.custom_evaluator import CustomEvaluator


class TestEvalRunnerSourceGT:
    """Tests that EvalRunner forwards expected_sources to the evaluator."""

    def test_source_ground_truth_forwarded(self) -> None:
        """EvalRunner should include expected_sources in ground_truth dict."""
        from src.observability.evaluation.eval_runner import (
            EvalRunner, GoldenTestCase,
        )

        # A fake hybrid_search returning one chunk with a known source
        fake_chunk = {"id": "c1", "source": "paperA.pdf", "text": "some text"}
        fake_search = MagicMock()
        fake_search.search.return_value = [fake_chunk]

        evaluator = CustomEvaluator(
            metrics=["source_recall_at_k"], source_top_k=5,
        )
        runner = EvalRunner(
            settings=None, hybrid_search=fake_search, evaluator=evaluator,
        )

        # Write a tiny golden set to a temp file
        import json, tempfile, os
        golden = {
            "test_cases": [
                {
                    "query": "which paper?",
                    "expected_sources": ["paperA.pdf"],
                    "expected_chunk_ids": [],
                }
            ]
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(golden, f)
            tmp_path = f.name

        try:
            report = runner.run(tmp_path, top_k=5)
        finally:
            os.unlink(tmp_path)

        # source_recall should be 1.0 because paperA.pdf was retrieved
        assert report.query_results[0].metrics["source_recall_at_k"] == 1.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n langchain-test python -m pytest tests/unit/test_eval_runner.py::TestEvalRunnerSourceGT -v`
Expected: FAIL —— `source_recall_at_k` 为 0.0(source GT 未透传,`_extract_ground_truth_sources` 收到空列表)。

- [ ] **Step 3: 修改 EvalRunner 透传 sources**

打开 [src/observability/evaluation/eval_runner.py](src/observability/evaluation/eval_runner.py),找到 `_evaluate_single` 里的 ground truth 构造段([:288-293](src/observability/evaluation/eval_runner.py#L288-L293)):

```python
        # Step 3: Build ground truth
        ground_truth = (
            {"ids": test_case.expected_chunk_ids}
            if test_case.expected_chunk_ids
            else None
        )
```

替换为(始终传 dict,合并 ids + sources):

```python
        # Step 3: Build ground truth — always a dict so source-level metrics
        # (which read ground_truth["sources"]) work even when chunk ids absent.
        ground_truth: Dict[str, Any] = {}
        if test_case.expected_chunk_ids:
            ground_truth["ids"] = test_case.expected_chunk_ids
        if test_case.expected_sources:
            ground_truth["sources"] = test_case.expected_sources
        if not ground_truth:
            ground_truth = None
```

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n langchain-test python -m pytest tests/unit/test_eval_runner.py -v`
Expected: 全部 PASS(原有测试 + 新增 source GT 测试)。

- [ ] **Step 5: 提交**

```bash
git add src/observability/evaluation/eval_runner.py tests/unit/test_eval_runner.py
git commit -m "feat(eval): EvalRunner 透传 expected_sources 到 evaluator ground_truth"
```

---

## Task 3: RagasEvaluator 增加 Ollama judge 分支

**Files:**
- Modify: `src/observability/evaluation/ragas_evaluator.py`
- Test: `tests/unit/test_ragas_evaluator.py`

**Interfaces:**
- Consumes: `AsyncOpenAI`(openai SDK),`llm_factory`(ragas.llms),`os.environ["OLLAMA_BASE_URL"]`
- Produces: `RagasEvaluator._build_wrappers()` 支持 `provider == "ollama"` 分支;新增类常量 `_JUDGE_MODEL = "granite4.1:8b"`、`_JUDGE_DEFAULT_BASE_URL = "http://localhost:11434/v1"`

**注意**:本任务只加 Ollama 分支,**不改现有 azure/openai 分支、不改指标**(指标替换在 Task 4)。这样若 Task 4 出问题可单独回退。

**关键决策**:judge 配置不读 `settings.llm`(那是检索链路的 LLM)。judge 模型名和端点硬编码为类常量 + env var,与检索链路解耦。`_build_wrappers` 原先从 `self.settings.llm` 读 provider —— 我们改为:优先检查 env `RAGAS_JUDGE_PROVIDER`(默认 `ollama`),只有显式设为 `azure`/`openai` 时才走原有云端逻辑。

- [ ] **Step 1: 写失败测试 —— Ollama wrapper 构建(mock 网络)**

在 `tests/unit/test_ragas_evaluator.py` 末尾新增(若文件没有这些 import 则加到顶部):

```python
from unittest.mock import patch, MagicMock


class TestRagasOllamaJudge:
    """Tests that RagasEvaluator builds an Ollama-backed judge wrapper."""

    def test_build_wrappers_ollama_branch(self, monkeypatch) -> None:
        """_build_wrappers with ollama provider creates AsyncOpenAI with ollama base_url."""
        # Force ollama branch
        monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Mock at the source modules — _build_wrappers imports these names
        # function-locally (from openai / ragas.llms), so module-attribute
        # patches on the evaluator module won't resolve. Patch the origin.
        with patch("ragas.llms.llm_factory") as mock_factory, patch(
            "openai.AsyncOpenAI"
        ) as mock_openai:
            mock_factory.return_value = MagicMock(name="ragas_llm")
            mock_openai.return_value = MagicMock(name="openai_client")

            # Build a minimal settings stub — _build_wrappers reads settings.embedding
            # for the embeddings wrapper, but in ollama branch we reuse the same client
            from src.observability.evaluation.ragas_evaluator import RagasEvaluator
            settings = MagicMock()
            settings.llm.provider = "ollama"
            settings.embedding.provider = "ollama"

            evaluator = RagasEvaluator.__new__(RagasEvaluator)  # bypass __init__ (ragas import)
            evaluator.settings = settings

            llm, embeddings = evaluator._build_wrappers()

            # AsyncOpenAI called with the ollama base_url (/v1 appended)
            call_kwargs = mock_openai.call_args.kwargs
            assert "localhost:11434" in call_kwargs["base_url"]
            assert call_kwargs["api_key"] == "ollama"
            # llm_factory called with granite model
            factory_args = mock_factory.call_args.args
            assert "granite4.1:8b" in factory_args
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n langchain-test python -m pytest tests/unit/test_ragas_evaluator.py::TestRagasOllamaJudge -v`
Expected: FAIL —— `_build_wrappers` 当前对 `provider == "ollama"` 抛 `ValueError: Unsupported LLM provider`([ragas_evaluator.py:232](src/observability/evaluation/ragas_evaluator.py#L232))。

- [ ] **Step 3: 修改 RagasEvaluator —— 加 Ollama judge 分支**

打开 [src/observability/evaluation/ragas_evaluator.py](src/observability/evaluation/ragas_evaluator.py)。

**(a) 增加 import 和类常量**(在文件顶部 import 区加 `import os`;在 `class RagasEvaluator` 内、`def __init__` 之前加类常量):

```python
import os  # 若文件顶部已有则跳过
```

类常量(加在 `class RagasEvaluator(BaseEvaluator):` 之后、docstring 之后):

```python
    # Judge LLM is decoupled from the retrieval pipeline's settings.llm.
    # Configured via env vars so the judge is stable across provider swaps.
    _JUDGE_MODEL = "granite4.1:8b"
    _JUDGE_DEFAULT_BASE_URL = "http://localhost:11434/v1"

    @staticmethod
    def _resolve_judge_provider() -> str:
        """Which provider family to use for the judge LLM.

        Reads RAGAS_JUDGE_PROVIDER (default 'ollama'). Set to 'azure' or
        'openai' to use a cloud judge instead.
        """
        return os.environ.get("RAGAS_JUDGE_PROVIDER", "ollama").lower()

    def _resolve_ollama_base_url(self) -> str:
        """Ollama endpoint: OLLAMA_BASE_URL env > default, with /v1 suffix."""
        env_url = os.environ.get("OLLAMA_BASE_URL", "").rstrip("/")
        if not env_url:
            return self._JUDGE_DEFAULT_BASE_URL
        if env_url.endswith("/v1"):
            return env_url
        return f"{env_url}/v1"
```

**(b) 改写 `_build_wrappers`** —— 在方法最前面插入 ollama 分支(保留原有 azure/openai 逻辑作为 fallback)。找到 `def _build_wrappers(self) -> tuple:` 方法,在 `# ── LLM ──` 注释行**之前**插入:

```python
        judge_provider = self._resolve_judge_provider()
        if judge_provider == "ollama":
            base_url = self._resolve_ollama_base_url()
            client = AsyncOpenAI(base_url=base_url, api_key="ollama")
            llm = llm_factory(self._JUDGE_MODEL, client=client, max_tokens=8192)
            # Ollama embeddings for AnswerRelevancy (if used) — reuse same endpoint.
            # nomic-embed-text is the project's embedding model.
            embeddings = OpenAIEmbeddings(
                model="nomic-embed-text", client=client,
            )
            return llm, embeddings

        # Fallback: original cloud-judge logic (azure/openai) below
```

> 说明:`AsyncOpenAI`、`llm_factory`、`OpenAIEmbeddings` 已在 [:204-206](src/observability/evaluation/ragas_evaluator.py#L204-L206) 函数内 import。但类方法 `_resolve_judge_provider` 是 staticmethod 不需要它们。类常量 `_JUDGE_MODEL` 等不依赖运行时 import,安全。

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n langchain-test python -m pytest tests/unit/test_ragas_evaluator.py::TestRagasOllamaJudge -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/observability/evaluation/ragas_evaluator.py tests/unit/test_ragas_evaluator.py
git commit -m "feat(eval): RagasEvaluator 增加本地 Ollama judge 分支(granite4.1:8b)"
```

---

## Task 4: RagasEvaluator 指标替换为 Context Relevance + Context Precision

**Files:**
- Modify: `src/observability/evaluation/ragas_evaluator.py`
- Test: `tests/unit/test_ragas_evaluator.py`

**Interfaces:**
- Consumes: `ragas.metrics.collections.ContextRelevance` / `ContextPrecision`(ragas 0.4.3,已验证可导入,需 vertexai stub)
- Produces: `SUPPORTED_METRICS = {"context_relevance", "context_precision"}`;`_run_ragas` 用 `ContextRelevance(llm=llm).score(user_input=, retrieved_contexts=)` 和 `ContextPrecision(llm=llm).score(user_input=, response=, retrieved_contexts=)`;**不再要求 generated_answer 非空**(context_relevance 不需要 answer)

**关键变更**:
1. `SUPPORTED_METRICS` 常量替换。
2. `FAITHFULNESS` / `ANSWER_RELEVANCY` 常量删除,新增 `CONTEXT_RELEVANCE`。
3. `_run_ragas` 的 metric 分支重写。
4. `evaluate()` 里"generated_answer 非空"的硬要求([:132-136](src/observability/evaluation/ragas_evaluator.py#L132-L136))放宽:context_relevance 不需要 answer,但 context_precision 需要 response——由于系统不生成答案,context_precision 的 `response` 传空串(让 judge 只看 retrieved_contexts 与 query 的关系)。这是本场景的权衡:context_precision 退化为"检索结果排序质量"判断。
5. 顶部加 vertexai stub workaround。

- [ ] **Step 1: 写失败测试 —— context_relevance 指标路由(mock judge)**

在 `tests/unit/test_ragas_evaluator.py` 新增:

```python
class TestRagasMetricRouting:
    """Tests that RagasEvaluator routes to context_relevance / context_precision."""

    def test_supported_metrics_replaced(self) -> None:
        """SUPPORTED_METRICS should now be context_relevance + context_precision."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        assert RagasEvaluator.SUPPORTED_METRICS == {
            "context_relevance", "context_precision",
        }

    def test_context_relevance_called(self) -> None:
        """_run_ragas with context_relevance should call ContextRelevance.score."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator._metric_names = ["context_relevance"]
        evaluator.settings = MagicMock()

        fake_metric = MagicMock()
        fake_result = MagicMock()
        fake_result.value = 0.8
        fake_metric.score.return_value = fake_result

        with patch(
            "ragas.metrics.collections.ContextRelevance"
        ) as mock_cr, patch.object(
            evaluator, "_build_wrappers", return_value=(MagicMock(), MagicMock())
        ):
            mock_cr.return_value = fake_metric
            scores = evaluator._run_ragas(
                query="q", contexts=["ctx text"], answer="",
            )

        assert scores["context_relevance"] == 0.8
        mock_cr.return_value.score.assert_called_once()

    def test_no_answer_required_for_context_relevance(self) -> None:
        """evaluate() should NOT raise when generated_answer is empty."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator._metric_names = ["context_relevance"]
        evaluator.settings = MagicMock()

        with patch.object(
            evaluator, "_run_ragas", return_value={"context_relevance": 0.7}
        ) as mock_run:
            metrics = evaluator.evaluate(
                query="q",
                retrieved_chunks=[{"id": "c1", "text": "ctx"}],
                generated_answer="",
            )

        assert metrics["context_relevance"] == 0.7
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n langchain-test python -m pytest tests/unit/test_ragas_evaluator.py::TestRagasMetricRouting -v`
Expected: FAIL —— `SUPPORTED_METRICS` 还是旧的 faithfulness/answer_relevancy/context_precision;`evaluate("")` 抛 `ValueError: requires a non-empty 'generated_answer'`。

- [ ] **Step 3a: 加 vertexai stub workaround**

打开 [src/observability/evaluation/ragas_evaluator.py](src/observability/evaluation/ragas_evaluator.py),在 `from __future__ import annotations` 之后、其它 import 之前插入:

```python
# ── ragas 0.4.3 import workaround ──────────────────────────────────
# ragas eagerly imports langchain_community.chat_models.vertexai during its
# own import; that package is not installed here. Inject a stub so the import
# succeeds. (Mirrors the agentic-rag-for-dummies evaluation notebook.)
import sys as _sys
import types as _types
try:  # pragma: no cover — only triggers when langchain_community is missing vertexai
    import langchain_community.chat_models.vertexai  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    if "langchain_community.chat_models.vertexai" not in _sys.modules:
        _stub = _types.ModuleType("langchain_community.chat_models.vertexai")

        class _ChatVertexAI:  # minimal stub
            pass

        _stub.ChatVertexAI = _ChatVertexAI  # type: ignore
        _sys.modules["langchain_community.chat_models.vertexai"] = _stub
```

- [ ] **Step 3b: 替换 metric 常量**

找到(文件上部):

```python
FAITHFULNESS = "faithfulness"
ANSWER_RELEVANCY = "answer_relevancy"
CONTEXT_PRECISION = "context_precision"

SUPPORTED_METRICS = {FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION}
```

替换为:

```python
CONTEXT_RELEVANCE = "context_relevance"
CONTEXT_PRECISION = "context_precision"

SUPPORTED_METRICS = {CONTEXT_RELEVANCE, CONTEXT_PRECISION}
```

- [ ] **Step 3c: 重写 `_run_ragas` 的 metric 分支**

找到 `def _run_ragas` 内的 import 和 for 循环([:163-193](src/observability/evaluation/ragas_evaluator.py#L163-L193)):

```python
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecisionWithoutReference,
        )

        # Build LLM / Embedding wrappers from settings
        llm, embeddings = self._build_wrappers()

        scores: Dict[str, float] = {}

        for metric_name in self._metric_names:
            if metric_name == FAITHFULNESS:
                m = Faithfulness(llm=llm)
                result = m.score(
                    user_input=query, response=answer, retrieved_contexts=contexts,
                )
            elif metric_name == ANSWER_RELEVANCY:
                m = AnswerRelevancy(llm=llm, embeddings=embeddings)
                result = m.score(user_input=query, response=answer)
            elif metric_name == CONTEXT_PRECISION:
                m = ContextPrecisionWithoutReference(llm=llm)
                result = m.score(
                    user_input=query, response=answer, retrieved_contexts=contexts,
                )
            else:
                continue

            scores[metric_name] = float(result.value) if result.value is not None else 0.0

        return scores
```

替换为:

```python
        from ragas.metrics.collections import ContextRelevance, ContextPrecision

        # Build LLM / Embedding wrappers from settings (ollama judge by default)
        llm, embeddings = self._build_wrappers()

        scores: Dict[str, float] = {}

        for metric_name in self._metric_names:
            if metric_name == CONTEXT_RELEVANCE:
                m = ContextRelevance(llm=llm)
                result = m.score(
                    user_input=query, retrieved_contexts=contexts,
                )
            elif metric_name == CONTEXT_PRECISION:
                m = ContextPrecision(llm=llm)
                # This system does not generate answers; pass response as the
                # query text so the judge ranks contexts by query relevance.
                result = m.score(
                    user_input=query,
                    response=answer or query,
                    retrieved_contexts=contexts,
                )
            else:
                continue

            scores[metric_name] = float(result.value) if result.value is not None else 0.0

        return scores
```

- [ ] **Step 3d: 放宽 evaluate() 对 generated_answer 的要求**

找到 [eval_runner.py:132-136](src/observability/evaluation/ragas_evaluator.py#L132-L136) 这段(在 `evaluate` 方法内):

```python
        if not generated_answer or not generated_answer.strip():
            raise ValueError(
                "RagasEvaluator requires a non-empty 'generated_answer'. "
                "Ragas uses LLM-as-Judge and needs the answer text to evaluate."
            )
```

替换为(去掉硬要求,因为 context_relevance 不需要 answer):

```python
        # Note: this system does not generate answers (retrieval + citation only).
        # context_relevance needs no answer; context_precision receives query as
        # response fallback (see _run_ragas). So empty answer is allowed.
```

- [ ] **Step 3e: 更新 `_metrics_from_settings` 过滤**

找到 `_metrics_from_settings` 末尾([:301](src/observability/evaluation/ragas_evaluator.py#L301)):

```python
        return [m for m in raw_metrics if m.lower() in SUPPORTED_METRICS]
```

这句本身正确(过滤到新 SUPPORTED_METRICS),**不用改**。但确认 `__init__` 里的 `normalised` 默认值会取到新指标 —— 当 `metrics=None` 且 settings 无 metrics 时,[:90-91](src/observability/evaluation/ragas_evaluator.py#L90-L91) `normalised = sorted(SUPPORTED_METRICS)` 会变成 `["context_precision", "context_relevance"]`,正确。

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n langchain-test python -m pytest tests/unit/test_ragas_evaluator.py -v`
Expected: 全部 PASS。原有测试中引用 `FAITHFULNESS` / `ANSWER_RELEVANCY` 的测试(如果有)会因常量删除而 error —— 需在 Step 4 后检查并修正(见 Step 4b)。

- [ ] **Step 4b: 修正原有测试中对旧常量的引用**

如果 `tests/unit/test_ragas_evaluator.py` 里有测试引用了 `FAITHFULNESS` / `ANSWER_RELEVANCY` 或断言旧 `SUPPORTED_METRICS`,需更新为新的 context_relevance / context_precision。检查命令:

```bash
grep -n "FAITHFULNESS\|ANSWER_RELEVANCY\|faithfulness\|answer_relevancy" tests/unit/test_ragas_evaluator.py
```

对命中的测试:若是测"不支持指标报错",改成用 `"bogus_metric"` 触发;若是测旧指标行为,删除或改测 context_relevance。

重新运行:
Run: `conda run -n langchain-test python -m pytest tests/unit/test_ragas_evaluator.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add src/observability/evaluation/ragas_evaluator.py tests/unit/test_ragas_evaluator.py
git commit -m "feat(eval): RagasEvaluator 指标替换为 context_relevance + context_precision(去答案类)"
```

---

## Task 5: 重写 golden_test_set.json 为 source 级结构

**Files:**
- Modify: `tests/fixtures/golden_test_set.json`

**Interfaces:**
- Produces: golden set 符合 `load_test_set()`([eval_runner.py:113-138](src/observability/evaluation/eval_runner.py#L113-L138)) 解析的 schema;每条 test_case 含 `query` / `expected_sources` / `expected_chunk_ids`(空)/ `reference`

**说明**:真正的 golden set 内容取决于你 ingest 哪些论文。这里写一个**结构模板**占位(3 条示例),实际 QA 内容在 ingest 论文后由你填充。模板的价值是:让 Task 6 的端到端验证有可用 fixture,即使论文还没选也能跑通流程(用论文文件名占位)。

> ⚠️ 这一步不遵循严格 TDD(它是 fixture 数据,不是代码)。但写入后需验证 `load_test_set` 能正确解析。

- [ ] **Step 1: 重写 golden_test_set.json**

将 [tests/fixtures/golden_test_set.json](tests/fixtures/golden_test_set.json) 全部内容替换为:

```json
{
  "description": "Source-level golden test set for retrieval + provenance evaluation. expected_sources lists the paper file(s) that should appear in top-k retrieval for each query. reference is a short ground-truth answer used only as judge input for RAGAS context_precision (the system does not generate answers).",
  "version": "2.0",
  "test_cases": [
    {
      "query": "PLACEHOLDER — replace with real query about paper 1",
      "expected_sources": ["paper_1.pdf"],
      "expected_chunk_ids": [],
      "reference": "PLACEHOLDER — short reference answer (English, judge input only)."
    },
    {
      "query": "PLACEHOLDER — replace with real query about paper 2",
      "expected_sources": ["paper_2.pdf"],
      "expected_chunk_ids": [],
      "reference": "PLACEHOLDER — short reference answer (English, judge input only)."
    },
    {
      "query": "PLACEHOLDER — replace with real query about paper 3",
      "expected_sources": ["paper_3.pdf"],
      "expected_chunk_ids": [],
      "reference": "PLACEHOLDER — short reference answer (English, judge input only)."
    }
  ]
}
```

- [ ] **Step 2: 验证 load_test_set 能解析**

Run:
```bash
conda run -n langchain-test python -c "from src.observability.evaluation.eval_runner import load_test_set; tcs = load_test_set('tests/fixtures/golden_test_set.json'); print(f'{len(tcs)} test cases loaded'); print(tcs[0].expected_sources)"
```
Expected: 输出 `3 test cases loaded` 和 `['paper_1.pdf']`。

- [ ] **Step 3: 提交**

```bash
git add tests/fixtures/golden_test_set.json
git commit -m "feat(eval): golden set 重写为 source 级结构(v2.0,模板占位待填真实 QA)"
```

---

## Task 6: config/settings.yaml 切到 CompositeEvaluator 并端到端验证

**Files:**
- Modify: `config/settings.yaml`

**Interfaces:**
- Consumes: Task 1(source 指标)+ Task 3/4(RagasEvaluator ollama + context 指标)
- Produces: `evaluation.provider: composite`,`evaluation.backends: [custom, ragas]`,`evaluation.metrics` 含 source 指标 + context 指标

**背景**:CompositeEvaluator 的 `_build_from_settings`([composite_evaluator.py:163-230](src/observability/evaluation/composite_evaluator.py#L163-L230))会按 `evaluation.backends` 列表分别创建 custom / ragas 子 evaluator,各子 evaluator 从 `evaluation.metrics` 取自己支持的指标。但注意:custom 的 source 指标需要 `source_top_k`,而 CompositeEvaluator 用 MagicMock 包装 settings 传给子 evaluator([:212-218](src/observability/evaluation/composite_evaluator.py#L212-L218)),**不会传 `source_top_k`**。CustomEvaluator 的 `__init__` 有默认 `source_top_k=5`,所以走默认值即可 —— 无需改 CompositeEvaluator。

- [ ] **Step 1: 修改 settings.yaml 的 evaluation 段**

打开 [config/settings.yaml](config/settings.yaml),找到 `evaluation:` 段([:75-80](config/settings.yaml#L75-L80)):

```yaml
evaluation:
  enabled: false
  provider: "custom"
  metrics:
    - "hit_rate"
    - "mrr"
    - "faithfulness"
```

替换为:

```yaml
evaluation:
  enabled: true
  provider: "composite"
  backends:
    - "custom"
    - "ragas"
  metrics:
    - "source_recall_at_k"
    - "source_precision_at_k"
    - "context_relevance"
    - "context_precision"
```

- [ ] **Step 2: 验证 settings 加载 + evaluator 创建(不需网络)**

Run:
```bash
conda run -n langchain-test python -c "from src.core.settings import load_settings; from src.libs.evaluator.evaluator_factory import EvaluatorFactory; s = load_settings(); e = EvaluatorFactory.create(s); print(type(e).__name__); print([type(sub).__name__ for sub in e.evaluators])"
```
Expected: 输出 `CompositeEvaluator` 和 `['CustomEvaluator', 'RagasEvaluator']`。

> 若报 ragas import 错,确认 Task 4 的 vertexai stub 已写入。

- [ ] **Step 3: 准备测试论文 + ingest(需要用户操作 + Ollama 在线)**

> 这一步需要你放入真实论文 PDF。创建目录 `tests/fixtures/eval_docs/`,放入 ≥1 篇英文论文 PDF。

Run(ingest 到 evaluation collection):
```bash
conda run -n langchain-test python scripts/ingest.py --path tests/fixtures/eval_docs/ --collection evaluation
```
Expected: 输出 ingestion summary,`[OK] Successful: N`,chunk 数 > 0。

- [ ] **Step 4: 用真实论文文件名更新 golden_test_set.json**

查看 ingest 后实际的 source 文件名:
```bash
conda run -n langchain-test python -c "import chromadb; c = chromadb.PersistentClient(path='data/db/chroma'); col = c.get_collection('evaluation'); metas = col.get(limit=5, include=['metadatas'])['metadatas']; sources = set(m.get('source_path','?') for m in metas if m); print(sources)"
```
把 golden_test_set.json 里的 `paper_1.pdf` 等占位换成真实文件名,并写真实 query / reference。

- [ ] **Step 5: 端到端跑评估(需 Ollama granite4.1:8b 在线)**

Run:
```bash
conda run -n langchain-test python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json --collection evaluation --json
```
Expected: JSON 报告,`aggregate_metrics` 含 `source_recall_at_k` / `source_precision_at_k` / `context_relevance` / `context_precision` 四个 key(RAGAS 指标首次跑可能较慢,granite 逐条 judge)。

- [ ] **Step 6: 提交**

```bash
git add config/settings.yaml
git commit -m "feat(eval): evaluation 配置切到 composite(custom source 指标 + ragas context 指标)"
```

---

## Task 7: 文档校准 + 完成验证

**Files:**
- Modify: `.claude/rules/extending-backends.md`(若 Evaluator 行需更新)
- Modify: `CLAUDE.md`(若评估命令描述需更新)

**说明**:这是收尾,确保文档反映新的评估能力。

- [ ] **Step 1: 检查 extending-backends.md 的 Evaluator 行**

打开 [.claude/rules/extending-backends.md](.claude/rules/extending-backends.md),看 Evaluator 行。当前标注 "(scaffolded — see factory)"。更新为:

```markdown
| Evaluator | `BaseEvaluator` (`libs/evaluator/`) | `EvaluatorFactory` | custom, ragas(lazy), composite(lazy) |
```

- [ ] **Step 2: 跑全套评估相关单元测试**

Run:
```bash
conda run -n langchain-test python -m pytest tests/unit/test_custom_evaluator.py tests/unit/test_ragas_evaluator.py tests/unit/test_eval_runner.py tests/unit/test_composite_evaluator.py -v
```
Expected: 全部 PASS。

- [ ] **Step 3: 提交文档**

```bash
git add .claude/rules/extending-backends.md
git commit -m "docs(rules): Evaluator provider 行校准——custom/ragas/composite"
```

---

## 实施顺序与依赖

```
Task 1 (CustomEvaluator source 指标)
   ↓
Task 2 (EvalRunner 透传 source GT)  ← 依赖 Task 1 的 source 指标
   ↓
Task 3 (RagasEvaluator ollama judge) ← 独立,可并行于 Task 1/2
   ↓
Task 4 (RagasEvaluator 指标替换)     ← 依赖 Task 3 的 ollama 分支
   ↓
Task 5 (golden set 重写)             ← 独立
   ↓
Task 6 (config + 端到端)             ← 依赖 Task 1-5 全部
   ↓
Task 7 (文档收尾)
```
