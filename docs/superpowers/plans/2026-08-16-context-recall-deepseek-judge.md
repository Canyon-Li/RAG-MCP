# context_recall + DeepSeek Judge 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 RagasEvaluator 加 `context_recall` 指标(RAGAS ContextRecall,GT 复用 golden set 既有 `reference` 字段),并把 judge LLM 通道扩展出 deepseek 分支(`deepseek-v4-flash`),同时清理无人消费的 embeddings 死代码、给 evaluate.py 加 dotenv 支持。

**Architecture:** 全部改动集中在 `RagasEvaluator`(src/observability/evaluation/ragas_evaluator.py)——指标常量 + `_run_ragas` 分支 + `_build_wrappers` 新增 deepseek 分支并收窄返回值为单 `llm`。judge 仍走 env 驱动(`RAGAS_JUDGE_PROVIDER`/`RAGAS_JUDGE_MODEL`/`DEEPSEEK_API_KEY`),与 settings.llm 解耦(D-014 一贯设计)。路由零改动:`CompositeEvaluator._backend_supported_metrics` 读 `SUPPORTED_METRICS` 自动生效,配置侧只加一行 metrics。spec: [docs/superpowers/specs/2026-08-16-context-recall-deepseek-judge-design.md](../specs/2026-08-16-context-recall-deepseek-judge-design.md)。

**Tech Stack:** Python 3.10(conda env `langchain-test`)、ragas 0.4.3(collections metrics + instructor)、openai SDK(AsyncOpenAI,DeepSeek OpenAI 兼容端点)、pytest + unittest.mock、python-dotenv(新依赖)。

## Global Constraints

- 运行环境:conda env `langchain-test`(全局 python 无 chromadb/ragas;跑测试用 `/d/Anaconda/envs/langchain-test/python.exe -m pytest`)。
- ragas 0.4.3 import 前必须先有 vertexai stub(ragas_evaluator.py 模块顶部已有;测试文件如直接 import ragas 需同样处理——现有测试通过 `from src.observability...` 间接触发,无需处理)。
- **trust_env 方向性**:ollama 分支 `trust_env=False`(localhost 绕代理,D-014);deepseek 分支**不禁**(云端出网,走系统代理是正确路径)。不得"顺手统一"。
- ContextRecall 签名(已实测):`score(user_input=..., retrieved_contexts=..., reference=...)`。
- golden set **零改动**;检索链路(`src/core/`)零改动;settings.yaml 只加一行 `- "context_recall"`。
- 默认路径不变:不设任何 env 时 `RAGAS_JUDGE_PROVIDER` 默认 `ollama`、`_DEFAULT_JUDGE_MODEL` 仍 `llama3`。
- Windows 控制台:新脚本代码不涉及;shell 命令按 PowerShell/Git-Bash 各自语法。
- 已知未知:DeepSeek 端点对 `deepseek-v4-flash` 的兼容性由 Task 6 冒烟解决;若 404/模型名错,修正 `RAGAS_JUDGE_BASE_URL` 默认值或模型名拼写(在 Task 6 内闭环,不改架构)。
- 每个任务独立可测、独立提交;TDD:先红后绿。

---

### Task 1: context_recall 指标常量 + `_run_ragas` 分支

**Files:**
- Modify: `src/observability/evaluation/ragas_evaluator.py:44-48`(常量区)、`:205-253`(`_run_ragas`)
- Test: `tests/unit/test_ragas_evaluator.py`

**Interfaces:**
- Consumes: `RagasEvaluator._run_ragas(query, contexts, reference)` 既有结构;`_extract_reference` 已返回 `ground_truth["reference"]`。
- Produces: 模块常量 `CONTEXT_RECALL = "context_recall"`;`SUPPORTED_METRICS` 扩为三元素集合。后续任务(Task 3/4/5)与 `CompositeEvaluator` 依赖此常量。

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_ragas_evaluator.py` 末尾追加(沿用文件内既有 mock 模式——`_build_wrappers` patch 为返回元组的形式**在本任务不改**,Task 3 收窄时统一改):

```python
class TestContextRecallMetric:
    """Tests for the context_recall metric branch in _run_ragas."""

    def test_supported_metrics_includes_context_recall(self) -> None:
        from src.observability.evaluation.ragas_evaluator import SUPPORTED_METRICS

        assert "context_recall" in SUPPORTED_METRICS

    def test_init_accepts_context_recall_metric(self) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["context_recall"])
        assert evaluator._metric_names == ["context_recall"]

    def test_run_ragas_calls_context_recall_with_reference(self) -> None:
        """context_recall branch invokes ContextRecall.score with
        (user_input, retrieved_contexts, reference) and records the value."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["context_recall"])
        mock_result = MagicMock()
        mock_result.value = 0.75

        with patch(
            "src.observability.evaluation.ragas_evaluator.RagasEvaluator._build_wrappers",
            return_value=(MagicMock(), MagicMock()),
        ), patch(
            "ragas.metrics.collections.ContextRecall"
        ) as MockContextRecall:
            MockContextRecall.return_value.score.return_value = mock_result
            scores = evaluator._run_ragas(
                query="What is LCU?",
                contexts=["LCU-based approach with query complexity O(1/sqrt(eps))."],
                reference="The paper uses LCU with complexity O(1/sqrt(eps)).",
            )

        assert scores["context_recall"] == 0.75
        MockContextRecall.return_value.score.assert_called_once_with(
            user_input="What is LCU?",
            contexts=["LCU-based approach with query complexity O(1/sqrt(eps))."],
            retrieved_contexts=["LCU-based approach with query complexity O(1/sqrt(eps))."],
            reference="The paper uses LCU with complexity O(1/sqrt(eps)).",
        )

    def test_run_ragas_context_recall_without_reference_scores_zero(self) -> None:
        """Missing reference → warning logged, score 0.0 (symmetric with precision)."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(metrics=["context_recall"])

        with patch(
            "src.observability.evaluation.ragas_evaluator.RagasEvaluator._build_wrappers",
            return_value=(MagicMock(), MagicMock()),
        ), patch(
            "ragas.metrics.collections.ContextRecall"
        ) as MockContextRecall:
            scores = evaluator._run_ragas(
                query="What is LCU?", contexts=["some context"], reference=None,
            )

        assert scores["context_recall"] == 0.0
        MockContextRecall.return_value.score.assert_not_called()
```

> 注:Step 1 第三个测试对 `score.assert_called_once_with` 的 kwargs 断言——若实现传参名与断言不一致(如漏 `user_input`),测试会抓出来。若 ragas 0.4.3 的 `ContextRecall.score` 实际接受位置参数以外的多余 kwargs(如实现把 `contexts` 与 `retrieved_contexts` 都传),以**实现后跑通的真签名为准**修断言——但 `(user_input, retrieved_contexts, reference)` 三参是 spec 实测基线,预计无需改。上面 `assert_called_once_with` 里同时写了 `contexts=` 与 `retrieved_contexts=` 是**笔误防御**:实现只应传 `retrieved_contexts`,跑红后把断言里的 `contexts=...` 一行删掉再继续(保留这条注释提醒执行者)。

- [ ] **Step 2: 跑测试确认失败**

```bash
/d/Anaconda/envs/langchain-test/python.exe -m pytest tests/unit/test_ragas_evaluator.py::TestContextRecallMetric -v
```
预期:`test_supported_metrics_includes_context_recall` FAIL(常量不存在),其余 3 个也 FAIL。

- [ ] **Step 3: 最小实现**

3a. 常量区([ragas_evaluator.py:44-48](../../src/observability/evaluation/ragas_evaluator.py#L44))改为:

```python
# Metric name constants
CONTEXT_RELEVANCE = "context_relevance"
CONTEXT_PRECISION = "context_precision"
CONTEXT_RECALL = "context_recall"

SUPPORTED_METRICS = {CONTEXT_RELEVANCE, CONTEXT_PRECISION, CONTEXT_RECALL}
```

3b. `_run_ragas` 内 import 行([:218](../../src/observability/evaluation/ragas_evaluator.py#L218))改为:

```python
from ragas.metrics.collections import (
    ContextRelevance,
    ContextPrecision,
    ContextRecall,
)
```

3c. `_run_ragas` 的 metric 循环里,`elif metric_name == CONTEXT_PRECISION:` 块之后、`else: continue` 之前,插入:

```python
            elif metric_name == CONTEXT_RECALL:
                m = ContextRecall(llm=llm)
                # ContextRecall decomposes the reference into atomic claims
                # and scores the fraction supported by the retrieved contexts.
                # Symmetric with precision: missing reference → warn + 0.0.
                if not reference:
                    logger.warning(
                        "context_recall skipped: no reference answer provided "
                        "(golden set missing 'reference')."
                    )
                    scores[metric_name] = 0.0
                    continue
                result = m.score(
                    user_input=query,
                    retrieved_contexts=contexts,
                    reference=reference,
                )
```

3d. `_run_ragas` docstring 的签名清单([:213-216](../../src/observability/evaluation/ragas_evaluator.py#L213))补一行:

```python
        - ContextRecall: (user_input, retrieved_contexts, reference)
```

- [ ] **Step 4: 跑测试确认通过**

```bash
/d/Anaconda/envs/langchain-test/python.exe -m pytest tests/unit/test_ragas_evaluator.py::TestContextRecallMetric -v
```
预期:4 个全 PASS。(若 `assert_called_once_with` 因 Step 1 注释的笔误防御失败,删断言里 `contexts=...` 那行重跑。)

- [ ] **Step 5: 跑整个测试文件回归**

```bash
/d/Anaconda/envs/langchain-test/python.exe -m pytest tests/unit/test_ragas_evaluator.py -q
```
预期:全 PASS(既有测试不受影响——`SUPPORTED_METRICS` 扩集不破坏"metrics 过滤到支持集"的既有测试;若 `test_init_default_metrics` 断言默认指标集恰为两元素集合,会 FAIL,此时把该断言更新为三元素 `{CONTEXT_RELEVANCE, CONTEXT_PRECISION, CONTEXT_RECALL}`——这是指标面扩张的预期联动,不是回归)。

- [ ] **Step 6: Commit**

```bash
git add src/observability/evaluation/ragas_evaluator.py tests/unit/test_ragas_evaluator.py
git commit -m "feat(eval): RagasEvaluator 新增 context_recall 指标(RAGAS ContextRecall,GT 复用 reference)"
```

---

### Task 2: `_build_wrappers` 新增 deepseek 分支

**Files:**
- Modify: `src/observability/evaluation/ragas_evaluator.py:269-302`(`_build_wrappers` 的 ollama 分支之后)
- Test: `tests/unit/test_ragas_evaluator.py`

**Interfaces:**
- Consumes: `_resolve_judge_provider()`(env `RAGAS_JUDGE_PROVIDER`,默认 `ollama`)、`_resolve_judge_model()`(env `RAGAS_JUDGE_MODEL`,call-time 读取)。
- Produces: deepseek 分支返回 `(llm, embeddings=None)`——**本任务保持元组返回**(embeddings 清理是 Task 3,先加分支后收窄,两步各自可测)。

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_ragas_evaluator.py` 追加(紧邻既有 `test_build_wrappers_ollama_branch` 的测试类,或新建类):

```python
class TestBuildWrappersDeepseekBranch:
    """Tests for the deepseek judge branch in _build_wrappers."""

    def test_deepseek_branch_builds_llm_with_env_key(self, monkeypatch) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "deepseek")
        monkeypatch.setenv("RAGAS_JUDGE_MODEL", "deepseek-v4-flash")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")

        evaluator = RagasEvaluator()
        # Mock llm_factory + AsyncOpenAI at their import sources
        with patch("ragas.llms.llm_factory") as mock_factory, \
             patch("openai.AsyncOpenAI") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            result = evaluator._build_wrappers()

        mock_client_cls.assert_called_once()
        _, kwargs = mock_client_cls.call_args
        assert kwargs["api_key"] == "sk-test-123"
        assert kwargs["base_url"] == "https://api.deepseek.com"
        mock_factory.assert_called_once_with(
            "deepseek-v4-flash", client=mock_client_cls.return_value,
            max_tokens=8192,
        )
        # Task 3 会把返回值收窄为单 llm;当前任务仍是元组
        assert isinstance(result, tuple)

    def test_deepseek_branch_missing_key_raises(self, monkeypatch) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "deepseek")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        evaluator = RagasEvaluator()
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            evaluator._build_wrappers()

    def test_deepseek_branch_custom_base_url(self, monkeypatch) -> None:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
        monkeypatch.setenv("RAGAS_JUDGE_BASE_URL", "https://api.deepseek.com/v1")

        evaluator = RagasEvaluator()
        with patch("ragas.llms.llm_factory"), \
             patch("openai.AsyncOpenAI") as mock_client_cls:
            evaluator._build_wrappers()

        _, kwargs = mock_client_cls.call_args
        assert kwargs["base_url"] == "https://api.deepseek.com/v1"
```

- [ ] **Step 2: 跑测试确认失败**

```bash
/d/Anaconda/envs/langchain-test/python.exe -m pytest tests/unit/test_ragas_evaluator.py::TestBuildWrappersDeepseekBranch -v
```
预期:3 个 FAIL——`_build_wrappers` 无 deepseek 分支,走到 azure/openai fallback 后因 settings 为 None/`settings.llm` 不匹配而抛错或构造错客户端。

- [ ] **Step 3: 最小实现**

`_build_wrappers` 里,ollama 分支的 `return llm, embeddings`([:302](../../src/observability/evaluation/ragas_evaluator.py#L302))之后、`# Fallback: original cloud-judge logic` 注释之前,插入:

```python
        if judge_provider == "deepseek":
            # Cloud judge via DeepSeek's OpenAI-compatible endpoint.
            # NOTE: unlike the ollama branch, trust_env is deliberately NOT
            # disabled here — deepseek is a REMOTE endpoint, so going through
            # the system proxy is the correct path (D-014 only applies to
            # localhost traffic).
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError(
                    "RAGAS_JUDGE_PROVIDER=deepseek but DEEPSEEK_API_KEY "
                    "is not set"
                )
            base_url = os.environ.get(
                "RAGAS_JUDGE_BASE_URL", "https://api.deepseek.com"
            )
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            llm = llm_factory(
                self._resolve_judge_model(), client=client, max_tokens=8192,
            )
            return llm, None
```

同时把 `AsyncOpenAI` 的 import 从函数体内的 `from openai import AsyncAzureOpenAI, AsyncOpenAI` 保持不动(deepseek 分支在同一函数内,共用该 import)。

- [ ] **Step 4: 跑测试确认通过**

```bash
/d/Anaconda/envs/langchain-test/python.exe -m pytest tests/unit/test_ragas_evaluator.py::TestBuildWrappersDeepseekBranch -v
```
预期:3 个 PASS。

- [ ] **Step 5: 文件级回归**

```bash
/d/Anaconda/envs/langchain-test/python.exe -m pytest tests/unit/test_ragas_evaluator.py -q
```
预期:全 PASS。

- [ ] **Step 6: Commit**

```bash
git add src/observability/evaluation/ragas_evaluator.py tests/unit/test_ragas_evaluator.py
git commit -m "feat(eval): RagasEvaluator judge 新增 deepseek 分支(env 驱动,trust_env 走代理)"
```

---

### Task 3: embeddings 死代码清理(返回值收窄为单 llm)

**Files:**
- Modify: `src/observability/evaluation/ragas_evaluator.py:269-361`(`_build_wrappers` 全函数)、`:221`(`_run_ragas` 调用点)、模块 docstring(`:1-14`)
- Test: `tests/unit/test_ragas_evaluator.py`(更新 `test_build_wrappers_ollama_branch` 及 Task 1/2 新增测试的元组断言)

**Interfaces:**
- Consumes: Task 1/2 后的 `_build_wrappers`(返回 `(llm, embeddings)`)。
- Produces: `_build_wrappers() -> llm`(单值)。**破坏性内部签名变更**——所有调用点(仅 `_run_ragas` 一处)和所有 mock 它的测试同步更新。

- [ ] **Step 1: 更新测试到目标形态(先红)**

3a. 既有 `test_build_wrappers_ollama_branch`([:234-274 附近](../../tests/unit/test_ragas_evaluator.py))中 `llm, embeddings = evaluator._build_wrappers()` 改为 `llm = evaluator._build_wrappers()`,并删除对 embeddings 的断言(如有)。

3b. Task 1 的两个 `_build_wrappers` mock: `return_value=(MagicMock(), MagicMock())` 改为 `return_value=MagicMock()`。

3c. Task 2 的 `test_deepseek_branch_builds_llm_with_env_key` 末尾 `assert isinstance(result, tuple)` 改为:

```python
        # embeddings 清理后返回单值 llm
        assert result is mock_factory.return_value
```

3d. 新增收窄断言测试(追加到 TestBuildWrappersDeepseekBranch 类):

```python
    def test_build_wrappers_returns_single_llm_not_tuple(self, monkeypatch) -> None:
        """After embeddings cleanup, _build_wrappers returns the llm alone."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")

        evaluator = RagasEvaluator()
        with patch("ragas.llms.llm_factory") as mock_factory, \
             patch("openai.AsyncOpenAI"):
            result = evaluator._build_wrappers()

        assert result is mock_factory.return_value
```

- [ ] **Step 2: 跑测试确认失败**

```bash
/d/Anaconda/envs/langchain-test/python.exe -m pytest tests/unit/test_ragas_evaluator.py -q
```
预期:改动过的断言 FAIL(当前实现仍返回元组)。

- [ ] **Step 3: 实现**

3a. `_run_ragas` 调用点([:221](../../src/observability/evaluation/ragas_evaluator.py#L221)):
```python
        # Build the judge LLM wrapper from env-driven provider config
        llm = self._build_wrappers()
```
(删 `llm, embeddings = ...` 的元组解包。)

3b. `_build_wrappers` 签名与 docstring 改为:

```python
    def _build_wrappers(self) -> Any:
        """Build the Ragas judge LLM wrapper.

        Judge provider is env-driven (RAGAS_JUDGE_PROVIDER: ollama | deepseek
        | azure | openai), decoupled from settings.llm so swapping the
        retrieval LLM never affects the judge.

        The three active metrics (relevance / precision / recall) consume no
        embeddings — the historical embeddings wrapper was dead code and has
        been removed (spec 2026-08-16 §2).
        """
```

3c. 删除:`from ragas.embeddings import OpenAIEmbeddings` import 行([:280](../../src/observability/evaluation/ragas_evaluator.py#L280));ollama 分支的 `embeddings = OpenAIEmbeddings(...)` 两行与 `return llm, embeddings` → `return llm`;deepseek 分支 `return llm, None` → `return llm`;azure/openai fallback 里 `# ── Embeddings ──` 整段(从 `emb_cfg = self.settings.embedding` 到 `embeddings = OpenAIEmbeddings(...)`,[:334-359](../../src/observability/evaluation/ragas_evaluator.py#L334))全删,末尾 `return llm, embeddings` → `return llm`。

3d. 模块 docstring([:1-14](../../src/observability/evaluation/ragas_evaluator.py#L1))的 "Build Ragas LLM and Embedding wrappers" 相关表述同步为 judge 单职责。

- [ ] **Step 4: 跑测试确认通过 + 文件回归**

```bash
/d/Anaconda/envs/langchain-test/python.exe -m pytest tests/unit/test_ragas_evaluator.py -q
```
预期:全 PASS。

- [ ] **Step 5: 全局回归(mock _build_wrappers 的其他文件)**

```bash
grep -rn "_build_wrappers" tests/ src/ | grep -v test_ragas_evaluator
```
若有其他文件 mock/调用 `_build_wrappers`(预计无——grep 验证),同步更新;然后:
```bash
/d/Anaconda/envs/langchain-test/python.exe -m pytest tests/unit -q
```
预期:除 main 预存 10 失败外无新增失败(基线:1352 passed / 10 failed)。

- [ ] **Step 6: Commit**

```bash
git add src/observability/evaluation/ragas_evaluator.py tests/unit/test_ragas_evaluator.py
git commit -m "refactor(eval): _build_wrappers 清理 embeddings 死代码,返回值收窄为单 judge llm"
```

---

### Task 4: settings.yaml 启用 context_recall + 复合路由验证

**Files:**
- Modify: `config/settings.yaml:81-85`(evaluation.metrics 列表)
- Test: `tests/unit/test_composite_evaluator.py`(追加路由测试)

**Interfaces:**
- Consumes: Task 1 的 `SUPPORTED_METRICS`(含 context_recall)。
- Produces: 配置态五指标列表;composite → ragas 路由含 context_recall。

- [ ] **Step 1: 写失败测试**

`tests/unit/test_composite_evaluator.py` 追加:

```python
class TestContextRecallRouting:
    """context_recall routes to the ragas backend via SUPPORTED_METRICS."""

    def test_backend_supported_metrics_includes_context_recall(self) -> None:
        from src.observability.evaluation.composite_evaluator import CompositeEvaluator

        supported = CompositeEvaluator._backend_supported_metrics("ragas")
        assert "context_recall" in supported
        # custom backend must NOT claim it (deterministic evaluator)
        custom_supported = CompositeEvaluator._backend_supported_metrics("custom")
        assert "context_recall" not in custom_supported
```

- [ ] **Step 2: 跑测试确认失败**

```bash
/d/Anaconda/envs/langchain-test/python.exe -m pytest tests/unit/test_composite_evaluator.py::TestContextRecallRouting -v
```
预期:FAIL(ragas 的 SUPPORTED_METRICS 尚无 context_recall——**注意**:若 Task 1 已完成此测试应直接 PASS,因为 `_backend_supported_metrics` 动态读常量。此时该测试的角色是"路由不回归"的守护测试,Step 2 记录实际结果即可)。

- [ ] **Step 3: 配置改动**

`config/settings.yaml` 的 `evaluation.metrics` 列表([:81-85](../../config/settings.yaml#L81))末尾加:

```yaml
    - "context_recall"
```

- [ ] **Step 4: 跑测试 + 配置加载冒烟**

```bash
/d/Anaconda/envs/langchain-test/python.exe -m pytest tests/unit/test_composite_evaluator.py -q
/d/Anaconda/envs/langchain-test/python.exe -c "import sys; sys.path.insert(0,'.'); from src.core.settings import load_settings; s = load_settings(); print(s.evaluation.metrics)"
```
预期:测试全 PASS;输出含 `context_recall`。

- [ ] **Step 5: Commit**

```bash
git add config/settings.yaml tests/unit/test_composite_evaluator.py
git commit -m "feat(eval): settings 启用 context_recall,composite 路由守护测试"
```

---

### Task 5: dotenv 支持(python-dotenv + evaluate.py 挂载)

**Files:**
- Modify: `pyproject.toml:26-44`(dependencies)、`scripts/evaluate.py:94-100`(main 入口)
- Test: `tests/unit/test_evaluate_dotenv.py`(新建)

**Interfaces:**
- Consumes: 无。
- Produces: evaluate.py 启动时自动加载 `.env`(存在则读,不存在静默跳过;`override=False` 默认——会话内已设的 env 优先于 .env 文件)。

- [ ] **Step 1: 写失败测试**

`tests/unit/test_evaluate_dotenv.py` 新建:

```python
"""Unit tests for dotenv loading in scripts/evaluate.py.

Verifies main() calls load_dotenv BEFORE constructing the evaluator, so
RAGAS_JUDGE_* / DEEPSEEK_API_KEY written in .env take effect at call time.
"""

from __future__ import annotations

from unittest.mock import patch


class TestEvaluateDotenv:
    def test_main_calls_load_dotenv_before_settings(self) -> None:
        """load_dotenv must run before load_settings (call order guard)."""
        import scripts.evaluate as evaluate_mod

        calls = []

        def fake_dotenv() -> None:
            calls.append("dotenv")

        def fake_settings(*a, **kw):
            calls.append("settings")
            raise SystemExit(0)  # stop main() right after settings load

        with patch.object(evaluate_mod, "__name__", "scripts.evaluate"), \
             patch("dotenv.load_dotenv", side_effect=fake_dotenv) as mock_ld, \
             patch("src.core.settings.load_settings", side_effect=fake_settings):
            try:
                evaluate_mod.main()
            except SystemExit:
                pass

        mock_ld.assert_called_once()
        assert calls == ["dotenv", "settings"], (
            f"dotenv must load before settings, got {calls}"
        )
```

> 注:evaluate.py 里 load_settings 是函数内 import(`from src.core.settings import load_settings`),patch 目标用 `src.core.settings.load_settings`;若该 patch 因 import 绑定时机不生效,改为 patch `scripts.evaluate` 模块内 main 里实际引用的路径(执行时用 `grep -n "load_settings" scripts/evaluate.py` 确认 import 形态后调整 patch 目标——原则:patch 消费点,不是定义点)。

- [ ] **Step 2: 跑测试确认失败**

```bash
/d/Anaconda/envs/langchain-test/python.exe -m pytest tests/unit/test_evaluate_dotenv.py -v
```
预期:FAIL——`dotenv` 模块不存在(ImportError)或 main 未调 load_dotenv(`calls == ["settings"]`)。

- [ ] **Step 3: 实现**

3a. `pyproject.toml` `dependencies` 列表([:26](../../pyproject.toml#L26))内加一行:

```toml
    "python-dotenv>=1.0",  # evaluate.py .env loading (DEEPSEEK_API_KEY etc.)
```

3b. 安装:
```bash
/d/Anaconda/envs/langchain-test/python.exe -m pip install -e ".[dev]"
```

3c. `scripts/evaluate.py` main() 开头([:94-96](../../scripts/evaluate.py#L94))改为:

```python
def main() -> int:
    """Main entry point."""
    # Load .env (gitignored) before anything reads env vars — judge keys
    # (DEEPSEEK_API_KEY / RAGAS_JUDGE_*) live there. Missing file is a silent
    # no-op; already-set session vars win over .env values (override=False).
    from dotenv import load_dotenv

    load_dotenv()

    args = parse_args()
```

- [ ] **Step 4: 跑测试确认通过**

```bash
/d/Anaconda/envs/langchain-test/python.exe -m pytest tests/unit/test_evaluate_dotenv.py -v
```
预期:PASS。

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml scripts/evaluate.py tests/unit/test_evaluate_dotenv.py
git commit -m "feat(eval): evaluate.py 挂载 python-dotenv(.env 自动加载,会话 env 优先)"
```

---

### Task 6: E2E 冒烟 + 全量验收(真 DeepSeek,需用户配 key)

**Files:**
- 无代码改动(纯验证;产出验收记录追加到 spec 末尾或 DEV_CHANGELOG D-028 草稿)
- 前置:用户在 `.env` 写 `DEEPSEEK_API_KEY=sk-...`(已 gitignore 验证)

**Interfaces:**
- Consumes: Task 1-5 全部成果。
- Produces: 冒烟结论(base_url/模型名兼容性)+ 23 条 QA 五指标全量结果 + 漏检证明实验数据。

- [ ] **Step 1: 冒烟——1 条 QA 验证 deepseek-v4-flash 连通**

```powershell
# .env 已写 DEEPSEEK_API_KEY(evaluate.py 会自动加载)
$env:RAGAS_JUDGE_PROVIDER = "deepseek"
$env:RAGAS_JUDGE_MODEL = "deepseek-v4-flash"
python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json --collection evaluation --top-k 5 --json
# (Ctrl+C 中断即可——目的只是看前 1-2 条是否出分;或临时做一个 1 条 QA 的子集 json)
```
判定:
- 出分(哪怕部分条目)→ 端点/模型名兼容,继续;
- 404 / model not found / 401 → 修 `RAGAS_JUDGE_BASE_URL`(试 `https://api.deepseek.com/v1`)或模型名拼写,重试;
- JSON 解析错(instructor 失败)→ `RAGAS_JUDGE_MODEL` 换 `deepseek-chat` 复测,记录 v4-flash 不兼容结论。

- [ ] **Step 2: 全量 23 条**

```powershell
.\scripts\switch-to-eval.ps1   # embedding 仍走本地 ollama,卸 granite/llava 仍有价值
python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json --collection evaluation --top-k 5 --json
```
验收:五指标全出;`context_recall` 量级合理(非全 0/全 1,预期 0.6-1.0 区间);总耗时应显著低于 llama3 版本(云端 1-3s/指标/条)。

- [ ] **Step 3: 漏检证明实验(spec §4 验收项)**

用跨源条目(如 #18,AES qubit 对比类问题)做对照:
```powershell
# 正常跑一次记录 #18 的 context_recall
# 然后临时改 #18 的 query,加 "source:Optimized-quantum-implementation-of-AES.pdf" 强制偏向单源(内联 filter 语法,D-022/D-023 支持)
# 再跑,对比:context_recall 应显著下降,v1 四指标基本不动
```
预期:`context_recall` 降 ≥0.2,v1 指标波动 <0.1。完成后**还原 golden set 改动**(git checkout tests/fixtures/golden_test_set.json 前先确认只改了 #18 的 query)。

- [ ] **Step 4: 记录验收结果**

把冒烟结论 + 全量分数 + 漏检实验数据写入 DEV_CHANGELOG 的 D-028 条目(格式对齐 D-024~D-027)。

- [ ] **Step 5: Commit**

```bash
git add DEV_CHANGELOG.md
git commit -m "docs(changelog): D-028 context_recall + DeepSeek judge(含 e2e 验收数据)"
```

---

## Self-Review 结果

1. **Spec 覆盖**:§3 指标本体 → Task 1/4;§2 judge 通道(deepseek 分支)→ Task 2;§2 embeddings 清理 → Task 3;§2 dotenv → Task 5;§4 测试验收 → Task 1-4 单测 + Task 6 e2e;§1 八项决策全部落 Global Constraints 或对应 Task。**无缺口**。
2. **占位扫描**:Task 6 Step 3 的"临时改 query"给出了具体改法(内联 source: filter);Task 2/5 的 mock patch 目标给了"若不生效"的排查指令而非 TBD。无占位。
3. **类型一致性**:`_build_wrappers` 返回值在 Task 2(元组过渡)与 Task 3(单值终态)显式区分,Task 3 Step 1 同步更新了 Task 1/2 的测试断言;`CONTEXT_RECALL` 常量名全文一致;`context_recall` 字符串值与 settings.yaml/测试一致。
