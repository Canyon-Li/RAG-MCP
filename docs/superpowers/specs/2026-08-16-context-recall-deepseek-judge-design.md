# context_recall 指标 + DeepSeek judge —— 设计方案

**日期**:2026-08-16
**分支**:feat/eval-speedup
**性质**:评估体系 v2 升级的实现 spec(对比论证见 [docs/eval-metrics-upgrade-v2.md](../../docs/eval-metrics-upgrade-v2.md),本文件只写"怎么做")

---

## 0. 设计输入(全部已实测/已确认)

- **RAGAS 0.4.3 ContextRecall API**(vertexai stub 前提下 inspect 验证):
  `score(user_input=..., retrieved_contexts=..., reference=...)` → `MetricResult.value ∈ [0,1]`。
  机制:judge 把 reference 拆成原子论断,逐条判"检索上下文能否支撑",分值 = 被支撑数/总数。
- **golden set 实况**(2026-08-16 实查):**23 条 QA**(非 PR#6 时的 10 条),`reference` 23/23 全有、192-566 字符论断式长句,含 6 条跨源条目(#18-23,2-3 篇论文)。**GT 侧零标注活**。
- **EvalRunner 已透传 reference**([eval_runner.py:298-299](../../src/observability/evaluation/eval_runner.py#L298)),`_extract_reference` 已在读([ragas_evaluator.py:255-267](../../src/observability/evaluation/ragas_evaluator.py#L255))——context_precision 铺好的链路,recall 直接复用。
- **DeepSeek 接入先例**:[deepseek_llm.py](../../src/libs/llm/deepseek_llm.py),OpenAI 兼容端点 `https://api.deepseek.com`,env `DEEPSEEK_API_KEY`。
- **`.env` 已在 .gitignore(:54)**,项目当前无 dotenv loader(git grep 零命中)。

---

## 1. 目标与范围

**做**(两件事一个交付):

1. **context_recall 指标**:RagasEvaluator 新增,调 RAGAS ContextRecall,GT 复用 golden set 既有 `reference` 字段,零 golden set 结构变更。
2. **judge 切 DeepSeek**:`_build_wrappers()` 新增 `deepseek` 分支,三个语义指标(relevance/precision/recall)统一走 `deepseek-v4-flash`。

**用户已拍板的决策**:

| # | 决策 | 内容 |
|---|---|---|
| D-范围 | judge 切换范围 | 全部语义指标(A),不做"仅 recall 用 deepseek" |
| D-模型 | judge 模型 | `deepseek-v4-flash` |
| D-清理 | embeddings 死代码 | 删(A)——三条在用指标均不消费 embeddings |
| D-ollama | ollama judge 分支 | **保留,仅显式 env 时生效**(`RAGAS_JUDGE_PROVIDER` 默认仍 ollama) |
| D-脚本 | switch-to-eval.ps1 | **保留不动**(llama3 预热行无害,后续可能还用) |
| D-base_url | DeepSeek 端点 | `https://api.deepseek.com`(冒烟验证 v4-flash 兼容,若 404/模型名错按报错修正——**已知未知**) |
| D-隐私 | 数据出境边界 | 上云仅 query + 检索文本 + reference(已脱敏公开论文内容);责任在用户,已接受 |
| D-dotenv | .env loader | **加**(A):evaluate.py 启动时 `load_dotenv()`,依赖 python-dotenv |

**不做**(YAGNI):deepseek 失败自动 fallback 本地;settings.yaml 加 judge 配置块;golden set 结构改动;检索链路任何改动。

**成功标准**:
- 23 条 QA 五指标全量跑通,context_recall 量级合理(非全 0/全 1);
- 语义指标单条耗时显著下降(llama3 本地 ~10s+/条 → 云端 ~1-3s/指标/条);
- 默认路径(不设 env)行为不变:ollama + llama3 + 四指标。

---

## 2. judge 通道设计(deepseek 分支)

### 分派结构

```
现在:  _build_wrappers() → ollama(默认) → azure → openai,返回 (llm, embeddings)
改后:  _build_wrappers() → ollama(默认) → deepseek(新) → azure → openai,返回 llm
```

### deepseek 分支(~15 行)

```python
elif judge_provider == "deepseek":
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "RAGAS_JUDGE_PROVIDER=deepseek but DEEPSEEK_API_KEY is not set"
        )
    base_url = os.environ.get("RAGAS_JUDGE_BASE_URL", "https://api.deepseek.com")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    llm = llm_factory(self._resolve_judge_model(), client=client, max_tokens=8192)
    return llm
```

设计点:
1. **env 名对齐 DeepSeekLLM**(`DEEPSEEK_API_KEY`)——单一真相,一个 key 两处可用。
2. **`RAGAS_JUDGE_BASE_URL` 可覆盖**,默认 `https://api.deepseek.com`;冒烟若需 `/v1` 后缀改默认值。
3. **trust_env 不禁**(与 ollama 分支相反且正确):deepseek 是云端出网,走系统代理是正确路径;ollama 走 localhost 才要 `trust_env=False`(D-014)。**注释写明,防后人"顺手统一"**。
4. **`_DEFAULT_JUDGE_MODEL` 不动**(llama3):默认 provider 是 ollama,默认模型与它配套;deepseek 是显式组合(`PROVIDER=deepseek` + `MODEL=deepseek-v4-flash`)。

### embeddings 清理

- ollama 分支删 `OpenAIEmbeddings(...)` 构造;azure/openai 分支删对应 embeddings 段;
- 返回值收窄为 `llm`;调用点 `_run_ragas` 同步;
- 模块 docstring 同步(judge/embeddings 双职责 → judge 单职责)。

### dotenv(新增依赖)

- `pyproject.toml` 加 `python-dotenv`;
- [evaluate.py](../../scripts/evaluate.py) main 入口处 `load_dotenv()`(默认读 `.env`,文件不存在静默跳过——dotenv 默认行为);
- `.env` 已 gitignore,写入 `DEEPSEEK_API_KEY=sk-...` 即自动生效。

### 使用方式(用户侧)

`.env` 写 key 后,评估命令:
```powershell
$env:RAGAS_JUDGE_PROVIDER = "deepseek"   # 或一并写进 .env
$env:RAGAS_JUDGE_MODEL = "deepseek-v4-flash"
python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json --collection evaluation --top-k 5 --json
```
(PROVIDER/MODEL 也可写进 .env——load_dotenv 在 evaluate.py 生效;RAGAS_JUDGE_* 是 call-time 读取,与 dotenv 时序兼容)

---

## 3. context_recall 指标本体

### 代码改动(三处,~20 行)

1. **模块常量**([ragas_evaluator.py:45-48](../../src/observability/evaluation/ragas_evaluator.py#L45)):
   ```python
   CONTEXT_RECALL = "context_recall"
   SUPPORTED_METRICS = {CONTEXT_RELEVANCE, CONTEXT_PRECISION, CONTEXT_RECALL}
   ```
2. **import**:`from ragas.metrics.collections import ContextRelevance, ContextPrecision, ContextRecall`
3. **`_run_ragas` 加分支**(抄 context_precision 结构,[:231-247](../../src/observability/evaluation/ragas_evaluator.py#L231)):
   ```python
   elif metric_name == CONTEXT_RECALL:
       m = ContextRecall(llm=llm)
       if not reference:
           logger.warning("context_recall skipped: no reference answer provided")
           scores[metric_name] = 0.0
           continue
       result = m.score(user_input=query, retrieved_contexts=contexts, reference=reference)
   ```
   reference 缺失 → warning + 0.0(与 precision 对称,报告可见而非悄悄消失)。

### 指标路由(自动)

`CompositeEvaluator._backend_supported_metrics` 读 SUPPORTED_METRICS 自动路由——零代码改动。**配置唯一改动**:settings.yaml `evaluation.metrics` 加 `- "context_recall"`。

### golden set

**零标注活**(23 条 reference 已就绪)。实现后跑一轮看分布,异常条目逐条诊断(验证,非标注)。

### 错误处理

- judge 调用失败:沿用外层 try/except → RuntimeError → EvalRunner 捕获记 `{}` 继续下一条(既有行为);
- reference 缺失:warning + 0.0;
- None 值:`float(result.value) if result.value is not None else 0.0`(抄 precision)。

---

## 4. 测试与验收

### 单元测试(mock,不调真 API)

`tests/unit/test_ragas_evaluator.py`(沿用项目 mock 模式):

| # | 测试 | 验证什么 |
|---|---|---|
| T1 | SUPPORTED_METRICS 含 context_recall;`metrics=["context_recall"]` 构造成功 | 注册面 |
| T2 | mock ContextRecall.score 返回固定值 → 分数字典含 context_recall | 分支走通 |
| T3 | reference 空 → warning + 0.0 | 兜底(与 precision 对称) |
| T4 | PROVIDER=deepseek + key 设 → 构造不抛;key 缺 → ValueError(monkeypatch env) | deepseek env 协议 |
| T5 | `_build_wrappers` 返回单值;ollama/azure/openai 分支不炸 | 清理不破坏现有分支 |

### E2E 验收(真 deepseek)

**冒烟(1 条 QA,先跑)**:验证 base_url/模型名兼容(已知未知在此解决)。

**全量(23 条)**:五指标全出;context_recall 量级合理;耗时 vs llama3 显著下降。

**漏检证明实验**(v2 价值证据):取跨源条目(如 #18),临时排除其中一篇的 chunks(或改 query 偏向单源)→ context_recall 显著下降,v1 指标基本不动——证明"量到了旧指标量不到的东西"。

### 回归确认

- 默认路径(零 env)行为不变;
- `pytest tests/unit` 全绿(既有 10 个 main 预存失败不新增)。

---

## 5. 风险与已知未知

| 风险 | 缓解 |
|---|---|
| `deepseek-v4-flash` 端点/模型名不兼容(base_url 或拼写) | 冒烟先行;`RAGAS_JUDGE_BASE_URL`/`RAGAS_JUDGE_MODEL` 可热切 |
| v4-flash JSON 结构化输出不稳(ragas instructor 依赖) | 冒烟暴露;可切 deepseek-chat/reasoner(env 一改) |
| 云端 judge 有方差 | 对比实验跑 2-3 次取均值(既有纪律);确定性 source 指标不受影响 |
| python-dotenv 新依赖 | 无害轻依赖;load_dotenv 文件缺失静默跳过 |
