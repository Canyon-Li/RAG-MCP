# 检索 + 溯源能力评估闭环 —— 设计方案

**日期**:2026-08-13
**分支**:eval/evaluation-work
**性质**:为项目的检索 + 溯源能力搭建评估闭环(golden QA → ingest → 评估 → 混合指标报告)

---

## 0. 设计依据(根植于代码,不依赖外部场景文档)

本方案的所有结论锚定在**代码事实**上,可独立验证、不引用任何会随其他任务线消失的文档。

**核心代码事实**:项目的查询链路是**纯检索 + 引用**,不生成综述答案。

核实证据:
- [response_builder.py](src/core/response/response_builder.py):构建 MCP 响应,全文件无 LLM 调用(唯一的 `generate` 是 `citation_generator.generate(results)`,生成引用标记,非答案)。
- [query.py](scripts/query.py):查询止于 `hybrid_search.search()`,无答案合成步骤。
- 链路全貌:`QueryProcessor → Dense+Sparse → RRF fusion → CitationGenerator → 返回带引用的检索结果`。

**推论**:评估一个"不生成答案的检索系统"时,答案类指标(Answer Accuracy / Response Groundedness / Answer Relevancy / Faithfulness)**无意义**——它们需要一个不存在的"生成答案"来打分。`EvalRunner` 里的 `answer_generator` 是评估脚手架的遗留设计,默认用"拼接 chunk 文本"冒充答案([eval_runner.py:370](src/observability/evaluation/eval_runner.py#L370)),它测的从来不是项目的真实能力。

→ **评估对象 = 检索质量 + 溯源命中。全程不生成答案。**

---

## 1. 目标与边界

**做什么**:搭建端到端评估闭环 —— golden QA 集 → ingest → 跑评估 → 混合指标报告,CLI 一键出结果,可进 dashboard 展示。

**本轮不做(YAGNI)**:
- ❌ 答案生成类指标(系统无答案生成,见 §0)
- ❌ chunk_id 级 IR 指标(hit_rate/mrr)—— chunk_id 脆弱,见 §3
- ❌ 评估驱动调优(baseline → 改参 → 对比)—— Future Work
- ❌ 多维度能力画像 —— Future Work

---

## 2. 数据地基

| 项 | 值 |
|---|---|
| 测试文档 | 英文学术论文 PDF → `tests/fixtures/eval_docs/` |
| collection 名 | `evaluation`(与现有 `default` collection 共存,**不清空**) |
| parser | `docling`(`config/settings.yaml` 默认,测真实链路) |
| ingest 方式 | 复用现成 `python scripts/ingest.py --path tests/fixtures/eval_docs/ --collection evaluation` |
| 评估命令 | `python scripts/evaluate.py --test-set <golden> --collection evaluation` |
| BM25 一致性约束 | ingest 与 evaluate 的 collection 名必须**完全一致**(BM25 索引按 collection 名分目录 `data/db/bm25/{collection}/`),否则 sparse 路径查空,hybrid 退化为纯 dense |

---

## 3. Ground Truth 策略:source 级(论文级)

**放弃 chunk 级 GT,改用 source 级 GT。** 理由根植于代码:

1. **chunk_id 脆弱**:`chunk_id = {doc_id}_{index:04d}_{content_hash8}`,改文档 / 调 chunk_size / 重新 ingest 都会让 id 变化,golden set 立刻作废。
2. **source 字段已透传到检索结果**:[hybrid_search.py:64](src/core/query_engine/hybrid_search.py#L64) `"source": r.metadata.get("source_path", r.metadata.get("source", ""))` —— 每个 chunk 都带论文文件名,source 级 GT 有现成数据,零额外工作。
3. **source(论文文件名)远比 chunk 稳定**:只要论文不换,source 标识不变;且"命中正确论文"正是溯源的直接量度。

golden set 结构:
```json
{
  "test_cases": [
    {
      "source": "wang2020_fast_blackbox.pdf",
      "question": "Which paper uses Linear Combination of Unitaries for state preparation?",
      "expected_sources": ["wang2020_fast_blackbox.pdf"],
      "reference": "LCU-based approach with black-box query complexity O(1/√ε)."
    }
  ]
}
```

- `expected_sources`:期望命中的论文文件名列表。标注者熟悉论文内容,标注成本极低(本来就知道答案在哪篇)。
- `reference`:简短参考答案,**仅作 RAGAS Context Precision 的 LLM 判断依据**,不参与任何"答案对比"(系统不生成答案)。

---

## 4. 指标体系(k=5,混合:确定性 + 语义)

| 指标 | 类型 | 测什么 | 实现 | judge LLM | GT 来源 |
|---|---|---|---|---|---|
| **Source Recall@5** | 溯源(核心) | top-5 是否命中正确论文 | CustomEvaluator(集合运算) | 不需要 | `expected_sources` |
| **Source Precision@5** | 溯源 | top-5 里命中正确论文的比例 | CustomEvaluator(集合运算) | 不需要 | `expected_sources` |
| **Context Relevance** | 检索语义质量 | 检索上下文是否真的相关 | RAGAS | granite4.1:8b | 无需 GT |
| **Context Precision** | 排序质量 | 相关 chunk 是否排在前面 | RAGAS | granite4.1:8b | `reference`(判断依据) |

### 设计原则

- **确定性可量度的,不请 LLM**:source 命中是黑白分明的(命中那篇论文没有 / 有),用集合运算,零成本、完全可复现、无随机性。
- **需要语义判断的,才上 RAGAS**:Context Relevance / Precision 本质是语义问题,集合运算搞不定,用 LLM 裁判。
- **judge 与被测解耦**:granite4.1:8b(本地 Ollama)只当裁判,不参与检索链路;且与项目的 embedding(via Ollama)同源,无云端依赖。

### 组合方式

`CompositeEvaluator`([composite_evaluator.py](src/observability/evaluation/composite_evaluator.py))组合 `CustomEvaluator`(溯源)+ `RagasEvaluator`(语义),合并两者的指标字典。现有脚手架的设计意图在此归位:两个 evaluator 各司其职。

---

## 5. 代码改动清单

| # | 改动 | 文件 | 说明 |
|---|---|---|---|
| 1 | CustomEvaluator 加 source 级指标 | [custom_evaluator.py](src/libs/evaluator/custom_evaluator.py) | 新增 `_extract_sources()` + `_compute_source_recall_at_k()` + `_compute_source_precision_at_k()`;GT 结构从 `{"ids":[...]}` 扩展支持 `{"sources":[...]}`;k=5 通过构造参数 `source_top_k=5` 注入(复用现有 `__init__(metrics, **kwargs)` 签名,新增可选参数),**不硬编码**、不读 settings |
| 2 | RagasEvaluator 加 ollama judge | [ragas_evaluator.py](src/observability/evaluation/ragas_evaluator.py) | `_build_wrappers()`([:232](src/observability/evaluation/ragas_evaluator.py#L232))只认 azure/openai,加 ollama 分支:端点解析复用项目既有模式 `explicit > OLLAMA_BASE_URL env > http://localhost:11434/v1`(参考 [ollama_vision_llm.py:86-103](src/libs/llm/ollama_vision_llm.py#L86-L103));`AsyncOpenAI(base_url=..., api_key="ollama")` + `llm_factory("granite4.1:8b", client=...)`;模型名与端点**不读 settings**(judge 与检索链路的 LLM 解耦,硬配置在 evaluator 内) |
| 3 | RagasEvaluator 指标替换 | 同上 | 从 `faithfulness` / `answer_relevancy` / `context_precision` 换成 `context_relevance` + `context_precision`(去掉答案类,见 §0) |
| 4 | golden set 重写 | [golden_test_set.json](tests/fixtures/golden_test_set.json) | 采用 §3 的 source 级结构 |
| 5 | EvalRunner 传 source GT | [eval_runner.py](src/observability/evaluation/eval_runner.py) | `_evaluate_single` 的 ground_truth 构造(:289-293)从仅 `{"ids":...}` 扩展为同时传 `{"sources": expected_sources}` |
| 6 | answer_generator **保持不接 LLM** | 同上 | 系统无答案生成(§0),保留现有 fallback 即可。不新增 LLM 答案生成逻辑 |

> 改动 6 是对"接真实 LLM 生成答案"思路的明确否决:测一个系统不具备的功能毫无意义。

---

## 6. 输出

`python scripts/evaluate.py --test-set <golden> --collection evaluation --json` → `EvalReport`:

- 每条 QA 四项指标(Source Recall@5 / Source Precision@5 / Context Relevance / Context Precision)
- 全局平均(`aggregate_metrics`)
- 按 `source` 分组可见"每篇论文的检索表现"

报告可进 dashboard 的 [evaluation_panel.py](src/observability/dashboard/pages/evaluation_panel.py) 展示。

---

## 7. Future Work(明确不在本轮)

- **B 阶段 —— 评估驱动调优**:baseline → 改一个参数(rrf_k / dense_top_k / sparse_top_k / 英文 BM25 分词)→ 重新评估 → before/after 对比。评估闭环建好后,B 阶段是纯配置实验 + 重跑评估。
- **多维度能力画像**:若将来引入中文检索 / 图片检索等场景,按维度设计分类测试集。

---

## 8. 验证手段

- **溯源命中**:评估报告里 Source Recall@5 > 0(证明检索能找对论文)。
- **指标分离**:确认四项指标都有值(确定性指标零值 / RAGAS 指标 NaN 都需排查)。
- **无答案生成**:确认评估全程未调用 LLM 生成答案(只调 granite 当 judge)。
- **BM25 生效**:断言 sparse 路径有召回(非空),证明 collection 名一致、hybrid 未退化。
