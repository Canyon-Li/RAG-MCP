# 评估指标方案对比:四指标(v1)→ 五指标(v2)

> **日期**:2026-08-16 · **分支**:`feat/eval-speedup`
> **一句话**:v1 的四指标在文档级确定性 + chunk 级语义覆盖了"找对论文"和"排得对不对",但对**漏检**(该召回没召回)结构性失明;v2 加一个 `context_recall`,用语义级 chunk 召回补上盲区,GT 用"原文参考"而非 chunk_id,免疫重新切分。

---

## 1. 两套方案总览

### v1(现行,PR#6 上线)

| 指标 | 层 | 判定方式 | GT 来源 | 量什么 |
|---|---|---|---|---|
| source_recall@5 | 文档级 | **确定性集合运算**(零 LLM) | `expected_sources`(论文文件名) | top-5 里有没有命中正确论文(二元) |
| source_precision@5 | 文档级 | **确定性集合运算** | `expected_sources` | top-5 里命中正确论文的比例 |
| context_relevance | chunk 级(语义) | RAGAS + llama3 judge | 无需 GT | 检索回的上下文和问题相关吗 |
| context_precision | chunk 级(语义) | RAGAS(=ContextPrecisionWithReference) | `reference`(简短参考答案) | 相关 chunk 是否排在前面 |

设计原则:**确定性可量度的不上 LLM**(source 命中是黑白分明的);**需要语义判断的才上 LLM**。

### v2(本方案,= v1 + context_recall)

| 指标 | 层 | 判定方式 | GT 来源 | 量什么 |
|---|---|---|---|---|
| (v1 全部四个,不变) | | | | |
| **context_recall** | chunk 级(语义) | RAGAS `ContextRecall` + llama3 judge | `reference`(参考答案,LLM 拆成论断逐条判支撑) | **该召回的信息,检索结果覆盖了吗** |

RAGAS 0.4.3 实测 API(已验证,含 vertexai stub 前提):
```python
m = ContextRecall(llm=llm)
result = m.score(user_input=query, retrieved_contexts=contexts, reference=reference)
# 分数 = 被检索上下文支撑的论断数 / reference 拆出的论断总数
```

---

## 2. 更换(扩展)的原因

### 2.1 核心缺陷:precision 类指标对漏检天然失明

v1 的四个指标里,三个"看得见的东西"都以**检索结果本身**为分母:

- source_recall 只判"命中没有"——top-1 命中即满分,漏掉其余相关内容无感;
- source_precision 分母是 top-5,漏检的内容**不在分母里**;
- context_relevance / context_precision 逐 chunk 打分,同样只看回来的。

**漏掉的 chunk 不在检索结果里,任何以检索结果为分母的指标都看不见它。**这是指标性质决定的,与粒度无关(文档级 chunk 级都一样)。要量漏检,分母必须来自**外部标准**(人工标注的"应该召回什么"),v1 没有这个外部分母。

### 2.2 三个具体的失败模式(v1 全部测不出)

| 失败模式 | 场景 | v1 的读数 | 实际 |
|---|---|---|---|
| **部分漏检** | 问"T-depth 优化",该论文 method 节 2 个相关 chunk 只召回 1 个 | source 双满分,precision 正常 | 信息缺了一半 |
| **同篇错段** | top-5 全是正确论文,但全是引言/参考文献页 | **source_recall=1.0 且 source_precision=1.0** | 用户拿到垃圾 |
| **调参不可归因** | B 阶段把 dense_top_k 20→10,precision 涨了 | 分不清是(a)砍了噪音 还是 (b)砍了该召回的内容 | (b) 是退化,(a) 是优化 |

第三个对 B 阶段(评估驱动调优)是致命的:**top_k / rrf_k 这类参数直接控制召回数量,没有 recall 侧度量,调参实验无法归因**——两种相反的参数效果在 v1 报告里长得一样。

### 2.3 为什么 GT 不用 chunk_id(而是 reference)

chunk 级召回的传统做法是标注"正确 chunk 的 id 清单",本项目的 chunk_id = `{doc_id}_{index:04d}_{content_hash8}`,**任何重切都让 GT 全量作废**:

- 调 chunk_size / chunk_overlap → 全变;
- 换 parser(D-011 降级链)→ 全变;
- **D-027 分批转换(本周实锤)**:同样的文件、同样的参数,仅因修复 bad_alloc 换了转换路径,全库 chunk 404 → 576,所有 id 全变。

> 关键区分:**chunk_id 是"切分的产物",原文 reference 是"切分的输入"。** GT 不应随被测对象的一部分(切分方式)漂移。

### 2.4 为什么判定用 LLM 语义而非确定性文本匹配

考虑过在 golden set 里存"需命中的 chunk 原文片段"做确定性匹配(token 重叠率等),否决,原因根植于本管线:

- **管线会改文本**:`chunk_refiner` 规则层恒执行(use_llm=false 只关 LLM 层),docling 是布局重建而非原文透传——chunk 文本 ≠ PDF 原文,精确子串匹配必然大量假阴性;
- 模糊匹配需要调阈值 + 选分词,**评估器本身变成需要被调优的系统**,违反"评估器要尽可能确定性"的原则;
- RAGAS ContextRecall 的机制恰好绕开:judge 把 `reference` 拆成论断,逐条判"检索上下文有无支撑",对表面变形(空格/语序/局部清洗)不敏感。

### 2.5 参考来源

方案原型:[agentic-rag-for-dummies evaluation.ipynb](file:///d:/Desktop/git/repo/agentic-rag-for-dummies/notebooks/evaluation.ipynb)——五指标(含 ContextRecall)+ judge 与被测模型分离。两点本地化调整:① 该 notebook 用 `context_phrases` 锚点从原文动态截取参考上下文,本方案直接复用 golden set 既有的 `reference` 字段(ragas 0.4.3 的 ContextRecall 签名吃 `reference`,零结构变更);② notebook 是答案生成型系统(有 AnswerAccuracy 等),本系统是纯检索+引用,答案类指标继续排除。

---

## 3. 为什么不用/暂不用其他候选

| 候选 | 否决理由 |
|---|---|
| chunk_id 级 hit_rate / mrr(代码仍在 SUPPORTED_METRICS) | GT 脆弱(§2.3),D-027 已实锤;需标全相关 chunk 集合,成本不可持续 |
| 确定性原文段匹配 | §2.4,refiner/docling 文本变形击穿 |
| 答案类指标(faithfulness / answer_relevancy / AnswerAccuracy) | 系统无答案生成(response_builder 无 LLM),测不存在的功能无意义(spec §0 原论证) |
| ResponseGroundedness | 同上,依赖"生成答案" |
| NVIDIA 指标家族 | notebook 用了,但需要 nvidia API 端点,违背本系统"绝对本地"场景(D-024) |
| chunk 级确定性 recall(标注全部相关 chunk 原文) | "全部"标注缺一段即 GT 错;重切即漂移;标注成本最高 |

---

## 4. 代价与边界(诚实声明)

1. **评估耗时增加**:每条 QA 多一轮 judge 判定(与 reference 论断数成正比)。缓解:metrics 列表是开关,不加不算;与 v1 的加速成果(top-k 5 + 模型切换脚本)叠加后总耗时可控。
2. **LLM 裁判方差**:context_recall 与 context_relevance/precision 同性质,跑两遍分数会漂;确定性 source 指标不受影响。对比实验需同配置跑 2-3 次取均值(既有纪律)。
3. **judge 质量是天花板**:llama3 判长论文上下文偶有误判;若 recall 分数长期异常,先怀疑 judge(RAGAS_JUDGE_MODEL 可热切验证)再怀疑检索。
4. **语义 recall ≠ 精确 recall**:它判"reference 的论断有无支撑",不是"标注 chunk 有无命中"。论断粒度由 judge 拆分决定,与人工期望可能有小偏差——这是用稳定性换精确度的已知取舍。

---

## 5. 实施清单(依赖顺序)

| # | 改动 | 文件 | 规模 |
|---|---|---|---|
| 1 | SUPPORTED_METRICS + `context_recall` 分支 | [ragas_evaluator.py](../src/observability/evaluation/ragas_evaluator.py) | ~15 行,抄 context_precision 结构 |
| 2 | GT 透传 `reference`(已有;确认 EvalRunner 不需新字段) | [eval_runner.py](../src/observability/evaluation/eval_runner.py) | ~0 行(验证即可) |
| 3 | metrics 列表加 `context_recall` | [settings.yaml](../config/settings.yaml) | 1 行 |
| 4 | 单测:指标路由 + reference 缺失时的行为 | tests/unit/ | ~50 行 |
| 5 | golden set:核对既有 reference 质量(论断式、可判定) | [golden_test_set.json](../tests/fixtures/golden_test_set.json) | 标注活 |

**验收**:跑一轮全量评估,五指标全部有值且量级合理;人为构造一条漏检 case(如把 expected 的相关 chunk 从 top-5 里挤掉),context_recall 应显著下降而 v1 四指标基本不动——这是"新指标量到了旧指标量不到的东西"的直接证明。

---

## 6. 与 v1 的关系总结

**v2 不是推翻 v1,是补一角。**v1 的"确定性(文档级)+ 语义(排序/相关性)"双层结构保留;v2 在语义层补"召回覆盖",使指标体系与检索系统的三本性(找对来源 / 排好序 / 不漏关键信息)一一对应:

```
           确定性(可复现)         语义(LLM judge)
文档级     source_recall/precision   —
chunk 级   —                        context_relevance / precision
漏检侧     —                        context_recall   ← v2 新增
```
