# 评估体系 v2 首轮全量基线报告

> **日期**:2026-08-16 · **分支**:`feat/eval-speedup`
> **一句话**:v2 五指标体系(新增 context_recall)在 23 条修订版 golden set 上完成首轮全量基线;漏检证明实验实证了新指标的价值;诊断出语义覆盖(recall 0.49)为当前最大短板,并给出 B 阶段调参优先级。

---

## 1. 运行配置

| 项 | 值 |
|---|---|
| golden set | 23 条 QA(v4.0,人工修订版:reference 严格论断式、F2^4 格式统一、5 条跨源) |
| judge | `deepseek-v4-flash`(云端,`RAGAS_JUDGE_PROVIDER=deepseek`) |
| 检索侧 | 本地 ollama `nomic-embed-text`(embedding),top-k = 5 |
| 运行方式 | **分批**:2 条/批 × 12 批,批间 15s(绕 deepseek 连续请求限流) |
| collection | `evaluation`(576 chunks,6 篇 AES/SM4 量子电路论文,D-027 修复后完整语料) |
| 数据存档 | [logs/eval_v2_baseline_deepseek.json](../logs/eval_v2_baseline_deepseek.json)(12 批 aggregate + 加权聚合) |

## 2. 基线结果(v2 首轮)

| 指标 | 加权均值 | 对照 llama3 旧基线(PR#6,10 条旧 GT) | 读数解读 |
|---|---|---|---|
| source_recall@5 | **0.9565** | 1.00 | 23 条中 1 条整篇漏检(#4 Lin/Xiang) |
| source_precision@5 | **0.7130** | 0.66 | top-5 约 30% 是错误论文 |
| context_relevance | **0.8696** | 0.95 | 部分 chunk 与问题跑偏 |
| context_precision | **0.6557** | 0.97 | 相关 chunk 排序靠后,噪音重 |
| **context_recall** | **0.4942** | —(v1 无) | **一半 reference 论断未被 top-5 覆盖** |

> 旧基线数值不可直接对比(语料 2→6 篇、GT 全部重标、judge 换 deepseek),仅作氛围参考。

**核心结论**:召回缺口(recall 0.49)与排序噪音(precision 0.66)**并存**——调参需"扩池 + 精排"组合,不能单方向猛打。

## 3. 漏检证明实验(v2 价值的直接实证)

同一道跨源 QA("2023 qubit-record vs 2022 Science China 的 AES qubit 数对比"),两次评估:

| 指标 | 正常检索 | 限制到单篇(source: 内联过滤) | 变化 |
|---|---|---|---|
| source_recall@5 | 1.0 | **1.0** | **纹丝不动**(v1 失明) |
| source_precision@5 | 0.8 | **1.0** | **反而升高**(单源时无稀释,指标在奖励退化) |
| **context_recall** | **1.0** | **0.25** | **暴跌 75%,抓个正着** |

**结论**:人为制造"第二篇论文漏检"后,v1 指标完全无感(source_recall 满分、source_precision 甚至更好看),context_recall 从 1.0 → 0.25。**漏检这件事,v1 量不到,v2 量到了**——新指标存在意义的实验闭环。

**指标分工的澄清**(备 future 追问):source_recall 与 context_recall 不是上下位关系——context_recall 报警"出事了"(内容缺),source_recall 归因"事出在哪层"(整篇没来 vs 对的论文来了错的段落)。两者修法不同(前者调召回,后者调排序/切分),且 source 系是零 LLM、零方差的确定性锚点,judge 抽风时做故障对照。**都保留**。

## 4. 逐条分布的关键发现

按 context_recall 升序的 23 条明细(评测当轮数据,明细文件已失,aggregate 已存档):

- **7 条 recall = 0.00**:集中在**方法/概念解释类问题**("W-type/T-type 是什么"、"straight-line method 是什么"、"F2^4 inversion 怎么做")——概念答案只占论文一小段,1000 字符粗 chunk 稀释了 embedding,检索排名暴跌;
- **问具体数字的条目明显更好**(qubits 数、T-depth 等,recall 普遍 0.75-1.0);
- **#4 是唯一 source_recall=0**:Lin/Xiang 论文整篇漏检(跨源条目其中一篇);
- **高分段(#17-22,recall=1.0)证明 judge 无系统性偏差**:分数差异来自真实检索难度,不是评分模型漂移;
- 1 条(#23)语义指标缺值(judge 单条调用失败被记 {}),属已知 LLM judge 方差。

## 5. 诊断 → B 阶段调参优先级

当前生效参数:`dense_top_k=20, sparse_top_k=20, fusion_top_k=10, rrf_k=60, rerank.enabled=false, chunk_size=1000, overlap=200`

| 优先级 | 改动 | 针对症状 | 机制 | 代价/风险 |
|---|---|---|---|---|
| 🔴 1 | `rerank.enabled → true`(先 `llm` provider) | 双 precision(0.66/0.71) | 从 fusion 候选池精选排序,rerank 本职 | granite 当重排模型,评估变慢;cross_encoder 未充分测试,暂不用 |
| 🔴 2 | `fusion_top_k: 10 → 15~20` | context_recall(一半) | 慢热 chunk 排 11-20 名时永无出头 | 精度可能再降,**必须与 rerank 搭配**(扩池+精排组合拳) |
| 🟡 3 | `dense/sparse_top_k: 20 → 25~30` | context_recall(另一半)+ #4 整篇漏检 | 单路 20 名外的好 chunk 进不了 RRF | 轻微耗时;RRF 自会过滤 |
| 🟡 4 | `rrf_k: 60 → 40` | context_precision | 60 偏"民主"压平名次信号;40 更信头部 | 单路冠军可能压制另一路,前 3 轮后再试 |
| 🟢 5 | `chunk_size: 1000 → 600~800`(overlap 同步 200→100~120) | **7 条概念题 recall=0** | 小 chunk = 聚焦向量,概念段落不再被稀释 | **全量重新 ingest**(40min/轮),chunk_id 全变;留作概念题缺口的最后一招 |

**实验规则**(B 阶段执行模板):一次只改一个参数 → 跑同一 23 条基线(分批脚本)→ 五指标 before/after;context_recall 做主指标(优化目标),source_recall 做归因指标(解释变化)。

## 6. 过程踩坑(运行环境,非代码缺陷)

| 坑 | 现象 | 根因 | 解法 |
|---|---|---|---|
| deepseek 限流 | 全量 23 条连跑两次卡死(CPU 0 增长,ConnectTimeout) | 连续百+请求触发服务端限流/连接重置 | **分批**:2 条/批 + 15s 间隔,12 批零失败 |
| 系统代理抽风 | 同上 ConnectTimeout,但 curl 直连 0.03s 通 | 代理间歇拦截海外建连(与 D-014 同根因、反方向:localhost 要绕代理,海外直连更稳) | 关代理后恢复;judge 走 deepseek 时避免开代理 |
| JSON mode 空 content | 冒烟误判"v4-flash 坏了" | 官方文档:json_object 模式要求 prompt 含 "json" 字样+样例,且有概率空返回 | 按官方规范写 prompt 后三模型全通过;ragas instructor 自带合规 prompt |
| 明细文件丢失 | batched_results.json 未写成 | 脚本异常退出路径未覆盖 | aggregate 从运行 log 正则重建并存档 |

## 7. 遗留事项

- 逐条明细未持久化(aggregate 已存档;如需逐条,可按同法分批重跑或在未来跑评估时直接写明细);
- #23 的 judge 缺值属 LLM 方差,无行动项;
- "目标 chunk 有没有进 top-5"**当前无指标能量**(chunk_id GT 已弃,锚文本 GT 可作 v3 增量——`anchor_hit@5`,与 source 指标同构的确定性匹配);是否立项取决于是否需要测 chunk 边界切分质量;
- 本报告的调参计划(§5)即 B 阶段启动模板,第一轮实验建议:开 rerank(llm)。
