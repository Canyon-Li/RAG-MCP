# 纯本地英文文献 RAG 知识库 —— 场景契合度评估与适配方案

**日期**:2026-08-13
**分支**:feat/local-english-literature-rag(从 main 切出)
**性质**:需求 × 现有架构契合度评估(非新功能,是对现有项目能否支撑新场景的诊断 + 配置/组件级适配)

---

## 1. 场景画像

| 维度 | 明确值 |
|---|---|
| 隐私边界 | **绝对本地**——数据不能出本机(私有 PDF,防泄露) |
| LLM | 当前 zhipu glm-4-flash(云端)→ 切 **ollama granite4.1:8b**(本地) |
| 文献类型 | **原生数字版英文学术论文**(arXiv / 期刊,双栏 / 公式 / 表格 / 图表) |
| 规模 | **小规模 < 50 篇** |
| 核心场景 | **个人文献记忆 + 溯源检索**——RAG 充当"已读文献外脑",帮记住读过的文献;核心价值是"我之前在哪篇文献见过类似观点/方法",且**每条结果必须能精确指回具体论文出处** |
| 明确**不要**的 | 综述报告式 LLM 生成答案 |

---

## 2. 架构契合度总评

**结论:架构高度契合,不需要推倒重来。** 问题集中在 3 个"配置 / 组件级"的点。

| 子能力 | 现状 | 对本场景的评级 |
|---|---|---|
| 四层设计 + Factory/Registry | 成熟 | 🟢 强支撑——换 backend 零代码 |
| docling parser(学术 PDF 版面) | 命中 | 🟢 DocLayNet 专为学术版面训练 |
| section-aware chunker | 命中 | 🟢 标题合并、表格保整、列表按项切 |
| **溯源 page + bbox** | **现成** | 🟢 chunk 带 `page_num`/`bbox`/`source_ref`,直接命中"暴露来源" |
| Chroma 本地向量库 | 合适 | 🟢 < 50 篇毫无压力 |
| Dense(embedding)检索 | 合适 | 🟢 支撑"以文找文"语义联想 |
| MCP stdio + Copilot/Claude | 命中 | 🟢 IDE 内直接返回带引用的结果 |
| LLM provider | **云端 zhipu** | 🔴 **硬冲突**——与"绝对本地"矛盾 |
| BM25 分词 | jieba | 🟡 jieba 对英文仅"原样保留",缺 stemming/停用词 |

---

## 3. 必改点诊断

### 3.1 改造点 1:LLM 切到本地(🔴 必做,纯配置)

**现状**:`llm.provider: zhipu` + `glm-4-flash`,走云端智谱 API,**与"绝对本地"硬冲突**。

**方案**:`llm.provider` → `ollama`,`model` → `granite4.1:8b`。代码零改动——`OllamaLLM` 已注册在工厂([src/libs/llm/__init__.py:28](src/libs/llm/__init__.py#L28)),`trust_env=False` 绕代理已处理。

**关键洞察**:本场景里 LLM **只服务 ingestion 阶段**(chunk 精炼 / 元数据),**不服务查询阶段的答案生成**(场景明确不要综述)。所以本地 LLM "推理慢"只影响**一次性 ingest**,不影响日常查询体验——这是本场景对本地化特别友好的地方。

LLM 在链路中的使用点与本方案动作:

| 使用点 | 当前 | 切本地 granite 后动作 |
|---|---|---|
| `chunk_refiner` | `use_llm: true` | 🟡 先改 `false`(规则层保留,见 3.3.1) |
| `metadata_enricher` | `use_llm: false` | 🟢 保持 `false`(见 3.2) |
| `image_captioner`(走 Vision LLM) | llava-phi3 | 与主 LLM 解耦,不受影响 |
| 答案生成 | 本来就不用 | ✅ 不涉及 |

### 3.2 改造点 2:英文 BM25 分词(🟡 强烈建议)

**现状**:`SparseEncoder._tokenize`([src/ingestion/embedding/sparse_encoder.py:134](src/ingestion/embedding/sparse_encoder.py#L134))和 `QueryProcessor._tokenize`([src/core/query_engine/query_processor.py:212](src/core/query_engine/query_processor.py#L212))都用 `jieba.lcut`。注释称"English text is handled natively by jieba (preserved as-is)"——即对英文不做专门处理。

**问题**:
- **无词干化**:`optimization` / `optimize` / `optimized` / `optimal` 被当作 4 个不同 term,查 `optimization` 召不回 `optimal`。
- **停用词处理弱**:倒排表塞满 `the/a/an/of`,干扰 + 索引膨胀(IDF 会压权但不彻底)。
- 大小写/标点一致性需确认。

**方案(双语并行)**:`_tokenize` 内对文本分语言处理——中文走 jieba,英文走 `lowercase → 去标点 → Porter stemmer → 英文停用词表`。保留中文能力,代价与纯英文方案相同(万一中文提问也能命中)。

**Stemmer 选型:零依赖纯 Python Porter(选项 B)**。
- 选 B 而非 nltk:本项目是学习/面试项目,引 nltk 太重;Porter 算法单文件几十行即可实现。
- 选 B 而非 snowballstemmer:少一个包依赖,且 Porter 是 BM25 场景足够好的算法。

**关键约束(纪律级)**:`SparseEncoder`(索引侧)和 `QueryProcessor`(查询侧)的 `_tokenize` **必须用同一套逻辑**,否则 term 对不上,BM25 永远召不回。建议抽取为公共函数(如 `src/core/text/tokenizer.py`),两侧共用。

**副作用**:分词逻辑变了,**旧 BM25 索引失效,需重新 ingest 全库**。

### 3.3 减负项(本场景比通用 RAG 更简单)

| 组件 | 现状 | 动作 | 理由 |
|---|---|---|---|
| Rerank | `enabled: false` | 🟢 保持关 | 场景是召回 + 溯源,不是精排喂生成;cross-encoder 本地跑吃显存慢;< 50 篇噪音不明显 |
| Metadata enricher LLM | `use_llm: false` | 🟢 保持关 | tags/summary 不参与召回,只是展示;8B 逐 chunk 生成是浪费 ingest 时间 |
| Chunk refiner LLM | `use_llm: true` | 🟡 **先改 `false`,跑一轮不满意再开** | 见下方 3.3.1 专项分析 |

#### 3.3.1 chunk_refiner 专项:两层独立,关的只是 LLM 层

chunk_refiner 有**两条独立路径**,`use_llm` 只控制第二条,**规则层恒执行**:

- **规则层 `_rule_based_refine`**(无论 `use_llm` 都跑,[chunk_refiner.py:234-237](src/ingestion/transform/chunk_refiner.py#L234-L237)):去 PDF 提取噪声——页眉页脚分隔线(`──── Page 12 ────`)、HTML 注释/标签残留、规整空白。**纯确定性,不改文字内容**,对本场景有价值,保留。
- **LLM 层 `_llm_refine`**(仅 `use_llm: true`):prompt([chunk_refinement.txt](config/prompts/chunk_refinement.txt))设计上很克制——明确禁止总结/改义/缩短,要求保留专有名词、引用、`[IMAGE: id]` 占位符,只做去噪 + 自洽连贯。

**关掉 LLM 层的理由(溯源优先确定性)**:
1. 本场景核心诉求是**溯源回论文原话**,对"文本被模型动过"零容忍。规则层是确定性的,LLM 层是概率性的——即便 prompt 禁止改写,8B 模型**实际执行仍可能微调措辞**(标点/冠词等),有破坏逐字溯源的风险。
2. docling + section-aware chunker 产出的学术文本已足够干净,LLM 层边际收益小。
3. 省一次性 ingest 的本地推理时间。

**采用策略 C(先关、效果不满意再开)**:先 `use_llm: false` 跑一轮,检查 chunk 质量(是否有断句残留/连字符问题)。若规则层已够干净,永久保持关;若发现明显残留,再开 LLM 层并重新 ingest 验证。

### 3.4 溯源展示(已验证,现成)

`CitationGenerator` + `ResponseBuilder` 已组装好溯源链路:

- Citation 结构([src/core/response/citation_generator.py:117-139](src/core/response/citation_generator.py#L117-L139)):取 `page`/`page_num`、`title`、`section`、`chunk_index`、`doc_type`,带 1-based 序号。
- Markdown 输出([src/core/response/response_builder.py:275-309](src/core/response/response_builder.py#L275-L309)):每条结果带 `### [1]`、`**相关度:**`、`**来源:**`、`**页码:**`,底部汇总引用列表 `(p.3)`。
- 双形态返回:人类可读 Markdown(`content`)+ 结构化 citation(`structured_content`)。

**结论:核心需求"暴露来源"数据层/处理层/展示层全部就位,不改 ResponseBuilder。**

**可选小裂缝**:citation 当前带 `page` 但**未带 `bbox` 和 `section_type`**(CitationGenerator 字段白名单不含)。chunk metadata 里有这两个字段,只是没透传。
- 只需"第 3 页"级溯源 → 现状足够,零改动。
- 想要"第 3 页 Method 章节左栏"级溯源 → CitationGenerator 加 2 行把 `section_type`/`bbox` 放进 citation。属 P3 可选项。

---

## 4. 落地清单(按优先级)

**P0 — 隐私合规(必做)**:
1. `config/settings.yaml`:`llm.provider: ollama` + `model: granite4.1:8b`(纯配置)
2. 确认 ollama 已 `ollama pull granite4.1:8b`(运行时前置,非代码)

**P1 — 英文检索质量(强烈建议)**:
3. 抽取公共 tokenizer(如 `src/core/text/tokenizer.py`):中文 jieba + 英文(lowercase + Porter stem + 停用词),零依赖纯 Python Porter
4. `SparseEncoder._tokenize` 和 `QueryProcessor._tokenize` 改为调用公共 tokenizer(**两侧一致**)
5. 重新 ingest 全库(分词变了,旧 BM25 索引失效)
6. 跑评估(`scripts/evaluate.py`)对比改前/改后的 hit_rate / mrr

**P2 — 减负 + 避免改写原文**:
7. `ingestion.chunk_refiner.use_llm: false`
8. `ingestion.metadata_enricher.use_llm: false`(保持)

**P3 — 锦上添花(可选)**:
9. CitationGenerator 加 `section_type` / `bbox` 到 citation(章节级溯源)
10. 将来召回噪音大时再开 cross-encoder rerank

---

## 5. 明确**不做**的事(YAGNI)

- ❌ 不引入 nltk(选零依赖 Porter)
- ❌ 不换 BM25 库(rank_bm25 / bm25s)——现有 JSON 倒排索引设计是对的,只缺分词
- ❌ 不做综述生成——场景明确不要
- ❌ 不上 OCR——文献是原生数字版,无扫描件
- ❌ 不重构四层架构——契合度足够,只动配置和分词

---

## 6. 验证手段

- **分词一致性**:写单元测试,断言同一文本经 SparseEncoder 和 QueryProcessor 的 `_tokenize` 产出相同 token 序列。
- **英文召回**:构造英文同源词用例(`optimization` 查回 `optimal` 的 chunk),断言 BM25 命中。
- **本地化**:断网状态下跑 `python scripts/query.py`,确认仍可查询(证明无云端依赖)。
- **溯源**:查询后检查返回的 citation 带 `source` + `page`。
