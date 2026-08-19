# Retrieval Playground & Evaluation Panel v2 — Design

> **日期**:2026-08-19 · **分支**:`feat/dashboard-retrieval-playground`
> **一句话**:把 Streamlit 面板升级为"检索效果检视工作台"——新增 Retrieval Playground 页(当场查询→看 chunk→dense/sparse/RRF 三列对比),并重写 Evaluation Panel 对齐 v2 五指标体系。

---

## 1. 背景与目标

**现状痛点**:
1. Query Traces 页只能**事后翻日志**看检索结果——没有"当场输入查询→立刻看检索 chunk"的界面;用户不需要 RAG 的 LLM 生成步骤,核心诉求是**检视检索效果**。
2. Evaluation Panel 停留在 v1 之前的形态:要求用户**手动填 Answer** 才能跑 Ragas,完全不认识 v2 五指标(含 `context_recall`)与新版报告格式(`batches` + `weighted_aggregate`,如 `logs/eval_v2_baseline_deepseek.json`)。

**决策**:增强 Streamlit,不开新前端(定位为纯自用检视工具,够用、快落地;FastAPI+React 的投入产出不划算——已与用户确认)。

**范围(本次做)**:
1. 新增 **Retrieval Playground** 页(核心,A2 深度:三列分路对比)
2. **重写 Evaluation Panel**(对齐 v2)
3. 为后续 A3(参数实验台)留扩展缝

**范围(不做,YAGNI)**:
- LLM 生成回答的聊天界面(用户明确不需要 RAG 最后一步)
- 新前端(FastAPI/React/Vue)
- 评估"运行"功能重构——评估照旧用 CLI `python scripts/evaluate.py` 跑,面板只负责**展示**报告(在 Streamlit 进程里嵌 RAGAS+LLM judge 长任务会卡 UI,自用场景 CLI 更稳)
- 面板内像素级 UI 测试

---

## 2. 架构总览

```
Streamlit Dashboard (src/observability/dashboard/)
├── app.py                      # 导航注册:Overview → Data Browser → Retrieval Playground → Ingestion
│                               #   Manager → Ingestion Traces → Query Traces → Evaluation Panel
├── pages/
│   ├── retrieval_playground.py # 新增:输入 query → 三列对比 + 状态条
│   └── evaluation_panel.py     # 重写:读 v2 报告,五指标卡 + 分批表 + 历史对比
├── services/
│   ├── retrieval_service.py    # 新增:唯一检索入口(A3 扩展缝),包 HybridSearch(return_details=True)
│   └── evaluation_report_service.py  # 新增:只读、纯函数式读评估报告 JSON
└── components/
    └── chunk_list.py           # 新增:共享 chunk 渲染组件(从 query_traces._render_chunk_list 抽取)
```

**数据流**:
- Playground:页面 → `RetrievalService.search()` → `HybridSearch.search(query, top_k, collection, return_details=True)` → `HybridSearchResult`(含 dense/sparse 分路结果、错误、fallback 标记) → 页面渲染三列 + 状态条。**trace 不落盘**(当场检视;历史走 Query Traces 页)。
- Evaluation:页面 → `EvaluationReportService` 读 `logs/eval_*.json` → dataclass → 渲染。

---

## 3. Retrieval Playground 页

### 3.1 输入区
- query 文本框 + "检索"按钮
- collection 下拉(从 Chroma 列出现有 collection,默认 `default`)
- top-k 数字框(默认 10;A3 时可直接挪用)

### 3.2 RetrievalService(A3 扩展缝)
- 封装 `HybridSearch.search(query, top_k, collection, return_details=True)`,把 `HybridSearchResult` 转成 UI 友好的 dict
- **service 是唯一检索入口**:未来 A3 加参数(rerank 开关等)只动 service + 表单控件,不动渲染逻辑
- `@st.cache_resource` 缓存 `HybridSearch` 实例,换 collection 才重建(避免每次点击重载 embedding/BM25)

### 3.3 结果区——三列并排对比(核心)
```
Dense (cosine)    │  Sparse (BM25)      │  Fusion (RRF)
rank. score src   │  rank. score src    │  rank. score src
1 0.83  paperA…   │  1 0.71  paperB…    │  1 0.83  paperA…
2 0.79  paperC…   │  2 0.69  paperA…    │  2 0.71  paperB…
```
- 每列一个 chunk 列表(用共享组件 `render_chunk_list`)
- 每个 chunk 可展开:chunk_id、source、title、**完整原文**(用户此前看不到的就是这个)
- **排名对照**:以 **dense∩sparse 交集 chunk_id 集合**为高亮基准,三列统一应用——交集 chunk 🟢高亮,单路召回(差集)⚪普通,一眼看出 RRF 是否救回了某路漏掉的 chunk

### 3.4 状态条
- 各阶段耗时(query_processing / dense / sparse / fusion)
- fallback 警告(某路失败显示 `dense_error`/`sparse_error`)
- **分词结果**:显示 `processed_query.keywords`(共享 tokenizer 的分词/词干化结果,英文 Porter stem + 中文分词),标注"与索引侧同源"——用户可直接对照 BM25 列判断稀疏路为什么召回/漏召

> 注:CLAUDE.md 中"jieba keyword extraction"的说法已过时(现为 `src/core/text/tokenizer.py` 共享 tokenizer,query 侧与索引侧同源,见 D-025)。CLAUDE.md 由用户自行修正,本设计不改它。

### 3.5 错误处理
- 查询为空 → 按钮置灰
- 检索异常 → 页内 error banner,不白屏
- collection 不存在 → 提示并列出可用 collection

---

## 4. Evaluation Panel 重写

**核心转变**:从"面板里跑评估"改为"**面板里读报告**"。

### 4.1 报告选择区
- 扫描 `logs/` 下 `eval_*.json`,下拉选择,默认最新
- 显示报告元信息:run 名、日期、n_qa、生成方式

### 4.2 聚合指标卡片区
- 五指标卡:`source_recall@5` / `source_precision@5` / `context_relevance` / `context_precision` / `context_recall`
- 每张卡显示加权聚合值(`weighted_aggregate`),附一行"这个指标量什么"的说明

### 4.3 分批明细表
- `batches` 数组 → 表格:每行一个 batch,五列指标
- 支持按任一列排序;低分 batch 高亮(如 `context_recall < 0.4` 标红)——"哪个 source 组最弱,一眼看到"

### 4.4 历史对比(轻量)
- 多选历史报告 → 五指标对比视图(A/B 对比调参效果)

### 4.5 删除的过时内容
手动填 Answer 交互、custom/composite 后端选择(custom 至今是脚手架)、`eval_history.jsonl` 旧历史格式——全部移除。

### 4.6 数据层
`EvaluationReportService`(services/evaluation_report_service.py):只读、纯函数式(读 JSON → dataclass),页面零文件 IO。

### 4.7 错误处理
- 无报告文件 → 引导文案("先用 `python scripts/evaluate.py` 生成")
- JSON 损坏 → 明确报错不白屏
- v1 格式(无 `context_recall` 键)→ 兼容显示,缺的指标标"—"

---

## 5. 共享组件与导航

### 5.1 chunk_list 组件
- 从 `query_traces._render_chunk_list` 抽取到 `components/chunk_list.py`
- 签名:`render_chunk_list(chunks, *, prefix, highlight=None)`——`highlight` 是一组 chunk_id,命中显示🟢交集高亮,未命中⚪普通
- Query Traces 页改为 import 该组件(行为不变,纯搬家)

### 5.2 导航顺序
Overview → Data Browser → **Retrieval Playground** → Ingestion Manager → Ingestion Traces → Query Traces → Evaluation Panel(高频页靠前)

---

## 6. 测试策略

- **unit**:
  - `RetrievalService`:mock HybridSearch,验证 return_details 数据转换、collection 缓存逻辑
  - `EvaluationReportService`:v2/v1 报告解析、损坏 JSON、空目录
  - `chunk_list` 组件:Streamlit AppTest 冒烟(渲染不抛错)
  - **回归点**:Query Traces 页搬家后 fusion/rerank 阶段 chunk 渲染行为不变(现有 AppTest/冒烟继续过)
- **integration**(已有基建):`streamlit run app.py` 子进程能启动、新页面注册成功

---

## 7. 依赖与验收

**依赖**:零新依赖(纯 Streamlit 现有能力)。

**验收标准**(成功 = 以下全部成立):
1. Playground 输入一条真实英文文献查询 → <10s 出三列对比 + 可展开原文
2. 同一查询换 top-k=20 重跑 → 结果即时刷新
3. Evaluation Panel 打开 `logs/eval_v2_baseline_deepseek.json` → 五指标卡 + 分批表正确渲染
4. v1 旧报告(无 context_recall)不报错
5. 全部单测过、`ruff check` / `mypy src` 干净

**后续扩展(A3,不在本次)**:参数实验台——top_k / rerank 开关 / collection 切换等可调参数矩阵,基于 RetrievalService 扩展缝实现。
