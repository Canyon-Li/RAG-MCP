# DEV_CHANGELOG — 关键决策记录

> **本项目的设计决策真相源。** 这是项目里唯一记"为什么这么设计"的文档——ADR 风格，每条决策记背景 / 备选 / 理由 / 代价 / 踩坑。
>
> - **怎么用**：做新决策时，先翻索引表确认有没有相关的旧决策；新决策追加到对应分区末尾，并按模板补一条。
> - **和 git log 的区别**：git log 告诉你 `4f0c7e2` 加了 docling_vlm；本文件告诉你**为什么放弃了 ollama 远程 granite 方案**。
> - **和 README / CLAUDE.md 的区别**：README 和 CLAUDE.md 描述"系统现在长什么样"（稳定架构）；本文件描述"为什么定成这样、怎么演化来的"。
> - **状态约定**：`采纳` / `演化中` / `已弃用` / `待决`。
>
> **历史说明**：项目早期有一份 `DEV_SPEC.md`（初版设计草案 + 任务排期表），后被代码与本文件双重超越，已于 2026-08 删除。不要再引用它；如需"为什么"，看本文件；如需"现在长什么样"，看 README 架构图 + CLAUDE.md `## Architecture`。

---

## 决策索引

| ID | 日期 | 决策 | 状态 | 关联 |
|---|---|---|---|---|
| D-001 | 2026-03-06 | config-driven everything（settings.yaml 唯一真相源） | 采纳 | f658c5a / [settings.py](src/core/settings.py) |
| D-002 | 2026-03-06 | 全链路 Base+Factory+Registry 插件化 | 采纳 | [src/libs/](src/libs/) |
| D-003 | 2026-03-06 | 多模态走 Image-to-Text（缝进 chunk），不用 CLIP | 采纳 | [image_captioner.py](src/ingestion/transform/image_captioner.py) |
| D-004 | 2026-03-06 | MCP stdio stdout 严格 + preload chromedb 防死锁 | 采纳 | [server.py](src/mcp_server/server.py) |
| D-005 | 2026-03-06 | Graceful degradation 是设计规则；rerank/eval 默认关闭 | 采纳 | [settings.yaml](config/settings.yaml) |
| D-006 | 2026-03-06 | 双幂等：文件 SHA256 + chunk 确定性 id | 采纳 | [file_integrity.py](src/libs/loader/file_integrity.py) |
| D-007 | 2026-03-06 | 两条 pipeline 全链路 trace + 稳定 stage 名 | 采纳 | [trace_context.py](src/core/trace/trace_context.py) |
| D-008 | 2026-07-14 | Loader → Parser 正名 + Factory 统一 settings 构造 | 采纳 | 42b7c0d (K1) |
| D-009 | 2026-07-14 | section-aware 切分 + 表格 embed/display 分离 | 采纳 | 9848569 (K2) |
| D-010 | 2026-07-14 | pdf_table provider：pdfplumber 原生表格 + 降级链 | 采纳 | a6362f5 (K4′) |
| D-011 | 2026-07-15 | 默认 parser 升级 pdf_table → docling | 采纳 | 0791b79 |
| D-012 | 2026-07-16 | docling_vlm：放弃 ollama 远程，走本地 transformers | 采纳 | 4f0c7e2 |
| D-013 | 2026-07-15 | 默认 LLM 切 Zhipu glm-4-flash | 已被 D-024 取代 | b80907c |
| D-014 | 2026-07-15 | embedding 切本地 ollama + httpx trust_env=False | 采纳 | 14318b8 |
| D-015 | 2026-07-15 | vision_llm 默认 disabled | 已被 D-020 取代 | [settings.yaml](config/settings.yaml#L35) |
| D-016 | 2026-07 | 运行环境用 conda，不建 .venv | 采纳 | memory: runtime-env-conda |
| D-017 | — | evaluator / cross_encoder 框架已搭但未完整测试 | 待决 | [evaluator/](src/libs/evaluator/) |
| D-018 | 2026-07-16 | 图片结构化数据从 Chroma metadata 彻底解耦到 ImageStorage SQLite | 采纳 | refactor/pdf-image-retrieval |
| D-019 | 2026-07-21 | docling 图片抽取从 get_images 改为 bbox 区域渲染（矢量图支持） | 采纳 | feat/docling-figure-render |
| D-020 | 2026-07-21 | vision_llm 切本地 ollama（llava-phi3）并默认启用 | 采纳 | 9905e3f |
| D-021 | 2026-07-21 | tags 过滤改为 post-fusion-only（剥离 pre-fusion + 读逗号字符串，存储层不动） | 采纳 | fix/tags-filter |
| D-022 | 2026-07-22 | filter pre/post 双层语义对齐：collection/source_path 走 post-fusion（修 N1 杀零 + N3 不一致） | 采纳 | 5a12c8b, 4a3e96d |
| D-023 | 2026-07-22 | QueryProcessor filter 解析收口：未识别 word:value 不当 generic filter + 删单字母别名（修 N4） | 采纳 | 548d508, 7763c04 |
| D-024 | 2026-08-13 | 默认 LLM 从云端 Zhipu 切本地 ollama granite4.1:8b（场景驱动：绝对本地/防泄露） | 采纳 | PR #7 |
| D-025 | 2026-08-13 | 英文 BM25 分词改造：公共 tokenizer + 零依赖 Porter stemmer（索引/查询单一真相源） | 采纳 | PR #7 |
| D-026 | 2026-08-13 | chunk_refiner 关闭 LLM 精炼层（策略 C：溯源优先确定性，规则层恒执行） | 采纳 | PR #7 |

---

## A. 架构基线决策（2026-03-06 初版）

### D-001 config-driven everything
- **背景**：要支持多 provider 自由切换（LLM/embedding/reranker/vector store/parser），且脚本要能从任意 CWD 运行。
- **决策**：`config/settings.yaml` 是唯一真相源 → 解析为 frozen dataclass；路径用 `resolve_path()` 锚定 `REPO_ROOT`（从 `__file__` 推导，CWD 无关）。
- **理由**：换后端 = 改一行 yaml，零代码改动；脚本不依赖"在哪个目录跑"。
- **代价**：配置项增多时 yaml 会变长；validation 必须 fail-fast（`SettingsError`）。

### D-003 多模态走 Image-to-Text，不用 CLIP
- **背景**：要支持"搜文字出图"。
- **备选**：① CLIP 跨模态向量（图文共享向量空间）；② Vision LLM 把图描述成文字，缝进 chunk。
- **决策**：选 ②。图片 → Vision LLM 生成 caption → 文本缝进 chunk body → 复用纯文本检索链路。
- **理由**：复用现成 dense+sparse 链路，不引入第二套向量索引；caption 本身可被关键词命中。
- **代价**：检索质量受 caption 质量上限限制；vision_llm 已在 D-020 默认启用（本地 ollama）。

### D-004 MCP stdio stdout 严格 + preload chromedb
- **背景**：MCP stdio transport 把 stdout 留给 JSON-RPC，任何日志打到 stdout 都会污染协议流；另外 MCP SDK 用 anyio 后台线程做 I/O，工具 handler 在 `asyncio.to_thread` 里 `import chromedb` 会和 stdin-reader 线程抢 Python import lock → 死锁。
- **决策**：① 启动时把所有 root logger handler 重定向到 stderr；② 在主线程 preload chromadb 及重模块，后续 worker 线程 import 只命中 `sys.modules`。
- **理由**：两条都是踩过才会知道的坑；保持这个 pattern 才能稳定跑 stdio server。
- **关联**：[server.py:25-79](src/mcp_server/server.py#L25-L79)。

### D-005 Graceful degradation 是设计规则
- **背景**：LLM 调用可能失败、可能没钱、可能没配 key；reranker/evaluator 是重依赖。
- **决策**：LLM-backed transform（chunk_refiner / metadata_enricher）失败或 `use_llm:false` → 降级到 rule-based；reranker 失败 → 降级到 RRF 顺序；`rerank.enabled` / `evaluation.enabled` 默认 `false`。
- **理由**：这些组件"不能阻塞主链路"。检索本身必须永远能出结果。

---

## B. Parser 演进线（2026-07）

这条线决策密度最高，核心问题始终是：**PDF 表格怎么解析才不丢信息**。

### D-010 pdf_table provider + 降级链（K4′）
- **背景**：纯文本解析（pdf_text / MarkItDown）丢表格结构。
- **决策**：新增 `pdf_table` provider 用 pdfplumber 提取**线条/对齐型表格**（Word/Excel 导出的文本型 PDF）；表格进 `metadata.table_html`（展示）+ 清洗纯文本（检索）。扫描件/图片型 PDF 自动降级到 `pdf_text` 并标 `degraded=true`。
- **理由**：区分"检索文本"与"展示 HTML"两条用途（见 D-009）。
- **代价**：不支持扫描件表格（需 OCR）——这是主动接受的范围。

### D-011 默认 parser 升级到 docling
- **背景**：pdfplumber 对复杂版面/带 caption 的表格仍偏弱。
- **决策**：新增 `docling` provider（DocLayNet 版面 + TableFormer 表格），表格直接用 Docling GFM（text+html 双字段），docling 失败降级 pdf_text；默认 `pdf_table → docling`。
- **证据**：实测 `chinese_table_chart_doc.pdf` 5 表全 GFM + caption，17 chunks（vs pdfplumber 11）。
- **理由**：表格质量实测更好，且天然带 caption 关联。

### D-012 docling_vlm：放弃 ollama 远程，走本地 transformers ⚠️
- **背景**：想用 Granite-Docling 做版面/表格识别。两条路线：① C2 = ollama 远程跑 granite-docling；② C1 = 本地 transformers CPU。
- **决策**：选 C1（本地）。`DoclingVlmParser` 继承 `DoclingParser` 覆盖 `_build_converter` 用 VlmPipeline + 本地 Granite-Docling-258M；HF cache 钉到 `D:\Transformers\Model`。
- **理由（踩坑）**：C2 实测不可行——granite-docling 是 **completion** 模型，ollama 的 chat template 会破坏输出（`<loc_>` 无 `<doctag>`），已放弃。另外 HF cache 钉 ASCII 路径是为了**避开中文 user 目录**导致的加载问题。
- **现状**：C1 实测中文表格识别偏弱（1/5 表），目前定位在**扫描件/图片型表格**这一 pdfplumber/docling 搞不定的场景。
- **教训**：completion 模型不能直接套 chat template——这类"模型类型不匹配"的坑值得记。

---

## C. Provider / 运行时选型

### D-014 embedding 切本地 ollama + httpx trust_env=False ⚠️
- **背景**：embedding 从 Zhipu embedding-3(2048d) 切本地 ollama nomic-embed-text(768d)。但本地 ollama 调不通：`curl` 能通，`httpx` 却 502。
- **根因**：系统代理（clash/v2ray）会拦截 localhost 请求，httpx 默认读系统代理环境变量 → 命中代理 → 502。
- **决策**：ollama httpx Client 显式 `trust_env=False`，绕过系统代理。
- **教训**：`curl 通但 httpx 不通` 的典型根因就是系统代理拦截 localhost——以后遇到先查这。见 memory: system-proxy-intercepts-localhost。

### D-016 运行环境用 conda，不建 .venv
- **背景**：全局 Python 3.14 装不了 chromadb（依赖未跟上）。
- **决策**：项目跑在 conda env `langchain-test`，**不要建 `.venv`**。
- **理由**：chromadb 对 Python 版本敏感，需要一个受控的、版本合适的解释器。
- **关联**：memory: runtime-env-conda。

### D-024 默认 LLM 从云端 Zhipu 切本地 ollama granite4.1:8b（场景驱动：绝对本地）
- **状态**：采纳。
- **背景**：D-013 把默认 LLM 定为云端智谱 glm-4-flash。新场景"纯本地 + 英文学术文献 + 溯源检索"要求**私有 PDF 不得出本机**——这与云端 LLM 直接冲突。需要把主 LLM 也挪到本地 ollama（embedding/vision 已先后本地化，见 D-014 / D-020）。
- **备选**：① 保持云端但只传摘要/片段（治标，仍泄露）；② 主走本地、答案生成留云端（但本场景明确不要综述，LLM 只服务 ingestion）；③ **全本地**。
- **决策**：选 ③。`llm.provider: zhipu` → `ollama`，`model: granite4.1:8b`。`OllamaLLM` 早已注册（D-002 Factory），`trust_env` 在 D-014 已就绪（embedding/vision 侧）。
- **理由**：本场景里 LLM **只服务 ingestion 阶段**（chunk 精炼/元数据），**不服务查询阶段的答案生成**（场景明确不要综述，见 [spec](docs/superpowers/specs/2026-08-13-local-english-literature-rag-design.md)）。所以本地 8B "推理慢"只影响一次性 ingest，不影响日常查询——这是本场景对本地化特别友好的地方。granite4.1:8b 在结构化抽取（标题/摘要/tags）任务上够用。
- **代价 / 现状**：PR #7 `feat/local-english-literature-rag`。**code-review 暴露的预存 bug（已随本决策修复）**：`OllamaLLM._call_api`（文本 LLM 路径）历史遗留**缺 `trust_env=False`**（D-014 只补了 embedding/vision）——配置切到 ollama 后一旦文本 LLM 真被调用（`evaluation.enabled: true`）即踩 D-014 同款代理坑 502；且构造器不读 `settings.llm.base_url`（死配置），endpoint 是原生 `/api/chat` 而非 `/v1`，settings 的 `/v1` 后缀需剥离。两处已在同 PR 补齐 + 加 4 个测试锁住。
- **关联**：[ollama_llm.py](src/libs/llm/ollama_llm.py)；[settings.yaml](config/settings.yaml)；[spec](docs/superpowers/specs/2026-08-13-local-english-literature-rag-design.md)。与 [[D-013]]（被取代）、[[D-014]]（trust_env 模式源）、[[D-020]]（vision 本地化前置）关联。运行时前置：`ollama pull granite4.1:8b`。

### D-025 英文 BM25 分词改造：公共 tokenizer + 零依赖 Porter stemmer（索引/查询单一真相源）
- **状态**：采纳。
- **背景**：原 BM25 分词两侧都用 `jieba.lcut`——jieba 是中文分词器，对英文只是"原样保留"：**无词干化**（`optimization`/`optimize`/`optimal` 算 4 个不同 term，查一个召不回另三个）、**停用词处理弱**。对英文学术文献场景，BM25 召回质量明显打折。更隐蔽的问题是两侧 `_tokenize` **逻辑各自实现**，已有不一致风险。
- **备选**：① 换 BM25 库（rank_bm25 / bm25s，自带英文 tokenizer）——要适配现有 JSON 倒排索引结构，改动大；② 引 nltk / snowballstemmer——重依赖；③ **抽公共 tokenizer + 零依赖纯 Python Porter stemmer**。
- **决策**：选 ③。新增 `src/core/text/tokenizer.py` 的 `tokenize()` + `porter_stemmer.py` 的 `stem()`，作为**索引侧 `SparseEncoder` 与查询侧 `QueryProcessor` 共用的唯一分词函数**。中文走 jieba（不 stem），英文走 lowercase→停用词→Porter stem→长度过滤，双语并行。
- **理由**：**两侧共用同一函数是 BM25 召回的硬前提**——索引存的 term 必须和查询查的 term 一致，否则永远召不回。抽成单一真相源比"两侧各写一份保持同步"可靠。选零依赖 Porter 而非 nltk：本项目是学习/面试项目，引 nltk 太重，Porter 单文件纯 Python 几十行即够（BM25 场景 Porter 足够好）。双语并行而非纯英文：保留中文能力，万一中文提问也能命中，改动成本相同。
- **代价 / 现状**：PR #7。**关键设计取舍**（经终审 + code-review 修正后定型）：① `dedupe` 参数——索引侧 `dedupe=False` 保真词频（BM25 TF 信号，终审发现去重把 tf 抹成 1 的回归）；查询侧默认 `dedupe=True`。② 有意不对称——query 侧 `min_term_length=2` 对齐 index（code-review V3：旧 query=1 产生 index 不存的单字 term 零召回）。**分词逻辑变了，旧 BM25 索引失效，重新 ingest 必须 `--force`**。136 直接相关测试全绿，跨层一致性测试 `test_query_and_index_tokenization_match` 是召回保证的安全网（reviewer trace 确认非空转）。
- **实现期踩坑（code-review 高严重度，已修）**：① **V2** `_SPLIT_RE` 把 `C++`/`C#`/`R-CNN` 拆碎→技术符号 token 整体保留（`_TECH_TOKEN_RE`，纯字母词仍走 stem）；② **V4** use 家族 `used/using/uses` stem 到 `us`（非停用词 `use`）成近零 IDF 垃圾词→停用词表补 stem 后形态（`_build_stemmed_stopwords`）拦截。已知局限：连字符词 `K-Means` 被 jieba 预拆成 `['K','-','Means']`，需预粘合才完整可检索，超出本次范围。
- **关联**：[tokenizer.py](src/core/text/tokenizer.py) / [porter_stemmer.py](src/core/text/porter_stemmer.py)；[sparse_encoder.py](src/ingestion/embedding/sparse_encoder.py) / [query_processor.py](src/core/query_engine/query_processor.py)；[spec](docs/superpowers/specs/2026-08-13-local-english-literature-rag-design.md)。停用词表仍定义在 query_processor（后续可挪到 `src/core/text/`，见 D-017 同类"待整理"债）。

### D-026 chunk_refiner 关闭 LLM 精炼层（策略 C：溯源优先确定性）
- **状态**：采纳。
- **背景**：`chunk_refiner` 有两层独立路径——**规则层**（恒执行，去 PDF 提取噪声：页眉分隔线/HTML 残留/空白规整，纯确定性不改文字）和 **LLM 层**（`use_llm: true` 时，prompt 虽克制地要求"保留原意"，但仍是概率性精炼）。新场景核心诉求是**溯源回论文原话**，对"文本被模型动过"零容忍。
- **备选**：① 保持 LLM 层（信任 prompt 约束）；② 永久关闭；③ **策略 C：先关，跑一轮看 chunk 质量不满意再开**。
- **决策**：选 ③ 的第一步——`ingestion.chunk_refiner.use_llm: false`。规则层恒执行（`use_llm` 只控制 LLM 分支），所以只是关掉概率性精炼。
- **理由**：溯源场景优先**确定性**——即便 prompt 明令禁止改写，8B 模型实际执行仍可能微调措辞（标点/冠词），有破坏逐字溯源的风险；而 docling + section-aware chunker 产出的学术文本已足够干净，LLM 精炼的边际收益小。关掉还顺带省一次性 ingest 的本地推理时间。选策略 C 而非 ②：留观察窗口，若规则层不够干净（断句/连字符残留）再开。
- **代价 / 现状**：PR #7。`metadata_enricher.use_llm` 保持 `false`（tags/summary 不参与召回，本地 8B 逐 chunk 生成纯浪费 ingest 时间）。rerank 保持 `enabled: false`（场景是召回+溯源非精排喂生成，< 50 篇小库噪音不明显）。三者共同构成"本场景比通用 RAG 更轻"的减负。
- **关联**：[chunk_refiner.py](src/ingestion/transform/chunk_refiner.py)；[settings.yaml](config/settings.yaml)；[spec](docs/superpowers/specs/2026-08-13-local-english-literature-rag-design.md) §3.3.1。与 [[D-005]]（graceful degradation 规则）一致——规则层是确定性兜底。

---

## D. 待决与已知债

### D-017 evaluator / cross_encoder 未完整测试
- **状态**：待决。
- **现状**：`BaseEvaluator` + `EvaluatorFactory` + `custom_evaluator` / `ragas_evaluator` / `composite_evaluator` 框架已搭好，golden test set 也存在；但 `evaluation.enabled: false`，且 README 自述"未经过完整测试"。`cross_encoder_reranker` 同样是框架未测、默认关闭。
- **下一步建议**：见体检报告 P0——先打开 eval 用 golden set 跑通 hit_rate/mrr 基线，把这条从"待决"推进到"采纳"。

### D-018 图片结构化数据从 Chroma metadata 彻底解耦到 ImageStorage SQLite
- **状态**：采纳。
- **背景**：`PDF处理链路分析.md` 发现 PDF 图片存取闭环断链（G1）。根因是 Chroma 只接受标量 metadata，而代码把 `images`（dict 列表）/ `image_captions`（list of dict）等复合结构直接塞了进去，落盘即有损；同时 `MultimodalAssembler` 从未注入 `ImageStorage`，三层路径全失败。连带 G2（caption schema 不匹配）、G3（页码丢失）同根同源。
- **备选**：A = 保留 Chroma 字段打补丁（JSON 序列化进出）；B = 双写（Chroma + ImageStorage 各一份）；C = **彻底解耦**，图片结构化数据只存 ImageStorage SQLite，Chroma 只存文本+向量。
- **决策**：选 C。`[IMAGE: id]` 占位符是查询端反查图片的唯一线索；在 `image_index` 表加 `caption` 列（PRAGMA-based ALTER TABLE），`set_caption`/`get_image_meta` 统一读写；图片注册从 pipeline stage 6c 前移到 stage 2.5（parse 后立即注册）。
- **理由**：从根因消除有损标量化，G2 被结构性消掉（caption 不进 Chroma）；图片注册前移保证 `set_caption` 时行已存在；`[IMAGE: id]` 占位符作为唯一 key 比 metadata 里藏一份副本更可靠。选 C 而非 A 是因为 JSON 序列化仍绕不开 Chroma 的 metadata 长度限制和查询端反序列化。
- **代价 / 现状**：已合入 main（PR #3，8 commits，`refactor/pdf-image-retrieval` 分支）。G1/G2/G3 全部修复，含 6 个新单元测试 + 集成测试，端到端验证通过。历史 ingest 数据需重新摄取（已有幂等性，不受影响）。关联 memory: image-extraction-raster-only。
- **后续**：vision LLM caption 生成已随 D-020 默认启用（本地 ollama / llava-phi3）。G4–G7 留待后续。

---

## D. 图片抽取演进（2026-07-21）

### D-019 docling 图片抽取从 get_images 改为 bbox 区域渲染（get_pixmap）
- **状态**：采纳。
- **背景**：D-018 把图片结构化解耦到 ImageStorage 后，存取闭环通了，但底层抽取仍是 PyMuPDF `page.get_images(full=True)`——只查 PDF 图对象树，矢量图（matplotlib 图表、流程图、量子电路图）不是图对象，完全捕获不到。实测一篇全矢量的量子计算论文 `images=0`。关联 memory: image-extraction-raster-only。
- **备选**：A = 混合保留（get_images 抽光栅 + 仅对未覆盖区域 bbox 渲染，去重判断复杂）；B = **全部走 bbox 渲染**（丢掉 get_images，Docling 识别到的每个 Figure 区域都 `get_pixmap(clip=bbox)`）；C = 加 `image_strategy` 配置开关让用户选。
- **决策**：选 B。Docling 版面分析已经给出每个 Figure 的 bbox，一个 `get_pixmap(clip=bbox)` 把光栅图和矢量图一并解决，代码大幅简化。只改 docling_parser；pdf_text / pdf_table 无版面分析能力，保持 get_images 不变。
- **理由**：论文场景下矢量图是主流；Docling 已经做了版面分析，bbox 白拿；光栅图本来就是渲染产物，再渲染一次损失可忽略（DPI=200）。选 B 而非 A 是因为去重判断（bbox 是否已被光栅图覆盖）收益不大却显著增复杂度。
- **代价 / 现状**：PR #4（`feat/docling-figure-render`）。`RENDER_DPI=200` / `MIN_FIGURE_SIZE=50pt` 硬编码类常量（将来需要再加配置）。14 个单元测试覆盖渲染/过滤/降级路径。
- **踩坑（坐标系）**：Docling `prov.bbox` 用 PDF **bottom-left origin**（`t > b`），PyMuPDF `fitz.Rect` 用 **top-left origin**。初版代码 `height = b - t` 算出负数 → 所有真实图被 MIN_FIGURE_SIZE 误过滤（实测 0 张产出）；即使过了过滤，Rect 没翻转 y 轴渲染区域也是错的。修复：`height = abs(b - t)`；Rect 用 `page_height - y` 翻转。**真实 PDF 端到端验证才发现**——单元测试全 mock 了 `fitz.Rect`，坐标系假设没被覆盖。教训：坐标系这类底层假设必须有真实数据验证，不能停在 mock 层。
- **关联**：5400034；spec [docs/superpowers/specs/2026-07-21-docling-figure-render-design.md](docs/superpowers/specs/2026-07-21-docling-figure-render-design.md)；memory: image-extraction-raster-only。

### D-020 vision_llm 切本地 ollama（llava-phi3）并默认启用
- **状态**：采纳（取代 D-015 的"默认 disabled"）。
- **背景**：D-015 默认关闭 vision_llm 的理由是"需要云端 API key"。现在本地有 Ollama + llava-phi3:3.8b，无需 key、无外部依赖，D-015 的前提不再成立。D-018 的 caption 链路也要求 vision_llm 开启才能真正生成描述。
- **备选**：A = 继续 disabled，用户手动开；B = **默认启用本地 ollama**；C = 默认启用但走云端（azure/openai）。
- **决策**：选 B。新增 `OllamaVisionLLM`（继承 `OpenAIVisionLLM`，复用 OpenAI 兼容的 `/v1/chat/completions` 消息格式），覆盖初始化（API key 占位、base_url 默认 `http://localhost:11434/v1`）和 HTTP 客户端（`trust_env=False` 绕系统代理，同 D-014）。settings.yaml 切到 `provider: ollama`、`enabled: true`。
- **理由**：本地模型零成本、无 key、隐私友好；Ollama 的 `/v1` 端点完全 OpenAI 兼容，继承父类省掉图片处理逻辑；`trust_env=False` 是 D-014 已验证的本地服务调用必备（系统代理会拦 localhost）。选 B 而非 C 是因为本项目定位是学习/面试，本地优先。
- **代价 / 现状**：llava-phi3:3.8b 是小模型，复杂科学图描述质量有限（能用但不惊艳）；如需更强可换 llava-llama3 或走云端。已合入 main（`9905e3f`）。
- **关联**：9905e3f；[ollama_vision_llm.py](src/libs/llm/ollama_vision_llm.py)；memory: system-proxy-intercepts-localhost。

---

## E. 查询过滤（2026-07-21）

### D-021 tags 过滤改为 post-fusion-only（存储层不动）
- **状态**：采纳。
- **背景**：`PDF处理链路分析.md` §5 G4 发现 tags 元数据过滤完全不工作，三层故障叠加：① Chroma `_sanitize_metadata` 把 tags list 逗号拼接成字符串（标量约束的必然结果）；② `search()` 把含 tags 的 filters 直传 retrievers → Chroma `where={"tags":["azure"]}` 把 list 当 `$in` 去匹配逗号字符串 → 永不命中 → dense **杀零**；③ post-fusion 的 `_matches_filters` 对字符串做 `set()` → 按字符级拆分 → 语义错误。`QueryProcessor` 已能从 `tag:azure 架构图` 解析出 filters，**入口通但执行端全断**。
- **备选**：范围上 A=仅修内联 `tag:` 语法 / B=加 MCP 显式 filters 参数 / C=含 pre-fusion Chroma 过滤；存储上 A=保留逗号拼接 / B=改 JSON 数组字符串；剥离注入点 = `hybrid_search.search()` 剥离 / = `chroma_store._build_where_clause` 跳过 tags。
- **决策**：**范围 A + 存储 A + hybrid_search 层剥离**。`search()` 传 retrievers 前用字典推导剥离 `tags`（tags 只走 post-fusion）；`_matches_filters` tags 分支改读逗号字符串（`split(",")` + `strip()` 再求交，防御性兼容 list 形态）；存储层 `chroma_store.py` 零改动。
- **理由**：tags 是 list 语义，Chroma 对字符串字段做不了 "list contains"，下推到 Chroma where 无意义反而杀零；post-fusion 在 Python 内存里 list 语义天然好处理。存储层不动 → 向后兼容已摄取数据，免重新摄取。"tags 是 post-fusion 专属"语义集中在 query 层一处，存储层保持通用。范围选 A 是因为内联 `tag:` 入口已存在，修通即兑现能力，MCP 显式参数是 YAGNI。
- **代价 / 现状**：实现完成在分支 `fix/tags-filter`（commits `53c948b` + `389c293` + `bb16e85`），6 个新测试（5 单元 + 1 端到端），final review 判定 **Ready to merge**，待合并。**限制**：tags 过滤依赖 `metadata_filter_post=True`（默认 True）；若设 False 则 tags 静默失效（已在 `HybridSearchConfig` docstring 标注）；标量 filter（collection/doc_type/source_path）不受影响，仍可 pre-fusion。
- **关联**：[hybrid_search.py](src/core/query_engine/hybrid_search.py)；`PDF处理链路分析.md` §5 G4。与 [[D-018]] 形成对照——同根问题"Chroma 存不了复合类型"，D-018 把图片**移出** Chroma 到 SQLite，本决策把 tags **留在** Chroma（逗号字符串）但只 post-fusion 消费。G5–G7 留后续。

### D-022 filter pre/post 双层语义对齐（修 N1 + N3）
- **状态**：采纳。
- **背景**：QueryProcessor 支持内联 filter 语法（`collection:`/`source:` 等），但 ① `collection` 是物理隔离维度（parser/chunker/enricher/upserter 都不写进 metadata），下推 Chroma `where` 必杀零，post-fusion 对缺失 key 也排除 → Dense+Sparse 双杀零（N1）；② `source_path` Dense 下推用 exact、post-fusion 用 partial，同一条结果两层语义不一致（N3）。
- **备选**：① `source_path` 语义 partial vs exact；② generic 缺字段 排除 vs 保留；③ N4 是否纳入本批。详见 [spec](docs/superpowers/specs/2026-07-22-filter-pre-post-alignment-design.md) §3。
- **决策**：① source_path=**partial**（两层一致）；② generic 缺字段=**排除**；③ N4 拆单独议题。新增模块常量 `POST_ONLY_FILTERS = {tags, collection, source_path}`，在 `_run_dense_retrieval` 内剥离（不进 Chroma where，保 sparse 选 BM25 index 的 collection 路径）；post-fusion `_matches_filters` 对 `collection` 改 `continue` 放行（⚠️ 不能用 `return True`，会跳过后续 key 检查）。
- **理由**：物理隔离已保证 collection 隔离，不该再进 where；长路径 exact 几乎无法命中，partial 零增量成本；同一 key 两层语义必须一致，否则一层通过另一层干掉。
- **代价 / 现状**：改动集中在 `hybrid_search.py`（`search()`/`_run_dense_retrieval`/`_matches_filters`/`HybridSearchConfig` docstring），`QueryProcessor`/`ChromaStore`/`settings` 零改动。43 测试全绿；e2e dogfood（`default` collection, ollama nomic-embed-text）：`algorithm collection:default` → DENSE=20 + FUSION=10（修复前双杀零=0）。commits `5a12c8b` + `4a3e96d`，final review (opus) Ready to merge。
- **关联**：[hybrid_search.py](src/core/query_engine/hybrid_search.py)；[spec](docs/superpowers/specs/2026-07-22-filter-pre-post-alignment-design.md) / [plan](docs/superpowers/plans/2026-07-22-filter-pre-post-alignment.md)；`PDF处理链路分析.md` §5 N1+N3。与 [[D-021]] 同根（Chroma where 表达力不足 → post-fusion 兜底），本决策把 collection/source_path 也归入 post-only。N4 解析层根因见 [[D-023]]。

### D-023 QueryProcessor filter 解析收口（修 N4）
- **状态**：采纳。
- **背景**：N4 = 自然语言里随处可见的 `word:value`（`Azure:服务端`、`12:30`、`https://...`、`c:\Users\...`）被无边界正则 `(\w+):([^\s]+)` 匹配 + `else` 分支当 generic filter；因对不上任何真实 metadata 字段 → Dense `where` 杀零 + 关键词被 `sub` 删除 + post-fusion 排除，三路全废。generic filter 是"死功能"（metadata 字段固定，用户写的 custom key 永远命不中）。
- **备选**：① 删 generic 白名单化 / ② 保留 generic + 启发式排除（治标）/ ③ 改显式语法标记（破坏现有语法）；白名单来源 静态 vs config；范围 现有 4 类 vs 扩展；单字母别名 `c`/`s`/`t` 删否。详见 [spec](docs/superpowers/specs/2026-07-22-n4-filter-parsing-allowlist-design.md) §3。
- **决策**：**① 删 generic（白名单化）+ 静态写死 + 现有 4 类 + 删单字母别名**。`_extract_filters` 重构为 `finditer` 逐段处理：白名单 key（`collection/col`、`type/doc_type`、`source/src`、`tag/tags`）收 filter 并从 query 文本删除；未识别 `word:value` 原样保留为查询文本参与分词。正则 `FILTER_PATTERN` 不动。
- **理由**：generic 是死功能，放弃零损失；filter key 与下游处理逻辑强绑定（collection→物理隔离、tags→post-fusion、source_path→partial、doc_type→exact where），加新 key 不止改配置还得改下游分支，所以白名单是代码契约而非配置；单字母别名是 N4 近亲（`c:\路径`、`s:3`），多字母别名够便捷。
- **代价 / 现状**：改动集中在 `query_processor.py`（`_extract_filters` 重构 + 删别名 tuple），`hybrid_search`/`chroma_store`/`settings` 零改动（下游 generic else 成死代码，防御性保留不清理，spec §7）。43 测试全绿；e2e：`Azure:服务端 配置` → `filters=(none)` + FUSION=10（修复前 `filters={"azure":"服务端"}` 杀零=0）。commits `548d508` + `7763c04`（keyword 断言回归保险），final review (opus) Ready to merge。
- **关联**：[query_processor.py](src/core/query_engine/query_processor.py)；[spec](docs/superpowers/specs/2026-07-22-n4-filter-parsing-allowlist-design.md) / [plan](docs/superpowers/plans/2026-07-22-n4-filter-parsing-allowlist.md)；`PDF处理链路分析.md` §5 N4。与 [[D-021]]/[[D-022]] 同根（filter 收口系统化的第三层——解析层），从源头不让 generic filter 产生，下游不再误触发。

---

## 追加模板

做新决策时，复制以下结构追加到对应分区末尾，并在索引表加一行：

```markdown
### D-0XX <决策标题>
- **背景**：遇到什么问题 / 什么触发了这个决策。
- **备选**：考虑过哪些方案（如有）。
- **决策**：选了什么。
- **理由**：为什么选这个（含实测数据 / 踩坑教训）。
- **代价 / 现状**：接受了什么限制，或当前进展。
- **关联**：commit hash / 文件 / 相关 memory。
```

> 写作纪律：① 没有证据（commit / 文件 / 实测）的决策不要写；② "改了什么"交给 git，这里只写"为什么"；③ 弃用的方案和踩过的坑同样重要，写下来避免重蹈。
