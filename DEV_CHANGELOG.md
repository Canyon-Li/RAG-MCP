# DEV_CHANGELOG — 关键决策记录

> 项目 ledger。**只记"为什么这么决定"**（背景 / 备选 / 理由 / 踩坑），不记"改了什么"——那是 `git log` 的职责。
>
> - **怎么用**：做新决策时，先翻索引表确认有没有相关的旧决策；新决策追加到对应分区末尾，并按模板补一条。
> - **和 git log 的区别**：git log 告诉你 `4f0c7e2` 加了 docling_vlm；本文件告诉你**为什么放弃了 ollama 远程 granite 方案**。
> - **状态约定**：`采纳` / `演化中` / `已弃用` / `待决`。

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
| D-013 | 2026-07-15 | 默认 LLM 切 Zhipu glm-4-flash | 采纳 | b80907c |
| D-014 | 2026-07-15 | embedding 切本地 ollama + httpx trust_env=False | 采纳 | 14318b8 |
| D-015 | 2026-07-15 | vision_llm 默认 disabled | 采纳 | [settings.yaml](config/settings.yaml#L35) |
| D-016 | 2026-07 | 运行环境用 conda，不建 .venv | 采纳 | memory: runtime-env-conda |
| D-017 | — | evaluator / cross_encoder 框架已搭但未完整测试 | 待决 | [evaluator/](src/libs/evaluator/) |
| D-018 | 2026-07-16 | 图片结构化数据从 Chroma metadata 彻底解耦到 ImageStorage SQLite | 采纳 | refactor/pdf-image-retrieval |

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
- **代价**：检索质量受 caption 质量上限限制；当前 vision_llm disabled，退化为 `[IMAGE: id]` 占位（见 D-015）。

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
- **后续**：vision LLM caption 生成当前仍 disabled（D-015），结构化 caption 链路已通，开启即用。G4–G7 留待后续。

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
