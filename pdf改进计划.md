# PDF 改进计划

> 本文件记录 PDF 解析能力增强的架构决策与实施计划。由讨论沉淀,作为实施前对齐文档。**实施代码不在本计划范围;本文件锁定决策,供实施时对照。**
>
> **2026-07-14 修订**:文档画像明确为"多数文本型 PDF",**sidecar 方案砍掉**,改为 `pdf_table` 原生方案(本项目内 pdfplumber,零模型/零外部服务)。

## 1. 背景与目标

为项目增加 PDF 表格读取能力。**文档画像:多数文本型 PDF**(有文字层,Word/Excel 导出),少数扫描件。因此采用**本项目内纯算法表格方案**(pdfplumber),不走 sidecar、不装模型。

### 参考的开源项目
- RAGFlow(`D:\Desktop\git\repo\ragflow`)的 deepdoc 提供了"检索用纯文本 + 展示用 HTML"的表格处理思路(其 `tokenize` 对 HTML 去标签后进 BM25 字段)。本计划**借鉴其 embed/display 分离的设计,但不移植其代码、不用其 sidecar**——多数文本型 PDF 不需要 OCR/版面模型,纯算法表格库就够。

### 本项目关键约束
- 运行环境 python **3.13**
- 面试向:每个决策要能讲清"为什么";架构清晰度 > 功能堆砌
- 红线:**可插拔架构(Factory+Registry+Base*)不能破**;已有契约可删改

## 2. 架构决策

### 决策 1:两层 Parser / Chunker(不加 Loader)
- **Parser**:吃文件路径 → 产带类型的结构化分段。可插拔(`BaseParser` + `ParserFactory`)
- **Chunker**:吃分段 → 产最终 chunk。尊重 Parser 给的类型边界(表格不切碎)
- **不加 Loader 层**:本地文件项目不需要(Loader 只在多来源场景有价值)

### 决策 2:PDF 表格走 pdf_table 原生方案(本项目内,pdfplumber)
- 新增 `pdf_table` provider(`BaseParser` 子类),用 **pdfplumber `extract_tables()`** 提取表格(纯算法,基于线条/对齐)
- **零模型、零 sidecar、零 onnxruntime**:pdfplumber 纯 Python,python 3.13 兼容
- 表格 → HTML(`metadata.table_html`)+ 清洗纯文本(`chunk.text`),套用 embed/display 分离
- **砍掉原 sidecar 方案**(RAGFlow deepdoc_server + docker + 模型)——文本型 PDF 不需要

### 决策 3:降级链本项目内
- `pdf_table` → 无表格/失败 → `pdf_text`(MarkItDown 纯文本)
- 全进程内,无 HTTP 探测、无 sidecar

### 决策 4:WordLoader 跟随正名
- 基类 `BaseLoader` → `BaseParser`,注册 `LoaderFactory` → `ParserFactory`,解析逻辑不改

### 决策 5(未来可选):sidecar 留作扫描件增强
- 少数扫描件表格(需 OCR)`pdf_table` 做不到 → 现在接受降级到 `pdf_text`(表格丢)
- 若未来扫描件变多,再评估 sidecar(RAGFlow deepdoc)作可选增强——**本轮不实施**

## 3. 决策锁定表

| 决策项 | 结论 |
|---|---|
| 整体架构 | 两层 Parser / Chunker;不加 Loader |
| 本轮功能范围 | 只 PDF(`pdf_text` + `pdf_table`);架构铺垫跨格式;query 不动 |
| PDF 表格方案 | **pdf_table 原生**(pdfplumber,本项目内),零 sidecar/零模型 |
| WordLoader | 跟随正名(基类→`BaseParser`、注册→`ParserFactory`),解析不改 |
| 降级链 | `pdf_table` → `pdf_text`(本项目内,无 sidecar) |
| Chunker | 升级:读 `section_type`,表格整块保留、标题并入正文、超长正文段内递归切 |
| 数据结构 | `Chunk`/`ChunkRecord` 加可选 `section_type`/`bbox`/`table_html`(进 metadata,向后兼容) |
| 表格表示 | **HTML**(展示)+ 清洗纯文本(embed) |
| image_refs 契约 | Parser 必须遵守 `{id,path,page,...}` + `[IMAGE:{id}]` 占位符 |
| 数据迁移 | `chunk_id` 变 → `--force` 重灌或新 collection |
| query engine | 不动(只通过 `ChunkRecord` 存储契约耦合) |
| sidecar | **砍掉**(留作未来扫描件可选增强,本轮不实施) |

## 4. 范围

### 本轮做
- PDF:`pdf_text`(纯文本)+ `pdf_table`(原生表格)+ 降级链
- 架构铺垫(跨格式):
  - `BaseParser` + `ParserFactory`(新建)
  - `BaseLoader` → `BaseParser` 正名(含 WordLoader 跟随)
  - `Chunker` 升级(尊重类型边界)
  - `Chunk`/`ChunkRecord` 加可选字段(`section_type`/`bbox`/`table_html`)

### 本轮不做
- query engine(免疫,不动)
- 其他格式解析(docx/excel/ppt/md/...)
- Loader 层(未来按需)
- **sidecar / RAGFlow repo 改动 / docker / 模型**(砍掉,留作未来可选)

## 5. 关键约束(红线)

1. **可插拔架构不能破**:`Factory + Registry + Base*` 模式保留;新增 `BaseParser` + `ParserFactory` 与现有 9 个 `Base*` 同构
2. **query engine 不动**:ingestion 与 query 只通过 `ChunkRecord` 存储契约耦合;新字段是 metadata 增量,query 不读也能跑(ResponseBuilder 读 `table_html` 做展示是允许的可选增强,**不改检索逻辑**)
3. **image_refs 契约必须遵守**:`{id, path, page, text_offset, text_length, position}` + text 内 `[IMAGE:{id}]` 占位符
4. **新字段向后兼容**:`section_type`/`bbox`/`table_html` 可选,不破坏老数据结构(符合 `types.py` "最小必需 + 可扩展"原则)

## 6. 数据迁移

- `chunk_id = {doc_id}_{index:04d}_{content_hash8}`;Parser 改了 section 含义 → id 变
- 老数据变孤儿 → 用 `--force` 重灌,或开新 collection

## 7. 接口契约设计

### 7.1 现状关键事实(设计前提)
- 现 `DocumentChunker.split_document()` 吃 `Document`,`splitter.split_text(document.text)` 纯文本切分,**无分段类型概念**
- `chunk_id` 真实格式:`{doc_id}_{index:04d}_{content_hash8}`(注:CLAUDE.md 里写的 `hash(source_path+section+content_hash)` 不准确,需修正)
- 现 `BaseLoader.load() -> Document`,设计原则已声明 "No Splitting"
- 现 `ingestion.loader: {provider: pdf, extract_images: true}`;`chunk_size: 1000`,`chunk_overlap: 200`

### 7.2 BaseParser 接口
```python
class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str | Path) -> Document: ...
    # 复用 BaseLoader 的 helper:_validate_file / _compute_file_hash / _extract_title / _generate_image_id
```
- ✅ Parser 输出仍是 `Document`(向后兼容 Transform 层),结构化分段放 `document.metadata["sections"]`(可选)

### 7.3 Section 数据结构
```python
@dataclass
class Section:
    type: str            # title | text | table | figure | figure_caption | list | header | footer | equation
    text: str            # 表格:清洗纯文本("列名: 值");其他:纯文本
    page: int
    bbox: dict | None
    images: list[dict]   # figure 关联的图片引用
    html: str | None     # 表格 HTML(展示用)
```

### 7.4 pdf_table 输出契约(本项目内,无 sidecar)
- `pdf_table_parser.py` 用 `pdfplumber.open(fnm)` → 逐页用**带 bbox 的文本块**(`page.extract_words()` 或 `page.extract_text(layout=True)`)+ 表格(`page.find_tables()` / `extract_tables()` 的 bbox)**按 top 坐标交替**产出 sections(文本块和表格按垂直位置排序,表格插到文本流正确位置)。⚠️ `extract_text()` 默认只返回纯字符串**无 bbox**,**位置对齐必须用带 bbox 的 API**(extract_words / layout 文本 + 表格 bbox 按 top 排序)
- 每个 PDF 产出:
  - `document.metadata["sections"]`:文本段(type=text)+ 表格段(type=table,text=清洗纯文本,html=HTML)
  - 表格 HTML:由 `list[list]` 拼成 `<table><tr><td>...</td></tr></table>`
  - 表格纯文本:"列名: 值"序列化(首行作列名)
- 图片提取仍走现有 PyMuPDF(fitz)逻辑,`pdf_table` 不改图片处理

### 7.5 Chunker 升级规则(9 种 type 完整策略)
- Chunker 优先读 `document.metadata["sections"]`,无则退化 `split_text(text)` 老路径
- 切分规则:

| type | 策略 |
|---|---|
| title | 并入后邻 text |
| text | 段内递归切(chunk_size/overlap,复用 SplitterFactory) |
| table | 整块保留;`section.text`(Parser 已清洗的纯文本)进 `chunk.text`,`section.html` 进 `metadata.table_html`;**Chunker 不清洗,只搬运**;超大表格 >2×chunk_size 在行边界切,切后 table_html 只挂首块 |
| figure | 不单独成 chunk;`[IMAGE:id]` 并入相邻 text 或 figure_caption |
| figure_caption | 并入关联 figure(nearest)或相邻 text |
| list | 整块保留;超 chunk_size 按 item 边界切(项目符号正则) |
| equation | 整块保留 |
| header / footer | **丢弃**(噪声) |
| 未覆盖类型 | 默认按 text 段内切 |

### 7.6 settings.yaml 配置
```yaml
ingestion:
  parser:                    # 原 loader 段
    provider: "pdf_table"    # 默认 pdf_table;可选 pdf_text | docx
    extract_images: true
  chunk_size: 1000
  chunk_overlap: 200
```

### 7.7 降级链(本项目内)
- `pdf_table.parse()`:pdfplumber 提文本 + 表格,产完整 sections
- **`degraded=true` 只在 pdfplumber 异常时**(打不开/报错)→ 委托 `pdf_text`(MarkItDown 纯文本)
- **"无表格"不算降级**:PDF 本无表格,pdf_table 正常产纯文本 sections(`extract_text`),不标 degraded
- 降级住 `pdf_table` 内部(持有 `pdf_text` 引用,组合),不经 Factory

### 7.8 WordLoader 正名 + 目录策略
- **目录策略**:**新建 `src/libs/parser/`(与 `loader/` 同级)**;`loader/` 保留 `file_integrity.py`(文件 IO 工具),现有 `base_loader/loader_factory/pdf_loader/word_loader` 搬到 `parser/` 并正名。`data_service.py` 的 `from src.libs.loader.file_integrity` 不动

| 文件 | 改动 |
|---|---|
| 新建 `src/libs/parser/` | `base_parser.py`、`parser_factory.py`、`pdf_text_parser.py`(原 pdf_loader)、`pdf_table_parser.py`(新)、`word_parser.py`(原 word_loader)、`__init__.py` |
| `src/libs/loader/__init__.py` | 去掉 BaseLoader/PdfLoader/WordLoader/LoaderFactory 导出,保留 file_integrity |
| `src/ingestion/pipeline.py` | `from src.libs.parser.parser_factory import ParserFactory`;file_integrity 保持 `from src.libs.loader.file_integrity` |
| `scripts/ingest.py` | `from src.libs.parser.parser_factory import ParserFactory`、`ParserFactory.get_supported_extensions` |
| `settings.yaml` | `loader`→`parser` |
| `tests/` | loader 测试搬 parser,import 改 |
- **影响范围**(Grep 扫出):pipeline.py、scripts/ingest.py、loader/__init__、tests;`data_service.py` 因目录分离**不用改**;`mcp_server/` 零引用(query 免疫)

### 7.9 关键决策(已确认定稿)
> 2026-07-14 确认采纳全部三条:
- ✅ A:Parser 产 `Document`(带 `metadata.sections`)而非 `List[Section]` —— 向后兼容 Transform 层
- ✅ B:Chunker 优先 sections、退化 text 老路径 —— 平滑迁移、老数据兼容
- ✅ C:降级住 provider 内部(组合 fallback)—— 减少 Factory 编排层

## 8. 实施任务分解(DEVSPEC 阶段 K)

> 对齐 DEV_SPEC 排期原则(DEV_SPEC.md:1916):1h 可验收增量,每任务带验收标准 + 测试方法 + 文件清单。接在阶段 J(J1 LoaderFactory / J2 WordLoader,已完成)之后。

### 阶段 K 进度表(对齐 DEV_SPEC.md:1944 格式)

| 任务编号 | 任务名称 | 状态 | 依赖 | 备注 |
|---|---|---|---|---|
| K1 | Loader→Parser 正名 + Factory 统一 settings 构造 | [x] | — | 全局改动,独立先做(已 commit 42b7c0d) |
| K2 | Section 数据结构 + Chunker 升级(优先 sections 退化 text) | [x] | K1 | 老路径必须 100% 回归(已 commit 9848569) |
| K3 | pdf_text provider + 别名 + _PROVIDER_EXTENSIONS | [x] | K1 | 现 PdfLoader 降格(已 commit 2301a9e) |
| K2b | Chunker table 职责修正(清洗→搬运,对齐 7.5) | [ ] | K2 | K4′ 前置;修正 K2 的 sidecar 清洗假设 |
| K4′ | pdf_table provider(pdfplumber,本项目内) | [ ] | K2b, K3 | 原生表格能力 |
| K6 | 收尾:trace + 文档债 + QA/README | [ ] | K1–K4′ | 全局清理 |

### K1 Loader→Parser 正名 + Factory 统一
- **文件**:见 7.8
- **改动**:`BaseLoader`→`BaseParser`、`load`→`parse`;`BaseParser.__init__(self, settings, collection, image_storage_dir, **kwargs)`(见 14.4);`LoaderFactory`→`ParserFactory`,统一 settings 构造(对齐 LLM/Embedding Factory,消除异类)
- **验收**:`python -m compileall src` 通过;pytest(除 `llm`)全绿;`ingest --path x.pdf`/`x.docx` 跑通
- **测试**:loader 测试搬 parser 后通过;新增 Factory settings 构造契约测试

### K2 Section 数据结构 + Chunker 升级
- **文件**:`types.py`(加 `Section` dataclass + `Chunk.metadata` 可选 `section_type`/`bbox`/`table_html`)、`document_chunker.py`
- **改动**:Chunker 优先读 `document.metadata["sections"]`,无则退化 `split_text(text)` 老路径;规则见 7.5。**table 段:Chunker 只搬运(`seg["text"]`→`chunk.text`,`seg["html"]`→`metadata.table_html`),不清洗 HTML**(见 14.1)。K2 实施时**直接按"只搬运"写**(K2 当前状态 `[ ]` 未实施,不存在老清洗逻辑要改,不产生 `_html_table_to_text` 等清洗方法)
- **验收**:无 sections 的 Document 走老路径,输出与改造前**逐字节一致**;有 sections 的表格不被切碎
- **测试**:Chunker 老路径回归(diff 为零);sections 切分单测(9 种 type 边界)

### K3 pdf_text provider + 兼容
- **文件**:`pdf_text_parser.py`、`parser_factory.py`、`settings.yaml`
- **改动**:现 PdfLoader 降格为 `PdfTextParser`(MarkItDown 逻辑不变);注册别名 `pdf`→PdfTextParser、`pdf_text`→PdfTextParser;`_PROVIDER_EXTENSIONS` 显式 `pdf`/`pdf_text`→`[".pdf"]`
- **验收**:provider=pdf / pdf_text 行为一致;文本型 PDF ingest 跑通
- **测试**:别名测试;_PROVIDER_EXTENSIONS 测试

### K2b Chunker table 职责修正(清洗→搬运)
- **背景**:K2 实现时按老 sidecar 假设,Chunker 对 table 从 HTML 清洗(`_html_table_to_text`)。新计划 7.5/14.1 改为 Parser 清洗、Chunker 只搬运。本任务对齐 K2 实现与 7.5,是 K4′ 的前置。
- **文件**:`src/ingestion/chunking/document_chunker.py`、`tests/unit/test_document_chunker_sections.py`
- **改动**:
  - `_split_table_segment`:`chunk.text = seg["text"]`(Parser 已清洗纯文本),不再 `_html_table_to_text`;超大切块从纯文本按 `\n` 行切(不再从 HTML 切)
  - 弃用并删除 `_html_table_to_text` / `_split_html_table_rows`(table 清洗职责已移交 Parser)
  - `test_table_kept_whole_with_cleaned_text_and_html`:构造改 `Section(text=纯文本, html=HTML)`,断言 `chunk.text == 纯文本`(搬运,非清洗)
- **验收**:table segment 的 `chunk.text` == `section.text`(不重新清洗);老路径回归 + 其他 sections 测试不受影响
- **测试**:改 test_table_kept_whole;新增 test 验证 Chunker 不覆盖 Parser 纯文本(传入含 HTML 标签的 section.text 时 chunk.text 原样保留)

### K4′ pdf_table provider(原生表格)
- **文件**:`pdf_table_parser.py`(新)、`parser_factory.py`(注册 pdf_table)、`pyproject.toml`(加 `pdfplumber`)
- **改动**:`PdfTableParser(BaseParser)`:pdfplumber 逐页用**带 bbox 的文本块**(`extract_words` / `extract_text(layout=True)`)+ 表格(`find_tables` / `extract_tables` 的 bbox)**按 top 坐标交替**产出 sections;表格序列化 HTML + "列名: 值" 纯文本;持有 `pdf_text` 引用(**pdfplumber 异常时**降级)
- **依赖**:加 `pdfplumber`(纯 Python,3.13 兼容)
- **验收**:文本型 PDF 的表格被提取;表格进 `metadata.table_html`,纯文本进 `chunk.text`;**pdfplumber 异常 → 降级 pdf_text + `degraded=true`;无表格 PDF 正常产纯文本 sections,不 degraded**
- **测试**:pdf_table 单测(样例 PDF 表格 → sections);**降级测试(pdfplumber 异常→pdf_text + degraded);无表格测试(正常产纯文本,不 degraded)**;_PROVIDER_EXTENSIONS 测试

### K6 收尾
- **文件**:trace、`CLAUDE.md`、`DEV_SPEC.md`、`.claude/rules/extending-backends.md`、`qa_config.py`、`README`
- **改动**:trace load stage 补 method(`pdf_table`/`pdf_text`)+ degraded;文档措辞 BaseLoader→BaseParser + chunk_id 格式 + pdf_loader 注释;qa_config 加 pdf_table/pdf_text/降级 profile;README 写表格能力说明(无 docker)
- **验收**:文档无过时术语;qa-tester 跑 pdf_table/pdf_text/降级用例通过
- **测试**:qa-tester 用例

## 9. 测试策略
- **pdf_table 单测**:用含表格的样例 PDF,验证 `extract_tables` → sections(类型/HTML/纯文本);可选 mock pdfplumber 注入边界
- **降级测试**:**pdfplumber 异常** → 降级 pdf_text + `metadata.degraded=true`;无表格 PDF 正常产纯文本 sections(不 degraded)
- **Chunker 回归**:无 sections 的 Document,改造前后输出 diff 为零
- **golden set**:不破坏现有(占位,无 PDF 条目);pdf_table 的表格检索测试用独立 fixture
- **markers**:沿用现有 unit/integration/e2e/llm;**不需要 sidecar marker**(无外部服务)

## 10. 实施顺序与提交边界
1. **K1 正名** → 独立提交,全绿(红线检查点:可插拔模式未破)
2. **K2 Chunker 升级** → 独立提交,老路径回归全绿
3. **K3 pdf_text + 兼容** → 独立提交
4. **K2b Chunker table 职责修正** → 独立提交(K4′ 前置,对齐 7.5)
5. **K4′ pdf_table** → 独立提交
6. **K6 收尾** → 独立提交

每步独立 commit,可单独回滚。**K1 正名是全局改动,必须最先且独立**。

## 11. 文档债清单
| 位置 | 现状(错) | 修正 |
|---|---|---|
| CLAUDE.md | `chunk_id = hash(source_path + section + content_hash)` | `{doc_id}_{index:04d}_{content_hash8}` |
| pdf_loader.py:46 注释 | `data/images/{doc_hash}/` | `data/images/{collection}/` |
| CLAUDE.md / DEV_SPEC / extending-backends.md | BaseLoader / LoaderFactory 措辞 | BaseParser / ParserFactory(正名后同步) |
| DEV_SPEC 进度表 | 无阶段 K | 加 K1/K2/K2b/K3/K4′/K6 任务行 |

处理时机:K1 正名时顺手清代码注释;K6 收尾统一清文档。

## 12. 配置迁移与兼容
```yaml
ingestion:
  parser:                    # 原 loader 段
    provider: "pdf_table"    # 默认(表格优先);可选 pdf_text | docx
    extract_images: true
  chunk_size: 1000
  chunk_overlap: 200
```
- **provider 别名**:`pdf`→PdfTextParser、`pdf_text`→PdfTextParser、`pdf_table`→PdfTableParser、`docx`→WordParser
- **_PROVIDER_EXTENSIONS**:`pdf`/`pdf_text`/`pdf_table`→`[".pdf"]`,`docx`→`[".docx"]`
- **默认 provider**:`pdf_table`(体现新表格能力);纯文本场景可切 `pdf_text`
- **老配置兼容**:`ingestion.loader` 段若仍存在,启动时 warning + 当作 `ingestion.parser` 处理(不 fail-fast)

## 13. QA / README / setup 衔接
- **qa_config.py**:加 profile `pdf_table`(表格提取)、`pdf_text`(纯文本)、`degradation`(**pdfplumber 异常→fallback**)
- **README**:PDF 能力说明——"`pdf_table` 提取线条表格(pdfplumber);扫描件/图片型 PDF 表格不支持(降级纯文本)"
- **setup skill**:无需 docker/模型;装 `pdfplumber` 即可
- **QA_TEST_PLAN 用例**:pdf_table(文本型 PDF 表格)、pdf_text(纯文本)、降级(**pdfplumber 异常→fallback**)、Chunker 回归(无 sections 老路径不变)

## 14. 落地细节澄清(以本节为准)

### 14.1 表格 HTML 的 embed/display 分离 + 消费端
- **决策**:`chunk.text` 存清洗纯文本(表格"列名: 值");`metadata.table_html` 存 HTML(展示)
- **清洗逻辑放 Parser**(pdf_table 从 `list[list]` 同时产 text 纯文本 + html HTML);**Chunker 只搬运**(text→`chunk.text`, html→`metadata.table_html`),不清洗;下游 DenseEncoder 直接用 `chunk.text`,无感于表格
- **table_html 消费端**:K6 加 ResponseBuilder 读 `metadata.table_html` 做展示返回(展示层消费 metadata,**不破检索红线**——不改 Dense/Sparse/RRF/Rerank)
- **超大表格**:默认整块;>2×chunk_size 在行边界切;切后 `table_html` 只挂首块(chunk_index 最小者),其余 `section_type=table` 但 `table_html=None`

### 14.2 9 种 Section 类型完整策略
见 7.5(完整 9 种 + 默认按 text 切)。

### 14.3 图片占位符(本项目内注入)
- `pdf_table`/`pdf_text` 在产出 sections 时注入 `[IMAGE:{id}]` 到 `section.text`(走现有 PyMuPDF 提图逻辑)
- `text_offset`/`text_length` 基准是 section.text;**Chunker 切分后不保证 chunk 粒度准确**
- 图片实际定位靠 `id`/`path`(ImageStorage 按 id 索引),不靠 offset;`text_offset` 为 section 粒度历史值,下游不得靠它定位 chunk 内图片

### 14.4 BaseParser 构造契约
- `BaseParser.__init__(self, settings, collection: str, image_storage_dir: str | Path, **kwargs)`
- `ParserFactory.create(settings, collection="default")` 内部 `image_storage_dir = resolve_path(f"data/images/{collection}")`,传入 provider
- 对齐 LLM/Embedding 的"传 settings"模式 + 保留 Parser 特有 `collection`/`image_storage_dir` 显式形参

### 14.5 次要点
- **list item 边界**:list 超 chunk_size 按 item 边界切,靠项目符号正则(`^\s*[-*+]\s` / `^\s*\d+\.\s`)或换行
- **degraded 透传**:`metadata.degraded=true` 标在 Document.metadata,依赖现有 `_inherit_metadata` 的 `document.metadata.copy()`(document_chunker.py:211)自动透传到 chunk.metadata;trace 读 `chunk.metadata.degraded` 即可

## 15. 运行就绪检查清单(执行完→能跑的验收)

> K1–K4′ + K6 代码完成后,按本清单验证"能正常运行"。

### 15.1 必做项(默认路径能跑)
- [ ] K1 正名影响范围全改(见 7.8:pipeline / scripts/ingest.py / loader.__init__ / tests)
- [ ] `python -m compileall src` 通过(零 import 错误)
- [ ] `pytest -m "not llm"` 全绿
- [ ] `python scripts/ingest.py --path <test.pdf> --collection test` 跑通(默认 pdf_table)
- [ ] `python scripts/query.py --query "..." --collection test` 返回结果,表格以 HTML 呈现
- [ ] `python -m src.mcp_server.server` 启动无错(query 链路免疫,仍验证)
- [ ] `python scripts/start_dashboard.py` 启动,数据浏览页可开(验证 file_integrity 链路未断)

### 15.2 数据迁移(必做)
- [ ] reindex:`--force` 重灌现有 collection,或开新 collection(`chunk_id` 变,老数据孤儿)

### 15.3 不通过的处理
- 15.1 任一失败 → K1 正名遗漏或 pdf_table 实现问题,回 Grep 扫描 / 单测定位(`mcp_server/` 不应出现 loader 引用,出现即异常)

---

**附:可讲性要点(面试)**
- **两层 Parser/Chunker(不加 Loader)**:从 RAGFlow 反推(parser-centric 无 Loader),YAGNI,敢做减法
- **pdf_table 原生**:评估文档画像(文本型 PDF)后砍掉 sidecar/模型,选 pdfplumber 纯算法,零重依赖、零外部服务
- **embed/display 分离**:借鉴 RAGFlow(`content_ltks` 去标签进 BM25),`chunk.text` 纯文本检索 + `metadata.table_html` 展示
- **query 免疫**:ingestion 与 query 只通过存储契约耦合,改良 ingestion 不动 query 一行
- **降级链**:`pdf_table` → `pdf_text`,进程内 graceful degradation,无外部服务依赖
