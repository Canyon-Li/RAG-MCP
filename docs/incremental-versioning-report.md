# 研报版本管理 E2E 实测报告（D-031）

> 日期：2026-08-22 ｜ 决策记录：DEV_CHANGELOG D-031 ｜ 硬件：GTX 1660 Ti (6GB) / Ollama
> 栈：docling 解析 + bge-m3 嵌入 + qwen3-vl:4b 图表描述 ｜ 工作库：`zh_eval_bge_m3`

## 1. 实验设计

验证三层版本管理在**真实研报**上的闭环（不造数，全部 trace/log 实测）：

- **语料**：41 页中信证券研报（润泽科技 300442，102 chunks + 54 图）
- **场景**：`scripts/make_watermarked_copy.py` 制造平台重分发副本——首页打水印行 + `garbage=4, deflate` 重编码（模拟慧博式分发：**文件字节完全不同**，SHA256 `5aafa501…` → `b07ed6eb…`）
- **步骤 A**：原版摄取入库（建立 history + business_key，embedding 缓存已由前序冷跑填充）
- **步骤 B**：水印副本摄取（应触发：指纹一致 → supersede 旧版 → 双库清理 → 新 chunk 入库）

## 2. 结果一：版本识别与 supersede 级联（✅ 全链路正确）

| 检查项 | 结果 |
|---|---|
| 指纹抽取（真实研报首页） | business_key=`1a888f86a40813f1`，confidence=**high**，broker=中信证券，date=2023-06-12 |
| 水印副本指纹 | **与原版完全一致**（字节不同被正确识别为同稿新版本） |
| supersede 触发 | 日志实锤：`↪ Superseded old version 5aafa50184e7… (purged 102 vectors, same business_key)` |
| Chroma 终态 | 272 chunks **零重复**（研报 102 条归属新副本路径，旧路径 0 残留） |
| BM25 终态 | 旧前缀 `1b885809` 清除，新前缀 `75b0b805` ×102（前缀修复生效） |
| history 终态 | 1 条 superseded + 1 条 active（同 business_key） |
| 检索 sanity | "目标价和评级" top 命中水印版评级表（买入 / 46.00 元），引用回链正确 |

同时修复并验证的两个一致性缺陷：**孤儿向量**（内容变 → 新 chunk_id → 旧向量残留，现为 upsert 前按 source_path purge）；**BM25 清理失效**（旧代码按 `doc_` 前缀清理，与 postings 键 `{source_hash8}_` 永不匹配，现为真实 source 前缀）。

## 3. 结果二：增量收益（同字节重摄取，步骤 1 → A）

| 阶段 | 冷（缓存空） | 热（缓存满） | 说明 |
|---|---|---|---|
| load（docling 解析） | 400.2s | 235.3s | 全量解析是本栈地板（占热总耗时 95%） |
| transform | 1.4s | 1.0s | 图片 caption 命中 ImageStorage 已存记录 |
| **embed** | **88.1s** | **2.2s（-97.5%）** | **102/102 缓存命中，零嵌入调用** |
| upsert | ~5s | 5.3s | |
| **总耗时** | **8.2 min** | **4.15 min** | |

## 4. 结果三：重编码副本的增量失效根因（步骤 B，61.2 min——诚实记录）

水印副本（字节+路径都变）触发 supersede 正确，但总耗时 61.2 min，分解暴露两项**非确定性**：

1. **transform 3277.7s（54.6 min）**：水印副本的图片 ID 内嵌新文件哈希 → ImageStorage 无记录；caption 持久缓存（键=图片字节哈希）**也 miss**——docling 对重编码 PDF 的 bbox 渲染**不是字节确定的** → 54 图全部走 qwen3-vl:4b 重描述（~60s/图 @1660Ti）。
2. **embed 123.8s（~50/102 miss）**：重编码使 docling 文本抽取发生细微漂移 → chunk 边界漂移（前置章节文本微变 → 后续 chunk 哈希连锁变化）——此前理论预判的坑，现在有测量数据。

**后续方向（v2）**：① caption 缓存键升级为感知哈希（对渲染非确定鲁棒）；② **chunk-manifest 短路**：history 存业务键→chunk 哈希集合清单，同键且集合相同则免解析整库跳过（水印场景将降至秒级判重）。

## 5. 副产品：Vision 换型质量对比（同一张研报图表）

| 模型 | caption（原文摘录） |
|---|---|
| llava-phi3:3.8b（旧） | *"The image presents a line graph that illustrates the trend of two variables over four distinct years…"* ——英文、泛化、未识别业务含义，**中文查询无法命中** |
| **qwen3-vl:4b（新默认）** | *"2019-2022年营业收入持续增长，2022年达峰值；同比增长率（yoy）在2021年最高后有所回落"* ——中文、读出图表业务语义 |

## 6. 复现

```powershell
python scripts/make_watermarked_copy.py "<原版.pdf>" "<水印副本.pdf>"
python scripts/ingest.py --path "<原版.pdf>" --collection zh_eval_bge_m3        # 步骤 A
python scripts/ingest.py --path "<水印副本.pdf>" --collection zh_eval_bge_m3    # 步骤 B：supersede
```

trace 存档：`logs/traces.jsonl`（stage=fingerprint 含 business_key/superseded 明细；stage=embed 含 embedding_cache 命中统计）。

## 7. 可对外表述（实测摘录）

> 设计研报版本管理：从首页法定披露要素（分析师执业编号/证券代码/日期，版面无关）抽取业务主键，实测水印重分发副本（字节不同）被正确识别为同稿新版本并自动 supersede——旧版向量与 BM25 索引级联清理、终态零重复；embedding 按内容哈希持久缓存，同字节重摄取嵌入阶段 88s→2.2s（-97.5%）；实测定位两项增量失效根因（解析渲染非确定、chunk 边界漂移），据此提出感知哈希缓存键与 chunk-manifest 短路两个优化方向。
