# Embedding 中文选型 A/B 报告（D-030）

> 日期：2026-08-22 ｜ 决策记录：DEV_CHANGELOG D-030 ｜ 结论：**默认切 bge-m3**

## 1. 动机

D-014 定的 nomic-embed-text 是英文向模型（768 维）；项目叙事已转向**中文研报问答场景**，稠密向量质量是检索链路的第一变量。选型主张"中文向模型在中文语料上更优"必须在**中文语料**上验证——在英文论文语料上测中文模型，实验设计与主张脱节。

## 2. 实验设计（受控克隆法）

三份 collection 的 **chunk 文本、元数据、BM25 稀疏索引完全一致，唯一变量是稠密向量模型**：

1. `zh_eval`（272 chunks，nomic 摄入）：41 页中信证券研报（`--force` 补采，102 chunks + 54 图 Vision 描述）+ chinese_long_doc（124）+ chinese_technical_doc（29）+ chinese_table_chart_doc（17）；
2. `zh_eval_bge_m3`：`scripts/reembed_collection.py` 从 zh_eval 读全量 chunk → bge-m3 重嵌 → 新 collection + BM25 索引复制；
3. `zh_eval_qwen3e`：同法，qwen3-embedding:0.6b。

评测集：`tests/fixtures/golden_test_set_zh.json`（zh-v1.0，**15 题中文自然问句**，GT 事实全部从文档原文提取核实；8 题研报域 + 7 题技术域）。指标：确定性 source_recall@5 / source_precision@5（零 LLM，D-028 体系）。查询侧与文档侧模型一致（每轮评估前翻转 settings.yaml）。

## 3. 结果

| 模型 | 维度 | source_recall@5 | source_precision@5 | 嵌入吞吐（chunks/s，同机） |
|---|---|---|---|---|
| nomic-embed-text（原默认） | 768 | 1.000 | 0.6933 | — |
| **bge-m3（新默认）** | 1024 | 1.000 | **0.7867** | 2.6 |
| qwen3-embedding:0.6b | 1024 | 1.000 | 0.7733 | 1.8 |

- **precision@5 是本轮判别指标**：4 文档小语料下 recall 三方全 1.0（天花板效应，失去判别力——如实记录，语料扩充后 recall 恢复判别力）。
- bge-m3 vs nomic：**+9.3pt**；vs qwen3-embedding:0.6b：+1.3pt 且嵌入快 44%（2.6 vs 1.8 chunks/s）。

## 4. 局限（如实）

1. 语料仅 4 文档 / 272 chunks，规模小；结论方向可信、幅度待更大盘语料复核；
2. context_* 语义指标（D-028 五指标体系中的三个 judge 指标）本轮**未产出**——会话未配置 `RAGAS_JUDGE_PROVIDER=deepseek` + `DEEPSEEK_API_KEY`；三个 collection 已保留，配好 env 可补跑；
3. 单机单卡（Ollama），非并发压测。

## 5. 复现

```powershell
# 语料（研报需 --force，见 §6）
python scripts/ingest.py --path "<研报.pdf>" --collection zh_eval --force
python scripts/ingest.py --path tests/fixtures/sample_documents/chinese_long_doc.pdf --collection zh_eval
python scripts/ingest.py --path tests/fixtures/sample_documents/chinese_technical_doc.pdf --collection zh_eval
python scripts/ingest.py --path tests/fixtures/sample_documents/chinese_table_chart_doc.pdf --collection zh_eval
# 变体克隆
python scripts/reembed_collection.py --source zh_eval --target zh_eval_bge_m3 --model bge-m3 --dimensions 1024
python scripts/reembed_collection.py --source zh_eval --target zh_eval_qwen3e --model qwen3-embedding:0.6b --dimensions 1024
# 评估（每轮先改 settings.yaml 的 embedding.model/dimensions 保持查询/文档两侧一致）
python scripts/evaluate.py --test-set tests/fixtures/golden_test_set_zh.json --collection zh_eval --output logs/eval_zh_nomic.json
python scripts/evaluate.py --test-set tests/fixtures/golden_test_set_zh.json --collection zh_eval_bge_m3 --output logs/eval_zh_bgem3.json
python scripts/evaluate.py --test-set tests/fixtures/golden_test_set_zh.json --collection zh_eval_qwen3e --output logs/eval_zh_qwen3e.json
```

结果存档：`logs/eval_zh_nomic.json` / `logs/eval_zh_bgem3.json` / `logs/eval_zh_qwen3e.json` + `logs/eval_history.jsonl`。

## 6. 顺带实锤的工程问题

`SQLiteIntegrityChecker.should_skip` 按 file_hash **全局判重、不看 collection 作用域**：研报曾入其他 collection（success 记录），向 zh_eval 摄取被整体跳过（本次 `--force` 绕过）。多库场景（公有库+个人库）下"A 库采过的文档进不了 B 库"是缺陷——业务指纹方案（enterprise-rag-design §7）按 (业务键, collection) 判重才能根治。
