# 评估加速执行指南

> **一句话**:三项措施把单次 `evaluate.py` 耗时砍掉约一半(k=5 减 judge 调用 + 切换 ollama 常驻集 + 报告自动落盘)。
> 加速手段本身是纯命令;随实测顺带修的两处外围代码见 §7。

---

## 1. 背景:为什么会慢

评估的头号瓶颈是 **RAGAS 的 `context_precision`**:它对每条 QA 的**每个检索 chunk 都调一次本地 llama3 judge**,串行累积。

- 10 条 QA × top_k=10 ≈ **上百次**本地推理。
- 16G 内存下,granite(5.3G)+ llava(2.9G)若同时常驻,会挤占 llama3 推理内存 → **换页卡顿**。
- llama3 首次加载(4.7G)有**冷启动**开销。

评估链路实际只用两个 ollama 模型(已用代码核实):

| 模型 | 评估时用途 | 大小 |
|---|---|---|
| `llama3` | RAGAS judge | 4.7 GB |
| `nomic-embed-text` | HybridSearch 的 dense embedding | 0.27 GB |

另两个模型**评估时不调用**,纯占内存:

| 模型 | 不用的原因 | 大小 |
|---|---|---|
| `granite4.1:8b` | `chunk_refiner`/`metadata_enricher` 的 `use_llm: false`,对象构造了但从不 `.chat()` | 5.3 GB |
| `llava-phi3:3.8b` | 仅 ingest 的 ImageCaptioner 用 | 2.9 GB |

> **ingest 反过来**:只用 llava + nomic,不用 llama3/granite(详见 `switch-to-ingest.ps1` 注释)。

---

## 2. 三项加速措施

| # | 措施 | 怎么做 | 收益 |
|---|---|---|---|
| 1 | **k=5**(回到 spec 原定) | `evaluate.py --top-k 5` | judge 调用**砍半**(主力) |
| 2 | 卸载 granite + llava | `ollama stop ...` 或切换脚本 | 释放 ~8G,消除换页 |
| 3 | 预热 llama3 + nomic 常驻 | `ollama run ... --keepalive 30m` 或切换脚本 | 消除冷启动 |

---

## 3. 快速执行流程

### 3.1 准备(每次开终端)

```powershell
conda activate langchain-test
cd d:\Desktop\Code\RAG-MCP\MODULAR-RAG-MCP-SERVER
```

### 3.2 切到评估模式

```powershell
.\scripts\switch-to-eval.ps1
```

脚本做的事:
1. `ollama stop granite4.1:8b` + `ollama stop llava-phi3:3.8b` —— 立即卸载,不等默认 5 分钟
2. `ollama run llama3 --keepalive 30m` + `ollama run nomic-embed-text --keepalive 30m` —— 预热并设 30 分钟常驻

预期输出:
```
Switching to EVAL mode...
  unloaded: granite4.1:8b
  unloaded: llava-phi3:3.8b
  resident: llama3  (keepalive 30m)
  resident: nomic-embed-text  (keepalive 30m)

Ready to evaluate:
  python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json --collection evaluation --top-k 5 --json
```

### 3.3 跑评估

```powershell
python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json --collection evaluation --top-k 5 --json
```

> ⚠️ `--top-k 5` 是加速关键,别漏。`--collection evaluation` 必须和 ingest 时一致(BM25 一致性约束)。

**报告自动落盘(默认行为,无需任何参数)**:每次评估跑完自动追加一行到 `logs/eval_history.jsonl`(带 timestamp)。终端缓冲截断、忘加重定向都不会再丢数据——这是 dashboard Evaluation Panel 读的同一个文件,CLI 与 dashboard 共用评估历史。

| 参数 | 行为 |
|---|---|
| *(不带参数)* | 自动追加到 `logs/eval_history.jsonl`(stderr 会提示 `💾 Appended to: ...`) |
| `-o <file>` | 额外写一份**完整独立 JSON** 文件(适合命名存档,如 baseline) |
| `--no-save` | 跳过默认追加(确不需要历史时) |

存 baseline 的推荐写法:
```powershell
python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json --collection evaluation --top-k 5 --json -o logs/eval_baseline_k5.json
```

---

## 4. 验证加速效果

对比 **本次(k=5 + 切换脚本)** vs **上次(k=10,无切换)** 的总耗时。

**方法 A(推荐)**:直接看自动落盘的历史,不受终端缓冲影响:
```powershell
# 最近几次评估的总耗时
Get-Content logs\eval_history.jsonl | ForEach-Object { (ConvertFrom-Json $_).total_elapsed_ms }
```

**方法 B**:手表掐 `evaluate.py` 从启动到出结果的墙钟时间。

**实测参考**(2026-08-14,23 条 QA,6 篇论文,数据完整性修复后):k=5 下平均每条 ~169s(150-195s),全程 ~65 分钟。与 k=10 的每条 ~330s 相比,**同等条件加速约 50% 成立**;但 golden set 从 10 条扩到 23 条,绝对时间反而变长了——评估耗时同时受 top_k 和 QA 条数驱动。

> 📌 每条 QA 的耗时下限 ≈ 2 个 RAGAS 指标 × 5 chunk × llama3 单次推理(~15-20s)。想突破这个下限只能换更小 judge 或代码层 `judge_top_k`(见第 7 节)。

---

## 5. 任务切换:回到 ingest

ingest 和评估需要的常驻模型不同。跑 ingest 前切回去:

```powershell
.\scripts\switch-to-ingest.ps1
```

脚本做的事:
1. `ollama stop llama3` + `ollama stop granite4.1:8b`
2. `ollama run llava-phi3:3.8b --keepalive 30m` + `ollama run nomic-embed-text --keepalive 30m`

然后:
```powershell
python scripts/ingest.py --path tests/fixtures/eval_docs/ --collection evaluation
```

> **查询时**:需要 granite(LLM)+ nomic。查询脚本目前不在切换脚本里覆盖,查询前手动预热即可:
> ```powershell
> cmd /c "echo.|ollama run granite4.1:8b --keepalive 30m >nul 2>nul"
> ```

### 常驻模型速查表

| 任务 | 常驻 | 卸载 |
|---|---|---|
| 评估 | llama3, nomic | granite, llava |
| ingest | llava, nomic | llama3, granite |
| 查询 | granite, nomic | llama3, llava |

---

## 6. 排查与边界

### PowerShell 执行策略被拦
```
.\scripts\switch-to-eval.ps1 无法加载,因为在此系统上禁止运行脚本
```
解决(任选):
```powershell
# 方式 1:当前用户允许本地脚本(推荐,一次性)
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# 方式 2:临时绕过(每次)
powershell -ExecutionPolicy Bypass -File .\scripts\switch-to-eval.ps1
```

### `failed to get console mode for stderr: The handle is invalid`
ollama CLI 带 TUI,会调 `GetConsoleMode` 读 stderr 句柄。PowerShell 的
`2>$null` 把 stderr 重定向到非 console 句柄 → ollama 报错。
**两个坑合在一起**(都踩过):

1. **PowerShell 不支持 `<` 重定向**(cmd/bash 语法),报 `< operator is reserved`。
2. **`2>$null` 触发 ollama 的 console mode 错误**。

脚本里的统一解法是 **`cmd /c` 包一层 + cmd 自己的 `>nul 2>nul`**:
ollama 看到的仍是正常 console 句柄(不报错),而输出在 cmd 层被静默。
```powershell
cmd /c "ollama stop llama3 >nul 2>nul"
cmd /c "echo.|ollama run llama3 --keepalive 30m >nul 2>nul"
```
(`echo.` 喂一个空行给 stdin,让 `ollama run` 不进交互模式直接返回。)

手动敲单条命令(交互式)直接 `ollama run llama3 --keepalive 30m`,
看到 `>>>` 后输 `/bye`,模型留在内存。

### 确认哪些模型在内存
```powershell
ollama ps
```
评估模式下应只有 `llama3` + `nomic-embed-text`。

### `ollama stop` 对未加载的模型
无害,静默返回(脚本已用 `2>$null` 吞掉提示)。

### 加速不达预期(~40-50%)
可能原因:
1. **没切评估模式**:granite/llava 还在内存 → `ollama ps` 确认。
2. **没带 `--top-k 5`**:仍是 k=10 → 检查命令。
3. **模型冷启动没消除**:首次评估仍含加载时间,跑第二次对比更准。
4. **RAGAS 内部串行**:本方案只减 chunk 数,不并发化。若减半后还嫌慢,需上代码方案(`judge_top_k`),但那是另一个决策,不在本指南范围。

### 指标变化的诚实说明
k=5 vs k=10 会让 `context_relevance`/`context_precision` 数值略有变化(候选池变小)。`source_recall@5`/`source_precision@5` **零退化**(不依赖 judge 的 top_k)。这是"牺牲检索深度"的已知代价,符合设计取舍。

### ingest 时 `std::bad_alloc`(静默缺页)——已由 D-027 分批转换根治
**历史问题**:docling 预处理阶段每 converter 累积内存,~9 页后 `std::bad_alloc`,且 pipeline 照样报 ✅——35 页论文曾静默丢 25 页(每页仅 933 字符 vs 正常 4000+)。

**已修复**(D-027):DoclingParser 现按 8 页/批分批转换,**每批新建 converter** 重置内存累积;后批失败保留前批内容。长文档不再需要"一篇一进程"绕行。

**仍要留意**:若日志再现 `std::bad_alloc`(极端内存压力下批内也可能失败),判别法仍是 `Text length ÷ 页数 < 1000`;此时重灌单文件即可(幂等,BM25 按 doc_id 先删后加):
```powershell
python scripts/ingest.py --path "tests/fixtures/eval_docs/<受影响文件>.pdf" --collection evaluation --force
```

---

## 7. 本指南不做的事(边界演进记录)

加速方案本身保持**不改检索/评估逻辑**:
- ❌ 不换 judge 模型(保持 llama3)
- ❌ 不加缓存、不做并发
- ❌ 不动 `RagasEvaluator` 的指标计算

但随实测暴露的问题,本分支已顺带落地两处**外围**代码改动(均不影响指标语义):

| 改动 | 动机 | 详见 |
|---|---|---|
| `evaluate.py` 默认落盘 `logs/eval_history.jsonl`(`-o` / `--no-save`) | 23 条 QA 的 JSON 报告被终端缓冲截断,丢了 aggregate 头部 | §3.3 |
| DoclingParser 分批转换(D-027,批 8 页 × 每批新 converter) | docling ~9 页后 `std::bad_alloc` 静默丢正文,数据不完整则评估无意义 | §6 |

如果纯命令方案验证后**仍不够快**,再考虑代码层的 `judge_top_k`(检索取 N、judge 只判前 M),那需要另起一次 spec → plan → 实现。
