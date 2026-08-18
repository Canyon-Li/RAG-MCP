# Evaluation Documents

这个目录存放用于 RAG 检索 + 溯源评估的**英文论文 PDF**。论文本身因版权原因不入 git(见仓库根 `.gitignore` 规则 `tests/fixtures/eval_docs/`),这里只记录论文清单,以便复现评估时知道需要补齐哪些文件。

## 评估用的论文

评估 golden set([`../golden_test_set.json`](../golden_test_set.json))的 `expected_sources` 引用了以下文件名。复现评估时,请自行获取同名 PDF 放到本目录:

| 文件名 | 论文 | 来源 |
|---|---|---|
| `New-record-in-the-number-of-qubits-for-a-quantum-implementation-of-AES.pdf` | *New record in the number of qubits for a quantum implementation of AES* (2023) | Frontiers in Physics (OPEN ACCESS) |
| `Novel-quantum-circuit-implementation-of-Advanced-Encryption-Standard-with-low-costs.pdf` | *Novel quantum circuit implementation of Advanced Encryption Standard with low costs* (2022) | Science China |
| `Optimized-Quantum-Circuit-of-AES-with-Interlacing-Uncompute-Structure.pdf` | *Optimized Quantum Circuit of AES With Interlacing-Uncompute Structure* (2024) | IEEE Transactions on Computers |
| `Optimized-quantum-implementation-of-AES.pdf` | *Optimized quantum implementation of AES* (2023) | Science China (Springer) |
| `Optimizing-the-Depth-of-Quantum-Implementations-of-Linear-Layers.pdf` | *Optimizing the depth of quantum implementations of linear layers* (2024) | Springer (会议论文) |
| `Quantum-circuit-implementations-of-SM4-block-cipher-optimizing-the-number-of-qubits.pdf` | *Quantum circuit implementations of SM4 block cipher optimizing the number of qubits* (2024) | Springer (OPEN ACCESS) |

> 评估设计详见 [`docs/superpowers/specs/2026-08-13-retrieval-evaluation-design.md`](../../docs/superpowers/specs/2026-08-13-retrieval-evaluation-design.md)。
