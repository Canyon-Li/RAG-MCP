"""One-shot probe: drive the real MCP server over stdio and dump the raw
``content`` block structure returned by query_knowledge_hub.

Purpose: verify whether ``ResponseBuilder.to_mcp_content()`` actually emits a
second "References (JSON)" TextContent block, i.e. how many blocks the client
receives and what each contains.

Run (conda env):
    conda run -n langchain-test python scripts/_probe_citation_blocks.py
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Windows console: force UTF-8 so Chinese / box chars don't crash
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
        env=env,
    )


def send(proc: subprocess.Popen, messages: list, expected: int, timeout: float = 60.0):
    assert proc.stdin and proc.stdout
    for m in messages:
        proc.stdin.write(json.dumps(m) + "\n")
        proc.stdin.flush()

    responses: list[dict] = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            line = proc.stdout.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue
            try:
                d = json.loads(s)
            except json.JSONDecodeError:
                continue
            if "id" in d and ("result" in d or "error" in d):
                responses.append(d)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    deadline = time.time() + timeout
    while len(responses) < expected and time.time() < deadline:
        time.sleep(0.1)
    stop.set()
    return responses


def main() -> int:
    query = "SAT solver"
    init = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "clientInfo": {"name": "probe", "version": "1.0"},
            "capabilities": {},
        },
    }
    initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    call = {
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {
            "name": "query_knowledge_hub",
            "arguments": {"query": query, "top_k": 5},
        },
    }

    proc = start_server()
    stderr_text = ""
    try:
        responses = send(proc, [init, initialized, call], expected=2, timeout=90.0)
    finally:
        try:
            # drain stderr before terminating
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        finally:
            try:
                stderr_text = proc.stderr.read() if proc.stderr else ""
            except Exception:
                stderr_text = ""

    call_resp = next((r for r in responses if r.get("id") == 2), None)

    if call_resp is None:
        print("!! 没有收到 tools/call 响应")
        print("=== stderr (前 2000 字) ===")
        print(stderr_text[:2000])
        return 1

    if "error" in call_resp:
        print("!! 收到 error 响应:")
        print(json.dumps(call_resp["error"], ensure_ascii=False, indent=2))
        return 0

    result = call_resp["result"]
    content = result.get("content", [])

    print("=" * 60)
    print(f"query        = {query!r}")
    print(f"isError      = {result.get('isError')}")
    print(f"content blocks 数量 = {len(content)}")
    print("=" * 60)

    for i, block in enumerate(content):
        btype = block.get("type")
        text = block.get("text", "") or ""
        print(f"\n--- block[{i}]  type={btype}  text_len={len(text)} ---")
        markers = []
        if "## 检索结果" in text:
            markers.append("Markdown主体(检索结果)")
        if "## 引用来源" in text:
            markers.append("引用来源列表")
        if "References (JSON)" in text:
            markers.append("References-JSON-block")
        if "```json" in text:
            markers.append("json代码块")
        if "[1]" in text:
            markers.append("含[1]标记")
        print(f"    识别标记: {markers or '(无)'}")
        # 打印前 500 字预览
        preview = text[:500]
        print("    预览:")
        for line in preview.splitlines():
            print(f"      | {line}")
        if len(text) > 500:
            print(f"      | ...({len(text) - 500} 字省略)")

    # 结论
    has_md = any("## 检索结果" in (b.get("text", "") or "") for b in content)
    has_json = any("References (JSON)" in (b.get("text", "") or "") for b in content)
    print("\n" + "=" * 60)
    print("结论:")
    print(f"  - Markdown 主体 block : {'✅ 存在' if has_md else '❌ 缺失'}")
    print(f"  - References JSON block: {'✅ 存在' if has_json else '❌ 缺失'}")
    print(f"  - 总 block 数: {len(content)}")
    print("=" * 60)

    # stderr 里如果有 WARNING/ERROR 也提示一下
    for line in stderr_text.splitlines():
        if any(k in line for k in ("ERROR", "Traceback", "WARNING")):
            print(f"[stderr] {line.strip()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
