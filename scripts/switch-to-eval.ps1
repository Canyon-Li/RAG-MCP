<#
.SYNOPSIS
  Switch ollama's resident models to the EVAL set (nomic-embed-text only).

.DESCRIPTION
  Evaluation needs only:
    - nomic-embed-text (dense embedding for HybridSearch)
  The RAGAS judge is cloud DeepSeek (env-frozen, D-028 / T10 gauge
  calibration), so NO local LLM is needed. The other installed models are
  NOT used by evaluate.py and just tie up RAM:
    - granite4.1:8b   (LLM — never called during eval; chunk_refiner /
                       metadata_enricher have use_llm=false)
    - qwen2.5vl-3b:latest (Vision — only used by ingest's ImageCaptioner)

  This script:
    1. Immediately unloads granite + qwen2.5vl (don't wait for the 5m default).
    2. Pre-warms nomic with a 30m keep-alive so the first query doesn't pay
       the multi-second model-load cost and it stays resident across the
       whole eval run.

  Run this BEFORE `python scripts/evaluate.py ...`. Non-destructive: it only
  manages which models are in RAM, no data is touched.

.EXAMPLE
  .\scripts\switch-to-eval.ps1
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: 'ollama' not found on PATH. Is Ollama installed and running?" -ForegroundColor Red
    exit 2
}

Write-Host "Switching to EVAL mode..." -ForegroundColor Cyan

# NOTE on the cmd /c wrapper: ollama's CLI renders a TUI and calls
# GetConsoleMode on its stderr handle. PowerShell's `2>$null` redirects that
# handle to a non-console sink, triggering "failed to get console mode for
# stderr: The handle is invalid". Routing through cmd /c with cmd's own
# `>nul 2>nul` keeps a valid console handle from ollama's perspective, so the
# error disappears while output is still silenced.

# 1. Unload models eval doesn't use (free RAM for the eval loop).
foreach ($m in @("granite4.1:8b", "qwen2.5vl-3b:latest")) {
    cmd /c "ollama stop $m >nul 2>nul"
    Write-Host "  unloaded: $m" -ForegroundColor DarkGray
}

# 2. Pre-warm + keep resident. Feeding an empty line to `ollama run` via the
#    cmd pipe makes it return immediately (no interactive prompt) while leaving
#    the model loaded in RAM.
foreach ($m in @("nomic-embed-text")) {
    cmd /c "echo.|ollama run $m --keepalive 30m >nul 2>nul"
    Write-Host "  resident: $m  (keepalive 30m)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Ready to evaluate:" -ForegroundColor Green
Write-Host "  python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json --collection evaluation --top-k 5 --json" -ForegroundColor Green
