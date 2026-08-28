<#
.SYNOPSIS
  Switch ollama's resident models to the INGEST set (qwen2.5vl + nomic-embed-text).

.DESCRIPTION
  Ingest needs only:
    - qwen2.5vl-3b:latest  (Vision — ImageCaptioner; only when vision_llm.enabled)
    - nomic-embed-text (dense embedding for the DenseEncoder)
  granite4.1:8b is constructed by chunk_refiner / metadata_enricher but NEVER
  called (both have use_llm=false — rule-based fallback runs instead), and
  llama3 is the eval judge. Neither belongs in RAM during ingest.

  This script:
    1. Unloads granite (free RAM for qwen2.5vl's image captioning).
    2. Pre-warms qwen2.5vl + nomic with a 30m keep-alive.

  Run this BEFORE `python scripts/ingest.py ...`. Non-destructive.

  NOTE: vision captioning is the heavier ingest cost. If you're doing a
  text-only ingest and want to skip captioning entirely, set
  `vision_llm.enabled: false` in config/settings.yaml instead of running this.

.EXAMPLE
  .\scripts\switch-to-ingest.ps1
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: 'ollama' not found on PATH. Is Ollama installed and running?" -ForegroundColor Red
    exit 2
}

Write-Host "Switching to INGEST mode..." -ForegroundColor Cyan

# NOTE on the cmd /c wrapper: ollama's CLI renders a TUI and calls
# GetConsoleMode on its stderr handle. PowerShell's `2>$null` redirects that
# handle to a non-console sink, triggering "failed to get console mode for
# stderr: The handle is invalid". Routing through cmd /c with cmd's own
# `>nul 2>nul` keeps a valid console handle from ollama's perspective, so the
# error disappears while output is still silenced.

# 1. Unload models ingest doesn't use.
foreach ($m in @("granite4.1:8b")) {
    cmd /c "ollama stop $m >nul 2>nul"
    Write-Host "  unloaded: $m" -ForegroundColor DarkGray
}

# 2. Pre-warm + keep resident. Feeding an empty line to `ollama run` via the
#    cmd pipe makes it return immediately (no interactive prompt) while leaving
#    the model loaded in RAM.
foreach ($m in @("qwen2.5vl-3b:latest", "nomic-embed-text")) {
    cmd /c "echo.|ollama run $m --keepalive 30m >nul 2>nul"
    Write-Host "  resident: $m  (keepalive 30m)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Ready to ingest:" -ForegroundColor Green
Write-Host "  python scripts/ingest.py --path tests/fixtures/eval_docs/ --collection evaluation" -ForegroundColor Green
