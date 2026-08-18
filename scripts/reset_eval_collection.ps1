<#
.SYNOPSIS
  Reset (delete) ONE RAG collection's data from all stores — safely.

.DESCRIPTION
  Wipes a single collection from Chroma + BM25 + ImageStorage + SHA256 history.
  Use this AFTER changing the BM25 tokenizer (or any index-side transform), so
  the next `ingest.py` rebuilds the collection with the new logic instead of
  silently skipping unchanged files (FileIntegrity matches by SHA256).

  What gets deleted (collection-scoped only — other collections are untouched):
    1. Chroma collection         — via API delete + recreate (NOT by UUID dir)
    2. BM25 index dir            — data/db/bm25/{Collection}/
    3. ImageStorage rows         — image_index.db WHERE collection = ?
    4. SHA256 history rows (★)   — ingestion_history.db WHERE collection = ?
    5. (optional) image files    — data/images/{Collection}/  (-AlsoClearImages)

  ★ Step 4 is the critical one: without it, re-ingest sees the same SHA256 and
    SKIPS every file, so old docs are never rebuilt with the new tokenizer.

  Default mode is DRY RUN — nothing is deleted. Pass -Execute to actually reset.

  IMPORTANT: run inside `conda activate langchain-test` — this script needs
  chromadb + the project's src/ on path.

.PARAMETER Collection
  Collection name to reset (default: evaluation).

.PARAMETER Execute
  Actually perform the deletion. Without this flag the script only previews.

.PARAMETER AlsoClearImages
  Also delete data/images/{Collection}/ files. Default keeps them (harmless,
  will be regenerated on re-ingest); pass this to reclaim disk space.

.PARAMETER SkipConfirm
  Skip the interactive confirmation. Use with care.

.EXAMPLE
  # preview what would be deleted (no changes)
  .\scripts\reset_eval_collection.ps1

.EXAMPLE
  # actually reset the evaluation collection
  .\scripts\reset_eval_collection.ps1 -Execute

.EXAMPLE
  # reset a different collection and also wipe its image files
  .\scripts\reset_eval_collection.ps1 -Collection default -Execute -AlsoClearImages
#>

[CmdletBinding()]
param(
    [string]$Collection = "evaluation",
    [switch]$Execute,
    [switch]$AlsoClearImages,
    [switch]$SkipConfirm
)

$ErrorActionPreference = "Stop"

# --- locate repo root (script lives in scripts/, repo root is its parent) ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: 'python' not found on PATH. Activate the project env first:" -ForegroundColor Red
    Write-Host "  conda activate langchain-test" -ForegroundColor Red
    exit 2
}

# --- run a Python snippet (passed via stdin) using the repo as cwd ---
function Invoke-Py {
    param([string]$Code)
    $Code | python -
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: python exited with code $LASTEXITCODE." -ForegroundColor Red
        Write-Host "Make sure you are in the 'langchain-test' conda env (needs chromadb)." -ForegroundColor Red
        exit 1
    }
}

# Resolve the AlsoClearImages switch in a PS 5.1-compatible way.
$alsoImg = "0"
if ($AlsoClearImages) { $alsoImg = "1" }

# --- print the current state of all stores for the target collection ---
function Show-State {
    param([string]$Label)
    Write-Host ""
    Write-Host "=== $Label  (collection = $Collection) ===" -ForegroundColor Cyan
    $py = @"
import os, sqlite3, sys
repo = r"$RepoRoot"
sys.path.insert(0, repo); os.chdir(repo)
coll = "$Collection"

from pathlib import Path
from src.core.settings import load_settings
s = load_settings()
pdir = os.path.abspath(s.vector_store.persist_directory)   # .../data/db/chroma
dbdir = Path(pdir).parent                                  # .../data/db

# 1. chroma — use the project-standard ChromaSettings (mirrors reset_all in
#    data_service.py) so the read and reset clients are configured identically.
try:
    from chromadb.config import Settings as ChromaSettings
    import chromadb
    client = chromadb.PersistentClient(
        path=pdir,
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )
    names = {x.name for x in client.list_collections()}
    if coll in names:
        n = client.get_collection(coll).count()
    else:
        n = 0
    print(f"  chroma     : {n} items")
except Exception as e:
    print(f"  chroma     : ERR {e}")

# 2. bm25 index file
bm25 = dbdir / "bm25" / coll / f"{coll}_bm25.json"
print(f"  bm25 index : {'present' if bm25.exists() else 'absent '}  ({bm25})")

# 3. image_index.db rows
idb = dbdir / "image_index.db"
conn = sqlite3.connect(str(idb))
n = conn.execute("SELECT COUNT(*) FROM image_index WHERE collection=?", (coll,)).fetchone()[0]
conn.close()
print(f"  image rows : {n} in image_index.db")

# 4. ingestion_history.db rows  (the SHA256 dedup table)
hdb = dbdir / "ingestion_history.db"
conn = sqlite3.connect(str(hdb))
n = conn.execute("SELECT COUNT(*) FROM ingestion_history WHERE collection=?", (coll,)).fetchone()[0]
conn.close()
print(f"  sha256 rows: {n} in ingestion_history.db")

# 5. image files dir
idir = dbdir.parent / "images" / coll
print(f"  image dir  : {'present' if idir.exists() else 'absent '}  ({idir})")
"@
    Invoke-Py -Code $py
}

# --- perform the actual reset (only called under -Execute) ---
function Invoke-Reset {
    $py = @"
import os, shutil, sqlite3, sys
from pathlib import Path
repo = r"$RepoRoot"
sys.path.insert(0, repo); os.chdir(repo)
coll = "$Collection"
also_img = "$alsoImg" == "1"

from src.core.settings import load_settings
s = load_settings()
pdir = os.path.abspath(s.vector_store.persist_directory)
dbdir = Path(pdir).parent

# 1. Chroma — API only; never touch the UUID segment dirs by hand.
# ONE PersistentClient with the project-standard ChromaSettings (mirrors
# reset_all in data_service.py). A SECOND client with different settings on
# the same path trips chromadb's SharedSystemClient singleton guard:
#   "An instance of Chroma already exists ... with different settings"
from chromadb.config import Settings as ChromaSettings
import chromadb
client = chromadb.PersistentClient(
    path=pdir,
    settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
)
names = {x.name for x in client.list_collections()}
if coll in names:
    client.delete_collection(coll)
    print("[ok]   chroma collection deleted")
else:
    print(f"[skip] chroma collection '{coll}' absent (already clean)")

# 2. BM25 index dir (one JSON file per collection, no cross-collection state)
bm25_dir = dbdir / "bm25" / coll
if bm25_dir.exists():
    shutil.rmtree(str(bm25_dir))
    print(f"[ok]   removed bm25 dir  {bm25_dir}")
else:
    print(f"[skip] bm25 dir absent   {bm25_dir}")

# 3. ImageStorage rows (scoped by collection — do NOT drop the whole db)
idb = dbdir / "image_index.db"
conn = sqlite3.connect(str(idb))
before = conn.execute("SELECT COUNT(*) FROM image_index WHERE collection=?", (coll,)).fetchone()[0]
conn.execute("DELETE FROM image_index WHERE collection=?", (coll,))
conn.commit(); conn.close()
print(f"[ok]   image_index rows deleted: {before}")

# 4. SHA256 history (★ critical — prevents silent re-ingest skip)
hdb = dbdir / "ingestion_history.db"
conn = sqlite3.connect(str(hdb))
before = conn.execute("SELECT COUNT(*) FROM ingestion_history WHERE collection=?", (coll,)).fetchone()[0]
conn.execute("DELETE FROM ingestion_history WHERE collection=?", (coll,))
conn.commit(); conn.close()
print(f"[ok]   ingestion_history rows deleted: {before}  (prevents SHA256 silent-skip)")

# 5. (optional) image files
idir = dbdir.parent / "images" / coll
if also_img:
    if idir.exists():
        shutil.rmtree(str(idir))
        print(f"[ok]   removed image dir {idir}")
    else:
        print(f"[skip] image dir absent  {idir}")
else:
    print(f"[note] image files kept under {idir} (pass -AlsoClearImages to remove)")

print("done")
"@
    Invoke-Py -Code $py
}

# ── main flow ───────────────────────────────────────────────────────

Show-State -Label "BEFORE"

if (-not $Execute) {
    Write-Host ""
    Write-Host ">>> DRY RUN — nothing was changed." -ForegroundColor Yellow
    Write-Host ">>> Re-run with -Execute to actually reset '$Collection'." -ForegroundColor Yellow
    Write-Host ">>> Next steps after reset:" -ForegroundColor DarkGray
    Write-Host "      1. (optional) add more PDFs to tests/fixtures/eval_docs/" -ForegroundColor DarkGray
    Write-Host "      2. python scripts/ingest.py --path tests/fixtures/eval_docs/ --collection $Collection" -ForegroundColor DarkGray
    Write-Host "      3. python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json --collection $Collection --json" -ForegroundColor DarkGray
    return
}

if (-not $SkipConfirm) {
    Write-Host ""
    Write-Host "This will PERMANENTLY delete collection '$Collection' from all stores." -ForegroundColor Yellow
    $answer = Read-Host "Type the collection name ('$Collection') to confirm"
    if ($answer -cne $Collection) {
        Write-Host "Confirmation mismatch ('$answer' != '$Collection'). Aborted." -ForegroundColor Red
        return
    }
}

Write-Host ""
Write-Host "Resetting collection '$Collection'..." -ForegroundColor Yellow
Invoke-Reset

Show-State -Label "AFTER"

Write-Host ""
Write-Host "Done. Re-ingest + re-evaluate with:" -ForegroundColor Green
Write-Host "  python scripts/ingest.py --path tests/fixtures/eval_docs/ --collection $Collection" -ForegroundColor Green
Write-Host "  python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json --collection $Collection --json" -ForegroundColor Green
