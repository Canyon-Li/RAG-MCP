"""FastAPI application for the research-report knowledge service (D-032).

Endpoints:
    GET  /api/health              liveness + component summary
    GET  /api/users               demo user list (login picker for the chat page)
    GET  /api/libraries           visible libraries for the requesting user
    GET  /api/documents           ingested documents across visible libraries
    POST /api/query               multi-library RAG Q&A with citations
    POST /api/ingest              upload a PDF into a library (async task)
    GET  /api/tasks/{task_id}     poll an ingest task

Auth is intentionally minimal for the MVP: the caller identifies itself with
the ``X-User-Id`` header, and EVERY permission decision (visible libraries,
public-library write rights) is resolved server-side from that identity —
the design rule being exercised is "server-side ACL injection", not the auth
mechanism itself. Production deployments would front this with SSO.

Run:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8300
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from src.api import store
from src.api.service import RagService
from src.core.settings import resolve_path
from src.observability.logger import get_logger

logger = get_logger(__name__)

try:  # evaluate.py precedent: .env holds ZHIPUAI_API_KEY (gitignored)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

app = FastAPI(
    title="Research Report Knowledge Service",
    description="Multi-library RAG Q&A over broker research reports (D-032).",
    version="0.1.0",
)

UPLOAD_DIR = resolve_path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}

_service: Optional[RagService] = None


def get_service() -> RagService:
    global _service
    if _service is None:
        _service = RagService()
    return _service


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    libraries: Optional[List[str]] = None
    use_rerank: bool = True
    use_generate: bool = True


def _require_user(x_user_id: Optional[str]) -> str:
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-Id header required")
    user = store.get_user(x_user_id)
    if user is None:
        raise HTTPException(status_code=403, detail=f"unknown user {x_user_id}")
    return x_user_id


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "research-report-knowledge",
        "embedding": get_service().settings.embedding.model,
        "generation_enabled": bool(
            getattr(get_service().settings, "generation", None)
            and get_service().settings.generation.enabled
        ),
    }


@app.get("/api/users")
def users() -> dict:
    return {"users": store.list_users()}


@app.get("/api/libraries")
def libraries(x_user_id: Optional[str] = Header(None)) -> dict:
    user_id = _require_user(x_user_id)
    return {"libraries": get_service().list_libraries(user_id)}


@app.get("/api/documents")
def documents(x_user_id: Optional[str] = Header(None)) -> dict:
    user_id = _require_user(x_user_id)
    return {"documents": get_service().list_documents(user_id)}


@app.post("/api/query")
async def query(
    req: QueryRequest,
    x_user_id: Optional[str] = Header(None),
) -> dict:
    user_id = _require_user(x_user_id)
    # Blocking call (retrieval is fast; generation adds seconds). The
    # threadpool keeps the event loop free for concurrent requests.
    return await run_in_threadpool(
        get_service().answer,
        user_id=user_id,
        query=req.query,
        top_k=req.top_k,
        libraries=req.libraries,
        use_rerank=req.use_rerank,
        use_generate=req.use_generate,
    )


@app.post("/api/ingest")
async def ingest(
    background: BackgroundTasks,
    file: UploadFile,
    x_user_id: Optional[str] = Header(None),
    library: Optional[str] = None,
) -> dict:
    user_id = _require_user(x_user_id)

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"unsupported file type {suffix}")

    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)

    task_info = get_service().start_ingest(user_id, str(dest), library_id=library)
    if "error" in task_info:
        dest.unlink(missing_ok=True)
        raise HTTPException(403, task_info["error"])

    background.add_task(
        get_service().run_ingest_task,
        task_id=task_info["task_id"],
        user_id=user_id,
        file_path=str(dest),
        library_id=task_info["library"],
    )
    return task_info


@app.get("/api/tasks/{task_id}")
def task_status(task_id: str) -> dict:
    task = get_service().get_task(task_id)
    if task is None:
        raise HTTPException(404, f"unknown task {task_id}")
    return task
