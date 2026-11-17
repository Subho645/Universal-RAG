"""
FastAPI Web Interface for Universal RAG
Run: uvicorn app:app --reload
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List
from fastapi import Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import UniversalRAG, LOADERS

app = FastAPI(title="Universal RAG API", version="1.0")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

def check_admin(x_admin_key: str = Header(None)):
    if ADMIN_PASSWORD and x_admin_key != ADMIN_PASSWORD:
        raise HTTPException(401, detail="Admin password required")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag = UniversalRAG(
    db_path="./rag_db",
    embedding_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
    llm_model=os.getenv("LLM_MODEL", "llama3.2"),
    chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
    top_k=int(os.getenv("TOP_K", "4")),
)

UPLOAD_DIR = Path("./uploaded_docs")
UPLOAD_DIR.mkdir(exist_ok=True)


class QueryRequest(BaseModel):
    question: str


# ─── Routes ───────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return open("static/index.html", encoding="utf-8").read()

@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    return open("static/admin.html", encoding="utf-8").read()


@app.post("/upload", dependencies=[Depends(check_admin)])
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload one or more documents for ingestion."""
    saved = []
    unsupported = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in LOADERS:
            unsupported.append(file.filename)
            continue
        dest = UPLOAD_DIR / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(str(dest))

    if not saved:
        raise HTTPException(400, detail=f"No supported files uploaded. Unsupported: {unsupported}")

    try:
        rag.ingest(saved)
    except Exception as e:
        raise HTTPException(500, detail=f"Ingestion failed: {str(e)}")

    return {
        "status": "success",
        "ingested": [Path(p).name for p in saved],
        "skipped": unsupported,
    }


@app.post("/query")
def query(req: QueryRequest):
    """Ask a question against the ingested documents."""
    if not req.question.strip():
        raise HTTPException(400, detail="Question cannot be empty.")
    try:
        result = rag.query(req.question)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))


@app.get("/status")
def status():
    """Check Ollama connectivity and DB status."""
    import requests as req
    ollama_ok = False
    try:
        r = req.get("http://localhost:11434/api/tags", timeout=5)
        ollama_ok = r.status_code == 200
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        models = []

    try:
        from rag_pipeline import get_qdrant_client
        qc = get_qdrant_client()
        cols = qc.get_collections().collections
        db_exists = len(cols) > 0
    except Exception as e:
        logger.warning(f"Qdrant status check failed: {e}")
        db_exists = False

    return {
        "ollama_running": ollama_ok,
        "available_models": models,
        "db_exists": db_exists,
        "upload_dir": str(UPLOAD_DIR),
    }


@app.delete("/reset", dependencies=[Depends(check_admin)])
def reset_db():
    from rag_pipeline import get_qdrant_client, get_collection_name
    client = get_qdrant_client()
    for col in client.get_collections().collections:
        if col.name.startswith("doc_"):
            client.delete_collection(col.name)
    for f in UPLOAD_DIR.iterdir():
        f.unlink()
    rag.client = None
    return {"status": "reset complete"}
@app.get("/documents", dependencies=[Depends(check_admin)])
def list_documents():
    from rag_pipeline import get_qdrant_client
    qc = get_qdrant_client()
    docs = []
    for c in qc.get_collections().collections:
        if not c.name.startswith("doc_"):
            continue
        pts = qc.scroll(c.name, limit=1, with_payload=True)[0]
        label = pts[0].payload.get("source_file", c.name) if pts else c.name
        docs.append({"id": c.name, "name": label,
                     "chunks": qc.get_collection(c.name).points_count})
    return docs

@app.delete("/documents/{doc_id}", dependencies=[Depends(check_admin)])
def delete_document(doc_id: str):
    from rag_pipeline import get_qdrant_client
    if not doc_id.startswith("doc_"):
        raise HTTPException(400, detail="Invalid document id")
    get_qdrant_client().delete_collection(doc_id)
    return {"status": "deleted"}
  