"""
main.py — FastAPI app exposing the RAG system over HTTP.

Run:
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000
"""

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_engine import DocumentStore, generate_answer

load_dotenv()

app = FastAPI(title="RAG Document Q&A")
store = DocumentStore()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/")
def serve_ui():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/health")
def health():
    return {"status": "ok", "documents_indexed": len(store.list_documents())}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = store.ingest_file(tmp_path, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return JSONResponse(result)


@app.get("/documents")
def list_documents():
    return {"documents": store.list_documents()}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    deleted = store.delete_document(filename)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"filename": filename, "chunks_deleted": deleted}


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    hits = store.retrieve(req.query, k=req.top_k)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY is not set. Add it to a .env file or your environment.",
        )

    answer = generate_answer(req.query, hits, api_key=api_key)
    return {
        "answer": answer,
        "sources": [
            {"index": i + 1, "source": h["source"], "chunk_index": h["chunk_index"],
             "relevance": h["relevance"], "excerpt": h["text"][:220] + ("..." if len(h["text"]) > 220 else "")}
            for i, h in enumerate(hits)
        ],
    }
