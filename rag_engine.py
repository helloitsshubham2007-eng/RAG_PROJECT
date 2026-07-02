"""
rag_engine.py
Core RAG logic: document loading, chunking, embedding, vector storage,
retrieval, and answer generation with citations.

Design choices (why, so future-you can change them confidently):
- Embeddings: sentence-transformers "all-MiniLM-L6-v2" — runs locally, free,
  no API calls, fast enough for CPU. Swap EMBED_MODEL_NAME to upgrade quality.
- Vector store: ChromaDB persistent client — zero setup, stores to disk,
  good enough for up to ~100k chunks. Swap for pgvector/Qdrant at scale.
- Generation: Anthropic Claude — reads the retrieved chunks and answers
  ONLY from them, citing which chunk each claim came from.
"""

import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from docx import Document as DocxDocument
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE_WORDS = 220          # ~ 300-350 tokens, good recall/precision balance
CHUNK_OVERLAP_WORDS = 40        # keeps context continuous across chunk boundaries
TOP_K_DEFAULT = 5
CLAUDE_MODEL = "claude-sonnet-5"

STORAGE_DIR = Path(__file__).parent / "storage"
STORAGE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
def extract_text(file_path: str, filename: str) -> str:
    """Pull raw text out of pdf / docx / txt / md files."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        doc = DocxDocument(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    if suffix in (".txt", ".md"):
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: {suffix}. Use pdf, docx, txt, or md.")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS,
               overlap: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    """
    Word-based sliding-window chunking with overlap.
    Simple and robust across document types; good enough for most real docs.
    Collapses whitespace first so page-break artifacts don't create junk chunks.
    """
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")
    if not words or words == [""]:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap  # step forward, keeping overlap
    return chunks


# ---------------------------------------------------------------------------
# Vector store wrapper
# ---------------------------------------------------------------------------
class DocumentStore:
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path=str(STORAGE_DIR))
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL_NAME
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_file(self, file_path: str, filename: str) -> Dict[str, Any]:
        text = extract_text(file_path, filename)
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No extractable text found in this file.")

        ids = [f"{filename}-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        return {"filename": filename, "chunks_added": len(chunks)}

    def retrieve(self, query: str, k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        k = min(k, self.collection.count())
        results = self.collection.query(query_texts=[query], n_results=k)

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append({
                "text": doc,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "relevance": round(1 - dist, 4),  # cosine distance -> similarity
            })
        return hits

    def list_documents(self) -> List[Dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        all_items = self.collection.get()
        counts: Dict[str, int] = {}
        for meta in all_items["metadatas"]:
            counts[meta["source"]] = counts.get(meta["source"], 0) + 1
        return [{"filename": k, "chunks": v} for k, v in counts.items()]

    def delete_document(self, filename: str) -> int:
        all_items = self.collection.get(where={"source": filename})
        ids = all_items["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a precise research assistant. You answer ONLY using the \
numbered source excerpts provided in the user's message. Rules:
1. Every factual claim must be followed by a citation like [1], [2] referring to \
the excerpt number it came from.
2. If the excerpts don't contain enough information to answer, say so plainly \
instead of guessing or using outside knowledge.
3. Be concise and direct. Do not repeat the question back.
4. If multiple excerpts support one claim, cite all of them, e.g. [1][3]."""


def build_context_block(hits: List[Dict[str, Any]]) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] (source: {hit['source']}, section {hit['chunk_index']})\n{hit['text']}")
    return "\n\n".join(lines)


def generate_answer(query: str, hits: List[Dict[str, Any]], api_key: str = None) -> str:
    if not hits:
        return ("I couldn't find any relevant content in the uploaded documents "
                "to answer that. Try uploading a document first, or rephrase your question.")

    client = Anthropic(api_key=api_key) if api_key else Anthropic()
    context_block = build_context_block(hits)

    user_message = (
        f"Source excerpts:\n\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer the question using only the excerpts above, with citations."
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text
