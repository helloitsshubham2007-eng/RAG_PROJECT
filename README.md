# Marginal — a RAG document Q&A system

Upload documents (PDF, DOCX, TXT, MD), ask questions in plain English, and get
answers that cite the exact passage each claim came from — so you can trust,
and verify, what comes back. This is the real-world pattern behind internal
knowledge bases, policy/compliance assistants, contract review tools, and
customer-support "ask our docs" bots.

## How it works

```
 upload ──▶ extract text ──▶ chunk (220 words, 40 overlap) ──▶ embed locally
                                                                     │
 question ──▶ embed ──▶ vector search (ChromaDB, cosine) ◀──────────┘
                                │
                         top-k relevant chunks
                                │
                         Claude (Anthropic API)
                                │
                     answer + numbered citations
```

- **Embeddings** run locally via `sentence-transformers` (`all-MiniLM-L6-v2`) —
  free, no API calls, no data leaves your machine at the retrieval step.
- **Vector store** is ChromaDB, persisted to `./storage` so your index survives restarts.
- **Generation** uses the Anthropic API. The system prompt forces Claude to
  answer *only* from the retrieved excerpts and cite every claim — this is
  what keeps a RAG system honest instead of hallucinating.

## Setup

```bash
cd rag-project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set your Anthropic API key:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env
```

Get a key at https://console.anthropic.com/settings/keys if you don't have one.

## Run

```bash
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — a sample employee handbook is included in
`sample_docs/employee_handbook.txt`, drag it into the sidebar to try it
immediately, or upload your own files.

## API reference (if you want to use it headless / build another frontend)

| Method | Endpoint              | Purpose                          |
|--------|------------------------|-----------------------------------|
| POST   | `/upload`              | multipart file upload, indexes it |
| GET    | `/documents`           | list indexed documents            |
| DELETE | `/documents/{filename}`| remove a document from the index  |
| POST   | `/chat`                | `{"query": "...", "top_k": 5}` → answer + sources |
| GET    | `/health`              | status check                      |

## Making it production-ready

This is a solid, working core — here's what to add as real usage grows:

- **Bigger scale (100k+ chunks):** swap ChromaDB for a hosted vector DB
  (Qdrant, pgvector, Pinecone) — the `DocumentStore` class is the only place
  that needs to change.
- **Better retrieval:** add a re-ranking step (e.g. cross-encoder) after the
  initial vector search, and/or hybrid search (BM25 + embeddings) for exact
  keyword matches like part numbers or names.
- **Multi-user:** add auth and scope the Chroma collection per user/org.
- **Bigger files:** current PDF extraction is text-only; add OCR (e.g.
  `pytesseract`) for scanned documents.
- **Cost control:** cache repeated queries; use a cheaper/faster Claude model
  for simple factual lookups and reserve larger models for complex synthesis.
- **Observability:** log query → retrieved chunks → answer, so you can spot
  when retrieval is returning irrelevant chunks (the #1 cause of bad RAG answers).

## Project structure

```
rag-project/
├── main.py            FastAPI app / HTTP endpoints
├── rag_engine.py       chunking, embedding, retrieval, generation
├── static/index.html   web UI
├── sample_docs/        demo document
├── storage/             persisted vector index (created on first run)
├── requirements.txt
└── .env                 your API key (create this, don't commit it)
```
