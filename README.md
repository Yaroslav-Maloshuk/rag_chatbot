<img src="https://github.com/Yaroslav-Maloshuk/rag_chatbot/blob/main/rag_chatbot.jpeg" width="1000" height="1000">

# RAG Chatbot (FastAPI + pgvector + Celery + React)

Production-ready Retrieval-Augmented Generation (RAG) project for PDF-based Q&A:
- upload PDF files,
- extract text (with OCR fallback),
- split into chunks,
- generate embeddings,
- retrieve relevant context from pgvector + BM25,
- generate grounded answers with source citations.

Pipeline:
`PDF -> Extraction/OCR -> Chunking -> Embeddings -> pgvector/BM25 -> Rerank -> LLM`

## Stack

- **Backend**: FastAPI, SQLAlchemy, Supabase, Chroma, Qdrant, Redis, Celery
- **ML/NLP**: sentence-transformers, transformers, Gemini API, LangChain text splitters, LangGraph, LangFuse, LlamaIndex
- **Frontend**: React, Vite, Tailwind CSS
- **Infra**: Docker Compose + OS-specific start/stop scripts

## Architecture

Canonical backend layers:
- `app/bootstrap` — app startup and wiring
- `app/presentation/http` — HTTP layer (routes + dependencies)
- `app/services` — business orchestration
- `app/repositories` + `app/db` — data access and persistence
- `app/tasks` — asynchronous ingestion pipeline

Legacy imports under `app/api` are kept as backward-compatible shims.

Detailed structure guide: [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)

## Key Features

- PDF upload with file size limits (`MAX_UPLOAD_SIZE_MB`)
- Sync processing for small files, async queue for larger files
- Document lifecycle: `uploaded -> processing -> ready|failed`
- Chat over selected documents or all ready documents
- Streaming answers over SSE and standard non-stream endpoint
- Hybrid retrieval (vector + BM25), optional reranker, response cache
- Metrics endpoint (`/metrics`) when `ENABLE_METRICS=true`

## Requirements

### All platforms
- Docker + Docker Compose
- Git

### Additional for MPS mode
- Apple Silicon macOS (arm64)
- Python `3.11`

## Quick Start

### macOS / Linux

```bash
cp .env.example .env
./scripts/start.sh --mode auto
```

### Windows (PowerShell)

```powershell
Copy-Item .env.example .env -ErrorAction SilentlyContinue
.\scripts\start.ps1 -Mode docker
```

After startup:
- API docs: `http://localhost:18000/docs`
- Frontend: `http://localhost:15173`

## Run Modes

### `--mode docker` (full Docker)
Starts all services in containers:
- `api`
- `worker`
- `redis`
- `frontend`

### `--mode mps` (hybrid, Apple Silicon only)
Starts:
- Docker: `redis`, `frontend`
- Native macOS processes: `api`, `worker` with `MODEL_DEVICE=mps`

This mode exists because Apple MPS is not available inside Linux containers.

### `--mode auto`
Automatically selects:
- `mps` on Apple Silicon macOS
- `docker` everywhere else

## Service Management

### macOS / Linux

```bash
./scripts/status.sh
./scripts/stop.sh --mode all
```

`stop.sh --mode` options:
- `all`: stop local MPS processes and Docker services
- `docker`: stop Docker services only
- `mps`: stop local MPS processes + docker infra/frontend

### Windows (PowerShell)

```powershell
.\scripts\stop.ps1 -Mode all
```

## MPS Mode Details

`./scripts/start.sh --mode mps` performs:
1. Ensures `.env` exists (creates from `.env.example` if needed)
2. Starts `redis` + `frontend`
3. Prepares local `.venv` via `./scripts/setup_mps_env.sh`
4. Runs `api` and `worker` locally with `MODEL_DEVICE=mps`
5. Writes runtime PID/log files into `.runtime/`

Useful runtime files:
- `.runtime/api_mps.log`
- `.runtime/worker_mps.log`
- `.runtime/api_mps.pid`
- `.runtime/worker_mps.pid`

## Configuration (.env)

Full template: [`.env.example`](.env.example)

Most important variables:
- `API_HOST_PORT`, `FRONTEND_HOST_PORT`, `REDIS_HOST_PORT`
- `DATABASE_URL`, `REDIS_URL`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`
- `MODEL_DEVICE`, `LLM_MODEL_NAME`, `EMBEDDING_MODEL_NAME`
- `ENABLE_HYBRID_SEARCH`, `ENABLE_RERANKER`, `ENABLE_CACHE`
- `UPLOAD_DIR`, `MAX_UPLOAD_SIZE_MB`, `LARGE_FILE_THRESHOLD_MB`

## API Overview

Base prefix: `/api/v1`

### Health
- `GET /health/live`
- `GET /health/ready`

### Documents
- `POST /documents/upload`
- `GET /documents`
- `GET /documents/{document_id}`
- `POST /documents/delete`

### Chat
- `POST /chat`
- `POST /chat/stream` (SSE)

## API Examples

### 1) Upload PDF

```bash
curl -X POST "http://localhost:18000/api/v1/documents/upload" \
  -F "file=@/absolute/path/to/file.pdf"
```

### 2) List documents

```bash
curl "http://localhost:18000/api/v1/documents?limit=100"
```

### 3) Standard chat request

```bash
curl -X POST "http://localhost:18000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key points?",
    "document_ids": ["<DOCUMENT_ID>"],
    "top_k": 7,
    "use_hybrid_search": true,
    "use_reranker": true
  }'
```

### 4) Streaming chat request (SSE)

```bash
curl -N -X POST "http://localhost:18000/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize section 2",
    "document_ids": ["<DOCUMENT_ID>"]
  }'
```

SSE event types:
- `chunk`
- `sources`
- `done`

### 5) Delete ready documents

```bash
curl -X POST "http://localhost:18000/api/v1/documents/delete" \
  -H "Content-Type: application/json" \
  -d '{
    "document_ids": ["<DOCUMENT_ID_1>", "<DOCUMENT_ID_2>"]
  }'
```

## Frontend

Frontend runs at `http://localhost:15173` and supports:
- PDF upload/delete
- ready-document selection
- streaming/non-streaming chat
- hybrid search/reranker/top-k controls
- stop + regenerate controls

## Chat Controls Explained

- **Streaming**: returns tokens incrementally in real time instead of waiting for the full answer.
- **Hybrid search**: combines semantic vector search with keyword BM25 search to improve recall and robustness.
- **Reranker**: re-scores retrieved chunks with a cross-encoder model and reorders them by relevance before generation.
- **Top-K**: number of retrieved chunks used as context for answer generation.
  - Example: `Top-K = 5` means the model receives the 5 most relevant chunks.

Practical tuning guidance:
- Lower `Top-K` usually gives faster responses but may miss context.
- Higher `Top-K` can improve coverage but may add noise and latency.
- `Reranker` typically improves precision, with extra compute cost.

## Tests

```bash
pytest -q
```

## Troubleshooting

### UI changes are not visible

```bash
docker compose up -d --build frontend
```

### MPS is unavailable
- Confirm Apple Silicon macOS
- Run:

```bash
./scripts/setup_mps_env.sh
```

### First response is slow
The first requests may download and warm up model weights.

Check logs:

```bash
tail -n 120 .runtime/api_mps.log
tail -n 120 .runtime/worker_mps.log
```
