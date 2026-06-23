# SmartMove AI

A document intelligence platform with a FastAPI backend, AI agent core, LangChain vector store, and document ingestion utilities.

## Setup

1. Copy `.env.example` to `.env` and fill in credentials.
2. Install dependencies and create the virtual environment with Pipenv:
   ```bash
   pipenv install
   ```
3. Activate the Pipenv shell:
   ```bash
   pipenv shell
   ```
4. Run the backend:
   ```bash
   python backend/run.py
   ```

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/robots.txt` | Robots exclusion file |
| POST | `/api/v1/agent` | Execute agent query (cached, rate-limited) |
| POST | `/api/v1/agent/stream` | Stream agent response as SSE (rate-limited) |
| POST | `/api/documents/chat` | Document chat endpoint |

## Testing the APIs

### Health check
```bash
curl -s http://localhost:8000/
```
```json
{"status":"ok","service":"SmartMove AI"}
```

### Agent query
```bash
curl -s -X POST http://localhost:8000/api/v1/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What documents do you have about contracts?"}'
```

### Streaming agent query (SSE)
```bash
curl -s -N -X POST http://localhost:8000/api/v1/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the latest invoices"}'
```

### Document chat
```bash
curl -s -X POST http://localhost:8000/api/documents/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Find all documents from vendor ABC"}'
```

> **Note:** The agent and rate-limiting endpoints require a running Redis server on `localhost:6379`. Start one with `redis-server` or configure the connection via environment variables in `.env`.

## Document Ingestion Flow

Documents flow through three stages: **load**, **analyze**, and **index**.

```
Source (S3 / local file)
    │
    ▼
SmartDocumentProcessor
    ├── PDF  → PyPDF
    ├── CSV  → csv module
    ├── XLSX → openpyxl
    ├── DOCX → python-docx
    ├── TXT  → plain text
    └── Image → pytesseract OCR
    │
    ▼
VectorStore.add_documents()
    └── Embed via OpenAI / HuggingFace / Ollama
    └── Store in ChromaDB collection
```

### Ingest a local file

**CLI (recommended):**
```bash
# Basic ingestion
make ingest filepath=/path/to/invoice.pdf

# With analysis enrichment (amounts, dates, PO #s, suppliers)
make ingest filepath=/path/to/report.pdf analyze=yes

# Or directly via python
pipenv run python -m src.cli ingest /path/to/file.pdf --analyze
```

**Python API:**
```python
from src.knowledge_base.knowledge_base_manager import KnowledgeBaseManager

kbm = KnowledgeBaseManager()
records = kbm.add_from_local("/path/to/invoice.pdf")
# returns list of DocumentRecord with id, source, type, added_at
```

### Ingest from S3

**CLI:**
```bash
make ingest-s3 bucket=my-bucket prefix=invoices/
# Or directly
pipenv run python -m src.cli ingest-s3 my-bucket --prefix invoices/
```

**Python API:**
```python
from src.knowledge_base.document_indexer import DocumentIndexer

indexer = DocumentIndexer()
metrics = indexer.index_from_s3("my-bucket", prefix="invoices/")
# ProcessingMetrics with document_count, analysis summaries per file
```

### Load, analyze, and inspect without indexing

```python
from src.knowledge_base.document_loader import load_and_analyze

result = load_and_analyze("s3://my-bucket/report.pdf", metadata={"department": "finance"})
# result.analyses[i].analysis contains amounts, dates, PO numbers, suppliers
```

### Search ingested documents

```python
results = kbm.search("Q4 financial report", k=5)
for doc in results:
    print(doc.page_content)
```

Key modules:
- `src/document_processing/smart_document_processor.py` — file parsing and handler dispatch
- `src/document_processing/detailed_analysis.py` — regex extraction (amounts, dates, PO, suppliers)
- `src/ai_agent/document_analyzer.py` — LLM-powered classification, summarization, entity extraction
- `src/langchain_components/vector_store.py` — ChromaDB vector storage and retrieval
- `src/langchain_components/llm.py` — OpenAI / Claude / Ollama LLM dispatch
- `src/langchain_components/embeddings.py` — OpenAI / HuggingFace embedding dispatch
- `src/knowledge_base/document_loader.py` — S3 retry-safe loading with analysis
- `src/knowledge_base/knowledge_base_manager.py` — high-level CRUD
- `src/knowledge_base/document_indexer.py` — batch indexing with analysis enrichment

## Project Layout

- `backend/` - FastAPI application and API routes
- `src/` - AI agent core, LangChain components, document processing, and knowledge base helpers
- `tests/` - unit and integration test stubs
- `.env.example` - example environment variables

## Future Improvements

- **Redis graceful degradation** — Rate-limiter and cache should fall back to no-op when Redis is unavailable instead of crashing.
- **Startup validation** — Validate required env vars (API keys, Redis config) on startup and print clear error messages.
- **Async Redis connection pooling** — `cache_response` creates a new Redis connection per request; switch to a shared pool.
- **Streaming resilience** — `aexecute_stream` lacks error handling; failed streaming calls should return an error event instead of hanging.
- **Docker support** — Add `Dockerfile` and `docker-compose.yml` with Redis and the app service.
- **Pre-commit hooks** — Add lint (ruff) and format (black) checks via pre-commit.
- **CI secrets** — The GitHub Actions workflow needs real API keys (OpenAI, Pinecone) injected as secrets for integration tests.
