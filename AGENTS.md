# SmartMove AI — Agent Guide

Python 3.14 + Pipenv monorepo. Document intelligence platform with FastAPI backend, LangChain ReAct agent, ChromaDB (cloud) vector store, and Redis cache/rate-limiter.

## Commands (all via Makefile)

| Command | What it runs |
|---|---|
| `make install` | `pipenv install --dev` |
| `make test` | `pipenv run pytest tests/ -v` |
| `make test-coverage` | add `--cov=src --cov=backend --cov-report=term-missing` |
| `make run` | `pipenv run python backend/run.py` — starts uvicorn on `0.0.0.0:8000` with `reload=True` |
| `make lint` | `pipenv run ruff check .` (skips gracefully if ruff missing) |
| `make typecheck` | `pipenv run mypy .` (skips gracefully if mypy missing) |
| `make clean` | removes `__pycache__`, `.pyc`, `.pytest_cache`, `.coverage`, `htmlcov` |
| `make ingest filepath=<path> analyze=yes` | `python -m src.cli ingest <path> --analyze` |
| `make ingest-s3 bucket=<name> prefix=<prefix>` | `python -m src.cli ingest-s3 <name> --prefix <prefix>` |

## Architecture

- **`backend/run.py`** — entrypoint; patches `sys.path` so imports resolve from project root
- **`backend/app/main.py`** — FastAPI `create_app()` factory; mounts agent router at `/api` and document router at `/api/documents`
- **`backend/app/api/v1/agent.py`** — POST `/api/v1/agent` (cached + rate-limited) and POST `/api/v1/agent/stream` (SSE, rate-limited)
- **`backend/app/core/`** — `config.py` (pydantic-settings from `.env`), `cache.py` (Redis decorator), `rate_limit.py` (sliding-window)
- **`src/ai_agent/agent.py`** — `SmartMoveAgent` wrapping LangChain `create_agent()` (1.x API) with search + add-doc tools
- **`src/langchain_components/`** — `llm.py` (routing), `embeddings.py` (routing), `vector_store.py` (ChromaDB CloudClient)
- **`src/document_processing/`** — `smart_document_processor.py` (PDF/CSV/XLSX/DOCX/TXT/Image OCR), `detailed_analysis.py` (regex extraction)
- **`src/knowledge_base/`** — `document_loader.py` (S3 + retry), `document_indexer.py` (batch indexing), `knowledge_base_manager.py` (high-level CRUD)
- **`src/cli.py`** — CLI for `ingest` and `ingest-s3` subcommands
- **`tests/`** — pytest; all external services mocked in `conftest.py`

## LLM Routing (`src/langchain_components/llm.py`)

Model name contains `gpt`, `o1`, or `o3` → `ChatOpenAI`
Model name contains `claude` → `ChatAnthropic`
Anything else → `ChatOllama`

Same pattern in `embeddings.py`: `text-embedding` in name → OpenAI, else HuggingFace.

## Testing

- Tests use `pytest-asyncio`. Decorate async tests with `@pytest.mark.asyncio`.
- `conftest.py` adds project root to `sys.path`, sets dummy API keys, and mocks:
  - `chromadb`, `boto3`, `pypdf`, `python-docx`, `openpyxl`, `PIL`, `pytesseract`
  - `get_settings()` in all modules
  - `chromadb.HttpClient` → returns a mock collection
  - `get_embeddings()` → returns a mock embedding function
- Agent tests also mock `create_agent` from `src.ai_agent.agent`.
- No external services needed to run tests. Run: `pipenv run pytest tests/ -v`

## Notable Quirks

- **ChromaDB is cloud-based** — uses `chromadb.CloudClient()` with API key/tenant/database, not a local HTTP client.
- **Redis is required at runtime** (rate-limiter + cache). Tests mock it. If Redis is down, `/api/v1/agent` endpoints will fail.
- **Streaming** creates a *new* LLM instance with `streaming=True` rather than reusing the main one.
- **Cache** creates a new `Redis()` connection per request (no pooling).
- **VectorStore** stores text in both the ChromaDB document field AND in `metadatas["text"]` for retrieval.
- **`.env` and `.env.example` are committed** with real-looking credentials (Chroma API key, HF token). Treat with caution.
- **Agent uses `create_agent()`** from `langchain.agents` (1.x API, not LangGraph by default).
- **sentence-transformers can OOM** — default embedding model `sentence-transformers/all-MiniLM-L6-v2` loads locally (~80 MB). `src/langchain_components/embeddings.py` checks available memory before loading and gives clear instructions if memory is low. To avoid OOM:
  - **WSL2**: configure `~/.wslconfig` with `memory=8GB` + `swap=8GB`, then `wsl --shutdown && wsl`
  - **Or** set `embeddings_model_name=text-embedding-3-small` in `.env` to route to OpenAI (API calls, no local load)
