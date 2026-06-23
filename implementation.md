# SmartMove AI Implementation Plan

## Objective
Create a document intelligence platform with a FastAPI backend, AI agent core, LangChain vector store, document ingestion utilities, and supporting tests.

## 1. Project Setup
- [x] Confirm repository structure and add missing configuration files:
   - `Pipfile`
   - `.env.example`
   - `README.md` content for setup and run instructions
- [x] Document required environment variables (in `.env.example` and `backend/app/core/config.py`):
    - `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` — LLM provider keys
    - `PINECONE_API_KEY` — vector store
    - `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD` — cache and rate-limiter
    - `AGENT_MODEL_NAME`, `EMBEDDINGS_MODEL_NAME` — model selection
    - `BACKEND_CORS_ORIGINS` — CORS configuration
- [x] Ensure `backend/`, `src/`, and `tests/` folders are present and importable.

## 2. Backend API Layer
- [x] Implement `backend/run.py`:
   - import `setup_project_paths()`
   - launch Uvicorn with `app.main:app`
- [x] Implement `backend/app/main.py`:
   - create FastAPI app
   - add `CORSMiddleware` using `app.core.config.BACKEND_CORS_ORIGINS`
   - mount router under `/api`
   - add root `/` and `/robots.txt`
- [x] Implement API routes in `backend/app/api/v1/agent.py`:
    - `execute_agent` POST `/api/v1/agent`
      - call `SmartMoveAgent.process_query()`
      - apply caching decorator `cache_response()`
      - apply `RateLimiter` dependency
    - `execute_agent_stream` POST `/api/v1/agent/stream`
      - call `SmartMoveAgent.aexecute_stream()`
      - stream SSE events from the agent
- [x] Implement document chat router in `src/document_processing/process_documents.py`:
   - simple `/chat` endpoint
   - use `create_chat_chain()` from `backend/app/chains/__init__.py`

## 3. Config, Cache, and Rate Limiting
- [x] Implement `backend/app/core/config.py`:
   - load environment variables for model defaults, Redis, rate limits, and CORS
- [x] Implement `backend/app/core/cache.py`:
   - `cache_response()` decorator
   - `serialize_response()` and `deserialize_response()` helpers
   - apply caching to main agent endpoint
- [x] Implement `backend/app/core/rate_limit.py`:
   - `RateLimiter` dependency using Redis sliding-window logic
   - apply to both `/api/v1/agent` and stream endpoint

## 4. AI Agent Core
- [x] Implement `src/ai_agent/agent.py`:
    - `SmartMoveAgent` class
    - `process_query()` — async query with `create_agent` (LangChain 1.x)
    - `aexecute_stream()` — async SSE streaming via `astream`
- [x] Build agent initialization:
   - load LLM via `src/langchain_components/llm.get_llm()`
   - load embeddings via `src/langchain_components/embeddings.get_embeddings()`
   - ~~create `ConversationBufferMemory`~~ (LangChain 1.x `create_agent` handles memory internally)
   - instantiate `VectorStore`
   - register tools:
     - `Search Documents` using `VectorStore.llm_enhanced_similarity_search()`
     - `Add Document` using `VectorStore.add_documents()`
   - build a ReAct agent via `create_agent()` (LangChain 1.x API)
- [x] Add public methods:
   - `add_documents(documents)`
   - `process_query(query, context=None)`
   - `aexecute_stream(query, context=None, available_tools=None, callbacks=None)`
- [x] Add prompt and token helpers:
    - `calculate_tokens()` — approximate token count by word split
    - `_limit_prompt_tokens()` — truncate prompt to max token budget
- [x] Streaming emits JSON `{"type": "token", "content": "..."}` events via `astream`.

## 5. LangChain and Vector Store Layer
- [x] Implement `src/langchain_components/llm.py`:
    - `get_llm(model_name, temperature, streaming)`
    - `get_llm_openai()`, `get_llm_claude()`, `get_llm_ollama()` factories
    - routing: `gpt`/`o1`/`o3` → OpenAI, `claude` → Anthropic, everything else → Ollama
    - retry with `tenacity`
- [x] Implement `src/langchain_components/embeddings.py`:
   - `get_embeddings(model_name)`
   - OpenAI and HuggingFace embedding support
   - retry on rate-limit failures
- [x] Implement `src/langchain_components/vector_store.py`:
    - initialize ChromaDB HTTP client with tenant, database, and API key
    - implement `_get_or_create_collection()`
    - implement `add_documents()` with metadata size enforcement
    - implement `_clean_text()` helper
    - implement `similarity_search()`, `token_limited_similarity_search()`, and `llm_enhanced_similarity_search()`
- [x] Make `VectorStore` the core ingestion and retrieval primitive.

## 6. Document Processing Layer
- [x] Implement `src/document_processing/smart_document_processor.py`:
   - orchestrate S3 document ingestion
   - support handlers for PDF, CSV, Excel, Word, Text, Image (OCR)
   - create `langchain.Document` objects and push to `VectorStore.add_documents()`
   - record processing metrics and handle failures
- [x] Implement `src/knowledge_base/document_loader.py`:
   - load documents from S3 with retry-safe downloads via `tenacity`
   - use `SmartDocumentProcessor`
   - integrate with document analysis via `load_and_analyze()`
- [x] Implement `src/document_processing/detailed_analysis.py`:
   - parse output files and reports
   - extract amounts, dates, PO numbers, suppliers, patterns
   - generate markdown and JSON reports
- [x] Keep `src/document_processing/process_documents.py` as a lightweight chat endpoint.

## 7. Knowledge Base Cleanup
- [x] Clarify active versus placeholder modules in `src/knowledge_base/`:
    - `document_loader.py` — active (S3 retry + analysis integration)
    - `knowledge_base_manager.py` — implemented (high-level CRUD for documents)
    - `document_indexer.py` — implemented (indexing pipeline with analysis enrichment)
    - ~~`vector_store.py`~~ — removed (was a no-op re-export; imports updated to `src.langchain_components.vector_store` directly)
- [x] Review placeholder files in `src/ai_agent/`:
   - `document_analyzer.py` — implemented (LLM-powered classification, summarization, entity extraction)
   - `tools.py` — implemented (reusable tool factories: search, add, list documents)
- [x] Decide whether to wire placeholders into runtime flow or mark them for future development.
   - All placeholders are now implemented and importable.

## 8. Secondary Project Support
- [x] Review `langgraph_server` if present.
  - Not present in the repository. No action needed.
- [x] Keep it isolated unless it is part of the main API flow.
  - N/A — no langgraph_server to isolate.
- [x] Document its role only if needed.
  - N/A — nothing to document.

## 9. Tests and Validation
- [x] Implement tests for:
    - `SmartMoveAgent` initialization and query execution
    - `VectorStore` ingestion and retrieval
    - API route behavior and caching/rate limiting
    - streaming SSE behavior
- [x] Add or update test files:
    - `tests/test_agent.py` — agent init, query, streaming, tokens, cache key
    - `tests/test_vector_store.py` — similarity search, add, token-limited, LLM-enhanced
    - `tests/test_document_analyzer.py` — amounts, dates, PO, suppliers, patterns, reports
    - `tests/test_langchain_setup.py` — importability checks for all modules
    - `tests/test_llm.py` — LLM routing (OpenAI vs Claude, defaults)
    - `tests/test_chroma_store.py` — ChromaDB create/query/add
- [x] Add CI or local commands for test execution:
    - `Makefile` with `make test`, `make test-coverage`, `make run`, `make lint`, `make typecheck`, `make clean`
    - `.github/workflows/ci.yml` — GitHub Actions CI with Redis service, Python 3.14, test runner
- Results: **55/55 tests passing** (`pipenv run pytest tests/ -v`).

## 10. Documentation and Future Cleanup
- [x] Add developer documentation for:
    - environment setup
    - backend startup
    - document ingestion flow
    - test commands (`pipenv run pytest tests/ -v`)
- [x] Refine codebase after initial implementation:
    - review `_get_cache_key()` usage across agent and cache layers
      - removed dead `_get_cache_key()` method and `self.response_cache` from `SmartMoveAgent`; caching is handled by Redis `cache_response` decorator at the API layer
    - consolidate document analyzer responsibilities
      - moved `import json` to top level in `document_analyzer.py`; fixed duplicate "invoice" in `CLASSIFICATION_PROMPT`; separation is clean (regex for structured fields, LLM for classification/summarization/entities)
    - align `src/knowledge_base` with active architecture
      - removed deprecated re-export `vector_store.py`; updated imports in `knowledge_base_manager.py` and `document_indexer.py` to import directly from `src.langchain_components.vector_store`; added docstrings clarifying `KnowledgeBaseManager` vs `DocumentIndexer` use cases
- [x] Track follow-up improvements in `implementation.md` or `README.md` (see README.md → Future Improvements).
