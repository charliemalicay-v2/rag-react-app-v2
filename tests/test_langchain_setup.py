from unittest.mock import MagicMock, patch

import pytest


def test_import_langchain():
    import langchain
    assert hasattr(langchain, "__version__")


def test_import_agent():
    from langchain.agents import create_agent
    assert callable(create_agent)


def test_import_core_modules():
    import langchain_core.messages
    import langchain_core.tools
    import langchain_core.documents
    assert all([langchain_core.messages, langchain_core.tools, langchain_core.documents])


def test_import_vector_store():
    from src.langchain_components.vector_store import VectorStore
    assert hasattr(VectorStore, "similarity_search")
    assert hasattr(VectorStore, "add_documents")


def test_import_agent_module():
    from src.ai_agent.agent import SmartMoveAgent
    assert hasattr(SmartMoveAgent, "process_query")
    assert hasattr(SmartMoveAgent, "aexecute_stream")
    assert hasattr(SmartMoveAgent, "add_documents")


def test_import_embeddings():
    from src.langchain_components.embeddings import get_embeddings
    assert callable(get_embeddings)


def test_import_llm():
    from src.langchain_components.llm import get_llm
    assert callable(get_llm)


def test_import_cache():
    from backend.app.core.cache import cache_response, serialize_response, deserialize_response
    assert callable(cache_response)
    assert callable(serialize_response)
    assert callable(deserialize_response)


def test_import_rate_limit():
    from backend.app.core.rate_limit import RateLimiter
    assert hasattr(RateLimiter, "__call__")


def test_import_config():
    with patch("backend.app.core.config.get_settings") as mock:
        mock.return_value.openai_api_key = "test-key"
        from backend.app.core.config import get_settings, Settings
        assert callable(get_settings)


def test_import_tools():
    from src.ai_agent.tools import get_default_tools
    assert callable(get_default_tools)


def test_import_document_analyzer():
    from src.ai_agent.document_analyzer import DocumentAnalyzer
    assert hasattr(DocumentAnalyzer, "classify")
    assert hasattr(DocumentAnalyzer, "summarize")
    assert hasattr(DocumentAnalyzer, "extract_entities")


def test_import_knowledge_base():
    from src.knowledge_base.knowledge_base_manager import KnowledgeBaseManager
    assert hasattr(KnowledgeBaseManager, "add_documents")
    assert hasattr(KnowledgeBaseManager, "search")
