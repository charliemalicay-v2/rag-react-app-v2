import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set dummy env vars so ChatOpenAI etc. can instantiate without real keys
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-fake-key")

pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(autouse=True)
def mock_external_packages():
    packages = {
        "chromadb": MagicMock(),
        "boto3": MagicMock(),
        "pypdf": MagicMock(),
        "docx": MagicMock(),
        "openpyxl": MagicMock(),
        "PIL": MagicMock(),
        "pytesseract": MagicMock(),
    }
    for name, mock in packages.items():
        if name not in sys.modules:
            sys.modules[name] = mock


@pytest.fixture(autouse=True)
def mock_settings():
    settings = MagicMock()
    settings.openai_api_key = "sk-test-fake-key"
    settings.anthropic_api_key = "sk-ant-test-fake-key"
    settings.hf_token = "hf-test-token"
    settings.chroma_api_key = "test-chroma-key"
    settings.chroma_tenant = "test-tenant"
    settings.chroma_database = "test-db"
    settings.chroma_host = "test.chroma.app"
    settings.chroma_collection_name = "test-collection"
    settings.redis_host = "localhost"
    settings.redis_port = 6379
    settings.redis_password = None
    settings.backend_cors_origins = ["*"]
    settings.agent_model_name = "minimax-m3:cloud"
    settings.embeddings_model_name = "text-embedding-3-small"
    with (
        patch("backend.app.core.config.get_settings", return_value=settings),
        patch("src.langchain_components.llm.get_settings", return_value=settings),
        patch("src.langchain_components.embeddings.get_settings", return_value=settings),
        patch("src.langchain_components.vector_store.get_settings", return_value=settings),
        patch("src.ai_agent.agent.get_settings", return_value=settings),
    ):
        yield


@pytest.fixture(autouse=True)
def mock_chromadb():
    with patch("chromadb.CloudClient") as mock:
        client_instance = MagicMock()
        collection = MagicMock()
        collection.add = MagicMock()
        collection.query = MagicMock(return_value={
            "ids": [["1", "2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"text": "doc1"}, {"text": "doc2"}]],
        })
        client_instance.get_or_create_collection = MagicMock(return_value=collection)
        client_instance.get_collection = MagicMock(return_value=collection)
        client_instance.create_collection = MagicMock(return_value=collection)
        mock.return_value = client_instance
        yield collection


@pytest.fixture(autouse=True)
def mock_embeddings():
    embed_instance = MagicMock()
    embed_instance.embed_query = MagicMock(return_value=[0.1] * 1536)
    embed_instance.embed_documents = MagicMock(return_value=[[0.1] * 1536, [0.2] * 1536])
    embed_instance.aembed_query = MagicMock(return_value=[0.1] * 1536)
    embed_instance.aembed_documents = MagicMock(return_value=[[0.1] * 1536, [0.2] * 1536])

    with (
        patch("src.langchain_components.embeddings.get_embeddings", return_value=embed_instance),
        patch("src.langchain_components.vector_store.get_embeddings", return_value=embed_instance),
        patch("src.ai_agent.agent.get_embeddings", return_value=embed_instance),
    ):
        yield embed_instance
