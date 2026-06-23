from unittest.mock import MagicMock, patch

import pytest

from src.langchain_components.vector_store import VectorStore


def test_similarity_search():
    vs = VectorStore()
    vs._collection.query = MagicMock(return_value={
        "ids": [["1", "2"]],
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{"text": "doc1"}, {"text": "doc2"}]],
    })

    results = vs.similarity_search("test", k=2)
    assert len(results) == 2
    assert results[0].page_content == "doc1"


def test_similarity_search_empty():
    vs = VectorStore()
    vs._collection.query = MagicMock(return_value={
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
    })

    results = vs.similarity_search("test")
    assert results == []


def test_similarity_search_with_filter():
    vs = VectorStore()
    vs._collection.query = MagicMock(return_value={
        "ids": [["1"]],
        "documents": [["doc1"]],
        "metadatas": [[{"text": "doc1", "category": "A"}]],
    })

    results = vs.similarity_search("test", filter={"category": "A"})
    assert len(results) == 1


def test_add_documents():
    vs = VectorStore()
    vs._collection.add = MagicMock()

    vs.add_documents([
        {"text": "doc1", "metadata": {"source": "a"}},
        {"text": "doc2", "metadata": {"source": "b"}},
    ])

    vs._collection.add.assert_called_once()


def test_add_documents_empty():
    vs = VectorStore()
    vs.add_documents([])


def test_token_limited_similarity_search():
    vs = VectorStore()
    vs._collection.query = MagicMock(return_value={
        "ids": [["1"]],
        "documents": [["short text"]],
        "metadatas": [[{"text": "short text"}]],
    })

    results = vs.token_limited_similarity_search("test", max_tokens=10)
    assert len(results) == 1


def test_llm_enhanced_query():
    vs = VectorStore()
    vs._collection.query = MagicMock(return_value={
        "ids": [["1"]],
        "documents": [["relevant doc"]],
        "metadatas": [[{"text": "relevant doc"}]],
    })

    with patch("src.langchain_components.vector_store.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(return_value=MagicMock(content="0"))
        mock_get_llm.return_value = mock_llm

        results = vs.llm_enhanced_similarity_search("test")
        assert len(results) == 1
