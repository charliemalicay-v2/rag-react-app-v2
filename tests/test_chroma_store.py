from unittest.mock import MagicMock, patch

import pytest


def test_chroma_creates_collection_when_missing():
    with patch("chromadb.CloudClient") as mock:
        client = MagicMock()
        client.get_collection = MagicMock(side_effect=ValueError("not found"))
        collection = MagicMock()
        client.create_collection = MagicMock(return_value=collection)
        mock.return_value = client

        from src.langchain_components.vector_store import VectorStore
        vs = VectorStore(collection_name="new-collection")

        client.create_collection.assert_called_once_with(name="new-collection")


def test_chroma_uses_existing_collection():
    with patch("chromadb.CloudClient") as mock:
        client = MagicMock()
        collection = MagicMock()
        client.get_collection = MagicMock(return_value=collection)
        client.create_collection = MagicMock()
        mock.return_value = client

        from src.langchain_components.vector_store import VectorStore
        vs = VectorStore(collection_name="existing-collection")

        client.get_collection.assert_called_once_with(name="existing-collection")
        client.create_collection.assert_not_called()


def test_chroma_add_documents():
    with patch("chromadb.CloudClient") as mock:
        client = MagicMock()
        collection = MagicMock()
        collection.add = MagicMock()
        client.get_collection = MagicMock(return_value=collection)
        mock.return_value = client

        from src.langchain_components.vector_store import VectorStore
        vs = VectorStore(collection_name="test-collection")

        vs.add_documents([{"text": "test doc", "metadata": {"source": "test"}}])
        collection.add.assert_called_once()


def test_chroma_query():
    with patch("chromadb.CloudClient") as mock:
        client = MagicMock()
        collection = MagicMock()
        collection.query = MagicMock(return_value={
            "ids": [["1"]],
            "documents": [["result text"]],
            "metadatas": [[{"text": "result text"}]],
        })
        client.get_collection = MagicMock(return_value=collection)
        mock.return_value = client

        from src.langchain_components.vector_store import VectorStore
        vs = VectorStore(collection_name="test-collection")

        results = vs.similarity_search("test query")
        assert len(results) == 1
        assert results[0].page_content == "result text"
