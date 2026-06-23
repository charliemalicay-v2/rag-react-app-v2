from typing import Any, Dict, List, Optional

from langchain_core.tools import Tool

from src.langchain_components.vector_store import VectorStore


def search_documents_tool(vector_store: VectorStore) -> Tool:
    return Tool(
        name="Search Documents",
        func=lambda q: vector_store.llm_enhanced_similarity_search(q),
        description="Search the document store for relevant information. Input is a search query string.",
    )


def add_document_tool(vector_store: VectorStore) -> Tool:
    return Tool(
        name="Add Document",
        func=lambda docs: vector_store.add_documents(docs),
        description="Add a new document to the vector store. Input is a list of dicts with 'text' and 'metadata' keys.",
    )


def list_documents_tool(vector_store: VectorStore) -> Tool:
    return Tool(
        name="List Documents",
        func=lambda _: {"message": "Use the vector store to list available documents."},
        description="List available documents or namespaces in the store.",
    )


def get_default_tools(vector_store: Optional[VectorStore] = None) -> List[Tool]:
    vs = vector_store or VectorStore()
    return [
        search_documents_tool(vs),
        add_document_tool(vs),
        list_documents_tool(vs),
    ]
