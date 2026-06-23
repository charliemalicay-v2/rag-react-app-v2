import hashlib
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.errors import NotFoundError
from langchain_core.documents import Document

from backend.app.core.config import get_settings
from src.langchain_components.llm import get_llm


class VectorStore:
    def __init__(self, collection_name: str = "smartmove-collection") -> None:
        settings = get_settings()
        self.collection_name = collection_name

        self._client = chromadb.CloudClient(
            api_key=settings.chroma_api_key,
            tenant=settings.chroma_tenant,
            database=settings.chroma_database,
        )
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Any:
        try:
            return self._client.get_collection(name=self.collection_name)
        except NotFoundError:
            return self._client.create_collection(name=self.collection_name)

    @staticmethod
    def _clean_text(text: str) -> str:
        lines = text.split("\n")
        cleaned = [" ".join(line.split()) for line in lines]
        return "\n".join(cleaned).strip()

    @staticmethod
    def _enforce_metadata_size(metadata: Dict[str, Any], max_bytes: int = 40960) -> Dict[str, Any]:
        encoded = str(metadata).encode("utf-8")
        if len(encoded) <= max_bytes:
            return metadata
        truncated = {}
        for key, value in metadata.items():
            truncated[key] = str(value)[:1000]
        return truncated

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for doc in documents:
            text = self._clean_text(doc.get("text", ""))
            metadata = self._enforce_metadata_size(doc.get("metadata", {}))
            doc_id = hashlib.md5(text.encode("utf-8")).hexdigest()
            ids.append(doc_id)
            texts.append(text)
            metadatas.append(metadata)

        self._collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

    def similarity_search(self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        results = self._collection.query(
            query_texts=[query],
            n_results=k,
            where=filter,
        )
        docs: List[Document] = []
        for i in range(len(results["ids"][0])):
            doc_text = results["documents"][0][i]
            doc_meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            docs.append(Document(page_content=doc_text, metadata=doc_meta))
        return docs

    def token_limited_similarity_search(self, query: str, k: int = 5, max_tokens: int = 2000, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        results = self.similarity_search(query, k=k, filter=filter)
        token_count = 0
        limited: List[Document] = []
        for doc in results:
            approx_tokens = len(doc.page_content.split())
            if token_count + approx_tokens > max_tokens:
                remaining = max_tokens - token_count
                if remaining > 0:
                    limited.append(Document(page_content=" ".join(doc.page_content.split()[:remaining]), metadata=doc.metadata))
                break
            limited.append(doc)
            token_count += approx_tokens
        return limited

    def llm_enhanced_similarity_search(self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None, llm: Any = None) -> List[Document]:
        results = self.similarity_search(query, k=k * 2, filter=filter)
        if not results:
            return []
        if llm is None:
            llm = get_llm()
        context = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(results))
        prompt = (
            f"Given the query: '{query}'\n\n"
            f"Rank the following document excerpts by relevance (most relevant first). "
            f"Return the indices in order, comma-separated:\n\n{context}"
        )
        response = llm.invoke(prompt)
        try:
            indices = [int(i.strip()) for i in response.content.split(",") if i.strip().isdigit()]
            return [results[i] for i in indices if i < len(results)][:k]
        except (ValueError, IndexError):
            return results[:k]
