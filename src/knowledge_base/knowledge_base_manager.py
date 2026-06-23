import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.document_processing.smart_document_processor import SmartDocumentProcessor, ProcessingMetrics
from src.langchain_components.vector_store import VectorStore


@dataclass
class DocumentRecord:
    id: str
    source: str
    type: str
    added_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeBaseManager:
    """High-level CRUD for documents in the knowledge base.

    Use this for general-purpose document management (add, search, list, delete).
    For batch indexing with analysis enrichment, use DocumentIndexer instead.
    """

    def __init__(self, vector_store: Optional[VectorStore] = None) -> None:
        self._vector_store = vector_store or VectorStore()
        self._processor = SmartDocumentProcessor(vector_store=self._vector_store)
        self._documents: Dict[str, DocumentRecord] = {}

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        self._vector_store.add_documents(documents)

    def add_from_s3(self, bucket: str, prefix: str = "") -> ProcessingMetrics:
        return self._processor.process_batch_from_s3(bucket, prefix)

    def add_from_local(self, filepath: str) -> List[DocumentRecord]:
        docs = self._processor.process_local_file(filepath)
        records = []
        for doc in docs:
            clean_text = VectorStore._clean_text(doc.page_content)
            doc_id = hashlib.md5(clean_text.encode("utf-8")).hexdigest()
            record = DocumentRecord(
                id=doc_id,
                source=doc.metadata.get("source", filepath),
                type=doc.metadata.get("type", "unknown"),
                added_at=datetime.utcnow().isoformat(),
                metadata=doc.metadata,
            )
            self._documents[doc_id] = record
            records.append(record)
        self._vector_store.add_documents([
            {"text": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ])
        return records

    def search(self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Any]:
        return self._vector_store.similarity_search(query, k=k, filter=filter)

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        return self._documents.get(doc_id)

    def list_documents(self) -> List[DocumentRecord]:
        return list(self._documents.values())

    def delete_document(self, doc_id: str) -> bool:
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False
