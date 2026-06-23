from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.document_processing.detailed_analysis import analyze_document
from src.document_processing.smart_document_processor import SmartDocumentProcessor, ProcessingMetrics
from src.langchain_components.vector_store import VectorStore


@dataclass
class IndexRecord:
    source: str
    doc_type: str
    chunks: int
    indexed_at: str
    status: str
    error: Optional[str] = None


class DocumentIndexer:
    """Batch document indexer with analysis enrichment.

    For each file, runs regex-based extraction (amounts, dates, PO numbers, suppliers)
    and stores the results as metadata alongside the vector embeddings.
    Use KnowledgeBaseManager for simpler add/search/delete operations.
    """

    def __init__(self, vector_store: Optional[VectorStore] = None) -> None:
        self._vector_store = vector_store or VectorStore()
        self._processor = SmartDocumentProcessor(vector_store=self._vector_store)
        self._index_log: List[IndexRecord] = []

    def index_from_s3(self, bucket: str, prefix: str = "") -> ProcessingMetrics:
        return self._processor.process_batch_from_s3(bucket, prefix)

    def index_file(self, filepath: str) -> IndexRecord:
        try:
            docs = self._processor.process_local_file(filepath)
            if docs:
                enriched = []
                for doc in docs:
                    analysis = analyze_document(doc.page_content)
                    if analysis.amounts:
                        doc.metadata["extracted_amounts"] = analysis.amounts
                    if analysis.dates:
                        doc.metadata["extracted_dates"] = analysis.dates
                    if analysis.po_numbers:
                        doc.metadata["extracted_po_numbers"] = analysis.po_numbers
                    if analysis.suppliers:
                        doc.metadata["extracted_suppliers"] = analysis.suppliers
                    enriched.append({"text": doc.page_content, "metadata": doc.metadata})
                self._vector_store.add_documents(enriched)
            record = IndexRecord(
                source=filepath,
                doc_type=docs[0].metadata.get("type", "unknown") if docs else "unknown",
                chunks=len(docs),
                indexed_at=datetime.utcnow().isoformat(),
                status="indexed",
            )
            self._index_log.append(record)
            return record
        except Exception as exc:
            record = IndexRecord(
                source=filepath,
                doc_type="unknown",
                chunks=0,
                indexed_at=datetime.utcnow().isoformat(),
                status="failed",
                error=str(exc),
            )
            self._index_log.append(record)
            return record

    def get_index_log(self) -> List[IndexRecord]:
        return list(self._index_log)
