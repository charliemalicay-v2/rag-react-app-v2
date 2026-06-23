from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

from src.document_processing.detailed_analysis import analyze_document
from src.document_processing.smart_document_processor import SmartDocumentProcessor


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _s3_download(bucket: str, key: str) -> bytes:
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def load_documents(source: str, metadata: Optional[Dict[str, Any]] = None) -> List[Any]:
    if source.startswith("s3://"):
        parts = source[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        content = _s3_download(bucket, key)
        processor = SmartDocumentProcessor()
        docs = processor.process_file(content, key.split("/")[-1], source=source)
    elif source.startswith("/") or source.startswith("./") or source.startswith("~"):
        processor = SmartDocumentProcessor()
        docs = processor.process_local_file(source)
    else:
        raise ValueError(f"Unsupported source: {source}")

    if metadata:
        for doc in docs:
            doc.metadata.update(metadata)

    return docs


def load_and_analyze(source: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    docs = load_documents(source, metadata)
    analyses = []
    for doc in docs:
        analysis = analyze_document(doc.page_content)
        analyses.append({
            "source": doc.metadata.get("source", source),
            "content_preview": doc.page_content[:200],
            "analysis": {
                "amounts": analysis.amounts,
                "dates": analysis.dates,
                "po_numbers": analysis.po_numbers,
                "suppliers": analysis.suppliers,
                "patterns": analysis.patterns,
            },
        })
    return {"document_count": len(docs), "analyses": analyses}
