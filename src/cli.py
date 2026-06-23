import argparse
import sys
from pathlib import Path


def cmd_ingest(args: argparse.Namespace) -> None:
    filepath = args.filepath
    if not Path(filepath).exists():
        print(f"Error: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    from src.knowledge_base.document_indexer import DocumentIndexer

    indexer = DocumentIndexer()
    record = indexer.index_file(filepath)

    if record.status == "indexed":
        print(f"  ✓ {record.source}  ({record.chunks} chunks, {record.doc_type})")
    else:
        print(f"  ✗ {record.source}  (error: {record.error})", file=sys.stderr)


def cmd_ingest_s3(args: argparse.Namespace) -> None:
    from src.document_processing.smart_document_processor import SmartDocumentProcessor

    indexer = SmartDocumentProcessor()
    metrics = indexer.process_batch_from_s3(args.bucket, args.prefix or "")
    print(f"Processed {metrics.total_files} documents from s3://{args.bucket}/{args.prefix or ''}")
    print(f"  succeeded: {metrics.successful}, failed: {metrics.failed}")
    if metrics.errors:
        for err in metrics.errors[:5]:
            print(f"  ✗ {err}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(prog="smartmove", description="SmartMove AI document ingestion")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest", help="Ingest a local file")
    ingest_parser.add_argument("filepath", help="Path to the file")
    ingest_parser.add_argument("--analyze", "-a", action="store_true", help="Run analysis enrichment during indexing")
    ingest_parser.set_defaults(func=cmd_ingest)

    s3_parser = sub.add_parser("ingest-s3", help="Ingest all files from an S3 prefix")
    s3_parser.add_argument("bucket", help="S3 bucket name")
    s3_parser.add_argument("--prefix", default="", help="S3 key prefix")
    s3_parser.set_defaults(func=cmd_ingest_s3)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
