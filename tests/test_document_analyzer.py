import pytest

from src.document_processing.detailed_analysis import (
    ExtractedData,
    analyze_document,
    extract_amounts,
    extract_dates,
    extract_po_numbers,
    extract_suppliers,
    extract_patterns,
    generate_markdown_report,
    generate_json_report,
)


def test_extract_amounts():
    assert extract_amounts("Total: $1,234.56") == [1234.56]
    assert extract_amounts("Amount: 500.00 and 1,000") == [500.0, 1000.0]
    assert extract_amounts("No numbers here") == []
    assert extract_amounts("Price: 99") == [99.0]
    assert extract_amounts("Year 2024") == []


def test_extract_amounts_filters_dates():
    assert extract_amounts("Date: 2024-03-15") == []


def test_extract_dates():
    dates = extract_dates("Issued 2024-03-15 and due 04/30/2024")
    assert "2024-03-15" in dates
    assert "04/30/2024" in dates


def test_extract_dates_empty():
    assert extract_dates("No dates here") == []


def test_extract_po_numbers():
    assert extract_po_numbers("PO: PO-12345") == ["PO-12345"]
    assert extract_po_numbers("P.O. # 98765") == ["98765"]
    assert extract_po_numbers("Purchase Order 555") == ["555"]


def test_extract_po_numbers_empty():
    assert extract_po_numbers("No PO here") == []


def test_extract_po_numbers_case():
    assert extract_po_numbers("po# 123") == ["123"]
    assert extract_po_numbers("Purchase Order ABC-999") == ["ABC-999"]


def test_extract_suppliers():
    assert extract_suppliers("Supplier: Acme Corp") == ["Acme Corp"]
    assert extract_suppliers("Vendor: Widgets Inc") == ["Widgets Inc"]


def test_extract_suppliers_various_labels():
    assert extract_suppliers("Company: Test LLC") == ["Test LLC"]
    assert extract_suppliers("From: Big Corp") == ["Big Corp"]


def test_extract_suppliers_no_match():
    assert extract_suppliers("No supplier listed") == []


def test_extract_patterns():
    patterns = extract_patterns("Email: test@example.com")
    assert "email" in patterns
    assert "test@example.com" in patterns["email"]


def test_extract_patterns_phone():
    patterns = extract_patterns("Call 555-123-4567")
    assert "phone" in patterns


def test_extract_patterns_invoice():
    patterns = extract_patterns("INVOICE #INV-2024-001")
    assert "invoice" in patterns
    assert patterns["invoice"] == ["INV-2024-001"]


def test_extract_patterns_custom():
    patterns = extract_patterns("Ref: ABC-123", custom_patterns={"ref": r"REF:\s*(\w+-\d+)"})
    assert "ref" in patterns


def test_analyze_document():
    text = """INVOICE #INV-001
    Supplier: Test Corp
    Date: 2024-03-15
    PO: PO-123
    Amount: $1,500.00
    Contact: bob@test.com
    """
    data = analyze_document(text)
    assert isinstance(data, ExtractedData)
    assert len(data.amounts) > 0
    assert len(data.dates) > 0
    assert len(data.po_numbers) > 0
    assert len(data.suppliers) > 0
    assert "email" in data.patterns


def test_generate_markdown_report():
    data = ExtractedData(
        amounts=[1500.0],
        dates=["2024-03-15"],
        po_numbers=["PO-123"],
        suppliers=["Test Corp"],
        patterns={"email": ["bob@test.com"]},
    )
    report = generate_markdown_report(data)
    assert "# Analysis Report" in report
    assert "$1,500.00" in report
    assert "2024-03-15" in report
    assert "PO-123" in report
    assert "Test Corp" in report
    assert "bob@test.com" in report


def test_generate_markdown_report_empty():
    data = ExtractedData()
    report = generate_markdown_report(data)
    assert "_No data extracted._" in report


def test_generate_json_report():
    data = ExtractedData(amounts=[100.0])
    report = generate_json_report(data)
    assert '"amounts"' in report
    assert "100.0" in report
