import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ExtractedData:
    amounts: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    po_numbers: List[str] = field(default_factory=list)
    suppliers: List[str] = field(default_factory=list)
    patterns: Dict[str, List[str]] = field(default_factory=dict)


AMOUNT_PATTERN = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b")
DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{2}-\d{2}-\d{4}\b"),
    re.compile(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},?\s?\d{4}", re.IGNORECASE),
]
PO_PATTERN = re.compile(
    r"(?:PO|P\.?O\.?|Purchase\s+Order)\s*[#:;]?\s*(\w*[\d/-]+\w*)",
    re.IGNORECASE,
)
SUPPLIER_PATTERN = re.compile(
    r"(?:Supplier|Vendor|Provider|From|Company)\s*:\s*(\w[\w\s&.,'-]+)",
    re.IGNORECASE,
)


def extract_dates(text: str) -> List[str]:
    dates = []
    for pattern in DATE_PATTERNS:
        dates.extend(pattern.findall(text))
    return list(set(dates))


def _is_in_date_match(pos: int, text: str, date_ranges: List[range]) -> bool:
    for r in date_ranges:
        if pos in r:
            return True
    return False


def extract_amounts(text: str) -> List[float]:
    date_ranges: List[range] = []
    for pattern in DATE_PATTERNS:
        for m in pattern.finditer(text):
            date_ranges.append(range(m.start(), m.end()))

    matches = AMOUNT_PATTERN.finditer(text)
    amounts = []
    for m in matches:
        if _is_in_date_match(m.start(), text, date_ranges):
            continue
        cleaned = m.group(0).replace(",", "")
        val = float(cleaned)
        if val >= 10.0:
            amounts.append(val)
    return amounts


def extract_po_numbers(text: str) -> List[str]:
    return PO_PATTERN.findall(text)


def extract_suppliers(text: str) -> List[str]:
    return [s.strip() for s in SUPPLIER_PATTERN.findall(text)]


def extract_patterns(text: str, custom_patterns: Optional[Dict[str, str]] = None) -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {}
    builtins = {
        "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
        "phone": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "invoice": r"\b(?:INVOICE|Invoice|INV|Inv)[\s#:;]*(\w[\w/-]+)",
    }
    if custom_patterns:
        builtins.update(custom_patterns)
    for name, pattern_str in builtins.items():
        matches = re.findall(pattern_str, text, re.IGNORECASE)
        if matches:
            results[name] = matches
    return results


def analyze_document(text: str, custom_patterns: Optional[Dict[str, str]] = None) -> ExtractedData:
    return ExtractedData(
        amounts=extract_amounts(text),
        dates=extract_dates(text),
        po_numbers=extract_po_numbers(text),
        suppliers=extract_suppliers(text),
        patterns=extract_patterns(text, custom_patterns),
    )


def generate_markdown_report(data: ExtractedData, title: str = "Analysis Report") -> str:
    lines = [f"# {title}", ""]
    if data.amounts:
        lines.extend(["## Amounts", "", *[f"- ${a:,.2f}" for a in data.amounts], ""])
    if data.dates:
        lines.extend(["## Dates", "", *[f"- {d}" for d in data.dates], ""])
    if data.po_numbers:
        lines.extend(["## PO Numbers", "", *[f"- {po}" for po in data.po_numbers], ""])
    if data.suppliers:
        lines.extend(["## Suppliers", "", *[f"- {s}" for s in data.suppliers], ""])
    if data.patterns:
        lines.extend(["## Patterns", ""])
        for name, matches in data.patterns.items():
            lines.append(f"### {name}")
            lines.append("")
            for m in matches:
                lines.append(f"- {m}")
            lines.append("")
    if not any([data.amounts, data.dates, data.po_numbers, data.suppliers, data.patterns]):
        lines.append("_No data extracted._")
    return "\n".join(lines)


def generate_json_report(data: ExtractedData) -> str:
    return json.dumps(asdict(data), indent=2, default=str)
