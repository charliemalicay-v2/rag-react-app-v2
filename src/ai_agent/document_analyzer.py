import asyncio
import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from src.ai_agent.agent import SmartMoveAgent
from src.document_processing.detailed_analysis import analyze_document, ExtractedData


CLASSIFICATION_PROMPT = (
    "Classify the following document into one or more of these categories: "
    "invoice, contract, report, email, memo, proposal, receipt, other. "
    "Return only the category names as a comma-separated list.\n\n{document}"
)

SUMMARIZE_PROMPT = (
    "Provide a concise summary of the following document in 2-3 sentences.\n\n{document}"
)

ENTITIES_PROMPT = (
    "Extract key entities from the following document. "
    "Return them as a JSON object with keys: people, organizations, locations, dates, amounts.\n\n{document}"
)


class DocumentAnalyzer:
    def __init__(self, agent: Optional[SmartMoveAgent] = None) -> None:
        self._agent = agent or SmartMoveAgent()

    async def classify(self, document: str) -> List[str]:
        prompt = CLASSIFICATION_PROMPT.format(document=document[:4000])
        result = await self._agent.process_query(prompt)
        text = result.get("answer", "")
        return [c.strip().lower() for c in text.split(",") if c.strip()]

    async def summarize(self, document: str) -> str:
        prompt = SUMMARIZE_PROMPT.format(document=document[:4000])
        result = await self._agent.process_query(prompt)
        return result.get("answer", "")

    async def extract_entities(self, document: str) -> Dict[str, Any]:
        prompt = ENTITIES_PROMPT.format(document=document[:4000])
        result = await self._agent.process_query(prompt)
        raw = result.get("answer", "")
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"raw": raw}

    def extract_structured(self, document: str) -> ExtractedData:
        return analyze_document(document)

    async def full_analysis(self, document: str) -> Dict[str, Any]:
        structured = self.extract_structured(document)
        classification, summary, entities = await asyncio.gather(
            self.classify(document),
            self.summarize(document),
            self.extract_entities(document),
        )
        return {
            "classification": classification,
            "summary": summary,
            "entities": entities,
            "extracted": {
                "amounts": structured.amounts,
                "dates": structured.dates,
                "po_numbers": structured.po_numbers,
                "suppliers": structured.suppliers,
                "patterns": structured.patterns,
            },
        }
