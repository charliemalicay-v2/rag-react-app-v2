from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.app.chains import create_chat_agent_chain

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    context: Dict[str, Any] = {}


class ChatResponse(BaseModel):
    answer: str


def create_chat_chain() -> Any:
    return create_chat_agent_chain()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> Dict[str, str]:
    agent = create_chat_chain()
    try:
        result = await agent.process_query(request.query, context=request.context)
        return {"answer": result.get("answer", "")}
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
