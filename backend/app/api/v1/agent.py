from typing import Any, AsyncGenerator, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.app.chains import create_chat_agent_chain, create_streaming_agent_chain
from backend.app.core.cache import cache_response
from backend.app.core.rate_limit import RateLimiter

router = APIRouter()


class AgentRequest(BaseModel):
    query: str
    context: Dict[str, Any] = {}


class AgentResponse(BaseModel):
    query: str
    results: Any


@router.post("/v1/agent", response_model=AgentResponse)
@cache_response(ttl_seconds=30)
async def execute_agent(
    request: AgentRequest,
    _rate_limit: None = Depends(RateLimiter()),
) -> Dict[str, Any]:
    agent = create_chat_agent_chain()
    try:
        result = await agent.process_query(request.query, context=request.context)
        return {"query": request.query, "results": result}
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post("/v1/agent/stream")
async def execute_agent_stream(
    request: AgentRequest,
    _rate_limit: None = Depends(RateLimiter()),
) -> StreamingResponse:
    agent = create_streaming_agent_chain()

    async def event_generator() -> AsyncGenerator[str, None]:
        async for chunk in agent.aexecute_stream(request.query, context=request.context):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
