import json
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage

from backend.app.core.config import get_settings
from src.langchain_components.embeddings import get_embeddings
from src.langchain_components.llm import get_llm
from src.langchain_components.vector_store import VectorStore

SYSTEM_PROMPT = (
    "You are a helpful document intelligence assistant. "
    "Use the available tools to search documents and answer questions. "
    "Be concise and provide relevant information from the documents."
)


class SmartMoveAgent:
    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.agent_model_name
        self.embeddings_model_name = settings.embeddings_model_name
        self._llm = get_llm(model_name=self.model_name, temperature=0.0, streaming=False)
        self._embeddings = get_embeddings(model_name=self.embeddings_model_name)
        self._vector_store = VectorStore(collection_name=settings.chroma_collection_name)
        self._tools = self._build_tools()
        self._agent = create_agent(
            model=self._llm,
            tools=self._tools,
            system_prompt=SYSTEM_PROMPT,
        )
    def _build_tools(self) -> List[Tool]:
        return [
            Tool(
                name="Search Documents",
                func=lambda q: self._vector_store.llm_enhanced_similarity_search(q, llm=self._llm),
                description="Search the document store for relevant information. Input is a search query string.",
            ),
            Tool(
                name="Add Document",
                func=lambda docs: self._vector_store.add_documents(docs),
                description="Add a new document to the vector store. Input is a list of dicts with 'text' and 'metadata' keys.",
            ),
        ]

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        self._vector_store.add_documents(documents)

    def _limit_prompt_tokens(self, text: str, max_tokens: int = 4000) -> str:
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])

    @staticmethod
    def calculate_tokens(text: str) -> int:
        return len(text.split())

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        limited_query = self._limit_prompt_tokens(query)
        result = await self._agent.ainvoke({"messages": [HumanMessage(content=limited_query)]})
        return {"answer": result.get("messages", [])[-1].content if result.get("messages") else "", "raw": result}

    async def aexecute_stream(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        available_tools: Optional[List[Any]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> AsyncGenerator[str, None]:
        streaming_llm = get_llm(model_name=self.model_name, temperature=0.0, streaming=True)
        streaming_agent = create_agent(
            model=streaming_llm,
            tools=self._tools,
            system_prompt=SYSTEM_PROMPT,
        )

        async for event in streaming_agent.astream(
            {"messages": [HumanMessage(content=query)]},
            stream_mode="messages",
        ):
            chunk, metadata = event
            if hasattr(chunk, "content") and chunk.content:
                yield json.dumps({"type": "token", "content": chunk.content})
