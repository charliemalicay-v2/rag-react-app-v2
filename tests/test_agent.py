from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ai_agent.agent import SmartMoveAgent


@pytest.fixture(autouse=True)
def mock_create_agent():
    with patch("src.ai_agent.agent.create_agent") as mock:
        yield mock


@pytest.mark.asyncio
async def test_agent_initialization():
    agent = SmartMoveAgent()
    assert agent.model_name == "minimax-m3:cloud"
    assert len(agent._tools) == 2


@pytest.mark.asyncio
async def test_agent_process_query():
    agent = SmartMoveAgent()
    agent._agent.ainvoke = AsyncMock(return_value={
        "messages": [MagicMock(content="test answer")]
    })

    result = await agent.process_query("test query")

    assert result["answer"] == "test answer"


@pytest.mark.asyncio
async def test_agent_process_query_empty_result():
    agent = SmartMoveAgent()
    agent._agent.ainvoke = AsyncMock(return_value={})

    result = await agent.process_query("test")

    assert result["answer"] == ""


@pytest.mark.asyncio
async def test_agent_streaming():
    agent = SmartMoveAgent()

    async def mock_astream(*args, **kwargs):
        yield MagicMock(content="test"), {}

    agent._agent.astream = mock_astream

    gen = agent.aexecute_stream("test query")
    chunks = [c async for c in gen]

    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_agent_add_documents():
    with patch.object(SmartMoveAgent, "_build_tools") as mock_build:
        mock_build.return_value = [], MagicMock()
        agent = SmartMoveAgent()
        vs_mock = MagicMock()
        agent._vector_store = vs_mock

        agent.add_documents([{"text": "test", "metadata": {"source": "test"}}])
        vs_mock.add_documents.assert_called_once()


def test_calculate_tokens():
    assert SmartMoveAgent.calculate_tokens("hello world") == 2
    assert SmartMoveAgent.calculate_tokens("") == 0
    assert SmartMoveAgent.calculate_tokens("a b c d e") == 5


def test_limit_prompt_tokens():
    agent = SmartMoveAgent()
    text = " ".join(["word"] * 100)
    limited = agent._limit_prompt_tokens(text, max_tokens=10)
    assert len(limited.split()) == 10
