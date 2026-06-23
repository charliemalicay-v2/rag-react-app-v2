from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.app.core.config import get_settings


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_llm_openai(model_name: Optional[str] = None, temperature: float = 0.0, streaming: bool = False) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=model_name or settings.agent_model_name,
        temperature=temperature,
        streaming=streaming,
        api_key=settings.openai_api_key,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_llm_claude(model_name: str = "claude-3-haiku-20240307", temperature: float = 0.0, streaming: bool = False) -> ChatAnthropic:
    return ChatAnthropic(
        model=model_name,
        temperature=temperature,
        streaming=streaming,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_llm_ollama(model_name: str = "minimax-m3:cloud", temperature: float = 0.0, streaming: bool = False) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        streaming=streaming,
    )


def get_llm(model_name: Optional[str] = None, temperature: float = 0.0, streaming: bool = False) -> Any:
    settings = get_settings()
    if model_name is None:
        model_name = settings.agent_model_name
    if "gpt" in model_name or "o1" in model_name or "o3" in model_name:
        return get_llm_openai(model_name, temperature, streaming)
    if "claude" in model_name:
        return get_llm_claude(model_name, temperature, streaming)
    return get_llm_ollama(model_name, temperature, streaming)
