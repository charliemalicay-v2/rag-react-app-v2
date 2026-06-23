from unittest.mock import MagicMock, patch

import pytest


def test_get_llm_openai():
    with patch("src.langchain_components.llm.get_settings") as mock_settings:
        settings = MagicMock()
        settings.agent_model_name = "gpt-3.5-turbo"
        settings.openai_api_key = "test-key"
        mock_settings.return_value = settings

        with patch("src.langchain_components.llm.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            from src.langchain_components.llm import get_llm_openai
            llm = get_llm_openai()
            mock_chat.assert_called_once()
            assert llm is not None


def test_get_llm_claude():
    with patch("src.langchain_components.llm.ChatAnthropic") as mock_chat:
        mock_chat.return_value = MagicMock()
        from src.langchain_components.llm import get_llm_claude
        llm = get_llm_claude()
        mock_chat.assert_called_once()
        assert llm is not None


def test_get_llm_routes_to_openai():
    with patch("src.langchain_components.llm.get_settings") as mock_settings:
        settings = MagicMock()
        settings.agent_model_name = "gpt-4"
        settings.openai_api_key = "test-key"
        mock_settings.return_value = settings

        with patch("src.langchain_components.llm.get_llm_openai") as mock_openai:
            mock_openai.return_value = MagicMock()
            from src.langchain_components.llm import get_llm
            llm = get_llm()
            mock_openai.assert_called_once()


def test_get_llm_routes_to_claude():
    with patch("src.langchain_components.llm.get_settings") as mock_settings:
        settings = MagicMock()
        settings.agent_model_name = "claude-3-haiku-20240307"
        settings.openai_api_key = "test-key"
        mock_settings.return_value = settings

        with patch("src.langchain_components.llm.get_llm_claude") as mock_claude:
            mock_claude.return_value = MagicMock()
            from src.langchain_components.llm import get_llm
            llm = get_llm("claude-3-haiku-20240307")
            mock_claude.assert_called_once()


def test_get_llm_default_model():
    with patch("src.langchain_components.llm.get_settings") as mock_settings:
        settings = MagicMock()
        settings.agent_model_name = "gpt-4"
        settings.openai_api_key = "test-key"
        mock_settings.return_value = settings

        with patch("src.langchain_components.llm.get_llm_openai") as mock_openai:
            mock_openai.return_value = MagicMock()
            from src.langchain_components.llm import get_llm
            llm = get_llm()
            mock_openai.assert_called_once_with("gpt-4", 0.0, False)


def test_get_llm_default_ollama():
    with patch("src.langchain_components.llm.get_settings") as mock_settings:
        settings = MagicMock()
        settings.agent_model_name = "minimax-m3:cloud"
        settings.openai_api_key = "test-key"
        mock_settings.return_value = settings

        with patch("src.langchain_components.llm.get_llm_ollama") as mock_ollama:
            mock_ollama.return_value = MagicMock()
            from src.langchain_components.llm import get_llm
            llm = get_llm()
            mock_ollama.assert_called_once_with("minimax-m3:cloud", 0.0, False)
