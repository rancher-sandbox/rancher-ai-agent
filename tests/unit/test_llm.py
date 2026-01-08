import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi import WebSocketDisconnect

from app.services.llm import (
    get_llm,
)

@patch('app.services.llm.ChatOllama')
def test_get_llm_ollama(mock_chat_ollama):
    with patch.dict(os.environ, {"MODEL": "test-model", "OLLAMA_URL": "http://localhost:11434"}, clear=True):
        llm = get_llm()
        mock_chat_ollama.assert_called_once_with(model="test-model", base_url="http://localhost:11434")
        assert llm == mock_chat_ollama.return_value

@patch('app.services.llm.ChatGoogleGenerativeAI')
def test_get_llm_gemini(mock_chat_gemini):
    with patch.dict(os.environ, {"MODEL": "gemini-pro", "GOOGLE_API_KEY": "fake-key"}, clear=True):
        llm = get_llm()
        mock_chat_gemini.assert_called_once_with(model="gemini-pro")
        assert llm == mock_chat_gemini.return_value

@patch('app.services.llm.ChatOpenAI')
def test_get_llm_openai(mock_openai):
    with patch.dict(os.environ, {"MODEL": "gpt-4", "OPENAI_API_KEY": "fake-key"}, clear=True):
        llm = get_llm()
        mock_openai.assert_called_once_with(model="gpt-4")
        assert llm == mock_openai.return_value

def test_get_llm_no_model():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="LLM Model not configured."):
            get_llm()

def test_get_llm_no_provider():
    with patch.dict(os.environ, {"MODEL": "some-model"}, clear=True):
        with pytest.raises(ValueError, match="LLM not configured."):
            get_llm()
        
def test_get_llm_invalid_active_model():
    with patch.dict(os.environ, {"MODEL": "some-model", "ACTIVE_LLM": "invalid-llm"}, clear=True):
        with pytest.raises(ValueError, match="Unsupported Active LLM specified."):
            get_llm()
