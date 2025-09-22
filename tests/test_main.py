import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import WebSocketDisconnect

from main import (
    websocket_endpoint,
    is_context_message,
    get_llm,
)

class MockWebSocket:
    def __init__(self, messages=None):
        self.accepted = False
        self.closed = False
        self.cookies = {"R_SESS": "fake_token"}
        self.url = MagicMock()
        self.url.hostname = "fake.hostname"
        self.client = MagicMock()
        self.client.host = "fake_client_host"
        self._receive_queue = messages or []
        self._send_queue = []

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        if not self._receive_queue:
            raise WebSocketDisconnect("No more messages")
        return self._receive_queue.pop(0)

    async def send_text(self, data):
        self._send_queue.append(data)

    async def close(self, code=1000):
        self.closed = True

@pytest.mark.parametrize("message, expected", [
    ("<mcp_context>cluster=local</mcp_context>", ("cluster", "local")),
    ("just a normal message", None),
])
def test_is_context_message(message, expected):
    assert is_context_message(message) == expected

@patch('main.ChatOllama')
def test_get_llm_ollama(mock_chat_ollama):
    with patch.dict(os.environ, {"MODEL": "test-model", "OLLAMA_URL": "http://localhost:11434"}, clear=True):
        llm = get_llm()
        mock_chat_ollama.assert_called_once_with(model="test-model", base_url="http://localhost:11434")
        assert llm == mock_chat_ollama.return_value

@patch('main.ChatGoogleGenerativeAI')
def test_get_llm_gemini(mock_chat_gemini):
    with patch.dict(os.environ, {"MODEL": "gemini-pro", "GOOGLE_API_KEY": "fake-key"}, clear=True):
        llm = get_llm()
        mock_chat_gemini.assert_called_once_with(model="gemini-pro")
        assert llm == mock_chat_gemini.return_value

@patch('main.OpenAI')
def test_get_llm_openai(mock_openai):
    with patch.dict(os.environ, {"MODEL": "gpt-4", "OPENAI_API_KEY": "fake-key"}, clear=True):
        llm = get_llm()
        mock_openai.assert_called_once_with(model="gpt-4")
        assert llm == mock_openai.return_value

def test_get_llm_no_model():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Model not configured."):
            get_llm()

def test_get_llm_no_provider():
    with patch.dict(os.environ, {"MODEL": "some-model"}, clear=True):
        with pytest.raises(ValueError, match="LLM not configured."):
            get_llm()

@pytest.fixture
def mock_dependencies():
    with patch('main.streamablehttp_client') as mock_streamable_http, \
         patch('main.ClientSession') as mock_client_session, \
         patch('main.load_mcp_tools', new_callable=AsyncMock) as mock_load_tools, \
         patch('main.create_k8s_agent') as mock_create_agent, \
         patch('main.get_system_prompt', return_value="fake_prompt"), \
         patch('main.stream_agent_response', new_callable=AsyncMock) as mock_stream_response, \
         patch('main.init_config', {"llm": "fake_llm"}), \
         patch('main.logging') as mock_logging:

        mock_streamable_http.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock(), None)
        mock_compiled_agent = MagicMock()
        mock_agent = MagicMock()
        mock_agent.compile.return_value = mock_compiled_agent
        mock_create_agent.return_value = mock_agent
        mock_session = AsyncMock()
        mock_client_session.return_value.__aenter__.return_value = mock_session
        mock_load_tools.return_value = ["fake_tool"]

        yield {
            "streamablehttp_client": mock_streamable_http,
            "client_session": mock_session,
            "load_mcp_tools": mock_load_tools,
            "create_k8s_agent": mock_create_agent,
            "compiled_agent": mock_compiled_agent,
            "stream_agent_response": mock_stream_response,
            "logging": mock_logging,
        }

@pytest.mark.asyncio
async def test_websocket_endpoint(mock_dependencies):
    mock_ws = MockWebSocket(messages=["test message"])

    await websocket_endpoint(mock_ws)

    assert mock_ws.accepted
    mock_dependencies["streamablehttp_client"].assert_called_once_with(
        url="http://rancher-mcp-server",
        headers={
            "R_token": "fake_token",
            "R_url": "https://fake.hostname"
        }
    )
    mock_dependencies["client_session"].initialize.assert_awaited_once()
    mock_dependencies["load_mcp_tools"].assert_awaited_once_with(mock_dependencies["client_session"])
    mock_dependencies["create_k8s_agent"].assert_called_once_with("fake_llm", ["fake_tool"], system_prompt="fake_prompt")
    mock_dependencies["create_k8s_agent"].return_value.compile.assert_called_once()
    mock_dependencies["stream_agent_response"].assert_awaited_once()
    call_kwargs = mock_dependencies["stream_agent_response"].call_args.kwargs
    assert call_kwargs['agent'] == mock_dependencies["compiled_agent"]
    assert call_kwargs['input_data'] == {"messages": [{"role": "user", "content": "test message"}]}
    assert call_kwargs['config'] == {"thread_id": "thread-1"}
    assert call_kwargs['websocket'] == mock_ws
    assert call_kwargs['context'] == {}

    assert not mock_ws.closed

@pytest.mark.asyncio
async def test_websocket_endpoint_context_message(mock_dependencies):
    mock_ws = MockWebSocket(messages=[
        "<mcp_context>cluster=local</mcp_context>",
        "hello agent in local cluster"
    ])

    await websocket_endpoint(mock_ws)

    mock_dependencies["stream_agent_response"].assert_awaited_once()
    call_kwargs = mock_dependencies["stream_agent_response"].call_args.kwargs
    assert call_kwargs['agent'] == mock_dependencies["compiled_agent"]
    assert call_kwargs['input_data'] == {"messages": [{"role": "user", "content": "hello agent in local cluster"}]}
    assert call_kwargs['context'] == {"cluster": "local"} #
