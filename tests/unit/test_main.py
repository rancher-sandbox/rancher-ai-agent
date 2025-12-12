import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from fastapi import WebSocketDisconnect

from src.main import (
    websocket_endpoint,
    get_llm,
)

class MockWebSocket:
    def __init__(self, messages=None):
        self.accepted = False
        self.closed = False
        self.cookies = {"R_SESS": "fake_token"}
        self.url = MagicMock()
        self.url.hostname = "fake.hostname"
        self.url.port = None
        self.client = MagicMock()
        self.client.host = "fake_client_host"
        self.client_state = "connected"
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

    async def close(self):
        self.closed = True

@patch('src.main.ChatOllama')
def test_get_llm_ollama(mock_chat_ollama):
    with patch.dict(os.environ, {"MODEL": "test-model", "OLLAMA_URL": "http://localhost:11434"}, clear=True):
        llm = get_llm()
        mock_chat_ollama.assert_called_once_with(model="test-model", base_url="http://localhost:11434")
        assert llm == mock_chat_ollama.return_value

@patch('src.main.ChatGoogleGenerativeAI')
def test_get_llm_gemini(mock_chat_gemini):
    with patch.dict(os.environ, {"MODEL": "gemini-pro", "GOOGLE_API_KEY": "fake-key"}, clear=True):
        llm = get_llm()
        mock_chat_gemini.assert_called_once_with(model="gemini-pro")
        assert llm == mock_chat_gemini.return_value

@patch('src.main.ChatOpenAI')
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

@pytest.fixture
def mock_dependencies():
    with patch('src.main.streamablehttp_client') as mock_streamable_http, \
         patch('src.main.ClientSession') as mock_client_session, \
         patch('src.main.load_mcp_tools', new_callable=AsyncMock) as mock_load_tools, \
         patch('src.main.create_k8s_agent') as mock_create_agent, \
         patch('src.main.get_system_prompt', return_value="fake_prompt"), \
         patch('src.main.stream_agent_response', new_callable=AsyncMock) as mock_stream_response, \
         patch('src.main.init_config', {"llm": "fake_llm"}), \
         patch('src.main.logging') as mock_logging:

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
        url="https://rancher-mcp-server.cattle-ai-agent-system.svc",
        headers={
            "R_token": "fake_token",
            "R_url": "https://fake.hostname"
        }
    )
    mock_dependencies["client_session"].initialize.assert_awaited_once()
    mock_dependencies["load_mcp_tools"].assert_awaited_once_with(mock_dependencies["client_session"])
    mock_dependencies["create_k8s_agent"].assert_called_once_with("fake_llm", ["fake_tool"], "fake_prompt", ANY)
    mock_dependencies["stream_agent_response"].assert_awaited_once()
    call_kwargs = mock_dependencies["stream_agent_response"].call_args.kwargs
    assert call_kwargs['input_data'] == {"messages": [{"role": "user", "content": "test message"}]}
    assert call_kwargs['websocket'] == mock_ws

    assert not mock_ws.closed

@pytest.mark.asyncio
async def test_websocket_endpoint_context_message(mock_dependencies):
    mock_ws = MockWebSocket(messages=[
        '{"prompt": "show all pods", "context": { "namespace": "default", "cluster": "local"} }'
    ])

    await websocket_endpoint(mock_ws)

    mock_dependencies["client_session"].initialize.assert_awaited_once()
    mock_dependencies["load_mcp_tools"].assert_awaited_once_with(mock_dependencies["client_session"])
    mock_dependencies["create_k8s_agent"].assert_called_once_with("fake_llm", ["fake_tool"], "fake_prompt", ANY)
    mock_dependencies["stream_agent_response"].assert_awaited_once()
    call_kwargs = mock_dependencies["stream_agent_response"].call_args.kwargs
    assert call_kwargs['input_data'] == {"messages": [{"role": "user", "content": "show all pods. Use the following parameters to populate tool calls when appropriate. \n Only include parameters relevant to the user’s request (e.g., omit namespace for cluster-wide operations). \n Parameters (separated by ;): \n namespace:default;cluster:local;"}]}
    assert call_kwargs['websocket'] == mock_ws

