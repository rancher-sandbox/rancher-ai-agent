import pytest

from app.routers.websocket import websocket_endpoint
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from fastapi import WebSocketDisconnect

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

# TODO add more tests for different scenarios
@pytest.fixture
def mock_dependencies():
    with patch('app.routers.websocket.create_agent', new_callable=AsyncMock) as mock_create_agent, \
         patch('app.routers.websocket.stream_agent_response', new_callable=AsyncMock) as mock_stream_response:

        # Mock the create_agent to return (agent, session, client_ctx)
        mock_agent = MagicMock()
        mock_agent.astream_events = AsyncMock(return_value=iter([]))
        mock_session = MagicMock()
        mock_session.__aexit__ = AsyncMock()
        mock_client_ctx = MagicMock()
        mock_client_ctx.__aexit__ = AsyncMock()
        
        mock_create_agent.return_value = (mock_agent, mock_session, mock_client_ctx)

        yield {
            "create_agent": mock_create_agent,
            "agent": mock_agent,
            "session": mock_session,
            "client_ctx": mock_client_ctx,
            "stream_agent_response": mock_stream_response,
        }

@pytest.mark.asyncio
async def test_websocket_endpoint(mock_dependencies):
    mock_ws = MockWebSocket(messages=["test message"])
    mock_llm = MagicMock()

    await websocket_endpoint(mock_ws, mock_llm)

    assert mock_ws.accepted
    mock_dependencies["create_agent"].assert_awaited_once_with(llm=mock_llm, websocket=mock_ws)
    mock_dependencies["stream_agent_response"].assert_awaited_once()
    
    call_kwargs = mock_dependencies["stream_agent_response"].call_args.kwargs
    assert call_kwargs['input_data'] == {"messages": [{"role": "user", "content": "test message"}]}
    assert call_kwargs['websocket'] == mock_ws
    assert call_kwargs['agent'] == mock_dependencies["agent"]
    
    # Verify cleanup was called
    mock_dependencies["session"].__aexit__.assert_awaited_once()
    mock_dependencies["client_ctx"].__aexit__.assert_awaited_once()

@pytest.mark.asyncio
async def test_websocket_endpoint_context_message(mock_dependencies):
    mock_ws = MockWebSocket(messages=[
        '{"prompt": "show all pods", "context": { "namespace": "default", "cluster": "local"} }'
    ])
    mock_llm = MagicMock()

    await websocket_endpoint(mock_ws, mock_llm)

    mock_dependencies["create_agent"].assert_awaited_once_with(llm=mock_llm, websocket=mock_ws)
    mock_dependencies["stream_agent_response"].assert_awaited_once()
    
    call_kwargs = mock_dependencies["stream_agent_response"].call_args.kwargs
    expected_prompt = "show all pods. Use the following parameters to populate tool calls when appropriate. \n Only include parameters relevant to the user's request (e.g., omit namespace for cluster-wide operations). \n Parameters (separated by ;): \n namespace:default;cluster:local;"
    assert call_kwargs['input_data'] == {"messages": [{"role": "user", "content": expected_prompt}]}
    assert call_kwargs['websocket'] == mock_ws
