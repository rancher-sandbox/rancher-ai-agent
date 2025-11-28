from fastapi.testclient import TestClient
from src.main import app
from src.main import init_config, get_system_prompt
from langchain_core.language_models import FakeMessagesListChatModel
from mcp.server.fastmcp import FastMCP
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from _pytest.monkeypatch import MonkeyPatch
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.tools import BaseTool

import time
import multiprocessing
import requests
import pytest

mock_mcp = FastMCP("mock")


@mock_mcp.tool()
def add(a: int, b: int) -> str:
    """Add two numbers"""
    return f"sum is {a + b}"

def run_mock_mcp():
    """Runs the mock MCP server."""
    mock_mcp.run(transport="streamable-http")

client = TestClient(app)

class FakeMessagesListChatModelWithTools(FakeMessagesListChatModel):
    """
    A fake chat model that extends FakeMessagesListChatModel to support tool binding
    and capture the messages sent to the LLM for inspection in tests.
    """
    tools: list[BaseTool] = None
    messages_send_to_llm: LanguageModelInput = None

    def bind_tools(self, tools):
        self.tools = tools
        return self
    
    def invoke(self, input, config = None, *, stop = None, **kwargs):
        # Capture the input messages before invoking the parent method.
        self.messages_send_to_llm = remove_message_ids(input)
        return super().invoke(input, config, stop=stop, **kwargs)

def remove_message_ids(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Creates a new list of BaseMessage objects with the 'id' field removed
    from each message.
    """
    new_messages = []
    
    for message in messages:
        # Create a copy of the message, explicitly setting the 'id' field to None for consistent comparison.
        if isinstance(message, BaseMessage):
            new_message = message.model_copy(update={"id": None})
        else:
            new_message = message
        
        new_messages.append(new_message)
        
    return new_messages

@pytest.fixture(scope="module")
def module_monkeypatch(request):
    """
    A module-scoped version of the monkeypatch fixture.
    This fixture ensures that patches persist for the duration of the module,
    and cleanup happens only once at the end of the module.
    """
    mpatch = MonkeyPatch()

    yield mpatch

    mpatch.undo()

@pytest.fixture(scope="module", autouse=True)
def setup_mock_mcp_server(module_monkeypatch):
    """Sets up and tears down a mock MCP server for the duration of the test module."""
    module_monkeypatch.setenv("MCP_URL", "localhost:8000/mcp")
    module_monkeypatch.setenv("INSECURE_SKIP_TLS", "true")

    process = multiprocessing.Process(target=run_mock_mcp)
    process.start()

    # Wait for the mock server to be available before running tests.
    mcp_server_available = False

    while not mcp_server_available:
        try:
            requests.get("http://localhost:8000/mcp")
            mcp_server_available = True
        except requests.exceptions.ConnectionError:
            time.sleep(0.1)
       
    yield process

    process.terminate()

@pytest.mark.parametrize(
        ("fake_llm_responses", "expected_messages_send_to_llm", "expected_messages_send_to_websocket"),
        [
            (
                [
                    AIMessage(
                        content="", # The content is empty when a tool is called
                        tool_calls=[{
                            "id": "call_1",
                            "name": "add",
                            "args": {"a": 4, "b": 5}
                        }]
                    ),
                    AIMessage(
                        content="fake llm response",
                    ),
                ], 
                [
                    get_system_prompt(), 
                    HumanMessage(content="call add tool"), 
                    AIMessage(content="", tool_calls=[{"id": "call_1", "name": "add", "args": {"a": 4, "b": 5}}]),
                    ToolMessage(content="sum is 9", name="add", tool_call_id="call_1")
                ],
                ["<message>fake llm response</message>"]
             )
        ]
)
def test_websocket_connection_and_agent_interaction(fake_llm_responses: list[BaseMessage], expected_messages_send_to_llm: list[BaseMessage | str], expected_messages_send_to_websocket: list[str]):
    """Tests the full agent interaction flow through a WebSocket connection."""
    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    init_config["llm"] = fake_llm
    
    messages = []
    with client.websocket_connect("/agent/ws") as websocket:
        websocket.send_text("call add tool")
        
        receives_text = websocket.receive_text()
        msg = receives_text
        while not receives_text == "</message>":
            receives_text = websocket.receive_text()
            msg += receives_text
        messages.append(msg)
        
    assert messages == expected_messages_send_to_websocket
    assert expected_messages_send_to_llm == fake_llm.messages_send_to_llm
