from fastapi.testclient import TestClient
from app.main import app
from app.services.agent.factory import RANCHER_AGENT_PROMPT
from app.services.llm import LLMManager
from langchain_core.language_models import FakeMessagesListChatModel
from mcp.server.fastmcp import FastMCP
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage, SystemMessage
from _pytest.monkeypatch import MonkeyPatch
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

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
    _current_response_index: int = 0

    def bind_tools(self, tools):
        self.tools = tools
        return self
    
    def invoke(self, input, config = None, *, stop = None, **kwargs):
        # Capture the input messages before invoking the parent method.
        self.messages_send_to_llm = remove_message_ids(input)
        return super().invoke(input, config, stop=stop, **kwargs)
    
    def _stream(self, input, config = None, *, stop = None, **kwargs):
        """Override _stream to yield chunks from the response."""        
        # Capture the input messages
        self.messages_send_to_llm = remove_message_ids(input)
        
        # Get the next response from the list
        if self._current_response_index < len(self.responses):
            response = self.responses[self._current_response_index]
            self._current_response_index += 1
            
            # Create a message chunk
            if hasattr(response, 'content') and response.content:
                chunk = AIMessageChunk(
                    content=response.content, 
                    tool_calls=response.tool_calls if hasattr(response, 'tool_calls') else [],
                    id=response.id if hasattr(response, 'id') else None
                )
            elif hasattr(response, 'tool_calls') and response.tool_calls:
                chunk = AIMessageChunk(
                    content="", 
                    tool_calls=response.tool_calls,
                    id=response.id if hasattr(response, 'id') else None
                )
            else:
                chunk = AIMessageChunk(content="")
            
            # Wrap in ChatGenerationChunk
            yield ChatGenerationChunk(message=chunk)


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

def test_websocket_simple_prompt():
    """Tests a simple prompt-response interaction."""
    prompts = ["fake prompt"]
    fake_llm_responses = [
        AIMessage(content="fake llm response"),
    ]
    expected_messages_send_to_llm = [
        SystemMessage(content=RANCHER_AGENT_PROMPT),
        HumanMessage(content="fake prompt"),
    ]
    expected_messages_send_to_websocket = ["<message>fake llm response</message>"]
    
    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    LLMManager._instance = fake_llm
    
    try:
        messages = []
        with client.websocket_connect("/agent/ws") as websocket:
            for prompt in prompts:
                websocket.send_text(prompt)
                msg = ""
                while not msg.endswith("</message>"):
                    msg += websocket.receive_text()
                messages.append(msg)
            
        assert messages == expected_messages_send_to_websocket
        assert expected_messages_send_to_llm == fake_llm.messages_send_to_llm
    finally:
        LLMManager._instance = None

def test_websocket_multiple_prompts():
    """Tests multiple prompt-response interactions in sequence."""
    prompts = ["fake prompt 1", "fake prompt 2"]
    fake_llm_responses = [
        AIMessage(content="fake llm response 1"),
        AIMessage(content="fake llm response 2"),
    ]
    expected_messages_send_to_llm = [
        SystemMessage(content=RANCHER_AGENT_PROMPT),
        HumanMessage(content="fake prompt 1"),
        AIMessage(content="fake llm response 1"),
        HumanMessage(content="fake prompt 2"),
    ]
    expected_messages_send_to_websocket = [
        "<message>fake llm response 1</message>",
        "<message>fake llm response 2</message>"
    ]
    
    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    LLMManager._instance = fake_llm
    
    try:
        messages = []
        with client.websocket_connect("/agent/ws") as websocket:
            for prompt in prompts:
                websocket.send_text(prompt)
                msg = ""
                while not msg.endswith("</message>"):
                    msg += websocket.receive_text()
                messages.append(msg)
            
        assert messages == expected_messages_send_to_websocket
        assert expected_messages_send_to_llm == fake_llm.messages_send_to_llm
    finally:
        LLMManager._instance = None

def test_websocket_tool_call():
    """Tests agent interaction with tool calling."""
    prompts = ["sum 4 + 5"]
    fake_llm_responses = [
        AIMessage(
            content="",
            tool_calls=[{
                "id": "call_1",
                "name": "add",
                "args": {"a": 4, "b": 5}
            }]
        ),
        AIMessage(content="fake llm response"),
    ]
    expected_messages_send_to_llm = [
        SystemMessage(content=RANCHER_AGENT_PROMPT),
        HumanMessage(content="sum 4 + 5"),
        AIMessage(content="", tool_calls=[{"id": "call_1", "name": "add", "args": {"a": 4, "b": 5}}]),
        ToolMessage(content="sum is 9", name="add", tool_call_id="call_1")
    ]
    expected_messages_send_to_websocket = ["<message>fake llm response</message>"]
    
    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    LLMManager._instance = fake_llm
    
    try:
        messages = []
        with client.websocket_connect("/agent/ws") as websocket:
            for prompt in prompts:
                websocket.send_text(prompt)
                msg = ""
                while not msg.endswith("</message>"):
                    msg += websocket.receive_text()
                messages.append(msg)
            
        assert messages == expected_messages_send_to_websocket
        assert expected_messages_send_to_llm == fake_llm.messages_send_to_llm
    finally:
        LLMManager._instance = None
