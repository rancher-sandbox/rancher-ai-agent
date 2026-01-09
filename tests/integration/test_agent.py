from fastapi.testclient import TestClient
from app.main import app
from app.services.agent.agent import RANCHER_AGENT_PROMPT
from app.services.llm import LLMManager
from langchain_core.language_models import FakeMessagesListChatModel
from mcp.server.fastmcp import FastMCP
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage, SystemMessage
from _pytest.monkeypatch import MonkeyPatch
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

import logging
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
    logging.error("Starting mock MCP server...")
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

@pytest.fixture(scope="module")
def setup_mock_mcp_server(module_monkeypatch):
    """Sets up and tears down a mock MCP server for the duration of the test module."""
    module_monkeypatch.setenv("MCP_URL", "localhost:8000/mcp")
    module_monkeypatch.setenv("INSECURE_SKIP_TLS", "true")
    logging.debug("11Checking mock MCP server availability...")

    process = multiprocessing.Process(target=run_mock_mcp)
    process.start()

    # Wait for the mock server to be available before running tests.
    mcp_server_available = False
    max_attempts = 100  # 10 seconds total
    attempts = 0

    while not mcp_server_available and attempts < max_attempts:
        try:
            logging.error(f"Checking mock MCP server availability... (attempt {attempts + 1}/{max_attempts})")
            requests.get("http://localhost:8000/mcp", timeout=1)
            logging.error("Mock MCP server is available.")
            mcp_server_available = True
        except requests.exceptions.ConnectionError:
            logging.error("Mock MCP server not available yet, retrying...")
            attempts += 1
            time.sleep(0.1)
        except Exception as e:
            logging.error(f"Unexpected error checking MCP server: {e}")
            attempts += 1
            time.sleep(0.1)
    
    if not mcp_server_available:
        process.terminate()
        process.join(timeout=5)
        raise RuntimeError(f"Mock MCP server failed to start after {max_attempts} attempts")
    
    logging.error("Mock MCP server is ready!")
    yield process

    logging.debug("Shutting down mock MCP server...")
    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        logging.debug("Force killing mock MCP server...")
        process.kill()

@pytest.mark.parametrize(
        ("prompts", "fake_llm_responses", "expected_messages_send_to_llm", "expected_messages_send_to_websocket"),
        [
            (
                ["fake prompt"],
                [
                    AIMessage(
                        content="fake llm response",
                    ),
                ], 
                [
                    SystemMessage(content=RANCHER_AGENT_PROMPT),
                    HumanMessage(content="fake prompt"), 
                ],
                ["<message>fake llm response</message>"]
            ),
             (
                ["fake prompt 1",
                 "fake prompt 2"],
                [
                    AIMessage(
                        content="fake llm response 1",
                    ),
                     AIMessage(
                        content="fake llm response 2",
                    ),
                ], 
                [
                    SystemMessage(content=RANCHER_AGENT_PROMPT), 
                    HumanMessage(content="fake prompt 1"), 
                    AIMessage(
                        content="fake llm response 1",
                    ),
                    HumanMessage(content="fake prompt 2"), 
                ],
                ["<message>fake llm response 1</message>",
                 "<message>fake llm response 2</message>"]
            ),
             (
                ["sum 4 + 5"],
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
                    SystemMessage(content=RANCHER_AGENT_PROMPT), 
                    HumanMessage(content="sum 4 + 5"), 
                    AIMessage(content="", tool_calls=[{"id": "call_1", "name": "add", "args": {"a": 4, "b": 5}}]),
                    ToolMessage(content="sum is 9", name="add", tool_call_id="call_1")
                ],
                ["<message>fake llm response</message>"]
             )
        ]
)
def test_websocket_connection_and_agent_interaction(setup_mock_mcp_server, prompts: list[str], fake_llm_responses: list[BaseMessage], expected_messages_send_to_llm: list[BaseMessage | str], expected_messages_send_to_websocket: list[str]):
    """Tests the full agent interaction flow through a WebSocket connection."""
    fake_llm = FakeMessagesListChatModelWithTools(responses=fake_llm_responses)
    LLMManager._instance = fake_llm
    logging.debug("Starting test..")

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
        # Reset the LLM manager instance after each test
        LLMManager._instance = None
