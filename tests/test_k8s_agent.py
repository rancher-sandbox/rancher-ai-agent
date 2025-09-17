import pytest
from unittest.mock import AsyncMock, MagicMock
from langgraph.types import interrupt
from agents import create_k8s_agent
from langchain_core.messages import BaseMessage
from langchain_core.tools import StructuredTool

class FakeMessage:
    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls or []

class MockTool:
    def __init__(self, name, return_value):
        self.name = name
        self._return_value = return_value
        self.ainvoke = AsyncMock(return_value=return_value)

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.invoke = MagicMock(return_value="llm_response")
    return llm

fake_get_tool_execution_message = "fake k8s resource fetched"
fake_patch_tool_execution_message = "fake k8s resource patched"

@pytest.fixture
def mock_tools():
    return [MockTool("getKubernetesResource", fake_get_tool_execution_message), MockTool("patchKubernetesResource", fake_patch_tool_execution_message)]

def test_bind_tools(mock_llm, mock_tools):
    workflow = create_k8s_agent(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt"
    )

    assert workflow is not None
    mock_llm.bind_tools.assert_called_once_with(mock_tools)

@pytest.mark.asyncio
async def test_tool_node_executes_tool_human_verification_cancelled(mock_llm, mock_tools):
    workflow = create_k8s_agent(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt"
    )
    tool_node = workflow.nodes["tools"].runnable
    tool_call = {
        "id": "123",
        "name": "patchKubernetesResource",
        "args": {"namespace": "cattle-system", "name": "rancher", "kind": "Deployment", "cluster": "local", "patch": "[]"}
    }
    state = {"messages": [FakeMessage(tool_calls=[tool_call])]}

    interrupt_mock = MagicMock(return_value={"response": "no"})
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("langgraph.types.interrupt", interrupt_mock)

    result = await tool_node.ainvoke(state)

    assert result["messages"] == "the tool execution was cancelled by the user."

    monkeypatch.undo() 


@pytest.mark.asyncio
async def test_tool_node_executes_tool_human_verification_approved(mock_llm, mock_tools):
    workflow = create_k8s_agent(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt"
    )
    tool_node = workflow.nodes["tools"].runnable
    tool_call = {
        "id": "123",
        "name": "patchKubernetesResource",
        "args": {"namespace": "cattle-system", "name": "rancher", "kind": "Deployment", "cluster": "local", "patch": "[]"}
    }
    state = {"messages": [FakeMessage(tool_calls=[tool_call])]}

    interrupt_mock = MagicMock(return_value={"response": "yes"})
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("langgraph.types.interrupt", interrupt_mock)


    result = await tool_node.ainvoke(state)

    assert isinstance(result["messages"], list)
    assert result["messages"][0].name == "patchKubernetesResource"
    assert result["messages"][0].tool_call_id == "123"
    assert result["messages"][0].content == fake_patch_tool_execution_message

    monkeypatch.undo() 

@pytest.mark.asyncio
async def test_tool_node_executes_tool(mock_llm, mock_tools):
    workflow = create_k8s_agent(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt="system_prompt"
    )
    tool_node = workflow.nodes["tools"].runnable
    tool_call = {
        "id": "123",
        "name": "getKubernetesResource",
        "args": {"namespace": "cattle-system", "name": "rancher", "kind": "Deployment", "cluster": "local"}
    }
    state = {"messages": [FakeMessage(tool_calls=[tool_call])]}

    result = await tool_node.ainvoke(state)

    assert isinstance(result["messages"], list)
    assert result["messages"][0].name == "getKubernetesResource"
    assert result["messages"][0].tool_call_id == "123"
    assert result["messages"][0].content == fake_get_tool_execution_message

def test_call_model_includes_system_prompt(mock_llm, mock_tools):
    workflow = create_k8s_agent(
        llm=mock_llm,
        tools=[mock_tools],
        system_prompt="system_prompt"
    )
    call_model = workflow.nodes["agent"].runnable
    fake_message = FakeMessage()
    state = {"messages": [fake_message]}
    config = MagicMock()

    result = call_model.invoke(state, config)

    assert "llm_response" in result["messages"]
    mock_llm.invoke.assert_called_once_with(["system_prompt", fake_message], config)
 