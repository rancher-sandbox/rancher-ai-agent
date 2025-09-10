from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_core.tools import BaseTool, ToolException
from langgraph.graph import StateGraph, END
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer
from dataclasses import dataclass
from typing import Dict

@dataclass
class Context:
    context: Dict[str, str]

class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def create_k8s_agent(llm: BaseChatModel, tools: list[BaseTool], checkpointer: Checkpointer, system_prompt: str):
    llm_with_tools = llm.bind_tools(tools)

    async def tool_node(state: AgentState):
        tools_by_name = {tool.name: tool for tool in tools}
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            try:
                tool_result = await tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
                outputs.append(
                    ToolMessage(
                        content=tool_result,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            except ToolException as e:
                return {"messages": str(e)}
            
        return {"messages": outputs}

    def call_model(
        state: AgentState,
        config: RunnableConfig,
        runtime: Runtime[Context]
    ):
        messages_to_send = [system_prompt]
        if runtime.context.context:
            context_prompt = "Use the following values for the parameters for calling tools. If the user provides different values for these parameters use the values from the user input"
            for key, value in runtime.context.context.items():
                context_prompt += f"key={key}, value={value}\n"
            messages_to_send.append(context_prompt)
        response = llm_with_tools.invoke(messages_to_send + state["messages"], config)

        return {"messages": [response]}


    # Define the conditional edge that determines whether to continue or not
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)
