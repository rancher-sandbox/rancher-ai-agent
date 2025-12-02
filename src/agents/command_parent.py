"""
Parent agent implementation using LangGraph Command primitive for routing to specialized subagents.

This module provides a parent agent that uses an LLM to intelligently route user requests
to the most appropriate specialized subagent based on the request content.
"""

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langchain_core.tools import BaseTool, ToolException
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph, Checkpointer
from langchain_core.language_models.chat_models import BaseChatModel
from ollama import ResponseError
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict, Literal

from dataclasses import dataclass

@dataclass
class SubAgent:
    """
    Represents a specialized subagent that can handle specific types of requests.
    
    Attributes:
        name: Unique identifier for the subagent
        description: Description of the subagent's capabilities and use cases
        agent: The compiled LangGraph agent that handles the actual work
    """
    name: str
    description: str
    agent: CompiledStateGraph

class AgentState(TypedDict):
    """
    State structure for the parent agent workflow.
    
    Attributes:
        messages: Sequence of conversation messages with automatic message addition
        summary: Optional summary of the conversation context
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary: str

class ParentAgentBuilder:
    """
    Builder class for creating a parent agent that routes requests to specialized subagents.
    
    The parent agent uses an LLM to analyze incoming requests and route them to the
    most appropriate subagent using LangGraph's Command primitive for navigation.
    """
    
    def __init__(self, llm: BaseChatModel, subagents: list[SubAgent], checkpointer: Checkpointer):
        """
        Initialize the parent agent builder.
        
        Args:
            llm: Language model used for routing decisions
            subagents: List of available specialized subagents
            checkpointer: Checkpointer for persisting agent state
        """
        self.llm = llm
        self.subagents = subagents
        self.checkpointer = checkpointer
    
    def choose_subagent(self, state: AgentState, config: RunnableConfig):
        """
        Route the user request to the most appropriate subagent.
        
        Uses the LLM to analyze the user's request and select which subagent
        should handle it based on the subagent descriptions.
        
        Args:
            state: Current agent state containing messages
            config: Runtime configuration
            
        Returns:
            Command object directing workflow to the selected subagent
        """
        messages = state["messages"]
        
        # Build routing prompt with available subagents and their descriptions
        llm_route_prompt = "Based on the user request, decide which subagent is best suited to handle the user's request. Respond with only the name of the subagent.\n\n"
        llm_route_prompt += "Available subagents:\n"
        for sa in self.subagents:
            llm_route_prompt += f"- {sa.name}: {sa.description}\n"
        llm_route_prompt += f"\nUser's request: {messages[-1].content}"
        
        # Use LLM to select the appropriate subagent
        subagent = self.llm.invoke(llm_route_prompt).content
        print("LLM selected subagent:", subagent)

        # Return Command to navigate to the selected subagent
        return Command(
            goto=subagent,
        )

    def build(self) -> CompiledStateGraph:
        """
        Build and compile the parent agent workflow graph.
        
        Creates a LangGraph workflow with a routing node and nodes for each subagent.
        The workflow starts with routing and navigates to the appropriate subagent.
        
        Returns:
            Compiled state graph ready for execution
        """
        workflow = StateGraph(AgentState)
        
        # Add routing node as entry point
        workflow.add_node("choose_subagent", self.choose_subagent)
        
        # Add a node for each subagent
        for sa in self.subagents:
            workflow.add_node(sa.name, sa.agent)
        
        # Set the routing node as the entry point
        workflow.set_entry_point("choose_subagent")

        return workflow.compile(checkpointer=self.checkpointer)

def create_parent_agent(llm: BaseChatModel, subagents: list[SubAgent], checkpointer: Checkpointer) -> CompiledStateGraph:
    """
    Factory function to create a parent agent with routing capabilities.
    
    Args:
        llm: Language model for routing decisions
        subagents: List of specialized subagents to route between
        checkpointer: Checkpointer for state persistence
        
    Returns:
        Compiled parent agent ready to route requests to subagents
    """
    builder = ParentAgentBuilder(llm, subagents, checkpointer)

    return builder.build()
