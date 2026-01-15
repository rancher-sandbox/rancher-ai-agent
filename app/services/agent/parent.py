"""
Parent agent implementation using LangGraph Command primitive for routing to specialized child agents.

This module provides a parent agent that uses an LLM to intelligently route user requests
to the most appropriate specialized child agent based on the request content.
"""
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph, Checkpointer
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command
from langchain_core.callbacks.manager import dispatch_custom_event
from dataclasses import dataclass

from .base import BaseAgentBuilder, AgentState

@dataclass
class ChildAgent:
    """
    Represents a specialized child agent that can handle specific types of requests.
    
    Attributes:
        name: Unique identifier for the child agent
        description: Description of the child agent's capabilities and use cases
        agent: The compiled LangGraph agent that handles the actual work
    """
    name: str
    description: str
    agent: CompiledStateGraph


class ParentAgentBuilder(BaseAgentBuilder):
    """
    Builder class for creating a parent agent that routes requests to specialized child agents.
    
    The parent agent uses an LLM to analyze incoming requests and route them to the
    most appropriate child agent using LangGraph's Command primitive for navigation.
    """
    
    def __init__(self, llm: BaseChatModel, subagents: list[ChildAgent], checkpointer: Checkpointer):
        """
        Initialize the parent agent builder.
        
        Args:
            llm: Language model used for routing decisions
            subagents: List of available specialized child agents
            checkpointer: Checkpointer for persisting agent state
        """
        super().__init__(llm=llm, tools=[], system_prompt="", checkpointer=checkpointer)
        self.subagents = subagents
    
    def choose_subagent(self, state: AgentState, config: RunnableConfig):
        """
        Route the user request to the most appropriate child agent.
        
        Uses the LLM to analyze the user's request and select which child agent
        should handle it based on the child agent descriptions.
        
        Args:
            state: Current agent state containing messages
            config: Runtime configuration
            
        Returns:
            Command object directing workflow to the selected child agent
        """

        # UI override to force a specific child agent
        agent_override = config.get("configurable", {}).get("agent", "")
        if agent_override:
            dispatch_custom_event(
                "subagent_choice_event",
                f"_DEBUG MESSAGE: Using UI-specified agent: {agent_override}_ \n",
            )
            self.agent_selected = agent_override
            return Command(goto=agent_override)
        
        messages = state["messages"]

        # Build routing prompt with available child agents and their descriptions
        llm_route_prompt = "Based on the user request, decide which child agent is best suited to handle the user's request. Respond with only the name of the child agent.\n\n"
        llm_route_prompt += "Available child agents:\n"
        for sa in self.subagents:
            llm_route_prompt += f"- {sa.name}: {sa.description}\n"
        llm_route_prompt += f"\nUser's request: {messages[-1].content}"
        
        # Use LLM to select the appropriate child agent
        child_agent = self.llm.invoke(llm_route_prompt).content

        self.agent_selected = child_agent

        dispatch_custom_event(
            "subagent_choice_event",
            f"_DEBUG MESSAGE: LLM selected: {child_agent}_ \n",
        )

        # Return Command to navigate to the selected child agent
        return Command(goto=child_agent)

    def build(self) -> CompiledStateGraph:
        """
        Build and compile the parent agent workflow graph.
        
        Creates a LangGraph workflow with a routing node and nodes for each child agent.
        The workflow starts with routing and navigates to the appropriate child agent.
        
        Returns:
            Compiled state graph ready for execution
        """
        workflow = StateGraph(AgentState)
        
        workflow.add_node("choose_subagent", self.choose_subagent)
        workflow.add_node("summarize_conversation", self.summarize_conversation_node)

        # Add a node for each child agent
        for sa in self.subagents:
            workflow.add_node(sa.name, sa.agent)
            workflow.add_conditional_edges(
            sa.name,
            self.should_summarize_conversation,
            {
                "summarize_conversation": "summarize_conversation",
                "end": END,
            },
        )

        
        # Set the routing node as the entry point
        workflow.set_entry_point("choose_subagent")

        return workflow.compile(checkpointer=self.checkpointer)

def create_parent_agent(llm: BaseChatModel, subagents: list[ChildAgent], checkpointer: Checkpointer) -> CompiledStateGraph:
    """
    Factory function to create a parent agent with routing capabilities.
    
    Args:
        llm: Language model for routing decisions
        subagents: List of specialized child agents to route between
        checkpointer: Checkpointer for state persistence
        
    Returns:
        Compiled parent agent ready to route requests to child agents
    """
    builder = ParentAgentBuilder(llm, subagents, checkpointer)

    return builder.build()
