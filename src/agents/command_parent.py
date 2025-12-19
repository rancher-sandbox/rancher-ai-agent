"""
Parent agent implementation using LangGraph Command primitive for routing to specialized subagents.

This module provides a parent agent that uses an LLM to intelligently route user requests
to the most appropriate specialized subagent based on the request content.
"""

import logging
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, ToolException
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph, Checkpointer
from langchain_core.language_models.chat_models import BaseChatModel
from ollama import ResponseError
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict, Literal
from langchain_core.callbacks.manager import dispatch_custom_event
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

        # UI override to force a specific subagent
        agent_override = config.get("configurable", {}).get("agent", "")
        if agent_override:
            dispatch_custom_event(
                "subagent_choice_event",
                f"_DEBUG MESSAGE: Using UI-specified agent: {agent_override}_ \n",
            )
            return Command(goto=agent_override)
        
        messages = state["messages"]

        # Build routing prompt with available subagents and their descriptions
        llm_route_prompt = "Based on the user request, decide which subagent is best suited to handle the user's request. Respond with only the name of the subagent.\n\n"
        llm_route_prompt += "Available subagents:\n"
        for sa in self.subagents:
            llm_route_prompt += f"- {sa.name}: {sa.description}\n"
        llm_route_prompt += f"\nUser's request: {messages[-1].content}"
        
        # Use LLM to select the appropriate subagent
        subagent = self.llm.invoke(llm_route_prompt).content

        dispatch_custom_event(
            "subagent_choice_event",
            f"_DEBUG MESSAGE: LLM selected: {subagent}_ \n",
        )

        # Return Command to navigate to the selected subagent
        return Command(goto=subagent)
    
    def should_summarize_conversation(self, state: AgentState):
        """
        Determines the next step in the agent's workflow.

        This conditional edge checks the last message in the state to decide whether to
        continue with a tool call, summarize the conversation, or end the execution.

        Args:
            state: The current state of the agent.

        Returns:
            A string indicating the next node to transition to: "continue",
            "summarize_conversation", or "end"."""
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            if len(messages) > 7: ## TODO check tokens not len messages!
                return "summarize_conversation"
            return "end"
        else:
            return "continue"
        
    def summarize_conversation_node(self, state: AgentState):
        """
        Summarizes the conversation history.

        This node is invoked when the conversation becomes too long. It asks the LLM
        to create or extend a summary of the conversation, then replaces the
        previous messages with the new summary to keep the context concise.

        Args:
            state: The current state of the agent, containing messages and an optional summary.

        Returns:
            A dictionary with the updated summary and a condensed list of messages."""
        summary = state.get("summary", "")
        if summary:
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n"
                "Extend the summary by taking into account the new messages above:" )
        else:
            summary_message = "Create a summary of the conversation above:"

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = self.llm.invoke(messages)
        new_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
        new_messages = new_messages + [response]

        logging.debug("summarizing conversation")
        
        return {"summary": response.content, "messages": new_messages}


    def build(self) -> CompiledStateGraph:
        """
        Build and compile the parent agent workflow graph.
        
        Creates a LangGraph workflow with a routing node and nodes for each subagent.
        The workflow starts with routing and navigates to the appropriate subagent.
        
        Returns:
            Compiled state graph ready for execution
        """
        workflow = StateGraph(AgentState)
        
        workflow.add_node("choose_subagent", self.choose_subagent)
        workflow.add_node("summarize_conversation", self.summarize_conversation_node)

        # Add a node for each subagent
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
