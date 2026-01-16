"""
Child agent implementation for agents that work under a parent agent.

This module provides a child agent builder that delegates summarization
to its parent agent and focuses on executing tasks assigned by the parent.
"""

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph, Checkpointer
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from .builtin_agents import AgentConfig

from .base import BaseAgentBuilder, AgentState


class ChildAgentBuilder(BaseAgentBuilder):
    """Builder for child agents that work under a parent agent."""

    def build(self) -> CompiledStateGraph:
        """
        Builds and compiles the LangGraph child agent.

        Returns:
            A compiled LangGraph StateGraph ready to be invoked.
        """
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.call_model_node)
        workflow.add_node("tools", self.tool_node)

        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "tools",
            self.should_continue_after_interrupt,
            {
                "continue": "agent",
                "end": END,
            },
        )
        workflow.set_entry_point("agent")

        return workflow.compile(checkpointer=self.checkpointer)


def create_child_agent(llm: BaseChatModel, tools: list[BaseTool], system_prompt: str, checkpointer: Checkpointer, agent_config: AgentConfig) -> CompiledStateGraph:
    """
    Creates a LangGraph child agent capable of interacting with Rancher and Kubernetes resources.
    
    This factory function instantiates the ChildAgentBuilder, builds the agent graph,
    and returns the compiled agent.
    
    Args:
        llm: The language model to use for the agent's decisions.
        tools: A list of tools the agent can use (e.g., to interact with K8s).
        system_prompt: The initial system-level instructions for the agent.
        checkpointer: The checkpointer for persisting agent state.
        agent_config: Configuration for the agent's behavior and settings.
    Returns:
        A compiled LangGraph StateGraph ready to be invoked.
    """
    builder = ChildAgentBuilder(llm, tools, system_prompt, checkpointer, agent_config)
    return builder.build()
