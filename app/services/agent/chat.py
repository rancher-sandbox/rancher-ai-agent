from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph, Checkpointer

from .types import AgentState as ChatAgentState

class ChatAgentBuilder:
    """Builds a minimal agent for reading chat state."""
    
    def __init__(self, checkpointer: Checkpointer):
        """
        Initialize ChatAgentBuilder.
        
        Args:
            checkpointer: The checkpointer for reading agent state.
        """
        self.checkpointer = checkpointer
    
    def build(self) -> CompiledStateGraph:
        """
        Build and compile the chat state reader graph.
        
        Returns:
            A compiled state graph that can read and persist state.
        """
        workflow = StateGraph(ChatAgentState)
        
        # Add a simple passthrough node
        def passthrough_node(state: ChatAgentState) -> ChatAgentState:
            """Passthrough node that doesn't modify state."""
            return state
        
        workflow.add_node("reader", passthrough_node)
        workflow.set_entry_point("reader")
        workflow.add_edge("reader", "__end__")
        
        return workflow.compile(checkpointer=self.checkpointer)

def create_chat_agent(checkpointer: Checkpointer) -> CompiledStateGraph:
    """
    Create a minimal agent for reading chat state.
    
    Args:
        checkpointer: The checkpointer for reading state.
    
    Returns:
        Compiled agent ready to read state.
    """
    builder = ChatAgentBuilder(checkpointer)
    return builder.build()
