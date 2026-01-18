import json
import logging
import langgraph.types

from datetime import datetime
from typing import Annotated, Sequence, TypedDict, NotRequired
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, ToolException
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph, Checkpointer
from langchain_core.language_models.chat_models import BaseChatModel
from ollama import ResponseError
from langchain_core.callbacks.manager import dispatch_custom_event

from .types import AgentState as ChildAgentState

INTERRUPT_CANCEL_MESSAGE = "tool execution cancelled by the user"

class ChildAgentBuilder:
    def __init__(self, llm: BaseChatModel, tools: list[BaseTool], system_prompt: str, checkpointer: Checkpointer):
        """
        Initializes the ChildAgentBuilder.

        Args:
            llm: The language model to use for the agent's decisions.
            tools: A list of tools the agent can use.
            system_prompt: The initial system-level instructions for the agent.
            checkpointer: The checkpointer for persisting agent state.
        """
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.checkpointer = checkpointer
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
    
    def summarize_conversation_node(self, state: ChildAgentState):
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
        response = self.llm_with_tools.invoke(messages)

        # Mark this response explicitly as a summary so that it can be filtered
        # out from the memory endpoints results.
        # Summary are used to condense the conversation but should not appear
        # as part of the conversation history.
        if response.additional_kwargs is None:
            response.additional_kwargs = {}
        response.additional_kwargs["is_summary"] = True

        new_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
        new_messages = new_messages + [response]

        logging.debug("summarizing conversation")
        
        return {"summary": response.content, "messages": new_messages}

    def _invoke_llm_with_retry(self, messages: list, config: RunnableConfig):
        """
        Invokes the LLM, with a single retry for a tool call parsing error.
        Models can sometimes hallucinate and produce malformed JSON for tool calls.
        This method attempts the LLM call and retries if it fails with a specific
        "error parsing tool" """
        # 1 initial attempt + 1 retry
        for attempt in range(2):
            try:
                return self.llm_with_tools.invoke(messages, config)
            except ResponseError as e:
                if "error parsing tool call:" in str(e.error) and attempt < 1:
                    logging.warning(f"retrying due to tool call parsing error: {e.error}")
                    continue
                raise e

    def call_model_node(self, state: ChildAgentState, config: RunnableConfig):
        """
        Invokes the language model with the current state and context.

        This node prepares the messages for the LLM, including the system prompt,
        any contextual information (like current cluster), and the conversation history.
        It then calls the LLM to get the next response.

        Args:
            state: The current state of the agent.
            config: The runnable configuration.
            runtime: The LangGraph runtime, which holds the user's context.

        Returns:
            A dictionary containing the LLM's response message."""
        
        logging.debug("calling model")

        messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
        response = self._invoke_llm_with_retry(messages, config)

        logging.debug("model call finished")

        return {"messages": [response]}

    async def tool_node(self, state: ChildAgentState):
        """
        Executes tools based on the LLM's request.

        This node processes tool calls from the last message, handling user
        confirmation for sensitive operations. It invokes the appropriate tool
        and returns the results as ToolMessage objects.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary containing a list of ToolMessage objects with the tool results,
            or an error message if a tool fails or is cancelled."""
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            if not _handle_interrupt(tool_call):
                return {"messages": ToolMessage(
                        content=INTERRUPT_CANCEL_MESSAGE,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                        )}
            
            try:
                logging.debug("calling tool")
                tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
                logging.debug("tool call finished")
                processed_result = _process_tool_result(tool_result)
                outputs.append(
                    ToolMessage(
                        content=processed_result,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )
            except ToolException as e:
                return {"messages": str(e)}
            except Exception as e:
                logging.error(f"unexpected error during tool call: {e}")
                return {"messages": f"unexpected error during tool call: {e}"}

        return {"messages": outputs}
    
    def should_summarize_conversation(self, state: ChildAgentState):
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
            if len(messages) > 7:
                return "summarize_conversation"
            return "end"
        else:
            return "continue"
        
    def should_continue_after_interrupt(self, state: ChildAgentState):
        """
        Determines whether to continue execution after a tool interruption.

        This conditional edge checks if the last message indicates that a tool
        execution was cancelled by the user. If so, it ends the workflow;
        otherwise, it continues back to the agent node.

        Args:
            state: The current state of the agent.

        Returns:
            A string indicating the next node: "end" if the user cancelled,
            or "continue" to proceed with the agent."""
        messages = state["messages"]
        last_message = messages[-1]
        if isinstance(last_message, ToolMessage) and last_message.content == INTERRUPT_CANCEL_MESSAGE:
            return "end"
        
        return "continue"


    def build(self) -> CompiledStateGraph:
        """
        Builds and compiles the LangGraph agent.

        Returns:
            A compiled LangGraph StateGraph ready to be invoked.
        """
        workflow = StateGraph(ChildAgentState)
        workflow.add_node("agent", self.call_model_node)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("summarize_conversation", self.summarize_conversation_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self.should_summarize_conversation,
            {
                "continue": "tools",
                "summarize_conversation": "summarize_conversation",
                "end": END,
            },
        )
        workflow.add_conditional_edges(
            "tools",
            self.should_continue_after_interrupt,
            {
                "continue": "agent",
                "end": END,
            },
        )
        workflow.add_edge("summarize_conversation", END)

        return workflow.compile(checkpointer=self.checkpointer)

def create_child_agent(llm: BaseChatModel, tools: list[BaseTool], system_prompt: str, checkpointer: Checkpointer) -> CompiledStateGraph:
    """
    Creates a LangGraph agent capable of interacting with Rancher and Kubernetes resources.
    
    This factory function instantiates the AgentBuilder, builds the agent graph,
    and returns the compiled agent.
    
    Args:
        llm: The language model to use for the agent's decisions.
        tools: A list of tools the agent can use (e.g., to interact with K8s).
        system_prompt: The initial system-level instructions for the agent.
    
    Returns:
        A compiled LangGraph StateGraph ready to be invoked.
    """
    builder = ChildAgentBuilder(llm, tools, system_prompt, checkpointer)

    return builder.build()

def _create_confirmation_response(payload: str, type: str, name: str, kind: str, cluster: str, namespace: str):
    """
    Creates a structured confirmation response for the UI.

    This function formats a JSON payload that the UI can use to prompt the user
    for confirmation before executing a sensitive operation.

    Args:
        payload: The data for the operation (e.g., a patch or a resource definition).
        type: The type of operation (e.g., "patch").
        name: The name of the resource.
        kind: The kind of the resource (e.g., "Deployment").
        cluster: The target cluster.
        namespace: The target namespace.
    """
    payload_data = {
        "payload": payload,
        "type": type,
        "resource": {
            "name": name,
            "kind": kind,
            "cluster": cluster,
            "namespace": namespace
        }
    }

    json_payload = json.dumps(payload_data)

    return f'<confirmation-response>{json_payload}</confirmation-response>'

def _should_interrupt(tool_call: any) -> str:
    """
    Checks if a tool call requires user confirmation and generates an interrupt message.

    Args:
        tool_call: The tool call dictionary from the LLM.

    Returns:
        A formatted string to trigger a langgraph.types.interrupt, or an empty string
        if no interruption is needed.
    """
    if tool_call["name"] == "patchKubernetesResource":
        return _create_confirmation_response(tool_call['args']['patch'], "patch", tool_call['args']['name'], tool_call['args']['kind'], tool_call['args']['cluster'], tool_call['args']['namespace'])
    if tool_call["name"] == "createKubernetesResource":
        return _create_confirmation_response(tool_call['args']['resource'], "create", tool_call['args']['name'], tool_call['args']['kind'], tool_call['args']['cluster'], tool_call['args']['namespace'])
    return ""

def _extract_interrupt_message(interrupt_message:any) -> str: 
    """
    Extracts the user's response from an interrupt.

    The response from the UI might be a simple string or a JSON object.
    This function standardizes the extraction of the user's actual response.
    """
    try:
        json_response = json.loads(interrupt_message["response"])
        return json_response["prompt"]
    except Exception:
        return interrupt_message["response"]
    
def _handle_interrupt(tool_call: dict) -> bool:
    """Handles the user confirmation interrupt for a tool call."""
    if interrupt_message := _should_interrupt(tool_call):
        response = _extract_interrupt_message(langgraph.types.interrupt(interrupt_message))
        if response != "yes":
            return False
          
    return True 

def _process_tool_result(tool_result: str | list) -> str:
    """Processes the raw tool result, handling JSON and streaming UI context if necessary.
       MCP returns example: {"uiContext":{}, "llm": {}}"""
    try:
        # Handle list format: [{"type": "text", "text": "tool response", "id": "..."}]
        if isinstance(tool_result, list) and len(tool_result) > 0:
            if isinstance(tool_result[0], dict) and "text" in tool_result[0]:
                tool_result = tool_result[0]["text"]

        json_result = json.loads(tool_result)

        if "uiContext" in json_result:
            dispatch_custom_event(
            "ui_context",
            f"<mcp-response>{json.dumps(json_result['uiContext'])}</mcp-response>")
        if "docLinks" in json_result:
            for link in json_result['docLinks']:
                dispatch_custom_event(
                "dock_link",
                f"<mcp-doclink>{link}</mcp-doclink>")

        # Return the value for the LLM, or the full object if 'llm' key is not present
        return _convert_to_string_if_needed(json_result.get("llm", json_result))
    except (json.JSONDecodeError, TypeError):
        # If it's not a valid JSON, return the raw string result
        return tool_result

def _convert_to_string_if_needed(var):
    """
    Converts a variable to a JSON formatted string only if it is a dict or list
    
    Args:
        var: The variable to process.
        
    Returns:
        The JSON string representation if it was a dict/list,
        otherwise the original variable.
    """
    if isinstance(var, (dict, list)):
        return json.dumps(var)
    else:
        return var