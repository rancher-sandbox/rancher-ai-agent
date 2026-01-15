import os
import uuid
import logging
import json

from ..dependencies import get_llm
from ..services.agent.factory import create_agent
from dataclasses import dataclass
from fastapi import APIRouter
from fastapi import  WebSocket, WebSocketDisconnect, Depends
from starlette.websockets import WebSocketState
from langgraph.graph.state import CompiledStateGraph
from langfuse.langchain import CallbackHandler
from langchain_core.language_models.llms import BaseLanguageModel
from langgraph.types import  Command

router = APIRouter()

@dataclass
class WebSocketRequest:
    """Represents a parsed WebSocket request from the client."""
    prompt: str
    context: dict
    agent: str = ""

@router.websocket("/agent/ws")
async def websocket_endpoint(websocket: WebSocket, llm: BaseLanguageModel = Depends(get_llm)):
    """
    WebSocket endpoint for the agent.
    
    Accepts a WebSocket connection, sets up the agent and
    handles the back-and-forth communication with the client.
    """
    await websocket.accept()
    logging.debug("ws connection opened")
    
    async with create_agent(llm=llm, websocket=websocket) as agent:
        thread_id = str(uuid.uuid4())
                
        config = {
            "thread_id": thread_id,
        }
        if os.environ.get("LANGFUSE_SECRET_KEY") and os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_HOST"):
            langfuse_handler = CallbackHandler()
            config["callbacks"] = [langfuse_handler]

        while True:
            try:
                request = await websocket.receive_text()
                ws_request = _parse_websocket_request(request)
                prompt = _build_prompt_with_context(ws_request)
                config["agent"] = ws_request.agent or ""
                input_data = _build_input_data(agent, thread_id, prompt)
                
                await _call_agent(
                    agent=agent,
                    input_data=input_data,
                    config=config,
                    websocket=websocket)
                
            except WebSocketDisconnect:
                logging.info(f"Client {websocket.client.host} disconnected.")
                break
            except Exception as e:
                logging.error(f"An error occurred: {e}", exc_info=True)
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(f'<error>{{"message": "{str(e)}"}}</error>')
                else:
                    break
            finally:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text("</message>")

async def _call_agent(
    agent: CompiledStateGraph,
    input_data: any, 
    config: dict,
    websocket: WebSocket,
) -> None:
    """
    Streams the agent's response to a WebSocket connection, handling interruptions.
    
    Args:
        agent: The compiled LangGraph agent.
        input_data: The input data for the agent's run.
        config: The run configuration.
        websocket: The WebSocket connection.
        stream_mode: The types of events to stream from the agent.
    """

    await websocket.send_text("<message>")
    async for stream in agent.astream_events(
        input_data,
        config=config,
        stream_mode=["updates", "messages", "custom", "events"],
    ):
        if stream["event"] == "on_chat_model_stream":
            if text := _extract_streaming_text(stream):
                await websocket.send_text(text)
        
        if stream["event"] == "on_custom_event":
            await websocket.send_text(stream["data"])
    
        if stream["event"] == "on_chain_stream":
            if interrupt_value := _extract_interrupt_value(stream):
                await websocket.send_text(interrupt_value)


def _extract_streaming_text(stream: dict) -> str | None:
    """
    Extracts text content from a chat model stream event.
    
    Only extracts text from 'agent' or 'model' nodes to avoid streaming
    intermediate processing steps.
    
    Args:
        stream: The stream event dictionary from astream_events.
        
    Returns:
        The extracted text content, or None if not applicable.
    """
    STREAMABLE_NODES = ("agent", "model")
    
    node = stream.get("metadata", {}).get("langgraph_node")
    if node not in STREAMABLE_NODES:
        return None
    
    chunk = stream.get("data", {}).get("chunk")
    if not chunk or not chunk.content:
        return None
    
    return _extract_text_from_chunk_content(chunk.content)


def _extract_interrupt_value(stream: dict) -> str | None:
    """
    Extracts the interrupt value from a chain stream event.
    
    LangGraph sends interrupt signals through on_chain_stream events with a specific
    structure: data.chunk is a tuple like ("updates", {"__interrupt__": [Interrupt(...)]})
    
    Args:
        stream: The stream event dictionary from astream_events.
        
    Returns:
        The interrupt value string if present, None otherwise.
    """
    data = stream.get("data")
    if not isinstance(data, dict):
        return None
    
    chunk = data.get("chunk")
    if not isinstance(chunk, (list, tuple)) or len(chunk) < 2:
        return None
    
    if chunk[0] != "updates":
        return None
    
    updates = chunk[1]
    if not isinstance(updates, dict):
        return None
    
    interrupts = updates.get("__interrupt__", [])
    if not interrupts:
        return None
    
    return interrupts[0].value or None

    
def _extract_text_from_chunk_content(chunk_content: any) -> str:
    """
    Extracts the text content from a chunk received from the LLM.

    This function handles different formats that LLMs might return:
    1. A list of dictionaries, where each dictionary contains a 'text' key.
       This is common for models like Gemini that might structure their output.
    2. A single dictionary with a 'text' key.
    3. A simple string or other direct content.

    Args:
        chunk_content: The content field from an LLM chunk.

    Returns:
        str: The extracted text content, or an empty string if no text is found.
    """
    if isinstance(chunk_content, list):
        return "".join([item.get("text", "") for item in chunk_content if isinstance(item, dict)])
    elif isinstance(chunk_content, dict) and "text" in chunk_content:
        return chunk_content["text"]
    
    return str(chunk_content) if chunk_content is not None else ""

def _parse_websocket_request(request: str) -> WebSocketRequest:
    """
    Parses the incoming websocket request.

    The request can be a JSON string with 'prompt', 'context', and 'agent' keys,
    or a plain text string.

    Args:
        request: The raw request string from the websocket.

    Returns:
        A WebSocketRequest object containing the parsed data.
    """
    try:
        json_request = json.loads(request)
        return WebSocketRequest(
            prompt=json_request.get("prompt", ""),
            context=json_request.get("context", {}),
            agent=json_request.get("agent", "")
        )
    except json.JSONDecodeError:
        return WebSocketRequest(prompt=request, context={}, agent="")


def _build_prompt_with_context(ws_request: WebSocketRequest) -> str:
    """
    Builds the final prompt by appending context parameters if present.
    
    Context parameters are appended as key:value pairs to guide the LLM
    in populating tool calls with relevant values.
    
    Args:
        ws_request: The parsed WebSocket request.
        
    Returns:
        The prompt string, potentially enriched with context parameters.
    """
    if not ws_request.context:
        return ws_request.prompt
    
    context_parts = [f"{key}:{value}" for key, value in ws_request.context.items()]
    context_suffix = (
        ". Use the following parameters to populate tool calls when appropriate. \n"
        "Only include parameters relevant to the user's request "
        "(e.g., omit namespace for cluster-wide operations). \n"
        f"Parameters (separated by ;): \n {';'.join(context_parts)};"
    )
    return ws_request.prompt + context_suffix


def _build_input_data(agent: CompiledStateGraph, thread_id: str, prompt: str) -> dict | Command:
    """
    Builds the input data for the agent, handling interrupt resumption.
    
    If the agent is waiting on an interrupt, resumes with the user's response.
    Otherwise, creates a new user message.
    
    Args:
        agent: The compiled LangGraph agent.
        thread_id: The conversation thread ID.
        prompt: The user's prompt.
        
    Returns:
        Either a Command to resume an interrupt, or a messages dict for a new turn.
    """
    state = agent.get_state(config={"configurable": {"thread_id": thread_id}})
    
    if state.interrupts:
        return Command(resume=prompt)
    
    return {"messages": [{"role": "user", "content": prompt}]}
