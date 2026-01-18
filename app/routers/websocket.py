import os
import uuid
import logging
import json

from dataclasses import dataclass
from fastapi import APIRouter
from fastapi import  WebSocket, WebSocketDisconnect, Depends
from starlette.websockets import WebSocketState
from langgraph.graph.state import CompiledStateGraph
from langfuse.langchain import CallbackHandler
from langchain_core.language_models.llms import BaseLanguageModel

from ..services.auth import get_user_id
from ..dependencies import get_llm
from ..services.agent.agent import create_agent
from ..types import RequestType

router = APIRouter()

async def get_user_id_from_websocket(websocket: WebSocket) -> str:
    """
    Retrieves the user ID from the Rancher API using the session token from the WebSocket cookies.
    """
    cookies = websocket.cookies
    rancher_url = os.environ.get("RANCHER_URL","https://"+websocket.url.hostname)
    token = os.environ.get("RANCHER_API_TOKEN", cookies.get("R_SESS", ""))

    return await get_user_id(rancher_url, token)

@dataclass
class WebSocketRequest:
    """Represents a parsed WebSocket request from the client."""
    prompt: str
    context: dict
    tags: list[str] = None
    agent: str = ""

@router.websocket("/agent/ws/messages")
@router.websocket("/agent/ws/messages/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: str = None, llm: BaseLanguageModel = Depends(get_llm)):
    """
    WebSocket endpoint for the agent.
    
    Accepts a WebSocket connection, sets up the agent and
    handles the back-and-forth communication with the client.
    """
    
    user_id = await get_user_id_from_websocket(websocket)
    
    if not thread_id:
        thread_id = str(uuid.uuid4())
    logging.debug(f"Starting websocket session with thread_id: {thread_id}, user_id: {user_id}")
    
    await websocket.accept()
    logging.debug("ws/messages connection opened")
    
    await websocket.send_text(f'<chat-metadata>{{"chatId": "{thread_id}"}}</chat-metadata>')

    async with create_agent(llm=llm, websocket=websocket, request_type=RequestType.MESSAGE) as ctx:
        agent = ctx.agent

        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
            },
        }

        if os.environ.get("LANGFUSE_SECRET_KEY") and os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_HOST"):
            langfuse_handler = CallbackHandler()
            config["callbacks"] = [langfuse_handler]

        while True:
            try:
                request = await websocket.receive_text()
                request_id = str(uuid.uuid4())
                
                ws_request = _parse_websocket_request(request)
                content = ws_request.prompt
                if ws_request.context:
                    context_prompt = ". Use the following parameters to populate tool calls when appropriate. \n Only include parameters relevant to the user's request (e.g., omit namespace for cluster-wide operations). \n Parameters (separated by ;): \n "
                    for key, value in ws_request.context.items():
                        context_prompt += f"{key}:{value};"
                    content += context_prompt

                config["configurable"]["request_id"] = request_id

                if ws_request.agent:
                    config["configurable"]["agent"] = ws_request.agent
                else:
                    config["configurable"]["agent"] = ""

                input_messages = [{"role": "user", "content": content}]

                input_data = {
                    "messages": input_messages,
                    "agent_metadata": {
                        "prompt": ws_request.prompt,
                        "context": ws_request.context,
                        "tags": ws_request.tags,
                        "mcp_responses": [],
                    },
                }

                await stream_agent_response(
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

    # TODO: any additional cleanup if necessary and changes to support OAuth 2
    # - Clean up MCP session and client if needed.
    # - Each user requires their own session for token-based authentication.

async def stream_agent_response(
    agent: CompiledStateGraph,
    input_data: dict[str, list[dict[str, str]]],
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
            if stream["data"]["chunk"].content and (stream["metadata"]["langgraph_node"] == "agent" or stream["metadata"]["langgraph_node"] == "model"):
                await websocket.send_text(_extract_text_from_chunk_content(stream["data"]["chunk"].content))
        
        if stream["event"] == "on_custom_event":
            current_state = await agent.aget_state(config)
            metadata = current_state.values.get("agent_metadata", {})
            mcp_responses = metadata.get("mcp_responses", [])
            if mcp_responses is None:
                mcp_responses = []
            mcp_responses.append(stream["data"])
            metadata["mcp_responses"] = mcp_responses
            
            await agent.aupdate_state(
                config,
                {"agent_metadata": metadata}
            )
            
            await websocket.send_text(stream["data"])
    
        if stream["event"] == "on_chain_stream":
            data = stream.get("data")
            if isinstance(data, dict):
                chunk = data.get("chunk")
                if chunk and isinstance(chunk, (list, tuple)) and len(chunk) > 0 and chunk[0] == "updates":
                    if len(chunk) > 1 and isinstance(chunk[1], dict):
                        interrupts = chunk[1].get("__interrupt__", [])
                        if interrupts and len(interrupts) > 0:
                            interrupt_value = interrupts[0].value
                            if interrupt_value:
                                current_state = await agent.aget_state(config)
                                metadata = current_state.values.get("agent_metadata", {})
                                metadata["last_interrupt"] = interrupt_value
                                
                                # Store interrupt in agent state
                                await agent.aupdate_state(
                                    config,
                                    {"agent_metadata": metadata}
                                )
                                await websocket.send_text(interrupt_value)
    
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
            tags = json_request.get("tags", []),
            agent=json_request.get("agent", "")
        )
    except json.JSONDecodeError:
        return WebSocketRequest(prompt=request, context={}, tags=[], agent="")