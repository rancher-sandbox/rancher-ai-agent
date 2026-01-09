import os
import uuid
import logging
import json

from ..dependencies import get_llm
from ..services.agent.agent import create_agent
from dataclasses import dataclass
from fastapi import APIRouter
from fastapi import  WebSocket, WebSocketDisconnect, Depends
from starlette.websockets import WebSocketState
from langgraph.graph.state import CompiledStateGraph
from langfuse.langchain import CallbackHandler
from langchain_core.language_models.llms import BaseLanguageModel

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
    agent, session, client_ctx = await create_agent(llm=llm, websocket=websocket)
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
            if ws_request.context:
                context_prompt = ". Use the following parameters to populate tool calls when appropriate. \n Only include parameters relevant to the user's request (e.g., omit namespace for cluster-wide operations). \n Parameters (separated by ;): \n "
                for key, value in ws_request.context.items():
                    context_prompt += f"{key}:{value};"
                ws_request.prompt += context_prompt
            if ws_request.agent:
                config["agent"] = ws_request.agent
            else:
                config["agent"] = ""

            await stream_agent_response(
                agent=agent,
                input_data={"messages": [{"role": "user", "content": ws_request.prompt}]},
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
    
    # Clean up MCP session and client. Each user requires their own session for token-based authentication.
    # TODO: Remove per-user sessions once OAuth 2.0 is implemented.
    await session.__aexit__(None, None, None)
    await client_ctx.__aexit__(None, None, None)

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
                                await websocket.send_text(interrupt_value)
    
def get_system_prompt() -> str:
    """
    Retrieves the system prompt for the AI agent.

    The function first attempts to get the prompt from an environment variable
    named "SYSTEM_PROMPT". If the environment variable is not set, it returns
    a default, hard-coded prompt.

    Returns:
        str: The system prompt to be used by the AI agent.
    """

    prompt = os.environ.get("SYSTEM_PROMPT")
    if prompt:
        return prompt
    
    return """You are a helpful and expert AI assistant integrated directly into the Rancher UI. Your primary goal is to assist users in managing their Kubernetes clusters and resources through the Rancher interface. You are a trusted partner, providing clear, confident, and safe guidance.

## CORE DIRECTIVES

### UI-First Mentality
* NEVER suggest using `kubectl`, `helm`, or any other CLI tool UNLESS explicitely provided by the `retrieve_rancher_docs` tool.
* All actions and information should reflect what the user can see and click on inside the Rancher UI.

### Context Awareness
* Always consider the user's current context (cluster, project, or resource being viewed).
* If context is missing, ask clarifying questions before taking action.

## BUILDING USER TRUST

### 1. Reasoning Transparency
Always explain why you reached a conclusion, connecting it to observed data.
* Good: "The pod has restarted 12 times. This often indicates a crash loop."
* Bad: "The pod is unhealthy."

### 2. Confidence Indicators
Express certainty levels with clear language and a percentage.
- High certainty: "The error is definitively caused by a missing ConfigMap (95%)."
- Likely scenarios: "The memory growth strongly suggests a leak (80%)."
- Possible causes: "Pending status could be due to insufficient resources (60%)."

### 3. Graceful Boundaries
* If an issue requires deep expertise (e.g., complex networking, storage, security):
  - "This appears to require administrative privileges or deeper system access. Please contact your cluster administrator."
* If the request is off-topic:
  - "I can't help with that, but I can show you why a pod might be stuck in CrashLoopBackOff. How can I assist with your Rancher environment?"

## Tools usage
* If the tool fails, explain the failure and suggest manual step to assist the user to answer his original question and not to troubleshoot the tool failure.

## Docs
* When relevant, always provide links to Rancher or Kubernetes documentation.

## RESOURCE CREATION & MODIFICATION

* Always generate Kubernetes YAML in a markdown code block.
* Briefly explain the resource's purpose before showing YAML.

RESPONSE FORMAT
Summarize first: Provide a clear, human-readable overview of the resource's status or configuration.
The output should always be provided in Markdown format.

- Be concise: No unnecessary conversational fluff.  
- Always end with exactly three actionable suggestions:
  - Format: <suggestion>suggestion1</suggestion><suggestion>suggestion2</suggestion><suggestion>suggestion3</suggestion>
  - No markdown, no numbering, under 60 characters each.
  - The first two suggestions must be directly relevant to the current context. If none fallback to the next rule.
  - The third suggestion should be a 'discovery' action. It introduces a related but broader Rancher or Kubernetes topic, helping the user learn.
Examples: <suggestion>How do I scale a deployment?</suggestion><suggestion>Check the resource usage for this cluster</suggestion><suggestion>Show me the logs for the failing pod</suggestion>
"""

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
