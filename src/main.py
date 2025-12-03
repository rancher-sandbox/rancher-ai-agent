import logging
import os
import json
import uuid
import certifi

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from starlette.websockets import WebSocketState
from fastapi.responses import HTMLResponse
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama 
from langchain_mcp_adapters.tools import load_mcp_tools
from .agents import create_k8s_agent, fleet_documentation_retriever, init_rag_retriever, rancher_documentation_retriever, create_parent_agent, SubAgent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langchain_core.language_models.llms import BaseLanguageModel
from contextlib import asynccontextmanager
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

init_config = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
        logging.getLogger().setLevel(LOG_LEVEL)
        init_config["llm"] = get_llm()
        logging.info(f"Using model: {init_config['llm']}")
        if os.environ.get("ENABLE_RAG", "false").lower() == "true":
            init_rag_retriever()
        if os.environ.get('INSECURE_SKIP_TLS', 'false').lower() != "true":
            SimpleTruststore().set_truststore()
    except ValueError as e:
        logging.critical(e)
        raise e
    yield
    init_config.clear()

app = FastAPI(lifespan=lifespan)


def create_math_agent() -> CompiledStateGraph:
    """
    Creates a simple math agent that can sum two numbers.
    
    Returns:
        A compiled LangGraph agent capable of summing two numbers.
    """
    @tool
    def sum_two_numbers(a: int, b: int) -> int:
        """Sum two numbers together.
        
        Args:
            a: The first number to sum
            b: The second number to sum
        
        Returns:
            The sum of a and b
        """
        print("Summing", a, "and", b)
        return a + b


    return create_agent(model=get_llm(), tools=[sum_two_numbers])

async def create_rancher_agent():
    rancher_url = "https://raul-cabello.ngrok.app"
    mcpUrl = "http://localhost:9092"

    client_ctx = streamablehttp_client(
        url=mcpUrl,
        headers={
             "R_token":"",
             "R_url":rancher_url
        }
    )
    
    read, write, _ = await client_ctx.__aenter__()
    
    # Initialize MCP session
    session = ClientSession(read, write)
    await session.__aenter__()
    await session.initialize()
    tools = await load_mcp_tools(session)
    
    # if ENABLE_RAG is true, add the retriever tools to the tools list
    if os.environ.get("ENABLE_RAG", "false").lower() == "true":
        tools = [fleet_documentation_retriever, rancher_documentation_retriever] + tools

    agent = create_k8s_agent(init_config["llm"], tools, get_system_prompt(), InMemorySaver())
    
    return agent, session, client_ctx  # Return all three for cleanup

async def create_observability_agent():
    client_ctx = streamablehttp_client(
        url="http://localhost:9093",
    )
    
    read, write, _ = await client_ctx.__aenter__()
    
    # Initialize MCP session
    session = ClientSession(read, write)
    await session.__aenter__()
    await session.initialize()
    tools = await load_mcp_tools(session)
    
    agent = create_agent(init_config["llm"], tools)
    
    return agent, session, client_ctx  # Return all three for cleanup

@app.websocket("/agent/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for the agent.
    
    Accepts a WebSocket connection, sets up the agent and
    handles the back-and-forth communication with the client.
    """
    await websocket.accept()
    logging.debug("ws connection opened")
    rancher_agent, session, client_ctx = await create_rancher_agent()
    rancher_agent_sub = SubAgent(
        name="rancher-agent",
        description="Agent specialized in managing Rancher and Kubernetes resources through the Rancher UI.",
        agent=rancher_agent
    )
    suse_observability_agent, obs_session, obs_client_ctx = await create_observability_agent()
    suse_observability_agent_sub = SubAgent(
        name="suse-observability-agent",
        description="Agent specialized in managing SUSE Observability resources. Use it to help with questions related to monitoring, metrics, logging, and tracing within Kubernetes clusters.",
        agent=suse_observability_agent
    )

    math_agent_sub = SubAgent(
            name="math-agent",
            description="Agent that can help with math",
            agent=create_math_agent()
        )
    parent_agent = create_parent_agent(llm=init_config["llm"],subagents=[rancher_agent_sub, math_agent_sub, suse_observability_agent_sub],checkpointer=InMemorySaver())
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
            
            prompt, context = _parse_websocket_request(request)
            if context:
                context_prompt = ". Use the following parameters to populate tool calls when appropriate. \n Only include parameters relevant to the user’s request (e.g., omit namespace for cluster-wide operations). \n Parameters (separated by ;): \n "
                for key, value in context.items():
                    context_prompt += f"{key}:{value};"
                prompt += context_prompt

            await stream_agent_response(
                agent=parent_agent,
                input_data={"messages": [{"role": "user", "content": prompt}]},
                config=config,
                websocket=websocket)
        except WebSocketDisconnect:
            logging.info(f"Client {websocket.client.host} disconnected.")
            break
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(f'<error>{{"message": "{str(e)}"}}</error>')
            else:
                break
        finally:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text("</message>")
    
    # Close MCP session and client after WebSocket loop ends
    await session.__aexit__(None, None, None)
    await client_ctx.__aexit__(None, None, None)
    await obs_session.__aexit__(None, None, None)
    await obs_client_ctx.__aexit__(None, None, None)
    logging.debug("ws connection closed")

# This is the UI for testing. This will be replaced by the UI extension
@app.get("/agent")
async def get(request: Request):
    """Serves the main HTML page for the chat client."""
    with open("src/index.html") as f:
        html_content = f.read()
        modified_html = html_content.replace("{{ url }}", request.url.hostname)

    return HTMLResponse(modified_html)

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
    async for event in agent.astream_events(
        input_data,
        config=config,
        stream_mode=["updates", "messages", "custom"]
    ):
        print(f"event: {event}")
        if event["event"] == "on_chat_model_stream":
            if event["data"]["chunk"].content:
                # TODO filter by node! summary and subagent choice should not be sent to the user
                await websocket.send_text(_extract_text_from_chunk_content(event["data"]["chunk"].content))
        """ if event == "messages":
            chunk, metadata = data
            if metadata.get("langgraph_node") == "agent" and chunk.content:
                await websocket.send_text(_extract_text_from_chunk_content(chunk.content))

        if event == "updates":
            if interrupt_value := data.get("__interrupt__"):
                await websocket.send_text(interrupt_value[0].value)
                # Receive user response for the human verification
                user_response = await websocket.receive_text()
                await stream_agent_response(
                    agent=agent,
                    input_data=Command(resume={"response": user_response}),
                    config=config,
                    websocket=websocket)
                
        if event == "custom":
            await websocket.send_text(data) """
    

def get_llm() -> BaseLanguageModel:
    """
    Selects and returns a language model instance based on environment variables.
    
    Returns:
        An instance of a language model.
        
    Raises:
        ValueError: If no supported model or API key is configured.
    """

    model = os.environ.get("MODEL")
    if not model:
        raise ValueError("LLM Model not configured.")
    
    active = os.environ.get("ACTIVE_LLM", "")
    ollama_url = os.environ.get("OLLAMA_URL")
    gemini_key = os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_url = os.environ.get("OPENAI_URL")

    if active == "ollama":
        return ChatOllama(model=model, base_url=ollama_url)
    if active == "gemini":
        return ChatGoogleGenerativeAI(model=model)
    if active == "openai":
        if openai_url:
            return ChatOpenAI(model=model, base_url=openai_url)
        else:
            return ChatOpenAI(model=model)

    # default order if active is not specified
    if ollama_url:
        return ChatOllama(model=model, base_url=ollama_url)
    if gemini_key:
        return ChatGoogleGenerativeAI(model=model)
    if openai_key:
        if openai_url:
            return ChatOpenAI(model=model, base_url=openai_url)
        else:
            return ChatOpenAI(model=model)

    raise ValueError("LLM not configured.")

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

def _parse_websocket_request(request: str) -> tuple[str, dict]:
    """
    Parses the incoming websocket request.

    The request can be a JSON string with 'prompt' and 'context' keys,
    or a plain text string.

    Args:
        request: The raw request string from the websocket.

    Returns:
        A tuple containing the prompt (str) and the context (dict).
    """
    try:
        json_request = json.loads(request)
        prompt = json_request.get("prompt", "")
        context = json_request.get("context", {})
        
        return prompt, context
    except json.JSONDecodeError:
        return request, {}

# This will be removed once https://github.com/modelcontextprotocol/python-sdk/pull/1177 is merged
class SimpleTruststore:
    def get_default(self):
        """Get the default Python truststore"""
        return certifi.where()

    def create_combined(self, company_cert_path, output_path):
        """Create truststore with public CAs + company cert"""
        with open(output_path, "w") as combined:
            # Add public CAs 
            with open(certifi.where(), "r") as public_cas:
                combined.write(public_cas.read())

            # Add MCP self-signed cert
            with open(company_cert_path, "r") as company:
                combined.write("\n" + company.read())

        return output_path
    
    def use_truststore(self, truststore_path):
        """Set the global truststore"""
        os.environ["SSL_CERT_FILE"] = truststore_path

    def set_truststore(self):
        company_cert_path = "/etc/tls/tls.crt"
        output_path = "/combined.crt"
        truststore_path = self.create_combined(
            company_cert_path=company_cert_path, output_path=output_path
        )
        self.use_truststore(truststore_path=truststore_path)
