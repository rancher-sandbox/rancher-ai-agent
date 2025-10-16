import logging
import os
import json
import uuid
import httpx

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama 
from langchain_mcp_adapters.tools import load_mcp_tools
from agents import create_k8s_agent, Context, init_rag_rancher
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import Langfuse
from langgraph.store.memory import InMemoryStore
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from langchain_openai import OpenAI
from langchain_core.language_models.llms import BaseLanguageModel
from contextlib import asynccontextmanager
from langchain.tools.retriever import create_retriever_tool
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langfuse.langchain import CallbackHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

init_config = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_config["llm"] = get_llm()
        logging.info(f"Using model: {init_config['llm']}")
        # if ENABLE_RAG flag is set, initialize the RAG retriever tool
        if os.environ.get("ENABLE_RAG", "false").lower() == "true":
            retriever = init_rag_rancher(get_llm_embeddings())
            init_config["retriever_tool"] = create_retriever_tool(
                retriever,
                "retrieve_rancher_docs",
                "Search and return relevant passages from local Rancher/SUSE documentation.",
            )
    except ValueError as e:
        logging.critical(e)
        raise e
    yield
    init_config.clear()

app = FastAPI(lifespan=lifespan)

@app.websocket("/agent/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for the agent.
    
    Accepts a WebSocket connection, sets up the agent and
    handles the back-and-forth communication with the client.
    """
    await websocket.accept()
    cookies = websocket.cookies
    rancher_url = "https://"+websocket.url.hostname
    if websocket.url.port:
        rancher_url += ":"+str(websocket.url.port)

    async with streamablehttp_client(
        url="http://rancher-mcp-server",
        headers={
             "R_token":str(cookies.get("R_SESS")),
             "R_url":rancher_url
             }
    ) as (read, write, _):
        # This will create one mcp connection for each websocket connection. This is needed because we need to pass the rancher token in the header.
        async with ClientSession(read, write) as session:
            await session.initialize()
            thread_id = str(uuid.uuid4())
            tools = await load_mcp_tools(session)
            
            # if ENABLE_RAG is true, add the retriever tool to the tools list
            if os.environ.get("ENABLE_RAG", "false").lower() == "true":
                tools = [init_config["retriever_tool"]] + tools
            langfuse_handler = CallbackHandler()

            agent = create_k8s_agent(init_config["llm"], tools, get_system_prompt(), InMemorySaver())
            
            config = {
                "callbacks": [langfuse_handler],
                "thread_id": thread_id
            }

            try:
                while True:
                    request = await websocket.receive_text()
                    context = {}
                    try:
                        json_request = json.loads(request)
                        context = json_request["context"]
                        prompt = json_request["prompt"]
                    except json.JSONDecodeError:
                        prompt = request

                    await stream_agent_response(
                        agent=agent,
                        input_data={"messages": [{"role": "user", "content": prompt}]},
                        config=config,
                        websocket=websocket,
                        context=context)
            except WebSocketDisconnect:
                logging.info(f"Client {websocket.client.host} disconnected.")
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                await websocket.close()

# This is the UI for testing. This will be replaced by the UI extension
@app.get("/agent")
async def get(request: Request):
    """Serves the main HTML page for the chat client."""
    with open("index.html") as f:
        html_content = f.read()
        modified_html = html_content.replace("{{ url }}", request.url.hostname)

    return HTMLResponse(modified_html)

async def stream_agent_response(
    agent: CompiledStateGraph,
    input_data: dict[str, list[dict[str, str]]],
    config: dict,
    websocket: WebSocket,
    context: dict,
) -> None:
    """
    Streams the agent's response to a WebSocket connection, handling interruptions.
    
    Args:
        agent: The compiled LangGraph agent.
        input_data: The input data for the agent's run.
        config: The run configuration.
        websocket: The WebSocket connection.
        context: The context for default tool parameter values.
        stream_mode: The types of events to stream from the agent.
    """

    await websocket.send_text("<message>")
    try:
        async for event, data in agent.astream(
            input_data,
            config=config,
            stream_mode=["updates", "messages", "custom"],
            context=Context(context=context)
        ):
            if event == "messages":
                chunk, metadata = data
                if metadata.get("langgraph_node") == "agent" and chunk.content:
                    await websocket.send_text(chunk.content)
            if event == "updates":
                if interrupt_value := data.get("__interrupt__"):
                    await websocket.send_text(interrupt_value[0].value)
                    # Receive user response for the human verification
                    user_response = await websocket.receive_text()
                    await stream_agent_response(
                        agent=agent,
                        input_data=Command(resume={"response": user_response}),
                        config=config,
                        websocket=websocket,
                        context=context)
            if event == "custom":
                await websocket.send_text(data)
    finally:
        await websocket.send_text("</message>")

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

    ollama_url = os.environ.get("OLLAMA_URL")
    if ollama_url:
        return ChatOllama(model=model, base_url=ollama_url)
                          
    gemini_key = os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        return ChatGoogleGenerativeAI(model=model)
    
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return OpenAI(model=model)
    
    raise ValueError("LLM not configured.")

def get_llm_embeddings() -> Embeddings:
    """
    Selects and returns an embedding model instance based on environment variables.

    Returns:
        An instance of a LangChain embedding model that implements the Embeddings interface.

    Raises:
        ValueError: If a required environment variable (like EMBEDDING_MODEL for Ollama) is missing,
                    or if no supported embedding provider is configured at all.
    """

     # Provider 1: Ollama
    ollama_url = os.environ.get("OLLAMA_URL")
    embedding_model_name = os.environ.get("EMBEDDINGS_MODEL")
    if not embedding_model_name:
            raise ValueError("EMBEDDINGS_MODEL must be set.")
    if ollama_url:
        return OllamaEmbeddings(model=embedding_model_name, base_url=ollama_url)

    # Provider 2: Google Gemini
    gemini_key = os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        return GoogleGenerativeAIEmbeddings(model=embedding_model_name)

    # Provider 3: OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
            return OpenAIEmbeddings(model=embedding_model_name)

    raise ValueError("No embedding provider configured. Set OLLAMA_URL, GOOGLE_API_KEY, or OPENAI_API_KEY.")

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
  - Format: SUGGESTIONS: [suggestion1] | [suggestion2] | [suggestion3]
  - No markdown, no numbering, under 60 characters each.
  - The first two suggestions must be directly relevant to the current context. If none fallback to the next rule.
  - The third suggestion should be a 'discovery' action. It introduces a related but broader Rancher or Kubernetes topic, helping the user learn.
Examples: SUGGESTIONS: How do I scale this deployment? | Check the resource usage for this cluster | Show me the logs for the failing pod
"""