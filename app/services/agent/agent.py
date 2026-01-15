import os
import logging

from contextlib import asynccontextmanager, AsyncExitStack
from dataclasses import dataclass
from fastapi import  WebSocket
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph.state import CompiledStateGraph
from langchain_core.language_models.llms import BaseLanguageModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from .child import create_child_agent
from .chat import create_chat_agent
from ..rag import fleet_documentation_retriever, rancher_documentation_retriever
from ...types import RequestType

RANCHER_AGENT_PROMPT = """You are a helpful and expert AI assistant integrated directly into the Rancher UI. Your primary goal is to assist users in managing their Kubernetes clusters and resources through the Rancher interface. You are a trusted partner, providing clear, confident, and safe guidance.

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

SUMMARY_PROMPT = """Each message is a list of recent agent replies to the user. Your task is to generate a concise summary of these replies, focusing on key points and relevant information. Your response will be used to assign a title to a Chat.

## CORE DIRECTIVES

### Summary Focus
* Focus on user requests FIRST. The summary should reflect what the user asked for.
* The summary should capture the essence of requests, not what the agent replied.
  For example:
    * The user asked for "How is the weather in Florence?"
        * Good summary: "Weather in Florence"
        * Bad summary: "Can't answer weather questions"

### Conciseness
* The summary MUST BE MAX 40 characters.
* Summarize the content in a brief manner, highlighting only the most important aspects.
* Avoid unnecessary details or lengthy explanations.

### Consistency
* DO NOT include greetings or pleasantries in the summary.
* DO NOT include tags like <message>, </message> or any other keywords between < and >.
* DO NOT include question marks or suggestions in the summary.
* DO NOT include periods at the end of the summary.
* DO NOT include any special characters like '-', '_', or other symbols that humans usually use.
"""

@dataclass
class AgentContext:
    agent: CompiledStateGraph
    session: ClientSession
    client_ctx: any

@asynccontextmanager
async def create_agent(llm: BaseLanguageModel, websocket: WebSocket, request_type: RequestType, checkpointer = None):
    """
    TODO multiagent support

    Context manager that creates and manages agent lifecycle.
    """
    async with _create_rancher_core_agent(llm=llm, websocket=websocket, request_type=request_type, checkpointer=checkpointer) as agent_ctx:
        yield agent_ctx

@asynccontextmanager
async def _create_rancher_core_agent(llm: BaseLanguageModel, websocket: WebSocket, request_type: RequestType, checkpointer = None):
    """
    Creates a Rancher core agent with MCP client connection.
    
    This function sets up an agent specialized in managing Rancher and Kubernetes resources
    through the Rancher UI. It establishes a connection to the MCP server for tool execution
    and properly manages the lifecycle of the client connection.
    
    Args:
        llm: The language model to use for the agent.
        websocket: WebSocket connection to extract Rancher URL and authentication token.
    
    Yields:
        AgentContext containing:
            - agent: The compiled LangGraph agent ready to process requests
            - session: MCP ClientSession that needs to be closed after use
            - client_ctx: MCP client context manager that needs to be closed after use
    """
    cookies = websocket.cookies
    rancher_url = os.environ.get("RANCHER_URL","https://"+websocket.url.hostname)
    token = os.environ.get("RANCHER_API_TOKEN", cookies.get("R_SESS", ""))
    mcp_url = os.environ.get("MCP_URL", "rancher-mcp-server.cattle-ai-agent-system.svc")
    if os.environ.get('INSECURE_SKIP_TLS', 'false').lower() == "true":
        mcp_url = "http://" + mcp_url
    else:
        mcp_url = "https://" + mcp_url

    async with AsyncExitStack() as stack:
        client_ctx = await stack.enter_async_context(
            streamablehttp_client(
                url=mcp_url,
                headers={
                    "R_token": token,
                    "R_url": rancher_url
                }
            )
        )
        
        read, write, _ = client_ctx
        session = await stack.enter_async_context(ClientSession(read, write))
        
        # Initialize MCP session
        await session.initialize()
        tools = await load_mcp_tools(session)
        
        # if ENABLE_RAG is true, add the retriever tools to the tools list
        if os.environ.get("ENABLE_RAG", "false").lower() == "true":
            tools = [fleet_documentation_retriever, rancher_documentation_retriever] + tools
        
        # Initialize checkpointer for persisting agent state in Postgres DB if enabled
        if not checkpointer and websocket.app.db_manager:
            try:
                checkpointer = await stack.enter_async_context(
                    AsyncPostgresSaver.from_conn_string(websocket.app.db_manager.db_url)
                )
                logging.debug("Using PostgreSQL checkpointer for agent state.")
            except Exception as e:
                logging.warning(f"Using in-memory-saver checkpointer due to error initializing PostgreSQL saver: {e}")

        if not checkpointer:
            checkpointer = InMemorySaver()
            logging.debug("Using in-memory checkpointer for agent state.")

        agent = create_child_agent(llm, tools, _get_system_prompt(request_type), checkpointer)
        
        yield AgentContext(
            agent=agent,
            session=session,
            client_ctx=client_ctx
        )

    # All contexts automatically cleaned up here in reverse order since we used AsyncExitStack

def create_rest_api_agent(checkpointer):
    """
    Creates a chat agent for REST API endpoints.
    
    This is a minimal agent creation for REST API use cases where
    only reading chat state is needed (no LLM, tools, or MCP).
    
    Args:
        checkpointer: The checkpointer for reading agent state.
    
    Returns:
        CompiledStateGraph: The compiled agent ready to read state.
    """
    return create_chat_agent(checkpointer)

def _get_system_prompt(type: RequestType) -> str:
    """
    Retrieves the system prompt for the AI agent.

    The function first attempts to get the prompt from an environment variable
    named "SYSTEM_PROMPT". If the environment variable is not set, it returns
    a default, hard-coded prompt depending on the request type.

    Returns:
        str: The system prompt to be used by the AI agent.
    """
    match type:
        case RequestType.SUMMARY:
            return SUMMARY_PROMPT
        case RequestType.MESSAGE:
            prompt = os.environ.get("SYSTEM_PROMPT")
            if prompt:
                return prompt
            
            return RANCHER_AGENT_PROMPT
