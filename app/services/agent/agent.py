import os
import logging

from contextlib import asynccontextmanager, AsyncExitStack
from dataclasses import dataclass
from fastapi import  WebSocket
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph.state import CompiledStateGraph
from langchain_core.language_models.llms import BaseLanguageModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from .child import create_child_agent
from ..rag import fleet_documentation_retriever, rancher_documentation_retriever

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

@dataclass
class AgentContext:
    agent: CompiledStateGraph
    session: ClientSession
    client_ctx: any

@asynccontextmanager
async def create_agent(llm: BaseLanguageModel, websocket: WebSocket):
    """
    TODO multiagent support

    Context manager that creates and manages agent lifecycle.
    """
    async with _create_rancher_core_agent(llm=llm, websocket=websocket) as agent_ctx:
        yield agent_ctx

@asynccontextmanager
async def _create_rancher_core_agent(llm: BaseLanguageModel, websocket: WebSocket):
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
        
        # Initialize checkpointer for persisting agent state or fall back to in-memory
        checkpointer = InMemorySaver()
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            try:
                checkpointer = await stack.enter_async_context(
                    AsyncPostgresSaver.from_conn_string(db_url)
                )
                # Initialize database schema
                await checkpointer.setup()
                logging.info("Using PostgreSQL checkpointer for agent state.")
            except Exception as e:
                logging.warning(f"Failed to connect to PostgreSQL ({e}), falling back to in-memory-saver checkpointer")

        agent = create_child_agent(llm, tools, _get_system_prompt(), checkpointer)
        
        yield AgentContext(
            agent=agent,
            session=session,
            client_ctx=client_ctx
        )

    # All contexts automatically cleaned up here in reverse order since we used AsyncExitStack

def _get_system_prompt() -> str:
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
    
    return RANCHER_AGENT_PROMPT

