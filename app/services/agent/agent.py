import os
import logging
import json
from contextlib import asynccontextmanager, AsyncExitStack
from dataclasses import dataclass

from .child import create_child_agent
from .parent import create_parent_agent, ChildAgent
from ..rag import fleet_documentation_retriever, rancher_documentation_retriever
from fastapi import  WebSocket
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph.state import CompiledStateGraph
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models.llms import BaseLanguageModel

@dataclass
class AgentConfig:
    """Configuration for a child agent."""
    name: str
    description: str
    system_prompt: str
    mcp_url: str
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AgentConfig':
        """Create an AgentConfig from a dictionary."""
        return cls(
            name=data.get("agent", ""),
            description=data.get("description", ""),
            system_prompt=data.get("systemPrompt", ""),
            mcp_url=data.get("mcpURL", "")
        )

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

#TODO this is only for testing. Remove!!
MULTI_AGENT= """
[
  {
  "agent": "Weather Agent",
  "description": "Provides weather information for a given location",
  "systemPrompt": "answer the user",
  "mcpURL": "http://localhost:8001/mcp"
  },
  {
  "agent": "Math Agent",
  "description": "Performs mathematical calculations and problem solving",
  "systemPrompt": "answer the user",
  "mcpURL": "http://localhost:8002/mcp"
  }
]
"""

def parse_agent_configs(json_str: str) -> list[AgentConfig]:
    """
    Parse JSON string into a list of AgentConfig objects.
    
    Args:
        json_str: JSON string containing agent configurations
    
    Returns:
        List of AgentConfig objects
    """
    try:
        data = json.loads(json_str)
        return [AgentConfig.from_dict(agent_dict) for agent_dict in data]
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse agent configs: {e}")
        return []
    except Exception as e:
        logging.error(f"Error creating agent configs: {e}")
        return []

@asynccontextmanager
async def create_agent(llm: BaseLanguageModel, websocket: WebSocket):
    agents_config = parse_agent_configs(os.environ.get("AGENTS_CONFIG", MULTI_AGENT))
    print(agents_config)
    if len(agents_config) > 0:
        logging.info("Creating parent agent with child agents: " + ", ".join([agent.name for agent in agents_config]))
        async with AsyncExitStack() as stack:
            tools = await _create_rancher_core_mcp_tools(stack, llm, websocket)
            rancher_agent = ChildAgent(
                name="Rancher Core Agent",
                description="Handles Rancher core functionality and Kubernetes resource management",
                agent=create_child_agent(llm, tools, _get_system_prompt(), InMemorySaver())
            )
            child_agents = []
            for agent_cfg in agents_config:
                tools = await _create_mcp_tools(stack, llm, agent_cfg.mcp_url)
                child_agents.append(ChildAgent(
                    name=agent_cfg.name,
                    description=agent_cfg.description,
                    agent=create_child_agent(llm, tools, agent_cfg.system_prompt, InMemorySaver())
                ))
            
            parent_agent = create_parent_agent(llm, child_agents + [rancher_agent], InMemorySaver())
            yield parent_agent
    else:
        logging.info("single agent" )
        async with AsyncExitStack() as stack:
            tools = await _create_rancher_core_mcp_tools(stack, llm, websocket)
            agent = create_child_agent(llm, tools, _get_system_prompt(), InMemorySaver())
            
            yield agent

async def _create_rancher_core_mcp_tools(stack: AsyncExitStack, llm: BaseLanguageModel, websocket: WebSocket) -> list:
    """
    Creates a Rancher core agent by connecting to MCP server and loading tools.
    
    Args:
        stack: AsyncExitStack to manage async context managers
        llm: The language model to use for the agent
        websocket: WebSocket connection to extract Rancher URL and authentication token
    
    Returns:
        List of tools loaded from the MCP server
    """
    cookies = websocket.cookies
    rancher_url = os.environ.get("RANCHER_URL","https://"+websocket.url.hostname)
    token = os.environ.get("RANCHER_API_TOKEN", cookies.get("R_SESS", ""))
    mcp_url = os.environ.get("MCP_URL", "rancher-mcp-server.cattle-ai-agent-system.svc")
    if os.environ.get('INSECURE_SKIP_TLS', 'false').lower() == "true":
        mcp_url = "http://" + mcp_url
    else:
        mcp_url = "https://" + mcp_url

    read, write, _ = await stack.enter_async_context(
        streamablehttp_client(
            url=mcp_url,
            headers={
                "R_token": token,
                "R_url": rancher_url
            }
        )
    )
    
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    tools = await load_mcp_tools(session)

    # if ENABLE_RAG is true, add the retriever tools to the tools list
    if os.environ.get("ENABLE_RAG", "false").lower() == "true":
        tools = [fleet_documentation_retriever, rancher_documentation_retriever] + tools

    return tools

async def _create_mcp_tools(stack: AsyncExitStack, llm: BaseLanguageModel, mcp_url: str) -> list:
    """
    Creates a Rancher core agent by connecting to MCP server and loading tools.
    
    Args:
        stack: AsyncExitStack to manage async context managers
        llm: The language model to use for the agent
        websocket: WebSocket connection to extract Rancher URL and authentication token
    
    Returns:
        List of tools loaded from the MCP server
    """

    read, write, _ = await stack.enter_async_context(
        streamablehttp_client(
            url=mcp_url,
        )
    )
    
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    tools = await load_mcp_tools(session)

    # if ENABLE_RAG is true, add the retriever tools to the tools list
    if os.environ.get("ENABLE_RAG", "false").lower() == "true":
        tools = [fleet_documentation_retriever, rancher_documentation_retriever] + tools

    return tools


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

