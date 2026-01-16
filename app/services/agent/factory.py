import os
import logging
import json
import httpx
from contextlib import asynccontextmanager, AsyncExitStack
from dataclasses import dataclass

from .builtin_agents import BUILTIN_AGENTS, RANCHER_AGENT, AuthenticationType
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
    if len(BUILTIN_AGENTS) > 0:
        logging.info("Multi-agent setup detected, creating parent agent.")        
        async with AsyncExitStack() as stack:
            child_agents = []
            for agent_cfg in BUILTIN_AGENTS:
                tools = await _create_mcp_tools(stack, websocket, agent_cfg)
                child_agents.append(ChildAgent(
                    name=agent_cfg.name,
                    description=agent_cfg.description,
                    agent=create_child_agent(llm, tools, agent_cfg.system_prompt, InMemorySaver())
                ))
            parent_agent = create_parent_agent(llm, child_agents, InMemorySaver())

            yield parent_agent
    else:
        logging.info("Single agent" )
        
        async with AsyncExitStack() as stack:
            tools = await _create_mcp_tools(stack, websocket, RANCHER_AGENT)
            agent = create_child_agent(llm, tools, RANCHER_AGENT.system_prompt, InMemorySaver())
            
            yield agent

async def _create_mcp_tools(stack: AsyncExitStack, websocket: WebSocket, agent_config: AgentConfig) -> list:
    if agent_config.authentication == AuthenticationType.RANCHER:
        cookies = websocket.cookies
        rancher_url = os.environ.get("RANCHER_URL","https://"+websocket.url.hostname)
        token = os.environ.get("RANCHER_API_TOKEN", cookies.get("R_SESS", ""))
        mcp_url = os.environ.get("MCP_URL", agent_config.mcp_url)
        if os.environ.get('INSECURE_SKIP_TLS', 'false').lower() == "true":
            mcp_url = "http://" + mcp_url
        else:
            mcp_url = "https://" + mcp_url
        headers={
                "R_token": token,
                "R_url": rancher_url
            }
    else:
        mcp_url = agent_config.mcp_url
        headers = {}

    read, write, _ = await stack.enter_async_context(
        streamablehttp_client(
            url=mcp_url,
            headers=headers
        )
    )
    
    try:
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        tools = await load_mcp_tools(session)
    except Exception as e: #TODO not catching exception here!!!
        logging.error(f"Failed to connect to MCP server at {mcp_url}: {e}")
        logging.warning(f"Agent '{agent_config.name}' will run without MCP tools")
        tools = []

    if agent_config.name == "Rancher Core Agent" and os.environ.get("ENABLE_RAG", "false").lower() == "true":
        tools = [fleet_documentation_retriever, rancher_documentation_retriever] + tools

    return tools

