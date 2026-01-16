"""Built-in agent configurations."""

import json
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class AuthenticationType(str, Enum):
    """Authentication types for agents."""
    NONE = "NONE"
    RANCHER = "RANCHER"
    BASIC = "BASIC"


class ToolActionType(str, Enum):
    """Action types for human validation tools."""
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


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


class HumanValidationTool(BaseModel):
    name: str
    type: ToolActionType

class AgentConfig(BaseModel):
    """Configuration for a single agent."""
    name: str 
    description: str 
    system_prompt: str 
    mcp_url: str
    authentication: AuthenticationType = AuthenticationType.NONE
    toolset: Optional[str] = None
    human_validation_tools: list[HumanValidationTool] = []

WEATHER_AGENT = AgentConfig(
    name="Weather Agent",
    description="Provides weather information for a given location",
    system_prompt="answer the user",
    mcp_url="http://localhost:8001/mcp"
)

MATH_AGENT = AgentConfig(
    name="Math Agent",
    description="Performs mathematical calculations and problem solving",
    system_prompt="answer the user",
    mcp_url="http://localhost:8002/mcp"
)

RANCHER_AGENT = AgentConfig(
    name="Rancher Agent",
    description="Manages Rancher resources and operations",
    system_prompt=RANCHER_AGENT_PROMPT,
    mcp_url="rancher-mcp-server.cattle-ai-agent-system.svc",
    authentication=AuthenticationType.RANCHER,
    human_validation_tools=[
        HumanValidationTool(name="CREATE", type=ToolActionType.CREATE),
        HumanValidationTool(name="UPDATE", type=ToolActionType.UPDATE),
    ]
)

BUILTIN_AGENTS = [WEATHER_AGENT, MATH_AGENT, RANCHER_AGENT]

def parse_agents(json_str: str) -> list[AgentConfig]:
    """Parse JSON string into a list of AgentConfig objects."""
    data = json.loads(json_str)
    return [AgentConfig(**agent) for agent in data]
