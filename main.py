import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from mcp import ClientSession, StdioServerParameters
from mcp.client.streamable_http import streamablehttp_client
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.tools import load_mcp_tools
from agents import create_k8s_agent, Context
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import Langfuse
from langgraph.prebuilt import create_react_agent
from langfuse.langchain import CallbackHandler
from langgraph.types import interrupt, Command
from typing import Callable
from langchain_core.tools import BaseTool, tool as create_tool
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from langgraph.prebuilt.interrupt import HumanInterruptConfig, HumanInterrupt
from langchain_openai import OpenAI
from langchain_core.language_models.llms import BaseLanguageModel

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = FastAPI()

def add_human_in_the_loop(
    tool: Callable | BaseTool,
    *,
    interrupt_config: HumanInterruptConfig = None,
) -> BaseTool:
    """Wrap a tool to support human-in-the-loop review."""
    if not isinstance(tool, BaseTool):
        tool = create_tool(tool)

    if interrupt_config is None:
        interrupt_config = {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
        }

    @create_tool(  # (1)!
        tool.name,
        description=tool.description,
        args_schema=tool.args_schema
    )
    async def call_tool_with_interrupt(config: RunnableConfig, **tool_input):
        request: HumanInterrupt = {
            "action_request": {
                "action": tool.name,
                "args": tool_input
            },
            "config": interrupt_config,
            "description": "The following patch is going to be applied in the cluster. \n "+str(tool_input["patch"])+"\nWARNING! This can't be reverted! Confirm? (yes/no)"
        }
        response = interrupt([request])   # (2)!
        # approve the tool call
        if response["type"] == "yes":
            tool_response = await tool.ainvoke(tool_input, config)
        else:
            return "the tool execution was not approved by the user."

        return tool_response

    return call_tool_with_interrupt



@app.websocket("/agent/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cookies = websocket.cookies
    url = "https://"+websocket.url.hostname

    async with streamablehttp_client(
        url="http://rancher-mcp-server",
        headers={
             "R_token":str(cookies.get("R_SESS")),
             "R_url":url
             }
    ) as (read, write, _):
        # This will create one mcp connection for each websocket connection. This is needed because we need to pass the rancher token in the header.
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            tools = [add_human_in_the_loop(t) if t.name == "patchKubernetesResource" else t for t in tools]

            #langfuse_handler = CallbackHandler()
            try:
                llm = get_llm()
            except ValueError as e:
                logging.error(e)
                await websocket.send_text(f"<message>") #TODO check with UI how to handle errors. This will change!
                await websocket.send_text(f"ERROR: {e}")
                await websocket.send_text(f"</message>")
            checkpointer = InMemorySaver()
            agent = create_k8s_agent(llm, tools, checkpointer, system_prompt= get_system_prompt())
            config = {
    #            "callbacks": [langfuse_handler],
                "thread_id": "thread-1"
            }
            context = {}
            try:
                while True:
                    data = await websocket.receive_text()
                    if (context_tuple := is_context_message(data)) is not None:
                        key, value = context_tuple
                        context[key] = value
                    else:
                        await websocket.send_text(f"<message>")
                        async for event, data in agent.astream(
                            {"messages": [{"role": "user", "content": data}],},
                            config=config,
                            stream_mode=["updates", "messages"],
                            context=Context(context=context)
                        ):
                            if event == "messages":
                                chunk, metadata = data 
                                if metadata["langgraph_node"] == "agent" and chunk.content != "":
                                    await websocket.send_text(chunk.content)
                            if event == "updates": # TODO move human validation to graph?
                                if (interrupt_value := data.get('__interrupt__')) is not None:
                                    await websocket.send_text(interrupt_value[0].value[0]["description"])
                                    response = await websocket.receive_text()
                                    await websocket.send_text(f"<message>")
                                    # resume after interrupt
                                    async for event, data in agent.astream(
                                        Command(resume={"type": response}),  
                                        config,
                                        stream_mode=["updates", "messages"],
                                        context=Context(context=context)
                                    ):
                                        if event == "messages":
                                            chunk, metadata = data 
                                            if metadata["langgraph_node"] == "agent" and chunk.content != "":
                                                await websocket.send_text(chunk.content)
                                    await websocket.send_text(f"</message>")
                        await websocket.send_text(f"</message>")
            except WebSocketDisconnect:
                logging.info(f"Client {websocket.client.host} disconnected.")

@app.get("/agent")
async def get(request: Request):
    """Serves the main HTML page for the chat client."""
    with open("index.html") as f:
        html_content = f.read()
    
    modified_html = html_content.replace("{{ url }}", request.url.hostname)

    return HTMLResponse(modified_html)

def is_context_message(message: str) -> tuple[str, str]  | None:
    # Check for the tags first
    if not (message.startswith("<mcp_context>") and message.endswith("</mcp_context>")):
        return None

    content = message.removeprefix("<mcp_context>").removesuffix("</mcp_context>")
    equal_sign_pos = content.find("=")
    
    if equal_sign_pos == -1:
        # No key-value pair found
        return None
    
    # Split the string into key and value
    key = content[:equal_sign_pos]
    value = content[equal_sign_pos + 1:]
    
    return (key, value)

def get_llm() -> BaseLanguageModel:
    model = os.environ.get("MODEL")
    if not model:
        raise ValueError("Model not configured.")

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

def get_system_prompt() -> str:
    prompt = os.environ.get("SYSTEM_PROMPT")
    if prompt:
        return prompt
    
    return """You are a helpful and expert AI assistant integrated directly into the Rancher UI. Your primary goal is to assist users in managing their Kubernetes clusters and resources through the Rancher interface. You are a trusted partner, providing clear, confident, and safe guidance.

## CORE DIRECTIVES

### UI-First Mentality
* NEVER suggest using `kubectl`, `helm`, or any other CLI tool.
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

## RESOURCE CREATION & MODIFICATION

* Always generate Kubernetes YAML in a markdown code block.
* Briefly explain the resource's purpose before showing YAML.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  namespace: default
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.25
        ports:
        - containerPort: 80

RESPONSE FORMAT
Summarize first: Provide a clear, human-readable overview of the resource's status or configuration.
The output should always be provided in Markdown format.

- Be concise: No unnecessary conversational fluff.  
- Always end with exactly three actionable suggestions:
  - Format: SUGGESTIONS: [suggestion1] | [suggestion2] | [suggestion3]
  - No markdown, no numbering, under 60 characters each.
  - Must be relevant to the current Rancher UI context.

Examples: SUGGESTIONS: How do I scale this deployment? | Check the resource usage for this cluster | Show me the logs for the failing pod
"""
