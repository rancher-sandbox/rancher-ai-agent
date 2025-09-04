from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
import logging
from mcp import ClientSession, StdioServerParameters
from mcp.client.streamable_http import streamablehttp_client
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = FastAPI()

# TODO support multiple sessions
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
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            print(tools)
            #llm = ChatOllama(model="qwen3:4b")
            # Create and run the agent
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
            )
            memorysaver = InMemorySaver()
            prompt = "You are a helpful assistant that helps troubleshooting kubernetes clusters that are installed with Rancher."
            agent = create_react_agent(llm, tools, checkpointer=memorysaver, prompt=prompt)
            print("Agent created")
            try:
                while True:
                    # Wait for a message from the client
                    data = await websocket.receive_text()
                    await websocket.send_text(f"<message>")
                    async for token, metadata in agent.astream(
                        {"messages": [{"role": "user", "content": data}],},
                        {"configurable": {"thread_id": "thread-1"}},
                        stream_mode="messages",
                    ):
                        if metadata["langgraph_node"] != "tools":
                            await websocket.send_text(f"{token.content}")
                        print(metadata)
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
