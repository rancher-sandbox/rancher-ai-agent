from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()

# This is the UI for testing. This will be replaced by the UI extension
@router.get("/agent")
async def get(request: Request):
    """Serves the main HTML page for the chat client."""
    with open("app/index.html") as f:
        html_content = f.read()
        modified_html = html_content.replace("{{ url }}", request.url.hostname)

    return HTMLResponse(modified_html)