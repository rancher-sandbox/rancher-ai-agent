from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/agent")
async def get(request: Request):
    """ONLY FOR TESTING. This serves the test HTML page for the chat client."""
    with open("routers/testui.html") as f:
        html_content = f.read()
        modified_html = html_content.replace("{{ url }}", request.url.hostname)

    return HTMLResponse(modified_html)