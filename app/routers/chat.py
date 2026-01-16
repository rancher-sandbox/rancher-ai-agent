import logging
import os
from urllib.parse import urlparse
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response, JSONResponse

from ..services.auth import get_user_id

async def get_user_id_from_request(request: Request) -> str:
    """
    Retrieves the user ID from the Rancher API using the session token from the request cookies.
    """
    rancher_url = os.environ.get("RANCHER_URL", "")
    token = request.cookies.get("R_SESS")

    host = ""
    if not token:
        logging.warning("R_SESS cookie not found")
        return None

    if rancher_url:
        parsed = urlparse(rancher_url)
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc
        host = f"{scheme}://{netloc}"
    else:
        rancher_host = request.headers.get("Host", "localhost")
        host = f"https://{rancher_host}"

    return await get_user_id(host, token)

router = APIRouter(prefix="/agent/api", tags=["chats"])

@router.get("/chats")
async def get_chats(request: Request):
    """
    TODO: add filtering by tags from query parameters.
    Get all threads that have at least one user message and return them as chats list.

    Returns:
        A list of chat objects.
    """
    user_id = await get_user_id_from_request(request)

    try:
        chats = await request.app.db_manager.fetch_chats(
            user_id=user_id
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=chats
        )

    except Exception as e:
        logging.error(f"Error fetching chats for user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )
        
@router.delete("/chats")
async def delete_chats(request: Request) -> JSONResponse:
    """
    Delete all chats for the user.

    Returns:
        A success message and an HTTP status code.
    """
    
    user_id = await get_user_id_from_request(request)

    try:
        await request.app.db_manager.delete_chats(
            user_id=user_id
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    except Exception as e:
        logging.error(f"Error deleting chats for user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )

@router.get("/chats/{chat_id}")
async def get_chat(request: Request, chat_id: str) -> JSONResponse:
    """
    Get a chat by ID.

    Args:
        chat_id: The ID of the thread.

    Returns:
        A chat object and an HTTP status code.
    """
    
    user_id = await get_user_id_from_request(request)

    if not chat_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chat_id is required")
    
    try:
        chat = await request.app.db_manager.fetch_chat(
            chat_id=chat_id,
            user_id=user_id
        )
        if not chat:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=chat
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching chat for chat_id {chat_id} and user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )
        
@router.put("/chats/{chat_id}")
async def update_chat(request: Request, chat_id: str, chat_data: dict) -> JSONResponse:
    """
    Update a chat.
    Args:
        chat_id: The ID of the thread.
        chat_data: The chat data to update.
    Returns:
        The updated chat object and an HTTP status code.
    """
    
    user_id = await get_user_id_from_request(request)

    if not chat_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chat_id is required")
    
    if not chat_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chat_data is required")

    try:
        chat = await request.app.db_manager.fetch_chat(
            chat_id=chat_id,
            user_id=user_id
        )
        if not chat:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

        updated_chat = await request.app.db_manager.update_chat(
            chat_id=chat_id,
            user_id=user_id,
            chat_data=chat_data
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=updated_chat
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error updating chat for chat_id {chat_id} and user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )

@router.delete("/chats/{chat_id}")
async def delete_chat(request: Request, chat_id: str) -> JSONResponse:
    """
    Delete a specific chat by ID.

    Args:
        chat_id: The ID of the thread.

    Returns:
        A success message and an HTTP status code.
    """
    
    user_id = await get_user_id_from_request(request)

    if not chat_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chat_id is required")
    
    try:
        await request.app.db_manager.delete_chat(
            chat_id=chat_id,
            user_id=user_id
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting chat for chat_id {chat_id} and user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )

@router.get("/chats/{chat_id}/messages")
async def get_chat_messages(request: Request, chat_id: str) -> JSONResponse:
    """
    TODO: add filtering by tags from query parameters.
    Get messages for a specific thread.

    Args:
        chat_id: The ID of the thread.
    Returns:
        A list of message objects and an HTTP status code.
    """
    
    user_id = await get_user_id_from_request(request)

    if not chat_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chat_id is required")
    
    try:
        chat = await request.app.db_manager.fetch_chat(
            chat_id=chat_id,
            user_id=user_id
        )
        if not chat:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")

        messages = await request.app.db_manager.fetch_messages(
            chat_id=chat_id,
            user_id=user_id
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=messages
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching messages for chat_id {chat_id} and user_id {user_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )
