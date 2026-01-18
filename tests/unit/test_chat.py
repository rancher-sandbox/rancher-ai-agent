import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import status, HTTPException
from app.routers import chat as chat_router
import json

@pytest.fixture
def mock_request():
    req = MagicMock()
    req.app.memory_manager = AsyncMock()
    req.cookies = {"R_SESS": "token"}
    req.headers = {"Host": "localhost"}
    return req

@pytest.mark.asyncio
async def test_get_chats_success(mock_request):
    mock_request.app.memory_manager.fetch_chats.return_value = [
        {"id": "1", "userId": "u", "createdAt": 2},
        {"id": "2", "userId": "u", "createdAt": 1},
    ]
    with patch("app.routers.chat.get_user_id_from_request", AsyncMock(return_value="u")):
        resp = await chat_router.get_chats(mock_request)
        assert resp.status_code == status.HTTP_200_OK
        assert resp.body

@pytest.mark.asyncio
async def test_get_chats_sort_asc(mock_request):
    mock_request.app.memory_manager.fetch_chats.return_value = [
        {"id": "1", "userId": "u", "createdAt": 2},
        {"id": "2", "userId": "u", "createdAt": 1},
    ]
    with patch("app.routers.chat.get_user_id_from_request", AsyncMock(return_value="u")):
        resp = await chat_router.get_chats(mock_request, sort="createdAt:asc")
        result = json.loads(resp.body)
        assert result[0]["createdAt"] == 1
        assert result[1]["createdAt"] == 2

@pytest.mark.asyncio
async def test_get_chat_not_found(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = None
    with patch("app.routers.chat.get_user_id_from_request", AsyncMock(return_value="u")):
        with pytest.raises(HTTPException) as exc:
            await chat_router.get_chat(mock_request, chat_id="notfound")
        assert exc.value.status_code == status.HTTP_404_NOT_FOUND

@pytest.mark.asyncio
async def test_update_chat_success(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = {"id": "1", "userId": "u"}
    mock_request.app.memory_manager.update_chat.return_value = {"id": "1", "userId": "u", "name": "updated"}
    with patch("app.routers.chat.get_user_id_from_request", AsyncMock(return_value="u")):
        resp = await chat_router.update_chat(mock_request, chat_id="1", chat_data={"name": "updated"})
        assert resp.status_code == status.HTTP_200_OK
        assert b"updated" in resp.body

@pytest.mark.asyncio
async def test_delete_chat_success(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = {"id": "1", "userId": "u"}
    mock_request.app.memory_manager.delete_chat.return_value = None
    with patch("app.routers.chat.get_user_id_from_request", AsyncMock(return_value="u")):
        resp = await chat_router.delete_chat(mock_request, chat_id="1")
        assert resp.status_code == status.HTTP_204_NO_CONTENT

@pytest.mark.asyncio
async def test_get_chat_messages_not_found(mock_request):
    mock_request.app.memory_manager.fetch_chat.return_value = None
    with patch("app.routers.chat.get_user_id_from_request", AsyncMock(return_value="u")):
        with pytest.raises(HTTPException) as exc:
            await chat_router.get_chat_messages(mock_request, chat_id="notfound")
        assert exc.value.status_code == status.HTTP_404_NOT_FOUND
