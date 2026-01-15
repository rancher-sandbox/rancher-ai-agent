import logging
import httpx
from urllib.parse import urlparse

async def get_user_id(host: str, token: str) -> str:
    """
    Retrieves the user ID from the Rancher API using the session token.
    """
    url = f"{host}/v3/users?me=true"
    try:
        async with httpx.AsyncClient(timeout=5.0, verify=False) as client:
            resp = await client.get(url, headers={
                "Cookie": f"R_SESS={token}",
                "Accept": "application/json",
            })
            payload = resp.json()
            
            user_id = payload["data"][0]["id"]
            
            if user_id:
                logging.info("user API returned: %s - userId %s", resp.status_code, user_id)

                return user_id
    except Exception as e:
        logging.error("user API call failed: %s", e)

    return None