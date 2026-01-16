import os
import logging
from datetime import datetime
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.base import CheckpointTuple

from .agent.agent import create_rest_api_agent

class DatabaseManager:
    """
    Manages database connections and operations.
    """
    def __init__(self):
        DB_USER = os.environ.get("DB_USER", "postgres")
        DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
        DB_HOST = os.environ.get("DB_HOST", "localhost")
        DB_PORT = os.environ.get("DB_PORT", "5432")
        DB_NAME = os.environ.get("DB_NAME", "postgres")

        self.db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    async def initialize_database(self) -> None:
        """
        Initialize database schema.
        """

        try:
            # Initialize LangGraph schema
            async with AsyncPostgresSaver.from_conn_string(self.db_url) as checkpointer:
                await checkpointer.setup()
                logging.debug("PostgreSQL schema initialized")
        except Exception as e:
            logging.error(f"Failed to initialize database: {e}", exc_info=True)
            
    def _filter_by_tags(self, tag_filters: list[dict], tags: list, role: str) -> bool:
        """
        TODO: add custom tag filtering logic here.
        Check if any of the tags are in the excluded tags list.
        
        Args:
            tag_filters: list of tag filter dicts
            tags: list of tags to check
            role: "human" or "ai"
        Returns:
            bool: True if none of the tags are in the excluded tags, False otherwise.
        """
        for tag in tags:
            for excluded in tag_filters:
                for tag_role, excluded_tags in excluded.items():
                    if tag_role == role and tag in excluded_tags:
                        return False
        return True
    
    def _is_empty_chat(self, checkpoint_tuple: CheckpointTuple) -> bool:
        """
        Check if a chat is empty from the checkpoint tuple.
        A chat is considered empty if it does not contain any messages other than those with 'welcome' tag.
        """
        tag_filters = [
            { "human": ["welcome"] },
            { "ai": ["welcome"] },
        ]
        
        channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
        agent_metatdata = channel_values.get("agent_metadata", {})
        messages = channel_values.get("messages", [])
        tags = agent_metatdata.get("tags", [])

        is_empty = True
        for msg in messages:
            if self._filter_by_tags(tag_filters, tags, msg.type):
                is_empty = False
                break
            
        return is_empty
    
    def _get_chat_metadata(self, checkpointTuple: CheckpointTuple) -> dict:
        """
        Extract chat metadata from a checkpoint tuple.
        """
        channel_values = checkpointTuple.checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])

        name = ""
        created_at = None

        if messages and len(messages) > 0:
            # First message is used for chat metadata
            message = messages[0]

            additional_kwargs = message.additional_kwargs
            created_at = additional_kwargs.get("created_at") if additional_kwargs else None
            if created_at:
                name = f"Chat - {datetime.fromisoformat(created_at).strftime('%Y-%m-%d %H:%M')}"

        return {
            "name": name,
            "created_at": created_at
        }

    async def fetch_chats(self, user_id: str, filters: dict = {}) -> list:
        """
        Fetch chat threads from the database for a specific user.

        Args:
            user_id: The ID of the user.
            filters: A dictionary to filter chat threads.

        Returns:
            A list of chat thread records.
        """
        async with AsyncPostgresSaver.from_conn_string(self.db_url) as checkpointer:
            chat_list = []

            chat_ids = set()
            async for checkpoint_tuple in checkpointer.alist(config=None, filter={"user_id": user_id}):
                chat_id = checkpoint_tuple.config["configurable"]["thread_id"]
                user_id = checkpoint_tuple.metadata["user_id"]
                
                logging.debug(f"Processing checkpoint_tuple for chat_id: {chat_id}, user_id: {user_id}")

                if self._is_empty_chat(checkpoint_tuple):
                    logging.debug(f"Chat_id: {chat_id}, user_id: {user_id} is empty, skipping")
                    continue

                if chat_id not in chat_ids:
                    chat_ids.add(chat_id)
                    chat_list.append({
                        "id": chat_id,
                        "userId": user_id
                    })

            rows = []
            for chat in chat_list:
                chat_metadata = self._get_chat_metadata(checkpoint_tuple)
                row = {
                    "id": chat["id"],
                    "userId": chat["userId"],
                    "name": chat_metadata.get("name"),
                    "createdAt": chat_metadata.get("created_at"),
                }
                rows.append(row)
        return rows
    
    async def fetch_chat(self, chat_id: str, user_id: str) -> dict | None:
        """
        Fetch a specific chat thread from the database.

        Args:
            chat_id: The ID of the chat thread.
            user_id: The ID of the user.
        Returns:
            A chat thread record or None if not found.
        """
        async with AsyncPostgresSaver.from_conn_string(self.db_url) as checkpointer:
            config = {"configurable": {"thread_id": chat_id}}
            checkpoint_tuple = await checkpointer.aget_tuple(config=config)
            if checkpoint_tuple and checkpoint_tuple.metadata.get("user_id") == user_id:
                logging.debug(f"Found checkpoint_tuple for chat_id: {chat_id}, user_id: {user_id}")

                if self._is_empty_chat(checkpoint_tuple):
                    logging.debug(f"Chat_id: {chat_id}, user_id: {user_id} is empty, skipping")
                    return None

                chat_metadata = self._get_chat_metadata(checkpoint_tuple)
                chat = {
                    "id": chat_id,
                    "userId": user_id,
                    "name": chat_metadata.get("name"),
                    "createdAt": chat_metadata.get("created_at"),
                }
                return chat
        return None
    
    async def delete_chats(self, user_id: str) -> None:
        """
        Delete all chat threads for a specific user.

        Args:
            user_id: The ID of the user.
        """
        async with AsyncPostgresSaver.from_conn_string(self.db_url) as checkpointer:
            thread_ids = []
            async for checkpoint_tuple in checkpointer.alist(config=None, filter={"user_id": user_id}):
                thread_id = checkpoint_tuple.config['configurable']['thread_id']
                if thread_id not in thread_ids:
                    thread_ids.append(thread_id)
            
            # Then delete them after iteration is complete
            for thread_id in thread_ids:
                await checkpointer.adelete_thread(thread_id)
                logging.debug(f"Deleted thread: {thread_id}, user_id: {user_id}")

    async def update_chat(self, chat_id: str, user_id: str, chat_data: dict) -> dict:
        """
        Update a specific chat thread for a specific user.

        Args:
            chat_id: The ID of the chat thread.
            user_id: The ID of the user.
            chat_data: The chat data to update.
        Returns:
            The updated chat thread record.
        """
        # TODO: finish implementation of update_chat

        return chat_data  # Placeholder return
    
    async def delete_chat(self, chat_id: str, user_id: str) -> None:
        """
        Delete a specific chat thread for a specific user.

        Args:
            chat_id: The ID of the chat thread.
            user_id: The ID of the user.
        """
        async with AsyncPostgresSaver.from_conn_string(self.db_url) as checkpointer:
            config = {"configurable": {"thread_id": chat_id}}
            checkpoint_tuple = await checkpointer.aget_tuple(config=config)
            if checkpoint_tuple and checkpoint_tuple.metadata.get("user_id") == user_id:
                await checkpointer.adelete_thread(chat_id)
                logging.debug(f"Deleted thread: {chat_id}, user_id: {user_id}")

    async def fetch_messages(self, chat_id: str, user_id: str, filters: dict = {}) -> list:
        """
        TODO: implement tags filtering logic.
        Fetch messages from the database based on filter configuration.

        Args:
            chat_id: The ID of the chat thread.
            user_id: The ID of the user.
            filters: A dictionary to filter messages.
        Returns:
            A list of message records.
        """
        rows = []
        async with AsyncPostgresSaver.from_conn_string(self.db_url) as checkpointer:
            # Create agent with the checkpointer to access state history
            agent = create_rest_api_agent(checkpointer)
            
            # Filter by chat_id
            config = {"configurable": {"thread_id": chat_id}}
            
            limit = filters.get("limit")
            defaut_tag_filters = [
                { "human": ["welcome", "confirmation"] },
                { "ai": ["welcome"] },
            ]
            
            # Collect states grouped by request_id in reverse order
            states_list = []
            async for state in agent.aget_state_history(config, filter={"user_id": user_id}):
                # Filter by user_id
                if state.metadata.get("user_id") == user_id:
                    states_list.append(state)

            # Group states by request_id
            states_dict = {}
            for state in reversed(states_list):
                if state and state.values and state.metadata:
                    state_request_id = state.metadata.get("request_id")
                    if state_request_id:
                        if state_request_id not in states_dict:
                            states_dict[state_request_id] = []
                        states_dict[state_request_id].append(state)
            
            # Process states for each request_id
            processed_message_ids = []
            for request_id, states in states_dict.items():
                
                logging.debug(f"Processing state for chat_id: {chat_id}, request_id: {request_id}")
                
                user_row = None
                agent_row = None
                
                mcp_str = ""
                llm_str = ""

                for state in states:
                    agent_metadata = state.values.get("agent_metadata", {})
                    context = agent_metadata.get("context", {})
                    tags = agent_metadata.get("tags", [])
                    mcp_responses = agent_metadata.get("mcp_responses", [])
                    mcp_resp_str = "".join(mcp_responses) if mcp_responses else ""

                    # Filter out already processed messages
                    messages = [m for m in state.values.get("messages", []) if hasattr(m, "id") and m.id not in processed_message_ids]

                    for msg in messages:
                        if msg.type == 'human' and self._filter_by_tags(defaut_tag_filters, tags, msg.type):
                            if user_row is None:
                                text = agent_metadata.get("prompt", "")
                                user_row = {
                                    "chatId": chat_id,
                                    "requestId": request_id,
                                    "role": "user",
                                    "message": text if text else "",
                                    "context": context,
                                    "tags": tags,
                                    "createdAt": msg.additional_kwargs.get("created_at"),
                                }

                        if msg.type == 'ai' and self._filter_by_tags(defaut_tag_filters, tags, msg.type):
                            # Always concatenate MCP responses to agent message
                            if mcp_str == "":
                                mcp_str = mcp_resp_str
                            if llm_str == "":
                                llm_str = msg.content if msg.content else ""
                                
                            text = (mcp_str + llm_str) if (mcp_str or llm_str) else agent_row["message"] if agent_row else ""
                            if text:
                                agent_row = {
                                    "chatId": chat_id,
                                    "requestId": request_id,
                                    "role": "agent",
                                    "message": text,
                                    "context": None,
                                    "tags": tags,
                                    "createdAt": msg.additional_kwargs.get("created_at"), # Always the date from latest Agent node
                                }
                        processed_message_ids.append(msg.id)
                if user_row:
                    rows.append(user_row)
                if agent_row:
                    rows.append(agent_row)

                if limit and len(rows) >= limit:
                    return rows[:limit]

        return rows

async def create_database_manager() -> DatabaseManager:
    """
    Factory function to create a DatabaseManager instance.

    Returns:
        An instance of DatabaseManager.
    """
    
    manager = DatabaseManager()
    await manager.initialize_database()
    
    logging.info("DatabaseManager created and initialized")
    return manager