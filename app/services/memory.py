import os
import logging
from datetime import datetime
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import CheckpointTuple

class MemoryManager:
    """
    Manages chat memories using a database or in-memory storage.
    """
    def __init__(self):
        self.db_enabled = os.environ.get("DB_ENABLED", "false").lower() == "true"
        self.db_url = os.environ.get("DB_CONNECTION_STRING", "")
        self.db_pool = None
        self.checkpointer = None
        
    async def initialize(self):
        if os.environ.get("DB_ENABLED", "false").lower() == "true":
            logging.info("Initializing PostgreSQL Checkpointer...")

            self.db_pool = AsyncConnectionPool(
                conninfo=self.db_url, 
                max_size=20,
                kwargs={"autocommit": True},
                open=False
            )
            
            await self.db_pool.open()

            self.checkpointer = AsyncPostgresSaver(self.db_pool)

            await self.checkpointer.setup()
            logging.info("PostgreSQL Checkpointer ready.")
        else:
            logging.info("Using InMemorySaver for checkpoints.")
            self.checkpointer = InMemorySaver()

    async def destroy(self):
        if self.db_pool:
            await self.db_pool.close()
            logging.info("PostgreSQL pool closed.")

    def get_checkpointer(self):
        if not self.checkpointer:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")
        return self.checkpointer

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
    
    def _is_empty_chat(self, checkpoint_tuple: CheckpointTuple, tag_filters: list[dict] = []) -> bool:
        """
        Check if a chat is empty from the checkpoint tuple.
        A chat is considered empty if it does not contain any messages besides tag-filtered messages.
        
        Example:
            tag_filters = [
                { "human": ["welcome"] },
                { "ai": ["welcome"] },
            ]
        """        
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
        TODO: implement filtering logic.
        Fetch chat threads from the database for a specific user.

        Args:
            user_id: The ID of the user.
            filters: A dictionary to filter chat threads.

        Returns:
            A list of chat thread records.
        """
        rows = []

        chat_ids = set()
        async for checkpoint_tuple in self.checkpointer.alist(config=None, filter={"user_id": user_id}):
            chat_id = checkpoint_tuple.config["configurable"]["thread_id"]
            user_id = checkpoint_tuple.metadata["user_id"]
            
            if not chat_id or not user_id:
                continue

            if self._is_empty_chat(checkpoint_tuple):
                logging.debug(f"Chat_id: {chat_id}, user_id: {user_id} is empty, skipping")
                continue
            
            logging.debug(f"Processing checkpoint_tuple for chat_id: {chat_id}, user_id: {user_id}")

            if chat_id not in chat_ids:
                chat_ids.add(chat_id)
                
                chat_metadata = self._get_chat_metadata(checkpoint_tuple)
                rows.append({
                    "id": chat_id,
                    "userId": user_id,
                    "name": chat_metadata.get("name"),
                    "createdAt": chat_metadata.get("created_at"),
                })

        return rows
    
    async def delete_chats(self, user_id: str) -> None:
        """
        Delete all chat threads for a specific user.

        Args:
            user_id: The ID of the user.
        """
        thread_ids = []
        async for checkpoint_tuple in self.checkpointer.alist(config=None, filter={"user_id": user_id}):
            thread_id = checkpoint_tuple.config['configurable']['thread_id']
            if thread_id not in thread_ids:
                thread_ids.append(thread_id)
        
        # Then delete them after iteration is complete
        for thread_id in thread_ids:
            await self.checkpointer.adelete_thread(thread_id)
            logging.debug(f"Deleted thread: {thread_id}, user_id: {user_id}")

    async def fetch_chat(self, chat_id: str, user_id: str) -> dict | None:
        """
        Fetch a specific chat thread from the database.

        Args:
            chat_id: The ID of the chat thread.
            user_id: The ID of the user.
        Returns:
            A chat thread record or None if not found.
        """
        config = {"configurable": {"thread_id": chat_id}}
        checkpoint_tuple = await self.checkpointer.aget_tuple(config=config)

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
        config = {"configurable": {"thread_id": chat_id}}
        checkpoint_tuple = await self.checkpointer.aget_tuple(config=config)

        if checkpoint_tuple and checkpoint_tuple.metadata.get("user_id") == user_id:
            await self.checkpointer.adelete_thread(chat_id)
            logging.debug(f"Deleted thread: {chat_id}, user_id: {user_id}")

    async def fetch_messages(self, stateGraph: CompiledStateGraph, chat_id: str, user_id: str, filters: dict = {}) -> list:
        """
        TODO: implement tags filtering logic.
        Fetch messages from the database based on filter configuration.

        Args:
            chat_id: The ID of the chat thread.
            user_id: The ID of the user.
            filters: A dictionary to filter messages.
        Returns:
            A list of message records.
            
        Filters example:
            tag_filters = [
                { "human": ["welcome"] },
                { "ai": ["welcome"] },
            ]
        """
        rows = []
        
        # Filter by chat_id
        config = {"configurable": {"thread_id": chat_id}}
        
        limit = filters.get("limit")
        tag_filters = filters.get("tag_filters", [])
        
        # Collect states in reverse order
        states_list = []
        async for state in stateGraph.aget_state_history(config, filter={"user_id": user_id}):
            states_list.insert(0, state)

        # Group states by request_id
        states_dict = {}
        for state in states_list:
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
            
            # Accumulators for the current request_id
            all_mcp_responses = []
            last_interrupt_val = ""
            
            # First pass to collect all MCP responses and last_interrupt across all checkpoints of this request_id
            for state in states:
                metadata = state.values.get("agent_metadata", {})
                
                # Collect MCP responses
                mcp_in_state = metadata.get("mcp_responses", [])
                if mcp_in_state:
                    for resp in mcp_in_state:
                        if resp not in all_mcp_responses:
                            all_mcp_responses.append(resp)
                
                # Collect last_interrupt (take the most recent non-empty one)
                interrupt_in_state = metadata.get("last_interrupt", "")
                if interrupt_in_state:
                    last_interrupt_val = interrupt_in_state

            mcp_str = "".join(all_mcp_responses)
            interrupt_str = str(last_interrupt_val) if last_interrupt_val else ""

            for state in states:
                agent_metadata = state.values.get("agent_metadata", {})
                context = agent_metadata.get("context", {})
                tags = agent_metadata.get("tags", [])

                # Filter out already processed messages
                messages = [m for m in state.values.get("messages", []) if hasattr(m, "id") and m.id not in processed_message_ids]

                for msg in messages:
                    # Skip internal summary messages, marked at creation time in agent/parent.py and agent/child.py
                    if getattr(msg, "additional_kwargs", {}).get("is_summary"):
                        processed_message_ids.append(msg.id)
                        continue

                    if msg.type == 'human' and self._filter_by_tags(tag_filters, tags, msg.type):
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

                    if msg.type == 'ai' and self._filter_by_tags(tag_filters, tags, msg.type):
                        llm_str = msg.content if msg.content else ""
                        
                        text = llm_str
                        
                        if interrupt_str and llm_str:
                            text = llm_str
                        elif interrupt_str:
                            text = interrupt_str
                        elif mcp_str:
                            text = mcp_str + llm_str

                        text = text if text else agent_row["message"] if agent_row else ""
                        
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

async def create_memory_manager() -> MemoryManager:
    """
    Factory function to create a MemoryManager instance.

    Returns:
        An instance of MemoryManager.
    """
    
    manager = MemoryManager()
    await manager.initialize()

    logging.info("MemoryManager created and initialized")
    return manager
