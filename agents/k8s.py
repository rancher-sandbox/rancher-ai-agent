import os
import json
import logging
import langgraph.types 
import uuid

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langchain_core.tools import BaseTool, ToolException
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph, Checkpointer
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, ExperimentalMarkdownSyntaxTextSplitter
from langchain_classic.chains.summarize.chain import load_summarize_chain
from langchain_core.documents import Document
from typing import List
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever

class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary: str

class K8sAgentBuilder:
    def __init__(self, llm: BaseChatModel, tools: list[BaseTool], system_prompt: str, checkpointer: Checkpointer):
        """
        Initializes the K8sAgentBuilder.

        Args:
            llm: The language model to use for the agent's decisions.
            tools: A list of tools the agent can use.
            system_prompt: The initial system-level instructions for the agent.
            checkpointer: The checkpointer for persisting agent state.
        """
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.checkpointer = checkpointer
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
    
    def summarize_conversation_node(self, state: AgentState):
        """
        Summarizes the conversation history.

        This node is invoked when the conversation becomes too long. It asks the LLM
        to create or extend a summary of the conversation, then replaces the
        previous messages with the new summary to keep the context concise.

        Args:
            state: The current state of the agent, containing messages and an optional summary.

        Returns:
            A dictionary with the updated summary and a condensed list of messages."""
        summary = state.get("summary", "")
        if summary:
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n"
                "Extend the summary by taking into account the new messages above:" )
        else:
            summary_message = "Create a summary of the conversation above:"

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = self.llm_with_tools.invoke(messages)
        new_messages = [RemoveMessage(id=m.id) for m in messages[:-3]]
        new_messages = new_messages + [response]

        logging.debug("summarizing conversation")
        
        return {"summary": response.content, "messages": new_messages}

    def call_model_node(self, state: AgentState, config: RunnableConfig):
        """
        Invokes the language model with the current state and context.

        This node prepares the messages for the LLM, including the system prompt,
        any contextual information (like current cluster), and the conversation history.
        It then calls the LLM to get the next response.

        Args:
            state: The current state of the agent.
            config: The runnable configuration.
            runtime: The LangGraph runtime, which holds the user's context.

        Returns:
            A dictionary containing the LLM's response message."""
        
        logging.debug("calling model")
        response = self.llm_with_tools.invoke([self.system_prompt] + state["messages"], config)
        logging.debug("model call finished")


        return {"messages": [response]}

    async def tool_node(self, state: AgentState):
        """
        Executes tools based on the LLM's request.

        This node processes tool calls from the last message, handling user
        confirmation for sensitive operations. It invokes the appropriate tool
        and returns the results as ToolMessage objects.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary containing a list of ToolMessage objects with the tool results,
            or an error message if a tool fails or is cancelled."""
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            if not _handle_interrupt(tool_call):
                return {"messages": "the tool execution was cancelled by the user."}
            
            try:
                logging.debug("calling tool")
                tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
                logging.debug("tool call finished")
                processed_result = _process_tool_result(tool_result)
                outputs.append(
                    ToolMessage(
                        content=processed_result,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )
            except ToolException as e:
                return {"messages": str(e)}

        return {"messages": outputs}
    
    def should_continue(self, state: AgentState):
        """
        Determines the next step in the agent's workflow.

        This conditional edge checks the last message in the state to decide whether to
        continue with a tool call, summarize the conversation, or end the execution.

        Args:
            state: The current state of the agent.

        Returns:
            A string indicating the next node to transition to: "continue",
            "summarize_conversation", or "end"."""
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            if len(messages) > 7:
                return "summarize_conversation"
            return "end"
        else:
            return "continue"

    def build(self) -> CompiledStateGraph:
        """
        Builds and compiles the LangGraph agent.

        Returns:
            A compiled LangGraph StateGraph ready to be invoked.
        """
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.call_model_node)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("summarize_conversation", self.summarize_conversation_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "tools",
                "summarize_conversation": "summarize_conversation",
                "end": END,
            },
        )
        workflow.add_edge("tools", "agent")
        workflow.add_edge("summarize_conversation", END)

        return workflow.compile(checkpointer=self.checkpointer)

def create_k8s_agent(llm: BaseChatModel, tools: list[BaseTool], system_prompt: str, checkpointer: Checkpointer) -> CompiledStateGraph:
    """
    Creates a LangGraph agent capable of interacting with Rancher and Kubernetes resources.
    
    This factory function instantiates the K8sAgentBuilder, builds the agent graph,
    and returns the compiled agent.
    
    Args:
        llm: The language model to use for the agent's decisions.
        tools: A list of tools the agent can use (e.g., to interact with K8s).
        system_prompt: The initial system-level instructions for the agent.
    
    Returns:
        A compiled LangGraph StateGraph ready to be invoked.
    """
    builder = K8sAgentBuilder(llm, tools, system_prompt, checkpointer)

    return builder.build()
def init_rag_hierarchical_method2(embedding_model: Embeddings, llm: BaseChatModel) -> VectorStoreRetriever:
    """
    Creates a hierarchical retriever using ParentDocumentRetriever.
    """
    doc_path = os.environ.get("DOCS_PATH", "/rag_docs")
    if not os.path.exists(doc_path) or not os.listdir(doc_path):
        raise FileNotFoundError("The directory /rag_docs does not exist or is empty.")
    
    # 1. Load Documents
    logging.info("Loading RAG documents...")
    loader = DirectoryLoader(path=doc_path, glob="**/*.md")
    raw_docs: List[Document] = loader.load()
    logging.info(f"RAG → {len(raw_docs)} raw documents loaded")
    # Split your website into big chunks
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 * 4, chunk_overlap=0)

    # This text splitter is used to create the child documents. They should be small chunk size.
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=125*4)
    vectorstore=InMemoryVectorStore(embedding_model)
    docstore=InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore, 
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    # Add documents to the retriever in a task
    logging.info(f"Starting adding docs.")
    retriever.add_documents(raw_docs)
    logging.info(f"Hierarchical RAG initialized successfully.")
    return retriever
def init_rag_rancher_hierarchical(embedding_model: Embeddings, llm: BaseChatModel) -> VectorStoreRetriever:
    """
    Creates a hierarchical retriever using ParentDocumentRetriever.
    
    This implementation correctly:
    1. Summarizes each document using load_summarize_chain.
    2. Embeds and stores ONLY the summaries in the vector_store.
    3. Stores the full, small chunks in the doc_store.
    4. Links them via a manually-controlled ID.
    """
    doc_path = os.environ.get("DOCS_PATH", "/rag_docs")
    if not os.path.exists(doc_path) or not os.listdir(doc_path):
        raise FileNotFoundError("The directory /rag_docs does not exist or is empty.")
    
    # 1. Load Documents
    logging.info("Loading RAG documents...")
    loader = DirectoryLoader(path=doc_path, glob="**/*.md")
    raw_docs: List[Document] = loader.load()
    logging.info(f"RAG → {len(raw_docs)} raw documents loaded")

    # 2. Define Splitter and Stores
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    vector_store = InMemoryVectorStore(embedding_model) 
    doc_store = InMemoryStore()

    # PARENT splitter: Splits by Markdown headers
    headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]
    parent_splitter = ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False
    )
    # 3. Create the Summarization Chain
    # We use 'stuff' to summarize each document file as a whole.
    summarization_chain = load_summarize_chain(
        llm, 
        chain_type="stuff"
    )

    parent_summaries = []
    parent_chunks_to_store = {}

    logging.info("Starting document sectioning and summarization...")
    
    # 4. Process docs one by one
    for doc in raw_docs:
        # 4a. Split doc into PARENT chunks (sections)
        parent_chunks = parent_splitter.split_text(doc.page_content)
        
        logging.info(f"Splitting '{doc.metadata.get('source')}' into {len(parent_chunks)} sections.")

        for parent_chunk in parent_chunks:
            # Generate a unique ID for this parent chunk
            current_chunk_id = str(uuid.uuid4())
            
            # 4b. Store the PARENT CHUNK (section) in the doc_store
            # We update its metadata to include the original source
            parent_chunk.metadata["original_source"] = doc.metadata.get("source")
            parent_chunks_to_store[current_chunk_id] = parent_chunk
            
            # 4c. Create and store the summary
            try:
                # Summarize the parent chunk
                summary_result = summarization_chain.invoke([parent_chunk]) 
                summary = summary_result['output_text']
                
                # Create the parent summary doc (for the vector store)
                summary_doc = Document(
                    page_content=summary, 
                    metadata={
                        "source": doc.metadata.get("source"),
                        "title": parent_chunk.metadata.get("H1", doc.metadata.get("title", "Untitled Section")),
                        "section_header": parent_chunk.metadata.get("H2", parent_chunk.metadata.get("H3")),
                        # This ID is the link to the parent chunk in the doc_store
                        "original_id": current_chunk_id 
                    }
                )
                parent_summaries.append(summary_doc)
            except Exception as e:
                logging.error(f"Error summarizing chunk from {doc.metadata.get('source')}: {e}")
                # Fallback: use a snippet
                summary_doc = Document(
                    page_content=parent_chunk.page_content[:400] + "...",
                    metadata={
                        "source": doc.metadata.get("source"),
                        "title": parent_chunk.metadata.get("H1", doc.metadata.get("title", "Untitled Section")),
                        "section_header": parent_chunk.metadata.get("H2", parent_chunk.metadata.get("H3")),
                        "original_id": current_chunk_id
                    }
                )
                parent_summaries.append(summary_doc)

    # 5. Manually Populate the Stores
    
    # 5a. Populate Vector Store with SUMMARIES of sections
    logging.info(f"Adding {len(parent_summaries)} section summaries to vector store.")
    vector_store.add_documents(parent_summaries)
    
    # 5b. Populate Doc Store with full PARENT SECTION chunks
    logging.info(f"Adding {len(parent_chunks_to_store)} parent sections to doc store.")
    doc_store.mset(list(parent_chunks_to_store.items()))

    # 6. Instantiate the Retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=doc_store,
        id_key="original_id", # Links summaries to parent chunks
        child_splitter=child_splitter,
        search_kwargs={"k": 10}
        # We can also set search_kwargs here, e.g., {'k': 5}
        # to control how many sections are retrieved.
    )

    logging.info(f"Hierarchical (Section-based) RAG initialized successfully.")
    return retriever

def init_rag_rancher(embedding_model: Embeddings) -> VectorStoreRetriever:
    """
    Creates a retriever for Rancher documentation using RAG (Retrieval-Augmented Generation).

    Args:
        embedding_model: The embedding model to use for creating document embeddings.

    Returns:
        A VectorStoreRetriever that can be used to fetch relevant documents.
    """
    # test if `/rag_docs` exists and contains files
    doc_path = os.environ.get("DOCS_PATH", "/rag_docs")
    if not os.path.exists(doc_path) or not os.listdir(doc_path):
        raise FileNotFoundError("The directory /rag_docs does not exist or is empty.")
    # load all markdown files in the directory
    logging.info(f"loading RAG documents")
    loader = DirectoryLoader(path=doc_path, glob="**/*.md")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)
    # Initialize the vector store
    vector_store = InMemoryVectorStore(embedding_model)  
    vector_store.add_documents(documents=all_splits)
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    logging.info(f"RAG → {len(docs)} raw documents loaded")

    return retriever    

def _create_confirmation_response(payload: str, type: str, name: str, kind: str, cluster: str, namespace: str):
    """
    Creates a structured confirmation response for the UI.

    This function formats a JSON payload that the UI can use to prompt the user
    for confirmation before executing a sensitive operation.

    Args:
        payload: The data for the operation (e.g., a patch or a resource definition).
        type: The type of operation (e.g., "patch").
        name: The name of the resource.
        kind: The kind of the resource (e.g., "Deployment").
        cluster: The target cluster.
        namespace: The target namespace.
    """
    payload_data = {
        "payload": payload,
        "type": type,
        "resource": {
            "name": name,
            "kind": kind,
            "cluster": cluster,
            "namespace": namespace
        }
    }

    json_payload = json.dumps(payload_data)

    return f'<confirmation-response>{json_payload}</confirmation-response>'

def _should_interrupt(tool_call: any) -> str:
    """
    Checks if a tool call requires user confirmation and generates an interrupt message.

    Args:
        tool_call: The tool call dictionary from the LLM.

    Returns:
        A formatted string to trigger a langgraph.types.interrupt, or an empty string
        if no interruption is needed.
    """
    if tool_call["name"] == "patchKubernetesResource":
        return _create_confirmation_response(tool_call['args']['patch'], "patch", tool_call['args']['name'], tool_call['args']['kind'], tool_call['args']['cluster'], tool_call['args']['namespace'])
    if tool_call["name"] == "createKubernetesResource":
        return _create_confirmation_response(tool_call['args']['resource'], "create", tool_call['args']['name'], tool_call['args']['kind'], tool_call['args']['cluster'], tool_call['args']['namespace'])
    return ""

def _extract_interrupt_message(interrupt_message:any) -> str: 
    """
    Extracts the user's response from an interrupt.

    The response from the UI might be a simple string or a JSON object.
    This function standardizes the extraction of the user's actual response.
    """
    try:
        json_response = json.loads(interrupt_message["response"])
        return json_response["prompt"]
    except Exception:
        return interrupt_message["response"]
    
def _handle_interrupt(tool_call: dict) -> bool:
    """Handles the user confirmation interrupt for a tool call."""
    if interrupt_message := _should_interrupt(tool_call):
        response = _extract_interrupt_message(langgraph.types.interrupt(interrupt_message))
        if response != "yes":
            return False
          
    return True 

def _process_tool_result(tool_result: str) -> str:
    """Processes the raw tool result, handling JSON and streaming UI context if necessary.
       MCP returns examaple: {"uiContext":{}, "llm": {}}"""
    try:
        json_result = json.loads(tool_result)
        if "uiContext" in json_result:
            writer = get_stream_writer()
            if writer:
                writer(f"<mcp-response>{json.dumps(json_result['uiContext'])}</mcp-response>")
        
        # Return the value for the LLM, or the full object if 'llm' key is not present
        return _convert_to_string_if_needed(json_result.get("llm", json_result))
    except json.JSONDecodeError:
        # If it's not a valid JSON, return the raw string result
        return tool_result

def _convert_to_string_if_needed(var):
    """
    Converts a variable to a JSON formatted string only if it is a dict or list
    
    Args:
        var: The variable to process.
        
    Returns:
        The JSON string representation if it was a dict/list,
        otherwise the original variable.
    """
    if isinstance(var, (dict, list)):
        return json.dumps(var)
    else:
        return var