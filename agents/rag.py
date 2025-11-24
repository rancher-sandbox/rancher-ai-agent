import os
import json
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain.tools import tool
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings

FLEET_BASE_URL = "https://fleet.rancher.io"
RANCHER_BASE_URL = "https://ranchermanager.docs.rancher.com"
RAG_ADD_DOCS_BATCH_SIZE = 100

VECTOR_STORE_DIR = os.environ.get("RAG_VECTORSTORE_DIR", "/app/rag/vectorstore")
DOC_STORE_DIR = os.environ.get("RAG_DOCSTORE_DIR", "/app/rag/docstore")
FLEET_DOC_PATH = os.environ.get("FLEET_DOCS_PATH", "/fleet_docs")
RANCHER_DOC_PATH = os.environ.get("RANCHER_DOCS_PATH", "/rancher_docs")

def init_rag_retriever():
    """
    Initializes the RAG retrievers for Fleet and Rancher documentation.

    This function sets up two separate retrievers, one for Fleet and one for Rancher.
    It checks if persisted vector and document stores exist. If not, it loads
    the markdown documentation from the configured paths, processes the documents,
    and adds them to the persistent stores. This ensures that the document loading
    and embedding process only runs once.
    """
    fleet_vector_store_dir = VECTOR_STORE_DIR + "/fleet"
    fleet_docstore_dir = DOC_STORE_DIR + "/fleet"
    rancher_vector_store_dir = VECTOR_STORE_DIR + "/rancher"
    rancher_docstore_dir = DOC_STORE_DIR + "/rancher"

    fleet_retriever = hierarchical_retriever(fleet_vector_store_dir, fleet_docstore_dir, _get_llm_embeddings())

    if not os.path.exists(fleet_vector_store_dir) or not os.path.exists(fleet_docstore_dir):
        logging.info(f"loading Fleet RAG documents")
        _load_and_add_docs(fleet_retriever, FLEET_DOC_PATH)

    rancher_retriever = hierarchical_retriever(rancher_vector_store_dir, rancher_docstore_dir, _get_llm_embeddings())

    if not os.path.exists(rancher_vector_store_dir) or not os.path.exists(rancher_docstore_dir):
        logging.info(f"loading Rancher RAG documents")
        _load_and_add_docs(rancher_retriever, RANCHER_DOC_PATH)


@tool("fleet_documentation_retriever", description="A specialized Fleet Documentation Retriever tool. MUST be called for any question related to Fleet, Fleet documentation, resources, GitRepo, Bundle") 
def fleet_documentation_retriever(query: str) -> str:
    """
    Retrieves relevant documents from the Fleet documentation based on a query.

    This tool uses a hierarchical retriever to find documents related to the user's query.
    It then formats the results into a JSON string containing the content for the LLM
    and user-facing documentation links.

    Args:
        query: The user's question or search term.

    Returns:
        A JSON string with 'llm' and 'docLinks' keys.
    """
    vectore_store_dir = os.environ.get("RAG_VECTORSTORE_DIR", "/app/rag/vectorstore") + "/fleet"
    docstore_dir = os.environ.get("RAG_DOCSTORE_DIR", "/app/rag/docstore") + "/fleet"

    retriever = hierarchical_retriever(vectore_store_dir, docstore_dir, _get_llm_embeddings())

    docs = retriever.invoke(query)
    logging.debug(f"RAG retreived docs: {docs}")

    page_contents = [doc.page_content for doc in docs]
    # use set to remove duplicates
    doc_links_set = {_transform_source_to_url(doc.metadata["source"], FLEET_DOC_PATH, FLEET_BASE_URL) for doc in docs}
    doc_inks = list(doc_links_set)

    return json.dumps({
        "llm": page_contents,
        "docLinks": doc_inks
    })

@tool("rancher_documentation_retriever", description="A specialized Rancher Documentation Retriever tool. MUST be called for any question related to Rancher") 
def rancher_documentation_retriever(query: str) -> str:
    """
    Retrieves relevant documents from the Rancher documentation based on a query.

    This tool uses a hierarchical retriever to find documents related to the user's query.
    It then formats the results into a JSON string containing the content for the LLM
    and user-facing documentation links.

    Args:
        query: The user's question or search term.

    Returns:
        A JSON string with 'llm' and 'docLinks' keys.
    """
    vectore_store_dir = os.environ.get("RAG_VECTORSTORE_DIR", "/app/rag/vectorstore") + "/rancher"
    docstore_dir = os.environ.get("RAG_DOCSTORE_DIR", "/app/rag/docstore") + "/rancher"

    retriever = hierarchical_retriever(vectore_store_dir, docstore_dir, _get_llm_embeddings())

    docs = retriever.invoke(query)
    logging.debug(f"RAG retreived docs: {docs}")

    page_contents = [doc.page_content for doc in docs]
    doc_links_set = {_transform_source_to_url(doc.metadata["source"], RANCHER_DOC_PATH, RANCHER_BASE_URL) for doc in docs}
    docLinks = list(doc_links_set)

    return json.dumps({
        "llm": page_contents,
        "docLinks": docLinks
    })

def hierarchical_retriever(persist_dir: str, persist_dir_docstore: str, embeddings: Embeddings) -> BaseRetriever:
    """
    Creates and configures a ParentDocumentRetriever.

    This retriever is designed for effective RAG by splitting documents into smaller
    child chunks for embedding and similarity search, but returning the larger parent
    chunks to provide more context to the LLM. It uses a Chroma vector store for
    the embeddings and a local file store for the raw documents.

    Args:
        persist_dir: The directory to persist the Chroma vector store.
        persist_dir_docstore: The directory to persist the document store.
        embeddings: The embedding model to use for vectorizing documents.

    Returns:
        An instance of ParentDocumentRetriever.
    """
    child_chunk_size = int(os.environ.get("RAG_CHILD_CHUNK_SIZE", 500))
    child_chunk_overlap = int(os.environ.get("RAG_CHILD_CHUNK_OVERLAP", 150))
    parent_chunk_size = int(os.environ.get("RAG_PARENT_CHUNK_SIZE", 4000))
    parent_chunk_overlap = int(os.environ.get("RAG_PARENT_CHUNK_OVERLAP", 400))

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap)
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    local_file_store = LocalFileStore(persist_dir_docstore)
    docstore = create_kv_docstore(local_file_store)

    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 3}
    )

    return parent_retriever

def _transform_source_to_url(path: str, prefix_to_remove: str, base_url: str) -> str:
    """
    Transforms a local documentation file path into a public-facing URL.

    It removes a specified path prefix and the '.md' suffix from the file path
    and prepends a base URL to create a clean, shareable link.

    Args:
        path: The local file path of the document source.
        prefix_to_remove: The prefix of the path to be removed (e.g., the root docs directory).
        base_url: The base URL of the documentation site.

    Returns:
        The fully formed documentation URL.
    """
    SUFFIX_TO_REMOVE = ".md"

    if path.startswith(prefix_to_remove):
        intermediate_path = path.replace(prefix_to_remove, "", 1)
    else:
        intermediate_path = path

    if intermediate_path.endswith(SUFFIX_TO_REMOVE):
        url_path = intermediate_path[:-len(SUFFIX_TO_REMOVE)]
    else:
        url_path = intermediate_path
        
    return base_url + url_path

def _get_llm_embeddings() -> Embeddings:
    """
    Selects and returns an embedding model instance based on environment variables.

    Returns:
        An instance of a LangChain embedding model that implements the Embeddings interface.

    Raises:
        ValueError: If a required environment variable (like EMBEDDING_MODEL for Ollama) is missing,
                    or if no supported embedding provider is configured at all.
    """

    ollama_url = os.environ.get("OLLAMA_URL")
    embedding_model_name = os.environ.get("EMBEDDINGS_MODEL")
    if not embedding_model_name:
            raise ValueError("EMBEDDINGS_MODEL must be set.")
    if ollama_url:
        return OllamaEmbeddings(model=embedding_model_name, base_url=ollama_url)

    gemini_key = os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        return GoogleGenerativeAIEmbeddings(model=embedding_model_name)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
            return OpenAIEmbeddings(model=embedding_model_name)

    raise ValueError("No embedding provider configured. Set OLLAMA_URL, GOOGLE_API_KEY, or OPENAI_API_KEY.")

def _load_and_add_docs(retriever: BaseRetriever, doc_path: str):
    """
    Loads documents from a directory and adds them to the retriever in batches.

    This function uses a lazy loader to iterate through documents one by one,
    collecting them into batches and adding them to the retriever. This is more
    memory-efficient than loading all documents at once.

    Args:
        retriever: The retriever instance to add documents to.
        doc_path: The path to the directory containing markdown documents.
    """
    loader = DirectoryLoader(path=doc_path, glob="**/*.md")
    doc_iterator = loader.lazy_load()
    
    batch = []
    for doc in doc_iterator:
        batch.append(doc)
        if len(batch) >= RAG_ADD_DOCS_BATCH_SIZE:
            logging.info(f"Adding batch of {len(batch)} documents to persistent stores...")
            retriever.add_documents(batch)
            batch = []
    
    if batch:
        logging.info(f"Adding final batch of {len(batch)} documents to persistent stores...")
        retriever.add_documents(batch)
