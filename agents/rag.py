import os
import json
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter, ExperimentalMarkdownSyntaxTextSplitter
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
from langchain_core.documents import Document

def _transform_source_to_url(path: str) -> str:
    """
    Transforms a source file path to a Fleet URL.
    """
    BASE_URL = "https://fleet.rancher.io/"
    PREFIX_TO_REMOVE = "docs/fleet/"
    SUFFIX_TO_REMOVE = ".md"

    if path.startswith(PREFIX_TO_REMOVE):
        intermediate_path = path.replace(PREFIX_TO_REMOVE, "", 1)
    else:
        intermediate_path = path

    if intermediate_path.endswith(SUFFIX_TO_REMOVE):
        url_path = intermediate_path[:-len(SUFFIX_TO_REMOVE)]
    else:
        url_path = intermediate_path
        
    return BASE_URL + url_path

def _get_llm_embeddings() -> Embeddings:
    """
    Selects and returns an embedding model instance based on environment variables.

    Returns:
        An instance of a LangChain embedding model that implements the Embeddings interface.

    Raises:
        ValueError: If a required environment variable (like EMBEDDING_MODEL for Ollama) is missing,
                    or if no supported embedding provider is configured at all.
    """

     # Provider 1: Ollama
    ollama_url = os.environ.get("OLLAMA_URL")
    embedding_model_name = os.environ.get("EMBEDDINGS_MODEL")
    if not embedding_model_name:
            raise ValueError("EMBEDDINGS_MODEL must be set.")
    if ollama_url:
        return OllamaEmbeddings(model=embedding_model_name, base_url=ollama_url)

    # Provider 2: Google Gemini
    gemini_key = os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        return GoogleGenerativeAIEmbeddings(model=embedding_model_name)

    # Provider 3: OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
            return OpenAIEmbeddings(model=embedding_model_name)

    raise ValueError("No embedding provider configured. Set OLLAMA_URL, GOOGLE_API_KEY, or OPENAI_API_KEY.")

def init_retriever():
    vectoreStoreDir = os.environ.get("RAG_VECTORSTORE_DIR", "/app/rag/vectorstore")
    docstoreDir = os.environ.get("RAG_DOCSTORE_DIR", "/app/rag/docstore")
    ragApproach = os.environ.get("RAG_APPROACH", "hierarchical")

    retriever = fleet_hierarchical_retriever(vectoreStoreDir + "/" + ragApproach, docstoreDir + "/" +ragApproach, _get_llm_embeddings())

    if not os.path.exists(vectoreStoreDir + "/" + ragApproach) or not os.path.exists(docstoreDir + "/" +ragApproach):
        doc_path = os.environ.get("DOCS_PATH", "/docs/fleet")
        if not os.path.exists(doc_path) or not os.listdir(doc_path):
            raise FileNotFoundError("The directory /rancher_docs does not exist or is empty.")
        logging.info(f"loading Fleet RAG documents")
        loader = DirectoryLoader(path=doc_path, glob="**/*.md")
        docs = loader.load()

        logging.debug(f"Adding {len(docs)} new documents to hierarchical persistent stores...")
        retriever.add_documents(docs)

@tool("fleet_documentation_retriever", description="A specialized Fleet Documentation Retriever tool. MUST be called for any question related to Fleet, Fleet documentation, resources, GitRepo, Bundle") 
def fleet_documentation_retriever(query: str) -> str:
    vectoreStoreDir = os.environ.get("RAG_VECTORSTORE_DIR", "/app/rag/vectorstore")
    docstoreDir = os.environ.get("RAG_DOCSTORE_DIR", "/app/rag/docstore")
    ragApproach = os.environ.get("RAG_APPROACH", "hierarchical")

    retriever = fleet_hierarchical_retriever(vectoreStoreDir + "/" + ragApproach, docstoreDir + "/" +ragApproach, _get_llm_embeddings())

    docs = retriever.invoke(query)
    logging.debug(f"RAG retreived docs: {docs}")

    page_contents = [doc.page_content for doc in docs]
    docLinks = [_transform_source_to_url(doc.metadata["source"]) for doc in docs]

    return json.dumps({
        "llm": page_contents,
        "docLinks": docLinks
    })

def fleet_hierarchical_retriever(persist_dir: str, persist_dir_docstore: str, embeddings: Embeddings) -> BaseRetriever:
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    local_file_store = LocalFileStore(persist_dir_docstore)
    docstore = create_kv_docstore(local_file_store)

    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 2}
    )

    return parent_retriever

    