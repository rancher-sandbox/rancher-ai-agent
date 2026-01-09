import os
import pytest
import json
from unittest.mock import patch, MagicMock, call

from app.services.rag import (
    _get_llm_embeddings,
    _transform_source_to_url,
    fleet_documentation_retriever,
    rancher_documentation_retriever,
    init_rag_retriever,
    hierarchical_retriever,
)

@patch('app.services.rag.OllamaEmbeddings')
def test_get_llm_embeddings_ollama(mock_ollama_embeddings):
    with patch.dict(os.environ, {"EMBEDDINGS_MODEL": "test-model", "OLLAMA_URL": "http://localhost:11434"}, clear=True):
        embeddings = _get_llm_embeddings()
        mock_ollama_embeddings.assert_called_once_with(model="test-model", base_url="http://localhost:11434")
        assert embeddings == mock_ollama_embeddings.return_value

@patch('app.services.rag.GoogleGenerativeAIEmbeddings')
def test_get_llm_embeddings_gemini(mock_google_embeddings):
    with patch.dict(os.environ, {"EMBEDDINGS_MODEL": "gemini-pro", "GOOGLE_API_KEY": "fake-key"}, clear=True):
        embeddings = _get_llm_embeddings()
        mock_google_embeddings.assert_called_once_with(model="gemini-pro")
        assert embeddings == mock_google_embeddings.return_value

@patch('app.services.rag.OpenAIEmbeddings')
def test_get_llm_embeddings_openai(mock_openai_embeddings):
    with patch.dict(os.environ, {"EMBEDDINGS_MODEL": "gpt-4", "OPENAI_API_KEY": "fake-key"}, clear=True):
        embeddings = _get_llm_embeddings()
        mock_openai_embeddings.assert_called_once_with(model="gpt-4")
        assert embeddings == mock_openai_embeddings.return_value

def test_get_llm_embeddings_no_model():
    with patch.dict(os.environ, {"OLLAMA_URL": "http://localhost:11434"}, clear=True):
        with pytest.raises(ValueError, match="EMBEDDINGS_MODEL must be set."):
            _get_llm_embeddings()

def test_get_llm_embeddings_no_provider():
    with patch.dict(os.environ, {"EMBEDDINGS_MODEL": "some-model"}, clear=True):
        with pytest.raises(ValueError, match="No embedding provider configured."):
            _get_llm_embeddings()

@pytest.mark.parametrize("path, prefix, base_url, expected", [
    ("/docs/fleet/page.md", "/docs/fleet", "https://fleet.rancher.io", "https://fleet.rancher.io/page"),
    ("/docs/rancher/page.md", "/docs/rancher", "https://rancher.com", "https://rancher.com/page"),
])
def test_transform_source_to_url(path, prefix, base_url, expected):
    assert _transform_source_to_url(path, prefix, base_url) == expected

class MockDoc:
    def __init__(self, page_content, source):
        self.page_content = page_content
        self.metadata = {"source": source}

@pytest.fixture
def mock_fleet_rag_dependencies():
    with patch('app.services.rag._get_llm_embeddings', return_value=MagicMock()) as mock_get_embeddings, \
         patch('app.services.rag.hierarchical_retriever') as mock_hierarchical_retriever, \
         patch('app.services.rag.logging') as mock_logging:
        
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            MockDoc("Fleet content", "/fleet_docs/fleet.md"),
        ]
        mock_hierarchical_retriever.return_value = mock_retriever

        yield {
            "get_embeddings": mock_get_embeddings,
            "hierarchical_retriever": mock_hierarchical_retriever,
            "retriever": mock_retriever,
            "logging": mock_logging
        }

def test_fleet_documentation_retriever(mock_fleet_rag_dependencies):
    result = fleet_documentation_retriever.invoke("what is fleet")
    
    mock_fleet_rag_dependencies["get_embeddings"].assert_called_once()
    mock_fleet_rag_dependencies["hierarchical_retriever"].assert_called_once()
    mock_fleet_rag_dependencies["retriever"].invoke.assert_called_once_with("what is fleet")

    expected_json = {
        "llm": ["Fleet content"],
        "docLinks": [
            "https://fleet.rancher.io/fleet",
        ]
    }
    assert json.loads(result) == expected_json

@pytest.fixture
def mock_rancher_rag_dependencies():
    with patch('app.services.rag._get_llm_embeddings', return_value=MagicMock()) as mock_get_embeddings, \
         patch('app.services.rag.hierarchical_retriever') as mock_hierarchical_retriever, \
         patch('app.services.rag.logging') as mock_logging:
        
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            MockDoc("Rancher content", "/rancher_docs/rancher.md"),
        ]
        mock_hierarchical_retriever.return_value = mock_retriever

        yield {
            "get_embeddings": mock_get_embeddings,
            "hierarchical_retriever": mock_hierarchical_retriever,
            "retriever": mock_retriever,
            "logging": mock_logging
        }

def test_rancher_documentation_retriever(mock_rancher_rag_dependencies):
    result = rancher_documentation_retriever.invoke("what is rancher")

    mock_rancher_rag_dependencies["get_embeddings"].assert_called_once()
    mock_rancher_rag_dependencies["hierarchical_retriever"].assert_called_once()
    mock_rancher_rag_dependencies["retriever"].invoke.assert_called_once_with("what is rancher")

    expected_json = {
        "llm": ["Rancher content"],
        "docLinks": [
            "https://ranchermanager.docs.rancher.com/rancher"
        ]
    }
    assert json.loads(result) == expected_json

@patch('app.services.rag.hierarchical_retriever')
@patch('app.services.rag._get_llm_embeddings')
@patch('os.path.exists')
@patch('os.listdir')
@patch('app.services.rag.DirectoryLoader')
def test_init_rag_retriever_creates_new_vectorestore(mock_loader, mock_listdir, mock_exists, mock_get_embeddings, mock_hierarchical_retriever):
    mock_exists.return_value = False
    mock_listdir.return_value = ["some_file.md"]
    mock_docs = [MagicMock()]
    mock_loader.return_value.lazy_load.return_value = mock_docs
    
    mock_fleet_retriever = MagicMock()
    mock_rancher_retriever = MagicMock()
    mock_hierarchical_retriever.side_effect = [mock_fleet_retriever, mock_rancher_retriever]

    init_rag_retriever()

    assert mock_hierarchical_retriever.call_count == 2
    mock_fleet_retriever.add_documents.assert_called_once_with(mock_docs)
    mock_rancher_retriever.add_documents.assert_called_once_with(mock_docs)

@patch('app.services.rag.hierarchical_retriever')
@patch('app.services.rag._get_llm_embeddings')
@patch('os.path.exists')
def test_init_rag_retriever_already_persisted(mock_exists, mock_get_embeddings, mock_hierarchical_retriever):
    mock_exists.return_value = True
    mock_fleet_retriever = MagicMock()
    mock_rancher_retriever = MagicMock()
    mock_hierarchical_retriever.side_effect = [mock_fleet_retriever, mock_rancher_retriever]

    init_rag_retriever()

    assert mock_hierarchical_retriever.call_count == 2
    mock_fleet_retriever.add_documents.assert_not_called()
    mock_rancher_retriever.add_documents.assert_not_called()

@patch('app.services.rag.Chroma')
@patch('app.services.rag.LocalFileStore')
@patch('app.services.rag.create_kv_docstore')
@patch('app.services.rag.ParentDocumentRetriever')
def test_hierarchical_retriever(mock_parent_retriever, mock_create_docstore, mock_file_store, mock_chroma):
    mock_embeddings = MagicMock()
    retriever = hierarchical_retriever("persist_dir", "docstore_dir", mock_embeddings)

    mock_chroma.assert_called_once_with(persist_directory="persist_dir", embedding_function=mock_embeddings)
    mock_file_store.assert_called_once_with("docstore_dir")
    mock_create_docstore.assert_called_once_with(mock_file_store.return_value)
    mock_parent_retriever.assert_called_once()
    assert retriever == mock_parent_retriever.return_value