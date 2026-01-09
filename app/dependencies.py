from .services.llm import LLMManager
from langchain_core.language_models.llms import BaseLanguageModel

def get_llm() -> BaseLanguageModel:
    """
    FastAPI dependency that provides the singleton LLM instance.
    
    This function is used as a dependency injection in FastAPI endpoints
    to access the configured language model.
    
    Returns:
        The singleton language model instance.
    """
    return LLMManager.get_instance()