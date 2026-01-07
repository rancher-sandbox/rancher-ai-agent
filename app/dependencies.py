from .services.llm import LLMManager
from langchain_core.language_models.llms import BaseLanguageModel

def get_llm() -> BaseLanguageModel:
    return LLMManager.get_instance()