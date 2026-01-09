import os
import logging

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_aws import ChatBedrockConverse

class LLMManager:
    """
    Singleton manager for language model instances.
    
    This class ensures that only one instance of the language model is created
    and reused throughout the application, avoiding redundant initializations
    and ensuring consistent model configuration.
    """
    _instance: BaseLanguageModel = None

    @classmethod
    def get_instance(cls) -> BaseLanguageModel:
        """
        Retrieves the singleton instance of the language model.
        
        If the instance doesn't exist yet, it initializes it by calling get_llm().
        Subsequent calls return the same instance.
        
        Returns:
            The singleton language model instance.
        """
        if cls._instance is None:
            cls._instance = get_llm()
            logging.info(f"Using model: {cls._instance}")
        return cls._instance

def get_llm() -> BaseLanguageModel:
    """
    Selects and returns a language model instance based on environment variables.
    
    Returns:
        An instance of a language model.
        
    Raises:
        ValueError: If no supported model or API key is configured.
    """

    active = os.environ.get("ACTIVE_LLM", "")
    if active and active not in ["ollama", "gemini", "openai", "bedrock"]:
        raise ValueError("Unsupported Active LLM specified.")

    model = get_llm_model(active)
    
    ollama_url = os.environ.get("OLLAMA_URL")
    gemini_key = os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_url = os.environ.get("OPENAI_URL")
    aws_region = os.environ.get("AWS_REGION")

    if active == "ollama":
        return ChatOllama(model=model, base_url=ollama_url)
    if active == "gemini":
        return ChatGoogleGenerativeAI(model=model)
    if active == "openai":
        if openai_url:
            return ChatOpenAI(model=model, base_url=openai_url)
        else:
            return ChatOpenAI(model=model)
    if active == "bedrock":
        return ChatBedrockConverse(model=model)

    # default order if active is not specified
    if ollama_url:
        return ChatOllama(model=model, base_url=ollama_url)
    if gemini_key:
        return ChatGoogleGenerativeAI(model=model)
    if openai_key:
        if openai_url:
            return ChatOpenAI(model=model, base_url=openai_url)
        else:
            return ChatOpenAI(model=model)
    if aws_region:
        return ChatBedrockConverse(model=model)

    raise ValueError("LLM not configured.")

def get_llm_model(active_llm: str) -> str:
    """
    Retrieves the model name from environment variables.
    If an active LLM is specified, it looks for the corresponding model variable, otherwise it falls back to a general MODEL variable.
    
    Args:
        active_llm: The active LLM identifier, one of 'ollama', 'gemini', 'openai', 'bedrock'.

    Returns:
        The model name as a string.
    """

    model = None

    if active_llm:
        model = os.environ.get(f"{active_llm.upper()}_MODEL")

    if not model:
        model = os.environ.get("MODEL")

    if not model:
        raise ValueError("LLM Model not configured.")

    return model

