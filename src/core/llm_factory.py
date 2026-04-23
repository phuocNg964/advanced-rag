import logging
from typing import Any

from src.core.config import get_settings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

def get_llm(model_size: str = "small", **kwargs: Any):
    """
    Centralized factory to instantiate the configured LLM.
    
    Args:
        model_size (str): "small" for fast/cheap tasks (routing, summarization), 
                          "large" for heavy tasks (RAG generation).
        **kwargs: Additional overrides for the LLM constructor (e.g., temperature).
    """
    settings = get_settings()
    temperature = kwargs.pop("temperature", 0.3)
    
    if settings.use_local_llm or settings.llm_provider == "local":
        logger.info(f"Instantiating Local LLM: {settings.ollama_model}")
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_host,
            temperature=temperature,
            **kwargs
        )
        
    elif settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set if llm_provider is 'openai'")
        
        # Use gpt-5-mini for small/fast routing tasks (next-gen intelligence while keeping costs extremely low).
        # It supports custom temperatures unlike the nano models.
        base_model = "gpt-4.1-nano" if model_size == "small" else "gpt-4.1-mini"
        model_name = kwargs.pop("model", base_model)
        
        logger.info(f"Instantiating OpenAI LLM: {model_name}")
        return ChatOpenAI(
            model=model_name, 
            temperature=temperature, 
            api_key=settings.openai_api_key,
            max_retries=kwargs.pop("max_retries", 5),
            **kwargs
        )
        
    else: # Default to Gemini
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be set if llm_provider is 'gemini'")
            
        base_model = "gemini-2.0-flash"
        model_name = kwargs.pop("model", base_model)
        
        logger.info(f"Instantiating Gemini LLM: {model_name}")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=settings.gemini_api_key,
            max_retries=kwargs.pop("max_retries", 5),
            **kwargs
        )
