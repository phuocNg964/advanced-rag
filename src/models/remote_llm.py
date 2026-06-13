from typing import Any
import logging
from src.core.config import get_settings
from src.core.model_config import LLMProfile
from src.models.base import BaseModelFactory

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Provider → (ChatClass, api_key_field_name, api_key_kwarg_name)
_REMOTE_PROVIDERS = {
    "groq":   (ChatGroq,                  "groq_api_key",   "api_key"),
    "openai": (ChatOpenAI,                "openai_api_key", "api_key"),
    "gemini": (ChatGoogleGenerativeAI,    "gemini_api_key", "google_api_key"),
}

class RemoteLLMFactory(BaseModelFactory):
    @classmethod
    def supports(cls, provider: str) -> bool:
        return provider in _REMOTE_PROVIDERS

    @classmethod
    def instantiate(cls, profile: LLMProfile) -> Any:
        settings = get_settings()
        provider = profile.provider.lower()

        ChatClass, key_field, key_kwarg = _REMOTE_PROVIDERS[provider]
        api_key = getattr(settings, key_field, None)

        if not api_key:
            raise ValueError(f"{key_field.upper()} must be set when provider='{provider}'")

        kwargs = {
            "model": profile.model,
            "temperature": profile.temperature,
            "max_retries": profile.max_retries,
        }
        if profile.model_kwargs:
            kwargs["model_kwargs"] = profile.model_kwargs
            
        kwargs[key_kwarg] = api_key
        
        logger.info(f"Instantiating {provider} Remote LLM: {profile.model}")
        return ChatClass(**kwargs)
